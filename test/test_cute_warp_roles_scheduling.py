import itertools
import math
import random
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, cute

from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warp_roles import (
    Clc,
    DynamicPersistent,
    OneShot,
    PdlWait,
    ProtocolError,
    Raster,
    StaticPersistent,
    Tiles,
    TileStream,
    WarpSpecialized,
    role,
)

POLICIES = {
    "oneshot": OneShot(),
    "static": StaticPersistent(),
    "dynamic": DynamicPersistent(stages=2),
    "dynamic1": DynamicPersistent(stages=1),
    "clc": Clc(stages=2),
    "clc1": Clc(stages=1),
}


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("scheduling tests require SM100-family (CLC)")


# --------------------------------------------------------------------------- #
# CPU: order mappings
# --------------------------------------------------------------------------- #


def _all_coords(order, ext):
    total = ext[0] * ext[1] * ext[2]
    return [order.decode(i, ext) for i in range(total)]


EXTENTS = [(1, 1, 1), (1, 7, 1), (7, 1, 2), (5, 3, 1), (16, 16, 1), (13, 29, 3), (3, 100, 1)]
ORDERS = [
    Raster(),
    Raster(along="n"),
    Raster(swizzle=2),
    Raster(swizzle=8),
    Raster(along="n", swizzle=3),
    Raster(swizzle=8, reverse=("m",)),
    Raster(along="n", swizzle=5, reverse=("m", "n", "l")),
]


@pytest.mark.parametrize("ext", EXTENTS)
@pytest.mark.parametrize("order", ORDERS, ids=lambda o: o.get_key())
def test_raster_is_a_bijection(order, ext):
    coords = _all_coords(order, ext)
    assert sorted(coords) == sorted(itertools.product(*(range(e) for e in ext)))
    assert len(set(coords)) == len(coords)


@pytest.mark.parametrize("ext", [(13, 29, 1), (8, 64, 2), (40, 3, 1)])
def test_raster_along_and_l_slowest(ext):
    em, en, _ = ext
    for i, (m, n, l) in enumerate(_all_coords(Raster(), ext)):
        assert (m, n, l) == (i % em, (i // em) % en, i // (em * en))
    for i, (m, n, l) in enumerate(_all_coords(Raster(along="n"), ext)):
        assert (m, n, l) == ((i // en) % em, i % en, i // (em * en))


@pytest.mark.parametrize("swizzle", [2, 4, 8])
@pytest.mark.parametrize("ext", [(13, 29, 1), (32, 32, 1), (5, 17, 2)])
def test_raster_swizzle_bands(swizzle, ext):
    """Indices fill one band of ``swizzle`` n-columns before the next; within a band the band's
    columns are innermost, so any window of ``k * swizzle`` indices inside a band touches at most
    ``swizzle`` columns and ``k`` (+1) rows."""
    em, en, _ = ext
    coords = _all_coords(Raster(swizzle=swizzle), ext)
    for l in range(ext[2]):
        layer = coords[l * em * en : (l + 1) * em * en]
        bands = [n // swizzle for _, n, _ in layer]
        assert bands == sorted(bands)
        for start in range(0, en, swizzle):
            width = min(swizzle, en - start)
            band = [(m, n) for m, n, _ in layer if start <= n < start + width]
            for j, (m, n) in enumerate(band):
                assert (m, n) == (j // width, start + j % width)


def test_raster_lpt_reverses_m():
    ext = (6, 4, 2)
    plain = _all_coords(Raster(swizzle=2), ext)
    lpt = _all_coords(Raster(swizzle=2, reverse="m"), ext)
    assert [(ext[0] - 1 - m, n, l) for m, n, l in plain] == lpt
    assert lpt[0][0] == ext[0] - 1  # the longest query block goes first


def test_raster_validation():
    with pytest.raises(ValueError):
        Raster(along="k")
    with pytest.raises(ValueError):
        Raster(swizzle=0)
    with pytest.raises(ValueError):
        Raster(reverse=("k",))


def test_stream_keys_exclude_runtime_values():
    keys = {TileStream(order=Raster(swizzle=8), policy=p).get_key() for p in POLICIES.values()}
    assert len(keys) == len(POLICIES)
    assert TileStream().get_key() == "oneshot0_m1"


# --------------------------------------------------------------------------- #
# A toy kernel: every role logs, per CTA, the sequence of work items it saw.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SkipOdd(Raster):
    """Raster that skips tiles whose m + n is odd (an empty-item test mapping)."""

    skips = True

    def skip(self, item, space):
        m, n, _ = item
        return (m + n) % 2 == 1

    def get_key(self) -> str:
        return "skipodd_" + super().get_key()


class ScheduleToy(WarpSpecialized):
    """Roles ``a`` (warp 0) and ``b`` (warps 1-2) write each item's linear tile id into
    ``log[role, cta, k]`` (k = the CTA's k-th item) and bump ``hits[role, tile]``."""

    MAX_ITEMS = 64

    def __init__(self, policy, order=None, mode: str = "ok", regs: bool = False, ready=None):
        self.mode = mode
        self.tiles = TileStream(order=order or Raster(swizzle=3), policy=policy, ready=ready)
        self.regs = regs

    def get_name(self) -> str:
        return f"warp_roles_sched_toy_{self.tiles.get_key()}_{self.mode}"

    @cute.jit
    def _visit(self, r: int, tiles, count, ctx):
        m, n, l = tiles.work
        tile = (l * ctx.en + n) * ctx.em + m
        if cute.arch.lane_idx() == 0 and cute.arch.warp_idx() == r:
            bidx, _, _ = cute.arch.block_idx()
            if count < self.MAX_ITEMS:
                ctx.log[r, bidx, count] = tile
            cute.arch.atomic_add(ctx.hits.iterator + (r * ctx.hits.shape[1] + tile), Int32(1))

    @role(warps=0)
    @cute.jit
    def a(self, tiles: Tiles, ctx):
        count = Int32(0)
        while tiles.valid:
            self._visit(0, tiles, count, ctx)
            count += 1
            if cutlass.const_expr(self.mode == "conditional_advance"):
                if count < self.MAX_ITEMS:  # a runtime predicate: some iterations skip it
                    tiles.advance()
            else:
                tiles.advance()
            if cutlass.const_expr(self.mode == "double_advance"):
                tiles.advance()

    @role(warps=range(1, 3))
    @cute.jit
    def b(self, tiles: Tiles, ctx):
        count = Int32(0)
        if cutlass.const_expr(self.mode == "no_advance"):
            self._visit(1, tiles, count, ctx)
        else:
            while tiles.valid:
                self._visit(1, tiles, count, ctx)
                count += 1
                tiles.advance()

    @cute.kernel
    def kernel(self, log: cute.Tensor, hits: cute.Tensor, work):
        smem = self.begin_roles(tiles=work)
        # ~160 KB of SMEM: one CTA per SM, so CLC has unlaunched CTAs to cancel.
        smem.allocate_array(cutlass.Int32, 40000)
        em, en, _ = work.space
        self.run_roles(SimpleNamespace(log=log, hits=hits, em=em, en=en))

    @cute.jit
    def __call__(
        self,
        log: cute.Tensor,
        hits: cute.Tensor,
        workspace: cute.Tensor,
        em: Int32,
        en: Int32,
        el: Int32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        work = self.tiles.problem((em, en, el), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(log, hits, work).launch(
            grid=work.grid(), block=(self.num_threads, 1, 1), stream=stream
        )


_TOYS = {}


def compile_toy(op: ScheduleToy):
    key = op.get_name()
    if key not in _TOYS:
        i32 = cutlass.Int32
        _TOYS[key] = compile_tvm_ffi(
            op,
            make_fake_compact_tensor(i32, (2, cute.sym_int(), ScheduleToy.MAX_ITEMS)),
            make_fake_compact_tensor(i32, (2, cute.sym_int())),
            make_fake_compact_tensor(i32, (cute.sym_int(),)),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            name=key,
        )
    return _TOYS[key]


def run_toy(op, ext, max_ctas, reps=1):
    em, en, el = ext
    total = em * en * el
    persistent = isinstance(op.tiles.policy, (StaticPersistent, DynamicPersistent))
    grid = max(min(total, max_ctas), 1) if persistent else max(total, 1)
    log = torch.full((2, grid, ScheduleToy.MAX_ITEMS), -1, dtype=torch.int32, device="cuda")
    hits = torch.zeros(2, max(total, 1), dtype=torch.int32, device="cuda")
    ws, _ = op.tiles.launch_args("cuda")
    fn = compile_toy(op)
    for _ in range(reps):
        log.fill_(-1)
        hits.zero_()
        fn(log, hits, ws, em, en, el, max_ctas)
    torch.cuda.synchronize()
    return log.cpu(), hits.cpu()[:, :total], ws.cpu()


TOY_CASES = [
    ((1, 1, 1), 148),  # one tile
    ((5, 3, 1), 148),  # tiles < SMs
    ((13, 29, 3), 148),  # tiles >> SMs
    ((40, 40, 1), 7),  # capped grid: ~229 tiles per CTA for persistent policies
]


@pytest.mark.parametrize("ext,max_ctas", TOY_CASES)
@pytest.mark.parametrize("policy", list(POLICIES), ids=str)
def test_every_role_sees_every_tile_once(requires_blackwell, policy, ext, max_ctas):
    op = ScheduleToy(POLICIES[policy])
    log, hits, ws = run_toy(op, ext, max_ctas, reps=3)
    total = ext[0] * ext[1] * ext[2]
    assert (hits == 1).all(), f"tiles missed or repeated: {hits}"
    # Both roles of a CTA see the same sequence (for the first MAX_ITEMS items).
    assert torch.equal(log[0], log[1])
    if POLICIES[policy].persistent and total > max_ctas and not isinstance(op.tiles.policy, Clc):
        assert (log[0][:, 1] >= 0).all(), "persistent CTAs should run several tiles"
    if isinstance(op.tiles.policy, Clc) and total > 4 * 148:
        assert (log[0][:, 1] >= 0).any(), "no CLC cancellation succeeded"
    if policy == "static":
        grid = min(total, max_ctas)
        order = op.tiles.order
        for b in range(grid):
            for k in range(min(ScheduleToy.MAX_ITEMS, len(range(b, total, grid)))):
                m, n, l = order.decode(b + k * grid, ext)
                assert log[0, b, k] == (l * ext[1] + n) * ext[0] + m
    assert ws[:2].tolist() == [0, 0], "dynamic counter must self-reset"


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("policy", ["oneshot", "static", "dynamic", "clc"])
def test_skip_hook_hides_items_from_every_role(requires_blackwell, policy, ready):
    """Skip (and the readiness hook, a no-op wait outside PDL) in local and broadcast mode."""
    op = ScheduleToy(
        POLICIES[policy], order=SkipOdd(swizzle=2), mode="skip", ready=PdlWait() if ready else None
    )
    ext = (9, 7, 2)
    log, hits, _ = run_toy(op, ext, 16, reps=2)
    assert torch.equal(log[0], log[1])
    want = torch.tensor(
        [
            0 if (m + n) % 2 else 1
            for l in range(ext[2])
            for n in range(ext[1])
            for m in range(ext[0])
        ],
        dtype=torch.int32,
    )
    assert torch.equal(hits[0], want) and torch.equal(hits[1], want)


@dataclass(frozen=True)
class SkipAll(SkipOdd):
    def skip(self, item, space):
        m, _, _ = item
        return m >= 0


@pytest.mark.parametrize("policy", ["oneshot", "static", "dynamic", "clc"])
def test_all_items_skipped_terminates(requires_blackwell, policy):
    op = ScheduleToy(POLICIES[policy], order=SkipAll(), mode="skipall")
    _, hits, ws = run_toy(op, (5, 4, 1), 3, reps=2)
    assert (hits == 0).all() and ws[:2].tolist() == [0, 0]


def test_record_needs_a_bound_stream(requires_blackwell):
    stream = TileStream(policy=StaticPersistent(), record=4)
    with pytest.raises(ValueError, match="before launch_args"):
        stream.launch_args("cuda")


def test_describe_places_scheduler_and_counts(requires_blackwell):
    op = ScheduleToy(Clc(stages=3))
    op._collect()
    chans = op._ws_chans
    work = chans["tiles_work"]
    # consumers: a (32) + b (64) + scheduler (32) threads; producer: scheduler warp.
    assert work.slot_counts("consumer") == [128] * 3
    assert work.slot_counts("producer") == [32] * 3
    assert chans["tiles_clc"].slot_counts("producer") == [1]
    sched = next(r for r in op._ws_roles if r.name == "tiles.scheduler")
    assert sched.first_warp == 3
    assert "scheduler warp 3" in op.describe()
    local = ScheduleToy(StaticPersistent())
    local._collect()
    assert "tiles_work" not in local._ws_chans
    assert "decoded locally" in local.describe()


@pytest.mark.parametrize(
    "mode,message",
    [
        ("no_advance", "never calls advance"),
        ("double_advance", "advance\\(\\) traced 2 times"),
        ("conditional_advance", "not under a nested branch"),
    ],
)
def test_cursor_protocol_errors(requires_blackwell, mode, message):
    with pytest.raises(ProtocolError, match=message):
        compile_toy(ScheduleToy(StaticPersistent(), mode=mode))


class _IdleRole(WarpSpecialized):
    def __init__(self):
        self.tiles = TileStream(policy=StaticPersistent())

    @role(warps=0)
    @cute.jit
    def a(self, tiles: Tiles, ctx):
        pass

    @role(warps=1)
    @cute.jit
    def b(self, ctx):
        pass


def test_persistent_requires_every_role_to_loop():
    with pytest.raises(ProtocolError, match="do not take `tiles: Tiles`"):
        _IdleRole()._collect()


def test_random_extents_bijection_fuzz():
    rng = random.Random(0)
    for _ in range(200):
        ext = (rng.randint(1, 40), rng.randint(1, 40), rng.randint(1, 3))
        order = Raster(
            along=rng.choice("mn"),
            swizzle=rng.randint(1, 12),
            reverse=rng.choice([(), ("m",), ("n",)]),
        )
        coords = _all_coords(order, ext)
        assert len(set(coords)) == len(coords) == ext[0] * ext[1] * ext[2]
        assert all(0 <= c < e for xyz in coords for c, e in zip(xyz, ext, strict=True))


# --------------------------------------------------------------------------- #
# Persistent GEMM example: every policy, ragged shapes, tiles < SMs and >> SMs.
# --------------------------------------------------------------------------- #

GEMM_SHAPES = [
    (1, 8, 8),
    (300, 520, 136),  # ragged M, N, K
    (128, 256, 64),  # exactly one tile
    (1000, 1000, 1000),  # 8 x 4 tiles < SMs
    (4097, 2056, 520),  # 33 x 9 = 297 tiles, two waves, ragged
    (8192, 2048, 128),  # 512 tiles, short K: epilogue-bound, many tiles per CTA
]


def _check_gemm(policy, m, n, k, max_ctas=None, order=None):
    from transformer_nuggets.cute.warp_roles.examples.persistent_gemm import persistent_gemm

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):  # the second launch reuses the (self-reset) dynamic counter
        out.fill_(float("nan"))
        persistent_gemm(a, b, out, policy=policy, order=order, max_ctas=max_ctas)
        err = (out.float() - ref).abs().max().item()
        assert err <= 2 * torch_err + 1e-3, (policy, (m, n, k), err, torch_err)


@pytest.mark.parametrize("m,n,k", GEMM_SHAPES)
@pytest.mark.parametrize("policy", ["oneshot", "static", "dynamic", "clc"])
def test_persistent_gemm(requires_blackwell, policy, m, n, k):
    _check_gemm(POLICIES[policy], m, n, k)


@pytest.mark.parametrize("policy", ["static", "dynamic", "dynamic1", "clc1"])
@pytest.mark.parametrize("max_ctas", [1, 5])
def test_persistent_gemm_capped_grid(requires_blackwell, policy, max_ctas):
    """A tiny grid makes every CTA wrap the work, A/B, and accumulator rings many times."""
    _check_gemm(POLICIES[policy], 1100, 1032, 200, max_ctas=max_ctas)


@pytest.mark.parametrize("order", [Raster(), Raster(along="n", swizzle=3, reverse=("m", "n"))])
def test_persistent_gemm_orders(requires_blackwell, order):
    _check_gemm(POLICIES["dynamic"], 1500, 1800, 64, order=order)


def test_persistent_gemm_compiles_once_across_shapes(requires_blackwell, monkeypatch):
    """Tile counts, K, and the grid cap are runtime values: one compile per (policy, order)."""
    from transformer_nuggets.cute import cache
    from transformer_nuggets.cute.warp_roles.examples.persistent_gemm import (
        PersistentGemm,
        compile_persistent_gemm,
    )

    compiles = []
    real = cache.compile_tvm_ffi
    monkeypatch.setattr(
        cache, "compile_tvm_ffi", lambda *a, **kw: compiles.append(1) or real(*a, **kw)
    )
    op = PersistentGemm(policy=DynamicPersistent(stages=3), ab_stages=3)
    fns = set()
    # m: 1, multiple of 16, odd, tile edge, multi-wave; n, k: multiples of 8 on both sides of 16.
    for m, n, k, cap in [
        (1, 8, 8, 148),
        (16, 256, 64, 3),
        (129, 264, 136, 148),
        (4096, 520, 24, 64),
    ]:
        torch.manual_seed(m)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        fn = compile_persistent_gemm(op)
        fns.add(id(fn))
        ws, _ = op.tiles.launch_args("cuda")
        fn(a, b, out, ws, cap)
        ref = torch.nn.functional.linear(a.float(), b.float())
        assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item() + 1e-2
    assert len(compiles) <= 1 and len(fns) == 1


# --------------------------------------------------------------------------- #
# FA4-style forward on the same primitive: persistent policies.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "max_ctas,turnstile", [(None, False), (3, True)], ids=["sm_grid", "3ctas_turnstile"]
)
@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("b,h,sq,sk", [(2, 3, 640, 512), (8, 16, 1024, 384)])
@pytest.mark.parametrize("policy", ["static", "dynamic", "clc"])
def test_fa4_fwd_persistent(
    requires_blackwell, policy, b, h, sq, sk, nonuniform, max_ctas, turnstile
):
    """(8, 16, 1024) has 256 work items, so most CTAs run two (S/O/Q slots reused); with 3 CTAs
    each runs ~85, wrapping every slot and ring. sk=384 is an odd key-block count."""
    pytest.importorskip("flash_attn.cute")
    from test_cute_warp_roles import fa4_inputs

    from transformer_nuggets.cute.warp_roles.examples.fa4_fwd import fa4_fwd

    q, k, v = fa4_inputs(b, h, sq, sk, nonuniform)
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2)
    ref_lse = torch.logsumexp((qf @ kf.transpose(-1, -2)) / math.sqrt(128), dim=-1)
    tol = 2**-7 * ref.abs().max().item() + 4e-3
    for _ in range(2):
        o, lse = fa4_fwd(q, k, v, policy=POLICIES[policy], max_ctas=max_ctas, turnstile=turnstile)
        assert (o.float() - ref).abs().max().item() <= tol
        torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


def test_persistent_gemm_empty_k(requires_blackwell):
    from transformer_nuggets.cute.warp_roles.examples.persistent_gemm import persistent_gemm

    a = torch.empty(64, 0, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(24, 0, device="cuda", dtype=torch.bfloat16)
    assert (persistent_gemm(a, b) == 0).all()
