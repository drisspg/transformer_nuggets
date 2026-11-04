"""2-CTA (CtaPair) support in the warp-role layer: derived scopes, trace-time checks, a toy pair
protocol, and the 2-CTA persistent GEMM and FA4 examples under every scheduling policy."""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.nvgpu import tcgen05

from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warp_roles import (
    Clc,
    Consumer,
    CtaPair,
    DynamicPersistent,
    Instance,
    OneShot,
    Producer,
    ProtocolError,
    Raster,
    Ring,
    Slots,
    StaticPersistent,
    Threads,
    Tiles,
    TileStream,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    role,
)
from transformer_nuggets.cute.warp_roles.cluster import check_atoms

POLICIES = {
    "oneshot": OneShot(),
    "static": StaticPersistent(),
    "dynamic": DynamicPersistent(stages=2),
    "dynamic1": DynamicPersistent(stages=1),
    "clc": Clc(stages=2),
    "clc1": Clc(stages=1),
}
PAIR = CtaPair()


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("2-CTA tests require SM100-family tcgen05 and clusters")


# --------------------------------------------------------------------------- #
# Toy pair protocol: producers in both CTAs -> leader-only "MMA" role -> sinks in both CTAs.
# ``ch``: Threads producers (one per slot, both CTAs) joined on the leader's full barrier; the
# leader releases with a multicast tcgen05.commit (no MMAs needed), so producers tail.
# ``back``: the leader commits (multicast) and the sinks of both CTAs release on the leader.
# --------------------------------------------------------------------------- #


class PairToy(WarpSpecialized):
    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.cluster = PAIR
        self.ch = Slots(2, Threads, Umma)
        self.back = Ring(2, Umma, Threads)

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def prod(self, stage: Instance, ch: Producer[Instance], c):
        snapshot = ch  # "snapshot": a stale copy used later must not become the tail's state
        # Uneven commit counts per slot: slot 1 gets one more, so the slots' parities differ.
        for _ in cutlass.range(c.n + stage, unroll=1):
            h = ch.acquire(stage)
            if cutlass.const_expr(self.mode == "retire"):
                h.retire()
            else:
                h.commit()
        if cutlass.const_expr(self.mode == "snapshot"):  # noqa: SIM102 (runtime inside const)
            if c.n > 0:  # a dynamic region that carries (rebuilds) the stale copy
                c.out[3 + stage] = snapshot.full_barrier(stage).toint()

    @role(warps=2, cta="leader")
    @cute.jit
    def mma(self, ch: Consumer, back: Producer, c):
        for _ in cutlass.range(c.n, unroll=1):
            for j in cutlass.range_constexpr(2):
                with ch.consume(j), back.produce():
                    pass
        with ch.consume(1), back.produce():
            pass
        if cute.arch.lane_idx() == 0:
            c.out[2] = 2 * c.n + 1

    @role(warps=3)
    @cute.jit
    def sink(self, back: Consumer, c):
        count = cutlass.Int32(0)
        for _ in cutlass.range(2 * c.n + 1, unroll=1):
            with back.consume():
                count = count + 1
        rank = self.cluster.rank()  # warp-uniform: not inside the lane-0 branch
        if cute.arch.lane_idx() == 0:
            c.out[rank] = count

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(f"warp_roles_pair_toy_{self.mode}")
        self.kernel(out, n).launch(**self.launch_config(grid=(2, 1, 1)), stream=stream)


def compile_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), name=f"warp_roles_pair_toy_{op.mode}")


def test_pair_toy_derived_scopes():
    text = PairToy().describe()
    assert (
        "cluster (2, 1, 1)" in text and "role mma         warps  2-2  regs=None cta=leader" in text
    )
    # Threads from both CTAs join on the leader (32 + 32 per slot); the multicast commit lands
    # once on each CTA's empty barrier; producers drain because the release is asynchronous.
    assert (
        "chan ch     Slots(2) prod[0]@0, prod[1]@1 -> mma; pair-scoped: full leader [64, 64]"
        in text
    )
    assert "empty leader [1, 1], peer [1, 1] release_mask=0b11 remote_scope=cta tail" in text
    assert (
        "chan back   Ring(2)  mma -> sink; pair-scoped: full leader [1, 1], peer [1, 1]; " in text
    )
    assert "empty leader [64, 64]" in text


@pytest.mark.parametrize("mode", ["ok", "snapshot"])
@pytest.mark.parametrize("n", [1, 2, 5, 64])
def test_pair_toy_runs(requires_blackwell, n, mode):
    """Remote arrivals, multicast commits, slot-bound producer tails, and the exit protocol.
    Regression: tails once waited every slot, so prod[0] waited slot 1's parity and hung for even
    n (slot 1 completes n + 1 phases)."""
    out = torch.full((5,), -1, device="cuda", dtype=torch.int32)
    compile_toy(PairToy(mode))(out, n)
    torch.cuda.synchronize()
    assert out[:3].tolist() == [2 * n + 1] * 3


def test_retire_on_tailed_pair_channel_is_rejected(requires_blackwell):
    with pytest.raises(Exception, match="drain every stage before exit") as info:
        compile_toy(PairToy("retire"))
    assert isinstance(info.value, ProtocolError) or "ProtocolError" in repr(info.value)


# --------------------------------------------------------------------------- #
# Host-side placement and scope checks
# --------------------------------------------------------------------------- #


class _Unpinned(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.ch = Slots(1, Threads, Umma)

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, c):
        pass

    @role(warps=1)  # issues tcgen05 commits but runs in both CTAs
    @cute.jit
    def mma(self, ch: Consumer, c):
        pass


class _LeaderTmaToBoth(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.ch = Slots(1, Tma(1024), Threads)

    @role(warps=0, cta="leader")
    @cute.jit
    def load(self, ch: Producer, c):
        pass

    @role(warps=1)
    @cute.jit
    def use(self, ch: Consumer, c):
        pass


class _BothTmaToMixed(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.ch = Slots(1, Tma(1024), Threads)

    @role(warps=0)
    @cute.jit
    def load(self, ch: Producer, c):
        pass

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def use(self, ch: Consumer, c):
        pass

    @role(warps=3, cta="leader")
    @cute.jit
    def lead(self, ch: Consumer, c):
        pass


class _LeaderOnlyTmem(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.tmem_plan = TmemPlan().region("acc", 32)

    @role(warps=0, cta="leader")
    @cute.jit
    def mma(self, tmem: Tmem, c):
        pass


class _PairPayload(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.ch = Slots(1, Threads, Threads, payload=(cutlass.Int32, (4,)))

    @role(warps=0, cta="leader")
    @cute.jit
    def prod(self, ch: Producer, c):
        pass

    @role(warps=1)
    @cute.jit
    def use(self, ch: Consumer, c):
        pass


def test_pair_placement_errors():
    with pytest.raises(ProtocolError, match=r"pin it with @role\(\.\.\., cta='leader'\)"):
        _Unpinned()._collect()
    with pytest.raises(ProtocolError, match="would need multicast TMA"):
        _LeaderTmaToBoth()._collect()
    with pytest.raises(ProtocolError, match="can only feed leader-only consumers"):
        _BothTmaToMixed()._collect()
    with pytest.raises(ProtocolError, match="layer-owned payload is per-CTA SMEM"):
        _PairPayload()._collect()
    with pytest.raises(ProtocolError, match="every Tmem role .* is leader-only"):
        _LeaderOnlyTmem()._collect()
    with pytest.raises(ValueError, match="role cta must be one of"):
        role(warps=0, cta="peer")


def test_leader_placement_is_a_noop_for_one_cta():
    """The same role source runs on one CTA: `cta="leader"` places it in the only CTA."""
    from transformer_nuggets.cute.warp_roles.examples.persistent_gemm import PersistentGemm

    text = PersistentGemm().describe()
    assert "cta=leader" not in text and "pair-scoped" not in text
    assert "chan ab     Ring(4)  load -> mma; full arrivals [1, 1, 1, 1]" in text


def test_pair_gemm_declared_graph():
    from transformer_nuggets.cute.warp_roles.examples.persistent_gemm import PersistentGemm

    text = PersistentGemm(policy=Clc(), cluster=PAIR).describe()
    # Both CTAs' TMA bytes (2 x 32 KB) complete on the leader, armed once by its producer.
    assert (
        "chan ab     Ring(6)  load -> mma; pair-scoped: full leader [1, 1, 1, 1, 1, 1]; " in text
    )
    assert "tx=65536 release_mask=0b11 tail" in text
    assert (
        "chan acc    Ring(2)  mma -> epilogue; pair-scoped: full leader [1, 1], peer [1, 1]; "
        in text
    )
    assert "empty leader [256, 256] commit_mask=0b11" in text
    # The leader's scheduler publishes into both CTAs (st.async bytes), and every consumer
    # releases the leader's slot: epilogue and load in both CTAs, the leader-only MMA and
    # scheduler once: 2 x (128 + 32) + 32 + 32.
    assert "tiles.scheduler warps  6-6  regs=40 cta=leader" in text
    assert "chan tiles_work Ring(2)  tiles.scheduler -> " in text
    assert "full leader [1, 1], peer [1, 1]; empty leader [384, 384] tx=16" in text
    assert "chan tiles_clc Slots(1) tiles.scheduler -> tiles.scheduler; full arrivals [1]" in text


def test_fa4_pair_declared_graph():
    pytest.importorskip("flash_attn.cute")
    from transformer_nuggets.cute.warp_roles.examples.fa4_fwd import FlashAttentionForward

    text = FlashAttentionForward(cluster=PAIR).describe()
    assert (
        "chan kv     Ring(6)  load -> mma; pair-scoped: full leader" in text and "tx=32768" in text
    )
    assert "tx=65536" in text  # Q: two 32 KB halves
    # S joins softmax[j] and correction of both CTAs on the leader: 2 x (128 + 128).
    assert "empty leader [512, 512]" in text
    assert "chan stats  Slots(2) softmax[0]@0, softmax[1]@1 -> correction; full arrivals" in text


@cute.jit
def _mma_atom_check(cluster: cutlass.Constexpr, group: cutlass.Constexpr):
    bf16, major = cutlass.BFloat16, cute.nvgpu.OperandMajorMode
    m = 256 if group == tcgen05.CtaGroup.TWO else 128
    tiled = sm100_utils.make_trivial_tiled_mma(
        bf16, bf16, major.K, major.K, cutlass.Float32, group, (m, 256)
    )
    check_atoms({"tiled_mma": tiled}, cluster)


def test_mismatched_mma_atom_is_rejected(requires_blackwell):
    from transformer_nuggets.cute.warp_roles import SingleCta

    cute.compile(_mma_atom_check, PAIR, tcgen05.CtaGroup.TWO)
    with pytest.raises(Exception, match="spans 1 CTA.*declares CtaPair"):
        cute.compile(_mma_atom_check, PAIR, tcgen05.CtaGroup.ONE)
    with pytest.raises(Exception, match="spans 2 CTA.*declares SingleCta"):
        cute.compile(_mma_atom_check, SingleCta(), tcgen05.CtaGroup.TWO)


# --------------------------------------------------------------------------- #
# TileStream in a pair: work units are clusters
# --------------------------------------------------------------------------- #


class PairStream(WarpSpecialized):
    """Every stream user counts the items it sees; ``cta`` places them."""

    def __init__(self, policy, cta: str):
        self.cluster = PAIR
        self.cta = cta
        self.tiles = TileStream(order=Raster(), policy=policy)

    @role(warps=0, cta="leader")
    @cute.jit
    def lead(self, tiles: Tiles, c):
        while tiles.valid:
            m, n, _ = tiles.work
            if cute.arch.lane_idx() == 0:
                cute.arch.atomic_add(c.out.iterator + m * c.n_tiles + n, cutlass.Int32(1))
            tiles.advance()

    @role(warps=1, cta="leader")
    @cute.jit
    def other(self, tiles: Tiles, c):
        while tiles.valid:
            tiles.advance()

    @cute.kernel
    def kernel(self, out: cute.Tensor, n_tiles: cutlass.Int32, work):
        self.begin_roles(tiles=work)
        self.run_roles(SimpleNamespace(out=out, n_tiles=n_tiles))

    @cute.jit
    def __call__(
        self,
        out: cute.Tensor,
        m_tiles: cutlass.Int32,
        n_tiles: cutlass.Int32,
        workspace: cute.Tensor,
        max_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        work = self.tiles.problem((m_tiles, n_tiles, 1), workspace, max_ctas)
        self.kernel(out, n_tiles, work).launch(**self.launch_config(work), stream=stream)


@pytest.mark.parametrize("policy", ["oneshot", "static", "dynamic", "clc"])
def test_pair_stream_with_leader_only_users(requires_blackwell, policy):
    """All stream users leader-only: the broadcast ring is CTA-local to the leader, and its
    TMA-like publication still completes (st.async to the leader itself)."""
    op = PairStream(POLICIES[policy], "leader")

    def fake():
        return make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)

    i32 = cutlass.Int32(1)
    fn = compile_tvm_ffi(
        op, fake(), i32, i32, fake(), i32, name=f"warp_roles_pair_stream_{policy}"
    )
    for m_tiles, n_tiles, cap in [(1, 1, 148), (7, 9, 4), (40, 30, 148)]:
        out = torch.zeros(m_tiles * n_tiles, device="cuda", dtype=torch.int32)
        ws, _ = op.tiles.launch_args("cuda")
        fn(out, m_tiles, n_tiles, ws, cap)
        torch.cuda.synchronize()
        assert out.tolist() == [1] * (m_tiles * n_tiles), (m_tiles, n_tiles, cap)


def test_pair_record_workspace_covers_whole_clusters(requires_blackwell):
    op = PairStream(OneShot(), "leader")
    op.tiles = TileStream(order=Raster(), policy=StaticPersistent(), record=3)
    op.describe()  # binds the roles the log is sized by
    ws, cap = op.tiles.launch_args("cuda", max_ctas=1)
    roles = len(op.tiles.roles)
    assert cap == 1 and ws.numel() >= 4 + roles * 2 * 3 * 2  # one cluster = two CTAs


# --------------------------------------------------------------------------- #
# 2-CTA persistent GEMM
# --------------------------------------------------------------------------- #

GEMM_SHAPES = [
    (1, 8, 8),  # the peer CTA's half and most of the tile are out of bounds
    (300, 520, 136),  # ragged M, N, K
    (256, 256, 64),  # exactly one 2-CTA tile
    (1000, 1000, 1000),  # 4 x 4 tiles: far fewer clusters than SMs
    (4097, 2056, 520),  # 17 x 9 tiles, ragged, more than one wave
]


def _check_gemm(policy, m, n, k, max_ctas=None, reps=2, tile_n=256):
    from transformer_nuggets.cute.warp_roles.examples.persistent_gemm import persistent_gemm

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
    first = None
    for _ in range(reps):  # later launches reuse the (self-reset) dynamic counter
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        persistent_gemm(a, b, out, policy=policy, max_ctas=max_ctas, cluster=PAIR, tile_n=tile_n)
        err = (out.float() - ref).abs().max().item()
        assert err <= 2 * torch_err + 1e-3, (policy, (m, n, k), err, torch_err)
        if first is None:
            first = out
        assert torch.equal(out, first), "2-CTA GEMM is not bitwise deterministic"


@pytest.mark.parametrize("m,n,k", GEMM_SHAPES)
@pytest.mark.parametrize("policy", ["oneshot", "static", "dynamic", "clc"])
def test_pair_gemm(requires_blackwell, policy, m, n, k):
    _check_gemm(POLICIES[policy], m, n, k)


@pytest.mark.parametrize("tile_n", [224, 192, 128])
@pytest.mark.parametrize("policy", ["static", "clc"])
def test_pair_gemm_tile_n(requires_blackwell, policy, tile_n):
    """Narrower MMA tiles: each CTA loads tile_n / 2 rows of B; ragged M and N edges."""
    _check_gemm(POLICIES[policy], 3000, 1000, 520, tile_n=tile_n)


@pytest.mark.parametrize("policy", ["static", "dynamic", "dynamic1", "clc1"])
@pytest.mark.parametrize("max_ctas", [2, 3, 10])
def test_pair_gemm_capped_grid(requires_blackwell, policy, max_ctas):
    """2 and 3 CTAs launch one cluster, so its rings and the work ring wrap many times."""
    _check_gemm(POLICIES[policy], 1100, 1032, 200, max_ctas=max_ctas, reps=3)


def test_pair_gemm_counter_resets(requires_blackwell):
    from transformer_nuggets.cute.warp_roles.examples.persistent_gemm import _OPS

    _check_gemm(POLICIES["dynamic"], 1500, 1800, 64, max_ctas=8, reps=3)
    op = _OPS[(POLICIES["dynamic"], None, PAIR, 256)]
    ws, _ = op.tiles.launch_args("cuda")
    torch.cuda.synchronize()
    assert ws[:2].tolist() == [0, 0]


def test_pair_gemm_compiles_once_across_shapes(requires_blackwell, monkeypatch):
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
    op = PersistentGemm(policy=DynamicPersistent(stages=3), cluster=PAIR, ab_stages=5)
    fns = set()
    for m, n, k, cap in [
        (1, 8, 8, 148),
        (16, 256, 64, 3),
        (257, 264, 136, 148),
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
# 2-CTA FA4-style forward
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("split_p", [False, True], ids=["nosplit", "splitp"])
@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize(
    "b,h,sq,sk", [(1, 1, 256, 128), (1, 1, 384, 256), (2, 3, 640, 512), (8, 16, 1024, 384)]
)
@pytest.mark.parametrize("policy", ["oneshot", "static", "dynamic", "clc"])
def test_pair_fa4(requires_blackwell, policy, b, h, sq, sk, nonuniform, split_p):
    """sq=256/384: the second Q tile of the only item is (partly) out of bounds; (8, 16, 1024)
    has 128 items, so persistent clusters run several; sk=384 is an odd key-block count."""
    _check_fa4(POLICIES[policy], b, h, sq, sk, nonuniform, split_p)


@pytest.mark.parametrize("policy", ["static", "dynamic", "clc"])
def test_pair_fa4_three_clusters_turnstile(requires_blackwell, policy):
    """6 CTAs = 3 clusters, each running ~43 items: every slot and ring wraps."""
    _check_fa4(POLICIES[policy], 8, 16, 1024, 384, True, True, max_ctas=6, turnstile=True)


def _check_fa4(policy, b, h, sq, sk, nonuniform, split_p, max_ctas=None, turnstile=False):
    pytest.importorskip("flash_attn.cute")
    from test_cute_warp_roles import fa4_inputs

    from transformer_nuggets.cute.warp_roles.examples.fa4_fwd import fa4_fwd

    q, k, v = fa4_inputs(b, h, sq, sk, nonuniform)
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2)
    ref_lse = torch.logsumexp((qf @ kf.transpose(-1, -2)) / math.sqrt(128), dim=-1)
    tol = 2**-7 * ref.abs().max().item() + 4e-3
    first = None
    for _ in range(2):
        o, lse = fa4_fwd(
            q,
            k,
            v,
            policy=policy,
            max_ctas=max_ctas,
            turnstile=turnstile,
            cluster=PAIR,
            split_p=split_p,
        )
        assert (o.float() - ref).abs().max().item() <= tol
        torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)
        if first is None:
            first = o
        assert torch.equal(o, first)
