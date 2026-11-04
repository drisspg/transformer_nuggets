"""The public scheduler interface: custom policies and the raw-role work-item accessor, swapped
into the same kernels with one argument."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from test_cute_warpdrive_scheduling import (
    EMPTY_SPACES,
    OVERFLOWING_SPACES,
    CountToy,
    ScheduleToy,
    SkipOdd,
    _check_gemm,
    _count_items,
    compile_count_toy,
    run_toy,
)

from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    Clc,
    CtaPair,
    DynamicPersistent,
    OneShot,
    Policy,
    ProtocolError,
    Raster,
    SingleCta,
    Readiness,
    StaticPersistent,
    Tiles,
    TileStream,
    WarpSpecialized,
    role,
)


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("scheduling tests require SM100-family (CLC)")


# --------------------------------------------------------------------------- #
# Custom policies, written against the public protocol only.
# --------------------------------------------------------------------------- #


@cute.jit
def _countdown(index, problem):
    """``index`` while it is a valid index, else the terminal index ``total``."""
    return cutlass.select_(index >= 0, index, problem.total)


@dataclass(frozen=True)
class ReverseStatic(Policy):
    """Static persistent from the back: unit ``b`` runs ``total-1-b, total-1-b-U, ...``.
    Computed locally in every role."""

    def first(self, problem):
        return _countdown(problem.total - 1 - problem.first_unit(), problem)

    def step(self, index, problem):
        return _countdown(index - problem.units(), problem)


@dataclass(frozen=True)
class ReverseBroadcast(Policy):
    """The same sequence, claimed on the scheduler warp and broadcast (stateful through the
    previous index ``claim`` receives)."""

    stages: int = 2
    broadcast = True

    def first(self, problem):
        return _countdown(problem.total - 1 - problem.first_unit(), problem)

    def claim(self, fetch, problem, index):
        return _countdown(index - problem.units(), problem)


def reverse_static_items(total: int, units: int, unit: int) -> list[int]:
    return list(range(total - 1 - unit, -1, -units))


# --------------------------------------------------------------------------- #
# Host-side contract
# --------------------------------------------------------------------------- #


def test_custom_protocol_errors_at_declaration():
    class NoStep(Policy):
        pass

    class NoClaim(Policy):
        broadcast = True
        stages = 2

    with pytest.raises(TypeError, match="local policy but does not define step"):
        TileStream(policy=NoStep())
    with pytest.raises(TypeError, match="broadcast policy but does not define claim"):
        TileStream(policy=NoClaim())
    with pytest.raises(TypeError, match="TileStream.policy must be a Policy"):
        TileStream(policy=SimpleNamespace(step=None))
    with pytest.raises(TypeError, match="TileStream.order must be a WorkMapping"):
        TileStream(order=SimpleNamespace(words=3))
    with pytest.raises(TypeError, match="TileStream.ready must be a Readiness"):
        TileStream(ready=object())
    with pytest.raises(ValueError, match="broadcast policy stages"):
        TileStream(policy=ReverseBroadcast(stages=0))


@dataclass(frozen=True)
class Strided(Policy):
    """A policy with a compile-time field: the default key must name it."""

    stride: int = 1

    def step(self, index, problem):
        return index + self.stride * problem.units()


@dataclass(frozen=True)
class NoWait(Readiness):
    """A custom readiness hook (nothing to wait for)."""

    def wait(self):
        pass


def test_custom_policy_keys_and_launch_needs():
    assert TileStream(policy=ReverseStatic()).get_key() == "reversestatic0_m1"
    assert TileStream(policy=ReverseBroadcast(3)).get_key() == "reversebroadcast3_m1"
    assert TileStream(policy=Strided(2)).get_key() == "strided0_stride2_m1"
    assert TileStream(ready=NoWait()).get_key() == "oneshot0_m1_nowait"
    # Built-in keys are unchanged by the protocol (compile caches and kernel names).
    assert TileStream(order=Raster(swizzle=8), policy=Clc()).get_key() == "clc2_m8"
    assert ReverseStatic().uses_max_ctas and not ReverseStatic().uses_workspace
    with pytest.raises(ValueError, match="DynamicPersistent.*workspace"):
        TileStream(policy=DynamicPersistent()).problem((1, 1, 1))


def test_work_item_rejects_persistent_policies():
    stream = TileStream(policy=StaticPersistent())
    with pytest.raises(ProtocolError, match="non-persistent local policy"):
        stream.work_item()
    with pytest.raises(ProtocolError, match="after begin_roles"):
        TileStream().work_item()


# --------------------------------------------------------------------------- #
# GPU: custom policies on the schedule toy (single CTA) and the count toy (pairs)
# --------------------------------------------------------------------------- #

CUSTOM = {"reverse_static": ReverseStatic(), "reverse_broadcast": ReverseBroadcast(stages=2)}


@pytest.mark.parametrize("ext,max_ctas", [((5, 3, 1), 148), ((13, 29, 3), 148), ((40, 40, 1), 7)])
@pytest.mark.parametrize("policy", list(CUSTOM))
def test_custom_policy_runs_its_sequence(requires_blackwell, policy, ext, max_ctas):
    op = ScheduleToy(CUSTOM[policy], order=Raster())
    log, hits, _ = run_toy(op, ext, max_ctas, reps=2)
    assert (hits == 1).all(), f"tiles missed or repeated: {hits}"
    assert torch.equal(log[0], log[1])
    total = ext[0] * ext[1] * ext[2]
    units = min(total, max_ctas)
    for b in range(units):
        want = reverse_static_items(total, units, b)[: ScheduleToy.MAX_ITEMS]
        assert log[0, b, : len(want)].tolist() == want, b


@pytest.mark.parametrize("ready", [None, NoWait()], ids=["no_ready", "custom_ready"])
@pytest.mark.parametrize("policy", list(CUSTOM))
def test_custom_policy_with_skips(requires_blackwell, policy, ready):
    op = ScheduleToy(CUSTOM[policy], order=SkipOdd(swizzle=2), mode="skip", ready=ready)
    ext = (9, 7, 2)
    log, hits, _ = run_toy(op, ext, 16, reps=2)
    assert torch.equal(log[0], log[1])
    want = [
        0 if (m + n) % 2 else 1
        for li in range(ext[2])
        for n in range(ext[1])
        for m in range(ext[0])
    ]
    assert hits[0].tolist() == want and hits[1].tolist() == want


# --------------------------------------------------------------------------- #
# Pairs, empty spaces and overflow for custom policies and adapters
# --------------------------------------------------------------------------- #


def count_cases():
    """name -> (toy class, policy, order, overflow detected at launch)."""
    cases = {
        "reverse_static": (CountToy, ReverseStatic(), Raster(along="n", swizzle=3), True),
        "reverse_broadcast": (CountToy, ReverseBroadcast(), Raster(along="n", swizzle=3), True),
    }
    return cases


@pytest.mark.parametrize("cluster", [SingleCta(), CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize("case", list(count_cases()))
def test_swapped_schedules_cover_empty_and_overflowing_spaces(requires_blackwell, case, cluster):
    """Every CTA of a unit sees each item; empty spaces run nothing (and never divide by zero in
    the entry point); spaces past Int32 are rejected at launch (custom ``reach`` default and
    CUTLASS's Int64 ``size64``)."""
    toy, policy, order, rejects_overflow = count_cases()[case]
    op = toy(policy, order, cluster)
    fn = compile_count_toy(op)
    ws, _ = op.tiles.launch_args("cuda")
    for space in EMPTY_SPACES:
        assert _count_items(fn, ws, space, 148) == [0, 0], space
    for space in OVERFLOWING_SPACES if rejects_overflow else []:
        with pytest.raises(Exception, match="CUDA launch failed"):
            _count_items(fn, ws, space, 148)
    want = 5 * 3 * 2 * cluster.size
    assert _count_items(fn, ws, (5, 3, 2), 4) == [want, want]


# --------------------------------------------------------------------------- #
# Raw roles: the OneShot work item without a cursor
# --------------------------------------------------------------------------- #


class RawItemToy(WarpSpecialized):
    """A managed role (warp 0) loops the stream; a raw role (warps 1-2) reads its work item with
    ``self.tiles.work_item()``. Both count each tile in ``hits[role, tile]``."""

    def __init__(self, order, mode: str, cluster=None):
        self.mode, self.cluster = mode, cluster or SingleCta()
        self.tiles = TileStream(order=order, policy=OneShot())

    def get_name(self) -> str:
        return f"warpdrive_raw_item_{self.tiles.get_key()}_{self.mode}_cs{self.cluster.size}"

    @cute.jit
    def _hit(self, r: int, item, ctx):
        m, n, li = item
        if cute.arch.lane_idx() == 0 and cute.arch.warp_idx() == r:
            tile = (li * ctx.en + n) * ctx.em + m
            cute.arch.atomic_add(ctx.hits.iterator + (r * ctx.hits.shape[1] + tile), Int32(1))

    @role(warps=0)
    @cute.jit
    def managed(self, tiles: Tiles, ctx):
        while tiles.valid:
            self._hit(0, tiles.work, ctx)
            tiles.advance()
        return tiles

    @role(warps=range(1, 3), raw=True)
    @cute.jit
    def raw(self, ctx):
        valid, item = self.tiles.work_item()
        if valid:
            self._hit(1, item, ctx)

    @cute.kernel
    def kernel(self, hits: cute.Tensor, em: Int32, en: Int32, work):
        self.begin_roles(tiles=work)
        self.run_roles(SimpleNamespace(hits=hits, em=em, en=en))

    @cute.jit
    def __call__(self, hits: cute.Tensor, em: Int32, en: Int32, el: Int32, stream: cuda.CUstream):
        work = self.tiles.problem((em, en, el))
        self.kernel.set_name_prefix(self.get_name())
        cfg = self.launch_config(work)
        if cutlass.const_expr(self.mode == "1d"):  # a hand-built (total * cs, 1, 1) grid
            cfg["grid"] = (cutlass.max(work.total, 1) * self.cluster.size, 1, 1)
        self.kernel(hits, em, en, work).launch(**cfg, stream=stream)


@pytest.mark.parametrize(
    "order, mode, ext, cluster",
    [
        (Raster(), "grid", (5, 3, 2), SingleCta()),
        (Raster(), "grid", (2, 65536, 1), SingleCta()),  # n_tiles past gridDim.y: 1-D fallback
        (Raster(), "1d", (5, 3, 2), SingleCta()),
        (Raster(), "grid", (3, 70000, 1), CtaPair()),
        (Raster(swizzle=3, reverse="m"), "grid", (7, 5, 2), CtaPair()),
        (SkipOdd(swizzle=2), "grid", (9, 7, 2), SingleCta()),
        (Raster(), "grid", (0, 3, 1), SingleCta()),  # empty: one unit, no valid item
    ],
    ids=["tile_grid", "fallback", "hand_1d", "pair_fallback", "pair_swizzle", "skip",
         "empty"],
)  # fmt: skip
def test_raw_role_work_item(requires_blackwell, order, mode, ext, cluster):
    op = RawItemToy(order, mode, cluster)
    em, en, el = ext
    total = max(em * en * el, 0)
    hits = torch.zeros(2, max(total, 1), dtype=torch.int32, device="cuda")
    fn = compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (2, cute.sym_int())),
        Int32(1),
        Int32(1),
        Int32(1),
        name=op.get_name(),
    )
    fn(hits, em, en, el)
    torch.cuda.synchronize()
    hits = hits.cpu()[:, :total]
    kept = torch.tensor(
        [
            0 if isinstance(order, SkipOdd) and order.skip((m, n, li), ext) else cluster.size
            for li in range(el)
            for n in range(en)
            for m in range(em)
        ],
        dtype=torch.int32,
    )
    assert torch.equal(hits[0], kept) and torch.equal(hits[1], kept)


# --------------------------------------------------------------------------- #
# The persistent GEMM: one argument swaps the schedule.
# --------------------------------------------------------------------------- #


def gemm_schedules():
    schedules = {
        "oneshot": (OneShot(), None),
        "static": (StaticPersistent(), None),
        "dynamic": (DynamicPersistent(), None),
        "clc": (Clc(), None),
        "reverse_static": (ReverseStatic(), None),
        "reverse_broadcast": (ReverseBroadcast(), None),
    }
    return schedules


GEMM_SCHEDULES = list(gemm_schedules())


@pytest.mark.parametrize("cluster", [None, CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize("schedule", GEMM_SCHEDULES)
def test_persistent_gemm_swaps_schedules(requires_blackwell, schedule, cluster):
    policy, order = gemm_schedules()[schedule]
    for m, n, k, max_ctas in [
        (300, 520, 136, None),
        (4097, 2056, 520, None),
        (1100, 1032, 200, 5),
    ]:
        _check_gemm(policy, m, n, k, max_ctas=max_ctas, order=order, cluster=cluster)
