"""The public scheduler interface: custom schedules, external scheduler adapters, and the raw-role
work-item accessor, swapped into the same kernels with one argument."""

import importlib
import sys
import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass_static_persistent_tile_scheduler import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
)
from test_cute_warpdrive_scheduling import (
    EMPTY_SPACES,
    OVERFLOWING_SPACES,
    CountToy,
    ScheduleToy,
    SkipOdd,
    _check_gemm,
    _count_items,
    compile_count_toy,
    compile_toy,
    run_toy,
)

from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    ClusterLaunchControl,
    CtaPair,
    CutlassRaster,
    AtomicPersistent,
    Fa4Mapping,
    OneShot,
    IndexSchedule,
    ProtocolError,
    Raster,
    SingleCta,
    Readiness,
    StaticPersistent,
    WorkCursor,
    WorkStream,
    WarpSpecialized,
    role,
)

# The vendored CUTLASS oracle still creates V1 fast-divmod divisors (the adapter does not).
pytestmark = pytest.mark.filterwarnings(
    "ignore:FastDivmodDivisor is deprecated:DeprecationWarning"
)


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("scheduling tests require SM100-family (CLC)")


def fa4_tile_scheduler():
    return pytest.importorskip("flash_attn.cute.tile_scheduler")


# --------------------------------------------------------------------------- #
# Custom schedules, written against the public protocol only.
# --------------------------------------------------------------------------- #


@cute.jit
def _countdown(index, problem):
    """``index`` while it is a valid index, else the terminal index ``total``."""
    return cutlass.select_(index >= 0, index, problem.total)


@dataclass(frozen=True)
class ReverseStatic(IndexSchedule):
    """Static persistent from the back: unit ``b`` runs ``total-1-b, total-1-b-U, ...``.
    Computed locally in every role."""

    def first_index(self, problem):
        return _countdown(problem.total - 1 - problem.first_unit(), problem)

    def next_index(self, index, problem):
        return _countdown(index - problem.units(), problem)


@dataclass(frozen=True)
class ReverseBroadcast(IndexSchedule):
    """The same sequence, claimed on the scheduler warp and broadcast (stateful through the
    previous index ``claim_next`` receives)."""

    stages: int = 2
    broadcast = True

    def first_index(self, problem):
        return _countdown(problem.total - 1 - problem.first_unit(), problem)

    def claim_next(self, fetch, problem, index):
        return _countdown(index - problem.units(), problem)


def reverse_static_items(total: int, units: int, unit: int) -> list[int]:
    return list(range(total - 1 - unit, -1, -units))


# --------------------------------------------------------------------------- #
# Host-side contract
# --------------------------------------------------------------------------- #


def test_custom_protocol_errors_at_declaration():
    class NoStep(IndexSchedule):
        pass

    class NoClaim(IndexSchedule):
        broadcast = True
        stages = 2

    with pytest.raises(TypeError, match="local schedule but does not define next_index"):
        WorkStream(schedule=NoStep())
    with pytest.raises(TypeError, match="broadcast schedule but does not define claim_next"):
        WorkStream(schedule=NoClaim())
    with pytest.raises(
        TypeError, match="WorkStream.schedule: expected an instance of IndexSchedule"
    ):
        WorkStream(schedule=SimpleNamespace(next_index=None))
    with pytest.raises(TypeError, match="WorkStream.mapping: expected an instance of WorkMap"):
        WorkStream(mapping=SimpleNamespace(words=3))
    with pytest.raises(TypeError, match="WorkStream.ready: expected an instance of Readiness"):
        WorkStream(ready=object())
    with pytest.raises(ValueError, match="broadcast schedule stages"):
        WorkStream(schedule=ReverseBroadcast(stages=0))


@dataclass(frozen=True)
class Strided(IndexSchedule):
    """A schedule with a compile-time field: the default key must name it."""

    stride: int = 1

    def next_index(self, index, problem):
        return index + self.stride * problem.units()


@dataclass(frozen=True)
class NoWait(Readiness):
    """A custom readiness hook (nothing to wait for)."""

    def wait(self):
        pass


def test_custom_schedule_keys_and_launch_needs():
    assert WorkStream(schedule=ReverseStatic()).get_key() == "reversestatic0_m1"
    assert WorkStream(schedule=ReverseBroadcast(3)).get_key() == "reversebroadcast3_m1"
    assert WorkStream(schedule=Strided(2)).get_key() == "strided0_stride2_m1"
    assert WorkStream(ready=NoWait()).get_key() == "oneshot0_m1_nowait"
    # Built-ins take the same default key as custom schedules.
    stream = WorkStream(mapping=Raster(swizzle=8), schedule=ClusterLaunchControl())
    assert stream.get_key() == "clusterlaunchcontrol2_m8"
    assert ReverseStatic().uses_max_ctas and not ReverseStatic().uses_workspace
    with pytest.raises(ValueError, match="AtomicPersistent.*workspace"):
        WorkStream(schedule=AtomicPersistent()).bind((1, 1, 1))


def test_adapters_import_without_flash_attn(monkeypatch):
    """warpdrive imports without flash_attn; only constructing Fa4Mapping needs it."""
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.tile_scheduler", None)
    for name in [m for m in sys.modules if m.startswith("transformer_nuggets.cute.warpdrive")]:
        monkeypatch.delitem(sys.modules, name)
    warpdrive = importlib.import_module("transformer_nuggets.cute.warpdrive")
    warpdrive.WorkStream(mapping=warpdrive.CutlassRaster(swizzle=4))
    with pytest.raises(ImportError, match="flash-attn-4"):
        warpdrive.Fa4Mapping(object)


def test_adapter_validation():
    with pytest.raises(ValueError, match="along"):
        CutlassRaster(along="k")
    with pytest.raises(ValueError, match="swizzle"):
        CutlassRaster(swizzle=0)
    assert CutlassRaster(swizzle=8).skips and not CutlassRaster().skips
    ts = fa4_tile_scheduler()
    with pytest.raises(ValueError, match="one linear index"):
        Fa4Mapping(ts.SingleTileScheduler)


def test_single_item_rejects_persistent_schedules():
    stream = WorkStream(schedule=StaticPersistent())
    with pytest.raises(ProtocolError, match="non-persistent local schedule"):
        stream.single_item()
    with pytest.raises(ProtocolError, match="after begin_roles"):
        WorkStream().single_item()


# --------------------------------------------------------------------------- #
# GPU: custom schedules on the schedule toy (single CTA) and the count toy (pairs)
# --------------------------------------------------------------------------- #

CUSTOM = {"reverse_static": ReverseStatic(), "reverse_broadcast": ReverseBroadcast(stages=2)}


@pytest.mark.parametrize("ext,max_ctas", [((5, 3, 1), 148), ((13, 29, 3), 148), ((40, 40, 1), 7)])
@pytest.mark.parametrize("schedule", list(CUSTOM))
def test_custom_schedule_runs_its_sequence(requires_blackwell, schedule, ext, max_ctas):
    op = ScheduleToy(CUSTOM[schedule], mapping=Raster())
    log, hits, _ = run_toy(op, ext, max_ctas, reps=2)
    assert (hits == 1).all(), f"tiles missed or repeated: {hits}"
    assert torch.equal(log[0], log[1])
    total = ext[0] * ext[1] * ext[2]
    units = min(total, max_ctas)
    for b in range(units):
        want = reverse_static_items(total, units, b)[: ScheduleToy.MAX_ITEMS]
        assert log[0, b, : len(want)].tolist() == want, b


@pytest.mark.parametrize("ready", [None, NoWait()], ids=["no_ready", "custom_ready"])
@pytest.mark.parametrize("schedule", list(CUSTOM))
def test_custom_schedule_with_skips(requires_blackwell, schedule, ready):
    op = ScheduleToy(CUSTOM[schedule], mapping=SkipOdd(swizzle=2), mode="skip", ready=ready)
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
# External schedulers: item order equals the library's own sequence.
# --------------------------------------------------------------------------- #

MAX_REF = ScheduleToy.MAX_ITEMS


class ExternalReference:
    """Runs an external scheduler the way its library does (``create``,
    ``initial_work_tile_info``, ``advance_to_next_work``) and logs each unit's raw
    ``(m, n, l)`` coordinates in ``log[unit, k]``. CUTLASS: the vendored BSD-3 CuTeDSL helper
    (``cutlass_static_persistent_tile_scheduler.py``) over the space's tiles. FA4: its params
    built here from ``TileSchedulerArguments`` the way FA4's bf16 head-dim-128 forward builds
    them, independently of the adapter's ``bind``; ``lpt`` holds the LPT flags."""

    def __init__(self, adapter, kind: str, seqlen_k: int = 0, lpt=(True, False)):
        self.adapter, self.kind, self.seqlen_k, self.lpt = adapter, kind, seqlen_k, lpt
        self.unit_axis = 2 if kind == "cutlass" else 0  # CUTLASS launches (1, 1, units)

    def fa4_params(self, em, en, el):
        ts = fa4_tile_scheduler()
        args = ts.TileSchedulerArguments(
            num_block=em,
            num_head=en,
            num_batch=el,
            num_splits=Int32(1),
            seqlen_k=Int32(self.seqlen_k),
            headdim=128,
            headdim_v=128,
            total_q=Int32(0),
            tile_shape_mn=(128, 128),
            element_size=2,
            lpt=self.lpt[0],
            lpt_global=self.lpt[1],
        )
        return self.adapter.scheduler.to_underlying_arguments(args)

    def _create(self, params):
        if self.kind == "cutlass":
            return StaticPersistentTileScheduler.create(
                params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
        return self.adapter.scheduler.create(params)

    @cute.kernel
    def kernel(self, log: cute.Tensor, params):
        unit = cute.arch.block_idx()[self.unit_axis]
        scheduler = self._create(params)
        work = scheduler.initial_work_tile_info()
        k = Int32(0)
        while work.is_valid_tile:
            if (cute.arch.thread_idx()[0] == 0) & (k < MAX_REF):
                for a in cutlass.range_constexpr(3):
                    log[unit, k, a] = Int32(work.tile_idx[a])
            k += 1
            scheduler.advance_to_next_work()
            work = scheduler.get_current_work()

    @cute.jit
    def __call__(self, log: cute.Tensor, em: Int32, en: Int32, el: Int32, units: Int32,
                 stream: cuda.CUstream):  # fmt: skip
        if cutlass.const_expr(self.kind == "cutlass"):
            a = self.adapter
            params = PersistentTileSchedulerParams(
                (em, en, el), (1, 1, 1), a.swizzle, a.along == "m"
            )
            grid = (1, 1, units)
        else:
            params = self.fa4_params(em, en, el)
            grid = (units, 1, 1)
        self.kernel(log, params).launch(grid=grid, block=(32, 1, 1), stream=stream)


def reference_sequences(ref: ExternalReference, ext, units: int) -> list[list[int]]:
    """Per unit, the tile ids the external scheduler visits inside the space (CUTLASS hands
    out padded swizzle tiles too; the adapter skips them)."""
    em, en, el = ext
    log = torch.full((units, MAX_REF, 3), -1, dtype=torch.int32, device="cuda")
    i32 = cutlass.Int32
    fn = compile_tvm_ffi(
        ref,
        make_fake_compact_tensor(i32, (cute.sym_int(), MAX_REF, 3)),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        name=f"warpdrive_external_reference_{ref.adapter.get_key()}",
    )
    fn(log, em, en, el, units)
    torch.cuda.synchronize()
    out = []
    for row in log.cpu().tolist():
        out.append([(li * en + n) * em + m for m, n, li in row if m >= 0 and m < em and n < en])
    return out


def padded_count(adapter: CutlassRaster, ext) -> int:
    em, en, el = ext
    s = adapter.swizzle
    if adapter.along == "m":
        en = -(-en // s) * s
    else:
        em = -(-em // s) * s
    return em * en * el


def check_external_order(op, ref, ext, max_ctas, items):
    """Every tile reaches both roles once, and each unit's sequence (up to ``MAX_ITEMS``)
    equals the external scheduler's."""
    log, hits, _ = run_toy(op, ext, max_ctas, items=items)
    assert (hits == 1).all(), f"tiles missed or repeated: {hits}"
    assert torch.equal(log[0], log[1])
    units = max(min(items, max_ctas), 1) if op.work.schedule.uses_max_ctas else max(items, 1)
    for b, want in enumerate(reference_sequences(ref, ext, units)):
        want = want[:MAX_REF]
        got = [t for t in log[0, b].tolist() if t >= 0]
        assert got == want, (b, got, want)


EXTERNAL_SHAPES = [((5, 3, 1), 148), ((13, 29, 3), 148), ((7, 17, 2), 5)]


@pytest.mark.parametrize("ext,max_ctas", EXTERNAL_SHAPES)
@pytest.mark.parametrize(
    "adapter",
    [
        CutlassRaster(),
        CutlassRaster(along="n"),
        CutlassRaster(swizzle=3),
        CutlassRaster(swizzle=8, along="n"),
    ],
    ids=lambda a: a.get_key(),
)
def test_cutlass_static_persistent_order(requires_blackwell, adapter, ext, max_ctas):
    op = ScheduleToy(StaticPersistent(), mapping=adapter, mode="ext")
    ref = ExternalReference(adapter, "cutlass")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_toy(op)
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)], caught
    check_external_order(op, ref, ext, max_ctas, padded_count(adapter, ext))


@pytest.mark.parametrize("ext,max_ctas", EXTERNAL_SHAPES)
def test_fa4_static_persistent_order(requires_blackwell, ext, max_ctas):
    ts = fa4_tile_scheduler()
    adapter = Fa4Mapping(ts.StaticPersistentTileScheduler)
    op = ScheduleToy(StaticPersistent(), mapping=adapter, mode="ext")
    ref = ExternalReference(adapter, "fa4")
    check_external_order(op, ref, ext, max_ctas, ext[0] * ext[1] * ext[2])


# 4096 keys of head dim 128 in bf16 give 16-head L2 sections, so 29 * 3 heads exercise FA4's
# residual section.
SEQLEN_K = 4096


class LptToy(ScheduleToy):
    """Binds ``(num_block, num_head, num_batch, seqlen_k)``."""

    seqlen_k = SEQLEN_K

    def space(self, em, en, el):
        return (em, en, el, Int32(self.seqlen_k))


class LptCountToy(CountToy):
    seqlen_k = SEQLEN_K

    def space(self, em, en, el):
        return (em, en, el, Int32(self.seqlen_k))


class EmptyKvCountToy(LptCountToy):
    seqlen_k = 0  # FA4 would divide by the zero K/V bytes per head


@pytest.mark.parametrize("lpt", [(True, False), (False, False), (True, True)], ids=str)
@pytest.mark.parametrize("ext", [(5, 3, 1), (13, 29, 3), (2, 40, 1)])
def test_fa4_lpt_order(requires_blackwell, ext, lpt):
    ts = fa4_tile_scheduler()
    adapter = Fa4Mapping(ts.SingleTileLPTScheduler, lpt=lpt[0], lpt_global=lpt[1])
    op = LptToy(OneShot(), mapping=adapter, mode="ext")
    ref = ExternalReference(adapter, "fa4", SEQLEN_K, lpt)
    check_external_order(op, ref, ext, 148, ext[0] * ext[1] * ext[2])


def test_fa4_adapter_spaces_and_keys():
    ts = fa4_tile_scheduler()
    with pytest.raises(ValueError, match="seqlen_k"):
        Fa4Mapping(ts.SingleTileLPTScheduler).bind((1, 2, 3))
    with pytest.raises(ValueError, match="num_batch\\)"):
        Fa4Mapping(ts.StaticPersistentTileScheduler).bind((1, 2, 3, 4))
    with pytest.raises(ValueError, match="head_dim"):
        Fa4Mapping(ts.SingleTileLPTScheduler, head_dim=0)
    # Every compile-time argument of the LPT scheduler is in the key.
    keys = {
        Fa4Mapping(ts.SingleTileLPTScheduler, **kw).get_key()
        for kw in [{}, {"lpt": False}, {"lpt_global": True}, {"head_dim": 64}, {"element_size": 1}]
    }
    assert len(keys) == 5


@pytest.mark.parametrize("schedule", ["atomic", "clc", "reverse_broadcast"])
def test_external_mapping_under_other_schedules(requires_blackwell, schedule):
    """An adapter is a mapping: any schedule distributes its items."""
    schedules = {"atomic": AtomicPersistent(), "clc": ClusterLaunchControl(), **CUSTOM}
    adapter = CutlassRaster(swizzle=8)
    op = ScheduleToy(schedules[schedule], mapping=adapter, mode="ext")
    ext = (13, 29, 3)
    _, hits, ws = run_toy(op, ext, 16, reps=2, items=padded_count(adapter, ext))
    assert (hits == 1).all() and ws[:2].tolist() == [0, 0]


# --------------------------------------------------------------------------- #
# Pairs, empty spaces and overflow for custom schedules and adapters
# --------------------------------------------------------------------------- #


def fa4_or_none():
    try:
        return importlib.import_module("flash_attn.cute.tile_scheduler")
    except ImportError:
        return None


def count_cases():
    """name -> (toy class, schedule, mapping)."""
    cases = {
        "reverse_static": (CountToy, ReverseStatic(), Raster(along="n", swizzle=3)),
        "reverse_broadcast": (CountToy, ReverseBroadcast(), Raster(along="n", swizzle=3)),
        "cutlass_static": (CountToy, StaticPersistent(), CutlassRaster(swizzle=4)),
        "cutlass_clc": (CountToy, ClusterLaunchControl(), CutlassRaster(swizzle=4, along="n")),
    }
    ts = fa4_or_none()
    if ts is not None:
        static = Fa4Mapping(ts.StaticPersistentTileScheduler)
        lpt_global = Fa4Mapping(ts.SingleTileLPTScheduler, lpt_global=True)
        lpt = Fa4Mapping(ts.SingleTileLPTScheduler)
        cases["fa4_static"] = (CountToy, StaticPersistent(), static)
        cases["fa4_lpt_global"] = (LptCountToy, OneShot(), lpt_global)
        cases["fa4_lpt_empty_kv"] = (EmptyKvCountToy, OneShot(), lpt)
    return cases


@pytest.mark.parametrize("cluster", [SingleCta(), CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize("case", list(count_cases()))
def test_swapped_schedules_cover_empty_and_overflowing_spaces(requires_blackwell, case, cluster):
    """Every CTA of a unit sees each item; empty spaces (and an empty KV) run nothing or never
    divide by zero in the entry point; spaces past Int32 are rejected at launch (custom
    ``index_limit`` default and the adapters' Int64 ``count64``)."""
    toy, schedule, mapping = count_cases()[case]
    op = toy(schedule, mapping, cluster)
    fn = compile_count_toy(op)
    ws, _ = op.work.launch_args("cuda")
    for space in EMPTY_SPACES:
        assert _count_items(fn, ws, space, 148) == [0, 0], space
    for space in OVERFLOWING_SPACES:
        with pytest.raises(Exception, match="CUDA launch failed"):
            _count_items(fn, ws, space, 148)
    want = 5 * 3 * 2 * cluster.size
    assert _count_items(fn, ws, (5, 3, 2), 4) == [want, want]


# --------------------------------------------------------------------------- #
# Raw roles: the OneShot work item without a cursor
# --------------------------------------------------------------------------- #


class RawItemToy(WarpSpecialized):
    """A managed role (warp 0) loops the stream; a raw role (warps 1-2) reads its work item with
    ``self.work.single_item()``. Both count each tile in ``hits[role, tile]``."""

    def __init__(self, mapping, mode: str, cluster=None):
        self.mode, self.cluster = mode, cluster or SingleCta()
        self.work = WorkStream(mapping=mapping, schedule=OneShot())

    def get_name(self) -> str:
        return f"warpdrive_raw_item_{self.work.get_key()}_{self.mode}_cs{self.cluster.size}"

    @cute.jit
    def _hit(self, r: int, item, ctx):
        m, n, li = item
        if cute.arch.lane_idx() == 0 and cute.arch.warp_idx() == r:
            tile = (li * ctx.en + n) * ctx.em + m
            cute.arch.atomic_add(ctx.hits.iterator + (r * ctx.hits.shape[1] + tile), Int32(1))

    @role(warps=0)
    @cute.jit
    def managed(self, work: WorkCursor, ctx):
        while work.valid:
            self._hit(0, work.item, ctx)
            work.advance()
        return work

    @role(warps=range(1, 3), raw=True)
    @cute.jit
    def raw(self, ctx):
        valid, item = self.work.single_item()
        if valid:
            self._hit(1, item, ctx)

    @cute.kernel
    def kernel(self, hits: cute.Tensor, em: Int32, en: Int32, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(hits=hits, em=em, en=en))

    @cute.jit
    def __call__(self, hits: cute.Tensor, em: Int32, en: Int32, el: Int32, stream: cuda.CUstream):
        bound = self.work.bind((em, en, el))
        self.kernel.set_name_prefix(self.get_name())
        cfg = self.launch_config(bound)
        if cutlass.const_expr(self.mode == "1d"):  # a hand-built (total * cs, 1, 1) grid
            cfg["grid"] = (cutlass.max(bound.total, 1) * self.cluster.size, 1, 1)
        self.kernel(hits, em, en, bound).launch(**cfg, stream=stream)


@pytest.mark.parametrize(
    "mapping, mode, ext, cluster",
    [
        (Raster(), "grid", (5, 3, 2), SingleCta()),
        (Raster(), "grid", (2, 65536, 1), SingleCta()),  # n_tiles past gridDim.y: 1-D fallback
        (Raster(), "1d", (5, 3, 2), SingleCta()),
        (Raster(), "grid", (3, 70000, 1), CtaPair()),
        (Raster(swizzle=3, reverse="m"), "grid", (7, 5, 2), CtaPair()),
        (SkipOdd(swizzle=2), "grid", (9, 7, 2), SingleCta()),
        (CutlassRaster(swizzle=4), "grid", (7, 5, 2), SingleCta()),
        (Raster(), "grid", (0, 3, 1), SingleCta()),  # empty: one unit, no valid item
    ],
    ids=["tile_grid", "fallback", "hand_1d", "pair_fallback", "pair_swizzle", "skip", "cutlass",
         "empty"],
)  # fmt: skip
def test_raw_role_single_item(requires_blackwell, mapping, mode, ext, cluster):
    op = RawItemToy(mapping, mode, cluster)
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
            0 if isinstance(mapping, SkipOdd) and mapping.skip((m, n, li), ext) else cluster.size
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
    ts = fa4_or_none()
    schedules = {
        "oneshot": (OneShot(), None),
        "static": (StaticPersistent(), None),
        "atomic": (AtomicPersistent(), None),
        "clc": (ClusterLaunchControl(), None),
        "reverse_static": (ReverseStatic(), None),
        "reverse_broadcast": (ReverseBroadcast(), None),
        "cutlass_static": (StaticPersistent(), CutlassRaster(swizzle=8)),
        "cutlass_clc": (ClusterLaunchControl(), CutlassRaster(swizzle=2, along="n")),
    }
    if ts is not None:
        adapter = Fa4Mapping(ts.StaticPersistentTileScheduler)
        schedules["fa4_static"] = (StaticPersistent(), adapter)
        schedules["fa4_atomic"] = (AtomicPersistent(), adapter)
    return schedules


GEMM_SCHEDULES = list(gemm_schedules())


@pytest.mark.parametrize("cluster", [None, CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize("case", GEMM_SCHEDULES)
def test_persistent_gemm_swaps_schedules(requires_blackwell, case, cluster):
    schedule, mapping = gemm_schedules()[case]
    for m, n, k, max_ctas in [
        (300, 520, 136, None),
        (4097, 2056, 520, None),
        (1100, 1032, 200, 5),
    ]:
        _check_gemm(schedule, m, n, k, max_ctas=max_ctas, mapping=mapping, cluster=cluster)
