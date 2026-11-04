import math
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    Consumer,
    Instance,
    Producer,
    ProtocolError,
    Ring,
    Slots,
    Threads,
    TileStream,
    Tma,
    TmemPlan,
    Turn,
    Turnstile,
    Umma,
    WarpSpecialized,
    role,
)
from transformer_nuggets.cute.warpdrive.examples.gemm import gemm
from transformer_nuggets.cute.warpdrive.examples.persistent_gemm import persistent_gemm


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("warpdrive examples require SM100-family tcgen05")


# --------------------------------------------------------------------------- #
# A two-role toy kernel: producer warp 0 -> consumer warp 1 over two fixed slots with a
# layer-owned SMEM payload. ``mode`` injects protocol violations at trace time.
# --------------------------------------------------------------------------- #


class SlotsToy(WarpSpecialized):
    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.ch = Slots(2, Threads, Threads, producer_role="prod", payload=(cutlass.Int32, (32,)))

    def get_name(self) -> str:
        return f"warpdrive_toy_{self.mode}"

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(self.mode == "leak"):
            ch.acquire(0)
        if cutlass.const_expr(self.mode == "cross_region"):
            early = ch.acquire(0)
        for i in cutlass.range(ctx.n, unroll=1):
            for j in cutlass.range_constexpr(2):
                if cutlass.const_expr(self.mode == "cross_region" and j == 0):
                    early.commit()
                else:
                    h = ch.acquire(j)
                    h.data[lane] = (2 * i + j) * 32 + lane
                    h.commit()
                    if cutlass.const_expr(self.mode == "double"):
                        h.commit()
                    if cutlass.const_expr(self.mode == "wrong_side"):
                        h.release()

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        lane = cute.arch.lane_idx()
        acc = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            for j in cutlass.range_constexpr(2):
                with ch.consume(j) as h:
                    acc = acc + h.data[lane]
        ctx.out[lane] = acc

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(out, n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


class TurnToy(WarpSpecialized):
    """Two instances of one role append their seat to a log, one turn at a time."""

    def __init__(self, enabled: bool = True):
        self.ts = Turnstile(enabled=enabled)

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def worker(self, seat: Instance, ts: Turn, ctx):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ts.turn():
                if cute.arch.lane_idx() == 0:
                    pos = ctx.log[0]
                    ctx.log[pos + 1] = seat
                    ctx.log[0] = pos + 1
                    cute.arch.fence_acq_rel_cta()

    @cute.kernel
    def kernel(self, log: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(log=log, n=n))

    @cute.jit
    def __call__(self, log: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_turn_toy")
        self.kernel(log, n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


class BoundToy(WarpSpecialized):
    """Two producer instances, each bound to its own slot; one consumer reads both slots."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.ch = Slots(2, Threads, Threads, payload=(cutlass.Int32, (32,)))

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def prod(self, stage: Instance, ch: Producer[Instance], ctx):
        slot = 1 - stage if cutlass.const_expr(self.mode == "wrong_slot") else stage
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce(slot) as h:
                h.data[cute.arch.lane_idx()] = (stage + 1) * (i + 1)

    @role(warps=2)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        lane = cute.arch.lane_idx()
        for _ in cutlass.range(ctx.n, unroll=1):
            for j in cutlass.range_constexpr(2):
                with ch.consume(j) as h:
                    ctx.out[lane + 32 * j] = ctx.out[lane + 32 * j] + h.data[lane]

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_bound_toy")
        self.kernel(out, n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


class HoleToy(WarpSpecialized):
    """Producer warp 0 -> consumer warps 4-7, plus an idle role on warps 8-11. Warps 1-3 belong
    to no role but share warp 0's warpgroup, so they must take its register budget
    (setmaxnreg is .aligned per warpgroup). 12 warps: default 168 registers, so the producer
    and idle role decrease and the consumer increases."""

    def __init__(self):
        self.ch = Slots(1, Threads, Threads, payload=(cutlass.Int32, (32,)))

    @role(warps=0, regs=56)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce(0) as h:
                h.data[lane] = i * 32 + lane

    @role(warps=range(4, 8), regs=232)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        lane = cute.arch.lane_idx()
        acc = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume(0) as h:
                acc = acc + h.data[lane]
        if cute.arch.warp_idx() == 4:
            ctx.out[lane] = acc

    @role(warps=range(8, 12), regs=56)
    @cute.jit
    def idle(self, ctx):
        pass

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_hole_toy")
        self.kernel(out, n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


class OverBudgetHoleToy(HoleToy):
    """Warp 0's 232 registers also bind holes 1-3: 4 warpgroups' worth over 65536."""

    @role(warps=0, regs=232)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        pass


def compile_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), name="warpdrive_toy")


@pytest.mark.parametrize("n", [1, 2, 7, 64])
def test_slots_payload_round_trip(requires_blackwell, n):
    out = torch.full((32,), -1, device="cuda", dtype=torch.int32)
    compile_toy(SlotsToy())(out, n)
    lane = torch.arange(32, device="cuda", dtype=torch.int32)
    expected = sum((2 * i + j) * 32 for i in range(n) for j in range(2)) + 2 * n * lane
    torch.testing.assert_close(out, expected.to(torch.int32))


@pytest.mark.parametrize("n", [1, 5, 33])
def test_turnstile_alternates_seats(requires_blackwell, n):
    log = torch.zeros(2 * n + 1, device="cuda", dtype=torch.int32)
    compile_toy(TurnToy())(log, n)
    assert log[0].item() == 2 * n
    assert log[1:].tolist() == [0, 1] * n


@pytest.mark.parametrize("n", [1, 9])
def test_bound_producers_fan_in(requires_blackwell, n):
    out = torch.zeros(64, device="cuda", dtype=torch.int32)
    compile_toy(BoundToy())(out, n)
    tri = n * (n + 1) // 2
    assert out[:32].tolist() == [tri] * 32 and out[32:].tolist() == [2 * tri] * 32


@pytest.mark.parametrize("n", [1, 9])
def test_register_holes_run(requires_blackwell, n):
    out = torch.full((32,), -1, device="cuda", dtype=torch.int32)
    compile_toy(HoleToy())(out, n)
    torch.cuda.synchronize()
    lane = torch.arange(32, dtype=torch.int32)
    assert torch.equal(out.cpu(), 32 * n * (n - 1) // 2 + n * lane)


def test_register_holes_take_their_warpgroup_budget():
    # 232 x 32 x (4 + 4) + 56 x 32 x 4 = 66560: counting warp 0 alone would accept it.
    with pytest.raises(ValueError, match="register budgets need 66560 > 65536"):
        OverBudgetHoleToy().describe()
    op = HoleToy()
    assert "  1-3    (padding)    56\n" in op.describe()
    assert op._ws_padding == [(range(1, 4), 56)]


def test_register_budget_branches_are_warpgroup_uniform():
    """setmaxnreg is .sync.aligned per warpgroup: the layer emits one adjustment per budget
    under a warpgroup-uniform branch, so every warpgroup lands in exactly one branch and each
    role (and hole) is dispatched inside its warpgroup's branch."""
    kernel = HoleToy()
    kernel.describe()
    groups = kernel._ws_budget_groups
    assert len({regs for regs, _, _ in groups}) == len(groups)
    covered = sorted(g for _, wgs, _ in groups for g in wgs)
    assert covered == list(range(kernel.num_warps // 4))
    for _, wgs, role_ids in groups:
        for i in role_ids:
            assert {w // 4 for w in kernel._ws_roles[i].warps} <= set(wgs)


def test_trace_time_wrong_slot(requires_blackwell):
    with pytest.raises(Exception, match=r"bound to slot 0, but acquire\(1\)"):
        compile_toy(BoundToy("wrong_slot"))


@pytest.mark.parametrize(
    "mode, message",
    [
        ("leak", r"handles never closed: ch\[0\] acquire opened at test_cute_warpdrive.py"),
        ("double", r"ch\[[01]\] acquire closed twice: commit at .*, then commit at"),
        ("cross_region", r"different dynamic region"),
        ("wrong_side", r"release\(\) on a ch\[[01]\] acquire handle"),
    ],
)
def test_trace_time_handle_errors(requires_blackwell, mode, message):
    with pytest.raises(Exception, match=message) as info:
        compile_toy(SlotsToy(mode))
    assert "ProtocolError" in repr(info.value) or isinstance(info.value, ProtocolError)


# --------------------------------------------------------------------------- #
# Role context: a typed frozen dataclass, or a namespace
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToyCtx:
    out: cute.Tensor
    n: cutlass.Int32


@dataclass
class MutableToyCtx:
    out: cute.Tensor
    n: cutlass.Int32


class TupleCtx(NamedTuple):
    out: cute.Tensor
    n: cutlass.Int32


@dataclass(frozen=True)
class OtherCtx:
    out: cute.Tensor


class TypedCtxToy(WarpSpecialized):
    """Producer warp 0 -> consumer warp 1 over one slot; both roles take ``ctx: ToyCtx``, and the
    consumer writes through it inside a dynamic branch. ``ctx`` picks what the kernel passes to
    run_roles; ``seen`` records the context type each role receives at trace time."""

    def __init__(self, ctx: str = "typed"):
        self.ctx = ctx
        self.seen = []
        self.ch = Slots(1, Threads, Threads, payload=(cutlass.Int32, (32,)))

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx: ToyCtx):
        self.seen.append(type(ctx))
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce(0) as h:
                h.data[lane] = i * 32 + lane

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx: ToyCtx):
        self.seen.append(type(ctx))
        lane = cute.arch.lane_idx()
        acc = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume(0) as h:
                acc = acc + h.data[lane]
        if ctx.n > 0:
            ctx.out[lane] = acc

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        if cutlass.const_expr(self.ctx == "typed"):
            ctx = ToyCtx(out=out, n=n)
        elif cutlass.const_expr(self.ctx == "mutable"):
            ctx = MutableToyCtx(out=out, n=n)
        elif cutlass.const_expr(self.ctx == "tuple"):
            ctx = TupleCtx(out=out, n=n)
        else:
            ctx = SimpleNamespace(out=out, n=n)
        self.run_roles(ctx)

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(f"warpdrive_ctx_toy_{self.ctx}")
        self.kernel(out, n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


class NamespaceCtxToy(TypedCtxToy):
    """TypedCtxToy's roles with an unannotated context, as kernels built on SimpleNamespace
    write them."""

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        self.seen.append(type(ctx))
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce(0) as h:
                h.data[lane] = i * 32 + lane

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        self.seen.append(type(ctx))
        lane = cute.arch.lane_idx()
        acc = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume(0) as h:
                acc = acc + h.data[lane]
        if ctx.n > 0:
            ctx.out[lane] = acc


@pytest.mark.parametrize(
    "toy, seen",
    [(TypedCtxToy, ToyCtx), (NamespaceCtxToy, SimpleNamespace)],
    ids=["typed", "namespace"],
)
def test_context_reaches_roles(requires_blackwell, toy, seen):
    """A frozen dataclass reaches the roles as itself; a namespace still works, as a namespace."""
    op = toy("typed" if toy is TypedCtxToy else "namespace")
    out = torch.full((32,), -1, device="cuda", dtype=torch.int32)
    compile_toy(op)(out, 5)
    lane = torch.arange(32, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(out, 32 * 10 + 5 * lane)
    assert len(op.seen) == 2 and all(isinstance(t, type) and issubclass(t, seen) for t in op.seen)
    if toy is TypedCtxToy:
        assert op.seen == [ToyCtx, ToyCtx]


@pytest.mark.parametrize(
    "ctx, message",
    [
        (
            "namespace",
            r"run_roles got a SimpleNamespace context, but \w+\.ctx is annotated ToyCtx",
        ),
        ("mutable", r"MutableToyCtx is a mutable dataclass; declare it @dataclass\(frozen=True\)"),
        ("tuple", r"context TupleCtx has no attributes to share; pass a frozen dataclass"),
    ],
)
def test_context_type_checked_at_run_roles(requires_blackwell, ctx, message):
    toy = TypedCtxToy if ctx == "namespace" else NamespaceCtxToy  # unannotated: any type fits
    with pytest.raises(Exception, match=message) as info:
        compile_toy(toy(ctx))
    assert "ProtocolError" in repr(info.value) or isinstance(info.value, ProtocolError)


class TypoAnnotationToy(WarpSpecialized):
    def __init__(self):
        self.ab = Ring(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def load(self, ab: Producer, ctx: ToyCtx):
        pass

    @role(warps=1)
    @cute.jit
    def mma(self, ab: "Consumr", ctx: ToyCtx):  # noqa: F821 (the typo under test)
        pass


class LookalikeToy(WarpSpecialized):
    def __init__(self):
        self.tiles = TileStream()

    @role(warps=0)
    @cute.jit
    def work(self, tiles: TileStream, ctx):
        pass


class TwoContextTypesToy(TypedCtxToy):
    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx: OtherCtx):
        pass


class MutableAnnotationToy(TypedCtxToy):
    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx: MutableToyCtx):
        pass


class TwoUnannotatedToy(TypedCtxToy):
    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, extra, ctx):
        pass


@pytest.mark.parametrize(
    "toy, message",
    [
        (
            TypoAnnotationToy,
            r"role mma: parameter ab is annotated 'Consumr', which does not resolve",
        ),
        (LookalikeToy, r"parameter tiles is annotated TileStream.*did you mean `tiles: Tiles`"),
        (TwoContextTypesToy, r"unrelated types \(cons\.ctx: OtherCtx, prod\.ctx: ToyCtx\)"),
        (TwoUnannotatedToy, r"role cons: parameters \['extra', 'ctx'\] would all receive"),
        (MutableAnnotationToy, r"cons\.ctx is annotated MutableToyCtx, which is not a frozen"),
    ],
    ids=["typo", "lookalike", "two_types", "two_unannotated", "mutable_annotation"],
)
def test_unrecognized_role_parameters(toy, message):
    """Only the context-typed or single unannotated parameter receives the context; anything
    else is an error naming the parameter, not a silent context injection."""
    with pytest.raises(ProtocolError, match=message):
        toy().describe()


# --------------------------------------------------------------------------- #
# Host-side declaration checks (no compile)
# --------------------------------------------------------------------------- #


class WrongRoleToy(SlotsToy):
    def __init__(self):
        self.mode = "ok"
        self.ch = Slots(2, Threads, Threads, producer_role="cons")


class BoundSlotsToy(WarpSpecialized):
    """Two instances, each bound to its own slot of ``ch``; one consumer of both."""

    def __init__(self, n=2, producer=Threads):
        self.ch = Slots(n, producer, Threads)

    @role(warps=[range(4), range(4, 8)])
    @cute.jit
    def prod(self, stage: Instance, ch: Producer[Instance], ctx):
        pass

    @role(warps=range(8, 12))
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        pass


def test_wrong_role_on_channel():
    with pytest.raises(ProtocolError, match="producer is declared as role 'cons'"):
        WrongRoleToy().describe()


def test_bound_slot_arrival_counts():
    toy = BoundSlotsToy()
    toy.describe()
    assert toy.ch.slot_counts("producer") == [128, 128]
    assert toy.ch.slot_counts("consumer") == [128, 128]
    with pytest.raises(ProtocolError, match="slot 1 >= ch.n=1"):
        BoundSlotsToy(n=1).describe()
    with pytest.raises(ProtocolError, match="signals once"):
        BoundSlotsToy(producer=Umma).describe()


class WideRolesToy(WarpSpecialized):
    """One producer role on warps 0-3, one consumer role on warps 4-7."""

    def __init__(self, producer, consumer):
        self.ch = Ring(2, producer, consumer)

    @role(warps=range(4))
    @cute.jit
    def prod(self, ch: Producer, ctx):
        pass

    @role(warps=range(4, 8))
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        pass


@pytest.mark.parametrize(
    "producer, consumer, wide",
    [(Tma(1024), Umma, "prod"), (Umma, Threads, "prod"), (Threads, Umma, "cons")],
    ids=["tma_producer", "umma_producer", "umma_consumer"],
)
def test_single_signal_side_rejects_multi_warp_role(producer, consumer, wide):
    """Tma/Umma sides expect one arrival; each warp of a wider role would signal."""
    with pytest.raises(ProtocolError, match=f"role {wide} spans 4 warps"):
        WideRolesToy(producer, consumer).describe()
    toy = WideRolesToy(Threads, Threads)
    toy.describe()
    assert toy.ch.slot_counts("producer") == [128, 128]


def test_declaration_errors():
    with pytest.raises(ValueError, match="no commit to fence"):
        Slots(2, Tma(1024), Umma, commit_fence="smem_async")
    with pytest.raises(ValueError, match="exceeds S"):
        TmemPlan().region("S", 128, 2).alias("P", of="S", offset=96, cols=64)
    with pytest.raises(ValueError, match="needs 640 > 512"):
        _ = TmemPlan().region("S", 128, 5).alloc_cols
    plan = (
        TmemPlan().region("S", 128, 2).region("O", 128, 2).alias("P", of="S", offset=64, cols=64)
    )
    assert [plan.col("P", j) for j in range(2)] == [64, 192]
    assert [plan.col("O", j) for j in range(2)] == [256, 384]
    assert plan.alloc_cols == 512


# --------------------------------------------------------------------------- #
# GEMM example
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "m, n, k", [(128, 256, 64), (512, 512, 512), (300, 520, 136), (1, 8, 8), (1000, 1000, 1000)]
)
def test_gemm(requires_blackwell, m, n, k):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
    err = (gemm(a, b).float() - ref).abs().max().item()
    assert err <= 2 * torch_err + 1e-3


GEMMS = pytest.mark.parametrize("fn", [gemm, persistent_gemm], ids=lambda f: f.__name__)


def _out_view(buf: torch.Tensor, offset: int, m: int, n: int) -> torch.Tensor:
    return buf[offset : offset + m * n].view(m, n)


@GEMMS
@pytest.mark.parametrize("m, n, k", [(64, 24, 0), (0, 8, 64), (64, 0, 64), (0, 0, 0)])
def test_gemm_zero_work(requires_blackwell, fn, m, n, k):
    """Empty K gives zeros (no MMA writes the accumulator); empty outputs launch nothing."""
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert fn(a, b, out) is out
    torch.cuda.synchronize()
    assert torch.equal(out, torch.zeros_like(out))


@GEMMS
@pytest.mark.parametrize("alias", ["out_is_a", "out_is_b", "out_overlaps_half_of_a"])
def test_gemm_rejects_out_overlapping_operands(requires_blackwell, fn, alias):
    """512^3 has 8 output tiles: a tile's store would overwrite rows another tile still loads."""
    size = 512
    buf = torch.randn(3 * size * size, device="cuda", dtype=torch.bfloat16)
    a = _out_view(buf, 0, size, size)
    b = _out_view(buf, 2 * size * size, size, size)
    out = {"out_is_a": a, "out_is_b": b, "out_overlaps_half_of_a": None}[alias]
    if out is None:
        out = _out_view(buf, size * size // 2, size, size)
    with pytest.raises(ValueError, match="out must not overlap a or b"):
        fn(a, b, out)


@GEMMS
def test_gemm_accepts_out_adjacent_to_operands(requires_blackwell, fn):
    size = 512
    buf = torch.randn(3 * size * size, device="cuda", dtype=torch.bfloat16)
    a, out, b = (_out_view(buf, i * size * size, size, size) for i in range(3))
    ref = torch.nn.functional.linear(a.float(), b.float())
    fn(a, b, out)
    assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item()


@GEMMS
@pytest.mark.parametrize(
    "m, n, k, out, error",
    [
        (2, 8, 0, torch.ones(1), TypeError),  # float32 out for an empty-K problem
        (2, 8, 0, torch.ones(1, dtype=torch.bfloat16), ValueError),  # wrong shape, empty K
        (0, 8, 16, torch.empty(3, 8, dtype=torch.bfloat16), ValueError),  # wrong shape, M = 0
        (4, 8, 0, torch.empty(8, 4, dtype=torch.bfloat16).t(), ValueError),  # transposed out
    ],
    ids=["dtype_empty_k", "shape_empty_k", "shape_empty_m", "layout"],
)
def test_gemm_checks_out_before_zero_work_returns(fn, m, n, k, out, error):
    """Zero-work problems return before dispatch; ``out`` still has to fit the contract."""
    a = torch.zeros(m, k, dtype=torch.bfloat16)
    b = torch.zeros(n, k, dtype=torch.bfloat16)
    with pytest.raises(error, match="out must"):
        fn(a, b, out)


# --------------------------------------------------------------------------- #
# FA4-style attention forward example
# --------------------------------------------------------------------------- #


def fa4_inputs(b, h, sq, sk, nonuniform):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, s, h, 128, device="cuda") for s in (sq, sk, sk))
    if nonuniform:
        # Distinct magnitude per 128-row Q tile (slot/stage mixups), growing K blocks (forces
        # O rescales), per-key-block V offsets, per-head scale.
        q = q * (0.5 + 0.75 * (torch.arange(sq, device="cuda") // 128 % 4)).view(1, -1, 1, 1)
        kblk = torch.arange(sk, device="cuda") // 128
        k = k * (0.5 + 0.25 * kblk).view(1, -1, 1, 1)
        v = v + 0.5 * kblk.view(1, -1, 1, 1)
        q = q * (1 + torch.arange(h, device="cuda")).view(1, 1, -1, 1) ** 0.5
    return tuple(x.to(torch.bfloat16) for x in (q, k, v))


@pytest.mark.parametrize("turnstile", [False, True])
@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize(
    "b, h, sq, sk", [(1, 1, 256, 128), (1, 1, 256, 1024), (2, 3, 640, 512), (1, 2, 384, 256)]
)
def test_fa4_fwd(requires_blackwell, b, h, sq, sk, nonuniform, turnstile):
    pytest.importorskip("flash_attn.cute")
    from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import fa4_fwd

    q, k, v = fa4_inputs(b, h, sq, sk, nonuniform)
    o, lse = fa4_fwd(q, k, v, turnstile=turnstile)
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2)
    ref_lse = torch.logsumexp((qf @ kf.transpose(-1, -2)) / math.sqrt(128), dim=-1)
    # bf16 output rounding bounds the error: half an ulp of the largest |O|, plus slack for
    # bf16 P in the PV MMA.
    tol = 2**-7 * ref.abs().max().item() + 4e-3
    assert (o.float() - ref).abs().max().item() <= tol
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


def test_fa4_declared_graph():
    pytest.importorskip("flash_attn.cute")
    from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import FlashAttentionForward

    op = FlashAttentionForward()
    summary = op.describe()
    # Fan-out/join: S_j is consumed by softmax[j] and correction, so its empty barrier waits
    # for 128 + 128 thread arrivals; the producer side is one tcgen05.commit.
    assert op.s.slot_counts("consumer") == [256, 256]
    assert op.s.slot_counts("producer") == [1, 1]
    assert op.stats.slot_counts("producer") == [128, 128]
    assert op.num_threads == 512
    assert {r.name: r.warps for r in op._ws_roles}["softmax[1]"] == range(4, 8)
    assert re.search(r"\n  4-7 +softmax\[1\] +192  all\n", summary), summary


def test_fa4_rejects_key_tail(requires_blackwell):
    pytest.importorskip("flash_attn.cute")
    from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import fa4_fwd

    q = torch.randn(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 200, 1, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 128"):
        fa4_fwd(q, k, k)


def test_describe_lists_the_traced_protocol(requires_blackwell):
    """With trace_protocol, describe() after a compile adds each role's channel operations with
    source lines, indented by dynamic region, and the TMEM regions each role touched."""
    from transformer_nuggets.cute.warpdrive.examples.gemm import Gemm, compile_gemm

    op = Gemm(trace=True)
    assert "Traced protocol" not in op.describe()  # nothing traced before the compile
    compile_gemm(op)
    text = op.describe()
    protocol = text[text.index("Traced protocol\n") :]
    assert re.search(r"\n  role load:\n +ab\[i\] +acquire +@ gemm\.py:\d+\n", protocol), protocol
    assert re.search(r"\n    epilogue: acc\[0\]", protocol), protocol
    assert all(len(line) <= 100 for line in text.splitlines())


def test_streams_launch_through_their_work_problem():
    """A kernel with a TileStream must take its grid from the problem: a OneShot over plain
    Raster() reads its item from blockIdx.y/z, which a hand-built 1-D grid would not set."""
    from transformer_nuggets.cute.warpdrive.examples.gemm import Gemm

    with pytest.raises(ProtocolError, match=r"TileStream\(s\) \['tiles'\]; pass work="):
        Gemm().launch_config(grid=(4, 1, 1))
