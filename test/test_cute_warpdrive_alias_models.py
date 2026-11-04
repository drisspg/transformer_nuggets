"""Alias-diagnostic regression models: small single-CTA kernels (compiled, never launched) whose
alias orderings the diagnostic must flag or accept. FA4 mutants: test_cute_warpdrive_alias.py."""

import re
import warnings
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    Consumer,
    Producer,
    Ring,
    Slots,
    SmemPlan,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    named_barrier,
    role,
    unchecked,
)


def compile_quietly(op, name: str):
    """Compile ``op`` (entry point ``(unused, n)``), returning the alias warnings."""
    unused = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = compile_tvm_ffi(op, unused, cutlass.Int32(1), name=name)
    return fn, [str(w.message) for w in caught if "alias diagnostic" in str(w.message)]


def alias_section(op) -> str:
    text = op.describe()
    return text[text.index("Alias diagnostic") :]


def status(op, pair: str) -> str:
    match = re.search(rf"\n  {re.escape(pair)} \((?:tmem|smem)\)  ([a-z ]+)", alias_section(op))
    assert match, alias_section(op)
    return match.group(1).strip().split(" among")[0]


class Model(WarpSpecialized):
    """Roles loop ``n`` times; ``layouts()`` gives the SMEM plan's layouts, if any."""

    def layouts(self):
        return {}

    @cute.kernel
    def kernel(self, n: cutlass.Int32):
        self.begin_roles(layouts=self.layouts())
        self.run_roles(SimpleNamespace(n=n))

    @cute.jit
    def __call__(self, unused: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix("warpdrive_alias_model")
        self.kernel(n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


# --------------------------------------------------------------------------- #
# FA4-backward-style cross-warp case: eight warps share an S region's lanes in two warpgroups;
# each warpgroup overwrites its half of the P columns (an alias inside S) while the other
# warpgroup's warp on the same lanes may still be loading S.
# --------------------------------------------------------------------------- #


class CrossWarpPOverS(Model):
    def __init__(self, barrier: bool):
        self.barrier = barrier
        self.s = Slots(1, Umma, Threads, release_fence="tmem_store")
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=8)
    @cute.jit
    def mma(self, ctx, *, s: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with s.produce(0):
                tmem.view("S", write=True, by="umma")  # the pointer an S = Q K^T MMA writes
        return s

    @role(warps=range(8))
    @cute.jit
    def softmax(self, ctx, *, s: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with s.consume(0):
                tmem.access("S", read=True)  # every warp loads S on its 32 lanes
                if cutlass.const_expr(self.barrier):
                    named_barrier(3, 256)
                # Warpgroup g writes P columns [32 g, 32 g + 32): the other warpgroup's warp on
                # the same lanes reads those S columns.
                tmem.access(
                    "P", write=True, cols=lambda w: range(32 * (w // 4), 32 * (w // 4) + 32)
                )
        return s


def test_cross_warp_p_over_s_needs_a_barrier():
    op = CrossWarpPOverS(barrier=False)
    _, alias = compile_quietly(op, "cross_warp_0")
    assert len(alias) == 1 and "P~S" in alias[0]
    text = alias_section(op)
    assert status(op, "P ~ S") == "unverified"
    found = re.search(
        r"softmax read S\[0\] by threads @ .*and softmax write P\[0\] by threads @ .*: warps "
        r"(\d) and (\d) touch the same TMEM lanes and columns in one handle with no recorded "
        r"named_barrier",
        text,
    )
    assert found and abs(int(found[1]) - int(found[2])) == 4, text  # same lanes, other group

    fixed = CrossWarpPOverS(barrier=True)
    _, alias = compile_quietly(fixed, "cross_warp_1")
    assert alias == []
    assert status(fixed, "P ~ S") == "no hazard found"


def test_diagnostic_emits_no_device_code():
    """The same kernel lowers to identical IR with the diagnostic on and off."""
    ir = []
    for mode in ("warn", "off"):
        op = CrossWarpPOverS(barrier=True)
        op.alias_check = mode
        fn, _ = compile_quietly(op, "cross_warp_1")
        ir.append(str(fn.ir_module))
    assert ir[0] == ir[1]


# --------------------------------------------------------------------------- #
# A Ring hands storage over stage by stage: with two stages the MMA may rewrite one S region
# while the epilogue still reads the previous epoch's P from it.
# --------------------------------------------------------------------------- #


class RingTmemHandoff(Model):
    def __init__(self, stages: int):
        self.acc = Ring(stages, Umma, Threads)
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=4)
    @cute.jit
    def mma(self, ctx, *, acc: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with acc.produce():
                tmem.access("S", write=True, by="umma")
        return acc

    @role(warps=range(4))
    @cute.jit
    def epilogue(self, ctx, *, acc: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with acc.consume():
                tmem.access("P", read=True)
        return acc


@pytest.mark.parametrize("stages, expected", [(1, "no hazard found"), (2, "potential hazard")])
def test_ring_hands_over_unstaged_tmem_only_with_one_stage(stages, expected):
    op = RingTmemHandoff(stages)
    compile_quietly(op, f"ring_tmem_{stages}")
    assert status(op, "P ~ S") == expected, alias_section(op)
    if stages == 2:
        assert "no channel path orders them [channel]" in alias_section(op)


# --------------------------------------------------------------------------- #
# One role, no synchronization between iterations: an MMA's D write to S in iteration i and the
# A read of P in iteration i + 1 are not a pipelined pair (the trace order alone, A read then
# D write, is).
# --------------------------------------------------------------------------- #


class LoopCarriedMma(Model):
    def __init__(self):
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            tmem.access("P", read=True, by="umma")  # PV
            tmem.access("S", write=True, by="umma")  # next QK


def test_loop_carried_same_role_pair_is_checked():
    op = LoopCarriedMma()
    compile_quietly(op, "loop_carried")
    assert status(op, "P ~ S") == "potential hazard"
    assert re.search(
        r"mma write S\[0\] by umma @ .* then mma read P\[0\] by umma @ .*not a pipelined pair",
        alias_section(op),
    ), alias_section(op)


# --------------------------------------------------------------------------- #
# A TMA load is published only by its own handle's transaction count: a reader that waits on
# another TMA channel the loader fills is not ordered after it.
# --------------------------------------------------------------------------- #


class TmaThroughOtherChannel(Model):
    def __init__(self):
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        y = self.smem_plan.buffer("Y", cutlass.Int32, overlaps=x)
        self.a = Slots(1, Tma(128), Threads, buffer=x)
        self.b = Slots(1, Tma(128), Threads, buffer=y)  # another TMA transaction count

    def layouts(self):
        return {"X": cute.make_layout((32, 1)), "Y": cute.make_layout((32, 1))}

    @role(warps=0)
    @cute.jit
    def load(self, ctx, *, a: Producer, b: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with a.produce(0) as h:
                h.access(write=True, by="tma")
            with b.produce(0):
                pass
        return a, b

    @role(warps=1)
    @cute.jit
    def sink(self, ctx, *, a: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with a.consume(0):
                pass
        return a

    @role(warps=2)
    @cute.jit
    def user(self, ctx, *, b: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with b.consume(0) as h:
                h.access(read=True)
        return b


def test_tma_load_is_published_only_by_its_own_handle():
    op = TmaThroughOtherChannel()
    compile_quietly(op, "tma_other_channel")
    assert status(op, "Y ~ X") == "potential hazard", alias_section(op)
    assert "[complete_tx]" in alias_section(op)


class LoopInsideHandle(LoopCarriedMma):
    """The same loop inside one handle: the handle must not hide the loop-carried pair."""

    def __init__(self):
        super().__init__()
        self.s = Slots(1, Umma, Threads)

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, s: Producer, tmem: Tmem):
        with s.produce(0):
            for _ in cutlass.range(ctx.n, unroll=1):
                tmem.access("P", read=True, by="umma")
                tmem.access("S", write=True, by="umma")
        return s

    @role(warps=1)
    @cute.jit
    def sink(self, ctx, *, s: Consumer):
        with s.consume(0):
            pass
        return s


def test_a_handle_around_the_loop_does_not_hide_the_loop_carried_pair():
    op = LoopInsideHandle()
    compile_quietly(op, "loop_in_handle")
    assert status(op, "P ~ S") == "potential hazard", alias_section(op)


# --------------------------------------------------------------------------- #
# Runtime fuzz finding F-C1: a consumer that releases with signal=False and arrives itself.
# The reader -> next writer order exists only through that raw signal, which the diagnostic
# cannot see: unverified (its documented answer for raw signals), not a potential hazard.
# --------------------------------------------------------------------------- #


class RawReleaseOverlap(Model):
    """The producer writes Q, the consumer reads P (Q overlaps P) in both slots of one
    channel; ``raw``: the consumer releases with ``signal=False`` plus one elected arrival."""

    def __init__(self, raw: bool):
        self.raw = raw
        self.smem_plan = SmemPlan()
        p = self.smem_plan.buffer("P", cutlass.Int32)
        q = self.smem_plan.buffer("Q", cutlass.Int32, overlaps=p)
        override = {"empty_counts": 1} if raw else None
        self.c = Slots(2, Threads, Threads, buffer={"P": p, "Q": q}, override=override)

    def layouts(self):
        return {"P": cute.make_layout((32, 2)), "Q": cute.make_layout((32, 2))}

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, c: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            for s in cutlass.range_constexpr(2):
                with c.produce(s) as h:
                    h.access("Q", write=True)
        return c

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            for s in cutlass.range_constexpr(2):
                h = c.wait(s)
                h.access("P", read=True)
                if cutlass.const_expr(self.raw):
                    cute.arch.sync_warp()
                    if cute.arch.lane_idx() == 0:
                        cute.arch.mbarrier_arrive(h.empty_barrier)
                    h.release(signal=False)
                else:
                    h.release()
        return c


class RawCommitBesideTma(Model):
    """The producer writes Q and publishes it with ``commit(signal=False)`` plus an arrival,
    after committing a TMA channel the consumer also waits on. The TMA commit does not publish
    the write (complete_tx), but the raw signal might: unverified, not a potential hazard."""

    def __init__(self):
        self.smem_plan = SmemPlan()
        p = self.smem_plan.buffer("P", cutlass.Int32)
        q = self.smem_plan.buffer("Q", cutlass.Int32, overlaps=p)
        self.c = Slots(1, Threads, Threads, buffer={"P": p, "Q": q}, override={"full_counts": 1})
        self.t = Slots(1, Tma(128), Threads)

    def layouts(self):
        return {"P": cute.make_layout((32, 1)), "Q": cute.make_layout((32, 1))}

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, c: Producer, t: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            ht = t.acquire(0)
            h = c.acquire(0)
            h.access("Q", write=True)
            ht.commit()
            cute.arch.sync_warp()
            if cute.arch.lane_idx() == 0:
                cute.arch.mbarrier_arrive(h.full_barrier)
            h.commit(signal=False)
        return c, t

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer, t: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with t.consume(0):
                pass
            with c.consume(0) as h:
                h.access("P", read=True)
        return c, t


class UnclosedRelease(RawReleaseOverlap):
    """The consumer opens under ``unchecked("unclosed")`` and arrives on the empty barrier
    itself instead of releasing: the close the order needs is not traced at all."""

    def __init__(self):
        super().__init__(raw=False)

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            for s in cutlass.range_constexpr(2):
                with unchecked("unclosed", reason="released by the arrival below"):
                    h = c.wait(s)
                h.access("P", read=True)
                cute.arch.mbarrier_arrive(h.empty_barrier)
        return c


class UnclosedBesideUmma(UnclosedRelease):
    """As ``UnclosedRelease``, but the consumer also holds a stage of ``d``, a single-warp Umma
    consumer channel, around the read: ``d``'s release cannot publish the read (a failed edge),
    which must not outrank the never-closed handle that might."""

    def __init__(self):
        super().__init__()
        self.d = Slots(1, Threads, Umma)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, c: Producer, d: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with d.produce(0):
                pass
            for s in cutlass.range_constexpr(2):
                with c.produce(s) as h:
                    h.access("Q", write=True)
        return c, d

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer, d: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            g = d.wait(0)
            for s in cutlass.range_constexpr(2):
                with unchecked("unclosed", reason="released by the arrival below"):
                    h = c.wait(s)
                h.access("P", read=True)
                cute.arch.mbarrier_arrive(h.empty_barrier)
            g.release()
        return c, d


@pytest.mark.parametrize("model", [UnclosedRelease, UnclosedBesideUmma])
def test_unclosed_handle_is_unverified_not_a_hazard(model):
    op = model()
    op.alias_check = "strict"
    compile_quietly(op, f"unclosed_{model.__name__.lower()}")
    assert status(op, "Q ~ P") == "unverified", alias_section(op)
    assert "never closes" in alias_section(op)


class SameRoleRawRoundTrip(Model):
    """One role TMA-writes Q in one handle of ``t`` and reads P (Q overlaps P) in the next;
    ``sink`` releases both stages, with ``signal=False`` when ``raw``. The write -> read order
    is a round trip through that release; the read -> next write order has no edge at all."""

    def __init__(self, raw: bool):
        self.raw = raw
        self.smem_plan = SmemPlan()
        p = self.smem_plan.buffer("P", cutlass.Int32, align=128)
        q = self.smem_plan.buffer("Q", cutlass.Int32, overlaps=p)
        override = {"empty_counts": 1} if raw else None
        self.t = Slots(1, Tma(128), Threads, buffer={"P": p, "Q": q}, override=override)

    def layouts(self):
        return {"P": cute.make_layout((32, 1)), "Q": cute.make_layout((32, 1))}

    @role(warps=0)
    @cute.jit
    def load(self, ctx, *, t: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with t.produce(0) as h:
                h.access("Q", write=True, by="tma")
            with t.produce(0) as h:
                h.access("P", read=True)
        return t

    @role(warps=1)
    @cute.jit
    def sink(self, ctx, *, t: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            for _k in cutlass.range_constexpr(2):
                h = t.wait(0)
                if cutlass.const_expr(self.raw):
                    if cute.arch.lane_idx() == 0:
                        cute.arch.mbarrier_arrive(h.empty_barrier)
                    h.release(signal=False)
                else:
                    h.release()
        return t


def test_an_unverified_pair_does_not_hide_a_later_hazard():
    """The write -> read pair, ordered only through the raw release, is unverified. Reporting it
    must not end the scan before the read -> next-write pair, a potential hazard."""
    op = SameRoleRawRoundTrip(raw=True)
    compile_quietly(op, "same_role_raw")
    assert status(op, "Q ~ P") == "potential hazard", alias_section(op)
    assert "[pipelined]" in alias_section(op)


def test_raw_commit_beside_a_tma_commit_is_unverified():
    op = RawCommitBesideTma()
    op.alias_check = "strict"
    compile_quietly(op, "raw_commit_beside_tma")
    assert status(op, "Q ~ P") == "unverified", alias_section(op)


@pytest.mark.parametrize("raw, expected", [(False, "no hazard found"), (True, "unverified")])
def test_raw_release_is_unverified_not_a_hazard(raw, expected):
    """Once reported ``potential hazard: ... no channel path orders them`` (and raised under
    ``alias_check="strict"``) because closes with ``signal=False`` were dropped from the graph."""
    op = RawReleaseOverlap(raw)
    op.alias_check = "strict"
    compile_quietly(op, f"raw_release_overlap_{int(raw)}")
    assert status(op, "Q ~ P") == expected, alias_section(op)
    if raw:
        assert "only a raw signal (signal=False) would order them" in alias_section(op)
