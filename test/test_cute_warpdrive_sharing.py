"""Round-robin epoch ownership (``Sharing.ROUND_ROBIN``).

Round-robin instance j of k takes epochs j, j + k, ...: the static ownership check that
exclusive rings use (``test_cute_warpdrive_exclusive.py``) treats it as an owner skipping the
other instances' epochs, on either side. Compiled only; the decode toy and example run
round-robin rings on the GPU."""

from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from warpdrive_test_helpers import raises_protocol as _raises_protocol

from transformer_nuggets.cute.jit_cache import jit_cache
from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    Consumer,
    Instance,
    OneShot,
    Producer,
    Ring,
    Sharing,
    StaticPersistent,
    Threads,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)

RR = Sharing.ROUND_ROBIN


class Probe(WarpSpecialized):
    """Per work item, the producer instances acquire ``fills[j]`` epochs of ``ch`` and the
    consumer instances wait ``waits[j]``, in static loops (``dynamic``: the consumers' in a
    dynamic one). ``side`` names the round-robin side: its role has two instances."""

    def __init__(self, fills, waits, schedule=None, dynamic=False):
        self.fills, self.waits, self.dynamic = fills, waits, dynamic
        self.ch = Ring(2, Threads, Threads, **{f"{self.side}_sharing": RR})
        self.work = WorkStream(schedule=schedule or OneShot())

    @cute.jit
    def fill(self, ch, work, count: int):
        while work.valid:
            for _ in cutlass.range_constexpr(count):
                with ch.produce():
                    pass
            work.advance()
        return ch, work

    @cute.jit
    def drain(self, ch, work, count: int):
        while work.valid:
            if cutlass.const_expr(self.dynamic):
                for _ in cutlass.range(count, unroll=1):
                    ch.wait().release()
            else:
                for _ in cutlass.range_constexpr(count):
                    ch.wait().release()
            work.advance()
        return ch, work

    @cute.kernel
    def kernel(self, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace())

    @cute.jit
    def __call__(self, unused: cute.Tensor, n: cutlass.Int32, stream):
        bound = self.work.bind((n, 1, 1), max_ctas=cutlass.Int32(1))
        self.kernel.set_name_prefix("warpdrive_sharing_probe")
        self.kernel(bound).launch(**self.launch_config(bound), stream=stream)


class Consumers(Probe):
    side = "consumer"

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer, work: WorkCursor):
        return self.fill(ch, work, self.fills[0])

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def cons(self, ctx, *, stage: Instance, ch: Consumer, work: WorkCursor):
        return self.drain(ch, work, self.waits[stage])


class Producers(Probe):
    side = "producer"

    @role(warps=[range(0, 1), range(1, 2)])
    @cute.jit
    def prod(self, ctx, *, stage: Instance, ch: Producer, work: WorkCursor):
        return self.fill(ch, work, self.fills[stage])

    @role(warps=2)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer, work: WorkCursor):
        return self.drain(ch, work, self.waits[0])


@jit_cache
def compile_probe(op):
    unused = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    return compile_tvm_ffi(op, unused, cutlass.Int32(1), name="warpdrive_sharing_probe")


def epochs(op) -> str:
    compile_probe(op)
    return next(line.strip() for line in op.describe().splitlines() if "epochs  " in line)


@pytest.mark.parametrize(
    "op, verdict",
    [
        (Consumers((4,), (2, 2)), "4 epochs per work item (cons[0] 2, cons[1] 2)"),
        # One item per CTA: instance 0 may take the odd epoch out.
        (Consumers((3,), (2, 1)), "3 epochs per work item (cons[0] 2, cons[1] 1)"),
        (Consumers((4,), (2, 2), StaticPersistent()), "4 epochs per work item (cons[0] 2, "),
        (Producers((2, 2), (4,)), "4 epochs per work item (prod[0] 2, prod[1] 2)"),
        (Producers((2, 1), (3,)), "3 epochs per work item (prod[0] 2, prod[1] 1)"),
    ],
)
def test_round_robin_ownership_is_verified(op, verdict):
    assert epochs(op).startswith(f"epochs    verified: {verdict}")


def test_dynamic_round_robin_trips_leave_ownership_unverified():
    verdict = epochs(Consumers((4,), (2, 2), dynamic=True))
    assert verdict == "epochs    unverified (cons[0]'s wait runs inside a dynamic loop or branch)"


@pytest.mark.parametrize(
    "op, error",
    [
        (
            Consumers((4,), (2, 1)),
            r"epoch 3 of 4 per work item is waited by no consumer; round-robin consumers take "
            r"epochs j, j \+ k, \.\.\. in turn",
        ),
        (Consumers((4,), (3, 2)), r"cons\[0\] waits on epoch 4, but prod fills only 4 per"),
        # A persistent instance continues into the next item, where the odd epoch out would
        # shift every instance's turn.
        (
            Consumers((3,), (2, 1), StaticPersistent()),
            r"cons\[0\] advances 4 epochs per work item \(2 per wait\), but prod acquires 3",
        ),
        (
            Producers((2, 0), (2,)),
            r"epoch 1 of 2 per work item is acquired by no producer; round-robin producers",
        ),
        (
            Producers((1, 1), (3,)),
            r"cons waits on epoch 2, but prod\[0\] and prod\[1\] fill only 2 per work item",
        ),
        (
            Producers((2, 1), (3,), StaticPersistent()),
            r"prod\[0\] acquires 2 of the 3 epochs per work item; a round-robin producer "
            r"continues into the next item",
        ),
    ],
)
def test_round_robin_trip_counts_that_do_not_add_up_raise(op, error):
    _raises_protocol(error, lambda: compile_probe(op))
