"""Synthetic warpdrive bugs for checking that the runtime oracle notices them.

Each mutant monkeypatches one layer derivation in the case's child process (``WDFUZZ_MUTANT``),
so a generated kernel compiles against a deliberately broken layer. A mutant that the fuzzer
reports as passing on cases exercising its feature means the oracle is blind to that failure
class. ``FEATURE`` names the ``gen.features`` class a case needs for the mutant to matter.
"""

from __future__ import annotations

from cutlass import Int32, cute

from transformer_nuggets.cute.warpdrive import channels, cluster, kernel, scheduling
from transformer_nuggets.cute.warpdrive.channels import (
    Channel,
    ConsumerEnd,
    Handle,
    ProducerEnd,
    Threads,
    _End,
)
from transformer_nuggets.cute.warpdrive.tracing import wait_range


def join_short():
    """Joins expect one warp fewer: the empty barrier completes before the last consumer."""
    orig = Channel.slot_counts

    def slot_counts(self, side):
        counts = orig(self, side)
        if side == "consumer" and self.consumer is Threads and not self.internal:
            return [c - 32 if c > 32 else c for c in counts]
        return counts

    Channel.slot_counts = slot_counts


def ring_wait_phase():
    """Ring consumers wait on the wrong parity (a fresh full barrier passes immediately)."""

    def wait(self, slot=None):
        index, phase, label = self._next(slot, "wait")
        if not self._chan.fixed and not self._chan.internal:
            phase = phase ^ 1
        self._trace.record(self._chan.name, "wait", label)
        with wait_range(self._trace, f"{self._chan.name}.wait"):
            cute.arch.mbarrier_wait(self._full(index), phase)
        return Handle(self, index, phase, label, False)

    ConsumerEnd.wait = wait
    ConsumerEnd.consume = wait


def slots_shared_phase():
    """Using one slot flips every slot's phase bit (slots lose their independence)."""
    orig = _End._next

    def _next(self, slot, op):
        out = orig(self, slot, op)
        chan = self._chan
        if chan.fixed and not chan.internal and chan.n > 1:
            self._state = self._state ^ (((1 << chan.n) - 1) ^ (1 << slot))
        return out

    _End._next = _next


def static_stride():
    """StaticPersistent strides one unit too far: items are skipped."""
    scheduling.StaticPersistent.next_index = (
        lambda self, index, problem: index + problem.units() + 1
    )


@cute.jit
def _no_report(workspace, units):
    pass


def atomic_no_reset():
    """AtomicPersistent never reports or resets its counter: the next launch starts dirty."""
    scheduling._report_finished = _no_report


def turn_no_wait():
    """Turnstile seats enter without waiting for their turn."""

    def __enter__(self):
        e = self._end
        if e._ts.enabled:
            e._trace.record(e._ts.name, "turn", e._seat)

    channels._TurnScope.__enter__ = __enter__


def receive_early_release():
    """Work-ring consumers release the slot before reading the broadcast item."""

    def _receive(self):
        h = self._end.wait()
        h.release()
        self.index = Int32(h.data[0])
        self._words = [Int32(h.data[1 + i]) for i in range(self._stream.mapping.words)]

    scheduling.WorkCursor._receive = _receive


def pair_join_short():
    """Pair-scoped Threads joins expect one warp fewer on the leader's barrier."""
    orig = cluster.derive_scope

    def derive_scope(chan, clu, *, internal=False):
        scope = orig(chan, clu, internal=internal)
        if scope.pair and not internal:
            for counts, mech in (
                (scope.full_counts, chan.producer),
                (scope.empty_counts, chan.consumer),
            ):
                if mech is Threads:
                    for rank in counts:
                        counts[rank] = [c - 32 if c > 32 else c for c in counts[rank]]
        return scope

    cluster.derive_scope = derive_scope
    kernel.derive_scope = derive_scope


def no_pair_tail():
    """Producers of pair channels released by a remote tcgen05.commit do not drain at exit."""
    ProducerEnd.tail = lambda self: None


MUTANTS = {
    "join_short": (join_short, "join"),
    "ring_wait_phase": (ring_wait_phase, "payload"),
    "slots_shared_phase": (slots_shared_phase, "slots_threads_threads"),
    "static_stride": (static_stride, "schedule_static"),
    "atomic_no_reset": (atomic_no_reset, "schedule_atomic"),
    "turn_no_wait": (turn_no_wait, "turnstile"),
    "receive_early_release": (receive_early_release, "schedule_atomic"),
    "pair_join_short": (pair_join_short, "pair_scoped"),
    "no_pair_tail": (no_pair_tail, "pair_scoped"),
}


def apply(name: str) -> None:
    MUTANTS[name][0]()
