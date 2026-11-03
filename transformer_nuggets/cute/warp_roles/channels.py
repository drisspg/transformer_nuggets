"""Channels (rings and fixed slots), turnstiles, and their device-side ends and handles.

A channel is a full/empty mbarrier pair per stage or slot. Each side names its signalling
mechanism, which fixes how many arrivals the barrier expects:

* ``Tma(tx_bytes)``: one elected ``arrive.expect_tx``; the TMA engine completes the phase.
* ``Umma``: one elected ``tcgen05.commit`` (completes once all prior MMAs of the issuing
  thread have finished).
* ``Threads``: every thread of every role instance bound to the slot arrives.

The layer derives each slot's arrival count from the role instances bound to it, so a channel
with several consumer roles (fan-out) has a join barrier that completes only when all of them
have released.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cutlass import Int32, cute

from transformer_nuggets.cute.warp_roles.protocol import ProtocolError, current_block, user_site


@dataclass(frozen=True)
class Tma:
    """Producer side filled by TMA with ``tx_bytes`` per stage."""

    tx_bytes: int


class Umma:
    """Side signalled by one elected ``tcgen05.commit``."""


class Threads:
    """Side signalled by ``mbarrier.arrive`` from every thread of the bound role instances."""


def arrivals(mech, threads: int) -> int:
    return threads if mech is Threads else 1


def _fence(kind: str | None):
    if kind is None:
        return
    if kind == "tmem_store":  # tcgen05.st -> tcgen05.mma reads TMEM
        cute.arch.fence_view_async_tmem_store()
    elif kind == "smem_async":  # st.shared -> TMA store / UMMA reads SMEM (async proxy)
        cute.arch.fence_view_async_shared()
    else:
        raise ValueError(f"unknown fence kind {kind!r}")


FENCE_KINDS = (None, "tmem_store", "smem_async")


@dataclass
class Channel:
    """Shared declaration for ``Ring`` and ``Slots``.

    ``producer_role``/``consumer_roles`` optionally name the roles allowed on each side; the
    layer checks them against role annotations. ``payload=(dtype, shape)`` makes the layer own
    a per-slot SMEM buffer exposed as ``handle.data``. ``commit_fence``/``release_fence`` say
    what the producer/consumer wrote before signalling ("tmem_store" or "smem_async"); the
    layer emits the matching fence before every arrive on that side.
    """

    n: int
    producer: object = Threads
    consumer: object = Threads
    producer_role: str | None = None
    consumer_roles: tuple[str, ...] | None = None
    payload: tuple | None = None
    commit_fence: str | None = None
    release_fence: str | None = None
    name: str = field(default="", init=False)
    fixed: bool = field(default=False, init=False)
    producers: list = field(default_factory=list, init=False)  # (RoleInstance, slot | None)
    consumers: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.n <= 0:
            raise ValueError("channels need at least one stage")
        if self.commit_fence not in FENCE_KINDS or self.release_fence not in FENCE_KINDS:
            raise ValueError(f"fence kinds must be one of {FENCE_KINDS}")
        if isinstance(self.producer, Tma) and self.commit_fence is not None:
            raise ValueError("TMA producers do not arrive, so they have no commit to fence")

    def slot_counts(self, side: str) -> list[int]:
        mech = self.producer if side == "producer" else self.consumer
        bound = self.producers if side == "producer" else self.consumers
        return [
            sum(arrivals(mech, ri.num_threads) for ri, slot in bound if slot in (None, j))
            for j in range(self.n)
        ]


@dataclass
class Ring(Channel):
    """Rotating stages: ``acquire()``/``wait()`` take the next stage in order."""


@dataclass
class Slots(Channel):
    """Fixed slots: the caller names the slot; each slot keeps its own phase bit."""

    def __post_init__(self):
        super().__post_init__()
        self.fixed = True
        if self.n > 31:
            raise ValueError("Slots supports at most 31 slots (phase bits share one Int32)")


@dataclass
class Turnstile:
    """Round-robin critical section between the instances of one role.

    Seat p waits on barrier p and, on leaving, arrives on barrier (p + 1) % n; seat 0 starts
    with the turn. ``enabled=False`` keeps the API but emits nothing (for A/B runs).
    """

    enabled: bool = True
    name: str = field(default="", init=False)
    parties: list = field(default_factory=list, init=False)


# --------------------------------------------------------------------------- #
# Device side
# --------------------------------------------------------------------------- #


@dataclass
class HandleRecord:
    what: str
    site: str
    block: object
    closed_at: str | None = None
    closed_op: str | None = None


class Handle:
    """One acquired stage or slot. Close it exactly once, in the region that opened it."""

    def __init__(self, end, index, label, is_producer: bool):
        self._end = end
        self.index = index
        self._label = label
        self._is_producer = is_producer
        # Trace-time lifetime record, shared by loop-carried copies of this handle.
        self.rec = HandleRecord(self.describe(), user_site(), current_block())
        end._trace.opened(self.rec)

    # The preprocessor carries every name a loop or branch reassigns, so reusing a handle name
    # across a dynamic region boundary (``with kv.produce() as h`` before and inside a loop)
    # makes the handle a carried value. It carries its stage index; the trace-time state is
    # shared, so a carried copy still reports double closes and cross-region closes.
    def __extract_mlir_values__(self):
        return [] if isinstance(self.index, int) else [self.index.ir_value()]

    def __new_from_mlir_values__(self, values):
        new = object.__new__(Handle)
        new.__dict__.update(self.__dict__)
        if values:
            new.index = Int32(values[0])
        return new

    def describe(self) -> str:
        return (
            f"{self._end._chan.name}[{self._label}] {'acquire' if self._is_producer else 'wait'}"
        )

    @property
    def barrier(self):
        """This stage's full-barrier pointer (pass as ``tma_bar_ptr``)."""
        return self._end._full(self.index)

    @property
    def data(self):
        """This slot's view of the channel's ``payload`` buffer."""
        buf = self._end._payload
        if buf is None:
            raise ProtocolError(f"channel {self._end._chan.name} has no payload")
        return buf[(*([None] * (cute.rank(buf) - 1)), self.index)]

    def commit(self):
        """Producer: publish the stage (signal full)."""
        self._close(True, "commit")
        self._end._signal(self.index)

    def retire(self):
        """Producer: close without publishing, for a final acquire whose stage is never
        filled (for example the MMA's last S/P/O acquire, which only waits for P and O)."""
        self._close(True, "retire")

    def release(self):
        """Consumer: hand the stage back (signal empty)."""
        self._close(False, "release")
        self._end._signal(self.index)

    def _close(self, producer: bool, op: str):
        trace = self._end._trace
        if producer != self._is_producer:
            raise ProtocolError(
                f"{trace.role_name}: {op}() on a {self.describe()} handle; producer handles "
                "commit or retire, consumer handles release"
            )
        trace.closed(self.rec, op)
        trace.record(self._end._chan.name, op, self._label)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit() if self._is_producer else self.release()
        return False


class _End:
    """One role instance's end of a channel. A DSL value whose only runtime state is phases:
    a ``PipelineState`` for rings, an Int32 phase-bit mask for slots."""

    def __init__(self, chan, bars, payload, trace, bound_slot, state):
        self._chan = chan
        self._bars = bars  # full barriers at [0, n), empty barriers at [n, 2n)
        self._payload = payload
        self._trace = trace
        self._bound = bound_slot
        self._state = state

    def __extract_mlir_values__(self):
        if self._chan.fixed:
            return [self._state.ir_value()]
        return self._state.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        if self._chan.fixed:
            state = Int32(values[0])
        else:
            state = self._state.__new_from_mlir_values__(values)
        return type(self)(self._chan, self._bars, self._payload, self._trace, self._bound, state)

    def _full(self, index):
        return self._bars + index

    def _empty(self, index):
        return self._bars + (self._chan.n + index)

    @staticmethod
    def _arrive(mech, bar, fence):
        _fence(fence)
        if mech is Umma:
            with cute.arch.elect_one():
                cute.nvgpu.tcgen05.commit(bar)
        elif mech is Threads:
            cute.arch.mbarrier_arrive(bar)
        else:
            raise AssertionError(mech)

    def _next(self, slot, op):
        """(index, phase, label) of the next use, advancing this end's state."""
        chan, who = self._chan, self._trace.role_name
        if not chan.fixed:
            if slot is not None:
                raise ProtocolError(f"{who}: ring {chan.name}.{op}() takes no slot")
            index, phase = self._state.index, self._state.phase
            self._state.advance()
            return index, phase, "i"
        if not isinstance(slot, int):
            raise ProtocolError(
                f"{who}: {chan.name}.{op}(slot) needs a Python int slot (use "
                f"cutlass.range_constexpr), got {type(slot).__name__}"
            )
        if not 0 <= slot < chan.n:
            raise ProtocolError(f"{who}: {chan.name} has {chan.n} slots, got {slot}")
        if self._bound is not None and slot != self._bound:
            raise ProtocolError(
                f"{who}: this {chan.name} end is bound to slot {self._bound}, but {op}({slot}) "
                f"was called at {user_site()}"
            )
        phase = (self._state >> slot) & 1
        self._state = self._state ^ (1 << slot)
        return slot, phase, slot


class ProducerEnd(_End):
    def acquire(self, slot=None) -> Handle:
        """Wait until the stage is empty; returns a handle to commit or retire."""
        index, phase, label = self._next(slot, "acquire")
        self._trace.record(self._chan.name, "acquire", label)
        cute.arch.mbarrier_wait(self._empty(index), phase)
        if isinstance(self._chan.producer, Tma):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    self._full(index), self._chan.producer.tx_bytes
                )
        return Handle(self, index, label, True)

    produce = acquire  # ``with end.produce(...) as h:`` reads better for nested scopes

    def _signal(self, index):
        if isinstance(self._chan.producer, Tma):
            return  # completed by the TMA transaction count
        self._arrive(self._chan.producer, self._full(index), self._chan.commit_fence)


class ConsumerEnd(_End):
    def wait(self, slot=None) -> Handle:
        """Wait until the stage is full; returns a handle to release."""
        index, phase, label = self._next(slot, "wait")
        self._trace.record(self._chan.name, "wait", label)
        cute.arch.mbarrier_wait(self._full(index), phase)
        return Handle(self, index, label, False)

    consume = wait

    def _signal(self, index):
        self._arrive(self._chan.consumer, self._empty(index), self._chan.release_fence)


class TurnEnd:
    """One role instance's seat at a turnstile. The phase bit is loop-carried."""

    def __init__(self, ts: Turnstile, bars, seat: int, trace, phase):
        self._ts, self._bars, self._seat, self._trace, self._phase = ts, bars, seat, trace, phase

    def __extract_mlir_values__(self):
        return [self._phase.ir_value()]

    def __new_from_mlir_values__(self, values):
        return TurnEnd(self._ts, self._bars, self._seat, self._trace, Int32(values[0]))

    def turn(self):
        """``with seat.turn():`` runs the body when it is this seat's turn, then passes it on."""
        return _TurnScope(self)


class _TurnScope:
    def __init__(self, end: TurnEnd):
        self._end = end

    def __enter__(self):
        e = self._end
        if e._ts.enabled:
            e._trace.record(e._ts.name, "turn", e._seat)
            cute.arch.mbarrier_wait(e._bars + e._seat, e._phase)

    def __exit__(self, exc_type, exc, tb):
        e = self._end
        if exc_type is None and e._ts.enabled:
            nxt = (e._seat + 1) % len(e._ts.parties)
            cute.arch.mbarrier_arrive(e._bars + nxt)
            e._phase = e._phase ^ 1
            e._trace.record(e._ts.name, "pass", nxt)
        return False
