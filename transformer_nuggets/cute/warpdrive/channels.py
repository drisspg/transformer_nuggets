"""Channels (rings and fixed slots), turnstiles, and their device-side ends and handles.

A channel is a full/empty mbarrier pair per stage or slot. Each side names its signalling
mechanism, which fixes how many arrivals the barrier expects:

* ``Tma(tx_bytes)``: one elected ``arrive.expect_tx``; the TMA engine completes the phase.
* ``Umma``: one elected ``tcgen05.commit`` (completes once all prior MMAs of the issuing
  thread have finished).
* ``Threads``: every thread of every role instance bound to the slot arrives.

The layer derives each slot's arrival count from the role instances bound to it, so a channel
with several consumer roles (fan-out) has a join barrier that completes only when all of them
have released. A ``Tma`` or ``Umma`` side signals once, so it must be bound to one single-warp
role (the warp that issues the copies or MMAs). ``Channel(override={...})`` replaces any
derived value; ``Barriers`` reserves mbarriers that user code drives directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cutlass
from cutlass import Int32, cute
from cutlass._mlir.dialects import nvvm as _nvvm

from transformer_nuggets.cute.warpdrive.cluster import ALL, LEADER, SCOPE_OVERRIDES
from transformer_nuggets.cute.warpdrive.protocol import ProtocolError, current_block, user_site
from transformer_nuggets.cute.warpdrive.tma import TmaUse, stage_closed
from transformer_nuggets.cute.warpdrive.tracing import wait_range


@dataclass(frozen=True)
class Tma:
    """Producer side filled by TMA with ``tx_bytes`` per stage."""

    tx_bytes: int

    def __post_init__(self):
        if type(self.tx_bytes) is not int or self.tx_bytes <= 0:
            raise ValueError(f"Tma.tx_bytes must be a positive int, got {self.tx_bytes!r}")


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

    ``remote_scope``: memory scope of thread arrivals on a pair-scoped channel's barriers in the
    other CTA. ``"cta"`` (default without a payload; what CUTLASS and FA4 pipelines use) only
    counts arrivals: right for TMEM/TMA/MMA handoffs ordered by their own fences and for
    stage reuse. ``"cluster"`` (default with a layer payload) is a cluster-scope release.
    **Rule:** a pair-scoped channel whose signal publishes ordinary memory (SMEM read through
    DSMEM, or global memory) that the other CTA reads outside a layer payload must set
    ``remote_scope="cluster"``. The layer cannot detect this. The cluster scope is measurably
    slower on hot channels (FA4's S release: ~30%).

    ``override``: replace derived values of the channel's ``cluster.Scope``; the user value
    wins and ``describe()`` prints it next to the derived one, flagging differences. Keys:
    ``full_counts``/``empty_counts`` (an int, a per-slot list, or ``{cta rank: list}``),
    ``tx`` (bytes armed per full barrier), ``full_mask``/``empty_mask`` (``tcgen05.commit``
    multicast masks). Example: per-warp elected releases need ``empty_counts=4`` for a
    four-warp consumer, with the role signalling via ``handle.release(signal=False)``.
    """

    n: int
    producer: object = Threads
    consumer: object = Threads
    producer_role: str | None = None
    consumer_roles: tuple[str, ...] | None = None
    payload: tuple | None = None
    commit_fence: str | None = None
    release_fence: str | None = None
    remote_scope: str | None = None
    override: dict | None = None
    name: str = field(default="", init=False)
    fixed: bool = field(default=False, init=False)
    producers: list = field(default_factory=list, init=False)  # (RoleInstance, slot | None)
    consumers: list = field(default_factory=list, init=False)
    # Derived by the layer from the kernel's cluster and role placement (cluster.derive_scope);
    # ``scope`` has the overrides applied, ``derived_scope`` does not.
    scope: object = field(default=None, init=False, repr=False)
    derived_scope: object = field(default=None, init=False, repr=False)
    cluster: object = field(default=None, init=False, repr=False)
    derived_remote_scope: str = field(default="cta", init=False, repr=False)
    remote_scope_overridden: bool = field(default=False, init=False, repr=False)
    # Layer-owned channels (the scheduler's rings) publish without TMA copies.
    internal: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if type(self.n) is not int:
            raise ValueError(f"channel stages must be a positive int, got {self.n!r}")
        if self.n <= 0:
            raise ValueError("channels need at least one stage")
        for side, mech in (("producer", self.producer), ("consumer", self.consumer)):
            if mech is not Threads and mech is not Umma and not isinstance(mech, Tma):
                raise ValueError(
                    f"channel {side} mechanism must be Threads, Umma, or Tma(bytes), got {mech!r}"
                )
        if isinstance(self.consumer, Tma):
            raise ValueError("TMA is a producer mechanism")  # noqa: TRY004
        if self.commit_fence not in FENCE_KINDS or self.release_fence not in FENCE_KINDS:
            raise ValueError(f"fence kinds must be one of {FENCE_KINDS}")
        if isinstance(self.producer, Tma) and self.commit_fence is not None:
            raise ValueError("TMA producers do not arrive, so they have no commit to fence")
        if self.remote_scope not in (None, "cta", "cluster"):
            raise ValueError(f"remote_scope must be 'cta' or 'cluster', got {self.remote_scope!r}")
        unknown = set(self.override or {}) - set(SCOPE_OVERRIDES)
        if unknown:
            raise ValueError(f"unknown override keys {sorted(unknown)}; one of {SCOPE_OVERRIDES}")
        self.derived_remote_scope = "cluster" if self.payload is not None else "cta"
        self.remote_scope_overridden = self.remote_scope is not None
        if self.remote_scope is None:
            self.remote_scope = self.derived_remote_scope

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


@dataclass
class Barriers:
    """``n`` mbarriers that user code drives directly (an unmanaged channel).

    The layer allocates them with its own barriers (so they count in SMEM) and covers them with
    its init fence and CTA/cluster sync. ``counts`` (an int or a list of ``n``): the layer
    initializes them with these arrival counts. ``counts=None``: the user initializes them in
    the kernel between ``begin_roles()`` and ``run_roles()``, for example with
    ``cutlass.pipeline.PipelineUmmaAsync.create(barrier_storage=self.barrier_ptr(name),
    defer_sync=True)``. Roles receive the base pointer through a ``name: Barriers`` parameter.
    No protocol checks apply.
    """

    n: int
    counts: int | list[int] | None = None
    name: str = field(default="", init=False)

    def __post_init__(self):
        if type(self.n) is not int:
            raise ValueError(f"Barriers.n must be a positive int, got {self.n!r}")
        if self.n <= 0:
            raise ValueError("Barriers needs at least one mbarrier")
        if type(self.counts) is int:
            self.counts = [self.counts] * self.n
        if self.counts is not None and (
            not isinstance(self.counts, (list, tuple))
            or len(self.counts) != self.n
            or any(type(c) is not int or c <= 0 for c in self.counts)
        ):
            raise ValueError(f"Barriers counts must be {self.n} positive ints, got {self.counts}")


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
    unclosed_ok: object = None  # the unchecked("unclosed") opt-out active at open


class Handle:
    """One acquired stage or slot. Close it exactly once, in the region that opened it.

    Raw state: ``index`` (stage or slot), ``phase`` (the parity this handle waited on),
    ``full_barrier``/``barrier`` and ``empty_barrier`` (this CTA's pointers; a pair's peer has
    its barriers at the same offsets). ``commit(signal=False)``/``release(signal=False)`` close
    the handle for the checker while the caller signals the barrier itself.
    """

    def __init__(self, end, index, phase, label, is_producer: bool):
        self._end = end
        self.index = index
        self.phase = phase
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
        return [v.ir_value() for v in (self.index, self.phase) if not isinstance(v, int)]

    def __new_from_mlir_values__(self, values):
        new = object.__new__(Handle)
        new.__dict__.update(self.__dict__)
        values = list(values)
        if not isinstance(self.index, int):
            new.index = Int32(values.pop(0))
        if not isinstance(self.phase, int):
            new.phase = Int32(values.pop(0))
        return new

    def describe(self) -> str:
        return (
            f"{self._end._chan.name}[{self._label}] {'acquire' if self._is_producer else 'wait'}"
        )

    @property
    def full_barrier(self):
        """This stage's full-barrier pointer. On a TMA producer, pass it as ``tma_bar_ptr``: the
        layer checks the copies that complete on it (``tma.py``)."""
        return self._end._handed_full(self.index, self.rec if self._is_producer else None)

    barrier = full_barrier

    @property
    def empty_barrier(self):
        """This stage's empty-barrier pointer (escape hatch for raw ``cute.arch`` signalling)."""
        return self._end._empty(self.index)

    @property
    def data(self):
        """This slot's view of the channel's ``payload`` buffer."""
        buf = self._end._payload
        if buf is None:
            raise ProtocolError(f"channel {self._end._chan.name} has no payload")
        return buf[(*([None] * (cute.rank(buf) - 1)), self.index)]

    def commit(self, signal: bool = True):
        """Producer: publish the stage (signal full). ``signal=False``: the caller signals the
        full barrier itself with the derived (or overridden) counts."""
        self._close(True, "commit" if signal else "commit(raw signal)")
        if signal:
            self._end._signal(self.index)

    def retire(self):
        """Producer: close without publishing, for a final acquire whose stage is never
        filled (for example the MMA's last S/P/O acquire, which only waits for P and O)."""
        if self._end._chan.scope is not None and self._end._chan.scope.tail:
            raise ProtocolError(
                f"{self._end._trace.role_name}: retire() on {self.describe()}: producers of this "
                "pair-scoped channel drain every stage before exit, and a retired stage is never "
                "released; commit it instead"
            )
        self._close(True, "retire")

    def release(self, signal: bool = True):
        """Consumer: hand the stage back (signal empty). ``signal=False``: the caller signals the
        empty barrier itself with the derived (or overridden) counts."""
        self._close(False, "release" if signal else "release(raw signal)")
        if signal:
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
        if producer and isinstance(self._end._chan.producer, Tma):
            stage_closed(trace, self.rec)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit() if self._is_producer else self.release()
        return False


@dataclass
class _EndRecord:
    """Trace-time record shared by loop-carried copies of one end: the fixed ``slots`` it used
    (a tail drains only those)."""

    slots: set = field(default_factory=set)


class _End:
    """One role instance's end of a channel. A DSL value whose only runtime state is phases:
    a ``PipelineState`` for rings, an Int32 phase-bit mask for slots.

    ``cta``: the placement of the owning role (``"all"`` or ``"leader"``); with the channel's
    derived scope it selects local or remote arrivals and who arms transaction barriers.
    """

    def __init__(self, chan, bars, payload, trace, bound_slot, state, cta=ALL, rec=None):
        self._chan = chan
        self._bars = bars  # full barriers at [0, n), empty barriers at [n, 2n)
        self._payload = payload
        self._trace = trace
        self._bound = bound_slot
        self._state = state
        self._cta = cta
        self._rec = rec if rec is not None else _EndRecord()

    def __extract_mlir_values__(self):
        if self._chan.fixed:
            return [self._state.ir_value()]
        return self._state.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        if self._chan.fixed:
            state = Int32(values[0])
        else:
            state = self._state.__new_from_mlir_values__(values)
        return type(self)(
            self._chan,
            self._bars,
            self._payload,
            self._trace,
            self._bound,
            state,
            self._cta,
            self._rec,
        )

    def _full(self, index):
        return self._bars + index

    def _empty(self, index):
        return self._bars + (self._chan.n + index)

    def _handed_full(self, index, handle_rec=None):
        """A full-barrier pointer given to user code. On a TMA producer the layer records it, and
        at role exit checks the TMA copies that complete on it (``tma.py``)."""
        ptr = self._full(index)
        chan, trace = self._chan, self._trace
        if isinstance(self, ProducerEnd) and isinstance(chan.producer, Tma) and not chan.internal:
            site = user_site()
            opted_out = trace.opt_out("tma", f"{chan.name} barrier at {site}")
            trace.tma_bars.append(TmaUse(chan, ptr.value, site, handle_rec, opted_out))
        return ptr

    # Escape hatches: everything the layer derived, for a role that drops to raw
    # cute.arch / cutlass.pipeline calls on one edge. A raw signal must keep the derived (or
    # overridden) counts in ``scope``.
    def full_barrier(self, index):
        """This CTA's full barrier of a stage/slot (a pair's peer: same offset)."""
        return self._handed_full(index)

    def empty_barrier(self, index):
        """This CTA's empty barrier of a stage/slot (a pair's peer: same offset)."""
        return self._empty(index)

    @property
    def channel(self):
        """The channel declaration (``n``, mechanisms, ``remote_scope``, ``override``)."""
        return self._chan

    @property
    def scope(self):
        """The ``cluster.Scope`` in effect (overrides applied): barrier CTAs, per-CTA arrival
        counts, multicast masks, armed tx bytes, tail. ``channel.derived_scope`` has the
        layer's own values."""
        return self._chan.scope

    @property
    def state(self):
        """Ring: the ``PipelineState`` (index, phase, count); Slots: the Int32 phase-bit mask.
        Assign a new value to resynchronize after raw waits or signals on this edge."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def index(self):
        """Ring: the stage the next acquire/wait takes."""
        if self._chan.fixed:
            raise ProtocolError(f"{self._chan.name} is Slots: the caller names the slot")
        return self._state.index

    def phase(self, slot=None):
        """The parity the next acquire/wait on the stage (Ring) or ``slot`` (Slots) waits for."""
        if not self._chan.fixed:
            return self._state.phase
        if not isinstance(slot, int):
            raise ProtocolError(f"{self._chan.name}.phase(slot) needs a Python int slot")
        return (self._state >> slot) & 1

    @property
    def targets(self) -> tuple[int, ...]:
        """CTA ranks whose barrier this end's signal reaches (empty: this CTA only)."""
        sc = self._chan.scope
        if not sc.pair:
            return ()
        return sc.full_ctas if isinstance(self, ProducerEnd) else sc.empty_ctas

    def _arrive(self, mech, bar, fence, mask):
        """Signal ``bar`` in each CTA of ``self.targets`` (empty: this CTA only); ``mask``: the
        ``tcgen05.commit`` multicast mask."""
        _fence(fence)
        cluster, targets = self._chan.cluster, self.targets
        if mech is Umma:
            with cute.arch.elect_one():
                if cluster is None or cluster.size == 1:
                    cute.nvgpu.tcgen05.commit(bar)
                else:
                    # Every tcgen05 op of a pair kernel is cta_group::2; the multicast mask
                    # names the CTAs whose barrier (same offset) receives the arrival.
                    cute.nvgpu.tcgen05.commit(bar, mask, cluster.cta_group)
        elif mech is Threads:
            if not targets:
                cute.arch.mbarrier_arrive(bar)
                return
            # See Channel.remote_scope: a cluster-scope release orders generic-memory data for
            # the other CTA's waiter; "cta" (CUTLASS's default) only counts the arrival.
            scope = _nvvm.MemScopeKind.CLUSTER if self._chan.remote_scope == "cluster" else None
            for c in targets:
                if c == 0 and self._cta == LEADER:
                    cute.arch.mbarrier_arrive(bar, scope=scope)
                else:
                    cute.arch.mbarrier_arrive(bar, c, scope=scope)
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
        self._rec.slots.add(slot)
        return slot, phase, slot


class ProducerEnd(_End):
    def acquire(self, slot=None) -> Handle:
        """Wait until the stage is empty; returns a handle to commit or retire."""
        index, phase, label = self._next(slot, "acquire")
        self._trace.record(self._chan.name, "acquire", label)
        with wait_range(self._trace, f"{self._chan.name}.acquire"):
            cute.arch.mbarrier_wait(self._empty(index), phase)
        if isinstance(self._chan.producer, Tma):
            self._arm(self._full(index))
        return Handle(self, index, phase, label, True)

    produce = acquire  # ``with end.produce(...) as h:`` reads better for nested scopes

    def _arm(self, full):
        """Expect the stage's transaction bytes on every waited full barrier. In a pair-scoped
        channel only the leader's producer arms (one arrival per barrier, the bytes of every
        CTA's copies), as CUTLASS's ``PipelineTmaUmma`` does."""
        scope = self._chan.scope
        if not scope.pair:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(full, scope.tx)
        elif self._cta == LEADER:
            _arm_targets(full, scope.tx, scope.full_ctas)
        else:
            _arm_if_leader(full, scope.tx, scope.full_ctas)

    def _signal(self, index):
        if isinstance(self._chan.producer, Tma):
            return  # completed by the TMA transaction count
        chan = self._chan
        self._arrive(chan.producer, self._full(index), chan.commit_fence, chan.scope.full_mask)

    def tail(self):
        """Wait until every stage this end filled has been released (layer-emitted at role
        exit for pair-scoped channels whose consumer signals from another CTA)."""
        self._trace.record(self._chan.name, "tail", None)
        n = self._chan.n
        with wait_range(self._trace, f"{self._chan.name}.tail"):
            if self._chan.fixed:
                # Only slots this end acquired: another instance's slot keeps its own parity.
                for j in sorted(self._rec.slots):
                    cute.arch.mbarrier_wait(self._empty(j), (self._state >> j) & 1)
            else:
                st = self._state.clone()
                for _ in range(n):
                    cute.arch.mbarrier_wait(self._empty(st.index), st.phase)
                    st.advance()


class ConsumerEnd(_End):
    def wait(self, slot=None) -> Handle:
        """Wait until the stage is full; returns a handle to release."""
        index, phase, label = self._next(slot, "wait")
        self._trace.record(self._chan.name, "wait", label)
        with wait_range(self._trace, f"{self._chan.name}.wait"):
            cute.arch.mbarrier_wait(self._full(index), phase)
        return Handle(self, index, phase, label, False)

    consume = wait

    def _signal(self, index):
        chan = self._chan
        self._arrive(chan.consumer, self._empty(index), chan.release_fence, chan.scope.empty_mask)


def _arm_one(full, tx: int, cta: int):
    if cta == 0:  # the leader arms its own barrier
        cute.arch.mbarrier_arrive_and_expect_tx(full, tx)
    else:
        cute.arch.mbarrier_arrive_and_expect_tx(full, tx, cta)


def _arm_targets(full, tx: int, targets):
    with cute.arch.elect_one():
        for c in targets:
            _arm_one(full, tx, c)


@cute.jit
def _arm_if_leader(full, tx: cutlass.Constexpr, targets: cutlass.Constexpr):
    if cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0:
        _arm_targets(full, tx, targets)


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
            with wait_range(e._trace, f"{e._ts.name}.turn"):
                cute.arch.mbarrier_wait(e._bars + e._seat, e._phase)

    def __exit__(self, exc_type, exc, tb):
        e = self._end
        if exc_type is None and e._ts.enabled:
            nxt = (e._seat + 1) % len(e._ts.parties)
            cute.arch.mbarrier_arrive(e._bars + nxt)
            e._phase = e._phase ^ 1
            e._trace.record(e._ts.name, "pass", nxt)
        return False
