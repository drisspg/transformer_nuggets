"""Channels (rings and fixed slots), turnstiles, and their device-side ends and handles.

A channel is a full/empty mbarrier pair per stage or slot. Each side names its signalling
mechanism, which fixes how many arrivals the barrier expects:

* ``Tma(tx_bytes, multicast=...)``: one elected ``arrive.expect_tx``; the TMA engine completes
  the phase.
* ``Umma``: one elected ``tcgen05.commit`` (completes once all prior MMAs of the issuing
  thread have finished).
* ``Threads``: every thread of every role instance bound to the slot arrives.

The layer derives each slot's arrival count from the role instances bound to it, so a channel
with several consumer roles (fan-out) has a join barrier that completes only when all of them
have released. A ``Tma`` or ``Umma`` side signals once, so it must be bound to one single-warp
role (the warp that issues the copies or MMAs). ``Channel(override={...})`` replaces any
derived value; ``Barriers`` reserves mbarriers that user code drives directly.

Two declaration options change how a channel synchronizes:

* ``round_robin=True`` (``Ring`` only): the role instances on a side take the epochs in turn.
  With ``k`` instances, instance ``j`` handles epochs ``j, j + k, j + 2k, ...``, and each epoch's
  arrival count is one instance's contribution, not the join of all of them. FA4's decode
  kernel uses it for two softmax warpgroups that alternate KV blocks. On an mbarrier ring ``k``
  must divide the stage count: a parity wait is only exact when the waiter itself saw the
  stage's previous epoch, and with ``n % k != 0`` another instance did.
* ``sync="named"`` (``Threads`` on both sides): each slot is a pair of CTA named barriers
  (``bar.sync``/``bar.arrive``, unaligned) instead of mbarriers. A named barrier has no phase:
  it completes when the producer's and the consumer's threads have all arrived, so its count is
  their sum. Every consumer instance arrives once on its slots' empty barriers at role start, the
  equivalent of an mbarrier ring starting empty. IDs are allocated from 15 downward; see
  ``WarpSpecialized.named_barrier_ids``.

A named barrier cannot tell two instances apart: if round-robin instances share a stage's barrier
(``n % k != 0``, e.g. a one-stage mailbox written by two warpgroups in turn), their threads could
complete it together without the other side. Such a channel names a ``Turnstile`` over those
instances (``turnstile="mufu"``) and acquires/waits only while holding the turn (checked while
tracing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cutlass
from cutlass import Int32, cute
from cutlass._mlir.dialects import nvvm as _nvvm

from transformer_nuggets.cute.warpdrive.cluster import (
    ALL,
    MULTICAST_MODES,
    SCOPE_OVERRIDES,
    rank_in,
    rank_value,
)
from transformer_nuggets.cute.warpdrive.protocol import ProtocolError, current_block, user_site
from transformer_nuggets.cute.warpdrive.storage import SmemBuffer, StageView, project, stage_extent
from transformer_nuggets.cute.warpdrive.tma import TmaUse, stage_closed
from transformer_nuggets.cute.warpdrive.tracing import wait_range


@dataclass(frozen=True)
class Tma:
    """Producer side filled by TMA with ``tx_bytes`` per stage: the bytes landing in each
    producing CTA's SMEM, whichever CTA issued the copies.

    ``multicast``: the cluster modes ("m", "n", or both, as a string or tuple) along which the
    copies are multicast (``Cluster.multicast``). A copy multicast along N lands in every CTA
    with this CTA's coordinates except along N; the consumers there must release this CTA's
    stage too, which the layer derives from the declaration."""

    tx_bytes: int
    multicast: tuple[str, ...] | str = ()

    def __post_init__(self):
        if type(self.tx_bytes) is not int or self.tx_bytes <= 0:
            raise ValueError(f"Tma.tx_bytes must be a positive int, got {self.tx_bytes!r}")
        modes = self.multicast
        if not isinstance(modes, (str, tuple, list)) or any(
            m not in MULTICAST_MODES for m in modes
        ):
            raise ValueError(
                f"Tma.multicast must name cluster modes from {MULTICAST_MODES}, got {modes!r}"
            )
        object.__setattr__(self, "multicast", tuple(sorted(set(modes))))


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
SYNC_KINDS = ("mbarrier", "named")


@dataclass
class Channel:
    """Shared declaration for ``Ring`` and ``Slots``.

    ``producer_role``/``consumer_roles`` optionally name the roles allowed on each side; the
    layer checks them against role annotations. ``payload=(dtype, shape)`` makes the layer own
    a per-slot SMEM buffer exposed as ``handle.data``. ``buffer=`` binds buffers of the kernel's
    ``SmemPlan`` instead (one ``SmemBuffer``, or a ``{name: SmemBuffer}`` dict): ``handle.data`` is
    then the stage's canonical view and ``handle.view(name)`` a registered partition
    (``storage.py``). A buffer, unlike a payload, changes neither ``remote_scope`` nor fences
    and is allowed on pair-scoped channels. ``commit_fence``/``release_fence`` say
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

    ``round_robin`` and ``sync`` are described in the module docstring. A named channel takes
    no override and must stay CTA-local; ``named_ids`` holds its ``(full, empty)`` barrier IDs
    per slot once the kernel is collected.
    """

    n: int
    producer: object = Threads
    consumer: object = Threads
    producer_role: str | None = None
    consumer_roles: tuple[str, ...] | None = None
    payload: tuple | None = None
    buffer: SmemBuffer | dict | None = None
    commit_fence: str | None = None
    release_fence: str | None = None
    remote_scope: str | None = None
    override: dict | None = None
    round_robin: bool = False
    sync: str = "mbarrier"
    turnstile: str | None = None
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
    # ``buffer`` normalized to {name: SmemBuffer}; a single buffer is keyed by its plan name.
    buffers: dict = field(default_factory=dict, init=False, repr=False)
    # sync="named": ([full id per slot], [empty id per slot]), assigned by the kernel.
    named_ids: tuple | None = field(default=None, init=False, repr=False)

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
        if self.buffer is not None:
            if self.payload is not None:
                raise ValueError("a channel takes payload= or buffer=, not both")
            named = (
                {self.buffer.name: self.buffer}
                if isinstance(self.buffer, SmemBuffer)
                else self.buffer
            )
            if (
                not isinstance(named, dict)
                or not named
                or any(
                    not isinstance(k, str) or not isinstance(v, SmemBuffer)
                    for k, v in named.items()
                )
            ):
                raise ValueError(
                    "channel buffer must be a SmemBuffer or a non-empty {name: SmemBuffer} dict, "
                    f"got {self.buffer!r}"
                )
            self.buffers = dict(named)
        if type(self.round_robin) is not bool:
            raise ValueError(f"round_robin must be a bool, got {self.round_robin!r}")
        if self.sync not in SYNC_KINDS:
            raise ValueError(f"channel sync must be one of {SYNC_KINDS}, got {self.sync!r}")
        if self.turnstile is not None and not (self.named and self.round_robin):
            raise ValueError("turnstile= serializes round-robin instances of a named channel")
        if self.named:
            if self.producer is not Threads or self.consumer is not Threads:
                raise ValueError(
                    "a named-barrier channel synchronizes threads: both sides must be Threads"
                )
            if self.override:
                raise ValueError(
                    "a named-barrier channel takes no override: its count is the producer and "
                    "consumer threads of each slot"
                )
        unknown = set(self.override or {}) - set(SCOPE_OVERRIDES)
        if unknown:
            raise ValueError(f"unknown override keys {sorted(unknown)}; one of {SCOPE_OVERRIDES}")
        self.derived_remote_scope = "cluster" if self.payload is not None else "cta"
        self.remote_scope_overridden = self.remote_scope is not None
        if self.remote_scope is None:
            self.remote_scope = self.derived_remote_scope

    @property
    def named(self) -> bool:
        return self.sync == "named"

    def turns(self, side: str) -> int:
        """Instances taking turns on ``side``: ``k`` for a round-robin side, else 1."""
        bound = self.producers if side == "producer" else self.consumers
        return len(bound) if self.round_robin and len(bound) > 1 else 1

    def slot_counts(self, side: str) -> list[int]:
        mech = self.producer if side == "producer" else self.consumer
        bound = self.producers if side == "producer" else self.consumers
        if self.turns(side) > 1:  # one instance signals each epoch
            return [arrivals(mech, bound[0][0].num_threads)] * self.n
        return [
            sum(arrivals(mech, ri.num_threads) for ri, slot in bound if slot in (None, j))
            for j in range(self.n)
        ]

    def named_counts(self) -> list[int]:
        """Per-slot thread count of a named channel's barriers: producers plus consumers."""
        return [p + c for p, c in zip(self.slot_counts("producer"), self.slot_counts("consumer"))]


@dataclass
class Ring(Channel):
    """Rotating stages: ``acquire()``/``wait()`` take the next stage in order."""


@dataclass
class Slots(Channel):
    """Fixed slots: the caller names the slot; each slot keeps its own phase bit."""

    def __post_init__(self):
        super().__post_init__()
        if self.round_robin:
            raise ValueError("round_robin needs a Ring: a Slots caller names the slot itself")
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


@dataclass(eq=False)
class HandleRecord:
    what: str
    site: str
    block: object
    closed_at: str | None = None
    closed_op: str | None = None
    unclosed_ok: object = None  # the unchecked("unclosed") opt-out active at open
    chan: object = None  # the Channel
    slot: int | None = None  # the Slots slot; None for a Ring stage (dynamic)
    producer: bool = False


@dataclass(frozen=True)
class StageBuffer:
    """A channel buffer as an end sees it: the whole staged tensor and its stage mode."""

    tensor: object
    axis: int
    storage: str | None  # the SmemPlan buffer name; None for a layer payload


AGENTS = ("threads", "tma", "umma")


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
        self.rec = HandleRecord(
            self.describe(),
            user_site(),
            current_block(),
            chan=end._chan,
            slot=index if end._chan.fixed else None,
            producer=is_producer,
        )
        end._trace.opened(self.rec)
        end._trace.event("open", handle=self.rec)

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
        self._end._mbarriers("full_barrier")
        return self._end._handed_full(self.index, self.rec if self._is_producer else None)

    barrier = full_barrier

    @property
    def empty_barrier(self):
        """This stage's empty-barrier pointer (escape hatch for raw ``cute.arch`` signalling)."""
        self._end._mbarriers("empty_barrier")
        return self._end._empty(self.index)

    @property
    def data(self):
        """This stage's canonical view of the channel's storage: the ``payload`` tensor, the
        bound buffer, or ``{name: view}`` for named buffers. Every mode but the stage mode is
        kept. An unannotated use: for aliased storage prefer ``view(..., read=/write=)``."""
        buffers = self._end._buffers
        if not buffers:
            raise ProtocolError(f"channel {self._end._chan.name} has no payload or buffer")
        for b in buffers.values():
            self._record_raw(b)
        views = {k: project(b.tensor, b.axis, self.index) for k, b in buffers.items()}
        return next(iter(views.values())) if len(views) == 1 else views

    def view(
        self,
        name: str | None = None,
        *coord,
        read: bool = False,
        write: bool = False,
        by="threads",
    ):
        """Buffer ``name`` at this stage: the partition registered with ``end.views(name=...)``,
        else the canonical view. ``name`` may be omitted (``None``) for a single buffer.
        ``coord``: indices of the other modes, taken in the same slice as the stage
        (``h.view("A", None, None, kb)`` is ``tCrA[None, None, kb, h.index]``). ``read``/
        ``write`` with ``by`` ("threads", "tma", "umma") annotate the access for the alias
        diagnostic."""
        key, b = self._buffer(name)
        part = self._end._views.get(key)
        tensor, axis = (part.tensor, part.axis) if part is not None else (b.tensor, b.axis)
        if read or write:
            self._record_access(b, read, write, by)
        else:
            self._record_raw(b)
        return project(tensor, axis, self.index, coord)

    def access(
        self, name: str | None = None, *, read: bool = False, write: bool = False, by="threads"
    ):
        """Annotate an access to buffer ``name`` at this stage made through a tensor obtained
        elsewhere (for example a TMA copy function that selects the stage itself). No IR."""
        if not (read or write):
            raise ValueError("access() needs read=True and/or write=True")
        _, b = self._buffer(name)
        self._record_access(b, read, write, by)

    def _buffer(self, name):
        buffers, chan = self._end._buffers, self._end._chan
        if name is None:
            if len(buffers) != 1:
                raise ProtocolError(
                    f"channel {chan.name} has buffers {sorted(buffers)}; name the one to view"
                )
            name = next(iter(buffers))
        if name not in buffers:
            raise ProtocolError(
                f"channel {chan.name} has no buffer {name!r} (have {sorted(buffers)})"
            )
        return name, buffers[name]

    def _record_access(self, b: StageBuffer, read: bool, write: bool, by):
        if by not in AGENTS:
            raise ValueError(f"access agent must be one of {AGENTS}, got {by!r}")
        if b.storage is None:
            return  # a layer payload is never aliased
        self._end._trace.event(
            "access",
            storage=("smem", b.storage),
            slot=self.rec.slot,
            read=read,
            write=write,
            agent=by,
            handle=self.rec,
        )

    def _record_raw(self, b: StageBuffer):
        if b.storage is not None:
            self._end._trace.event("raw", storage=("smem", b.storage))

    def commit(self, signal: bool = True, wait: bool = False):
        """Producer: publish the stage (signal full). ``signal=False``: the caller signals the
        full barrier itself with the derived (or overridden) counts.

        ``wait=True`` (named channels only) makes the commit a rendezvous: the producer threads
        also wait on the full barrier, so when it returns every producer and consumer thread has
        arrived, and the producer threads see each other's writes made before the commit."""
        if wait and not self._end._chan.named:
            raise ProtocolError(
                f"{self._end._trace.role_name}: commit(wait=True) on {self.describe()}: only a "
                "named-barrier channel's producer can wait on its own full barrier"
            )
        if wait and not signal:
            raise ProtocolError("commit(wait=True) signals the full barrier; drop signal=False")
        self._close(True, "commit" if signal else "commit(raw signal)")
        if wait:
            _fence(self._end._chan.commit_fence)
            self._end._named_wait(self._end._chan.named_ids[0], self.index, "commit")
        elif signal:
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
        trace.event("close", handle=self.rec, signal=op in ("commit", "release"))
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

    def __init__(
        self,
        chan,
        bars,
        buffers,
        trace,
        bound_slot,
        state,
        cta=ALL,
        rec=None,
        views=None,
        stride=1,
    ):
        self._chan = chan
        self._stride = stride  # round-robin: epochs between this end's turns
        self._bars = bars  # full barriers at [0, n), empty barriers at [n, 2n)
        self._buffers = buffers  # {name: StageBuffer}: the payload or the bound plan buffers
        self._trace = trace
        self._bound = bound_slot
        self._state = state
        self._cta = cta
        self._rec = rec if rec is not None else _EndRecord()
        self._views = views or {}  # {buffer name: StageView} registered with views()

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
            self._buffers,
            self._trace,
            self._bound,
            state,
            self._cta,
            self._rec,
            self._views,
            self._stride,
        )

    def views(self, **named: StageView):
        """This end with partitions of its channel's buffers: ``ab = ab.views(A=StageView(tAsA,
        axis=1))`` makes ``h.view("A")`` return ``tAsA`` at ``h.index``. The name must be one of
        the channel's buffers and the view's stage mode must have the channel's stage count.
        The returned end shares this end's state; use it (and return it) in place of this one."""
        chan = self._chan
        for name, part in named.items():
            if name not in self._buffers or self._buffers[name].storage is None:
                raise ProtocolError(
                    f"{self._trace.role_name}: {chan.name}.views({name}=...): the channel has no "
                    f"buffer {name!r} (have {sorted(k for k, b in self._buffers.items() if b.storage)})"
                )
            if not isinstance(part, StageView):
                raise ProtocolError(
                    f"{chan.name}.views({name}=...) takes a StageView, got {type(part).__name__}"
                )
            extent = stage_extent(part.tensor, part.axis, f"{chan.name}.views({name}=...)")
            if extent != chan.n:
                raise ProtocolError(
                    f"{self._trace.role_name}: {chan.name}.views({name}=...): the view's stage "
                    f"mode {part.axis} has {extent} stages, but {chan.name} has {chan.n}"
                )
        return type(self)(
            chan,
            self._bars,
            self._buffers,
            self._trace,
            self._bound,
            self._state,
            self._cta,
            self._rec,
            {**self._views, **named},
            self._stride,
        )

    def _mbarriers(self, what: str):
        if self._chan.named:
            raise ProtocolError(
                f"{self._trace.role_name}: {self._chan.name} is a named-barrier channel and has no "
                f"mbarriers ({what}); its barrier IDs are channel.named_ids"
            )

    def _named_id(self, ids, index):
        if isinstance(index, int):
            return ids[index]
        return Int32(ids[0]) + index  # ring stages have consecutive IDs

    def _named_count(self, index):
        counts = self._chan.named_counts()
        return counts[index] if isinstance(index, int) else counts[0]

    def _named_wait(self, ids, index, op):
        with wait_range(self._trace, f"{self._chan.name}.{op}"):
            _named_sync(self._named_id(ids, index), self._named_count(index))

    def _named_arrive(self, ids, index):
        _named_arrive(self._named_id(ids, index), self._named_count(index))

    def _check_turn(self, side: str, op: str):
        chan = self._chan
        if chan.turnstile is None or chan.turns(side) == 1:
            return
        if chan.turnstile not in self._trace.open_turns:
            raise ProtocolError(
                f"{self._trace.role_name}: {chan.name}.{op}() at {user_site()} outside "
                f"`with {chan.turnstile}.turn():`; round-robin {side}s of this named channel share "
                "a barrier and must take it in turn"
            )

    def seed(self, slots):
        """Named channel, consumer: arrive once on the empty barriers of ``slots`` (layer-emitted
        at role start; the equivalent of mbarriers starting empty)."""
        for j in slots:
            _named_arrive(self._chan.named_ids[1][j], self._chan.named_counts()[j])

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
        self._mbarriers("full_barrier")
        return self._handed_full(index)

    def empty_barrier(self, index):
        """This CTA's empty barrier of a stage/slot (a pair's peer: same offset)."""
        self._mbarriers("empty_barrier")
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
    def ctas(self) -> tuple[int, ...]:
        """CTA ranks this end's role runs in."""
        cluster = self._chan.cluster
        return (0,) if cluster is None else cluster.ctas(self._cta)

    @property
    def targets(self) -> dict:
        """For each CTA rank this end runs in, the ranks whose barrier its signal reaches."""
        sc = self._chan.scope
        table = sc.full_targets if isinstance(self, ProducerEnd) else sc.empty_targets
        return {r: table.get(r, ()) for r in self.ctas}

    def _arrive(self, mech, bar, fence, mask):
        """Signal ``bar`` in each CTA of ``self.targets``; ``mask``: the ``tcgen05.commit``
        multicast mask (an int, or ``{rank: mask}``)."""
        _fence(fence)
        cluster, targets = self._chan.cluster, self.targets
        if mech is Umma:
            # Without pairs a commit to this CTA's own barrier needs no mask. Every tcgen05 op of
            # a pair kernel is cta_group::2; the multicast mask names the CTAs whose barrier
            # (same offset) receives the arrival.
            multicast = cluster is not None and (
                cluster.pair or any(cs != (r,) for r, cs in targets.items())
            )
            if multicast and isinstance(mask, dict):
                mask = rank_value({r: mask[r] for r in self.ctas}, cluster_rank)
            with cute.arch.elect_one():
                if multicast:
                    cute.nvgpu.tcgen05.commit(bar, mask, cluster.cta_group)
                else:
                    cute.nvgpu.tcgen05.commit(bar)
        elif mech is Threads:
            if not self._chan.scope.pair:
                cute.arch.mbarrier_arrive(bar)
                return
            # See Channel.remote_scope: a cluster-scope release orders generic-memory data for
            # the other CTA's waiter; "cta" (CUTLASS's default) only counts the arrival.
            scope = _nvvm.MemScopeKind.CLUSTER if self._chan.remote_scope == "cluster" else None
            shared = set(targets.values())
            if len(shared) == 1:  # every CTA of the role signals the same ranks
                for c in shared.pop():
                    if self.ctas == (c,):
                        cute.arch.mbarrier_arrive(bar, scope=scope)
                    else:
                        cute.arch.mbarrier_arrive(bar, c, scope=scope)
                return
            # Targets relative to the signalling CTA (its MMA pair): pick them by rank.
            widths = {len(cs) for cs in targets.values()}
            assert len(widths) == 1, targets
            rank = cluster_rank()
            for k in range(widths.pop()):
                c = rank_value({r: cs[k] for r, cs in targets.items()}, rank)
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
            self._state.advance(self._stride)
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
        if self._chan.named:
            self._check_turn("producer", "acquire")
            self._named_wait(self._chan.named_ids[1], index, "acquire")
            return Handle(self, index, phase, label, True)
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
            return
        arms = {r: cs for r, cs in self.targets.items() if cs}
        if all(cs == (r,) for r, cs in arms.items()):
            targets = (None,)  # each arming CTA arms its own barrier
        else:
            assert len(arms) == 1, arms  # one CTA arms the barriers of its targets
            ((rank, targets),) = arms.items()
            targets = tuple(None if c == rank else c for c in targets)
        if len(arms) == len(self.ctas):
            _arm_targets(full, scope.tx, targets)
        else:
            _arm_if(full, scope.tx, targets, tuple(arms), self._chan.cluster.size)

    def _signal(self, index):
        if isinstance(self._chan.producer, Tma):
            return  # completed by the TMA transaction count
        chan = self._chan
        if chan.named:
            _fence(chan.commit_fence)
            self._named_arrive(chan.named_ids[0], index)
            return
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
        if self._chan.named:
            self._check_turn("consumer", "wait")
            self._named_wait(self._chan.named_ids[0], index, "wait")
            return Handle(self, index, phase, label, False)
        with wait_range(self._trace, f"{self._chan.name}.wait"):
            cute.arch.mbarrier_wait(self._full(index), phase)
        return Handle(self, index, phase, label, False)

    consume = wait

    def _signal(self, index):
        chan = self._chan
        if chan.named:
            _fence(chan.release_fence)
            self._named_arrive(chan.named_ids[1], index)
            return
        self._arrive(chan.consumer, self._empty(index), chan.release_fence, chan.scope.empty_mask)


def _named_sync(barrier_id, threads: int):
    """Unaligned ``barrier.cta.sync``: roles reach a channel's barriers from different code
    sites, and lanes of one warp may arrive from divergent paths."""
    _nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(), thread_count=Int32(threads).ir_value(), aligned=False
    )


def _named_arrive(barrier_id, threads: int):
    cute.arch.barrier_arrive(barrier_id=barrier_id, number_of_threads=threads, aligned=False)


def cluster_rank():
    """This CTA's cluster rank without the warp-uniform hint (safe in divergent code)."""
    return cute.arch.block_idx_in_cluster()


def _arm_one(full, tx: int, cta: int | None):
    if cta is None:  # this CTA's own barrier
        cute.arch.mbarrier_arrive_and_expect_tx(full, tx)
    else:
        cute.arch.mbarrier_arrive_and_expect_tx(full, tx, cta)


def _arm_targets(full, tx: int, targets):
    with cute.arch.elect_one():
        for c in targets:
            _arm_one(full, tx, c)


@cute.jit
def _arm_if(
    full,
    tx: cutlass.Constexpr,
    targets: cutlass.Constexpr,
    ctas: cutlass.Constexpr,
    size: cutlass.Constexpr,
):
    """Arm ``targets`` from the CTAs ``ctas`` only (a pair's leader)."""
    if rank_in(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()), ctas, size):
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
        e._trace.open_turns.add(e._ts.name)
        if e._ts.enabled:
            e._trace.record(e._ts.name, "turn", e._seat)
            with wait_range(e._trace, f"{e._ts.name}.turn"):
                cute.arch.mbarrier_wait(e._bars + e._seat, e._phase)

    def __exit__(self, exc_type, exc, tb):
        e = self._end
        e._trace.open_turns.discard(e._ts.name)
        if exc_type is None and e._ts.enabled:
            nxt = (e._seat + 1) % len(e._ts.parties)
            cute.arch.mbarrier_arrive(e._bars + nxt)
            e._phase = e._phase ^ 1
            e._trace.record(e._ts.name, "pass", nxt)
        return False
