"""Role declarations and the ``WarpSpecialized`` kernel base."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace
from typing import TypeVar

import cutlass
from cutlass import Int32, cute, memory, pipeline
from cutlass._mlir.dialects import nvvm

from transformer_nuggets.cute.warpdrive import alias
from transformer_nuggets.cute.warpdrive.channels import (
    Barriers,
    Channel,
    ConsumerEnd,
    ProducerEnd,
    StageBuffer,
    Threads,
    Tma,
    TurnEnd,
    Turnstile,
    Umma,
)
from transformer_nuggets.cute.warpdrive.cluster import (
    ALL,
    LEADER,
    PLACEMENTS,
    SingleCta,
    apply_overrides,
    check_atoms,
    derive_scope,
)
from transformer_nuggets.cute.warpdrive.describe import describe_kernel
from transformer_nuggets.cute.warpdrive.protocol import (
    ProtocolError,
    ProtocolLog,
    RoleTrace,
    take_opt_outs,
    tracing_role,
)
from transformer_nuggets.cute.warpdrive.scheduling import (
    WorkCursor,
    WorkStream,
    _Fetch,
    scheduler_role,
)
from transformer_nuggets.cute.warpdrive.storage import SmemPlan, allocate, stage_extent
from transformer_nuggets.cute.warpdrive.tma import TmaReport
from transformer_nuggets.cute.warpdrive.tma import check_role as check_tma
from transformer_nuggets.cute.warpdrive.tmem import TmemPlan, TmemView
from transformer_nuggets.cute.warpdrive.tracing import pop, pop_if, push

WARP_SIZE = 32
WARPGROUP = 4
REGS_PER_SM = 65536
MIN_REGS = 24  # setmaxnreg floor: the budget of a warpgroup no role owns
MAX_WARPS = 32  # CUDA's 1024-thread block limit
# Named barrier ids owned by the layer; user code must not reuse them.
TMEM_PTR_BARRIER_ID = 1
TMEM_TEARDOWN_BARRIER_ID = 2
# Named-barrier channels take IDs from the top down, so hand-picked user IDs from 3 upward
# (persistent_gemm's epilogue barrier) stay clear of them.
MAX_NAMED_BARRIER_ID = 15


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #


class _Bound:
    """``Consumer[Instance]``: an end restricted to the slot equal to the instance index."""

    def __init__(self, side):
        self.side = side


class _SideMeta(type):
    def __getitem__(cls, key) -> _Bound:
        if key is not Instance:
            raise TypeError(f"{cls.__name__}[...] takes Instance, got {key!r}")
        return _Bound(cls)


class Producer(metaclass=_SideMeta):
    """Annotation: producer end of the channel with the parameter's name."""


class Consumer(metaclass=_SideMeta):
    """Annotation: consumer end of the channel with the parameter's name."""


class Tmem:
    """Annotation: the role receives a ``TmemView``."""


class Instance:
    """Annotation: the role receives its instance index as a Python int (constexpr)."""


class Turn:
    """Annotation: the role receives its seat at the turnstile with the parameter's name."""


class _SchedulerCursor:
    """Internal annotation: the scheduler warp's own cursor (it decodes and publishes)."""


class _SchedulerFetch:
    """Internal annotation: the scheduler warp's private schedule channels (both ends)."""


class _Context:
    """Internal annotation: the parameter receives the context passed to ``run_roles``."""


_ANNOTATIONS = {
    "Producer": Producer,
    "Consumer": Consumer,
    "Tmem": Tmem,
    "Instance": Instance,
    "Turn": Turn,
    "WorkCursor": WorkCursor,
    "Barriers": Barriers,
}
# Parameters a raw role may take: no channel ends, cursors, or seats (they carry protocol state).
_RAW_ANNOTATIONS = (Instance, Tmem, Barriers, _Context)
# Layer classes a role might write in place of an annotation, and the annotation meant.
_LOOKALIKES = (
    (ProducerEnd, "Producer"),
    (ConsumerEnd, "Consumer"),
    (Channel, "Producer or Consumer"),
    (WorkStream, "WorkCursor"),
    (TmemPlan, "Tmem"),
    (TmemView, "Tmem"),
    (Turnstile, "Turn"),
    (TurnEnd, "Turn"),
)
_CTX_RULE = (
    "a parameter receives the context when it is annotated with the kernel's context type (the "
    "type of the object passed to run_roles, or a base of it) or is the role's single "
    "unannotated parameter"
)


def _resolve_param(role_name: str, pname: str, ann, fn) -> tuple[object, type | None]:
    """(annotation, context type) of one role parameter; context parameters get ``_Context``
    and their annotated type (``None`` when unannotated)."""
    if ann is inspect.Parameter.empty:
        return _Context, None
    if isinstance(ann, str):
        try:
            ann = eval(ann, {**fn.__globals__, **_ANNOTATIONS})
        except Exception as exc:
            raise ProtocolError(
                f"role {role_name}: parameter {pname} is annotated {ann!r}, which does not "
                f"resolve ({type(exc).__name__}: {exc})"
            ) from exc
    if isinstance(ann, _Bound) or ann in _ANNOTATIONS.values():
        return ann, None
    if isinstance(ann, type):
        for cls, meant in _LOOKALIKES:
            if issubclass(ann, cls):
                raise ProtocolError(
                    f"role {role_name}: parameter {pname} is annotated {ann.__name__}, a layer "
                    f"class rather than a role annotation; did you mean `{pname}: {meant}`?"
                )
        return _Context, ann
    raise ProtocolError(
        f"role {role_name}: parameter {pname} has unrecognized annotation {ann!r}; {_CTX_RULE}"
    )


@dataclass
class RoleInstance:
    name: str  # "softmax[0]", or the method name for single-instance roles
    role: str
    index: int
    first_warp: int
    num_warps: int
    regs: int | None
    fn: object
    params: list
    uses_tmem: bool = False
    cta: str = ALL  # "all": runs in every CTA of the cluster; "leader": leader CTA only
    raw: bool = False  # dispatched without ends, protocol log, or trace checks
    keyword_only: frozenset[str] = frozenset()

    @property
    def warps(self) -> range:
        return range(self.first_warp, self.first_warp + self.num_warps)

    @property
    def num_threads(self) -> int:
        return self.num_warps * WARP_SIZE


@dataclass(frozen=True)
class RoleConfig:
    """Host-side role placement, resolved before channels and launch resources are built.

    An int names one warp; a range names a contiguous group; a sequence of ranges creates role
    instances. Warps are normalized to an immutable tuple of ranges. ``regs=None`` requests no
    register adjustment, not automatic tuning. All roles in a warpgroup must agree.
    """

    warps: int | range | list[range] | tuple[range, ...]
    regs: int | None = None
    cta: str = ALL
    raw: bool = False

    def __post_init__(self):
        if self.cta not in PLACEMENTS:
            raise ValueError(f"role cta must be one of {PLACEMENTS}, got {self.cta!r}")
        if self.regs is not None and (
            type(self.regs) is not int or not MIN_REGS <= self.regs <= 256 or self.regs % 8 != 0
        ):
            raise ValueError(
                f"role regs must be None or an int multiple of 8 in [24, 256], got {self.regs!r}"
            )
        if type(self.raw) is not bool:
            raise ValueError(f"role raw must be a bool, got {self.raw!r}")
        if type(self.warps) is int:
            ranges = (range(self.warps, self.warps + 1),)
        elif isinstance(self.warps, range):
            ranges = (self.warps,)
        else:
            try:
                ranges = tuple(self.warps)
            except TypeError as exc:
                raise ValueError(
                    f"role warps must be an int, range, or list of ranges, got {self.warps!r}"
                ) from exc
        if not ranges:
            raise ValueError(f"role warps must contain at least one range, got {self.warps!r}")
        for r in ranges:
            if not isinstance(r, range) or r.step != 1 or len(r) == 0:
                raise ValueError(
                    f"role warps must be non-empty unit-step ranges, got {self.warps!r}"
                )
            if r.start < 0 or r.stop > MAX_WARPS:
                raise ValueError(f"role warps must lie in [0, {MAX_WARPS}), got {self.warps!r}")
        object.__setattr__(self, "warps", ranges)


KernelT = TypeVar("KernelT", bound="WarpSpecialized")


def role(
    warps: int | range | list[range] | tuple[range, ...] | None = None,
    regs: int | None = None,
    cta: str = ALL,
    raw: bool = False,
    *,
    config: Callable[[KernelT], RoleConfig] | None = None,
):
    """Mark a method as a warp role, using fixed placement or a per-instance config selector.

    ``@role(warps=4, regs=40)`` supplies a fixed RoleConfig. Alternatively,
    ``@role(config=lambda kernel: kernel.roles.load)`` selects one after construction, on first
    collection (describe, compile-cache lookup, or tracing). A named function or callable object
    works too. The selector is host code, never a device call; its result is snapshotted for
    that kernel instance. Construct a new instance to change a collected configuration.

    A list of warp ranges instantiates the method once per range; an Instance parameter receives
    its index. ``cta="leader"`` restricts execution to the leader of a pair; "all" is the default.
    ``raw=True`` accepts Instance, Tmem, Barriers and a context, but no managed channel ends or
    work cursors. Raw roles still take part in register allocation and TMEM teardown. Put this
    decorator directly above ``@cute.jit``; CuTeDSL re-applies it when preprocessing the source.
    """
    if config is not None:
        if not callable(config):
            raise TypeError("role config must be a callable returning RoleConfig")
        if warps is not None or regs is not None or cta != ALL or raw is not False:
            raise ValueError("role config cannot be combined with warps, regs, cta, or raw")
        declaration = config
    else:
        declaration = RoleConfig(warps, regs, cta, raw)

    def deco(fn):
        fn._warpdrive = declaration
        return fn

    return deco


# --------------------------------------------------------------------------- #
# Kernel base
# --------------------------------------------------------------------------- #


class WarpSpecialized:
    """Base for warp-specialized kernels.

    Subclasses declare ``Ring``/``Slots``/``Turnstile`` attributes and an optional
    ``tmem_plan`` in ``__init__``, and decorate role methods with ``@role``. Inside the
    subclass's ``@cute.kernel``::

        smem = self.begin_roles()   # barriers + payloads allocated and initialized, TMEM alloc
        ...user SMEM tensors and context...
        self.run_roles(ctx)         # init fence + CTA sync, dispatch, TMEM teardown

    The layer owns barrier storage and initialization, arrival counts, TMEM allocation and
    teardown, ``setmaxnreg`` per role, warpgroup padding, and the trace-time protocol checks.
    """

    tmem_plan: TmemPlan | None = None
    smem_plan: SmemPlan | None = None  # SMEM buffers channels bind with buffer= (storage.py)
    # Alias diagnostic (alias.py): "warn" (describe() plus one warning on findings), "strict"
    # (raise on a potential hazard), or "off". Host-only; it never changes the generated code.
    alias_check: str = "warn"
    trace_protocol: bool = False
    iket: bool = False  # emit IKET ranges named after roles and channels (tracing.py)
    iket_waits: bool = True  # with iket: also a range around every blocking channel wait
    cluster: SingleCta = SingleCta()  # or CtaPair(): one line switches to 2-CTA MMAs

    def __setattr__(self, name, value):
        if isinstance(value, WorkStream):
            value.owner = self  # the stream's work unit follows this kernel's cluster
        super().__setattr__(name, value)

    # ---- host-side analysis ------------------------------------------------ #

    def _collect(self):
        if getattr(self, "_ws_roles", None) is not None:
            return
        chans = {k: v for k, v in vars(self).items() if isinstance(v, Channel)}
        turns = {k: v for k, v in vars(self).items() if isinstance(v, Turnstile)}
        raw_bars = {k: v for k, v in vars(self).items() if isinstance(v, Barriers)}
        for name, b in raw_bars.items():
            b.name = name
        streams = {k: v for k, v in vars(self).items() if isinstance(v, WorkStream)}
        if len(streams) > 1:
            raise ProtocolError(
                "one WorkStream per kernel (every role starts at this CTA's first item), got "
                f"{list(streams)}"
            )
        seen = {}
        for name, obj in (*chans.items(), *turns.items(), *raw_bars.items(), *streams.items()):
            if id(obj) in seen:
                raise ProtocolError(
                    f"attributes {seen[id(obj)]!r} and {name!r} use the same declaration; "
                    "give each channel, turnstile, barrier set, and stream its own object"
                )
            seen[id(obj)] = name
        cluster = self.cluster
        for name, st in streams.items():
            st.name = name
            st.owner = self
            if st.schedule.broadcast:
                generated = {f"{name}_ring": st.channel()}
                generated.update(
                    {f"{name}_{k}": ch for k, ch in st.schedule.fetch_channels().items()}
                )
                for k, ch in generated.items():
                    if k in vars(self):
                        raise ProtocolError(
                            f"WorkStream {name}: attribute {k!r} is reserved for its scheduler"
                        )
                    chans[k] = ch
        for name, obj in (*chans.items(), *turns.items()):
            obj.name = name
            if isinstance(obj, Channel):
                obj.producers.clear()
                obj.consumers.clear()
            else:
                obj.parties.clear()
        plan = self.smem_plan
        for name, ch in chans.items():
            for key, buf in ch.buffers.items():
                if plan is None or plan.buffers.get(buf.name) is not buf:
                    raise ProtocolError(
                        f"channel {name}: buffer {key!r} ({buf.name}) is not a buffer of this "
                        "kernel's smem_plan; declare it with self.smem_plan.buffer(...)"
                    )
        roles = []
        ctx_types = {}  # annotated context type -> first "role.param" using it
        for name in dir(type(self)):
            fn = getattr(type(self), name)
            meta = getattr(fn, "_warpdrive", None)
            if meta is None:
                continue
            resolved = meta if isinstance(meta, RoleConfig) else meta(self)
            if not isinstance(resolved, RoleConfig):
                raise ProtocolError(
                    f"role {name}: config selector must return RoleConfig, "
                    f"got {type(resolved).__name__}"
                )
            ranges, regs, cta, raw = resolved.warps, resolved.regs, resolved.cta, resolved.raw
            if not hasattr(fn, "__wrapped__"):
                raise TypeError(f"role {name}: put @role(...) directly above @cute.jit")
            signature = inspect.signature(fn)
            for p in signature.parameters.values():
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise ProtocolError(
                        f"role {name}: variadic parameter {p.name} cannot be injected; "
                        "declare each parameter explicitly"
                    )
            keyword_only = frozenset(
                p.name
                for p in signature.parameters.values()
                if p.kind is inspect.Parameter.KEYWORD_ONLY
            )
            params = []
            for pname, p in list(signature.parameters.items())[1:]:
                ann, ctx_type = _resolve_param(name, pname, p.annotation, fn)
                if raw and ann not in _RAW_ANNOTATIONS:
                    raise ProtocolError(
                        f"raw role {name}: parameter {pname} is {getattr(ann, '__name__', ann)}; "
                        "raw roles take Instance, Tmem, Barriers, and the context only (drive "
                        "edges through Barriers, or make the role managed)"
                    )
                if ann is Barriers and pname not in raw_bars:
                    raise ProtocolError(f"role {name}: no Barriers named {pname!r}")
                if ann is WorkCursor and pname not in streams:
                    raise ProtocolError(f"role {name}: no WorkStream named {pname!r}")
                if ctx_type is not None:
                    ctx_types.setdefault(ctx_type, f"{name}.{pname}")
                params.append((pname, ann))
            ctx_params = [p for p, a in params if a is _Context]
            if len(ctx_params) > 1:
                raise ProtocolError(
                    f"role {name}: parameters {ctx_params} would all receive the context; "
                    f"{_CTX_RULE}"
                )
            for idx, wr in enumerate(ranges):
                inst = f"{name}[{idx}]" if len(ranges) > 1 else name
                ri = RoleInstance(
                    inst,
                    name,
                    idx,
                    wr.start,
                    len(wr),
                    regs,
                    fn,
                    params,
                    cta=cta,
                    raw=raw,
                    keyword_only=keyword_only,
                )
                ri.uses_tmem = any(a is Tmem for _, a in params)
                roles.append(ri)
                self._bind(ri, chans, turns)
        for t, site in ctx_types.items():
            if not (is_dataclass(t) and t.__dataclass_params__.frozen):
                raise ProtocolError(
                    f"{site} is annotated {t.__name__}, which is not a frozen dataclass; an "
                    "annotated context reaches the roles as itself, which the layer supports "
                    "for frozen dataclasses (leave a SimpleNamespace context unannotated)"
                )
        # The context type is the most derived annotation; the others must be its bases.
        derived = [t for t in ctx_types if all(issubclass(t, u) for u in ctx_types)]
        if ctx_types and not derived:
            sites = ", ".join(f"{site}: {t.__name__}" for t, site in ctx_types.items())
            raise ProtocolError(
                f"roles annotate the context with unrelated types ({sites}); {_CTX_RULE}"
            )
        self._ws_ctx_type = derived[0] if derived else None
        self._ws_ctx_site = ctx_types.get(self._ws_ctx_type)
        for name, st in streams.items():
            self._bind_stream(name, st, roles, chans, cluster)
        roles.sort(key=lambda r: r.first_warp)
        self._validate(chans, turns, roles)
        self._validate_placement(chans, turns, roles)
        self._ws_named = _allocate_named(chans)
        self._ws_chans, self._ws_turns, self._ws_roles = chans, turns, roles
        self._ws_raw_bars = raw_bars
        self._ws_streams = streams
        self._ws_tma = {}  # channel name -> tma.TmaReport, filled while tracing
        self._ws_smem = {}  # SmemPlan buffer name -> storage.ResolvedBuffer, after begin_roles
        self._ws_alias = None  # alias.AliasReport, after a trace
        self._ws_opt_outs = []
        self._ws_log = ProtocolLog() if self.trace_protocol else None

    @staticmethod
    def _bind_stream(name: str, st: WorkStream, roles: list, chans: dict, cluster):
        """Check stream users; for broadcast schedules add the scheduler role and bind every
        stream user (and the scheduler) as a consumer of the work ring."""
        users = [r for r in roles if any(a is WorkCursor for _, a in r.params)]
        st.roles = [r.name for r in sorted(users, key=lambda r: r.first_warp)]
        for r in users:
            for pname, ann in r.params:
                if ann is WorkCursor and pname != name:
                    raise ProtocolError(f"role {r.name}: no WorkStream named {pname!r}")
        if st.schedule.persistent:
            idle = sorted({r.role for r in roles if not r.raw} - {r.role for r in users})
            if idle:
                raise ProtocolError(
                    f"{st.schedule.describe()} runs several work items per CTA, but roles {idle} "
                    f"do not take `{name}: WorkCursor`; every role must loop over the stream "
                    "(a role with no per-item work drains it: "
                    f"`while {name}.valid: {name}.advance()`)"
                )
        if not st.schedule.broadcast:
            return
        owned = {w for r in roles for w in r.warps}
        warp = st.scheduler_warp
        if warp is None:
            warp = next(w for w in range(max(owned, default=-1) + 2) if w not in owned)
        elif warp in owned:
            raise ValueError(f"WorkStream {name}: scheduler_warp {warp} is claimed by a role")
        group = range(warp // WARPGROUP * WARPGROUP, (warp // WARPGROUP + 1) * WARPGROUP)
        # The scheduler shares its warpgroup's budget (setmaxnreg is per warpgroup), including a
        # deliberate None; only an otherwise idle warpgroup takes the smallest declared budget.
        budgets = {r.regs for r in roles if set(r.warps) & set(group)}
        any_regs = [r.regs for r in roles if r.regs is not None]
        if budgets:
            regs = budgets.pop() if len(budgets) == 1 else min(b for b in budgets if b is not None)
        else:
            regs = min(any_regs) if any_regs else None
        ring = f"{name}_ring"
        fetch = [f"{name}_{k}" for k in st.schedule.fetch_channels()]
        params = [(ring, Producer), (name, _SchedulerCursor), (name, _SchedulerFetch)]
        # In a CTA pair only the leader claims; it publishes each item into both CTAs.
        sched = RoleInstance(
            f"{name}.scheduler",
            f"{name}.scheduler",
            0,
            warp,
            1,
            regs,
            scheduler_role,
            params,
            cta=LEADER if cluster.size > 1 else ALL,
        )
        roles.append(sched)
        st.roles.append(sched.name)
        chans[ring].producers.append((sched, None))
        for r in [*users, sched]:
            chans[ring].consumers.append((r, None))
        for f in fetch:
            chans[f].producers.append((sched, None))
            chans[f].consumers.append((sched, None))

    @staticmethod
    def _bind(ri: RoleInstance, chans, turns):
        for pname, ann in ri.params:
            side = ann.side if isinstance(ann, _Bound) else ann
            if side is Producer or side is Consumer:
                if pname not in chans:
                    raise ProtocolError(f"role {ri.name}: no channel named {pname!r}")
                chan = chans[pname]
                slot = None
                if isinstance(ann, _Bound):
                    if not chan.fixed:
                        raise ProtocolError(f"role {ri.name}: ring {pname} cannot be slot-bound")
                    slot = ri.index
                    if slot >= chan.n:
                        raise ProtocolError(f"role {ri.name}: slot {slot} >= {pname}.n={chan.n}")
                if side is Producer:
                    declared = chan.producer_role
                    if declared is not None and declared != ri.role:
                        raise ProtocolError(
                            f"channel {pname}: producer is declared as role {declared!r}, but "
                            f"role {ri.name} takes its Producer end"
                        )
                    chan.producers.append((ri, slot))
                else:
                    declared = chan.consumer_roles
                    if declared is not None and ri.role not in declared:
                        raise ProtocolError(
                            f"channel {pname}: consumers are declared as {declared}, but role "
                            f"{ri.name} takes its Consumer end"
                        )
                    chan.consumers.append((ri, slot))
            elif ann is Turn:
                if pname not in turns:
                    raise ProtocolError(f"role {ri.name}: no turnstile named {pname!r}")
                turns[pname].parties.append(ri)

    def _validate_placement(self, chans, turns, roles):
        """Cluster checks on role placement, then derive every channel's scope."""
        cluster = self.cluster
        if cluster.size > 1:
            for name, ch in chans.items():
                for side, mech, bound in (
                    ("producer", ch.producer, ch.producers),
                    ("consumer", ch.consumer, ch.consumers),
                ):
                    for ri, _ in bound:
                        if mech is Umma and ri.cta != LEADER:
                            raise ProtocolError(
                                f"role {ri.name} signals channel {name} ({side} side) with "
                                f"tcgen05.commit, so it issues the MMAs; in a "
                                f"{type(cluster).__name__} kernel CtaGroup.TWO MMAs are issued "
                                "by the leader CTA only: pin it with @role(..., cta='leader')"
                            )
            for name, ts in turns.items():
                if len({r.cta for r in ts.parties}) != 1:
                    raise ProtocolError(f"turnstile {name}: seats run in different CTAs")
            tmem_roles = [r for r in roles if r.uses_tmem]
            if tmem_roles and not any(r.cta == ALL for r in tmem_roles):
                raise ProtocolError(
                    "2-CTA TMEM is allocated by one warp in each CTA of the pair, but every "
                    f"Tmem role {[r.name for r in tmem_roles]} is leader-only"
                )
        for name, ch in chans.items():
            ch.cluster = cluster
            ch.derived_scope = derive_scope(ch, cluster, internal=ch.internal)
            ch.scope = apply_overrides(ch, ch.derived_scope, cluster)
            if ch.scope.pair and (ch.named or ch.round_robin):
                what = "named barriers are per CTA" if ch.named else "round_robin is CTA-local"
                raise ProtocolError(
                    f"channel {name}: its roles run in different CTAs of the pair, but {what}; "
                    "give both sides the same placement"
                )

    def _validate(self, chans, turns, roles):
        if not roles:
            raise ValueError("warp-specialized kernels need at least one role")
        owner = {}
        for r in roles:
            for w in r.warps:
                if w in owner:
                    raise ValueError(f"warp {w} claimed by {owner[w]} and {r.name}")
                owner[w] = r.name
        for name, chan in chans.items():
            for side in ("producer", "consumer"):
                mech = chan.producer if side == "producer" else chan.consumer
                bound = chan.producers if side == "producer" else chan.consumers
                counts = chan.slot_counts(side)
                if 0 in counts:
                    raise ProtocolError(
                        f"channel {name}: slot {counts.index(0)} has no {side} role"
                    )
                if chan.round_robin and len(bound) > 1:
                    _check_turns(name, side, bound)
                if mech is Threads:
                    continue
                how = getattr(mech, "__name__", repr(mech))  # Umma, or Tma(tx_bytes=...)
                if len(bound) > 1 and not chan.round_robin:
                    raise ProtocolError(
                        f"channel {name}: {side} side signals once ({how}) but has several "
                        f"roles {[r.name for r, _ in bound]}"
                    )
                # One elected thread signals; every warp of a wider role would issue its own
                # arrive (and its own TMA copies or MMAs).
                for r, _ in bound:
                    if r.num_warps > 1:
                        raise ProtocolError(
                            f"channel {name}: {side} side signals once ({how}) from one "
                            f"elected thread, but role {r.name} spans {r.num_warps} warps and "
                            "each would signal; issue it from a single-warp role"
                        )
            if chan.round_robin and chan.turns("producer") == chan.turns("consumer") == 1:
                raise ProtocolError(
                    f"channel {name}: round_robin needs several instances of one role on a side "
                    f"(producers {[r.name for r, _ in chan.producers]}, consumers "
                    f"{[r.name for r, _ in chan.consumers]})"
                )
            if chan.named:
                _check_named_turns(name, chan, turns)
        for name, ts in turns.items():
            if len(ts.parties) < 2 or len({r.role for r in ts.parties}) != 1:
                raise ProtocolError(f"turnstile {name} needs >= 2 instances of one role")
            ts.parties.sort(key=lambda r: r.index)
        if self.tmem_plan is not None:
            _ = self.tmem_plan.alloc_cols  # raises when over capacity
        elif any(r.uses_tmem for r in roles):
            raise ProtocolError("roles take Tmem but the kernel has no tmem_plan")
        num_warps = max(owner) + 1
        if min(owner) < 0 or num_warps > MAX_WARPS:
            raise ValueError(f"role warps must lie in [0, {MAX_WARPS})")
        uses_regs = any(r.regs is not None for r in roles)
        # setmaxnreg is .aligned per warpgroup: pad to whole warpgroups and require every warp
        # of a warpgroup to request the same budget.
        self.num_warps = -(-num_warps // WARPGROUP) * WARPGROUP if uses_regs else num_warps
        self.default_regs = min(REGS_PER_SM // (self.num_warps * WARP_SIZE) // 8 * 8, 256)
        # (warps, regs) for each run of warps no role owns, holes included: they take their
        # warpgroup's budget (MIN_REGS for an all-idle warpgroup).
        self._ws_padding = []
        # One budget per warpgroup, applied before role dispatch (see run_roles).
        self._ws_wg_regs = []
        if uses_regs:
            regs_of = {w: r.regs for r in roles for w in r.warps}
            for g in range(self.num_warps // WARPGROUP):
                group = range(g * WARPGROUP, (g + 1) * WARPGROUP)
                budgets = {owner[w]: regs_of[w] for w in group if w in owner}
                vals = set(budgets.values())
                if len(vals) > 1:
                    raise ValueError(f"warpgroup {g} has mixed register budgets {budgets}")
                regs = vals.pop() if vals else MIN_REGS
                self._ws_wg_regs.append(regs)
                idle = [w for w in group if w not in owner]
                self._ws_padding += [(run, regs) for run in _runs(idle)]
            total = sum(
                (regs if regs is not None else self.default_regs) * len(warps) * WARP_SIZE
                for warps, regs in [*((r.warps, r.regs) for r in roles), *self._ws_padding]
            )
            if total > REGS_PER_SM:
                raise ValueError(f"register budgets need {total} > {REGS_PER_SM}")
        # (regs, warpgroups, role indices): one dispatch branch per register budget.
        wg_of = [w // WARPGROUP for w in range(self.num_warps)]
        if self._ws_wg_regs:
            budgets = list(dict.fromkeys(self._ws_wg_regs))
            self._ws_budget_groups = [
                (
                    b,
                    tuple(g for g, regs in enumerate(self._ws_wg_regs) if regs == b),
                    tuple(
                        i
                        for i, r in enumerate(roles)
                        if self._ws_wg_regs[wg_of[r.first_warp]] == b
                    ),
                )
                for b in budgets
            ]
        else:
            self._ws_budget_groups = [
                (None, tuple(range(-(-self.num_warps // WARPGROUP))), tuple(range(len(roles))))
            ]

    def role_config_key(self) -> tuple:
        """Resolved placement/budgets, independent of selector identity. First use collects and
        freezes the role configuration. Include this value in custom compilation caches."""
        self._collect()
        return tuple(
            (r.role, r.index, r.first_warp, r.num_warps, r.regs, r.cta, r.raw)
            for r in self._ws_roles
        )

    def __cute_compile_key__(self) -> tuple:
        """Extra key consumed automatically by transformer_nuggets.cute.cache. This supplements
        the author's algorithm key; it does not replace keys for tiles, dtypes or other flags."""
        return (
            type(self).__module__,
            type(self).__qualname__,
            self.role_config_key(),
            self.cluster.get_key(),
            self.iket_key(),
            self.trace_key(),
            self.alias_check,  # a cached kernel skips the trace-time diagnostic (strict too)
        )

    def enable_iket(self, enabled: bool = True, *, waits: bool = True):
        """Emit IKET ranges in the next compile (``tracing.py`` lists the names). Off: no IR.
        ``waits=False`` keeps only the role/work-item ranges (fine-grained wait ranges cost
        ~35% on FA4 under ``run-iket``; role ranges alone cost about nothing). Include
        ``iket_key()`` in the compile-cache key (the examples' ``get_name`` does)."""
        self.iket, self.iket_waits = enabled, waits
        return self

    def trace_key(self) -> str:
        """Compile-key suffix of ``trace_protocol``: a cached kernel was never traced, so a
        traced instance needs its own entry to fill the protocol log."""
        return "_trace" if self.trace_protocol else ""

    def iket_key(self) -> str:
        """Compile-key suffix of the IKET configuration ('' when off)."""
        if not self.iket:
            return ""
        return "_iket" if self.iket_waits else "_iketroles"

    @contextmanager
    def trace_region(self, name: str):
        """``with self.trace_region("exp"):`` inside a role: a named IKET range when IKET is
        enabled, nothing otherwise."""
        if self.iket:
            push(name)
        yield
        if self.iket:
            pop()

    @property
    def num_threads(self) -> int:
        self._collect()
        return self.num_warps * WARP_SIZE

    def describe(self) -> str:
        """Plain-text summary of what the layer derived: roles and register budgets, channels
        (arrival counts per CTA, TMA bytes, masks, overrides, TMA check status), streams, TMEM,
        and, after a compile, check opt-outs and the traced protocol (``describe.py``)."""
        self._collect()
        return describe_kernel(self)

    def _barrier_plan(self):
        """(owner, offset, {cta rank: arrival counts}) for every mbarrier, in allocation order.

        Every CTA of a cluster allocates the same barriers at the same offsets (remote arrivals
        address the peer's barrier by offset); counts may differ per rank."""
        plan, off = [], 0
        ranks = range(self.cluster.size)
        for name, ch in self._ws_chans.items():
            if ch.named:
                continue  # hardware named barriers: no SMEM, no initialization
            sc = ch.scope
            plan.append((name, off, {c: sc.full_counts[c] + sc.empty_counts[c] for c in ranks}))
            off += 2 * ch.n
        for name, ts in self._ws_turns.items():
            if ts.enabled:
                n = len(ts.parties)
                counts = [ts.parties[(p - 1) % n].num_threads for p in range(n)]
                plan.append((name, off, {c: counts for c in ranks}))
                off += n
        for name, b in self._ws_raw_bars.items():
            counts = b.counts or [None] * b.n  # None: initialized by user code
            plan.append((name, off, {c: counts for c in ranks}))
            off += b.n
        return plan, off

    @property
    def named_barrier_ids(self) -> dict:
        """``{channel: ([full id per slot], [empty id per slot])}`` of the named-barrier channels.
        They are allocated from 15 downward, after the layer's IDs 1 and 2; hand-picked IDs in
        user code must avoid them (``alias.named_barrier`` calls are checked while tracing)."""
        self._collect()
        return dict(self._ws_named)

    def barrier_ptr(self, name: str):
        """Pointer to the first mbarrier of a channel (full barriers at ``[0, n)``, empty at
        ``[n, 2n)``), an enabled turnstile, or a ``Barriers`` declaration. Valid after
        ``begin_roles()``; every CTA of a cluster has them at the same offsets."""
        if name not in self._ws_bars:
            raise ProtocolError(f"no barriers named {name!r} (have {sorted(self._ws_bars)})")
        return self._ws_bars[name]

    def launch_config(self, work=None, grid=None) -> dict:
        """``grid``/``block``/``cluster`` for ``kernel(...).launch(**cfg, stream=stream)``.
        Kernels with a ``WorkStream`` pass its ``BoundWork``: the grid depends on the schedule
        (a ``OneShot`` over plain ``Raster()`` launches the tile space)."""
        if work is not None:
            grid = work.grid()
        return {
            "grid": grid,
            "block": (self.num_threads, 1, 1),
            "cluster": self.cluster.launch_cluster(),
        }

    # ---- device-side ------------------------------------------------------- #

    def begin_roles(self, layouts: dict | None = None, **problems):
        """Allocate and initialize barriers and payload buffers; allocate TMEM; allocate the
        ``smem_plan`` buffers.

        ``problems``: the ``BoundWork`` of each declared ``WorkStream``, by name
        (``self.begin_roles(work=bound)``). ``layouts``: the layout (plain or swizzled) of every
        ``smem_plan`` buffer, by name. Plan buffers are allocated last, in declaration order, so
        ``smem`` allocations after this call follow them; ``self.buffer(name)`` returns each.

        Plain Python at trace time (the loops are over host-side declarations); only the
        barrier initialization is a device-side branch.
        """
        self._collect()
        if set(problems) != set(self._ws_streams):
            raise ProtocolError(
                f"begin_roles needs a problem for each WorkStream {sorted(self._ws_streams)}, "
                f"got {sorted(problems)}"
            )
        self._ws_problems = problems
        for problem in problems.values():
            problem.start()  # per-CTA start values, computed once ahead of role dispatch
        self._ws_tma = {}
        self._ws_alias_roles = []  # alias.RoleEvents of every dispatched role
        self._ws_uses = {}  # (role, channel) -> acquires/waits per work item (0: inner loop)
        take_opt_outs()  # drop opt-outs left over from an earlier, failed trace
        smem = memory.SmemAllocator()
        plan, total = self._barrier_plan()
        bars = smem.allocate_array(cutlass.Int64, max(total, 1))
        self._ws_bars = {name: bars + off for name, off, _ in plan}
        self._ws_bar_counts = [
            [c for _, _, counts in plan for c in counts[rank]] for rank in range(self.cluster.size)
        ]
        self._init_barriers(bars)
        self._ws_buffers = {}  # channel name -> {buffer name: StageBuffer}
        for name, ch in self._ws_chans.items():
            if ch.payload is not None:
                dtype, shape = ch.payload
                payload = smem.allocate_tensor(dtype, cute.make_layout((*shape, ch.n)), 16)
                self._ws_buffers[name] = {"payload": StageBuffer(payload, -1, None)}
        self._ws_tmem = None
        pair = self.cluster.size > 1
        tmem_roles = [r for r in self._ws_roles if r.uses_tmem]
        if tmem_roles:
            # In a pair both CTAs allocate together (tcgen05.alloc.cta_group::2), so the
            # allocator is the first TMEM role that runs in both CTAs.
            alloc_role = next(r for r in tmem_roles if not pair or r.cta == ALL)
            holder = smem.allocate_array(cutlass.Int32, 1)
            dealloc_bar = smem.allocate_array(cutlass.Int64, 1) if pair else None
            tmem = memory.TmemAllocator(
                holder,
                barrier_for_retrieve=pipeline.NamedBarrier(
                    barrier_id=TMEM_PTR_BARRIER_ID, num_threads=self.num_threads
                ),
                allocator_warp_id=alloc_role.first_warp,
                is_two_cta=pair,
                two_cta_tmem_dealloc_mbar_ptr=dealloc_bar,
            )
            self._ws_tmem = tmem
            self._ws_tmem_alloc_role = alloc_role
            if not pair:  # a pair allocates after the cluster sync in run_roles
                # Only the allocator warp executes tcgen05.alloc (checked inside).
                tmem.allocate(self.tmem_plan.alloc_cols)
                tmem.relinquish_alloc_permit()
        self._allocate_plan(smem, layouts or {})
        return smem

    def _allocate_plan(self, smem, layouts: dict):
        """Allocate every ``smem_plan`` buffer and check each channel binding's stage count."""
        plan = self.smem_plan
        names = set() if plan is None else set(plan.buffers)
        if set(layouts) != names:
            raise ProtocolError(
                f"begin_roles(layouts=...) needs a layout for each smem_plan buffer "
                f"{sorted(names)}, got {sorted(layouts)}"
            )
        self._ws_smem = {}
        if plan is None:
            return
        for buf in plan.buffers.values():
            self._ws_smem[buf.name] = allocate(smem, buf, layouts[buf.name], self._ws_smem)
        for name, ch in self._ws_chans.items():
            if not ch.buffers:
                continue
            bound = {}
            for key, buf in ch.buffers.items():
                res = self._ws_smem[buf.name]
                res.channels.append(name)
                what = f"channel {name} buffer {key!r}"
                extent = stage_extent(res.tensor, buf.stage_axis, what)
                if extent != ch.n:
                    raise ProtocolError(
                        f"{what}: stage mode {buf.stage_axis} of {buf.name}'s layout has "
                        f"{extent} stages, but {name} has {ch.n}"
                    )
                bound[key] = StageBuffer(res.tensor, buf.stage_axis, buf.name)
            self._ws_buffers[name] = bound

    def buffer(self, name: str):
        """The whole tensor of ``smem_plan`` buffer ``name`` (valid after ``begin_roles``)."""
        if name not in self._ws_smem:
            raise ProtocolError(
                f"no allocated SMEM buffer named {name!r} (have {sorted(self._ws_smem)}); "
                "declare it in smem_plan and call begin_roles(layouts=...) first"
            )
        return self._ws_smem[name].tensor

    @cute.jit
    def _init_barriers(self, bars):
        leader = self._ws_bar_counts[0]
        peer = self._ws_bar_counts[-1]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            rank = self.cluster.rank()
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(len(leader)):
                    if cutlass.const_expr(leader[i] is None):
                        pass  # user-initialized Barriers
                    elif cutlass.const_expr(leader[i] == peer[i]):
                        cute.arch.mbarrier_init(bars + i, leader[i])
                    else:  # pair: rank 0 is the leader, rank 1 the peer
                        cute.arch.mbarrier_init(bars + i, leader[i] + (peer[i] - leader[i]) * rank)

    @cute.jit
    def run_roles(self, params):
        """Dispatch roles with the kernel-specific input params, then perform teardown."""
        params = self._role_context(params)
        pair = self.cluster.size > 1
        if cutlass.const_expr(not pair):
            # Fences mbarrier.init and syncs the CTA; the same bar.sync publishes the TMEM base
            # address written by tcgen05.alloc in begin_roles().
            pipeline.pipeline_init_arrive()
            pipeline.pipeline_init_wait()
        else:
            # Every barrier of the pair is initialized before any CTA arrives remotely.
            pipeline.pipeline_init_arrive(cluster_shape_mn=self.cluster.shape, is_relaxed=True)
            pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster.shape)
            if cutlass.const_expr(self._ws_tmem is not None):
                self._ws_tmem.allocate(self.tmem_plan.alloc_cols)
                self._ws_tmem.relinquish_alloc_permit()
                cute.arch.sync_threads()  # publishes the TMEM base address
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = self.cluster.rank()
        roles = self._ws_roles
        tmem_threads = (
            sum(r.num_threads for r in roles if r.uses_tmem),
            sum(r.num_threads for r in roles if r.uses_tmem and r.cta == ALL),
        )
        # setmaxnreg is .sync.aligned over a warpgroup: every warp of a group must execute the
        # same instruction. Branch once per register budget on a warpgroup-uniform condition,
        # adjust there, then split into roles inside that branch (as CUTLASS does), so ptxas
        # still allocates each role's code under its budget.
        groups = self._ws_budget_groups
        for gi in cutlass.range_constexpr(len(groups)):
            if _in_warpgroups(warp_idx, groups[gi][1]):
                self._dispatch_group(groups[gi], params, tmem_threads, warp_idx, rank)
        if cutlass.const_expr(pair):
            # Remote arrivals into a CTA must land before it exits: every thread-issued remote
            # arrive precedes its thread's cluster arrive (release), and producers whose stages
            # are released by the other CTA's tcgen05.commit drained them (tails) above.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        self._finish_trace()

    def _role_context(self, ctx):
        """The context the roles receive: a frozen dataclass as is, any other object (such as a
        ``SimpleNamespace``) as a ``RoleContext`` copy of its attributes.

        Checked against the roles' annotated context type and the cluster's MMA/TMA atoms."""
        want = self._ws_ctx_type
        if want is not None and not isinstance(ctx, want):
            raise ProtocolError(
                f"run_roles got a {type(ctx).__name__} context, but {self._ws_ctx_site} is "
                f"annotated {want.__name__}; {_CTX_RULE}"
            )
        if not hasattr(ctx, "__dict__") and not is_dataclass(ctx):
            raise ProtocolError(
                f"context {type(ctx).__name__} has no attributes to share; pass a frozen "
                "dataclass (or a SimpleNamespace)"
            )
        if is_dataclass(ctx):
            if not type(ctx).__dataclass_params__.frozen:
                raise ProtocolError(
                    f"context {type(ctx).__name__} is a mutable dataclass; declare it "
                    "@dataclass(frozen=True): every role shares it, and a field rebound inside "
                    "one role's dynamic branch would leak that region's values into other roles"
                )
            check_atoms({f.name: getattr(ctx, f.name) for f in fields(ctx)}, self.cluster)
            return ctx
        ctx = ctx if isinstance(ctx, RoleContext) else RoleContext(**vars(ctx))
        check_atoms(vars(ctx), self.cluster)
        return ctx

    def _finish_trace(self):
        """Collect check opt-outs; warn once about TMA channels the check could not verify; run
        the alias diagnostic."""
        self._ws_opt_outs = take_opt_outs()
        taken = {
            i: f"channel {name}"
            for name, ids in self._ws_named.items()
            for i in (*ids[0], *ids[1])
        }
        taken.update({TMEM_PTR_BARRIER_ID: "the layer", TMEM_TEARDOWN_BARRIER_ID: "the layer"})
        for role_events in self._ws_alias_roles:
            for ev in role_events.events:
                if ev.kind == "barrier" and ev.data.get("id") in taken:
                    raise ProtocolError(
                        f"{role_events.name}: named_barrier({ev.data['id']}) at {ev.site} uses a "
                        f"barrier ID owned by {taken[ev.data['id']]}; pick an unused ID"
                    )
        self._ws_alias = None
        if self.alias_check not in alias.MODES:
            raise ValueError(f"alias_check must be one of {alias.MODES}, got {self.alias_check!r}")
        if self.alias_check != "off":
            self._ws_alias = alias.check(self, self._ws_alias_roles)
            alias.report(self._ws_alias, self.alias_check, type(self).__name__)
        for name, ch in self._ws_chans.items():
            if isinstance(ch.producer, Tma) and not ch.internal:
                report = self._ws_tma.setdefault(name, TmaReport())
                if not (report.copies or report.opted_out or report.unverified):
                    report.unverified.append(
                        "no TMA copy completes on a barrier taken from the layer (raw pointer?)"
                    )
        unverified = [n for n, rep in self._ws_tma.items() if rep.unverified]
        if unverified:
            details = "; ".join(f"{n}: {self._ws_tma[n].status()}" for n in unverified)
            warnings.warn(
                f"warpdrive: TMA check could not verify channel(s) {unverified} (no error "
                f"raised; see describe()): {details}",
                stacklevel=2,
            )

    @cute.jit
    def _dispatch_group(self, group, ctx, tmem_threads, warp_idx, rank):
        regs, _, role_ids = group
        self._set_regs(regs)
        pair = self.cluster.size > 1
        for i in cutlass.range_constexpr(len(role_ids)):
            r = self._ws_roles[role_ids[i]]
            # Only a call on `self` inside the dynamic `if`: an assignment through `self`
            # would make the preprocessor carry `self` as an if-result.
            if cutlass.const_expr(pair and r.cta == LEADER):
                if _in_warps(warp_idx, r.first_warp, r.num_warps) & (rank == 0):
                    self._dispatch(r, ctx, tmem_threads)
            else:
                if _in_warps(warp_idx, r.first_warp, r.num_warps):
                    self._dispatch(r, ctx, tmem_threads)

    def _dispatch(self, r: RoleInstance, ctx, tmem_threads: tuple[int, int]):
        # Plain Python (no dynamic control flow): runs at trace time inside the role's `if`.
        trace = RoleTrace(
            r.name, None if r.raw else self._ws_log, iket=self.iket, iket_waits=self.iket_waits
        )
        args = self._role_args(r, ctx, trace)
        # A role looping over a WorkStream gets one IKET range per work item (WorkCursor);
        # any other role gets one over its whole body.
        whole_range = self.iket and not any(isinstance(arg, WorkCursor) for arg in args)
        if whole_range:
            push(r.name)
        positional, keywords = [], {}
        for (name, _), arg in zip(r.params, args):
            if name in r.keyword_only:
                keywords[name] = arg
            else:
                positional.append(arg)
        with tracing_role(trace):
            result = r.fn(self, *positional, **keywords)
        if not r.raw:
            state_args = [
                (pname, arg)
                for (pname, _), arg in zip(r.params, args)
                if isinstance(arg, _ROLE_STATE_TYPES)
            ]
            final = _returned_state(r.name, result, state_args)
            if self.iket:
                for _, arg in state_args:
                    if isinstance(arg, WorkCursor):
                        cur = final[_state_identity(arg)]
                        pop_if(cur.index < cur._problem.total)
            for _, arg in state_args:
                if isinstance(arg, ProducerEnd) and arg.scope.tail:
                    final[_state_identity(arg)].tail()
        if whole_range:
            pop()
        if not r.raw:
            check_tma(trace, self.cluster, self._ws_tma)
            trace.finish()
        for chan, n in trace.uses_per_item().items():
            self._ws_uses[(r.name, chan)] = n
        # The role's regions are live now; the diagnostic keeps plain data only.
        self._ws_alias_roles.append(alias.RoleEvents.from_trace(trace, r))
        if r.uses_tmem:
            # Every TMEM user of this CTA finishes before the allocator warp frees it (a pair's
            # peer has no leader-only TMEM roles, so its count is smaller).
            leader, peer = tmem_threads
            count = leader
            if r.cta == ALL and leader != peer:
                count = leader + (peer - leader) * self.cluster.rank()
            # Roles reach this from different code sites after per-lane mbarrier spins, so use
            # the unaligned form (compute-sanitizer synccheck flags the aligned bar.sync).
            _barrier_unaligned(TMEM_TEARDOWN_BARRIER_ID, count)
            if r is self._ws_tmem_alloc_role:
                ptr = self._ws_tmem.retrieve_ptr(cutlass.Float32)
                # Pair: the allocator warps of both CTAs meet on the dealloc mbarrier first.
                self._ws_tmem.free(ptr, self.tmem_plan.alloc_cols)

    def _set_regs(self, regs):
        if regs is None:
            return
        if regs > self.default_regs:
            cute.arch.setmaxregister_increase(regs)
        elif regs < self.default_regs:
            cute.arch.setmaxregister_decrease(regs)

    def _role_args(self, r: RoleInstance, ctx, trace: RoleTrace):
        args = []
        for pname, ann in r.params:
            side = ann.side if isinstance(ann, _Bound) else ann
            if side is Producer or side is Consumer:
                bound = r.index if isinstance(ann, _Bound) else None
                args.append(self._make_end(pname, side is Producer, bound, trace, r.cta, r))
            elif ann is Tmem:
                base = self._ws_tmem.retrieve_ptr(cutlass.Float32)
                args.append(TmemView(base, self.tmem_plan, r.name, trace.log, trace))
            elif ann is Barriers:
                args.append(self._ws_bars[pname])
            elif ann is WorkCursor or ann is _SchedulerCursor:
                st = self._ws_streams[pname]
                end = None
                if st.schedule.broadcast:
                    end = self._make_end(f"{pname}_ring", False, None, trace, r.cta)
                args.append(
                    WorkCursor.create(
                        st,
                        self._ws_problems[pname],
                        end,
                        trace,
                        scheduler=ann is _SchedulerCursor,
                        slot=(st.roles.index(r.name), r.first_warp),
                    )
                )
            elif ann is _SchedulerFetch:
                st = self._ws_streams[pname]
                ends = {
                    k: (
                        self._make_end(f"{pname}_{k}", True, None, trace, r.cta),
                        self._make_end(f"{pname}_{k}", False, None, trace, r.cta),
                    )
                    for k in st.schedule.fetch_channels()
                }
                args.append(_Fetch(st.schedule, ends, self._ws_problems[pname]))
            elif ann is Instance:
                args.append(r.index)
            elif ann is Turn:
                ts = self._ws_turns[pname]
                seat = ts.parties.index(r)
                args.append(
                    TurnEnd(ts, self._ws_bars.get(pname), seat, trace, Int32(int(seat == 0)))
                )
            else:  # _Context
                args.append(ctx)
        return args

    def _make_end(self, pname: str, is_prod: bool, bound, trace, cta: str = ALL, ri=None):
        chan = self._ws_chans[pname]
        side = "producer" if is_prod else "consumer"
        # Round-robin: instance j of k starts at epoch j and steps k epochs per operation.
        turns = chan.turns(side)
        turn = (
            [r for r, _ in (chan.producers if is_prod else chan.consumers)].index(ri)
            if (turns > 1)
            else 0
        )
        if chan.fixed:
            # Producers start each slot at phase 1 (a fresh empty barrier passes).
            state = Int32((1 << chan.n) - 1 if is_prod else 0)
        else:
            state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer
                if is_prod
                else pipeline.PipelineUserType.Consumer,
                chan.n,
            )
            for _ in range(turn):
                state.advance()
        cls = ProducerEnd if is_prod else ConsumerEnd
        end = cls(
            chan,
            self._ws_bars.get(pname),
            self._ws_buffers.get(pname, {}),
            trace,
            bound,
            state,
            cta,
            stride=turns,
        )
        if chan.named and not is_prod:
            if bound is not None:
                end.seed([bound])
            else:  # each stage's first acquire needs one consumer token, from its first turn
                end.seed([j for j in range(chan.n) if j % turns == turn])
        return end


class RoleContext(SimpleNamespace):
    """A namespace context as the roles see it; opaque to CuTeDSL control flow.

    The preprocessor treats ``ctx.x[i] = v`` inside a dynamic ``if``/``for`` as a write of
    ``ctx`` and would carry a plain namespace through the region, rebinding its attributes to
    region results that do not dominate other roles. Context values are defined before role
    dispatch and never rebound, so carrying them is unnecessary: this class carries nothing
    and reconstructs as itself. A frozen dataclass needs no wrapper: its fields cannot be
    rebound, and the preprocessor does not carry it (CuTeDSL 4.8.0: the FA4 2-CTA kernel's IR
    and the examples' SASS matched the wrapped form).
    """

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self


_ROLE_STATE_TYPES = (ProducerEnd, ConsumerEnd, WorkCursor, TurnEnd)


def _state_identity(state):
    if isinstance(state, TurnEnd):
        return id(state._trace), id(state._ts), state._seat
    return id(state._rec)


def _returned_state(role_name: str, result, state_args: list[tuple[str, object]]) -> dict:
    """Match explicit state results to inputs, independent of names and tuple order.

    Reconstructed endpoints/cursors share a record; turnstile copies share a role, turnstile
    and seat. This checks ownership, not freshness: the author must return loop-carried state
    rather than a snapshot saved before the loop.
    """
    values = () if result is None else result if isinstance(result, tuple) else (result,)
    owned = {_state_identity(arg) for _, arg in state_args}
    final = {}
    for value in values:
        if not isinstance(value, _ROLE_STATE_TYPES):
            raise ProtocolError(
                f"{role_name}: return channel endpoints, work cursors and turnstile seats "
                f"(a single value or tuple), not {type(value).__name__}"
            )
        key = _state_identity(value)
        if key not in owned:
            raise ProtocolError(f"{role_name}: returned state does not belong to this role")
        if key in final:
            raise ProtocolError(f"{role_name}: returned the same state argument more than once")
        final[key] = value
    missing = [name for name, arg in state_args if _state_identity(arg) not in final]
    if missing:
        names = ", ".join(name for name, _ in state_args)
        raise ProtocolError(
            f"{role_name}: return all warpdrive state arguments; missing {', '.join(missing)} "
            f"(for example, return {names})"
        )
    return final


def _check_turns(name: str, side: str, bound):
    roles = {r.role for r, _ in bound}
    widths = {r.num_threads for r, _ in bound}
    if len(roles) != 1 or len(widths) != 1:
        raise ProtocolError(
            f"channel {name}: round-robin {side}s {[r.name for r, _ in bound]} must be "
            "instances of one role with the same width (each epoch counts one instance)"
        )


def _check_named_turns(name: str, chan, turns: dict):
    """Round-robin instances of a named channel that share a stage barrier need a turnstile."""
    for side in ("producer", "consumer"):
        k = chan.turns(side)
        if k == 1 or chan.n % k == 0:
            continue
        bound = {r.name for r, _ in (chan.producers if side == "producer" else chan.consumers)}
        ts = turns.get(chan.turnstile)
        if ts is None or not ts.enabled or {r.name for r in ts.parties} != bound:
            raise ProtocolError(
                f"channel {name}: {k} round-robin {side}s share the barriers of {chan.n} named "
                f"stage(s) ({chan.n} % {k} != 0), so two of them could complete a barrier "
                f"without the other side; declare an enabled Turnstile over {sorted(bound)} and "
                "pass its name as turnstile=, or use a multiple of the instance count as stages"
            )


def _allocate_named(chans: dict) -> dict:
    """Assign named barrier IDs to named channels, from ``MAX_NAMED_BARRIER_ID`` down: per
    channel, ``n`` consecutive full IDs then ``n`` empty IDs (ring stages index them)."""
    out, top = {}, MAX_NAMED_BARRIER_ID
    for name, ch in chans.items():
        if not ch.named:
            continue
        base = top - 2 * ch.n + 1
        if base <= TMEM_TEARDOWN_BARRIER_ID:
            need = sum(2 * c.n for c in chans.values() if c.named)
            raise ProtocolError(
                f"named-barrier channels need {need} barrier IDs; "
                f"{MAX_NAMED_BARRIER_ID - TMEM_TEARDOWN_BARRIER_ID} are free (0 is "
                "__syncthreads, 1 and 2 belong to the layer); use mbarrier channels for some"
            )
        ch.named_ids = (list(range(base, base + ch.n)), list(range(base + ch.n, top + 1)))
        out[name] = ch.named_ids
        top = base - 1
    return out


def _barrier_unaligned(barrier_id: int, num_threads):
    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(), thread_count=Int32(num_threads).ir_value(), aligned=False
    )


def _runs(warps: list[int]) -> list[range]:
    """Each run of consecutive warps in a sorted list."""
    runs = []
    for w in warps:
        if runs and runs[-1].stop == w:
            runs[-1] = range(runs[-1].start, w + 1)
        else:
            runs.append(range(w, w + 1))
    return runs


def _in_warpgroups(warp_idx, groups: tuple[int, ...]):
    """Warpgroup-uniform predicate: does ``warp_idx`` fall in one of ``groups``?"""
    cond = None
    for run in _runs(list(groups)):
        term = _in_warps(warp_idx, run.start * WARPGROUP, len(run) * WARPGROUP)
        cond = term if cond is None else cond | term
    return cond


def _in_warps(warp_idx, first: int, count: int):
    if count == 1:
        return warp_idx == first
    return (warp_idx >= first) & (warp_idx < first + count)
