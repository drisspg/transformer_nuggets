"""Role declarations and the ``WarpSpecialized`` kernel base."""

from __future__ import annotations

import inspect
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace

import cutlass
from cutlass import Int32, cute, memory, pipeline
from cutlass._mlir.dialects import nvvm

from transformer_nuggets.cute.warp_roles.channels import (
    Barriers,
    Channel,
    ConsumerEnd,
    ProducerEnd,
    Threads,
    Tma,
    TurnEnd,
    Turnstile,
    Umma,
)
from transformer_nuggets.cute.warp_roles.cluster import (
    ALL,
    LEADER,
    PLACEMENTS,
    SingleCta,
    apply_overrides,
    check_atoms,
    derive_scope,
)
from transformer_nuggets.cute.warp_roles.describe import describe_kernel
from transformer_nuggets.cute.warp_roles.protocol import (
    ProtocolError,
    ProtocolLog,
    RoleTrace,
    take_opt_outs,
)
from transformer_nuggets.cute.warp_roles.scheduling import (
    TileCursor,
    Tiles,
    TileStream,
    _Fetch,
    scheduler_role,
)
from transformer_nuggets.cute.warp_roles.tma import TmaReport
from transformer_nuggets.cute.warp_roles.tma import check_role as check_tma
from transformer_nuggets.cute.warp_roles.tmem import TmemPlan, TmemView
from transformer_nuggets.cute.warp_roles.tracing import pop, pop_if, push

WARP_SIZE = 32
WARPGROUP = 4
REGS_PER_SM = 65536
MIN_REGS = 24  # setmaxnreg floor: the budget of a warpgroup no role owns
# Named barrier ids owned by the layer; user code must not reuse them.
TMEM_PTR_BARRIER_ID = 1
TMEM_TEARDOWN_BARRIER_ID = 2


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


class _SchedulerTiles:
    """Internal annotation: the scheduler warp's own cursor (it decodes and publishes)."""


class _SchedulerFetch:
    """Internal annotation: the scheduler warp's private policy channels (both ends)."""


class _Context:
    """Internal annotation: the parameter receives the context passed to ``run_roles``."""


_ANNOTATIONS = {
    "Producer": Producer,
    "Consumer": Consumer,
    "Tmem": Tmem,
    "Instance": Instance,
    "Turn": Turn,
    "Tiles": Tiles,
    "Barriers": Barriers,
}
# Parameters a raw role may take: no channel ends, cursors, or seats (they carry protocol state).
_RAW_ANNOTATIONS = (Instance, Tmem, Barriers, _Context)
# Layer classes a role might write in place of an annotation, and the annotation meant.
_LOOKALIKES = (
    (ProducerEnd, "Producer"),
    (ConsumerEnd, "Consumer"),
    (Channel, "Producer or Consumer"),
    (TileStream, "Tiles"),
    (TileCursor, "Tiles"),
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

    @property
    def warps(self) -> range:
        return range(self.first_warp, self.first_warp + self.num_warps)

    @property
    def num_threads(self) -> int:
        return self.num_warps * WARP_SIZE


def role(warps, regs: int | None = None, cta: str = ALL, raw: bool = False):
    """Mark a method as a warp role. ``warps``: an int, a range, or a list of ranges.

    A list instantiates the method once per range; an ``Instance`` parameter receives the
    instance index. ``cta``: ``"all"`` (default) runs the role in every CTA of the kernel's
    cluster; ``"leader"`` runs it in the leader CTA only (the MMA issuer of a ``CtaPair``; a
    no-op placement for a single CTA). ``raw=True`` reserves the warps and register budget for
    a user function the layer only dispatches: it takes ``Instance``, ``Tmem``, ``Barriers``,
    and the context, but no channel ends, cursors, or seats, gets no protocol log or trace
    checks, and is exempt from the "every role loops over the TileStream" rule. It still counts
    in warpgroup padding, register budgets, and TMEM teardown. Put this decorator directly above
    ``@cute.jit``: the preprocessor only rewrites functions whose innermost decorator is
    ``<x>.jit``/``<x>.kernel`` and then re-applies the outer decorators to the rewritten
    function.
    """
    if cta not in PLACEMENTS:
        raise ValueError(f"role cta must be one of {PLACEMENTS}, got {cta!r}")
    if isinstance(warps, int):
        ranges = [range(warps, warps + 1)]
    elif isinstance(warps, range):
        ranges = [warps]
    else:
        ranges = list(warps)
    for r in ranges:
        if not isinstance(r, range) or r.step != 1 or len(r) == 0:
            raise ValueError(f"role warps must be non-empty unit-step ranges, got {warps!r}")

    def deco(fn):
        fn._warp_roles = (ranges, regs, cta, raw)
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
    trace_protocol: bool = False
    iket: bool = False  # emit IKET ranges named after roles and channels (tracing.py)
    iket_waits: bool = True  # with iket: also a range around every blocking channel wait
    cluster: SingleCta = SingleCta()  # or CtaPair(): one line switches to 2-CTA MMAs

    def __setattr__(self, name, value):
        if isinstance(value, TileStream):
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
        streams = {k: v for k, v in vars(self).items() if isinstance(v, TileStream)}
        if len(streams) > 1:
            raise ProtocolError(
                f"one TileStream per kernel (every role starts at blockIdx.x), got {list(streams)}"
            )
        cluster = self.cluster
        for name, st in streams.items():
            st.name = name
            st.owner = self
            if st.policy.broadcast:
                chans[f"{name}_work"] = st.channel()
                for k, ch in st.policy.fetch_channels().items():
                    chans[f"{name}_{k}"] = ch
        for name, obj in (*chans.items(), *turns.items()):
            obj.name = name
            if isinstance(obj, Channel):
                obj.producers.clear()
                obj.consumers.clear()
            else:
                obj.parties.clear()
        roles = []
        ctx_types = {}  # annotated context type -> first "role.param" using it
        for name in dir(type(self)):
            fn = getattr(type(self), name)
            meta = getattr(fn, "_warp_roles", None)
            if meta is None:
                continue
            ranges, regs, cta, raw = meta
            if not hasattr(fn, "__wrapped__"):
                raise TypeError(f"role {name}: put @role(...) directly above @cute.jit")
            params = []
            for pname, p in list(inspect.signature(fn).parameters.items())[1:]:
                ann, ctx_type = _resolve_param(name, pname, p.annotation, fn)
                if raw and ann not in _RAW_ANNOTATIONS:
                    raise ProtocolError(
                        f"raw role {name}: parameter {pname} is {getattr(ann, '__name__', ann)}; "
                        "raw roles take Instance, Tmem, Barriers, and the context only (drive "
                        "edges through Barriers, or make the role managed)"
                    )
                if ann is Barriers and pname not in raw_bars:
                    raise ProtocolError(f"role {name}: no Barriers named {pname!r}")
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
                    inst, name, idx, wr.start, len(wr), regs, fn, params, cta=cta, raw=raw
                )
                ri.uses_tmem = any(a is Tmem for _, a in params)
                roles.append(ri)
                self._bind(ri, chans, turns)
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
        self._ws_chans, self._ws_turns, self._ws_roles = chans, turns, roles
        self._ws_raw_bars = raw_bars
        self._ws_streams = streams
        self._ws_tma = {}  # channel name -> tma.TmaReport, filled while tracing
        self._ws_opt_outs = []
        self._ws_log = ProtocolLog() if self.trace_protocol else None

    @staticmethod
    def _bind_stream(name: str, st: TileStream, roles: list, chans: dict, cluster):
        """Check stream users; for broadcast policies add the scheduler role and bind every
        stream user (and the scheduler) as a consumer of the work ring."""
        users = [r for r in roles if any(a is Tiles for _, a in r.params)]
        st.roles = [r.name for r in sorted(users, key=lambda r: r.first_warp)]
        for r in users:
            for pname, ann in r.params:
                if ann is Tiles and pname != name:
                    raise ProtocolError(f"role {r.name}: no TileStream named {pname!r}")
        if st.policy.persistent:
            idle = sorted({r.role for r in roles if not r.raw} - {r.role for r in users})
            if idle:
                raise ProtocolError(
                    f"{st.policy.describe()} runs several work items per CTA, but roles {idle} "
                    f"do not take `{name}: Tiles`; every role must loop over the stream (a role "
                    f"with no per-item work drains it: `while {name}.valid: {name}.advance()`)"
                )
        if not st.policy.broadcast:
            return
        owned = {w for r in roles for w in r.warps}
        warp = st.scheduler_warp
        if warp is None:
            warp = next(w for w in range(max(owned, default=-1) + 2) if w not in owned)
        elif warp in owned:
            raise ValueError(f"TileStream {name}: scheduler_warp {warp} is claimed by a role")
        group = range(warp // WARPGROUP * WARPGROUP, (warp // WARPGROUP + 1) * WARPGROUP)
        budgets = {r.regs for r in roles if r.regs is not None and set(r.warps) & set(group)}
        any_regs = [r.regs for r in roles if r.regs is not None]
        regs = budgets.pop() if len(budgets) == 1 else (min(any_regs) if any_regs else None)
        work = f"{name}_work"
        fetch = [f"{name}_{k}" for k in st.policy.fetch_channels()]
        params = [(work, Producer), (name, _SchedulerTiles), (name, _SchedulerFetch)]
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
        chans[work].producers.append((sched, None))
        for r in [*users, sched]:
            chans[work].consumers.append((r, None))
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
        for ch in chans.values():
            ch.cluster = cluster
            ch.derived_scope = derive_scope(ch, cluster, internal=ch.internal)
            ch.scope = apply_overrides(ch, ch.derived_scope, cluster)

    def _validate(self, chans, turns, roles):
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
                if mech is Threads:
                    continue
                how = getattr(mech, "__name__", repr(mech))  # Umma, or Tma(tx_bytes=...)
                if len(bound) > 1:
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
        for name, ts in turns.items():
            if len(ts.parties) < 2 or len({r.role for r in ts.parties}) != 1:
                raise ProtocolError(f"turnstile {name} needs >= 2 instances of one role")
            ts.parties.sort(key=lambda r: r.index)
        if self.tmem_plan is not None:
            _ = self.tmem_plan.alloc_cols  # raises when over capacity
        elif any(r.uses_tmem for r in roles):
            raise ProtocolError("roles take Tmem but the kernel has no tmem_plan")
        num_warps = max(owner) + 1
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

    def enable_iket(self, enabled: bool = True, *, waits: bool = True):
        """Emit IKET ranges in the next compile (``tracing.py`` lists the names). Off: no IR.
        ``waits=False`` keeps only the role/work-item ranges (fine-grained wait ranges cost
        ~35% on FA4 under ``run-iket``; role ranges alone cost about nothing). Include
        ``iket_key()`` in the compile-cache key (the examples' ``get_name`` does)."""
        self.iket, self.iket_waits = enabled, waits
        return self

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

    def barrier_ptr(self, name: str):
        """Pointer to the first mbarrier of a channel (full barriers at ``[0, n)``, empty at
        ``[n, 2n)``), an enabled turnstile, or a ``Barriers`` declaration. Valid after
        ``begin_roles()``; every CTA of a cluster has them at the same offsets."""
        if name not in self._ws_bars:
            raise ProtocolError(f"no barriers named {name!r} (have {sorted(self._ws_bars)})")
        return self._ws_bars[name]

    def launch_config(self, work=None, grid=None) -> dict:
        """``grid``/``block``/``cluster`` for ``kernel(...).launch(**cfg, stream=stream)``."""
        if work is not None:
            grid = work.grid()
        return {
            "grid": grid,
            "block": (self.num_threads, 1, 1),
            "cluster": self.cluster.launch_cluster(),
        }

    # ---- device-side ------------------------------------------------------- #

    def begin_roles(self, **problems):
        """Allocate and initialize barriers and payload buffers; allocate TMEM.

        ``problems``: the runtime ``TileProblem`` of each declared ``TileStream``, by name
        (``self.begin_roles(tiles=work)``).

        Plain Python at trace time (the loops are over host-side declarations); only the
        barrier initialization is a device-side branch.
        """
        self._collect()
        if set(problems) != set(self._ws_streams):
            raise ProtocolError(
                f"begin_roles needs a problem for each TileStream {sorted(self._ws_streams)}, "
                f"got {sorted(problems)}"
            )
        self._ws_problems = problems
        for problem in problems.values():
            problem.start()  # per-CTA start values, computed once ahead of role dispatch
        self._ws_tma = {}
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
        self._ws_payload = {}
        for name, ch in self._ws_chans.items():
            if ch.payload is not None:
                dtype, shape = ch.payload
                self._ws_payload[name] = smem.allocate_tensor(
                    dtype, cute.make_layout((*shape, ch.n)), 16
                )
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
        return smem

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
    def run_roles(self, ctx):
        """CTA sync (barrier init + TMEM address), role dispatch, TMEM teardown."""
        ctx = self._role_context(ctx)
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
                self._dispatch_group(groups[gi], ctx, tmem_threads, warp_idx, rank)
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
        """Collect check opt-outs; warn once about TMA channels the check could not verify."""
        self._ws_opt_outs = take_opt_outs()
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
                f"warp_roles: TMA check could not verify channel(s) {unverified} (no error "
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
        # A role looping over a TileStream gets one IKET range per work item (TileCursor);
        # any other role gets one over its whole body.
        cursors = [
            "tiles" if a is _SchedulerTiles else p  # scheduler_role's parameter name
            for p, a in r.params
            if a in (Tiles, _SchedulerTiles)
        ]
        whole_range = self.iket and not cursors
        if whole_range:
            push(r.name)
        if r.raw:
            r.fn(self, *args)
        elif trace.tails or (self.iket and cursors):
            final = _call_capturing_locals(r.fn, r.fn.__name__, (self, *args))
            for pname in cursors if self.iket else ():
                cur = final.get(pname)
                if not isinstance(cur, TileCursor):
                    raise ProtocolError(
                        f"{r.name}: IKET closes the last work item's range at role exit using "
                        f"the cursor bound to `{pname}`, but `{pname}` is then "
                        f"{type(cur).__name__}; keep the cursor bound to its parameter name"
                    )
                pop_if(cur.index < cur._problem.total)
            for pname, rec in trace.tails:
                end = final.get(pname)
                if not isinstance(end, ProducerEnd) or end._rec is not rec:
                    raise ProtocolError(
                        f"{r.name}: the layer drains channel {pname} at role exit using the end "
                        f"bound to `{pname}` when the role returns, but `{pname}` is then "
                        f"{type(end).__name__}; keep the end bound to its parameter name "
                        f"(reassign it from helpers: `{pname} = helper({pname}, ...)`)"
                    )
                end.tail()
        else:
            r.fn(self, *args)
        if whole_range:
            pop()
        if not r.raw:
            check_tma(trace, self.cluster, self._ws_tma)
            trace.finish()
        for chan, n in trace.uses_per_item().items():
            self._ws_uses[(r.name, chan)] = n
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
                args.append(self._make_end(pname, side is Producer, bound, trace, r.cta))
            elif ann is Tmem:
                base = self._ws_tmem.retrieve_ptr(cutlass.Float32)
                args.append(TmemView(base, self.tmem_plan, r.name, trace.log))
            elif ann is Barriers:
                args.append(self._ws_bars[pname])
            elif ann is Tiles or ann is _SchedulerTiles:
                st = self._ws_streams[pname]
                end = None
                if st.policy.broadcast:
                    end = self._make_end(f"{pname}_work", False, None, trace, r.cta)
                args.append(
                    TileCursor.create(
                        st,
                        self._ws_problems[pname],
                        end,
                        trace,
                        scheduler=ann is _SchedulerTiles,
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
                    for k in st.policy.fetch_channels()
                }
                args.append(_Fetch(st.policy, ends, self._ws_problems[pname]))
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

    def _make_end(self, pname: str, is_prod: bool, bound, trace, cta: str = ALL):
        chan = self._ws_chans[pname]
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
        cls = ProducerEnd if is_prod else ConsumerEnd
        end = cls(
            chan, self._ws_bars[pname], self._ws_payload.get(pname), trace, bound, state, cta
        )
        if is_prod and chan.scope.tail:
            trace.tails.append((pname, end._rec))
        return end


class RoleContext(SimpleNamespace):
    """A namespace context as the roles see it; opaque to CuTeDSL control flow.

    The preprocessor treats ``ctx.x[i] = v`` inside a dynamic ``if``/``for`` as a write of
    ``c`` and would carry a plain namespace through the region, rebinding its attributes to
    region results that do not dominate other roles. Context values are defined before role
    dispatch and never rebound, so carrying them is unnecessary: this class carries nothing
    and reconstructs as itself. A frozen dataclass needs no wrapper: its fields cannot be
    rebound, and the preprocessor does not carry it (the examples' IR and SASS are unchanged).
    """

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self


def _call_capturing_locals(fn, name: str, args) -> dict:
    """Call a role at trace time and return its frame's locals at return: the ends bound to the
    parameter names then hold the role's final phases (loops rebind names to their results)."""
    captured = {}
    previous = sys.getprofile()

    def profile(frame, event, arg):
        # The outermost frame of the (preprocessor-rewritten) role returns last.
        code = frame.f_code
        if (
            event == "return"
            and code.co_name == name
            and code.co_argcount
            and frame.f_locals.get(code.co_varnames[0]) is args[0]  # the role's `self`
        ):
            captured.clear()
            captured.update(frame.f_locals)
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        fn(*args)
    finally:
        sys.setprofile(previous)
    return captured


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
