"""Role declarations and the ``WarpSpecialized`` kernel base."""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import cutlass
from cutlass import Int32, cute, memory, pipeline
from cutlass._mlir.dialects import nvvm

from transformer_nuggets.cute.warp_roles.channels import (
    Channel,
    ConsumerEnd,
    ProducerEnd,
    Threads,
    TurnEnd,
    Turnstile,
    Umma,
)
from transformer_nuggets.cute.warp_roles.cluster import (
    ALL,
    LEADER,
    PLACEMENTS,
    SingleCta,
    check_atoms,
    derive_scope,
)
from transformer_nuggets.cute.warp_roles.protocol import ProtocolError, ProtocolLog, RoleTrace
from transformer_nuggets.cute.warp_roles.scheduling import (
    TileCursor,
    Tiles,
    TileStream,
    _Fetch,
    scheduler_role,
)
from transformer_nuggets.cute.warp_roles.tmem import TmemPlan, TmemView

WARP_SIZE = 32
WARPGROUP = 4
REGS_PER_SM = 65536
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


_ANNOTATIONS = {
    "Producer": Producer,
    "Consumer": Consumer,
    "Tmem": Tmem,
    "Instance": Instance,
    "Turn": Turn,
    "Tiles": Tiles,
}


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

    @property
    def warps(self) -> range:
        return range(self.first_warp, self.first_warp + self.num_warps)

    @property
    def num_threads(self) -> int:
        return self.num_warps * WARP_SIZE


def role(warps, regs: int | None = None, cta: str = ALL):
    """Mark a method as a warp role. ``warps``: an int, a range, or a list of ranges.

    A list instantiates the method once per range; an ``Instance`` parameter receives the
    instance index. ``cta``: ``"all"`` (default) runs the role in every CTA of the kernel's
    cluster; ``"leader"`` runs it in the leader CTA only (the MMA issuer of a ``CtaPair``; a
    no-op placement for a single CTA). Put this decorator directly above ``@cute.jit``: the
    preprocessor only rewrites functions whose innermost decorator is ``<x>.jit``/``<x>.kernel``
    and then re-applies the outer decorators to the rewritten function.
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
        fn._warp_roles = (ranges, regs, cta)
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
        for name in dir(type(self)):
            fn = getattr(type(self), name)
            meta = getattr(fn, "_warp_roles", None)
            if meta is None:
                continue
            ranges, regs, cta = meta
            if not hasattr(fn, "__wrapped__"):
                raise TypeError(f"role {name}: put @role(...) directly above @cute.jit")
            params = []
            for pname, p in list(inspect.signature(fn).parameters.items())[1:]:
                ann = p.annotation
                if isinstance(ann, str):
                    ann = eval(ann, {**fn.__globals__, **_ANNOTATIONS})
                params.append((pname, ann))
            for idx, wr in enumerate(ranges):
                inst = f"{name}[{idx}]" if len(ranges) > 1 else name
                ri = RoleInstance(inst, name, idx, wr.start, len(wr), regs, fn, params, cta=cta)
                ri.uses_tmem = any(a is Tmem for _, a in params)
                roles.append(ri)
                self._bind(ri, chans, turns)
        for name, st in streams.items():
            self._bind_stream(name, st, roles, chans, cluster)
        roles.sort(key=lambda r: r.first_warp)
        self._validate(chans, turns, roles)
        self._validate_placement(chans, turns, roles)
        self._ws_chans, self._ws_turns, self._ws_roles = chans, turns, roles
        self._ws_streams = streams
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
            idle = sorted({r.role for r in roles} - {r.role for r in users})
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
            ch.scope = derive_scope(ch, cluster, internal=getattr(ch, "internal", False))

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
                if mech is not Threads and len(bound) > 1:
                    raise ProtocolError(
                        f"channel {name}: {side} side signals once ({mech}) but has several "
                        f"roles {[r.name for r, _ in bound]}"
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
        self.num_warps_used = num_warps
        uses_regs = any(r.regs is not None for r in roles)
        # setmaxnreg is .aligned per warpgroup: pad to whole warpgroups and require every warp
        # of a warpgroup to request the same budget.
        self.num_warps = -(-num_warps // WARPGROUP) * WARPGROUP if uses_regs else num_warps
        self.default_regs = min(REGS_PER_SM // (self.num_warps * WARP_SIZE) // 8 * 8, 256)
        self.pad_regs = None
        if uses_regs:
            for g in range(self.num_warps // WARPGROUP):
                budgets = {
                    owner.get(w, f"pad{w}"): next((r.regs for r in roles if w in r.warps), "pad")
                    for w in range(g * WARPGROUP, (g + 1) * WARPGROUP)
                }
                vals = {v for v in budgets.values() if v != "pad"}
                if len(vals) > 1:
                    raise ValueError(f"warpgroup {g} has mixed register budgets {budgets}")
                if "pad" in budgets.values():
                    self.pad_regs = min(vals) if vals else 24
            total = sum(
                (r.regs if r.regs is not None else self.default_regs) * r.num_threads
                for r in roles
            ) + (self.num_warps - num_warps) * WARP_SIZE * (self.pad_regs or 0)
            if total > REGS_PER_SM:
                raise ValueError(f"register budgets need {total} > {REGS_PER_SM}")

    @property
    def num_threads(self) -> int:
        self._collect()
        return self.num_warps * WARP_SIZE

    def describe(self) -> str:
        """Static role/channel summary, plus the recorded protocol when traced."""
        self._collect()
        pair = self.cluster.size > 1
        lines = [f"{self.num_warps} warps ({self.num_threads} threads)"]
        if pair:
            m, n, k = self.cluster.shape
            v = self.cluster.size
            lines[0] += (
                f"; {self.cluster.describe()}; cta_layout_vmnk shape (({v}),{m // v},{n},{k})"
            )
        for r in self._ws_roles:
            where = " cta=leader" if pair and r.cta == LEADER else ""
            lines.append(
                f"  role {r.name:<11} warps {r.warps.start:>2}-{r.warps.stop - 1:<2} "
                f"regs={r.regs}{where}"
            )
        for name, ch in self._ws_chans.items():
            kind = f"Slots({ch.n})" if ch.fixed else f"Ring({ch.n})"
            prods = ", ".join(r.name + ("" if s is None else f"@{s}") for r, s in ch.producers)
            cons = ", ".join(r.name + ("" if s is None else f"@{s}") for r, s in ch.consumers)
            sc = ch.scope
            if sc.pair:
                full = ", ".join(
                    f"{'leader' if c == 0 else 'peer'} {sc.full_counts[c]}" for c in sc.full_ctas
                )
                empty = ", ".join(
                    f"{'leader' if c == 0 else 'peer'} {sc.empty_counts[c]}" for c in sc.empty_ctas
                )
                extra = f" tx={sc.tx}" if sc.tx is not None else ""
                if ch.producer is Umma:
                    extra += f" commit_mask={sc.full_mask:#04b}"
                if ch.consumer is Umma:
                    extra += f" release_mask={sc.empty_mask:#04b}"
                if Threads in (ch.producer, ch.consumer):
                    extra += f" remote_scope={ch.remote_scope}"
                extra += " tail" if sc.tail else ""
                lines.append(
                    f"  chan {name:<6} {kind:<8} {prods} -> {cons}; pair-scoped: full {full}; "
                    f"empty {empty}{extra}"
                )
                continue
            lines.append(
                f"  chan {name:<6} {kind:<8} {prods} -> {cons}; full arrivals "
                f"{ch.slot_counts('producer')}, empty arrivals {ch.slot_counts('consumer')}"
            )
        for name, ts in self._ws_turns.items():
            seats = " -> ".join(r.name for r in ts.parties)
            lines.append(f"  turnstile {name}: {seats} (enabled={ts.enabled})")
        for name, st in self._ws_streams.items():
            sched = next((r for r in self._ws_roles if r.fn is scheduler_role), None)
            dist = (
                f"scheduler warp {sched.first_warp} broadcasts over {name}_work"
                if sched
                else "decoded locally in every role"
            )
            lines.append(f"  stream {name}: {st.describe()}; {dist}")
        if self.tmem_plan is not None:
            plan = self.tmem_plan
            lines.append(f"  tmem: {plan.used_cols} columns used, {plan.alloc_cols} allocated")
            lines.append(plan.describe())
        if self._ws_log is not None and self._ws_log.ops:
            lines.append(self._ws_log.describe())
        return "\n".join(lines)

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
        return plan, off

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
                    if cutlass.const_expr(leader[i] == peer[i]):
                        cute.arch.mbarrier_init(bars + i, leader[i])
                    else:  # pair: rank 0 is the leader, rank 1 the peer
                        cute.arch.mbarrier_init(bars + i, leader[i] + (peer[i] - leader[i]) * rank)

    @cute.jit
    def run_roles(self, ctx):
        """CTA sync (barrier init + TMEM address), role dispatch, TMEM teardown."""
        ctx = ctx if isinstance(ctx, RoleContext) else RoleContext(**vars(ctx))
        check_atoms(vars(ctx), self.cluster)
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
        for i in cutlass.range_constexpr(len(roles)):
            r = roles[i]
            # Only a call on `self` inside the dynamic `if`: an assignment through `self`
            # would make the preprocessor carry `self` as an if-result.
            if cutlass.const_expr(pair and r.cta == LEADER):
                if _in_warps(warp_idx, r.first_warp, r.num_warps) & (rank == 0):
                    self._dispatch(r, ctx, tmem_threads)
                # setmaxnreg is per warpgroup: the peer's idle warps take the same budget.
                if _in_warps(warp_idx, r.first_warp, r.num_warps) & (rank != 0):
                    self._set_regs(r.regs)
            else:
                if _in_warps(warp_idx, r.first_warp, r.num_warps):
                    self._dispatch(r, ctx, tmem_threads)
        if cutlass.const_expr(self.pad_regs is not None):  # noqa: SIM102 (runtime inside const)
            if warp_idx >= self.num_warps_used:
                cute.arch.setmaxregister_decrease(self.pad_regs)
        if cutlass.const_expr(pair):
            # Remote arrivals into a CTA must land before it exits: every thread-issued remote
            # arrive precedes its thread's cluster arrive (release), and producers whose stages
            # are released by the other CTA's tcgen05.commit drained them (tails) above.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    def _dispatch(self, r: RoleInstance, ctx, tmem_threads: tuple[int, int]):
        # Plain Python (no dynamic control flow): runs at trace time inside the role's `if`.
        self._set_regs(r.regs)
        trace = RoleTrace(r.name, self._ws_log)
        args = self._role_args(r, ctx, trace)
        if trace.tails:
            final = _call_capturing_locals(r.fn, r.role, (self, *args))
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
        trace.finish()
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
                args.append(TmemView(base, self.tmem_plan, r.name, self._ws_log))
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
            else:
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
    """Kernel-prologue values shared by all roles; opaque to CuTeDSL control flow.

    The preprocessor treats ``c.x[i] = v`` inside a dynamic ``if``/``for`` as a write of
    ``c`` and would carry a plain namespace through the region, rebinding its attributes to
    region results that do not dominate other roles. Context values are defined before role
    dispatch and never rebound, so carrying them is unnecessary: this class carries nothing
    and reconstructs as itself.
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
        if event == "return" and frame.f_code.co_name == name and "self" in frame.f_locals:
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


def _in_warps(warp_idx, first: int, count: int):
    if count == 1:
        return warp_idx == first
    return (warp_idx >= first) & (warp_idx < first + count)
