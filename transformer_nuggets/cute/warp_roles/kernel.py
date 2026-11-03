"""Role declarations and the ``WarpSpecialized`` kernel base."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import cutlass
from cutlass import Int32, cute, memory, pipeline

from transformer_nuggets.cute.warp_roles.channels import (
    Channel,
    ConsumerEnd,
    ProducerEnd,
    Threads,
    TurnEnd,
    Turnstile,
)
from transformer_nuggets.cute.warp_roles.protocol import ProtocolError, ProtocolLog, RoleTrace
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


_ANNOTATIONS = {
    "Producer": Producer,
    "Consumer": Consumer,
    "Tmem": Tmem,
    "Instance": Instance,
    "Turn": Turn,
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

    @property
    def warps(self) -> range:
        return range(self.first_warp, self.first_warp + self.num_warps)

    @property
    def num_threads(self) -> int:
        return self.num_warps * WARP_SIZE


def role(warps, regs: int | None = None):
    """Mark a method as a warp role. ``warps``: an int, a range, or a list of ranges.

    A list instantiates the method once per range; an ``Instance`` parameter receives the
    instance index. Put this decorator directly above ``@cute.jit``: the preprocessor only
    rewrites functions whose innermost decorator is ``<x>.jit``/``<x>.kernel`` and then
    re-applies the outer decorators to the rewritten function.
    """
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
        fn._warp_roles = (ranges, regs)
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

    # ---- host-side analysis ------------------------------------------------ #

    def _collect(self):
        if getattr(self, "_ws_roles", None) is not None:
            return
        chans = {k: v for k, v in vars(self).items() if isinstance(v, Channel)}
        turns = {k: v for k, v in vars(self).items() if isinstance(v, Turnstile)}
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
            ranges, regs = meta
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
                ri = RoleInstance(inst, name, idx, wr.start, len(wr), regs, fn, params)
                ri.uses_tmem = any(a is Tmem for _, a in params)
                roles.append(ri)
                self._bind(ri, chans, turns)
        roles.sort(key=lambda r: r.first_warp)
        self._validate(chans, turns, roles)
        self._ws_chans, self._ws_turns, self._ws_roles = chans, turns, roles
        self._ws_log = ProtocolLog() if self.trace_protocol else None

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
        lines = [f"{self.num_warps} warps ({self.num_threads} threads)"]
        for r in self._ws_roles:
            lines.append(
                f"  role {r.name:<11} warps {r.warps.start:>2}-{r.warps.stop - 1:<2} regs={r.regs}"
            )
        for name, ch in self._ws_chans.items():
            kind = f"Slots({ch.n})" if ch.fixed else f"Ring({ch.n})"
            prods = ", ".join(r.name + ("" if s is None else f"@{s}") for r, s in ch.producers)
            cons = ", ".join(r.name + ("" if s is None else f"@{s}") for r, s in ch.consumers)
            lines.append(
                f"  chan {name:<6} {kind:<8} {prods} -> {cons}; full arrivals "
                f"{ch.slot_counts('producer')}, empty arrivals {ch.slot_counts('consumer')}"
            )
        for name, ts in self._ws_turns.items():
            seats = " -> ".join(r.name for r in ts.parties)
            lines.append(f"  turnstile {name}: {seats} (enabled={ts.enabled})")
        if self.tmem_plan is not None:
            plan = self.tmem_plan
            lines.append(f"  tmem: {plan.used_cols} columns used, {plan.alloc_cols} allocated")
            lines.append(plan.describe())
        if self._ws_log is not None and self._ws_log.ops:
            lines.append(self._ws_log.describe())
        return "\n".join(lines)

    def _barrier_plan(self):
        """(owner, offset, arrival counts) for every mbarrier, in allocation order."""
        plan, off = [], 0
        for name, ch in self._ws_chans.items():
            plan.append((name, off, ch.slot_counts("producer") + ch.slot_counts("consumer")))
            off += 2 * ch.n
        for name, ts in self._ws_turns.items():
            if ts.enabled:
                n = len(ts.parties)
                plan.append((name, off, [ts.parties[(p - 1) % n].num_threads for p in range(n)]))
                off += n
        return plan, off

    # ---- device-side ------------------------------------------------------- #

    def begin_roles(self):
        """Allocate and initialize barriers and payload buffers; allocate TMEM.

        Plain Python at trace time (the loops are over host-side declarations); only the
        barrier initialization is a device-side branch.
        """
        self._collect()
        smem = memory.SmemAllocator()
        plan, total = self._barrier_plan()
        bars = smem.allocate_array(cutlass.Int64, max(total, 1))
        self._ws_bars = {name: bars + off for name, off, _ in plan}
        self._ws_bar_counts = [c for _, _, counts in plan for c in counts]
        self._init_barriers(bars)
        self._ws_payload = {}
        for name, ch in self._ws_chans.items():
            if ch.payload is not None:
                dtype, shape = ch.payload
                self._ws_payload[name] = smem.allocate_tensor(
                    dtype, cute.make_layout((*shape, ch.n)), 16
                )
        self._ws_tmem = None
        tmem_roles = [r for r in self._ws_roles if r.uses_tmem]
        if tmem_roles:
            holder = smem.allocate_array(cutlass.Int32, 1)
            tmem = memory.TmemAllocator(
                holder,
                barrier_for_retrieve=pipeline.NamedBarrier(
                    barrier_id=TMEM_PTR_BARRIER_ID, num_threads=self.num_threads
                ),
                allocator_warp_id=tmem_roles[0].first_warp,
            )
            # Only the allocator warp executes tcgen05.alloc (checked inside).
            tmem.allocate(self.tmem_plan.alloc_cols)
            tmem.relinquish_alloc_permit()
            self._ws_tmem = tmem
        return smem

    @cute.jit
    def _init_barriers(self, bars):
        counts = self._ws_bar_counts
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(len(counts)):
                    cute.arch.mbarrier_init(bars + i, counts[i])

    @cute.jit
    def run_roles(self, ctx):
        """CTA sync (barrier init + TMEM address), role dispatch, TMEM teardown."""
        # Fences mbarrier.init and syncs the CTA; the same bar.sync publishes the TMEM base
        # address written by tcgen05.alloc in begin_roles().
        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        ctx = ctx if isinstance(ctx, RoleContext) else RoleContext(**vars(ctx))
        roles = self._ws_roles
        tmem_threads = sum(r.num_threads for r in roles if r.uses_tmem)
        for i in cutlass.range_constexpr(len(roles)):
            r = roles[i]
            # Only a call on `self` inside the dynamic `if`: an assignment through `self`
            # would make the preprocessor carry `self` as an if-result.
            if _in_warps(warp_idx, r.first_warp, r.num_warps):
                self._dispatch(r, ctx, tmem_threads)
        if cutlass.const_expr(self.pad_regs is not None):  # noqa: SIM102 (runtime inside const)
            if warp_idx >= self.num_warps_used:
                cute.arch.setmaxregister_decrease(self.pad_regs)

    def _dispatch(self, r: RoleInstance, ctx, tmem_threads: int):
        # Plain Python (no dynamic control flow): runs at trace time inside the role's `if`.
        self._set_regs(r.regs)
        trace = RoleTrace(r.name, self._ws_log)
        r.fn(self, *self._role_args(r, ctx, trace))
        trace.finish()
        if r.uses_tmem:
            # Every TMEM user finishes before the allocator warp frees it.
            cute.arch.barrier(barrier_id=TMEM_TEARDOWN_BARRIER_ID, number_of_threads=tmem_threads)
            if r is next(x for x in self._ws_roles if x.uses_tmem):  # allocator role
                ptr = self._ws_tmem.retrieve_ptr(cutlass.Float32)
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
                chan = self._ws_chans[pname]
                is_prod = side is Producer
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
                bound = r.index if isinstance(ann, _Bound) else None
                args.append(
                    cls(
                        chan,
                        self._ws_bars[pname],
                        self._ws_payload.get(pname),
                        trace,
                        bound,
                        state,
                    )
                )
            elif ann is Tmem:
                base = self._ws_tmem.retrieve_ptr(cutlass.Float32)
                args.append(TmemView(base, self.tmem_plan, r.name, self._ws_log))
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


def _in_warps(warp_idx, first: int, count: int):
    if count == 1:
        return warp_idx == first
    return (warp_idx >= first) & (warp_idx < first + count)
