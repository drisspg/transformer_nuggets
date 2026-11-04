"""Alias-hazard diagnostic: a bug finder for declared SMEM/TMEM overlaps. Never a prover.

Scope (phase 1): single-CTA kernels, declared overlaps only (``TmemPlan.alias`` regions and
``SmemPlan.buffer(..., overlaps=...)`` buffers), and only accesses the author annotates:
``tmem.view``/``tmem.access`` and ``handle.view``/``handle.access`` with ``read=``/``write=`` and
an agent (``by="threads"``, ``"tma"``, ``"umma"``). Footprints are whole regions or buffer stages,
narrowed by optional TMEM ``lanes``/``cols`` per warp. Ordering comes from the traced managed
channel operations and the recording wrappers below; raw ``cute.arch`` ordering instructions stay
legal but are invisible, so what they would order is reported ``unverified``.

For every pair of conflicting annotated accesses (overlapping footprints, at least one write,
through two different declared names) the diagnostic looks for an ordering edge in both
directions and reports, per alias:

* ``potential hazard``: a conflicting pair and the edge that is missing, with source lines;
* ``unverified``: an access it cannot see (raw acquisition, raw signal, raw ordering
  instruction, a CTA pair), or an edge only a raw instruction could supply;
* ``no hazard found``: no finding among the annotated accesses, which is not a proof.

Two orderings are modeled. Accesses inside handles of the same channel (and slot) are ordered by
the channel's whole-region handoff, provided each access completes before its handle's close
signals (``RULES``). Other pairs are checked on a happens-before graph over the traced roles,
with loops unrolled twice and two work items under a persistent schedule (one otherwise), linked
by epoch-matched channel edges. Every rule lives in ``RULES`` with its PTX ISA citation.

Default ``alias_check = "warn"``: the report is in ``describe()`` and one warning names the
findings. ``"strict"`` raises ``ProtocolError`` on a potential hazard; ``"off"`` skips the
analysis. ``with unchecked("alias", reason=...)`` leaves the accesses annotated inside the block
out. The diagnostic is host-only: annotations and wrappers emit no extra device code.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

from cutlass import cute

from transformer_nuggets.cute.warpdrive.protocol import ProtocolError, current_trace

MODES = ("warn", "strict", "off")
PTX = "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"


@dataclass(frozen=True)
class Rule:
    statement: str
    citation: str


# Every rule the diagnostic applies, with its source. Update here when the PTX ISA changes.
RULES = {
    "pipelined": Rule(
        "Same-warp tcgen05 pairs execute in issue order only when pipelined: mma A read -> mma D "
        "write (same cta_group), mma D write -> mma D write (same accumulator), st -> ld, "
        "st -> st, ld -> st. mma D write -> mma A read is not pipelined; different warps never "
        "are.",
        f"PTX ISA 9.5 9.7.18.6.2 Pipelined tcgen05 Instructions "
        f"({PTX}#tcgen05-memory-consistency-model-pipelined-instructions)",
    ),
    "commit": Rule(
        "tcgen05.commit tracks the earlier tcgen05.mma/cp/shift of the issuing thread with the "
        "same cta_group, not tcgen05.ld/st, TMA or other threads' operations.",
        f"PTX ISA 9.5 9.7.18.12.1 tcgen05.commit ({PTX}#tcgen-async-sync-operations-commit) and "
        f"9.7.18.6.2.1 ({PTX}#tcgen05-memory-consistency-model-pipelined-instructions-implicit)",
    ),
    "wait_group_read": Rule(
        "cp.async.bulk.wait_group.read N returns once all but the N most recent bulk groups of "
        "the executing thread finished reading their sources; other threads learn it only "
        "through a later synchronization by that thread.",
        f"PTX ISA 9.5 9.7.10.28.6.2 cp.async.bulk.wait_group "
        f"({PTX}#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)",
    ),
    "proxy": Rule(
        "Generic-proxy and async-proxy (TMA, tcgen05.mma operand) accesses to the same SMEM need "
        "fence.proxy.async in the generic thread before the synchronization.",
        f"PTX ISA 9.5 9.7.10.28.2 Async Proxy ({PTX}#async-proxy) and 9.7.18.6.5 "
        f"({PTX}#tcgen05-memory-consistency-model-smem-access)",
    ),
    "tmem_store": Rule(
        "tcgen05.st completes only at tcgen05.wait::st; a thread handing stored columns to "
        "another thread or to tcgen05.mma waits (warpdrive's tmem_store fence) before it "
        "signals.",
        f"PTX ISA 9.5 9.7.18.6.2.1.2 and 9.7.18.6.3 "
        f"({PTX}#tcgen05-memory-consistency-model-inter-thread-sync)",
    ),
    "complete_tx": Rule(
        "A TMA load's complete-tx publishes only that copy's bytes on its barrier; it does not "
        "order the issuing thread's other memory operations.",
        f"PTX ISA 9.5 8.9.1.1 Asynchronous Operations ({PTX}#program-order-async-operations)",
    ),
    "channel": Rule(
        "A managed signal (commit/release) followed by the matching wait/acquire of the same "
        "channel, slot and epoch orders what the signaller completed before it; the consumers "
        "of one fan-out epoch (and the producers of one fan-in epoch) are mutually unordered.",
        "warpdrive GUARANTEES.md R2-R5; PTX ISA 9.5 9.7.15.16.16 mbarrier.arrive "
        f"({PTX}#parallel-synchronization-and-communication-instructions-mbarrier-arrive)",
    ),
    "barrier": Rule(
        "Warps of one role are ordered within a handle only by a barrier over the whole role "
        "between the two accesses (record it with named_barrier). Different warps are never "
        "pipelined; PTX also requires tcgen05.wait::ld/st and tcgen05.fence::before/"
        "after_thread_sync around that sync, which phase 1 assumes rather than checks.",
        f"PTX ISA 9.5 9.7.18.6.2 and 9.7.18.6.3 "
        f"({PTX}#tcgen05-memory-consistency-model-inter-thread-sync)",
    ),
    "stages": Rule(
        "A ring hands over one stage at a time: overlapping buffers bound to one channel must "
        "overlap only in the same stage.",
        "warpdrive GUARANTEES.md R2",
    ),
}

HAZARD = "potential hazard"
UNVERIFIED = "unverified"


# --------------------------------------------------------------------------- #
# Recording wrappers: the same instruction as the cute.arch call, plus a trace event
# --------------------------------------------------------------------------- #


def _record(kind: str, **data):
    trace = current_trace()
    if trace is not None:
        trace.event(kind, **data)


def fence_async_shared():
    """``cute.arch.fence_view_async_shared()`` (fence.proxy.async), visible to the diagnostic."""
    cute.arch.fence_view_async_shared()
    _record("fence", kind="smem_async")


def bulk_commit_group():
    """``cute.arch.cp_async_bulk_commit_group()``, visible to the diagnostic."""
    cute.arch.cp_async_bulk_commit_group()
    _record("bulk_commit")


def bulk_wait_group(n: int, *, read: bool = True):
    """``cute.arch.cp_async_bulk_wait_group(n, read=read)``, visible to the diagnostic."""
    cute.arch.cp_async_bulk_wait_group(n, read=read)
    _record("bulk_wait", n=n, read=read)


def named_barrier(barrier_id: int, number_of_threads: int):
    """``cute.arch.barrier(...)``, visible to the diagnostic. Only a barrier over every thread
    of the calling role (``number_of_threads == 32 * warps``) orders the role's warps."""
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=number_of_threads)
    # A Python int id lets the kernel reject a collision with a named-barrier channel's IDs.
    _record(
        "barrier", threads=number_of_threads, id=barrier_id if type(barrier_id) is int else None
    )


# --------------------------------------------------------------------------- #
# Per-role trace data
# --------------------------------------------------------------------------- #


@dataclass
class RoleEvents:
    """One traced role instance's events, as plain data (no MLIR objects)."""

    name: str
    warps: tuple[int, ...]
    raw: bool
    events: list
    item_key: int | None

    @classmethod
    def from_trace(cls, trace, ri) -> RoleEvents:
        item = trace.block_key(trace.item_block) if trace.item_block is not None else None
        return cls(ri.name, tuple(ri.warps), ri.raw, list(trace.events), item)


@dataclass
class Access:
    role: RoleEvents
    index: int  # event index in role.events
    storage: tuple[str, str]  # ("tmem" | "smem", name)
    slot: int | None
    read: bool
    write: bool
    agent: str
    parts: list  # [(warp | None, lanes | None, span)]
    handles: tuple  # open HandleRecords
    site: str
    handle: object = None  # the handle whose stage view an SMEM access goes through

    def describe(self) -> str:
        mode = "read/write" if self.read and self.write else "write" if self.write else "read"
        slot = "" if self.slot is None else f"[{self.slot}]"
        return f"{self.role.name} {mode} {self.storage[1]}{slot} by {self.agent} @ {self.site}"


@dataclass
class Finding:
    severity: str
    text: str
    rule: str

    def describe(self) -> str:
        return f"{self.severity}: {self.text} [{self.rule}]"


@dataclass
class AliasResult:
    names: tuple[str, str]
    kind: str  # "tmem" | "smem"
    pairs: int = 0  # conflicting annotated access pairs examined
    findings: list = field(default_factory=list)
    unverified: list = field(default_factory=list)  # reasons

    @property
    def status(self) -> str:
        if any(f.severity == HAZARD for f in self.findings):
            return HAZARD
        if self.unverified or self.findings:
            return UNVERIFIED
        return "no hazard found"

    def describe(self) -> str:
        head = f"  {self.names[0]} ~ {self.names[1]} ({self.kind})  {self.status}"
        if self.status == "no hazard found":
            head += f" among {self.pairs} conflicting annotated access pair(s)"
        lines = [head]
        lines += [f"      {f.describe()}" for f in self.findings]
        lines += [f"      unverified: {r}" for r in dict.fromkeys(self.unverified)]
        return "\n".join(lines)


@dataclass
class AliasReport:
    results: list

    def describe(self) -> str:
        if not self.results:
            return ""
        head = "Alias diagnostic (bug finder: reports potential hazards; never proves safety)"
        return "\n".join([head, *(r.describe() for r in self.results)])

    @property
    def hazards(self) -> list:
        return [r for r in self.results if r.status == HAZARD]

    @property
    def flagged(self) -> list:
        """Aliases with a finding (a potential hazard, or an edge only a raw instruction could
        supply): what the default warning names."""
        return [r for r in self.results if r.findings]


# --------------------------------------------------------------------------- #
# Declared overlaps and footprints
# --------------------------------------------------------------------------- #


def _overlap(a: range | None, b: range | None) -> bool:
    if a is None or b is None:
        return True
    return a.start < b.stop and b.start < a.stop


def _declared_pairs(kernel) -> list[tuple[str, tuple[str, str]]]:
    pairs = []
    plan = kernel.tmem_plan
    if plan is not None:
        regions = list(plan.regions.values())
        for i, x in enumerate(regions):
            for y in regions[i + 1 :]:
                if x.alias_of is None and y.alias_of is None:
                    continue  # plan regions never overlap each other
                spans = [
                    range(plan.col(r.name, j), plan.col(r.name, j) + r.cols)
                    for r in (x, y)
                    for j in range(r.slots)
                ]
                xs, ys = spans[: x.slots], spans[x.slots :]
                if any(_overlap(p, q) for p in xs for q in ys):
                    pairs.append(("tmem", (y.name, x.name)))
    for i, (xn, x) in enumerate(kernel._ws_smem.items()):
        for yn, y in list(kernel._ws_smem.items())[i + 1 :]:
            if x.buffer.root is not y.buffer.root:
                continue
            if _overlap(
                range(x.nbytes) if x.nbytes is not None else None,
                range(y.nbytes) if y.nbytes is not None else None,
            ):
                pairs.append(("smem", (yn, xn)))
    return pairs


def _span(value, warp: int) -> range:
    return value(warp) if callable(value) else value


def _parts(kernel, ev, role: RoleEvents) -> list:
    kind, name = ev.data["storage"]
    if kind == "tmem":
        plan = kernel.tmem_plan
        base, width = plan.col(name, ev.data["slot"]), plan.regions[name].cols
        parts = []
        warps = ev.data["warps"] or role.warps
        if not set(warps) <= set(role.warps):
            raise ProtocolError(
                f"{role.name}: access to {name} @ {ev.site} names warps {warps}, outside the "
                f"role's warps {role.warps}"
            )
        for w in warps:
            lanes = ev.data["lanes"]
            if lanes is None:
                lanes = (
                    range(128)
                    if ev.data["agent"] == "umma"
                    else range(32 * (w % 4), 32 * (w % 4) + 32)
                )
            cols = ev.data["cols"]
            cols = range(width) if cols is None else _span(cols, w)
            lanes = _span(lanes, w)
            if cols.start < 0 or cols.stop > width or lanes.start < 0 or lanes.stop > 128:
                raise ProtocolError(
                    f"{role.name}: access to {name} @ {ev.site}: warp {w}'s footprint (lanes "
                    f"{lanes}, cols {cols}) leaves the region (128 lanes, {width} cols)"
                )
            parts.append((w, lanes, range(base + cols.start, base + cols.stop)))
        return parts
    res = kernel._ws_smem[name]
    span = None  # a ring stage (dynamic) or a layout without static stage spans
    if ev.data["slot"] is not None and res.stage_span(ev.data["slot"]) is not None:
        span = range(*res.stage_span(ev.data["slot"]))
    return [(None, None, span)]


def _conflicts(a: Access, b: Access) -> list[tuple]:
    """Warp pairs whose footprints overlap (``(None, None)`` for SMEM)."""
    out = []
    for wa, la, sa in a.parts:
        for wb, lb, sb in b.parts:
            if _overlap(la, lb) and _overlap(sa, sb):
                out.append((wa, wb))
    return out


# --------------------------------------------------------------------------- #
# Completion of one access at a signal (the first hop of an ordering edge)
# --------------------------------------------------------------------------- #


def _between(role: RoleEvents, seq: list[int], lo: int, hi: int):
    """Events of ``role`` at positions ``lo < p < hi`` of ``seq`` (event indices)."""
    return [role.events[seq[p]] for p in range(lo + 1, hi)]


def _bulk_done(role: RoleEvents, seq: list[int], pos_a: int, pos_e: int) -> bool:
    """Does a recorded wait_group.read between the TMA store at ``pos_a`` and ``pos_e`` cover
    the store's bulk group?"""
    commits = sum(1 for p in range(pos_a) if role.events[seq[p]].kind == "bulk_commit")
    group = commits  # the store joins the next committed group
    seen = commits
    committed = False
    for p in range(pos_a + 1, pos_e):
        ev = role.events[seq[p]]
        if ev.kind == "bulk_commit":
            seen += 1
            committed = True
        elif ev.kind == "bulk_wait" and committed:
            if seen - ev.data["n"] > group:
                return True
    return False


def _completion(
    a: Access, role: RoleEvents, seq: list[int], pos_a: int, pos_e: int, close, other: str
) -> Finding | None:
    """None when access ``a`` (at ``pos_a``) has completed, and is published to a thread
    synchronizing with the signal at ``pos_e`` (a close event), for a conflicting access by
    agent ``other``; otherwise the missing edge."""
    rec = close.data["handle"]
    chan = rec.chan
    side = "producer" if rec.producer else "consumer"
    mech = chan.producer if rec.producer else chan.consumer
    mech_name = getattr(mech, "__name__", type(mech).__name__)
    fence = chan.commit_fence if rec.producer else chan.release_fence
    where = f"{a.describe()} before {chan.name} {close.site}"
    if not close.data["signal"]:
        return Finding(UNVERIFIED, f"{where}: raw signal (signal=False) is not visible", "channel")
    if a.agent == "tma" and a.write:
        if rec is a.handle and rec.producer and mech_name == "Tma":
            return None
        return Finding(
            HAZARD,
            f"{where}: a TMA load is published only by the transaction count of the handle "
            f"whose barrier it uses, not by this {mech_name} {side} signal",
            "complete_tx",
        )
    if a.agent == "tma":
        if mech_name != "Threads":
            return Finding(
                HAZARD,
                f"{where}: nothing waits for the TMA store's source reads",
                "wait_group_read",
            )
        if _bulk_done(role, seq, pos_a, pos_e):
            return None
        return Finding(
            UNVERIFIED,
            f"{where}: no recorded bulk_wait_group(read=True) covers the store before the signal",
            "wait_group_read",
        )
    if a.agent == "umma":
        if mech_name == "Umma":
            return None
        return Finding(
            HAZARD,
            f"{where}: tcgen05.mma completes only through tcgen05.commit, not this {mech_name} "
            f"{side} signal",
            "commit",
        )
    if mech_name != "Threads":
        return Finding(
            HAZARD,
            f"{where}: a {mech_name} {side} signal does not publish this thread's accesses",
            "commit" if mech_name == "Umma" else "complete_tx",
        )
    if a.storage[0] == "tmem" and a.write and fence != "tmem_store":
        return Finding(
            UNVERIFIED,
            f"{where}: tcgen05.st is not waited (channel tmem_store fence) before the signal "
            f"that hands its columns to a later {other} access",
            "tmem_store",
        )
    if other in ("tma", "umma") and a.storage[0] == "smem":
        fences = {e.data["kind"] for e in _between(role, seq, pos_a, pos_e) if e.kind == "fence"}
        if fence != "smem_async" and "smem_async" not in fences:
            return Finding(
                UNVERIFIED,
                f"{where}: no fence.proxy.async between a generic access and a later {other} "
                "access (channel smem_async fence or fence_async_shared())",
                "proxy",
            )
    return None


# --------------------------------------------------------------------------- #
# Same-thread ordering inside one handle period
# --------------------------------------------------------------------------- #


def _pipelined(first: Access, second: Access) -> bool:
    if first.storage[0] == "smem":
        return first.agent == second.agent == "threads"
    if first.agent == second.agent == "threads":
        return True  # st -> ld, st -> st, ld -> st (ld -> ld does not conflict)
    if first.agent == second.agent == "umma":
        if first.read and not first.write:
            return True  # A-operand read -> D write
        same_acc = first.storage == second.storage and first.slot == second.slot
        return same_acc and second.write and not second.read  # D write -> D write
    return False


def _role_barrier(role: RoleEvents, between) -> bool:
    """A recorded named_barrier over every thread of the role among ``between``."""
    return any(e.kind == "barrier" and e.data["threads"] == 32 * len(role.warps) for e in between)


def _same_period(
    a: Access, b: Access, warp_pairs, between=None, first_is_a=None
) -> Finding | None:
    """Program order inside one thread set: ``between`` are the events separating the two
    (default: the trace order of one handle period)."""
    if first_is_a is None:
        first_is_a = a.index < b.index
    first, second = (a, b) if first_is_a else (b, a)
    if between is None:
        between = first.role.events[first.index + 1 : second.index]
    barrier = _role_barrier(first.role, between)
    for wa, wb in warp_pairs:
        if wa is not None and wb is not None and wa != wb:
            if not barrier:
                return Finding(
                    UNVERIFIED,
                    f"{first.describe()} and {second.describe()}: warps {wa} and {wb} touch the "
                    "same TMEM lanes and columns in one handle with no recorded named_barrier "
                    "between them",
                    "barrier",
                )
        elif not _pipelined(first, second):
            return Finding(
                HAZARD,
                f"{first.describe()} then {second.describe()}: not a pipelined pair in one "
                "thread, and no completion wait between them",
                "pipelined",
            )
    return None


# --------------------------------------------------------------------------- #
# Whole-region handoff through one channel
# --------------------------------------------------------------------------- #


def _handoff_handles(acc: Access) -> list:
    """Handles whose channel hands ``acc``'s storage over as a whole: its own stage view's
    handle (a ring stage is handed over only through that ring), else any open handle of a
    channel with one stage per slot (Slots, or a one-stage Ring). A deeper Ring orders epochs
    e and e + n, not consecutive ones, so it hands over only storage indexed by its stage."""
    if acc.handle is not None:
        return [acc.handle]
    # A round-robin ring's epochs do not follow one role's program order, which the model's
    # epoch matching assumes: it adds no edges (accesses ordered only through it are flagged).
    return [
        h
        for h in acc.handles
        if not h.chan.internal and not h.chan.round_robin and (h.chan.fixed or h.chan.n == 1)
    ]


def _handle_on(acc: Access, chan, slot):
    return next((h for h in _handoff_handles(acc) if h.chan is chan and h.slot == slot), None)


def _close_after(role: RoleEvents, index: int, rec) -> int | None:
    for i in range(index + 1, len(role.events)):
        ev = role.events[i]
        if ev.kind == "close" and ev.data["handle"] is rec:
            return i
    return None


def _open_before(role: RoleEvents, index: int, rec) -> int:
    for i in range(index - 1, -1, -1):
        ev = role.events[i]
        if ev.kind == "open" and ev.data["handle"] is rec:
            return i
    return -1


def _handoff(a: Access, b: Access, chan, slot, warp_pairs) -> Finding | None:
    ha, hb = _handle_on(a, chan, slot), _handle_on(b, chan, slot)
    if a.role is b.role:
        # One role: the channel separates the two accesses only when the first one's handle
        # closes before the second one's opens; overlapping lifetimes are one period.
        first, hf, second, hs = (a, ha, b, hb) if a.index < b.index else (b, hb, a, ha)
        close = _close_after(first.role, first.index, hf)
        separated = (
            hf is not hs
            and close is not None
            and close < _open_before(second.role, second.index, hs)
        )
        if not separated:
            found = _same_period(a, b, warp_pairs)
            if found is not None:
                return found
    elif ha.producer == hb.producer:
        side = "producers" if ha.producer else "consumers"
        return Finding(
            HAZARD,
            f"{a.describe()} and {b.describe()}: {side} of one {chan.name} epoch are unordered",
            "channel",
        )
    # Other epochs: each access must complete before its own handle's close signals.
    for x, hx, other in ((a, ha, b), (b, hb, a)):
        seq = list(range(len(x.role.events)))
        close = _close_after(x.role, x.index, hx)
        if close is None:
            return Finding(
                UNVERIFIED, f"{x.describe()}: its {chan.name} handle never closes", "channel"
            )
        found = _completion(x, x.role, seq, x.index, close, x.role.events[close], other.agent)
        if found is not None:
            return found
    return None


# --------------------------------------------------------------------------- #
# Cross-channel happens-before graph
# --------------------------------------------------------------------------- #


UNROLL_LIMIT = 100_000  # events per role after unrolling (each loop level doubles it)
SEARCH_LIMIT = 5_000_000  # graph states visited per compile


class _TooLarge(Exception):
    pass


def _unroll(role: RoleEvents, items: int) -> list[tuple[int, int]]:
    """``[(event index, item)]`` in program order: dynamic loops twice, the work-item loop
    ``items`` times (prologue item -1, epilogue item ``items``), branches once."""
    root: list = []
    stack: list = [(None, root)]
    for i, ev in enumerate(role.events):
        keys = [k for k, _ in ev.where]
        depth = 0
        while depth < len(stack) - 1 and depth < len(keys) and stack[depth + 1][0] == keys[depth]:
            depth += 1
        del stack[depth + 1 :]
        for key, kind in ev.where[depth:]:
            node = (key, kind, [])
            stack[-1][1].append(node)
            stack.append((key, node[2]))
        stack[-1][1].append(i)
    out: list = []

    def emit(children, item):
        for c in children:
            if len(out) > UNROLL_LIMIT:
                raise _TooLarge(role.name)
            if isinstance(c, int):
                out.append((c, item))
            elif c[0] == role.item_key:
                for it in range(items):
                    emit(c[2], it)
            else:
                for _ in range(2 if c[1] == "loop" else 1):
                    emit(c[2], item)

    item = -1
    for c in root:
        if not isinstance(c, int) and c[0] == role.item_key:
            emit([c], item)
            item = items
        else:
            emit([c], item)
    return out


class _Graph:
    def __init__(self, roles: list[RoleEvents], items: int):
        self.roles = roles
        self.seq = {id(r): _unroll(r, items) for r in roles}
        self.pos = {}  # (role id, event index) -> positions
        for r in roles:
            for p, (e, _) in enumerate(self.seq[id(r)]):
                self.pos.setdefault((id(r), e), []).append(p)
        self.edges: dict = {}  # (role id, pos) -> [(role id, pos)]
        self.steps = 0  # search work, bounded by SEARCH_LIMIT
        self.cache: dict = {}  # (access, position, other agent) -> (reachable, failures, any)
        self._link(items)

    def _ops(self, role, want):
        """Per item: positions of ``want`` ("open" or signalling "close") events per
        ``(channel, slot, producer)``."""
        out: dict = {}
        for p, (e, item) in enumerate(self.seq[id(role)]):
            ev = role.events[e]
            if ev.kind != want or (want == "close" and not ev.data["signal"]):
                continue
            rec = ev.data["handle"]
            if rec.chan.internal or rec.chan.round_robin:  # see _handoff_handles
                continue
            key = (rec.chan.name, rec.slot, rec.producer)
            out.setdefault(key, {}).setdefault(item, []).append(p)
        return out

    def _link(self, items: int):
        opens = {id(r): self._ops(r, "open") for r in self.roles}
        closes = {id(r): self._ops(r, "close") for r in self.roles}
        chans = {
            ev.data["handle"].chan.name: ev.data["handle"].chan
            for r in self.roles
            for ev in r.events
            if ev.kind == "open"
        }
        span = range(-1, items + 1)
        for src in self.roles:
            for (name, slot, producer), by_item in closes[id(src)].items():
                chan = chans[name]
                for dst in self.roles:
                    if dst is src:
                        continue
                    waits = opens[id(dst)].get((name, slot, not producer))
                    if not waits:
                        continue
                    if producer:  # publication: commit k -> wait k of the same item
                        for item in span:
                            cs, ws = by_item.get(item, []), waits.get(item, [])
                            if not ws:
                                continue
                            for k, c in enumerate(cs):
                                w = ws[k] if len(cs) == len(ws) else ws[-1]
                                self._edge(src, c, dst, w)
                    else:
                        self._reuse(src, by_item, dst, waits, chan, span)

    def _reuse(self, src, releases, dst, acquires, chan, span):
        """release of epoch e -> acquire of epoch e + stages (one stage per Slots slot)."""
        equal = all(len(releases.get(i, [])) == len(acquires.get(i, [])) for i in span)
        flat_r = [p for i in span for p in releases.get(i, [])]
        flat_a = [p for i in span for p in acquires.get(i, [])]
        if equal and chan.fixed:
            for k, r in enumerate(flat_r):
                if k + 1 < len(flat_a):
                    self._edge(src, r, dst, flat_a[k + 1])
        elif equal:  # a ring: exact only when no op of the channel sits in an inner loop
            if any(
                len(releases.get(i, [])) != len(set(self._events(src, releases.get(i, []))))
                for i in span
            ):
                return
            for k, r in enumerate(flat_r):
                if k + chan.n < len(flat_a):
                    self._edge(src, r, dst, flat_a[k + chan.n])
        elif chan.fixed:  # unmatched counts: the next item's first acquire follows them all
            for i in span:
                nxt = acquires.get(i + 1, [])
                for r in releases.get(i, []) if nxt else []:
                    self._edge(src, r, dst, nxt[0])

    def _events(self, role, positions):
        return [self.seq[id(role)][p][0] for p in positions]

    def _edge(self, src, p, dst, q):
        self.edges.setdefault((id(src), p), []).append((dst, q))

    def reachable(self, a: Access, pos_a: int, check) -> set:
        """``(role id, position)`` of every instance a path from ``a``'s instance reaches after
        leaving ``a``'s role through a signal at which ``check(pos_e)`` holds."""
        start = (id(a.role), pos_a, False)
        seen, todo = {start}, deque([start])
        while todo:
            self.steps += 1
            if self.steps > SEARCH_LIMIT:
                raise _TooLarge(f"the ordering search passed {SEARCH_LIMIT} steps")
            rid, p, left = todo.popleft()
            nxt = []
            if p + 1 < len(self.seq[rid]):
                nxt.append((rid, p + 1, left))
            for dst, q in self.edges.get((rid, p), []):
                if not left and rid == id(a.role) and not check(p):
                    continue
                nxt.append((id(dst), q, True))
            for state in nxt:
                if state not in seen:
                    seen.add(state)
                    todo.append(state)
        return {(rid, p) for rid, p, left in seen if left}


def _cross_channel(graph: _Graph, a: Access, b: Access) -> Finding | None:
    """Every instance pair of ``a`` and ``b`` in the unrolled window must be ordered one way."""
    for pa in graph.pos.get((id(a.role), a.index), []):
        for pb in graph.pos.get((id(b.role), b.index), []):
            found = None
            for x, px, y, py in ((a, pa, b, pb), (b, pb, a, pa)):
                ordered, failures = _graph_order(graph, x, px, y, py)
                if ordered:
                    break
                if failures:
                    found = found or failures[0]
            else:
                items = (graph.seq[id(a.role)][pa][1], graph.seq[id(b.role)][pb][1])
                return found or Finding(
                    HAZARD,
                    f"{a.describe()} (work item {items[0]}) and {b.describe()} (work item "
                    f"{items[1]}): no channel path orders them",
                    "channel",
                )
    return None


def _reach(graph: _Graph, x: Access, px: int, other: str) -> tuple[set, list, set]:
    """What a path from instance ``px`` of ``x`` orders it before (for a conflicting ``other``
    agent), the completion failures of the signals it could not use, and what it would reach
    ignoring completion. Cached per instance."""
    key = (id(x), px, other)
    if key not in graph.cache:
        seq = [e for e, _ in graph.seq[id(x.role)]]
        failures: list = []
        verdicts: dict = {}

        def check(pos_e):
            if pos_e not in verdicts:
                close = x.role.events[seq[pos_e]]
                verdicts[pos_e] = _completion(x, x.role, seq, px, pos_e, close, other)
                if verdicts[pos_e] is not None:
                    failures.append(verdicts[pos_e])
            return verdicts[pos_e] is None

        ordered = graph.reachable(x, px, check)
        anyway = graph.reachable(x, px, lambda pos_e: True) if failures else ordered
        graph.cache[key] = (ordered, failures, anyway)
    return graph.cache[key]


def _graph_order(graph: _Graph, x: Access, px: int, y: Access, py: int) -> tuple[bool, list]:
    """Does a path order instance ``px`` of ``x`` before ``py`` of ``y``? Also the completion
    failures of the signals that would have ordered it."""
    ordered, failures, anyway = _reach(graph, x, px, y.agent)
    target = (id(y.role), py)
    return target in ordered, failures if target in anyway else []


def _same_role(graph: _Graph, a: Access, b: Access, warp_pairs) -> Finding | None:
    """One role, every instance order in the unrolled window (loop-carried pairs included):
    program order inside one thread when pipelined (a recorded role barrier across warps), else
    a round trip through another role that the earlier access completed before."""
    seq = graph.seq[id(a.role)]
    for pa in graph.pos.get((id(a.role), a.index), []):
        for pb in graph.pos.get((id(b.role), b.index), []):
            if pa == pb:
                continue
            (x, px), (y, py) = sorted(((a, pa), (b, pb)), key=lambda t: t[1])
            between = [a.role.events[e] for e, _ in seq[px + 1 : py]]
            found = _same_period(a, b, warp_pairs, between, first_is_a=x is a)
            if found is None:
                continue
            ordered, failures = _graph_order(graph, x, px, y, py)
            if not ordered:
                return failures[0] if failures and found.severity != HAZARD else found
    return None


def _stage_alignment(kernel, x: str, y: str) -> Finding | str | None:
    rx, ry = kernel._ws_smem[x], kernel._ws_smem[y]
    for chan in set(rx.channels) & set(ry.channels):
        if rx.stage_span(0) is None or ry.stage_span(0) is None:
            return f"channel {chan}: stage spans of {x} and {y} are not static and disjoint"
        for s in range(rx.stages):
            for t in range(ry.stages):
                if s != t and _overlap(range(*rx.stage_span(s)), range(*ry.stage_span(t))):
                    return Finding(
                        HAZARD,
                        f"channel {chan}: stage {s} of {x} overlaps stage {t} of {y}",
                        "stages",
                    )
    return None


def check(kernel, roles: list[RoleEvents]) -> AliasReport:
    """Run the diagnostic over the roles traced in one compile."""
    results = []
    pairs = _declared_pairs(kernel)
    if not pairs:
        return AliasReport(results)
    accesses: list[Access] = []
    for role in roles:
        for i, ev in enumerate(role.events):
            if ev.kind == "access" and not ev.opted_out:
                d = ev.data
                accesses.append(
                    Access(
                        role,
                        i,
                        d["storage"],
                        d["slot"],
                        d["read"],
                        d["write"],
                        d["agent"],
                        _parts(kernel, ev, role),
                        ev.open,
                        ev.site,
                        d.get("handle"),
                    )
                )
    streams = getattr(kernel, "_ws_streams", {})
    items = 2 if any(st.schedule.persistent for st in streams.values()) else 1
    graph: list = []  # built on the first cross-channel pair

    def get_graph() -> _Graph | str:
        if not graph:
            try:
                graph.append(_Graph([r for r in roles if not r.raw], items))
            except _TooLarge as exc:
                graph.append(f"role {exc} unrolls past {UNROLL_LIMIT} events; not modeled")
        return graph[0]

    for kind, (x, y) in pairs:
        res = AliasResult((x, y), kind)
        results.append(res)
        if kernel.cluster.size > 1:
            res.unverified.append(
                f"{type(kernel.cluster).__name__} kernels are not modeled (phase 1 is single-CTA)"
            )
            continue
        mine = {n: [a for a in accesses if a.storage == (kind, n)] for n in (x, y)}
        for n in (x, y):
            if not mine[n]:
                res.unverified.append(f"no annotated access to {n}")
            for role in roles:
                raw = [
                    e for e in role.events if e.kind == "raw" and e.data["storage"] == (kind, n)
                ]
                if raw and not any(a.role is role for a in mine[n]):
                    res.unverified.append(
                        f"{role.name} acquires {n} without annotating its accesses @ {raw[0].site}"
                    )
        for a in mine[x] + mine[y]:
            if a.role.raw:
                res.unverified.append(
                    f"raw role {a.role.name}: its synchronization is not visible"
                )
        if kind == "smem":
            found = _stage_alignment(kernel, x, y)
            if isinstance(found, str):
                res.unverified.append(found)
            elif found is not None:
                res.findings.append(found)
        for a in mine[x]:
            for b in mine[y]:
                if not (a.write or b.write) or a.role.raw or b.role.raw:
                    continue
                warp_pairs = _conflicts(a, b)
                if not warp_pairs:
                    continue
                res.pairs += 1
                found = _order(a, b, warp_pairs, get_graph)
                if isinstance(found, str):
                    res.unverified.append(found)
                elif found is not None and all(f.text != found.text for f in res.findings):
                    res.findings.append(found)
    return AliasReport(results)


def _order(a: Access, b: Access, warp_pairs, get_graph) -> Finding | str | None:
    """The missing edge between conflicting ``a`` and ``b`` (a str: why it is unverified).
    Any one channel both accesses are inside that orders them is enough."""
    found = []
    ring_stage = a.storage[0] == "smem" and (a.slot is None or b.slot is None)
    # One role and static storage: a handle open around a loop would hide the loop-carried
    # pair, so only the unrolled graph decides (trace order, loop-carried instances and round
    # trips). A ring stage is one epoch per handle, so its own handles still decide.
    for ha in _handoff_handles(a) if a.role is not b.role or ring_stage else ():
        if _handle_on(b, ha.chan, ha.slot) is not None:
            why = _handoff(a, b, ha.chan, ha.slot, warp_pairs)
            if why is None:
                return None
            found.append(why)
    if found and ring_stage:
        return next((f for f in found if f.severity == HAZARD), found[0])
    if ring_stage:
        return f"{a.describe()} and {b.describe()}: ring-stage accesses on different channels"
    # Another path (another channel, a later signal, or a round trip through another role) may
    # order what a shared channel does not; the graph knows which instances follow each other.
    graph = get_graph()
    if isinstance(graph, str):
        return graph
    try:
        if a.role is b.role:
            return _same_role(graph, a, b, warp_pairs)
        return _cross_channel(graph, a, b)
    except _TooLarge as exc:
        return f"{a.describe()} and {b.describe()}: {exc}; not modeled"


def report(rep: AliasReport, mode: str, kernel_name: str):
    """Warn once (or raise in strict mode) about the findings of ``rep``."""
    if mode == "strict" and rep.hazards:
        raise ProtocolError(f"warpdrive alias diagnostic (strict):\n{rep.describe()}")
    if rep.flagged:
        names = ", ".join(f"{r.names[0]}~{r.names[1]}" for r in rep.flagged)
        first = rep.flagged[0].findings[0].describe()
        warnings.warn(
            f"warpdrive: alias diagnostic flagged {names} in {kernel_name} (no error raised; see "
            f"describe()): {first}",
            stacklevel=3,
        )
