"""Trace-time check of TMA loads against the channel they complete on.

A TMA load names its barrier through ``cute.copy(atom, src, dst, tma_bar_ptr=ptr)``. Whether the
bytes land where the layer armed them depends on the atom: a ``CtaGroup.TWO`` load completes on
the leader CTA's barrier from either CTA of a pair, a ``CtaGroup.ONE`` load on the issuing CTA's
own barrier. A mismatch with the channel's derived scope, or a ``Tma(tx_bytes)`` that differs from
the bytes actually copied, leaves a barrier that never completes (a hang) or completes early.

The layer binds copies to channels by usage, with no change to user code. Every full-barrier
pointer of a ``Tma`` channel it hands out (``handle.barrier``, ``end.full_barrier(i)``) is recorded
with its SSA value. When the role returns, the check follows that value's uses in the traced IR to
``cute_nvgpu.atom.set_value<tma_bar>`` ops, reads the atom's type (load kind, ``copy_bits`` per
box) and each ``cute.copy`` destination layout, and compares:

* the atom's CTA group with the one the channel's scope requires;
* per acquired stage, the bytes of every copy issued in the stage's region with the bytes the
  layer arms (``Scope.tx``, per producing CTA).

The check fails open: anything it cannot recognize (a CuTeDSL IR change, an unknown atom kind, a
copy inside a dynamic loop, a pointer the layer did not hand out) marks the channel *unverified*
in ``describe()`` and emits one warning per kernel. It raises only on a positively identified
mismatch. Values are inspected without the DSL's value casters, which emit ops when invoked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from transformer_nuggets.cute.warpdrive.cluster import LEADER
from transformer_nuggets.cute.warpdrive.protocol import ProtocolError

SET_VALUE = "cute_nvgpu.atom.set_value"
MAKE_EXEC = "cute_nvgpu.atom.make_exec_tma"
COPY = "cute.copy"
# TiledTmaLoadEnum spelling -> (CtaGroup, multicast)
KINDS = {
    "sm_90": ("ONE", False),
    "sm_100_2sm": ("TWO", False),
    "sm_90_multicast": ("ONE", True),
    "sm_100_2sm_multicast": ("TWO", True),
}


# Multicast masks handed out by ``Cluster.multicast`` in the kernel being traced:
# {hash of the mask's ir.Value: (mode, CTAs reached)}. Tracing is single-threaded; cleared at
# ``begin_roles``.
_MASKS: dict[int, tuple[str, int]] = {}


def record_multicast_mask(mask, mode: str, ctas: int) -> None:
    _MASKS[hash(mask.ir_value())] = (mode, ctas)


def clear_multicast_masks() -> None:
    _MASKS.clear()


@dataclass
class TmaUse:
    """A full-barrier pointer of a TMA channel handed to user code."""

    chan: object
    value: object  # the pointer's ir.Value
    site: str
    handle: object | None  # the HandleRecord of the acquired stage; None for end.full_barrier
    opted_out: bool = False  # taken inside unchecked("tma"): its copies are not checked
    # (copies, set_value ops, problem), computed when the stage's handle closes: its region is
    # live then, while a region the tracer discards later must not be walked.
    result: tuple | None = None


@dataclass
class TmaReport:
    """What the check established for one channel (shown by ``describe()``)."""

    copies: set = field(default_factory=set)  # sites of verified copies
    groups: set = field(default_factory=set)
    stage_bytes: set = field(default_factory=set)
    unverified: list = field(default_factory=list)
    opted_out: bool = False

    def status(self, fmt_sites=None) -> str:
        """``verified ...``, ``unverified (reasons)``, or ``opted out``; ``fmt_sites`` joins the
        copy sites (default: comma-separated)."""
        if self.unverified:
            return f"unverified ({'; '.join(dict.fromkeys(self.unverified))})"
        if not self.copies:
            return "opted out (unchecked)" if self.opted_out else "no TMA copies traced"
        groups = "/".join(f"CtaGroup.{g}" for g in sorted(self.groups))
        nbytes = ", ".join(str(b) for b in sorted(self.stage_bytes)) or "not counted"
        sites = (fmt_sites or (lambda s: ", ".join(sorted(s))))(self.copies)
        partial = "; some barriers opted out" if self.opted_out else ""
        return f"verified {groups}, {nbytes} B/stage per CTA @ {sites}{partial}"


@dataclass
class _Copy:
    site: str
    group: str
    nbytes: int | None  # None: not countable (dynamic layout or unparsed type)
    in_stage_region: bool  # issued in the region that acquired the stage (runs once per stage)
    # The cluster mode of a multicast copy's mask ("?": a mask not from Cluster.multicast);
    # None when not multicast.
    multicast: str | None = None


@dataclass
class _Stage:
    """Every barrier use of one acquired stage; bytes are compared only when complete."""

    use: TmaUse
    copies: list = field(default_factory=list)
    incomplete: list = field(default_factory=list)  # why the byte total may be partial
    opted_out: bool = False  # a barrier of this stage was taken inside unchecked("tma")


def required_group(chan, cluster) -> str | None:
    """The CTA group a TMA load into ``chan`` must use (None: either works)."""
    if not cluster.pair:
        return "ONE"
    if chan.derived_scope.tx_ctas > 1:
        return "TWO"  # both CTAs' bytes complete on the leader's barrier
    leaders = set(cluster.ctas(LEADER))
    if all(set(cluster.ctas(ri.cta)) <= leaders for ri, _ in [*chan.producers, *chan.consumers]):
        return None  # leader-only on both sides: a 2-CTA load also lands on the leader
    return "ONE"


def expected_stage_bytes(chan) -> int:
    """Bytes one producing CTA's copies must deliver per stage for the armed transaction count
    (a multicast copy delivers its box to every CTA of its mask)."""
    sc = chan.scope
    return sc.tx // chan.derived_scope.tx_ctas


def stage_closed(trace, handle) -> None:
    """Analyze the barrier uses of one TMA stage when its handle closes."""
    for use in trace.tma_bars:
        if use.handle is handle and use.result is None:
            use.result = _copies(use)


def check_role(trace, cluster, reports: dict) -> None:
    """Analyze every TMA barrier use recorded while tracing one role.

    The atom's CTA group is checked for every copy positively traced to a channel barrier.
    A stage's byte total is compared only when every barrier use of that stage was traced and
    the role issued no TMA copy the check could not attribute; otherwise the channel is marked
    unverified.
    """
    stages: dict[int, _Stage] = {}
    partial_chans: dict[str, str] = {}  # channel -> why its per-stage totals may be partial
    opted_out_chans: set[str] = set()  # end.full_barrier() taken inside unchecked("tma")
    reached = []
    for use in trace.tma_bars:
        report = reports.setdefault(use.chan.name, TmaReport())
        stage = None
        if use.handle is not None:
            stage = stages.setdefault(id(use.handle), _Stage(use))
        copies, ops, problem = use.result or _copies(use)
        reached.extend(ops)
        if use.opted_out:
            report.opted_out = True
            if stage is not None:
                stage.opted_out = True
            else:
                opted_out_chans.add(use.chan.name)
            continue
        if problem is None and not copies:
            problem = f"barrier taken at {use.site} reaches no recognized TMA copy"
        if problem is not None:
            report.unverified.append(problem)
            if stage is not None:
                stage.incomplete.append(problem)
            else:
                partial_chans.setdefault(use.chan.name, problem)
            continue
        need = required_group(use.chan, cluster)
        modes = use.chan.producer.multicast
        for cp in copies:
            if need is not None and cp.group != need:
                raise ProtocolError(_group_message(trace.role_name, use, cp, need, cluster))
            if cp.multicast is not None and (not modes or cp.multicast not in (*modes, "?")):
                raise ProtocolError(_multicast_message(trace.role_name, use, cp))
            if cp.multicast == "?":
                report.unverified.append(
                    f"multicast copy at {cp.site} takes a mask not from Cluster.multicast"
                )
            report.groups.add(cp.group)
            report.copies.add(cp.site)
        if stage is not None:
            stage.copies.extend(copies)
        else:  # end.full_barrier(i): stage unknown, so totals of every stage may be partial
            partial_chans.setdefault(use.chan.name, f"end.full_barrier() copy at {use.site}")
    untracked = _untracked_tma_copies(trace.root, reached)
    for stage in stages.values():
        use, rec = stage.use, stage.use.handle
        report = reports[use.chan.name]
        why = list(stage.incomplete)
        if use.chan.name in partial_chans:
            why.append(partial_chans[use.chan.name])
        if untracked:
            why.append(untracked)
        if any(cp.nbytes is None or not cp.in_stage_region for cp in stage.copies):
            why.append("a copy sits in a nested loop/branch or has a dynamic layout")
        if why:
            report.unverified.append(f"bytes of {rec.what} at {rec.site}: {why[0]}")
        if why or stage.opted_out or use.chan.name in opted_out_chans:
            continue
        got, want = sum(cp.nbytes for cp in stage.copies), expected_stage_bytes(use.chan)
        if got != want:
            raise ProtocolError(_bytes_message(trace.role_name, use, stage.copies, got, want))
        report.stage_bytes.add(got)


def _copies(use: TmaUse) -> tuple[list[_Copy], list, str | None]:
    """The TMA copies whose barrier is ``use.value``, the ``set_value`` ops that name it, or a
    reason they cannot be analyzed."""
    copies, ops = [], []
    try:
        for operand in use.value.uses:
            op = operand.owner.operation
            if not _is_tma_bar(op):
                continue
            ops.append(op)
            kind = _kind(op)
            if kind is None:
                return [], ops, f"unrecognized TMA atom {str(op.results[0].type)[:60]!r}"
            group, multicast = kind
            copy_bits = re.search(r"copy_bits = (\d+)", str(op.results[0].type))
            for copy_op in _copy_ops(op):
                nbytes = _copy_bytes(copy_op, copy_bits)
                in_stage = use.handle is not None and copy_op.block == use.handle.block
                mode = _mask_mode(copy_op) if multicast else None
                copies.append(_Copy(use.site, group, nbytes, in_stage, mode))
    except (AttributeError, KeyError, IndexError, ValueError, RuntimeError) as e:
        return [], ops, f"cannot inspect the IR at {use.site}: {type(e).__name__}: {e}"
    return copies, ops, None


def _is_tma_bar(op) -> bool:
    return op.name == SET_VALUE and "<tma_bar>" in str(op.attributes["field"])


def _untracked_tma_copies(root, reached: list) -> str | None:
    """A reason when the role issues TMA copies whose barrier the layer did not hand out
    (their bytes may complete on a layer channel), else None. Walks op names and attributes
    only."""
    try:
        found = []

        def walk(block):
            for op in block.operations:
                op = op.operation
                if _is_tma_bar(op):
                    found.append(op)
                for region in op.regions:
                    for b in region.blocks:
                        walk(b)

        walk(root)
        extra = sum(1 for op in found if not any(op == r for r in reached))
    except (AttributeError, KeyError, ValueError, RuntimeError) as e:
        return f"cannot scan the role's TMA copies: {type(e).__name__}"
    return f"{extra} TMA copy(ies) use a barrier the layer did not hand out" if extra else None


def _kind(set_value) -> tuple[str, bool] | None:
    """(CTA group, multicast) of the atom an exec-atom ``set_value`` chain starts from."""
    op = set_value
    while op.name == SET_VALUE:  # earlier fields set on the same exec atom
        op = op.operands[0].owner.operation  # atom-typed values have no DSL caster
    if op.name != MAKE_EXEC:
        return None
    kind = re.match(r"!cute_nvgpu\.atom\.non_exec_tiled_tma_load<(\w+),", str(op.operands[0].type))
    return KINDS.get(kind.group(1)) if kind else None


def _mask_mode(copy_op) -> str:
    """The cluster mode of the multicast mask set on a copy's atom, or "?" when the mask is not
    one ``Cluster.multicast`` handed out."""
    op = copy_op.operands[0].owner.operation
    while op.name == SET_VALUE:
        if "<mcast_mask>" in str(op.attributes["field"]):
            mode, _ = _MASKS.get(hash(op.operands[1]), ("?", 0))
            return mode
        op = op.operands[0].owner.operation
    return "?"


def _copy_ops(set_value) -> list:
    """``cute.copy`` ops consuming this exec atom, through any later ``set_value`` fields."""
    out, frontier = [], [set_value]
    while frontier:
        op = frontier.pop()
        for operand in op.results[0].uses:
            user = operand.owner.operation
            if user.name == COPY:
                out.append(user)
            elif user.name == SET_VALUE:
                frontier.append(user)
    return out


def _copy_bytes(copy_op, copy_bits) -> int | None:
    """Bytes one ``cute.copy`` delivers: every element of its static SMEM destination, which
    must be a whole number of ``copy_bits`` boxes. None when the type is not recognized."""
    if copy_bits is None:
        return None
    text = str(copy_op)
    dst = re.search(r'!cute\.memref<([a-z]+)(\d+)\w*, smem[^"]*"([^"]*)">\) -> \(\)\s*$', text)
    if "operandSegmentSizes = array<i32: 1, 1, 1, 0>" not in text or dst is None:
        return None
    shape = _parse_shape(dst.group(3).split(":")[0])
    if shape is None:
        return None
    bits = int(dst.group(2))
    for leaf in _leaves(shape):
        bits *= leaf
    box = int(copy_bits.group(1))
    return bits // 8 if bits % box == 0 and bits % 8 == 0 else None


def _parse_shape(text: str):
    """Nested tuple of ints for a CuTe shape string; None if any extent is dynamic."""
    if "?" in text:
        return None
    tokens = re.findall(r"\(|\)|\d+", text)
    pos = 0

    def parse():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok != "(":
            return int(tok)
        items = []
        while tokens[pos] != ")":
            items.append(parse())
        pos += 1
        return tuple(items)

    try:
        return parse()
    except (IndexError, ValueError):
        return None


def _leaves(shape):
    if isinstance(shape, int):
        yield shape
        return
    for s in shape:
        yield from _leaves(s)


def _group_message(role: str, use: TmaUse, cp: _Copy, need: str, cluster) -> str:
    chan = use.chan
    head = (
        f"{role}: the TMA copy at {cp.site} completes on channel {chan.name}'s full barrier with "
        f"a CtaGroup.{cp.group} atom, but {chan.name} needs CtaGroup.{need}: "
    )
    if cluster.size == 1:
        why = "a 2-CTA load in a single-CTA kernel completes on a peer CTA's barrier"
    elif need == "TWO":
        why = (
            f"{chan.name} is pair-scoped, so its full barrier lives in the leader, which arms "
            f"{chan.scope.tx} bytes for both CTAs' copies; a CtaGroup.ONE copy in the peer "
            "completes on the peer's own barrier, which nobody waits on, and the leader hangs"
        )
    else:
        why = (
            f"{chan.name} is CTA-local (each CTA arms its own barrier with {chan.scope.tx} bytes); "
            "a CtaGroup.TWO copy from the peer completes on the leader's barrier instead"
        )
    if need == "TWO" or cluster.size == 1:
        fix = "build the atom's op with self.cluster.cta_group"
    else:
        fix = "build the atom's op with CtaGroup.ONE"
    return f"{head}{why}; {fix}"


def _multicast_message(role: str, use: TmaUse, cp: _Copy) -> str:
    chan = use.chan
    modes = chan.producer.multicast
    along = "" if cp.multicast == "?" else f" along {cp.multicast}"
    declared = f"declares multicast={modes}" if modes else "declares no multicast"
    return (
        f"{role}: the TMA copy at {cp.site} is multicast{along} into channel {chan.name}, but "
        f"its Tma {declared}: consumers in the CTAs the copy reaches would not release this "
        f"CTA's stage, so it could be overwritten while they read it; declare "
        f"Tma(..., multicast=...) with the copy's mode"
    )


def _bytes_message(role: str, use: TmaUse, copies: list, got: int, want: int) -> str:
    chan, rec = use.chan, use.handle
    sites = ", ".join(dict.fromkeys(cp.site for cp in copies))
    effect = "never completes (hang)" if got < want else "completes before every copy lands"
    per_cta = " per CTA" if chan.scope.pair else ""
    return (
        f"{role}: {rec.what} opened at {rec.site}: the TMA copies at {sites} deliver {got} "
        f"bytes{per_cta}, but the layer arms {want}{per_cta} (channel {chan.name}, "
        f"Tma({chan.producer.tx_bytes})), so the stage's full barrier {effect}; declare "
        f"Tma({got}) or fix the copies"
    )
