"""Trace-time protocol bookkeeping: handle lifetimes, the protocol log, per-site check opt-outs,
and errors."""

from __future__ import annotations

import contextlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

from cutlass._mlir import ir

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTEXTLIB = os.path.abspath(contextlib.__file__)  # frames of @contextmanager wrappers


class ProtocolError(Exception):
    """A role protocol violation detected while declaring or tracing a kernel."""


# Checks a role can opt out of at one site (``with unchecked(check, reason=...)``):
#   unclosed      handles opened inside the block may stay open at role exit
#   cross_region  closes inside the block may sit in a different loop/branch than their open
#   tma           TMA copies into barriers taken inside the block skip the atom and byte checks
CHECKS = ("unclosed", "cross_region", "tma")


@dataclass
class OptOut:
    """One ``unchecked(...)`` block: which check, why, where, and what it let through."""

    check: str
    reason: str
    site: str
    role: str | None = None
    suppressed: list = field(default_factory=list)

    def describe(self) -> str:
        what = "; ".join(sorted(set(self.suppressed))) or "nothing (stale opt-out?)"
        who = f"{self.role}: " if self.role else ""
        return f"unchecked {self.check} @ {self.site}: {who}{self.reason!r} -> {what}"


_ACTIVE: list[OptOut] = []  # innermost last; tracing is single-threaded
_SEEN: list[OptOut] = []  # every opt-out entered since the last take_opt_outs()


@contextmanager
def unchecked(check: str, *, reason: str):
    """Turn off one trace-time check for the code inside the block. ``reason`` is required;
    ``describe()`` lists the opt-out with its source line and what it let through. Every other
    check stays on, here and elsewhere. Use it for an ordering argument the checker cannot see,
    for example a handle deliberately closed in the next loop iteration."""
    if check not in CHECKS:
        raise ValueError(f"unknown check {check!r}; one of {CHECKS}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("unchecked(...) needs a non-empty reason")
    opt = OptOut(check, reason, user_site())
    _SEEN.append(opt)
    _ACTIVE.append(opt)
    try:
        yield opt
    finally:
        _ACTIVE.remove(opt)


def active_opt_out(check: str) -> OptOut | None:
    return next((o for o in reversed(_ACTIVE) if o.check == check), None)


def take_opt_outs() -> list[OptOut]:
    """Opt-outs entered since the last call (the kernel collects them once per trace)."""
    seen = list(_SEEN)
    _SEEN.clear()
    return seen


def user_site() -> str:
    """``file:line`` of the innermost caller outside this package."""
    frame = sys._getframe(1)
    while frame is not None:
        path = os.path.abspath(frame.f_code.co_filename)
        if os.path.dirname(path) != _PACKAGE_DIR and path != _CONTEXTLIB:
            return f"{os.path.basename(path)}:{frame.f_lineno}"
        frame = frame.f_back
    return "?"


def current_block():
    return ir.InsertionPoint.current.block


def _block_depth(block, root) -> int | None:
    depth = 0
    try:
        while block is not None:
            if block == root:
                return depth
            block = block.owner.operation.block
            depth += 1
    except (AttributeError, ValueError, RuntimeError):
        return None
    return None


def _ancestors(block, root) -> list:
    """``block`` and its enclosing blocks up to ``root`` (innermost first)."""
    chain = []
    try:
        while block is not None:
            chain.append(block)
            if block == root:
                break
            block = block.owner.operation.block
    except (AttributeError, ValueError, RuntimeError):
        pass
    return chain


class ProtocolLog:
    """Per role instance: the static sequence of channel ops recorded while tracing.

    Dynamic loops are traced once, so an op inside a loop appears once. ``depth`` is the MLIR
    block nesting depth relative to the role body (0 = role body, 1 = inside one loop or if).
    """

    def __init__(self):
        self.ops: dict[str, list[tuple]] = {}
        self.tmem: dict[str, set] = {}

    def record(self, role_name, chan, op, slot, depth, site):
        self.ops.setdefault(role_name, []).append((chan, op, slot, depth, site))

    def describe(self) -> str:
        lines = []
        for role_name, ops in self.ops.items():
            lines.append(f"role {role_name}:")
            for chan, op, slot, depth, site in ops:
                where = "" if slot is None else f"[{slot}]"
                lines.append(f"  {'  ' * depth}{chan + where:<9} {op:<8} @ {site}")
        if self.tmem:
            lines.append("tmem access:")
            for role_name, regions in self.tmem.items():
                lines.append(f"  {role_name}: {', '.join(sorted(regions))}")
        return "\n".join(lines)


class RoleTrace:
    """Open-handle bookkeeping for one traced role instance."""

    def __init__(
        self, role_name: str, log: ProtocolLog | None, iket: bool = False, iket_waits: bool = True
    ):
        self.role_name = role_name
        self.log = log
        self.iket = iket  # emit IKET work-item/role ranges (tracing.py)
        self.iket_waits = iket and iket_waits  # and ranges around blocking waits
        self.ops: list = []  # (channel, op, enclosing blocks innermost first) per recorded op
        self.item_block = None  # the work-item loop body (set by the TileCursor's advance())
        self.open: list = []
        self.checks: list = []  # end-of-role trace checks (e.g. tile cursors)
        self.tails: list = []  # producer ends the layer drains at role exit (pair channels)
        self.tma_bars: list = []  # TMA-channel full-barrier pointers handed out (tma.py)
        self.root = current_block()

    def opt_out(self, check: str, what: str) -> bool:
        """True (and recorded) when an ``unchecked(check)`` block is active here."""
        opt = active_opt_out(check)
        if opt is None:
            return False
        opt.role = opt.role or self.role_name
        opt.suppressed.append(what)
        return True

    def depth(self) -> int:
        d = _block_depth(current_block(), self.root)
        return 0 if d is None else d

    def record(self, chan, op, slot):
        # Blocks are captured while live: a region the tracer later discards must not be
        # walked afterwards (dereferencing an erased block crashes).
        self.ops.append((chan, op, _ancestors(current_block(), self.root)))
        if self.log is not None:
            self.log.record(self.role_name, chan, op, slot, self.depth(), user_site())

    def uses_per_item(self) -> dict:
        """``{channel: n}``: acquires/waits per work item, 0 when inside an inner loop."""
        uses: dict = {}
        if self.item_block is None:
            return uses
        for chan, op, blocks in self.ops:
            if op not in ("acquire", "wait") or self.item_block not in blocks:
                continue
            depth = blocks.index(self.item_block)
            if depth == 0 and uses.get(chan) != 0:
                uses[chan] = uses.get(chan, 0) + 1
            elif depth:  # nested inside the item loop: a dynamic count per item
                uses[chan] = 0
        return uses

    def opened(self, handle):
        if active_opt_out("unclosed") is not None:
            handle.unclosed_ok = active_opt_out("unclosed")
        self.open.append(handle)

    def closed(self, handle, op: str):
        site = user_site()
        if handle.closed_at is not None:
            raise ProtocolError(
                f"{self.role_name}: {handle.what} closed twice: {handle.closed_op} at "
                f"{handle.closed_at}, then {op} at {site}"
            )
        cross = current_block() != handle.block
        if cross and not self.opt_out("cross_region", f"{handle.what} {op} at {site}"):
            raise ProtocolError(
                f"{self.role_name}: {handle.what} opened at {handle.site} but {op} at "
                f"{site} is in a different dynamic region (loop or branch), so it would run a "
                "different number of times than the acquire; close handles in the region "
                "that opened them"
            )
        handle.closed_at, handle.closed_op = site, op
        self.open.remove(handle)

    def finish(self):
        for check in self.checks:
            check()
        leaks = []
        for h in self.open:
            if h.unclosed_ok is None:
                leaks.append(h)
                continue
            h.unclosed_ok.role = h.unclosed_ok.role or self.role_name
            h.unclosed_ok.suppressed.append(f"{h.what} opened at {h.site} left open")
        if leaks:
            items = "; ".join(f"{h.what} opened at {h.site}" for h in leaks)
            raise ProtocolError(f"{self.role_name}: handles never closed: {items}")
