"""Trace-time protocol bookkeeping: handle lifetimes, the protocol log, and errors."""

from __future__ import annotations

import os
import sys

from cutlass._mlir import ir

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class ProtocolError(Exception):
    """A role protocol violation detected while declaring or tracing a kernel."""


def user_site() -> str:
    """``file:line`` of the innermost caller outside this package."""
    frame = sys._getframe(1)
    while frame is not None:
        path = os.path.abspath(frame.f_code.co_filename)
        if os.path.dirname(path) != _PACKAGE_DIR:
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

    def __init__(self, role_name: str, log: ProtocolLog | None):
        self.role_name = role_name
        self.log = log
        self.open: list = []
        self.checks: list = []  # end-of-role trace checks (e.g. tile cursors)
        self.root = current_block()

    def depth(self) -> int:
        d = _block_depth(current_block(), self.root)
        return 0 if d is None else d

    def record(self, chan, op, slot):
        if self.log is not None:
            self.log.record(self.role_name, chan, op, slot, self.depth(), user_site())

    def opened(self, handle):
        self.open.append(handle)

    def closed(self, handle, op: str):
        site = user_site()
        if handle.closed_at is not None:
            raise ProtocolError(
                f"{self.role_name}: {handle.what} closed twice: {handle.closed_op} at "
                f"{handle.closed_at}, then {op} at {site}"
            )
        if current_block() != handle.block:
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
        if self.open:
            items = "; ".join(f"{h.what} opened at {h.site}" for h in self.open)
            raise ProtocolError(f"{self.role_name}: handles never closed: {items}")
