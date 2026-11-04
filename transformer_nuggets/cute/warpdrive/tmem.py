"""Layer-owned tensor-memory plan: named per-slot column regions and aliases."""

from __future__ import annotations

from dataclasses import dataclass

from transformer_nuggets.cute.warpdrive.protocol import ProtocolError, ProtocolLog

TMEM_COLS = 512


@dataclass
class _Region:
    name: str
    cols: int
    slots: int
    offset: int = 0
    alias_of: str | None = None
    alias_offset: int = 0


class TmemPlan:
    """Named TMEM column regions; ``alias`` places a region inside each slot of another.

    Offsets are assigned in declaration order and aliases take no new columns. The layer
    allocates the smallest power-of-two column count (>= 32) covering the plan.
    """

    def __init__(self):
        self.regions: dict[str, _Region] = {}
        self._next = 0

    def region(self, name: str, cols: int, slots: int = 1) -> TmemPlan:
        if name in self.regions:
            raise ValueError(f"TMEM region {name!r} is already declared")
        if type(cols) is not int or type(slots) is not int or cols <= 0 or slots <= 0:
            raise ValueError(f"TMEM region {name}: cols and slots must be positive ints")
        self.regions[name] = _Region(name, cols, slots, offset=self._next)
        self._next += cols * slots
        return self

    def alias(self, name: str, of: str, offset: int, cols: int) -> TmemPlan:
        if name in self.regions:
            raise ValueError(f"TMEM region {name!r} is already declared")
        if of not in self.regions:
            raise ProtocolError(f"no TMEM region named {of!r}")
        if type(cols) is not int or cols <= 0:
            raise ValueError(f"TMEM alias {name}: cols must be a positive int")
        if type(offset) is not int:
            raise ValueError(f"TMEM alias {name}: offset must be a nonnegative int")
        parent = self.regions[of]
        if offset < 0 or offset + cols > parent.cols:
            raise ValueError(
                f"TMEM alias {name} [{offset}, {offset + cols}) exceeds {of} ({parent.cols} cols)"
            )
        self.regions[name] = _Region(name, cols, parent.slots, alias_of=of, alias_offset=offset)
        return self

    def col(self, name: str, slot: int = 0) -> int:
        if name not in self.regions:
            raise ProtocolError(f"no TMEM region named {name!r}")
        r = self.regions[name]
        if type(slot) is not int or not 0 <= slot < r.slots:
            raise ProtocolError(f"TMEM region {name} has {r.slots} slots; got slot {slot!r}")
        if r.alias_of is None:
            return r.offset + slot * r.cols
        return self.col(r.alias_of, slot) + r.alias_offset

    @property
    def used_cols(self) -> int:
        return self._next

    @property
    def alloc_cols(self) -> int:
        cols = 32
        while cols < self._next:
            cols *= 2
        if cols > TMEM_COLS:
            raise ValueError(f"TMEM plan needs {self._next} > {TMEM_COLS} columns")
        return cols

    def describe(self) -> str:
        lines = []
        for r in self.regions.values():
            spans = ", ".join(
                f"[{self.col(r.name, j)}, {self.col(r.name, j) + r.cols})" for j in range(r.slots)
            )
            alias = f" (inside {r.alias_of})" if r.alias_of else ""
            lines.append(f"  tmem {r.name:<3} {spans}{alias}")
        return "\n".join(lines)


class TmemView:
    """Device-side TMEM base pointer plus the plan; records which regions each role touches."""

    def __init__(self, base, plan: TmemPlan, role_name: str, log: ProtocolLog | None):
        self.base, self._plan, self._role, self._log = base, plan, role_name, log

    def cols(self, name: str, slot: int = 0) -> int:
        """Column offset of ``name[slot]`` relative to the allocation base."""
        col = self._plan.col(name, slot)
        if self._log is not None:
            self._log.tmem.setdefault(self._role, set()).add(f"{name}[{slot}]")
        return col

    def ptr(self, name: str, slot: int = 0):
        """FP32 TMEM pointer to ``name[slot]``."""
        return self.base + self.cols(name, slot)
