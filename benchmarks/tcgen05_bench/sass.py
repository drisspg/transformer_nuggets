"""SASS hot-loop checks for tcgen05 probes.

The probe's repeated region is the innermost backward-branch loop that contains the target
opcode. A clean probe has the target instruction there and no foreign memory, barrier, or other
tcgen05 opcodes; scalar loop control and descriptor arithmetic are allowed.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

LABEL = re.compile(r"^\s*(\.L_x_\d+):")
INSTRUCTION = re.compile(
    r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Za-z0-9_]+)*)(.*?);"
)
BRANCH_TARGET = re.compile(r"`\((\.L_x_\d+)\)")

# Expected target opcode per probe family/op (regex on the full dotted opcode).
TARGET_OPCODES = {
    "mma": r"^UTC\w*MMA",
    "ld": r"^LDTM",
    "st": r"^STTM",
    "cp": r"^UTC\w*CP",
    "shift": r"^UTC\w*SHIFT",
    "commit": r"^UTC\w*BAR",
    "alloc": r"^UTCATOM",
    "fence": r"^(FENCE|MEMBAR|UTC)",
}
# Opcode stems that indicate work other than the measured instruction.
FOREIGN = re.compile(
    r"^(LD|ST|LDS|STS|LDG|STG|LDL|STL|ATOM|RED|BAR|MEMBAR|FENCE|UTC|LDTM|STTM|SYNCS|CCTL|TMA)"
)
# Completion or ordering opcodes that a measured body legitimately contains.
ALLOWED_COMPANIONS = {
    "ld": re.compile(r"^(DEPBAR|LDTM|FENCE\.VIEW\.ASYNC\.T)"),
    "st": re.compile(r"^(DEPBAR|STTM|FENCE\.VIEW\.ASYNC\.T)"),
    "cp": re.compile(r"^(UTC\w*BAR|SYNCS)"),
    "shift": re.compile(r"^(UTC\w*BAR|SYNCS)"),
    "commit": re.compile(r"^(UTC\w*BAR|SYNCS)"),
    "alloc": re.compile(r"^(UTC|LDS|SYNCS|BAR|FENCE)"),
    "mma": re.compile(r"^(UTC\w*BAR|SYNCS)"),
    "fence": re.compile(r"."),
}


@dataclass
class LoopReport:
    """Opcode histogram of the probe's repeated region."""

    found: bool
    target_opcodes: dict[str, int] = field(default_factory=dict)
    foreign_opcodes: dict[str, int] = field(default_factory=dict)
    loop_opcodes: dict[str, int] = field(default_factory=dict)
    loop_instructions: int = 0
    expected_targets: int | None = None
    clean: bool = False

    def to_json(self) -> dict:
        """Return a JSON-serializable dict."""
        return asdict(self)


def parse_sass(text: str) -> tuple[list[tuple[int, str, str]], dict[str, int]]:
    """Return ``(instructions, labels)`` where labels map to instruction indices."""
    instructions: list[tuple[int, str, str]] = []
    labels: dict[str, int] = {}
    for line in text.splitlines():
        label = LABEL.match(line)
        if label:
            labels[label.group(1)] = len(instructions)
            continue
        match = INSTRUCTION.search(line)
        if match:
            instructions.append((int(match.group(1), 16), match.group(3), match.group(4)))
    return instructions, labels


def hot_loop(text: str, op: str, expected_targets: int | None = None) -> LoopReport:
    """Find the innermost loop containing the target opcode and classify its opcodes.

    ``expected_targets`` is the number of target instructions one loop trip must contain; a
    mismatch (for example ptxas dropping dead loads) marks the probe unclean.
    """
    instructions, labels = parse_sass(text)
    target = re.compile(TARGET_OPCODES[op])
    loops = []
    for index, (_, opcode, operands) in enumerate(instructions):
        if not opcode.startswith("BRA"):
            continue
        label = BRANCH_TARGET.search(operands)
        if label and labels.get(label.group(1), index + 1) <= index:
            loops.append((labels[label.group(1)], index))
    candidates = [
        (begin, end)
        for begin, end in loops
        if any(target.search(instructions[i][1]) for i in range(begin, end + 1))
    ]
    if not candidates:
        return LoopReport(found=False)
    begin, end = min(candidates, key=lambda loop: loop[1] - loop[0])
    opcodes = Counter(instructions[i][1] for i in range(begin, end + 1))
    targets = {name: count for name, count in opcodes.items() if target.search(name)}
    companion = ALLOWED_COMPANIONS[op]
    foreign = {
        name: count
        for name, count in opcodes.items()
        if name not in targets and FOREIGN.search(name) and not companion.search(name)
    }
    return LoopReport(
        found=True,
        target_opcodes=targets,
        foreign_opcodes=foreign,
        loop_opcodes=dict(opcodes),
        loop_instructions=end - begin + 1,
        expected_targets=expected_targets,
        clean=bool(targets)
        and not foreign
        and (expected_targets is None or sum(targets.values()) == expected_targets),
    )


def disassemble(cubin: Path) -> str:
    """Disassemble a cubin with nvdisasm (labels and branch targets preserved)."""
    return subprocess.run(
        ["nvdisasm", "-c", str(cubin)], check=True, capture_output=True, text=True
    ).stdout


def find_cubin(dump_dir: Path, name: str) -> Path | None:
    """Return the newest cubin in ``dump_dir`` whose file name contains ``name``."""
    matches = sorted(dump_dir.rglob(f"*{name}*.cubin"), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None
