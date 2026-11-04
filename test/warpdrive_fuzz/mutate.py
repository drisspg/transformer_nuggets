"""Compile-only source mutations of locally valid channel and tile-loop templates.

No compiled callable is ever invoked. Source files are retained under agent_space so a
finding is independently reproducible; an isolated process should bound long campaigns.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cutlass
from cutlass import cute

from test.warpdrive_fuzz.static import Finding, FuzzReport
from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import ProtocolError


@dataclass(frozen=True)
class Mutation:
    name: str
    contract: str
    body: str
    kind: str = "channel"
    error: type[Exception] | None = ProtocolError
    message: str = ""
    obligation: bool = False


CHANNEL_TEMPLATE = """\
import cutlass
from cutlass import cute
from types import SimpleNamespace
from transformer_nuggets.cute.warpdrive import (
    WarpSpecialized, Ring, Slots, Threads, Producer, Consumer, Instance, role, unchecked,
)

class Probe(WarpSpecialized):
    def __init__(self):
        self.ch = DECLARATION

    @cute.jit
    def step(self, ch):
        h = ch.acquire(SLOT)
        h.commit()

    @role(warps=0)
    @cute.jit
    def prod(self, ch: PRODUCER_ANNOTATION, ctx):
PRODUCER_BODY

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        for i in cutlass.range(ctx.n, unroll=1):
            h = ch.wait(SLOT)
            h.release()

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix("NAME")
        self.kernel(out, n).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)
"""

TILES_TEMPLATE = """\
import cutlass
from cutlass import cute
from types import SimpleNamespace
from transformer_nuggets.cute.warpdrive import (
    WarpSpecialized, TileStream, StaticPersistent, Tiles, role,
)

class Probe(WarpSpecialized):
    def __init__(self):
        self.tiles = TileStream(policy=StaticPersistent())

    @role(warps=0)
    @cute.jit
    def worker(self, tiles: Tiles, ctx):
BODY

    @cute.kernel
    def kernel(self, out: cute.Tensor, work):
        self.begin_roles(tiles=work)
        self.run_roles(SimpleNamespace(out=out, n=work.total))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream):
        work = self.tiles.problem((n, 1, 1), max_ctas=cutlass.Int32(1))
        self.kernel.set_name_prefix("NAME")
        self.kernel(out, work).launch(**self.launch_config(work), stream=stream)
"""


def mutations() -> list[Mutation]:
    loop = "for i in cutlass.range(ctx.n, unroll=1):\n"
    valid = loop + "    h = ch.acquire(SLOT)\n    h.commit()"
    return [
        Mutation("valid", "R2", valid, error=None),
        Mutation(
            "drop_commit", "E58", loop + "    h = ch.acquire(SLOT)", message="handles never closed"
        ),
        Mutation("double_commit", "E56", valid + "\n    h.commit()", message="closed twice"),
        Mutation(
            "wrong_close", "E55", valid.replace("h.commit()", "h.release()"), message="release()"
        ),
        Mutation(
            "cross_loop",
            "E57",
            "h = ch.acquire(SLOT)\n" + loop + "    h.commit()",
            message="different dynamic region",
        ),
        Mutation(
            "cross_branch",
            "E57",
            loop + "    h = ch.acquire(SLOT)\n    if ctx.n > 0:\n        h.commit()",
            message="different dynamic region",
        ),
        Mutation(
            "with_double_close",
            "E56",
            loop + "    with ch.produce(SLOT) as h:\n        h.commit()",
            message="closed twice",
        ),
        Mutation(
            "wrong_bound_slot",
            "E62",
            valid.replace("SLOT", "1"),
            kind="bound",
            message="bound to slot 0",
        ),
        Mutation(
            "ring_slot", "E59", valid.replace("SLOT", "0"), kind="ring", message="takes no slot"
        ),
        Mutation(
            "dynamic_slot",
            "E60",
            valid.replace("SLOT", "i"),
            kind="slots",
            message="Python int slot",
        ),
        Mutation(
            "negative_slot", "E61", valid.replace("SLOT", "-1"), kind="slots", message="got -1"
        ),
        Mutation(
            "out_of_range_slot", "E61", valid.replace("SLOT", "31"), kind="slots", message="got 31"
        ),
        Mutation(
            "helper_discards_end", "U10", loop + "    self.step(ch)", error=None, obligation=True
        ),
        Mutation(
            "unbalanced_epochs",
            "U1",
            valid.replace("ctx.n,", "ctx.n + 1,"),
            error=None,
            obligation=True,
        ),
        Mutation(
            "lane_predicated_pair",
            "U1",
            "if cute.arch.lane_idx() == 0:\n" + textwrap.indent(valid, "    "),
            error=None,
            obligation=True,
        ),
        Mutation(
            "unchecked_no_reason",
            "E77",
            'with unchecked("unclosed", reason=""):\n    ch.acquire(SLOT)',
            error=ValueError,
            message="non-empty reason",
        ),
        Mutation(
            "missing_advance",
            "E71",
            "m, n, l = tiles.work",
            kind="tiles",
            message="never calls advance",
        ),
        Mutation(
            "double_advance",
            "E72",
            "while tiles.valid:\n    tiles.advance()\n    tiles.advance()",
            kind="tiles",
            message="traced 2 times",
        ),
        Mutation(
            "conditional_advance",
            "E73",
            "while tiles.valid:\n    if ctx.n > 0:\n        tiles.advance()",
            kind="tiles",
            message="directly in that loop",
        ),
        Mutation(
            "nested_loop_advance",
            "E73",
            "while tiles.valid:\n    for i in cutlass.range(ctx.n, unroll=1):\n        tiles.advance()",
            kind="tiles",
            message="directly in that loop",
        ),
        Mutation(
            "compound_condition",
            "G1",
            "count = cutlass.Int32(0)\nwhile tiles.valid & (count < ctx.n):\n    count += 1\n    tiles.advance()",
            kind="tiles",
            message="condition",
        ),
        Mutation(
            "advance_before_work",
            "G1",
            "while tiles.valid:\n    tiles.advance()\n    m, n, l = tiles.work\n    ctx.out[0] = m",
            kind="tiles",
            message="after advance",
        ),
        Mutation(
            "valid_tiles",
            "R12",
            "while tiles.valid:\n    m, n, l = tiles.work\n    ctx.out[0] = m\n    tiles.advance()",
            kind="tiles",
            error=None,
        ),
    ]


def source_for(mutation: Mutation, seed: int, depth: int = 2) -> str:
    name = f"warpdrive_mutation_{seed}_{mutation.name}"
    if mutation.kind == "tiles":
        return TILES_TEMPLATE.replace("BODY", textwrap.indent(mutation.body, "        ")).replace(
            "NAME", name
        )
    fixed = mutation.kind in ("slots", "bound")
    if mutation.kind == "bound":
        depth = 2
    declaration = (
        f"Slots({depth}, Threads, Threads)" if fixed else f"Ring({depth}, Threads, Threads)"
    )
    body = mutation.body.replace("SLOT", "0" if fixed else "")
    source = CHANNEL_TEMPLATE.replace("PRODUCER_BODY", textwrap.indent(body, "        "))
    if mutation.kind == "bound":
        source = source.replace("@role(warps=0)", "@role(warps=[range(1), range(1, 2)])")
        source = source.replace("@role(warps=1)", "@role(warps=2)")
    return (
        source.replace("DECLARATION", declaration)
        .replace(
            "PRODUCER_ANNOTATION", "Producer[Instance]" if mutation.kind == "bound" else "Producer"
        )
        .replace("SLOT", "0" if fixed else "")
        .replace("NAME", name)
    )


def compile_mutation(mutation: Mutation, seed: int, depth: int = 2):
    """Compile fresh source without a runtime tensor, device query, or launch invocation."""
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
    source = source_for(mutation, seed, depth)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    root = Path(__file__).resolve().parents[2] / "agent_space" / "warpdrive" / "fuzz" / "mutations"
    root.mkdir(parents=True, exist_ok=True)
    source_file = root / f"mutation_{digest}.py"
    if not source_file.exists():
        source_file.write_text(source)
    module_name = f"warpdrive_mutation_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
        return compile_tvm_ffi(module.Probe(), out, cutlass.Int32(1), name=module_name)
    finally:
        del sys.modules[module_name]


def is_expected_error(exc: Exception, error: type[Exception], message: str) -> bool:
    seen = set()
    pending = [exc]
    while pending:
        cause = pending.pop()
        if id(cause) in seen:
            continue
        seen.add(id(cause))
        if isinstance(cause, error) and message in str(cause):
            return True
        pending.extend(e for e in (cause.__cause__, cause.__context__) if e is not None)
    return False


def run_mutation_fuzz(seed: int, budget: int) -> FuzzReport:
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = random.Random(seed)
    corpus = mutations()
    rng.shuffle(corpus)
    report = FuzzReport(seed)
    for i in range(budget):
        mutation = corpus[i % len(corpus)]
        depth = rng.choice((2, 3, 4))
        report.cases += 1
        try:
            compile_mutation(mutation, seed + i, depth)
        except Exception as exc:  # noqa: BLE001 — compiler diagnostics are fuzz outcomes.
            if mutation.error is not None and is_expected_error(
                exc, mutation.error, mutation.message
            ):
                report.rejected += 1
            else:
                report.findings.append(
                    Finding(
                        seed,
                        mutation.name,
                        mutation.contract,
                        "unexpected exception",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        else:
            if mutation.error is not None:
                report.findings.append(
                    Finding(
                        seed,
                        mutation.name,
                        mutation.contract,
                        "missing rejection",
                        f"expected {mutation.error.__name__}: {mutation.message}",
                    )
                )
            elif mutation.obligation:
                report.obligations += 1
            else:
                report.accepted += 1
    return report
