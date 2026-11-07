"""The Qwen3 megakernel's static schedule, drawn per CTA (``schedule_html``).

``build_table`` assigns the decode step's instructions to CTAs with a list schedule over a
latency model (``assign_static``: ``ROW_COST`` per row, ``DEPENDENCY_LATENCY`` per producer
group) and lays them out ``[slot][cta]`` for ``StaticPersistent``. This replays that model to
draw, for each CTA, its rows coloured by op and shaded by layer, their predicted start and
end, the producer rows each row waits for (the last row of each producer group: the one the
model says gates it), and the predicted critical path. ``WorkStream.assignment`` maps the
table rows back to CTAs the way the kernel runs them.

A capture can be overlaid (``--timeline``): a JSON file
``{"unit": "us", "ctas": 152, "splits": 16, "spans": [[table_row, cta, start, end], ...]}``
(``ctas`` and ``splits`` optional, checked against the drawn table), times from any common
origin, for example the ``item`` and ``done`` stamps of a ``timeline=True`` run converted to
microseconds.

    python -m transformer_nuggets.cute.warpdrive.examples.qwen_megakernel_viz OUT.html \\
        [--ctas 152] [--splits 16] [--timeline capture.json]
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from transformer_nuggets.cute.warpdrive.examples import qwen_megakernel as mk
from transformer_nuggets.cute.warpdrive.schedule_html import Item, schedule_html

OPS = ("QKV", "ATTN", "OPROJ", "GATEUP", "DOWN", "LMHEAD")


def instruction_rows(op: mk.QwenDecodeMegakernel, ctas: int):
    """``(rows, table)``: the ``(words, producer groups, completed group)`` rows, in
    dependency order, that ``build_table`` hands ``assign_static``, and the table it returns.
    ``build_table`` keeps the rows local, so this records the call."""
    rows = []

    def record(given, n):
        rows.extend(given)
        return assign_static(given, n)

    assign_static = mk.assign_static
    with mock.patch.object(mk, "assign_static", record):
        table = mk.build_table(op, ctas)
    return rows, table


@dataclass
class Predicted:
    """One row as the latency model schedules it."""

    cta: int
    start: float
    end: float
    gates: tuple[int, ...]  # the last row of each producer group it waits for
    after: int | None  # the row that set its start (a gate, or its CTA's previous row)


def predict(rows, ctas: int) -> list[Predicted]:
    """Replay ``assign_static``'s list schedule, recording each row's span and gates."""
    last: dict[tuple, int] = {}  # producer group -> its last-finishing row
    finished: dict[tuple, float] = {}
    free = [0.0] * ctas
    tail: list[int | None] = [None] * ctas  # each CTA's previous row
    out = []
    for i, (words, deps, group) in enumerate(rows):
        known = [d for d in deps if d in finished]
        ready = max((finished[d] + mk.DEPENDENCY_LATENCY for d in known), default=0.0)
        cta = min(range(ctas), key=lambda c: (max(free[c], ready), c))
        start = max(free[cta], ready)
        gate = max(known, key=finished.__getitem__, default=None)
        after = last[gate] if gate is not None and ready >= free[cta] else tail[cta]
        out.append(
            Predicted(
                cta, start, start + mk.ROW_COST[words[0]], tuple(last[d] for d in known), after
            )
        )
        free[cta], tail[cta] = out[-1].end, i
        if out[-1].end >= finished.get(group, 0.0):
            finished[group], last[group] = out[-1].end, i
    return out


def critical_path(pred: list[Predicted]) -> list[int]:
    """Rows from the start to the last-finishing row, each the one that set the next's start."""
    i = max(range(len(pred)), key=lambda r: pred[r].end)
    path = []
    while i is not None:
        path.append(i)
        i = pred[i].after
    return path[::-1]


def _label(words, deps, table_row: int) -> str:
    kind, layer, a, b = words
    match kind:
        case mk.QKV:
            what = f"QKV layer {layer} · rows {a * mk.ROWS}-{a * mk.ROWS + mk.ROWS - 1}"
        case mk.ATTN:
            what = f"ATTN layer {layer} · kv head {a}, split {b}"
        case mk.OPROJ | mk.DOWN:
            what = f"{OPS[kind]} layer {layer} · rows {a * mk.ROWS}.. · k split {b}"
        case mk.GATEUP:
            what = f"GATEUP layer {layer} · tile {a}"
        case _:
            what = f"LMHEAD · row tiles {a}-{a + b - 1}"
    waits = ", ".join(" ".join(map(str, d)) for d in deps) or "nothing"
    return f"{what}\nwaits for: {waits}\ntable row {table_row}"


def megakernel_view(ctas: int, splits: int = 16, measured: dict | None = None) -> str:
    op = mk.QwenDecodeMegakernel(splits=splits)
    rows, table = instruction_rows(op, ctas)
    pred = predict(rows, ctas)
    by_cta: list[list[int]] = [[] for _ in range(ctas)]
    for i, p in enumerate(pred):
        by_cta[p.cta].append(i)
    # The kernel's view: StaticPersistent over the [slot][cta] table, padding rows dropped.
    host = table.tolist()
    assigned = op.work.assignment(table, max_ctas=ctas, decode=lambda t: (host[t][0] >= 0, t))
    table_row = {}
    for cta, items in enumerate(assigned):
        assert len(items) == len(by_cta[cta]), "the replayed model disagrees with build_table"
        for (t, _), i in zip(items, by_cta[cta]):
            assert host[t][:4] == rows[i][0], "the replayed model disagrees with build_table"
            table_row[t] = i
    items = [
        Item(
            _label(words, deps, t),
            OPS[words[0]],
            shade=words[1] / (mk.LAYERS - 1) if words[0] != mk.LMHEAD else 0.0,
            start=pred[i].start,
            end=pred[i].end,
            deps=pred[i].gates,
        )
        for t, i in sorted(table_row.items(), key=lambda kv: kv[1])
        for words, deps, _ in [rows[i]]
    ]
    spans = None
    if measured is not None:
        spans = {table_row[t]: (cta, s, e) for t, cta, s, e in measured["spans"] if t in table_row}
    path = critical_path(pred)
    makespan = max(p.end for p in pred)
    return schedule_html(
        f"Qwen3-0.6B decode megakernel: {len(rows)} rows on {ctas} CTAs",
        f"{op.work.describe()} · splits {splits} · predicted makespan {makespan:.1f} us",
        items,
        by_cta,
        critical=path,
        measured=spans,
        kinds=OPS,
        notes=[
            "Predicted spans replay assign_static's list schedule: ROW_COST "
            + ", ".join(f"{OPS[k]} {c} us" for k, c in mk.ROW_COST.items())
            + f", plus {mk.DEPENDENCY_LATENCY} us per producer group. Illustrative, not "
            "measured.",
            "Arrows on hover: the last row of each producer group the row waits for (the one "
            "that gates it in the model). Critical path: rows that each set the next's start.",
            "Shade: layer (darker = earlier). CTA rows are WorkStream.assignment over the "
            "[slot][cta] table, as StaticPersistent runs them.",
            *(
                [f"Measured overlay: {len(spans)} spans from the --timeline capture."]
                if spans
                else []
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ctas", type=int, default=152, help="resident CTAs (GB300: 152 SMs)")
    parser.add_argument("--splits", type=int, default=16)
    parser.add_argument("--timeline", type=Path, help="measured spans (see the docstring)")
    args = parser.parse_args()
    measured = json.loads(args.timeline.read_text()) if args.timeline else None
    for key in ("ctas", "splits"):  # the capture's table must be the one drawn
        if measured and measured.get(key, getattr(args, key)) != getattr(args, key):
            parser.error(
                f"the capture ran with {key} = {measured[key]}; pass --{key} {measured[key]}"
            )
    args.out.write_text(megakernel_view(args.ctas, args.splits, measured))
    print(args.out)


if __name__ == "__main__":
    main()
