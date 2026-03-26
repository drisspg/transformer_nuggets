"""Merge per-rank Chrome/Perfetto traces into a single multi-process trace."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Merge per-rank Chrome/Perfetto traces into one file.")


def _open_trace(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _get_min_ts(events: list[dict]) -> float:
    return min(
        (ev["ts"] for ev in events if "ts" in ev and ev.get("ph") != "M"),
        default=0.0,
    )


def merge_traces(
    input_paths: list[str],
    output_path: str,
    labels: list[str] | None = None,
    align_timestamps: bool = False,
) -> None:
    merged_events: list[dict] = []

    for idx, path in enumerate(input_paths):
        with _open_trace(path, "r") as f:
            data = json.load(f)

        events = data.get("traceEvents", data) if isinstance(data, dict) else data

        ts_offset = _get_min_ts(events) if align_timestamps else 0.0
        label = labels[idx] if labels else f"Rank {idx}"

        merged_events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": idx,
                "tid": 0,
                "args": {"name": label},
            }
        )

        for ev in events:
            if ev.get("ph") == "M" and ev.get("name") == "process_name":
                continue
            ev["pid"] = idx
            if align_timestamps and "ts" in ev:
                ev["ts"] = ev["ts"] - ts_offset
            if "id" in ev and ev.get("ph") in ("s", "t", "f"):
                ev["id"] = ev["id"] + idx * (1 << 32)
            merged_events.append(ev)

    with _open_trace(output_path, "w") as f:
        json.dump({"traceEvents": merged_events}, f, indent=0)


@app.command()
def main(
    traces: Annotated[
        list[Path], typer.Argument(help="Input trace files, one per rank, in rank order.")
    ],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output path.")] = Path(
        "merged_trace.json.gz"
    ),
    label: Annotated[
        list[str] | None,
        typer.Option("-l", "--label", help="Label for each trace (repeat for each file)."),
    ] = None,
    align: Annotated[
        bool, typer.Option("--align", help="Align timestamps so all traces start at t=0.")
    ] = False,
):
    """Merge per-rank Chrome/Perfetto traces into a single multi-process Perfetto trace."""
    for p in traces:
        if not p.exists():
            typer.echo(f"Error: {p} not found", err=True)
            raise typer.Exit(1)

    merge_traces([str(p) for p in traces], str(output), labels=label, align_timestamps=align)
    typer.echo(f"Merged {len(traces)} traces -> {output}")


if __name__ == "__main__":
    app()
