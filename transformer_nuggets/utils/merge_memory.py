"""Merge two CUDA memory snapshots into a single side-by-side interactive HTML page."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Merge two memory snapshots into one side-by-side HTML page.")


@app.command()
def main(
    left: Annotated[Path, typer.Argument(help="Left snapshot pickle file.")],
    right: Annotated[Path, typer.Argument(help="Right snapshot pickle file.")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output HTML path.")] = Path(
        "merged_memory.html"
    ),
    device: Annotated[int, typer.Option("--device", help="CUDA device index for both sides.")] = 0,
    device_left: Annotated[int | None, typer.Option("--device-left")] = None,
    device_right: Annotated[int | None, typer.Option("--device-right")] = None,
    title_left: Annotated[str | None, typer.Option("--title-left")] = None,
    title_right: Annotated[str | None, typer.Option("--title-right")] = None,
):
    """Merge two memory snapshot pickles into one side-by-side interactive HTML page."""
    for path in (left, right):
        if not path.exists():
            typer.echo(f"Error: {path} not found", err=True)
            raise typer.Exit(1)

    from transformer_nuggets.utils.memory_viz import generate_memory_comparison_html

    with open(left, "rb") as f:
        snapshot_left = pickle.load(f)
    with open(right, "rb") as f:
        snapshot_right = pickle.load(f)

    html = generate_memory_comparison_html(
        snapshot_left,
        snapshot_right,
        device=device,
        device_left=device_left,
        device_right=device_right,
        title_left=title_left or left.stem,
        title_right=title_right or right.stem,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    typer.echo(f"Merged memory snapshots -> {output}")


if __name__ == "__main__":
    app()
