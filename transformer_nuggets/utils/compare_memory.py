"""Compare two CUDA memory snapshots side-by-side in a single interactive HTML page."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Compare two memory snapshots side-by-side.")


@app.command()
def main(
    left: Annotated[Path, typer.Argument(help="Left snapshot pickle file.")],
    right: Annotated[Path, typer.Argument(help="Right snapshot pickle file.")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output HTML path.")] = Path(
        "memory_comparison.html"
    ),
    device: Annotated[int, typer.Option("--device", help="CUDA device index.")] = 0,
    title_left: Annotated[str | None, typer.Option("--title-left")] = None,
    title_right: Annotated[str | None, typer.Option("--title-right")] = None,
):
    """Generate a side-by-side memory comparison HTML from two snapshot pickles."""
    for p in (left, right):
        if not p.exists():
            typer.echo(f"Error: {p} not found", err=True)
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
        title_left=title_left or left.stem,
        title_right=title_right or right.stem,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    typer.echo(f"Wrote comparison to {output}")


if __name__ == "__main__":
    app()
