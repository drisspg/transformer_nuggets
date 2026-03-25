"""
Run two transformer memory snapshot jobs sequentially and build one comparison HTML.

Usage:
    python examples/memory_viz_compare_transformer.py
    python examples/memory_viz_compare_transformer.py --left_num_heads 8 --right_num_heads 4
    python examples/memory_viz_compare_transformer.py --dry_run true
"""

from __future__ import annotations

from pathlib import Path
from jsonargparse import CLI
import shlex
import subprocess
import sys


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


def main(
    batch_size: int = 2,
    seq_len: int = 512,
    dim: int = 1024,
    num_layers: int = 8,
    left_num_heads: int = 8,
    right_num_heads: int = 4,
    left_output: str = "data/transformer_left_snapshot.pickle",
    right_output: str = "data/transformer_right_snapshot.pickle",
    comparison_output: str = "data/transformer_compare.html",
    left_title: str | None = None,
    right_title: str | None = None,
    dry_run: bool = False,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    snapshot_script = Path(__file__).with_name("memory_viz_transformer.py")
    left_output_path = repo_root / left_output
    right_output_path = repo_root / right_output
    comparison_output_path = repo_root / comparison_output

    left_output_path.parent.mkdir(parents=True, exist_ok=True)
    right_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)

    left_title = left_title or f"{left_num_heads} heads"
    right_title = right_title or f"{right_num_heads} heads"

    common_args = [
        str(snapshot_script),
        "--batch_size",
        str(batch_size),
        "--seq_len",
        str(seq_len),
        "--dim",
        str(dim),
        "--num_layers",
        str(num_layers),
    ]

    left_cmd = [
        sys.executable,
        *common_args,
        "--num_heads",
        str(left_num_heads),
        "--output",
        str(left_output_path),
    ]
    right_cmd = [
        sys.executable,
        *common_args,
        "--num_heads",
        str(right_num_heads),
        "--output",
        str(right_output_path),
    ]
    compare_cmd = [
        sys.executable,
        "-m",
        "transformer_nuggets.utils.compare_memory",
        str(left_output_path),
        str(right_output_path),
        "-o",
        str(comparison_output_path),
        "--title-left",
        left_title,
        "--title-right",
        right_title,
    ]

    _run(left_cmd, cwd=repo_root, dry_run=dry_run)
    _run(right_cmd, cwd=repo_root, dry_run=dry_run)
    _run(compare_cmd, cwd=repo_root, dry_run=dry_run)


if __name__ == "__main__":
    CLI(main, as_positional=False)
