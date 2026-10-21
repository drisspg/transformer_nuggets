"""CSV rows, markdown summary, heatmaps and A/B diff for the FlexGEMM benchmark.

    python benchmarks/flex_gemm/report.py summarize OUT/results.csv [--out OUT]
    python benchmarks/flex_gemm/report.py --diff A/results.csv B/results.csv

All ratios are time ratios: lower is better.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


COLUMNS = [
    "label", "gpu", "commit", "shape", "M", "N", "K", "n_phys", "E", "offs", "family", "mode",
    "t_cublas", "t_stock", "t_two_gemm", "t_flex", "t_flex_identity",
    "saved_bytes", "hbm_bw_gbs", "saved_time_at_bw",
    "flex_over_stock", "flex_over_roofline", "gemm_gap",
    "flex_tflops", "cublas_tflops",
    "flex_config", "identity_config", "ctas", "waves",
    "flex_spread", "stock_spread", "cublas_spread",
    "flex_nrmse", "stock_nrmse", "correctness", "route",
    "compile_s", "clocks", "rounds", "replays", "status", "error",
]  # fmt: skip

RATIO_COLUMNS = {
    "flex_over_stock": "t_flex / t_stock",
    "flex_over_roofline": "t_flex / (t_stock - saved_time_at_bw)",
    "gemm_gap": "t_flex_identity / t_cublas",
}
WIN, LOSS = 0.97, 1.03


def append_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in COLUMNS})


def _fmt(value) -> str:
    if isinstance(value, float):
        return (
            f"{value:.4g}"
            if abs(value) < 1e-2
            else f"{value:.2f}"
            if abs(value) >= 100
            else f"{value:.3f}"
        )
    return "" if value is None else str(value)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _num(row: dict, key: str) -> float | None:
    text = row.get(key, "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def _g(values: list[float]) -> str:
    g = geomean(values)
    return "" if g is None else f"{g:.3f}"


def _cell_name(row: dict) -> str:
    return row["shape"]


def summarize(rows: list[dict], label: str = "") -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    lines = [f"# FlexGEMM benchmark summary{(' - ' + label) if label else ''}", ""]
    if rows:
        head = rows[0]
        lines.append(
            f"GPU {head.get('gpu', '?')}, commit {head.get('commit', '?')}, label {head.get('label', '')}, {len(rows)} rows ({len(ok)} ok). Measured copy BW {head.get('hbm_bw_gbs', '?')} GB/s."
        )
    lines.append(
        "All ratios are time ratios (lower is better): "
        + ", ".join(f"`{k}` = {v}" for k, v in RATIO_COLUMNS.items())
        + "."
    )
    lines.append("")

    by_family: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in ok:
        by_family[(r["family"], r["mode"])].append(r)

    lines += [
        "## Per family (geomean over cells)",
        "",
        "| family | mode | cells | flex/stock | wins <0.97 | losses >1.03 | flex/roofline | gemm gap | flex TFLOP/s (geomean) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (family, mode), group in sorted(by_family.items()):
        ratios = [v for r in group if (v := _num(r, "flex_over_stock")) is not None]
        lines.append(
            f"| {family} | {mode} | {len(group)} | {_g(ratios)} | {sum(v < WIN for v in ratios)} | {sum(v > LOSS for v in ratios)} "
            f"| {_g([v for r in group if (v := _num(r, 'flex_over_roofline')) is not None])} "
            f"| {_g([v for r in group if (v := _num(r, 'gemm_gap')) is not None])} "
            f"| {_g([v for r in group if (v := _num(r, 'flex_tflops')) is not None])} |"
        )
    lines.append("")

    for axis in ("M", "N", "K", "E", "offs"):
        table = _region_table(by_family, axis)
        if table:
            lines += [
                f"## flex/stock by {axis} (geomean; lower is better)",
                "",
                *table,
                "",
            ]

    scored = [(v, r) for r in ok if (v := _num(r, "flex_over_stock")) is not None]
    scored.sort(key=lambda item: item[0])
    for title, picks in (
        ("Best 5 cells", scored[:5]),
        ("Worst 5 cells", scored[-5:][::-1]),
    ):
        lines += [
            f"## {title} (flex/stock, lower is better)",
            "",
            "| cell | family | mode | flex/stock | flex/roofline | t_flex us | t_stock us | t_cublas us | pinned config | ctas (waves) |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for v, r in picks:
            lines.append(
                f"| {_cell_name(r)} | {r['family']} | {r['mode']} | {v:.3f} | {r.get('flex_over_roofline', '')} | {r.get('t_flex', '')} | {r.get('t_stock', '')} | {r.get('t_cublas', '')} | {r.get('flex_config', '')} | {r.get('ctas', '')} ({r.get('waves', '')}) |"
            )
        lines.append("")

    bad = [r for r in rows if r.get("status") != "ok"]
    if bad:
        lines += [
            "## Skipped / failed cells",
            "",
            "| cell | family | mode | status | detail |",
            "|---|---|---|---|---|",
        ]
        for r in bad:
            lines.append(
                f"| {_cell_name(r)} | {r['family']} | {r.get('mode', '')} | {r.get('status', '')} | {str(r.get('error', ''))[:160]} |"
            )
        lines.append("")
    return "\n".join(lines)


def _region_table(by_family: dict[tuple[str, str], list[dict]], axis: str) -> list[str]:
    values = sorted(
        {
            r[axis]
            for group in by_family.values()
            for r in group
            if r.get(axis) not in ("", "0", None)
        },
        key=_sort_key,
    )
    if len(values) < 2:
        return []
    lines = [
        "| family | mode | " + " | ".join(f"{axis}={v}" for v in values) + " |",
        "|---|---|" + "---:|" * len(values),
    ]
    for (family, mode), group in sorted(by_family.items()):
        cells = []
        for v in values:
            cells.append(
                _g(
                    [
                        x
                        for r in group
                        if r.get(axis) == v and (x := _num(r, "flex_over_stock")) is not None
                    ]
                )
            )
        if any(cells):
            lines.append(f"| {family} | {mode} | " + " | ".join(cells) + " |")
    return lines


def _sort_key(text: str):
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def heatmaps(rows: list[dict], out_dir: Path) -> list[Path]:
    """One PNG per (metric, family, mode): rows M, columns (N, K). Dense cells only."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as e:
        print(f"heatmaps skipped: {e}")
        return []
    dense = [r for r in rows if r.get("status") == "ok" and r.get("E") in ("", "0")]
    dense.sort(key=lambda r: r.get("n_phys") != r.get("N"))
    written = []
    for metric, title in RATIO_COLUMNS.items():
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        seen = set()
        for r in dense:
            if _num(r, metric) is None:
                continue
            # The GEMM gap is a per-shape property (identity epilogue), not a per-family one.
            key = ("identity", r["mode"]) if metric == "gemm_gap" else (r["family"], r["mode"])
            if metric == "gemm_gap" and (key, r["shape"]) in seen:
                continue
            seen.add((key, r["shape"]))
            groups[key].append(r)
        for (family, mode), group in sorted(groups.items()):
            frame = pd.DataFrame(
                {
                    "M": int(r["M"]),
                    "NK": f"N{r['N']}\nK{r['K']}",
                    metric: _num(r, metric),
                }
                for r in group
            )
            pivot = frame.pivot_table(index="M", columns="NK", values=metric, aggfunc="median")
            if pivot.size < 2:
                continue
            plt.figure(figsize=(1.2 * pivot.shape[1] + 3, 0.7 * pivot.shape[0] + 2))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn_r",
                center=1.0,
                vmin=0.5,
                vmax=1.5,
                cbar_kws={"label": f"{title} (lower is better)"},
            )
            plt.title(f"{family} [{mode}]: {title}")
            plt.tight_layout()
            path = out_dir / f"heat_{metric}_{family}_{mode}.png"
            plt.savefig(path, dpi=120)
            plt.close()
            written.append(path)
    return written


def diff(a_rows: list[dict], b_rows: list[dict]) -> str:
    """Per-cell t_flex ratio B/A (lower is better = B faster), joined on (family, mode, shape)."""

    def key(r):
        return (r["family"], r["mode"], r["shape"])

    a_map = {key(r): r for r in a_rows if r.get("status") == "ok"}
    a_label = a_rows[0].get("label", "A") if a_rows else "A"
    b_label = b_rows[0].get("label", "B") if b_rows else "B"
    lines = [
        f"# A/B: t_flex[{b_label}] / t_flex[{a_label}] (lower is better; <1 means B is faster)",
        "",
        "| cell | family | mode | A us | B us | B/A | A config | B config |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    ratios = []
    for r in b_rows:
        a = a_map.get(key(r))
        ta, tb = (_num(a, "t_flex") if a else None), _num(r, "t_flex")
        if ta is None or tb is None or ta <= 0:
            continue
        ratios.append(tb / ta)
        lines.append(
            f"| {r['shape']} | {r['family']} | {r['mode']} | {ta:.2f} | {tb:.2f} | {tb / ta:.3f} | {a.get('flex_config', '')} | {r.get('flex_config', '')} |"
        )
    lines += [
        "",
        f"cells: {len(ratios)}, geomean B/A: {_g(ratios)}, min {min(ratios):.3f}, max {max(ratios):.3f}"
        if ratios
        else "no common cells",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", nargs="?", choices=["summarize"], help="summarize a results.csv"
    )
    parser.add_argument("csv", nargs="?", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="directory for summary.md and heatmaps (default: the CSV's directory)",
    )
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("A.csv", "B.csv"))
    args = parser.parse_args()
    if args.diff:
        print(diff(read_rows(args.diff[0]), read_rows(args.diff[1])))
        return
    if args.command != "summarize" or args.csv is None:
        parser.error("expected `summarize results.csv` or `--diff A.csv B.csv`")
    rows = read_rows(args.csv)
    out = args.out or args.csv.parent
    out.mkdir(parents=True, exist_ok=True)
    text = summarize(rows)
    (out / "summary.md").write_text(text + "\n")
    print(text)
    for path in heatmaps(rows, out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
