"""Sweep pinned GemmConfigs for one (shape, family) under the bench's CUDA-graph contract.

Run from the repo root:

    gpu-run auto -- python benchmarks/flex_gemm/config_sweep.py --shape 16384,8192,4096 --family relu

Compiles one variant per SM100 priority config via ``kernel_options={"config": {...}}`` (plus the
autotune=True autotuner pick and the heuristic default), checks each against the fp64 reference,
then times all of them interleaved. Every ratio printed is a time ratio, lower is better.
"""

from __future__ import annotations

import argparse
import sys

import torch

import epilogues
from harness import accept, compare, compile_and_run, route_info, time_variants
from shapes import parse_dense


# (tile_m, tile_n, cluster_m, cluster_n, is_dynamic_persistent): the dense SM100 priority order in
# torch/_inductor/heuristics/template/flex_gemm.py (_PRIORITY_RANK); keep in sync.
PRIORITY = (
    (128, 256, 2, 1, True),
    (128, 192, 2, 1, True),
    (256, 256, 2, 1, True),
    (256, 256, 2, 2, True),
    (256, 192, 2, 1, True),
    (128, 128, 1, 1, False),
    (128, 256, 1, 1, True),
    (128, 256, 1, 1, False),
    (128, 128, 2, 1, True),
    (256, 128, 2, 1, True),
    (128, 224, 1, 1, True),
    (128, 160, 1, 1, True),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=40)
    parser.add_argument("--settle", type=float, default=3.0)
    args = parser.parse_args()

    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 4096
    shape = parse_dense(args.shape)
    if args.family not in epilogues.FAMILIES:
        parser.error(
            f"unknown family {args.family!r}; expected one of {','.join(epilogues.FAMILIES)}"
        )
    try:
        case = epilogues.FAMILIES[args.family](epilogues.Problem(shape, torch.bfloat16))
    except epilogues.FamilyUnavailable as e:
        raise SystemExit(f"{args.family}: unavailable: {e}") from e

    variants = {}
    labels = {}
    stock = compile_and_run(case.stock)
    stock_err = compare(case.comparable(stock.output), case.reference)
    variants["stock"] = stock.variant
    variants["cublas"] = case.cublas

    candidates: list[tuple[str, dict]] = [
        ("default", {"backend": "QUACK"}),
        ("autotune", {"backend": "QUACK", "autotune": True}),
    ]
    for tm, tn, cm, cn, dyn in PRIORITY:
        name = f"{tm}x{tn}_c{cm}x{cn}_{'clc' if dyn else 'static'}"
        cfg = {
            "tile_m": tm,
            "tile_n": tn,
            "cluster_m": cm,
            "cluster_n": cn,
            "is_dynamic_persistent": dyn,
            "pingpong": False,
            "swap_ab": False,
        }
        candidates.append((name, {"backend": "QUACK", "config": cfg}))

    for name, kernel_options in candidates:
        if case.fast_math:
            kernel_options = {**kernel_options, "fast_math": True}
        try:
            compiled = compile_and_run(case.flex(kernel_options))
        except Exception as e:  # illegal config for this epilogue, or lowering rejection
            print(f"{name:>22}: skipped ({type(e).__name__}: {str(e).splitlines()[0][:90]})")
            continue
        route = route_info(compiled.code)
        err = compare(case.comparable(compiled.output), case.reference)
        if not route.fused:
            print(f"{name:>22}: fallback ({route.extern})")
            continue
        if not accept(err, stock_err):
            print(f"{name:>22}: correctness_failed nrmse={err.nrmse}")
            continue
        variants[name] = compiled.variant
        labels[name] = route.config_label
        print(
            f"{name:>22}: pinned {route.config_label} compile={compiled.compile_s:.1f}s",
            flush=True,
        )

    timings, clocks = time_variants(
        variants, rounds=args.rounds, replays=args.replays, settle_s=args.settle
    )
    t_stock = timings["stock"].median_us
    print(f"\n{shape.label} {args.family}  clocks per round: {clocks}")
    print(
        f"{'variant':>22} {'pinned config':<26} {'us':>9} {'/stock':>7} {'/cublas':>8} {'spread':>7}"
    )
    ordered = sorted(timings.items(), key=lambda kv: kv[1].median_us)
    for name, t in ordered:
        print(
            f"{name:>22} {labels.get(name, ''):<26} {t.median_us:9.1f} "
            f"{t.median_us / t_stock:7.3f} {t.median_us / timings['cublas'].median_us:8.3f} {t.spread:7.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
