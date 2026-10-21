"""One-stop FlexGEMM performance tool: fused QUACK epilogues vs stock torch.compile vs bare cuBLAS.

Run from the repo root:

    python benchmarks/flex_gemm/bench_flex_gemm.py --families relu,silu_fast --shapes grid
    python benchmarks/flex_gemm/bench_flex_gemm.py --families grouped_swiglu_interleaved --shapes moe_16b
    python benchmarks/flex_gemm/bench_flex_gemm.py --fast --out benchmarks/data/flex_gemm/smoke
    python benchmarks/flex_gemm/report.py --diff before/results.csv after/results.csv

Run GPU work through `gpu-run auto -- ...`. See README.md for the roofline model and the
measurement contract. Every printed ratio is a time ratio: lower is better.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from epilogues import Case, FAMILIES, FamilyUnavailable, Problem, Variant
from harness import (
    accept,
    compare,
    compile_and_run,
    Compiled,
    copy_bandwidth_gbs,
    launch_geometry,
    Route,
    route_info,
    time_variants,
)
from report import append_rows, heatmaps, read_rows, summarize
from shapes import FAST_SHAPES, parse_dense, parse_moe, Shape, SHAPE_SETS

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config


HERE = Path(__file__).resolve().parent
# benchmarks/data is gitignored; results default to benchmarks/data/flex_gemm/<run>/.
DEFAULT_RESULTS_DIR = HERE.parent / "data" / "flex_gemm"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
FAST_FAMILIES = ["relu", "silu_fast", "swiglu_interleaved"]
# default: QuACK's heuristic GemmConfig (autotune=False); autotune: Inductor benchmarks the
# priority subset (autotune=True); exhaustive: autotune=True over every legal config.
MODES = ("default", "autotune", "exhaustive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--families",
        default="",
        help=f"comma list from: {','.join(FAMILIES)} (default: all dense pointwise)",
    )
    parser.add_argument(
        "--shapes", default="", help=f"comma list of shape sets: {','.join(SHAPE_SETS)}"
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="M,N,K",
        help="dense shape (repeatable)",
    )
    parser.add_argument(
        "--moe",
        action="append",
        default=[],
        metavar="E,K,N,M[,balanced|skewed|hot]",
        help="grouped_mm shape (repeatable)",
    )
    parser.add_argument(
        "--modes",
        default="default,autotune",
        help=f"comma list from: {', '.join(MODES)}",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="add the `exhaustive` mode (autotune=True under max_autotune_gemm_search_space=EXHAUSTIVE)",
    )
    parser.add_argument(
        "--baselines",
        default="cublas,stock",
        help="comma list from: cublas, stock, two_gemm",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=100, help="graph replays per round")
    parser.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="seconds of replay before timing (DVFS settle)",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument(
        "--label",
        default="",
        help="free-form tag written to every row (use with report.py --diff)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output dir (default {DEFAULT_RESULTS_DIR}/<timestamp>[_<label>]_<torch commit>/)",
    )
    parser.add_argument("--isolate", action="store_true", help="one subprocess per shape")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="smoke: 2 shapes x 3 families, 3 rounds x 30 replays",
    )
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--_shape-json", dest="shape_json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.fast:
        args.rounds, args.replays, args.settle = 3, 30, 1.0
    modes = [m for m in args.modes.split(",") if m]
    if args.exhaustive and "exhaustive" not in modes:
        modes.append("exhaustive")
    for mode in modes:
        if mode not in MODES:
            parser.error(f"unknown mode {mode!r}; expected one of {MODES}")
    args.mode_list = modes
    args.baseline_list = [b for b in args.baselines.split(",") if b]
    families = [f for f in args.families.split(",") if f]
    for family in families:
        if family not in FAMILIES:
            parser.error(f"unknown family {family!r}; expected one of {','.join(FAMILIES)}")
    if not families:
        families = (
            FAST_FAMILIES
            if args.fast
            else ["identity", "relu", "silu", "silu_fast", "bias_gelu", "residual"]
        )
    args.family_list = families[:3] if args.fast else families
    if args.out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = f"{stamp}_{args.label}" if args.label else stamp
        args.out = DEFAULT_RESULTS_DIR / f"{tag}_{git_commit()}"
    return args


def resolve_shapes(args: argparse.Namespace) -> list[Shape]:
    if args.shape_json:
        return [Shape.from_json(args.shape_json)]
    shapes: list[Shape] = []
    for name in (s for s in args.shapes.split(",") if s):
        if name not in SHAPE_SETS:
            raise SystemExit(f"unknown shape set {name!r}; expected one of {','.join(SHAPE_SETS)}")
        shapes += SHAPE_SETS[name]
    shapes += [parse_dense(s) for s in args.shape] + [parse_moe(s) for s in args.moe]
    if not shapes:
        shapes = FAST_SHAPES if args.fast else SHAPE_SETS["grid"]
    if args.fast:
        shapes = shapes[:2]
    return list(dict.fromkeys(shapes))


def kernel_options(mode: str, case: Case) -> dict[str, Any]:
    options: dict[str, Any] = {"backend": "QUACK", "autotune": mode != "default"}
    if case.fast_math:
        options["fast_math"] = True
    return options


def git_commit() -> str:
    """Short commit of the torch checkout under test (editable install), else torch.__version__."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(torch.__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return torch.__version__


def log(msg: str) -> None:
    print(msg, flush=True)


class ShapeRun:
    """Compiles, validates and times every requested variant of one shape."""

    def __init__(
        self,
        shape: Shape,
        args: argparse.Namespace,
        meta: dict[str, Any],
        codes_dir: Path,
    ) -> None:
        self.shape = shape
        self.args = args
        self.meta = meta
        self.codes_dir = codes_dir
        self.dtype = DTYPES[args.dtype]
        self.rows: list[dict[str, Any]] = []
        self.variants: dict[str, Variant] = {}
        self.compiled: dict[str, Compiled] = {}
        self.routes: dict[str, Route] = {}
        self.errors: dict[str, Any] = {}
        self.status: dict[str, tuple[str, str]] = {}

    def run(self) -> list[dict[str, Any]]:
        torch._dynamo.reset()
        log(f"=== {self.shape.label}")
        problem = Problem(self.shape, self.dtype)
        cases: dict[str, Case] = {}
        for family in self.args.family_list:
            if family != "identity" and family.startswith("grouped_") != self.shape.grouped:
                log(f"{family}: skipped ({'MoE' if self.shape.grouped else 'dense'} shape)")
                continue
            try:
                cases[family] = FAMILIES[family](problem)
            except FamilyUnavailable as e:
                self.skip(family, "unavailable", str(e))
        identity_cases = self.identity_cases(problem, cases)

        for n_phys, case in identity_cases.items():
            self.variants[f"cublas:bf16:n{n_phys}"] = case.cublas
            for mode in self.args.mode_list:
                self.compile_flex(f"flex:identity:n{n_phys}:{mode}", case, mode)
        for family, case in cases.items():
            if family == "identity":
                continue
            if case.gemm_key != "bf16":
                self.variants[f"cublas:{case.gemm_key}:n{case.lanes * self.shape.n}"] = case.cublas
            if "stock" in self.args.baseline_list:
                self.compile_plain(f"stock:{family}", case.stock, case)
            if "two_gemm" in self.args.baseline_list and case.two_gemm is not None:
                self.compile_plain(f"two_gemm:{family}", case.two_gemm, case)
            for mode in self.args.mode_list:
                self.compile_flex(f"flex:{family}:{mode}", case, mode)
        if "cublas" not in self.args.baseline_list:
            for key in [k for k in self.variants if k.startswith("cublas:")]:
                del self.variants[key]

        timings, clocks = time_variants(
            self.variants,
            rounds=self.args.rounds,
            replays=self.args.replays,
            settle_s=self.args.settle,
        )
        log(f"clocks: {clocks}")
        for family, case in cases.items():
            for mode in self.args.mode_list:
                self.rows.append(self.row(family, case, mode, timings, clocks))
        return self.rows

    def identity_cases(self, problem: Problem, cases: dict[str, Case]) -> dict[int, Case]:
        """Identity (bare QuACK GEMM) at every physical width used by a bf16 family; keyed by N_phys."""
        widths = sorted(
            {c.lanes * self.shape.n for c in cases.values() if c.gemm_key == "bf16"}
            | {self.shape.n}
        )
        out: dict[int, Case] = {}
        for n_phys in widths:
            if n_phys == self.shape.n:
                out[n_phys] = cases.get("identity") or FAMILIES["identity"](problem)
            else:
                out[n_phys] = FAMILIES["identity"](
                    Problem(dataclasses.replace(self.shape, n=n_phys), self.dtype)
                )
        return out

    def skip(self, family: str, status: str, detail: str) -> None:
        log(f"{family}: {status}: {detail}")
        for mode in self.args.mode_list:
            self.rows.append({**self.base_row(family, mode), "status": status, "error": detail})

    def compile_plain(self, key: str, variant: Variant, case: Case) -> None:
        try:
            compiled = compile_and_run(variant)
        except Exception as e:
            self.status[key] = ("compile_error", repr(e)[-1500:])
            log(f"{key}: COMPILE ERROR {e!r}"[:400])
            return
        self.compiled[key] = compiled
        self.routes[key] = route_info(compiled.code)
        self.errors[key] = compare(case.comparable(compiled.output), case.reference)
        self.variants[key] = compiled.variant
        log(
            f"{key}: extern={self.routes[key].extern} triton={self.routes[key].triton_kernels} nrmse={[f'{v:.2e}' for v in self.errors[key].nrmse]} compile={compiled.compile_s:.1f}s"
        )

    def compile_flex(self, key: str, case: Case, mode: str) -> None:
        options = kernel_options(mode, case)
        search = "EXHAUSTIVE" if mode == "exhaustive" else "DEFAULT"
        try:
            with inductor_config.patch(max_autotune_gemm_search_space=search):
                compiled = compile_and_run(case.flex(options))
        except Exception as e:
            self.status[key] = ("compile_error", repr(e)[-1500:])
            log(f"{key}: COMPILE ERROR {e!r}"[:400])
            return
        (self.codes_dir / f"{self.shape.label}__{key.replace(':', '_')}.py").write_text(
            compiled.code
        )
        route = route_info(compiled.code)
        err = compare(case.comparable(compiled.output), case.reference)
        self.compiled[key], self.routes[key], self.errors[key] = compiled, route, err
        stock_err = self.errors.get(f"stock:{case.family}")
        if not route.fused:
            self.status[key] = (
                "fallback",
                f"flex_calls={route.flex_calls} extern={route.extern}",
            )
            log(f"{key}: ROUTE FALLBACK {route.extern}")
        elif not accept(err, stock_err):
            self.status[key] = (
                "correctness_failed",
                f"nrmse={err.nrmse} max_abs={err.max_abs} finite={err.finite}",
            )
            log(f"{key}: CORRECTNESS FAILED {self.status[key][1]}")
        else:
            self.variants[key] = compiled.variant
            log(
                f"{key}: config={route.config_label} nrmse={[f'{v:.2e}' for v in err.nrmse]} compile={compiled.compile_s:.1f}s"
            )

    def base_row(self, family: str, mode: str) -> dict[str, Any]:
        s = self.shape
        return {
            **self.meta,
            "shape": s.label,
            "M": s.m,
            "N": s.n,
            "K": s.k,
            "E": s.experts,
            "offs": s.offs_kind,
            "family": family,
            "mode": mode,
            "rounds": self.args.rounds,
            "replays": self.args.replays,
        }

    def row(
        self, family: str, case: Case, mode: str, timings: dict, clocks: list[str]
    ) -> dict[str, Any]:
        n_phys = case.lanes * self.shape.n
        flex_key = (
            f"flex:{family}:{mode}" if family != "identity" else f"flex:identity:n{n_phys}:{mode}"
        )
        stock_key = f"cublas:bf16:n{n_phys}" if family == "identity" else f"stock:{family}"
        cublas_key = f"cublas:{case.gemm_key}:n{n_phys}"
        identity_key = f"flex:identity:n{n_phys}:{mode}"
        row = {
            **self.base_row(family, mode),
            "n_phys": n_phys,
            "clocks": clocks[0] if clocks else "",
        }
        t_flex, t_stock, t_cublas, t_identity, t_two = (
            timings[name].median_us if name in timings else None
            for name in (
                flex_key,
                stock_key,
                cublas_key,
                identity_key,
                f"two_gemm:{family}",
            )
        )
        bw = self.meta["hbm_bw_gbs"]
        saved_us = case.saved_bytes / (bw * 1e9) * 1e6
        row.update(
            t_flex=t_flex,
            t_stock=t_stock,
            t_cublas=t_cublas,
            t_flex_identity=t_identity if case.gemm_key == "bf16" else None,
            t_two_gemm=t_two,
            saved_bytes=case.saved_bytes,
            saved_time_at_bw=saved_us,
            flex_over_stock=t_flex / t_stock if t_flex and t_stock else None,
            flex_over_roofline=t_flex / (t_stock - saved_us)
            if t_flex and t_stock and t_stock > saved_us
            else None,
            gemm_gap=t_identity / t_cublas
            if t_identity and t_cublas and case.gemm_key == "bf16"
            else None,
            flex_tflops=case.flops / (t_flex * 1e-6) / 1e12 if t_flex else None,
            cublas_tflops=case.flops / (t_cublas * 1e-6) / 1e12 if t_cublas else None,
            flex_spread=timings[flex_key].spread if flex_key in timings else None,
            stock_spread=timings[stock_key].spread if stock_key in timings else None,
            cublas_spread=timings[cublas_key].spread if cublas_key in timings else None,
        )
        route = self.routes.get(flex_key)
        if route is not None:
            cfg = route.configs[0] if route.configs else None
            geometry = launch_geometry(cfg, self.shape.m, n_phys)
            row.update(
                flex_config=route.config_label,
                ctas=geometry.get("ctas"),
                waves=geometry.get("waves"),
                route=json.dumps(
                    {
                        "flex_calls": route.flex_calls,
                        "extern": route.extern,
                        "triton": route.triton_kernels,
                        "fast_math": route.fast_math,
                    }
                ),
            )
        if (identity_route := self.routes.get(identity_key)) is not None:
            row["identity_config"] = identity_route.config_label
        if (err := self.errors.get(flex_key)) is not None:
            row["flex_nrmse"] = max(err.nrmse, default=None)
            row["correctness"] = json.dumps(
                {"nrmse": err.nrmse, "max_abs": err.max_abs, "finite": err.finite}
            )
        if (stock_err := self.errors.get(stock_key)) is not None:
            row["stock_nrmse"] = max(stock_err.nrmse, default=None)
        if flex_key in self.compiled:
            row["compile_s"] = round(self.compiled[flex_key].compile_s, 1)
        status, detail = self.status.get(flex_key, ("ok", ""))
        if status == "ok" and stock_key in self.status:
            detail = f"stock: {self.status[stock_key][1]}"
        row.update(status=status, error=detail)
        return row


def child_argv(args: argparse.Namespace, shape: Shape) -> list[str]:
    argv = [
        sys.executable,
        str(HERE / "bench_flex_gemm.py"),
        "--_shape-json",
        shape.to_json(),
        "--families",
        ",".join(args.family_list),
        "--modes",
        ",".join(args.mode_list),
        "--baselines",
        args.baselines,
        "--rounds",
        str(args.rounds),
        "--replays",
        str(args.replays),
        "--settle",
        str(args.settle),
        "--dtype",
        args.dtype,
        "--label",
        args.label,
        "--out",
        str(args.out),
        "--no-heatmaps",
    ]
    return argv


def main() -> None:
    args = parse_args()
    shapes = resolve_shapes(args)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "results.csv"
    if args.isolate and not args.shape_json:
        for shape in shapes:
            log(f"--- isolated child for {shape.label}")
            result = subprocess.run(child_argv(args, shape), check=False, env=os.environ.copy())
            if result.returncode:
                log(f"child for {shape.label} exited with {result.returncode}")
    else:
        dynamo_config.recompile_limit = 256
        dynamo_config.accumulated_recompile_limit = 4096
        dynamo_config.automatic_dynamic_shapes = False
        props = torch.cuda.get_device_properties(0)
        bw = copy_bandwidth_gbs()
        meta = {
            "label": args.label,
            "gpu": f"{props.name} ({props.multi_processor_count} SMs)",
            "commit": git_commit(),
            "hbm_bw_gbs": bw,
        }
        log(
            f"{meta['gpu']} commit={meta['commit']} copy BW (read+write) = {bw:.0f} GB/s; {len(shapes)} shapes x {args.family_list} x {args.mode_list}"
        )
        codes_dir = args.out / "codes"
        codes_dir.mkdir(exist_ok=True)
        for shape in shapes:
            rows = ShapeRun(shape, args, meta, codes_dir).run()
            append_rows(csv_path, rows)
            for r in rows:
                if r.get("status") == "ok":
                    log(
                        f"{r['shape']:>28} {r['family']:<26} {r['mode']:<10} flex {r['t_flex']:8.2f} us  stock {r['t_stock'] or float('nan'):8.2f}  cublas {r['t_cublas'] or float('nan'):8.2f}  flex/stock {r['flex_over_stock'] or float('nan'):.3f}  flex/roofline {r['flex_over_roofline'] or float('nan'):.3f}  gemm_gap {r['gemm_gap'] or float('nan'):.3f} (lower is better)  {r.get('flex_config', '')}"
                    )
            torch.cuda.empty_cache()
    if args.shape_json:
        return
    rows = read_rows(csv_path)
    text = summarize(rows, args.label)
    (args.out / "summary.md").write_text(text + "\n")
    log(text)
    if not args.no_heatmaps:
        for path in heatmaps(rows, args.out):
            log(f"wrote {path}")
    log(f"results: {csv_path}")


if __name__ == "__main__":
    main()
