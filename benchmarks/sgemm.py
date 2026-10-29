"""Benchmark FP32 TMA/FFMA2 SGEMM against cuBLAS with TF32 disabled.

Logical inputs are A (M, K) and B (K, N). All four row-/column-major pairs
are measured by default; these are tensor storage layouts, not BLAS N/T flags.
The TF32 baseline is reduced-precision context, not an equivalent SGEMM.

Contract: warm, fixed-pointer CUDA graph replay at externally managed clocks and
power. All variants pass eager and replay correctness before timing begins.
Each graph contains a bounded number of GEMMs; CUDA-event times are divided by
that count and include replay overhead, not compilation, capture, or allocation.
Rounds are deterministically interleaved. This is not a cold-DRAM benchmark or
a claim of settled clocks. --csv-path also writes a .details.json sidecar with
all configuration samples, kernel names, failures, and environment metadata.

    python benchmarks/sgemm.py --shapes 128,128,128
    python benchmarks/sgemm.py --shapes 4096,4096,4096 --a-layout col --b-layout row
"""

import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from tabulate import tabulate

from transformer_nuggets.cute.sgemm import SgemmConfig, naive_sgemm, sgemm
from transformer_nuggets.utils.benchmark import benchmark_cuda_graph_stats
from transformer_nuggets.utils.tracing import cuda_kernel_profiler

DEFAULT_SHAPES = (
    "512,512,512",
    "1024,1024,1024",
    "2048,2048,2048",
    "4096,4096,4096",
    "8192,8192,8192",
    "8192,4096,4096",
    "4096,14336,4096",
    "4096,4096,14336",
    "128,4096,4096",
    "16,4096,4096",
)

# Exploration candidates, not a shape-dependent production selector.
CONFIGS = (
    SgemmConfig(),
    SgemmConfig(tile_k=32, num_stages=2, threads_n=8),
    SgemmConfig(tile_m=64, tile_k=32, num_stages=2, num_threads=128, threads_n=8),
    # K-major B needs a different register-load/arithmetic tradeoff.
    SgemmConfig(tile_k=32, num_stages=2, k_group=1),
)
NAIVE_MAX_FLOPS = 2 * 4096**3
WARMUP_ITERS = 3


class Layout(str, Enum):
    row = "row"
    col = "col"


def parse_shape(shape: str) -> tuple[int, int, int]:
    try:
        m, n, k = (int(dim) for dim in shape.split(","))
    except ValueError as exc:
        raise typer.BadParameter(f"Expected an M,N,K triple, got {shape!r}") from exc
    if any(dim <= 0 or dim % 4 for dim in (m, n, k)):
        raise typer.BadParameter("M, N, and K must be positive multiples of 4")
    return m, n, k


def make_operand(rows: int, cols: int, layout: Layout) -> torch.Tensor:
    tensor = torch.empty(rows, cols, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    return tensor if layout == Layout.row else tensor.t().contiguous().t()


def tflops(m: int, n: int, k: int, time_us: float) -> float:
    return 2 * m * n * k / (time_us * 1e6)


@dataclass
class CapturedGraph:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor  # Keep graph-owned allocations alive for replay.
    kernel_names: list[str]


def capture_checked(
    fn: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    graph_ops: int,
) -> CapturedGraph:
    for _ in range(WARMUP_ITERS):
        out = fn()
    torch.testing.assert_close(out, reference, atol=atol, rtol=rtol)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_ops):
            out = fn()
    # A correct eager result must not hide an empty graph or a stale output.
    out.fill_(math.nan)
    graph.replay()
    torch.testing.assert_close(out, reference, atol=atol, rtol=rtol)
    torch.cuda.synchronize()
    with cuda_kernel_profiler() as profile:
        graph.replay()
        torch.cuda.synchronize()
    names = sorted(set(profile["kernel_names"]))
    if not names:
        raise RuntimeError("Captured graph replay produced no profiled CUDA kernels")
    return CapturedGraph(graph, out, names)


def environment_metadata() -> dict[str, object]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    root = Path(__file__).resolve().parents[1]
    sources = (Path(__file__).resolve(), root / "transformer_nuggets/cute/sgemm.py")
    metadata = {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "gpu": props.name,
        "gpu_uuid": str(props.uuid),
        "compute_capability": [props.major, props.minor],
        "sms": props.multi_processor_count,
        "source_sha256": {
            str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources
        },
        "environment": {
            name: value
            for name, value in os.environ.items()
            if name.startswith(
                ("CUDA_", "CUBLAS_", "CUTLASS_", "CUTE_DSL_", "SGEMM_", "TNUGGETS_", "CUTRACER_")
            )
        },
        "timing": "fixed-pointer graph replay; CUDA events; allocation/setup excluded",
        "clock_policy": "externally managed; no clock locks or DVFS stabilization imposed",
        "fill": "float32 uniform[-1,1]; same logical values for each layout pair",
    }
    for label, command in (
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--short"]),
        (
            "nvidia_smi_snapshot",
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version,clocks.sm,power.draw,power.limit,temperature.gpu",
                "--format=csv",
            ],
        ),
    ):
        try:
            result = subprocess.run(
                command, cwd=root, capture_output=True, text=True, timeout=5, check=False
            )
            metadata[label] = result.stdout.strip() if result.returncode == 0 else result.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            metadata[label] = f"unavailable: {exc}"
    return metadata


def benchmark_case(
    m: int,
    n: int,
    k: int,
    a_layout: Layout,
    b_layout: Layout,
    *,
    num_iters: int,
    rounds: int,
    replays: int,
    seed: int,
    results: list[dict[str, object]],
) -> dict[str, object]:
    # Reset for each layout so transpose-copy changes storage, not the input values.
    torch.manual_seed(seed)
    a = make_operand(m, k, a_layout)
    b = make_operand(k, n, b_layout)
    reference = (a.double() @ b.double()).float()
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    variants = {
        "cuBLAS": (partial(torch.mm, a, b, out=out), None, "ieee"),
        "cuBLAS TF32 (context)": (partial(torch.mm, a, b, out=out), None, "tf32"),
        "cute default": (partial(sgemm, a, b, out), CONFIGS[0], "ieee"),
    }
    for config in dict.fromkeys(CONFIGS[1:]):
        if config != CONFIGS[0]:
            variants[config.name] = (partial(sgemm, a, b, out, config), config, "ieee")
    if 2 * m * n * k <= NAIVE_MAX_FLOPS and a_layout == b_layout == Layout.row:
        variants["naive"] = (partial(naive_sgemm, a, b), None, "ieee")

    graphs = {}
    records = {}
    for name, (fn, config, precision) in variants.items():
        # The naive API allocates its output, so keep its graph to one GEMM.
        graph_ops = 1 if name == "naive" else num_iters
        # Retain the existing FP32 gate; TF32 has a distinct input-rounding budget.
        epsilon = 1e-3 if precision == "tf32" else 1e-5
        record = {
            "M": m,
            "N": n,
            "K": k,
            "a_layout": a_layout.value,
            "b_layout": b_layout.value,
            "a_stride": list(a.stride()),
            "b_stride": list(b.stride()),
            "variant": name,
            "config": asdict(config) if config is not None else None,
            "precision": precision,
            "graph_ops": graph_ops,
            "atol": epsilon * math.sqrt(k),
            "rtol": epsilon,
            "samples_us": [],
            "round_medians_us": [],
            "round_positions": [],
            "status": "checking",
        }
        results.append(record)
        records[name] = record
        torch.backends.cuda.matmul.fp32_precision = precision
        try:
            graphs[name] = capture_checked(
                fn, reference, atol=record["atol"], rtol=epsilon, graph_ops=graph_ops
            )
            record["kernel_names"] = graphs[name].kernel_names
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        record["status"] = "validated"
    torch.backends.cuda.matmul.fp32_precision = "ieee"

    # All compilation, correctness checks, and route profiling finish before timing.
    order = list(graphs)
    random.Random(seed).shuffle(order)
    for round_idx in range(rounds):
        rotated = order[round_idx % len(order) :] + order[: round_idx % len(order)]
        if round_idx % 2:
            rotated.reverse()
        for position, name in enumerate(rotated):
            record = records[name]
            try:
                stats = benchmark_cuda_graph_stats(
                    graphs[name].graph, num_iters=replays, warmup_iters=1, seed=seed + round_idx
                )
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            samples = [sample / record["graph_ops"] for sample in stats.samples_us]
            record["samples_us"].append(samples)
            record["round_medians_us"].append(statistics.median(samples))
            record["round_positions"].append(position)

    cublas_us = statistics.median(records["cuBLAS"]["round_medians_us"])
    for record in records.values():
        time_us = statistics.median(record["round_medians_us"])
        record.update(
            status="measured",
            median_us=time_us,
            tflops=tflops(m, n, k, time_us),
            speedup_vs_cublas=cublas_us / time_us,
        )
    cute_names = [name for name, record in records.items() if record["config"] is not None]
    best = min(cute_names, key=lambda name: records[name]["median_us"])
    print(
        tabulate(
            [
                {
                    key: record[key]
                    for key in ("variant", "median_us", "tflops", "speedup_vs_cublas")
                }
                for record in records.values()
            ],
            headers="keys",
            floatfmt=".2f",
        ),
        flush=True,
    )
    return {
        "M,N,K": f"{m},{n},{k}",
        "A layout": f"{a_layout.value}-major",
        "B layout": f"{b_layout.value}-major",
        "cuBLAS us": cublas_us,
        "cuBLAS TF/s": records["cuBLAS"]["tflops"],
        "cute default TF/s": records["cute default"]["tflops"],
        "cute best TF/s": records[best]["tflops"],
        "best speedup vs cuBLAS": records[best]["speedup_vs_cublas"],
        "best config": variants[best][1].name,
        "naive TF/s": records["naive"]["tflops"] if "naive" in records else None,
        "cuBLAS TF32 TF/s": records["cuBLAS TF32 (context)"]["tflops"],
        "cuBLAS kernel": ";".join(records["cuBLAS"]["kernel_names"]),
    }


def main(
    shapes: Annotated[
        list[str] | None, typer.Option(help="M,N,K triples; multiples of 4.")
    ] = None,
    a_layout: Annotated[
        Layout | None, typer.Option(help="A (M,K) storage; default: both.")
    ] = None,
    b_layout: Annotated[
        Layout | None, typer.Option(help="B (K,N) storage; default: both.")
    ] = None,
    num_iters: Annotated[int, typer.Option(min=1, max=1024, help="GEMMs per graph.")] = 20,
    rounds: Annotated[int, typer.Option(min=1, max=31, help="Odd interleaved round count.")] = 5,
    replays: Annotated[
        int, typer.Option(min=1, max=1000, help="Timed graph replays per round.")
    ] = 5,
    seed: int = 0,
    csv_path: Path | None = None,
) -> None:
    parsed_shapes = [parse_shape(shape) for shape in shapes or DEFAULT_SHAPES]
    if rounds % 2 == 0:
        raise typer.BadParameter("rounds must be odd")
    if os.environ.get("SGEMM_PROFILE", "0") != "0":
        raise typer.BadParameter("SGEMM_PROFILE must be 0 for uninstrumented timing")
    layouts = list(product([a_layout] if a_layout else Layout, [b_layout] if b_layout else Layout))
    metadata = environment_metadata()
    metadata.update(
        shapes=parsed_shapes,
        layouts=[(a.value, b.value) for a, b in layouts],
        configs=[asdict(config) for config in CONFIGS],
        num_iters=num_iters,
        rounds=rounds,
        replays=replays,
        seed=seed,
        eager_warmup_iters=WARMUP_ITERS,
        graph_warmup_replays_per_round=1,
    )
    print(json.dumps(metadata, indent=2), flush=True)
    rows = []
    results = []
    previous_precision = torch.backends.cuda.matmul.fp32_precision
    try:
        for m, n, k in parsed_shapes:
            for a_order, b_order in layouts:
                print(
                    f"\n{m},{n},{k}: A={a_order.value}-major, B={b_order.value}-major", flush=True
                )
                rows.append(
                    benchmark_case(
                        m,
                        n,
                        k,
                        a_order,
                        b_order,
                        num_iters=num_iters,
                        rounds=rounds,
                        replays=replays,
                        seed=seed,
                        results=results,
                    )
                )
        print(tabulate(rows, headers="keys", floatfmt=".2f"))
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous_precision
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            csv_path.with_suffix(".details.json").write_text(
                json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
            )


if __name__ == "__main__":
    typer.run(main)
