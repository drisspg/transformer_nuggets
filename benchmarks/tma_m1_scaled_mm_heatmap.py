from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Annotated
from collections.abc import Callable

import pandas as pd
import seaborn as sns
import torch
import typer
from matplotlib import pyplot as plt
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

from transformer_nuggets.cute import (
    DEFAULT_PERSISTENT_CTAS_PER_SM,
    GridScheduler,
    mxfp8_tma_scaled_mm,
    nvfp4_tma_scaled_mm,
    select_mxfp8_tma_compute_warps,
    select_nvfp4_tma_config,
    select_nvfp4_tma_split_k,
    select_nvfp4_tma_stage_weight_scales,
)
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds


app = typer.Typer()


@dataclass(frozen=True)
class KernelConfig:
    block_n: int
    num_stages: int
    num_compute_warps: int
    grid_scheduler: GridScheduler
    split_k: int = 1
    stage_weight_scales: bool = False


@dataclass(frozen=True)
class BenchmarkResult:
    format: str
    n: int
    k: int
    tma_us: float
    scaled_mm_us: float
    speedup: float
    block_n: int | None
    num_stages: int
    num_compute_warps: int | None
    grid_scheduler: str
    split_k: int
    stage_weight_scales: bool


def parse_sizes(value: str, dimension: str, multiple: int) -> list[int]:
    """Parse positive comma-separated dimensions with the format's required alignment."""
    sizes = [int(size) for size in value.split(",")]
    if not sizes or any(size <= 0 or size % multiple for size in sizes):
        raise typer.BadParameter(f"{dimension} values must be positive multiples of {multiple}")
    if dimension == "K" and any(size < 2048 for size in sizes):
        raise typer.BadParameter("K values must be at least 2048 for the two-stage TMA pipeline")
    return sizes


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert natural block scales to the scaled_mm SWIZZLE_32_4_4 storage contract."""
    rows, columns = scales.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_columns = ((columns + 3) // 4) * 4
    padded = torch.zeros((padded_rows, padded_columns), dtype=scales.dtype, device=scales.device)
    padded[:rows, :columns] = scales
    return (
        padded.view(padded_rows // 128, 128, padded_columns // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
        .contiguous()
    )


def quantize_mxfp8(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 rows into E4M3 values and natural E8M0 scales."""
    blocks = value.float().reshape(value.shape[0], -1, 32)
    exponent = torch.ceil(
        torch.log2(blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny) / 448.0)
    ).clamp(-126, 127)
    quantized = (blocks / torch.exp2(exponent).unsqueeze(-1)).clamp(-448, 448)
    return quantized.to(torch.float8_e4m3fn).reshape_as(value), (exponent + 127).to(torch.uint8)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack low-nibble-first E2M1 codes into PyTorch's FP4 shell dtype."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)


def make_mxfp8_case(n: int, k: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Create MXFP8 operands with the swizzled scale layout required by scaled_mm."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    mat_a, scale_a = quantize_mxfp8(
        torch.randn((1, k), dtype=torch.bfloat16, device="cuda", generator=generator)
    )
    weight, scale_b = quantize_mxfp8(
        torch.randn((n, k), dtype=torch.bfloat16, device="cuda", generator=generator)
    )
    return (
        mat_a,
        weight.t(),
        swizzle_scales(scale_a).view(torch.float8_e8m0fnu),
        swizzle_scales(scale_b).view(torch.float8_e8m0fnu),
    )


def make_nvfp4_case(n: int, k: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Create packed NVFP4 operands with nonuniform swizzled E4M3 block scales."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    mat_a = pack_fp4(
        torch.randint(0, 16, (1, k), device="cuda", dtype=torch.uint8, generator=generator)
    )
    weight = pack_fp4(
        torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8, generator=generator)
    )
    scale_choices = torch.tensor([0.5, 1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    scale_a = scale_choices[
        torch.randint(0, 4, (1, k // 16), device="cuda", generator=generator)
    ].to(torch.float8_e4m3fn)
    scale_b = scale_choices[
        torch.randint(0, 4, (n, k // 16), device="cuda", generator=generator)
    ].to(torch.float8_e4m3fn)
    return mat_a, weight.t(), swizzle_scales(scale_a), swizzle_scales(scale_b)


def select_mxfp8_config(n: int, k: int, device: torch.device) -> KernelConfig:
    """Select the measured B200 MXFP8 configuration for the heatmap grid."""
    if k <= 2048:
        block_n, num_stages = 16, 2
    elif k <= 4096:
        block_n, num_stages = 8, 2
    elif k >= 12288:
        block_n, num_stages = 4, 3
    else:
        block_n, num_stages = 4, 2
    if k >= 24576:
        num_compute_warps = 4 if n <= 4096 else 2
        grid_scheduler = GridScheduler.STATIC
    elif k >= 8192:
        num_compute_warps = 4
        grid_scheduler = GridScheduler.STATIC
    else:
        num_compute_warps = select_mxfp8_tma_compute_warps(k, block_n)
        requested_ctas = (
            DEFAULT_PERSISTENT_CTAS_PER_SM
            * torch.cuda.get_device_properties(device).multi_processor_count
        )
        grid_scheduler = (
            GridScheduler.PERSISTENT
            if (n + block_n - 1) // block_n > requested_ctas
            else GridScheduler.STATIC
        )
    return KernelConfig(block_n, num_stages, num_compute_warps, grid_scheduler)


def select_nvfp4_config(n: int, k: int, device: torch.device) -> KernelConfig:
    """Select measured NVFP4 exceptions around the generic B200 selector."""
    block_n, num_stages, num_compute_warps = select_nvfp4_tma_config(n, k, device)
    if n >= 16384 and 6144 <= k <= 8192:
        block_n, num_compute_warps = 8, 4
    elif n >= 12288 and k == 12288:
        block_n, num_compute_warps = 16, 4
    split_k = select_nvfp4_tma_split_k(n, k, device)
    stage_weight_scales = select_nvfp4_tma_stage_weight_scales(
        n,
        k,
        block_n,
        num_compute_warps,
        GridScheduler.STATIC,
        split_k,
        device,
    )
    return KernelConfig(
        block_n,
        num_stages,
        num_compute_warps,
        GridScheduler.STATIC,
        split_k,
        stage_weight_scales,
    )


def scaled_mm_mxfp8(
    mat_a: torch.Tensor, mat_b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
):
    """Run the cuBLASLt MXFP8 block-scaled GEMV baseline."""
    return scaled_mm(
        mat_a,
        mat_b,
        scale_a=[scale_a],
        scale_recipe_a=[ScalingType.BlockWise1x32],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4],
        scale_b=[scale_b],
        scale_recipe_b=[ScalingType.BlockWise1x32],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4],
        output_dtype=torch.bfloat16,
    )


def scaled_mm_nvfp4(
    mat_a: torch.Tensor, mat_b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
):
    """Run the cuBLASLt NVFP4 block-scaled GEMV baseline."""
    return scaled_mm(
        mat_a,
        mat_b,
        scale_a=[scale_a],
        scale_recipe_a=[ScalingType.BlockWise1x16],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4],
        scale_b=[scale_b],
        scale_recipe_b=[ScalingType.BlockWise1x16],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4],
        output_dtype=torch.bfloat16,
    )


def median_interleaved_latency(
    tma: Callable[[], torch.Tensor],
    baseline: Callable[[], torch.Tensor],
    rounds: int,
    iterations: int,
) -> tuple[float, float]:
    """Measure TMA and cuBLASLt with alternating round order under CUDA graph replay."""
    samples = {"tma": [], "baseline": []}
    for round_index in range(rounds):
        order = (("tma", tma), ("baseline", baseline))
        if round_index % 2:
            order = tuple(reversed(order))
        for name, kernel in order:
            samples[name].append(
                benchmark_cuda_function_in_microseconds(
                    kernel,
                    NUM_ITERS=iterations,
                    USE_CUDA_GRAPHS=True,
                )
            )
    return median(samples["tma"]), median(samples["baseline"])


def run_mxfp8(n: int, k: int, rounds: int, iterations: int, seed: int) -> BenchmarkResult:
    """Benchmark the MXFP8 TMA specialization against the matching scaled_mm contract."""
    mat_a, mat_b, scale_a, scale_b = make_mxfp8_case(n, k, seed)
    config = select_mxfp8_config(n, k, mat_a.device)
    output = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    tma = lambda: mxfp8_tma_scaled_mm(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        block_n=config.block_n,
        num_stages=config.num_stages,
        num_compute_warps=config.num_compute_warps,
        grid_scheduler=config.grid_scheduler,
        output=output,
    )
    baseline = lambda: scaled_mm_mxfp8(mat_a, mat_b, scale_a, scale_b)
    torch.testing.assert_close(tma(), baseline(), atol=1.0, rtol=0.05)
    torch.cuda.synchronize()
    tma_us, scaled_mm_us = median_interleaved_latency(tma, baseline, rounds, iterations)
    return BenchmarkResult(
        "mxfp8",
        n,
        k,
        tma_us,
        scaled_mm_us,
        scaled_mm_us / tma_us,
        config.block_n,
        config.num_stages,
        config.num_compute_warps,
        config.grid_scheduler.value,
        config.split_k,
        config.stage_weight_scales,
    )


def run_nvfp4(n: int, k: int, rounds: int, iterations: int, seed: int) -> BenchmarkResult:
    """Benchmark the NVFP4 TMA specialization against the matching scaled_mm contract."""
    mat_a, mat_b, scale_a, scale_b = make_nvfp4_case(n, k, seed)
    config = select_nvfp4_config(n, k, mat_a.device)
    output = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    partial_output = (
        torch.empty((config.split_k, n), dtype=torch.float32, device="cuda")
        if config.split_k > 1
        else None
    )
    tma = lambda: nvfp4_tma_scaled_mm(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        block_n=config.block_n,
        num_stages=config.num_stages,
        num_compute_warps=config.num_compute_warps,
        grid_scheduler=config.grid_scheduler,
        split_k=config.split_k,
        stage_weight_scales=config.stage_weight_scales,
        output=output,
        partial_output=partial_output,
    )
    baseline = lambda: scaled_mm_nvfp4(mat_a, mat_b, scale_a, scale_b)
    torch.testing.assert_close(tma(), baseline(), atol=2.0, rtol=0.05)
    torch.cuda.synchronize()
    tma_us, scaled_mm_us = median_interleaved_latency(tma, baseline, rounds, iterations)
    return BenchmarkResult(
        "nvfp4",
        n,
        k,
        tma_us,
        scaled_mm_us,
        scaled_mm_us / tma_us,
        config.block_n,
        config.num_stages,
        config.num_compute_warps,
        config.grid_scheduler.value,
        config.split_k,
        config.stage_weight_scales,
    )


def write_heatmap(results: list[BenchmarkResult], output: Path) -> None:
    """Write a CSV and side-by-side speedup heatmaps for MXFP8 and NVFP4."""
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(asdict(result) for result in results)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    formats = ("mxfp8", "nvfp4")
    figure, axes = plt.subplots(1, len(formats), figsize=(16, 7), constrained_layout=True)
    for axis, format_name in zip(axes, formats, strict=True):
        matrix = frame[frame["format"] == format_name].pivot(
            index="k", columns="n", values="speedup"
        )
        sns.heatmap(
            matrix,
            ax=axis,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=1.0,
            cbar_kws={"label": "F.scaled_mm latency / TMA latency"},
        )
        axis.set_title(f"{format_name.upper()} TMA speedup over F.scaled_mm")
        axis.set_xlabel("N")
        axis.set_ylabel("K")
        axis.tick_params(axis="x", labelrotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    figure.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(figure)


@app.command()
def benchmark(
    n: Annotated[str, typer.Option(help="Comma-separated N values, each a multiple of 128")] = (
        "1024,2048,4096,4608,6144,8192,10240,12288,14336,16384,20480,24576,28672,32768"
    ),
    k: Annotated[str, typer.Option(help="Comma-separated K values, each a multiple of 1024")] = (
        "2048,4096,6144,8192,12288,16384,24576,32768"
    ),
    rounds: Annotated[int, typer.Option(min=1)] = 7,
    iterations: Annotated[int, typer.Option(min=1)] = 100,
    output: Path = Path("agent_space/tma_m1_scaled_mm_heatmap"),
    seed: int = 0,
) -> None:
    """Compare M=1 MXFP8 and NVFP4 TMA GEMV with F.scaled_mm over an N×K grid."""
    if torch.cuda.get_device_capability() != (10, 0):
        raise typer.BadParameter("the tuned heatmap benchmark requires an SM100 GPU")
    n_values = parse_sizes(n, "N", 128)
    k_values = parse_sizes(k, "K", 1024)
    results = []
    total = len(n_values) * len(k_values) * 2
    with typer.progressbar(length=total, label="Benchmarking") as progress:
        for format_name, runner in (("mxfp8", run_mxfp8), ("nvfp4", run_nvfp4)):
            for k_value in k_values:
                for n_value in n_values:
                    result = runner(n_value, k_value, rounds, iterations, seed + n_value + k_value)
                    results.append(result)
                    typer.echo(
                        f"{format_name} N={n_value} K={k_value}: "
                        f"TMA {result.tma_us:.3f} us, F.scaled_mm {result.scaled_mm_us:.3f} us, "
                        f"{result.speedup:.2f}x"
                    )
                    progress.update(1)
    write_heatmap(results, output)
    typer.echo(f"Wrote {output.with_suffix('.csv')} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    app()
