import itertools
from dataclasses import dataclass
from typing import List, Optional
import torch
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from jsonargparse import CLI
from pathlib import Path
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds
from torchao.float8.inference import (
    addmm_float8_unwrapped_inference,
    preprocess_data,
    Float8MMConfig,
)

try:
    from transformer_nuggets.fp8.fp8_matmul import (
        matmul_persistent,
        matmul_tma_persistent,
        matmul_device_tma_persistent,
    )
except ModuleNotFoundError:
    print("Triton version not new enough")
    pass

from datetime import datetime
from enum import Enum
import csv
import logging

from torchao.ops import mx_fp8_bf16

def ceil_div(a, b):
    return (a + b - 1) // b


torch._dynamo.config.cache_size_limit = 10000
logging.getLogger("transformer_nuggets").setLevel(logging.INFO)
torch._inductor.config.max_autotune_gemm_backends = "TRITON"
CHECK = False


class FP8Kernel(Enum):
    PERSISTENT = "Persistent"
    PERSISTENT_TMA = "Persistent-TMA"
    DEVICE_TMA = "Device-TMA"
    SCALED_MM = "Scaled-MM"
    CUTLASS_MX = "Cutlass-MX-FP8"


class ScalingStrategy(Enum):
    PER_TENSOR = "PerTensor"
    PER_ROW = "PerRow"
    E8M0 = "E8M0"


def is_col_major(stride):
    assert len(stride) == 2, "is_col_major only supports 2D tensors"
    return stride[1] > stride[0] and stride[0] == 1


def get_e8_scales(A: torch.Tensor, B: torch.Tensor):
    M, K = A.shape
    _, N = B.shape
    n_a_rows = ceil_div(M, 128) * 128
    n_a_cols = ceil_div(K, 32)
    n_b_rows = ceil_div(N, 128) * 128
    n_b_cols = ceil_div(K, 32)
    a_scales = torch.randn(n_a_rows, n_a_cols, dtype=torch.float32, device="cuda").to(torch.float8_e8m0fnu)
    b_scales = torch.randn(n_b_rows, n_b_cols, dtype=torch.float32, device="cuda").to(torch.float8_e8m0fnu)

    return a_scales, b_scales


def get_fp8_matmul(
    A: torch.Tensor, B: torch.Tensor, scaling_strategy: ScalingStrategy, fp8_kernel: FP8Kernel
):
    A_fp8 = A.to(torch.float8_e4m3fn)
    B_fp8 = B.to(torch.float8_e4m3fn)
    A_fp8, B_fp8 = preprocess_data(A_fp8, B_fp8, Float8MMConfig(use_fast_accum=True))

    # Handle E8M0 format for supported kernels
    if scaling_strategy == ScalingStrategy.E8M0:
        if fp8_kernel not in [FP8Kernel.CUTLASS_MX, FP8Kernel.SCALED_MM]:
            raise ValueError(
                "E8M0 scaling strategy is only supported by MX_FP8 and SCALED_MM kernels"
            )

    if scaling_strategy == ScalingStrategy.PER_TENSOR:
        a_scale = torch.tensor(1, device="cuda", dtype=torch.float32)
        b_scale = torch.tensor(1, device="cuda", dtype=torch.float32)
    elif scaling_strategy == ScalingStrategy.PER_ROW:
        a_scale = torch.ones((A_fp8.size(0), 1), device="cuda", dtype=torch.float32)
        b_scale = torch.ones((B_fp8.size(1), 1), device="cuda", dtype=torch.float32).T
    elif scaling_strategy == ScalingStrategy.E8M0:
        a_scale, b_scale = get_e8_scales(A_fp8, B_fp8)
    else:
        raise ValueError(f"Invalid scaling strategy: {scaling_strategy}")

    if fp8_kernel == FP8Kernel.PERSISTENT:
        return lambda: matmul_persistent(A_fp8, a_scale, B_fp8, b_scale, torch.bfloat16)
    elif fp8_kernel == FP8Kernel.PERSISTENT_TMA:
        return lambda: matmul_tma_persistent(A_fp8, a_scale, B_fp8, b_scale, torch.bfloat16)
    elif fp8_kernel == FP8Kernel.DEVICE_TMA:
        return lambda: matmul_device_tma_persistent(A_fp8, a_scale, B_fp8, b_scale, torch.bfloat16)
    elif fp8_kernel == FP8Kernel.SCALED_MM:
        if scaling_strategy == ScalingStrategy.E8M0:
            # Use the scales we computed earlier for E8M0
            return lambda: torch._scaled_mm(
                A_fp8,
                B_fp8,
                a_scale,
                b_scale,
                out_dtype=torch.bfloat16,
            )
        return lambda: addmm_float8_unwrapped_inference(
            A_fp8, a_scale, B_fp8, b_scale, output_dtype=torch.bfloat16, use_fast_accum=True
        )
    elif fp8_kernel == FP8Kernel.CUTLASS_MX:
        assert (
            scaling_strategy == ScalingStrategy.E8M0
        ), "E8M0 scaling strategy is required for MX_FP8"
        return lambda: mx_fp8_bf16(A_fp8, B_fp8, a_scale, b_scale)
    else:
        raise ValueError(f"Invalid FP8 kernel: {fp8_kernel}")


@dataclass(frozen=True)
class ExperimentConfig:
    M: int
    K: int
    N: int
    scaling_strategy: ScalingStrategy
    fp8_kernel: FP8Kernel
    compile: bool
    bf16: bool


@dataclass(frozen=True)
class ExperimentResult:
    bf16_time: Optional[float]
    fp8_time: float
    bf16_tflops: Optional[float]
    fp8_tflops: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def calculate_tflops(M: int, N: int, K: int, time_us: float) -> float:
    """Calculate TFLOPS (Tera Floating Point Operations Per Second)"""
    flops = 2 * M * N * K  # Number of floating point operations for matrix multiplication
    tflops = (flops / time_us) / 1e6  # Convert to TFLOPS
    return tflops


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(config.M, config.K, device=device, dtype=torch.bfloat16)
    B = torch.randn(config.K, config.N, device=device, dtype=torch.bfloat16)

    bf16_matmul = lambda x, y: torch.matmul(x, y)
    fp8_matmul = get_fp8_matmul(A, B, config.scaling_strategy, config.fp8_kernel)

    if config.compile and config.fp8_kernel == FP8Kernel.SCALED_MM:
        bf16_matmul = torch.compile(bf16_matmul)
        fp8_matmul = torch.compile(fp8_matmul, mode="max-autotune-no-cudagraphs", dynamic=False)

    # Warmup phase
    warmup_iterations = 5
    for _ in range(warmup_iterations):
        if config.bf16:
            _ = bf16_matmul(A, B)
        _ = fp8_matmul()
    torch.cuda.synchronize()

    # Actual benchmarking

    bf16_time = (
        benchmark_cuda_function_in_microseconds(lambda: bf16_matmul(A, B)) if config.bf16 else None
    )
    fp8_time = benchmark_cuda_function_in_microseconds(fp8_matmul)

    # Calculate TFLOPS
    bf16_tflops = calculate_tflops(config.M, config.N, config.K, bf16_time) if bf16_time else None
    fp8_tflops = calculate_tflops(config.M, config.N, config.K, fp8_time)

    # Baseline fp8_matmul correctness
    if CHECK:
        scaled_mm_base = get_fp8_matmul(A, B, config.scaling_strategy, FP8Kernel.SCALED_MM)
        out_base = scaled_mm_base()
        out = fp8_matmul()
        # Failing on one sample with large N
        torch.testing.assert_close(out, out_base)

    return ExperimentResult(
        bf16_time=bf16_time, fp8_time=fp8_time, bf16_tflops=bf16_tflops, fp8_tflops=fp8_tflops
    )


def print_results(experiments: List[Experiment], save_path: Optional[Path] = None):
    headers = [
        "M",
        "K",
        "N",
        "Scaling Strategy",
        "Fp8 Kernel",
        "Compiled",
        "BF16 Time (ms)",
        "FP8 Time (ms)",
        "Speedup",
        "BF16 TFLOPS",
        "FP8 TFLOPS",
        "TFLOPS Ratio",
    ]
    rows = []
    for experiment in experiments:
        config = experiment.config
        result = experiment.result

        # Format values handling None cases
        bf16_time = f"{result.bf16_time:.4f}" if result.bf16_time is not None else "N/A"
        fp8_time = f"{result.fp8_time:.4f}"
        bf16_tflops = f"{result.bf16_tflops:.2f}" if result.bf16_tflops is not None else "N/A"
        fp8_tflops = f"{result.fp8_tflops:.2f}"

        # Calculate ratios only if bf16 results exist
        if result.bf16_time is not None:
            speedup = f"{(result.bf16_time / result.fp8_time):.2f}x"
            tflops_ratio = f"{(result.fp8_tflops / result.bf16_tflops):.2f}x"
        else:
            speedup = "N/A"
            tflops_ratio = "N/A"

        rows.append(
            [
                config.M,
                config.K,
                config.N,
                config.scaling_strategy.value,
                config.fp8_kernel.value,
                config.compile,
                bf16_time,
                fp8_time,
                speedup,
                bf16_tflops,
                fp8_tflops,
                tflops_ratio,
            ]
        )

    print(tabulate(rows, headers=headers, floatfmt=".4f"))

    if save_path is not None:
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"💾 Results saved to: {save_path}")


def get_configs_varying_k(
    M: int = 8192, N: int = 8192, bf16: bool = False
) -> List[ExperimentConfig]:
    shapes = [(M, K, N) for K in range(1024, 16385, 1024)]
    scaling_strategies = [ScalingStrategy.E8M0]
    compile_options = [False]
    configs = []
    fp8_kernels = [
        FP8Kernel.SCALED_MM,
        # FP8Kernel.PERSISTENT,
        # FP8Kernel.PERSISTENT_TMA,
        # FP8Kernel.DEVICE_TMA,
        FP8Kernel.CUTLASS_MX,
    ]

    for (M, K, N), strategy, compile, kernel in itertools.product(
        shapes, scaling_strategies, compile_options, fp8_kernels
    ):
        configs.append(
            ExperimentConfig(
                M=M,
                K=K,
                N=N,
                scaling_strategy=strategy,
                compile=compile,
                fp8_kernel=kernel,
                bf16=bf16,
            )
        )
    return configs


def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    # df["Speedup"] = df["Speedup"].str.rstrip("x").astype(float)
    # df["TFLOPS Ratio"] = df["TFLOPS Ratio"].str.rstrip("x").astype(float)
    return df


def plot_tflops_comparison(df, save_path: Path):
    plt.figure(figsize=(12, 6))
    grouped = df.groupby(["K", "Fp8 Kernel"])
    k_values = sorted(df["K"].unique())
    kernel_types = df["Fp8 Kernel"].unique()
    scaling_strategy = df["Scaling Strategy"].iloc[0]
    m_value = df["M"].iloc[0]
    n_value = df["N"].iloc[0]

    # Plot FP8 kernel performance
    for kernel in kernel_types:
        tflops_values = [grouped.get_group((k, kernel))["FP8 TFLOPS"].values[0] for k in k_values]
        plt.plot(k_values, tflops_values, marker="o", label=f"FP8 - {kernel}")

    # Check if BF16 data exists and plot it
    has_bf16 = df["BF16 TFLOPS"].notna().any()
    if has_bf16:
        # Since there's only one BF16 implementation, we can extract it directly
        bf16_tflops_values = []
        for k in k_values:
            # Get the first entry for each K value since BF16 is the same regardless of kernel
            k_group = df[df["K"] == k]
            if not k_group.empty and not k_group["BF16 TFLOPS"].isna().all():
                bf16_tflops_values.append(k_group["BF16 TFLOPS"].iloc[0])
            else:
                # Handle case where BF16 data might be missing for some K values
                bf16_tflops_values.append(None)
        
        # Filter out None values for plotting
        valid_indices = [i for i, val in enumerate(bf16_tflops_values) if val is not None]
        if valid_indices:
            valid_k_values = [k_values[i] for i in valid_indices]
            valid_bf16_values = [bf16_tflops_values[i] for i in valid_indices]
            plt.plot(valid_k_values, valid_bf16_values, marker="s", linestyle="--", 
                    color="red", linewidth=2, label="BF16 Baseline")

    plt.xlabel("K (Matrix Dimension)")
    plt.ylabel("TFLOPS")
    title = f"Matrix Multiplication Performance Comparison\nM={m_value}, N={n_value}\nScaling Strategy: {scaling_strategy}"
    if has_bf16:
        title = "FP8 vs BF16 " + title
    else:
        title = "FP8 " + title
    plt.title(title)
    
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xticks(k_values, rotation=45, ha="right")
    plt.tight_layout()

    # Generate the file name and save in the same directory as the CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "fp8_bf16_comparison" if has_bf16 else "fp8_kernel_comparison"
    file_name = f"{prefix}_{m_value}_{n_value}_{timestamp}.png"
    graph_path = save_path.parent / file_name
    plt.savefig(graph_path, dpi=300)
    print(f"TFLOPS comparison plot saved as {graph_path}")


def main(
    save_path: Optional[str] = None,
    M: int = 8192,
    N: int = 8192,
    graph: bool = False,
    bf_16: bool = False,
):
    """Benchmark FP8 MatMul with different configurations and optionally graph results.

    Args:
        save_path (Optional[str], optional): Path to save the results. Defaults to None.
        M (int, optional): Number of rows in the first matrix. Defaults to 8192.
        N (int, optional): Number of columns in the second matrix. Defaults to 8192.
        graph_results (bool, optional): Whether to create a graph of the results. Defaults to False.
        bf_16 (bool, optional): Whether to use BF16 for the baseline. Defaults to False.
    """
    torch.random.manual_seed(123)
    configs = get_configs_varying_k(M, N, bf16=bf_16)
    results = []
    if save_path is not None:
        save_path = Path(save_path)
        save_path = save_path.with_suffix(".csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))
    print_results(results, save_path)

    if graph and save_path is not None:
        df = load_and_process_data(save_path)
        plot_tflops_comparison(df, save_path)


if __name__ == "__main__":
    CLI(main)
