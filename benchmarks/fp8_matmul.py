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
from transformer_nuggets.fp8.fp8_matmul import (
    matmul_persistent,
    matmul_tma_persistent,
    matmul_device_tma_persistent,
)
from enum import Enum
import csv

torch._dynamo.config.cache_size_limit = 1000


class FP8Kernel(Enum):
    PERSISTENT = "Persistent"
    PERSISTENT_TMA = "Persistent-TMA"
    DEVICE_TMA = "Device-TMA"
    SCALED_MM = "Scaled-MM"


class ScalingStrategy(Enum):
    PER_TENSOR = "PerTensor"
    PER_ROW = "PerRow"


def is_col_major(stride):
    assert len(stride) == 2, "is_col_major only supports 2D tensors"
    return stride[1] > stride[0] and stride[0] == 1


def get_fp8_matmul(
    A: torch.Tensor, B: torch.Tensor, scaling_strategy: ScalingStrategy, fp8_kernel: FP8Kernel
):
    A_fp8 = A.to(torch.float8_e4m3fn)
    B_fp8 = B.to(torch.float8_e4m3fn)
    A_fp8, B_fp8 = preprocess_data(A_fp8, B_fp8, Float8MMConfig(use_fast_accum=True))

    if scaling_strategy == ScalingStrategy.PER_TENSOR:
        a_scale = torch.tensor(1, device="cuda", dtype=torch.float32)
        b_scale = torch.tensor(1, device="cuda", dtype=torch.float32)
    elif scaling_strategy == ScalingStrategy.PER_ROW:
        a_scale = torch.ones((A_fp8.size(0), 1), device="cuda", dtype=torch.float32)
        b_scale = torch.ones((B_fp8.size(1), 1), device="cuda", dtype=torch.float32).T
    else:
        raise ValueError(f"Invalid scaling strategy: {scaling_strategy}")
    if fp8_kernel == FP8Kernel.PERSISTENT:
        return lambda: matmul_persistent(A_fp8, a_scale, B_fp8, b_scale, torch.bfloat16)
    elif fp8_kernel == FP8Kernel.PERSISTENT_TMA:
        return lambda: matmul_tma_persistent(A_fp8, a_scale, B_fp8, b_scale, torch.bfloat16)
    elif fp8_kernel == FP8Kernel.DEVICE_TMA:
        return lambda: matmul_device_tma_persistent(A_fp8, a_scale, B_fp8, b_scale, torch.bfloat16)
    elif fp8_kernel == FP8Kernel.SCALED_MM:
        return lambda: addmm_float8_unwrapped_inference(
            A_fp8, a_scale, B_fp8, b_scale, output_dtype=torch.bfloat16, use_fast_accum=True
        )
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


@dataclass(frozen=True)
class ExperimentResult:
    bf16_time: float
    fp8_time: float
    bf16_tflops: float
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
        fp8_matmul = torch.compile(fp8_matmul, mode="max-autotune")

    # Warmup phase
    warmup_iterations = 5
    for _ in range(warmup_iterations):
        _ = bf16_matmul(A, B)
        _ = fp8_matmul()
    torch.cuda.synchronize()

    # Actual benchmarking
    bf16_time = benchmark_cuda_function_in_microseconds(lambda: bf16_matmul(A, B))
    fp8_time = benchmark_cuda_function_in_microseconds(fp8_matmul)

    # Calculate TFLOPS
    bf16_tflops = calculate_tflops(config.M, config.N, config.K, bf16_time)
    fp8_tflops = calculate_tflops(config.M, config.N, config.K, fp8_time)

    # Baseline fp8_matmul correctness
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
        speedup = result.bf16_time / result.fp8_time
        tflops_ratio = result.fp8_tflops / result.bf16_tflops
        rows.append(
            [
                config.M,
                config.K,
                config.N,
                config.scaling_strategy,
                config.fp8_kernel,
                config.compile,
                f"{result.bf16_time:.4f}",
                f"{result.fp8_time:.4f}",
                f"{speedup:.2f}x",
                f"{result.bf16_tflops:.2f}",
                f"{result.fp8_tflops:.2f}",
                f"{tflops_ratio:.2f}x",
            ]
        )
    print(tabulate(rows, headers=headers, floatfmt=".4f"))

    if save_path is not None:
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"💾 Results saved to: {save_path}")


def get_configs_varying_k(M: int = 8192, N: int = 8192) -> List[ExperimentConfig]:
    shapes = [(M, K, N) for K in range(512, 8193, 512)]
    scaling_strategies = [ScalingStrategy.PER_ROW]
    compile_options = [False]
    configs = []
    fp8_kernels = [
        FP8Kernel.SCALED_MM,
        # FP8Kernel.PERSISTENT,
        FP8Kernel.PERSISTENT_TMA,
        FP8Kernel.DEVICE_TMA,
    ]

    for (M, K, N), strategy, compile, kernel in itertools.product(
        shapes, scaling_strategies, compile_options, fp8_kernels
    ):
        configs.append(
            ExperimentConfig(
                M=M, K=K, N=N, scaling_strategy=strategy, compile=compile, fp8_kernel=kernel
            )
        )
    return configs


def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df["Speedup"] = df["Speedup"].str.rstrip("x").astype(float)
    df["TFLOPS Ratio"] = df["TFLOPS Ratio"].str.rstrip("x").astype(float)
    return df


def plot_tflops_comparison(df, save_path: Path):
    plt.figure(figsize=(12, 6))
    grouped = df.groupby(["K", "Fp8 Kernel"])
    k_values = sorted(df["K"].unique())
    kernel_types = df["Fp8 Kernel"].unique()
    scaling_strategy = df["Scaling Strategy"].iloc[0]
    m_value = df["M"].iloc[0]
    n_value = df["N"].iloc[0]

    for kernel in kernel_types:
        tflops_values = [grouped.get_group((k, kernel))["FP8 TFLOPS"].values[0] for k in k_values]
        plt.plot(k_values, tflops_values, marker="o", label=kernel.split(".")[-1])

    plt.xlabel("K (Matrix Dimension)")
    plt.ylabel("TFLOPS")
    plt.title(
        f"FP8 Kernel Performance Comparison\nM={m_value}, N={n_value}\nScaling Strategy: {scaling_strategy}"
    )
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xticks(k_values, rotation=45, ha="right")
    plt.tight_layout()

    # Generate the file name and save in the same directory as the CSV file
    file_name = f"fp8_kernel_comparison_{m_value}_{n_value}.png"
    graph_path = save_path.parent / file_name
    plt.savefig(graph_path, dpi=300)
    print(f"TFLOPS comparison plot saved as {graph_path}")


def main(save_path: Optional[str] = None, M: int = 8192, N: int = 8192, graph: bool = False):
    """Benchmark FP8 MatMul with different configurations and optionally graph results.

    Args:
        save_path (Optional[str], optional): Path to save the results. Defaults to None.
        M (int, optional): Number of rows in the first matrix. Defaults to 8192.
        N (int, optional): Number of columns in the second matrix. Defaults to 8192.
        graph_results (bool, optional): Whether to create a graph of the results. Defaults to False.
    """
    torch.random.manual_seed(123)
    configs = get_configs_varying_k(M, N)
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
