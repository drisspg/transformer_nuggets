#!/usr/bin/env python3
"""
Benchmark comparing PyTorch native attention vs CUTE implementation.
This benchmark has no dependencies on other flash-attn components.
"""

import csv
import itertools
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from jsonargparse import CLI
from tabulate import tabulate
from tqdm import tqdm

from torch.nn.attention import sdpa_kernel, SDPBackend

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

try:
    # pyrefly: ignore  # import-error
    from flash_attn.cute import flash_attn_func

    CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False
    print(
        "Warning: CUTE implementation not available. Install with: pip install nvidia-cutlass-dsl>=4.1.0.dev0"
    )


# Type definitions
DtypeString = Literal["bfloat16", "float16", "float32"]
Backend = Literal["pytorch", "cute"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment."""

    batch_size: int
    seqlen: int
    nheads: int
    headdim: int
    causal: bool
    dtype: torch.dtype

    def asdict(self):
        """Convert to dictionary for display."""
        d = asdict(self)
        d["dtype"] = str(self.dtype).replace("torch.", "")
        d["shape(B,S,H,D)"] = (self.batch_size, self.seqlen, self.nheads, self.headdim)
        d.pop("batch_size")
        d.pop("seqlen")
        d.pop("nheads")
        d.pop("headdim")
        return d


@dataclass(frozen=True)
class ExperimentResults:
    """Results from a single experiment."""

    time_ms: float
    tflops: float
    bandwidth_gb_s: float
    memory_footprint_mb: float
    max_diff: float | None = None
    mean_diff: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class Experiment:
    """Complete experiment with config and results."""

    config: ExperimentConfig
    results: dict[str, ExperimentResults]  # backend -> results

    def asdict(self):
        """Flatten for tabular display."""
        dict1 = self.config.asdict()
        dict2 = {}
        for backend, result in self.results.items():
            if result.error:
                dict2[f"{backend}_time_ms"] = float("nan")
                dict2[f"{backend}_tflops"] = float("nan")
                dict2[f"{backend}_bandwidth_gb_s"] = float("nan")
                dict2[f"{backend}_error"] = result.error
            else:
                dict2[f"{backend}_time_ms"] = result.time_ms
                dict2[f"{backend}_tflops"] = result.tflops
                dict2[f"{backend}_bandwidth_gb_s"] = result.bandwidth_gb_s
                if backend == "cute":
                    dict2["max_diff"] = result.max_diff
                    dict2["mean_diff"] = result.mean_diff
        return {**dict1, **dict2}


def compute_flops(
    batch_size: int, seqlen: int, nheads: int, headdim: int, causal: bool = False
) -> int:
    """Compute theoretical FLOPs for attention."""
    batch_heads = batch_size * nheads

    qk_flops = batch_heads * seqlen * seqlen * headdim * 2
    softmax_flops = batch_heads * seqlen * seqlen
    av_flops = batch_heads * seqlen * seqlen * headdim * 2

    total_flops = qk_flops + softmax_flops + av_flops

    # Causal attention reduces computation by ~half
    if causal:
        total_flops = total_flops // 2

    return total_flops


def compute_bandwidth(config: ExperimentConfig, time_us: float) -> float:
    """Compute achieved memory bandwidth in GB/s."""
    batch_size, seqlen, nheads, headdim = (
        config.batch_size,
        config.seqlen,
        config.nheads,
        config.headdim,
    )
    dtype_size = torch.finfo(config.dtype).bits / 8

    # Memory accesses: Q, K, V reads + O write
    qkv_size = 3 * batch_size * seqlen * nheads * headdim * dtype_size
    o_size = batch_size * seqlen * nheads * headdim * dtype_size
    total_bytes = qkv_size + o_size

    # Convert to GB/s
    time_seconds = time_us / 1e6
    bandwidth_gb_s = (total_bytes / 1e9) / time_seconds

    return bandwidth_gb_s


def compute_memory_footprint(config: ExperimentConfig) -> float:
    """Compute memory footprint in MB."""
    batch_size, seqlen, nheads, headdim = (
        config.batch_size,
        config.seqlen,
        config.nheads,
        config.headdim,
    )
    dtype_size = torch.finfo(config.dtype).bits / 8

    # Q, K, V, O tensors
    tensor_size = batch_size * seqlen * nheads * headdim * dtype_size
    total_size = 4 * tensor_size  # Q, K, V, O

    return total_size / (1024 * 1024)  # Convert to MB


def generate_experiment_configs(
    batch_sizes: list[int],
    seqlens: list[int],
    num_heads: list[int],
    head_dims: list[int],
    causal_modes: list[bool],
    dtype: torch.dtype,
) -> list[ExperimentConfig]:
    """Generate all experiment configurations."""
    configs = []

    for batch_size, seqlen, nheads, headdim, causal in itertools.product(
        batch_sizes, seqlens, num_heads, head_dims, causal_modes
    ):
        configs.append(
            ExperimentConfig(
                batch_size=batch_size,
                seqlen=seqlen,
                nheads=nheads,
                headdim=headdim,
                causal=causal,
                dtype=dtype,
            )
        )

    return configs


def run_pytorch_attention(
    config: ExperimentConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> ExperimentResults:
    """Run PyTorch scaled_dot_product_attention."""
    # Transpose for PyTorch (B, H, S, D) format
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    attention_pytorch = lambda q, k, v: F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=config.causal
    )

    try:
        # Warmup and benchmark
        attention_pytorch(q_t, k_t, v_t)
        time_us = benchmark_cuda_function_in_microseconds(attention_pytorch, q_t, k_t, v_t)

        # Compute metrics
        time_ms = time_us / 1000
        flops = compute_flops(
            config.batch_size, config.seqlen, config.nheads, config.headdim, config.causal
        )
        tflops = (flops / time_us) / 1e6
        bandwidth = compute_bandwidth(config, time_us)
        memory_mb = compute_memory_footprint(config)

        return ExperimentResults(
            time_ms=time_ms,
            tflops=tflops,
            bandwidth_gb_s=bandwidth,
            memory_footprint_mb=memory_mb,
        )
    except Exception as e:
        return ExperimentResults(
            time_ms=float("nan"),
            tflops=float("nan"),
            bandwidth_gb_s=float("nan"),
            memory_footprint_mb=compute_memory_footprint(config),
            error=str(e),
        )


def run_cute_attention(
    config: ExperimentConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pytorch_output: torch.Tensor,
) -> ExperimentResults:
    """Run CUTE flash attention."""
    if not CUTE_AVAILABLE:
        return ExperimentResults(
            time_ms=float("nan"),
            tflops=float("nan"),
            bandwidth_gb_s=float("nan"),
            memory_footprint_mb=compute_memory_footprint(config),
            error="CUTE not available",
        )

    try:
        # Benchmark CUTE
        time_us = benchmark_cuda_function_in_microseconds(
            flash_attn_func, q, k, v, causal=config.causal
        )

        # Get output for correctness check
        cute_output, _ = flash_attn_func(q, k, v, causal=config.causal)

        # Check correctness
        max_diff = torch.max(torch.abs(pytorch_output - cute_output.transpose(1, 2))).item()
        mean_diff = torch.mean(torch.abs(pytorch_output - cute_output.transpose(1, 2))).item()

        # Compute metrics
        time_ms = time_us / 1000
        flops = compute_flops(
            config.batch_size, config.seqlen, config.nheads, config.headdim, config.causal
        )
        tflops = (flops / time_us) / 1e6
        bandwidth = compute_bandwidth(config, time_us)
        memory_mb = compute_memory_footprint(config)

        return ExperimentResults(
            time_ms=time_ms,
            tflops=tflops,
            bandwidth_gb_s=bandwidth,
            memory_footprint_mb=memory_mb,
            max_diff=max_diff,
            mean_diff=mean_diff,
        )
    except Exception as e:
        return ExperimentResults(
            time_ms=float("nan"),
            tflops=float("nan"),
            bandwidth_gb_s=float("nan"),
            memory_footprint_mb=compute_memory_footprint(config),
            error=str(e),
        )


def run_single_experiment(config: ExperimentConfig) -> Experiment:
    """Run a single experiment configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA not available. This benchmark requires GPU.")

    # Generate inputs
    q = torch.randn(
        config.batch_size,
        config.seqlen,
        config.nheads,
        config.headdim,
        device=device,
        dtype=config.dtype,
    )
    k = torch.randn(
        config.batch_size,
        config.seqlen,
        config.nheads,
        config.headdim,
        device=device,
        dtype=config.dtype,
    )
    v = torch.randn(
        config.batch_size,
        config.seqlen,
        config.nheads,
        config.headdim,
        device=device,
        dtype=config.dtype,
    )

    # Run PyTorch first
    pytorch_results = run_pytorch_attention(config, q, k, v)

    # Get PyTorch output for correctness check
    pytorch_output = None
    if not pytorch_results.error:
        q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        pytorch_output = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=config.causal
        )

    # Run CUTE
    # pyrefly: ignore  # bad-argument-type
    cute_results = run_cute_attention(config, q, k, v, pytorch_output)

    return Experiment(
        config=config,
        results={
            "pytorch": pytorch_results,
            "cute": cute_results,
        },
    )


def calculate_speedup(experiments: list[Experiment]) -> dict:
    """Calculate average, min, and max speedups."""
    speedups = []

    for exp in experiments:
        if "pytorch" in exp.results and "cute" in exp.results:
            pytorch_time = exp.results["pytorch"].time_ms
            cute_time = exp.results["cute"].time_ms

            if not (np.isnan(pytorch_time) or np.isnan(cute_time)):
                speedup = pytorch_time / cute_time
                speedups.append(speedup)

    if not speedups:
        return {"avg": float("nan"), "min": float("nan"), "max": float("nan")}

    return {
        "avg": np.mean(speedups),
        "min": np.min(speedups),
        "max": np.max(speedups),
    }


def print_results(experiments: list[Experiment], save_path: str | None = None):
    """Print results in tabular format."""
    # Convert experiments to table data
    table_data = [exp.asdict() for exp in experiments]

    # Print main results
    print("\n" + "=" * 80)
    print("PyTorch vs CUTE Flash Attention Benchmark Results")
    print("=" * 80)
    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    # Calculate and print speedup summary
    speedup_stats = calculate_speedup(experiments)
    print("\n" + "=" * 80)
    print("SPEEDUP SUMMARY (PyTorch time / CUTE time)")
    print("=" * 80)
    print(f"Average speedup: {speedup_stats['avg']:.2f}x")
    print(f"Min speedup:     {speedup_stats['min']:.2f}x")
    print(f"Max speedup:     {speedup_stats['max']:.2f}x")

    # Calculate average metrics
    valid_experiments = [
        exp
        for exp in experiments
        if not exp.results.get("cute", ExperimentResults(0, 0, 0, 0)).error
    ]

    if valid_experiments:
        avg_cute_tflops = np.mean([exp.results["cute"].tflops for exp in valid_experiments])
        avg_pytorch_tflops = np.mean([exp.results["pytorch"].tflops for exp in valid_experiments])
        max_errors = [exp.results["cute"].max_diff for exp in valid_experiments]
        mean_errors = [exp.results["cute"].mean_diff for exp in valid_experiments]

        print(f"\nAverage CUTE performance:    {avg_cute_tflops:.2f} TFLOPs/s")
        print(f"Average PyTorch performance: {avg_pytorch_tflops:.2f} TFLOPs/s")
        print(f"Max numerical difference:    {max(max_errors):.2e}")
        # pyrefly: ignore  # no-matching-overload
        print(f"Mean numerical difference:   {np.mean(mean_errors):.2e}")

    # Save to CSV if requested
    if save_path:
        with open(save_path, "w", newline="") as csvfile:
            if table_data:
                writer = csv.DictWriter(csvfile, fieldnames=table_data[0].keys())
                writer.writeheader()
                writer.writerows(table_data)
        print(f"\nResults saved to {save_path}")


@sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
def main(
    dtype: DtypeString = "float16",
    batch_sizes: list[int] = [4],
    seqlens: list[int] = [32768],
    num_heads: list[int] = [32],
    head_dims: list[int] = [128],
    causal: bool = False,
    include_non_causal: bool = False,
    save_path: str | None = None,
) -> None:
    """Run benchmark comparing PyTorch native attention vs CUTE implementation.

    Usage Examples:
        # Basic usage with defaults
        python benchmark_cute.py

        # Benchmark specific configurations
        python benchmark_cute.py --batch_sizes 1 2 4 --seqlens 1024 2048 4096

        # Test both causal and non-causal
        python benchmark_cute.py --include_non_causal

        # Save results to CSV
        python benchmark_cute.py --save_path results.csv

        # Use different dtype
        python benchmark_cute.py --dtype bfloat16

    Args:
        dtype: Data type for tensors (bfloat16, float16, float32)
        batch_sizes: List of batch sizes to benchmark
        seqlens: List of sequence lengths to benchmark
        num_heads: List of number of attention heads to benchmark
        head_dims: List of head dimensions to benchmark
        causal: Whether to test causal attention
        include_non_causal: Whether to also test non-causal attention
        save_path: Path to save results as CSV
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    causal_modes = []
    if causal:
        causal_modes.append(True)
    if include_non_causal:
        causal_modes.append(False)

    if not causal_modes:
        causal_modes = [True]  # Default to causal

    # Check prerequisites
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires GPU.")
        return

    if not CUTE_AVAILABLE:
        print("CUTE implementation not available. Exiting.")
        return

    print("Device: cuda")
    print(f"Dtype: {dtype}")
    print(f"CUTE Available: {CUTE_AVAILABLE}")
    print()

    configs = generate_experiment_configs(
        batch_sizes=batch_sizes,
        seqlens=seqlens,
        num_heads=num_heads,
        head_dims=head_dims,
        causal_modes=causal_modes,
        dtype=torch_dtype,
    )

    experiments = []
    # pyrefly: ignore  # not-iterable
    for config in tqdm(configs, desc="Running experiments"):
        experiment = run_single_experiment(config)
        experiments.append(experiment)

    # Print results
    print_results(experiments, save_path)


if __name__ == "__main__":
    torch.manual_seed(0)
    CLI(main, as_positional=False)
