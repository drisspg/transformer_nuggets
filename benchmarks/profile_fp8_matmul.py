import torch
from dataclasses import dataclass
from jsonargparse import CLI
import logging
from pathlib import Path

from transformer_nuggets.utils.benchmark import ProfileConfig, profile_function
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

logging.getLogger("transformer_nuggets").setLevel(logging.INFO)


class FP8Kernel(Enum):
    PERSISTENT = "Persistent"
    PERSISTENT_TMA = "Persistent-TMA"
    DEVICE_TMA = "Device-TMA"
    SCALED_MM = "Scaled-MM"


class ScalingStrategy(Enum):
    PER_TENSOR = "PerTensor"
    PER_ROW = "PerRow"


@dataclass(frozen=True)
class ExperimentConfig:
    M: int
    K: int
    N: int
    scaling_strategy: ScalingStrategy
    fp8_kernel: FP8Kernel
    compile: bool


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


def profile_matmul(config: ExperimentConfig, profile_config: ProfileConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(config.M, config.K, device=device, dtype=torch.bfloat16)
    B = torch.randn(config.K, config.N, device=device, dtype=torch.bfloat16)

    fp8_matmul = get_fp8_matmul(A, B, config.scaling_strategy, config.fp8_kernel)
    bf16_matmul = lambda x, y: torch.matmul(x, y)

    if config.compile and config.fp8_kernel == FP8Kernel.SCALED_MM:
        bf16_matmul = torch.compile(bf16_matmul)
        fp8_matmul = torch.compile(fp8_matmul, mode="max-autotune")

    # Warmup phase
    warmup_iterations = 5
    for _ in range(warmup_iterations):
        _ = bf16_matmul(A, B)
        _ = fp8_matmul()
    torch.cuda.synchronize()

    logging.info("Profiling FP8 MatMul")
    fp8_profile = profile_function(profile_config, fp8_matmul)

    return fp8_profile


def main():
    torch.random.manual_seed(123)

    # Define your experiment configuration here
    config = ExperimentConfig(
        M=8192,
        K=8192,
        N=8192,
        scaling_strategy=ScalingStrategy.PER_TENSOR,
        fp8_kernel=FP8Kernel.PERSISTENT_TMA,
        compile=False,
    )

    base = Path(__file__).resolve().parent / Path("data")
    path = base / Path(f"matmul_profile_{config.fp8_kernel.name}.csv")
    # Define your profile configuration here
    profile_config = ProfileConfig(
        file_path=str(path),
        name=f"MatMul Profiling {config.fp8_kernel}",
        cuda=True,
        iters=3,
        warmup_iters=5,
        sync=True,
    )

    fp8_profile = profile_matmul(config, profile_config)

    print(f"\nProfile for config: {config}")
    print("\nFP8 Profile:")
    print(fp8_profile.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    CLI(main)
