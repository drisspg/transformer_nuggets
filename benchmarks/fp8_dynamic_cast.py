import itertools

from dataclasses import dataclass

import torch

from tabulate import tabulate
from tqdm import tqdm

from transformer_nuggets.fp8.scaled_quant import (
    dynamic_scaled_quant,
    eager_dynamic_scaled_quant,
)
from transformer_nuggets.utils import benchmark_cuda_function_in_microseconds

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    numel: int
    high_precision_dtype: torch.dtype
    low_precision_dtype: torch.dtype


@dataclass(frozen=True)
class ExperimentResult:
    triton_time: float
    pytorch_time: float
    compiled_pytorch_time: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> list[ExperimentConfig]:
    # We hang for anything bigger than this
    # sizes = [2**21, 2**22, 2**23, 2**24]
    sizes = [2**21, 2**22]
    high_precision_dtypes = [torch.float32]
    low_precision_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2]
    configs = []
    for size, high_precision_dtype, low_precision_dtype in itertools.product(
        sizes, high_precision_dtypes, low_precision_dtypes
    ):
        configs.append(
            ExperimentConfig(
                numel=size,
                high_precision_dtype=high_precision_dtype,
                low_precision_dtype=low_precision_dtype,
            )
        )
    return configs


def correctness_check(hp_tensor, triton_tensor, config):
    # Correctness check:
    nuggets_out = dynamic_scaled_quant(
        triton_tensor,
        config.low_precision_dtype,
    ).to(config.high_precision_dtype)

    eager_out = eager_dynamic_scaled_quant(
        hp_tensor,
        config.low_precision_dtype,
    ).to(config.high_precision_dtype)

    compiled_pytorch_fn = torch.compile(eager_dynamic_scaled_quant, fullgraph=True)
    compiled_out = compiled_pytorch_fn(
        hp_tensor,
        config.low_precision_dtype,
    ).to(config.high_precision_dtype)

    print(f"Deviation between Triton and Nuggets: {torch.abs(nuggets_out - eager_out).max()}")
    print(
        f"Deviation between Eager and Compiled PyTorch: {torch.abs(eager_out - compiled_out).max()}"
    )

    # Find the index of the maximum deviation
    max_dev_index = torch.abs(nuggets_out - eager_out).argmax().item()
    print(f"nuggets_out tensor value: {nuggets_out.flatten()[max_dev_index]:.4f}")
    print(f"eager_out tensor value: {eager_out.flatten()[max_dev_index]:.4f}")


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    high_precision_tensor = torch.randn(
        config.numel, dtype=config.high_precision_dtype, device=device
    )
    triton_hp_tensor = high_precision_tensor.clone()

    # Triton does different rounding as far as I can tell
    if True:
        correctness_check(high_precision_tensor, triton_hp_tensor, config)

    triton_time = benchmark_cuda_function_in_microseconds(
        dynamic_scaled_quant,
        triton_hp_tensor,
        config.low_precision_dtype,
    )
    pytorch_time = benchmark_cuda_function_in_microseconds(
        eager_dynamic_scaled_quant,
        high_precision_tensor,
        config.low_precision_dtype,
    )
    compiled_pytorch_fn = torch.compile(eager_dynamic_scaled_quant, fullgraph=True)
    compiled_pytorch_time = benchmark_cuda_function_in_microseconds(
        compiled_pytorch_fn,
        high_precision_tensor,
        config.low_precision_dtype,
    )
    return ExperimentResult(
        triton_time=triton_time,
        pytorch_time=pytorch_time,
        compiled_pytorch_time=compiled_pytorch_time,
    )


def print_results(experiments: list[Experiment]):
    headers = [
        "numel",
        "high_precision_dtype",
        "low_precision_dtype",
        "triton_time",
        "pytorch_time",
        "compiled_pytorch_time",
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                experiment.config.numel,
                experiment.config.high_precision_dtype,
                experiment.config.low_precision_dtype,
                experiment.result.triton_time,
                experiment.result.pytorch_time,
                experiment.result.compiled_pytorch_time,
            ]
        )
    print(tabulate(rows, headers=headers))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    main()
