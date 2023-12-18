import itertools
from dataclasses import dataclass

from typing import List

import torch

from tabulate import tabulate
from tqdm import tqdm

from transformer_nuggets.fp8.scaled_quant import eager_scaled_quant, scaled_quant
from transformer_nuggets.utils import benchmark_torch_function_in_microseconds

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    numel: int
    high_precision_dtype: torch.dtype
    low_precision_dtype: torch.dtype
    saturated: bool = False


@dataclass(frozen=True)
class ExperimentResult:
    triton_time: float
    pytorch_time: float
    compiled_pytorch_time: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    sizes = [2**21, 2**22, 2**23, 2**24]
    high_precision_dtypes = [torch.bfloat16, torch.float32]
    low_precision_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2]
    saturated = [True, False]
    configs = []
    for size, high_precision_dtype, low_precision_dtype, sat in itertools.product(
        sizes, high_precision_dtypes, low_precision_dtypes, saturated
    ):
        configs.append(
            ExperimentConfig(
                numel=size,
                high_precision_dtype=high_precision_dtype,
                low_precision_dtype=low_precision_dtype,
                saturated=sat,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    high_precision_tensor = torch.randn(
        config.numel, dtype=config.high_precision_dtype, device=device
    )
    triton_hp_tensor = high_precision_tensor.clone()

    eager_abs_max = torch.empty(1, dtype=torch.float32, device=device)
    triton_abs_max = torch.empty(1, dtype=torch.float32, device=device)

    scale = torch.rand(1, dtype=torch.float32, device=device)

    triton_time = benchmark_torch_function_in_microseconds(
        scaled_quant,
        triton_hp_tensor,
        scale,
        triton_abs_max,
        config.low_precision_dtype,
        config.saturated,
    )
    pytorch_time = benchmark_torch_function_in_microseconds(
        eager_scaled_quant,
        high_precision_tensor,
        scale,
        eager_abs_max,
        config.low_precision_dtype,
        config.saturated,
    )
    compiled_pytorch_fn = torch.compile(eager_scaled_quant, fullgraph=True)
    compiled_pytorch_time = benchmark_torch_function_in_microseconds(
        compiled_pytorch_fn,
        high_precision_tensor,
        scale,
        eager_abs_max,
        config.low_precision_dtype,
        config.saturated,
    )
    return ExperimentResult(triton_time=triton_time, pytorch_time=pytorch_time, compiled_pytorch_time=compiled_pytorch_time)


def print_results(experiments: List[Experiment]):
    headers = [
        "numel",
        "high_precision_dtype",
        "low_precision_dtype",
        "saturated",
        "triton_time",
        "pytorch_time",
        "compiled_pytorch_time"
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                experiment.config.numel,
                experiment.config.high_precision_dtype,
                experiment.config.low_precision_dtype,
                experiment.config.saturated,
                experiment.result.triton_time,
                experiment.result.pytorch_time,
                experiment.result.compiled_pytorch_time,
            ]
        )
    print(tabulate(rows, headers=headers))


def main():
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    main()
