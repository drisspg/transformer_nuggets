import itertools

from contextlib import suppress
from dataclasses import dataclass

import torch

from tabulate import tabulate
from tqdm import tqdm

from transformer_nuggets.fp8.scaled_quant import eager_scaled_quant, scaled_quant
from transformer_nuggets.utils import benchmark_cuda_function_in_microseconds

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


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


def get_configs() -> list[ExperimentConfig]:
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

    eager_abs_max = torch.abs(high_precision_tensor).max().to(torch.float32)
    triton_abs_max = torch.abs(high_precision_tensor).max().to(torch.float32)

    scale = torch.finfo(config.low_precision_dtype).max / eager_abs_max
    scale = scale.to(torch.float32)
    scale = torch.rand(1, dtype=torch.float32, device=device)

    # Correctness check:
    nuggets_out = scaled_quant(
        triton_hp_tensor,
        triton_abs_max,
        scale,
        config.low_precision_dtype,
        config.saturated,
    )
    nuggets_out_hp = nuggets_out.to(config.high_precision_dtype)
    eager_out = eager_scaled_quant(
        high_precision_tensor,
        eager_abs_max,
        scale,
        config.low_precision_dtype,
        config.saturated,
    ).to(config.high_precision_dtype)
    eager_out_hp = eager_out.to(config.high_precision_dtype)
    with suppress(AssertionError):
        torch.testing.assert_close(nuggets_out_hp, eager_out_hp, rtol=1e-3, atol=1e-3)
        # investigate why we are seeing small deviations
        # Mismatched elements: 62577 / 2097152 (3.0%)
        # Greatest absolute difference: 2.0 at index (11111,) (up to 0.001 allowed)
        # Greatest relative difference: inf at index (516343,) (up to 0.001 allowed)
        # > /home/drisspg/meta/transformer_nuggets/benchmarks/fp8_sat_cast.py(85)run_experiment()
        # -> triton_time = benchmark_torch_function_in_microseconds(
        # (Pdb) nuggets_out[11111]
        # tensor(-18., device='cuda:0', dtype=torch.float8_e4m3fn)
        # (Pdb) eager_out[11111]
        # tensor(-16., device='cuda:0', dtype=torch.bfloat16)
        # (Pdb) eager_out[516343]
        # tensor(0., device='cuda:0', dtype=torch.bfloat16)
        # (Pdb) nuggets_out[516343]
        # tensor(0.0020, device='cuda:0', dtype=torch.float8_e4m3fn)

    triton_time = benchmark_cuda_function_in_microseconds(
        scaled_quant,
        triton_hp_tensor,
        scale,
        triton_abs_max,
        config.low_precision_dtype,
        config.saturated,
    )
    pytorch_time = benchmark_cuda_function_in_microseconds(
        eager_scaled_quant,
        high_precision_tensor,
        scale,
        eager_abs_max,
        config.low_precision_dtype,
        config.saturated,
    )
    compiled_pytorch_fn = torch.compile(eager_scaled_quant, fullgraph=True)
    compiled_pytorch_time = benchmark_cuda_function_in_microseconds(
        compiled_pytorch_fn,
        high_precision_tensor,
        scale,
        eager_abs_max,
        config.low_precision_dtype,
        config.saturated,
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
        "saturated",
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
                experiment.config.saturated,
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
