import csv
import itertools

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import transformer_nuggets as nugs
import transformer_nuggets.quant.qlora as qlora
from jsonargparse import CLI
from tabulate import tabulate
from tqdm import tqdm
from transformer_nuggets.quant import NF4Tensor

bnb_available = False


logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    import bitsandbytes as bnb  # noqa: F401

    bnb_available = True
except ImportError:
    logging.warning(
        "Could not import bitsandbytes, make sure you have installed it `pip install bitsandbytes` "
    )


@dataclass
class ExperimentConfig:
    embed_dim: int
    bsz: int
    seqlen: int
    device: torch.device
    op: str
    dynamic: bool


@dataclass
class ExperimentResult:
    unquantized: float
    eager_qlora: float
    compiled_qlora: float
    bnb_time: float
    triton_eager_time: float


def linear_experiment(config: ExperimentConfig) -> ExperimentResult:
    input_weight = qlora.build_input_weight(config.embed_dim, config.device)
    sample_input = qlora.get_sample_inputs(
        config.bsz,
        config.seqlen,
        config.embed_dim,
        config.device,
    )
    qlora_weight = NF4Tensor.from_tensor(input_weight.clone())
    compiled_qlora_linear = torch.compile(qlora.linear_nf4, fullgraph=True, dynamic=config.dynamic)
    if bnb_available:
        bnb_linear = qlora.build_bitsandbytes_linear(input_weight, config.device)

    # warmup
    for _ in range(3):
        F.linear(sample_input, input_weight)
        qlora.linear_nf4(sample_input, qlora_weight)
        compiled_qlora_linear(sample_input, qlora_weight)
        qlora.linear_nf4_trtion(sample_input, qlora_weight)
        if bnb_available:
            bnb_linear(sample_input)

    linear_time = nugs.utils.benchmark_torch_function_in_microseconds(
        F.linear, sample_input, input_weight
    )
    qlora_linear_time = nugs.utils.benchmark_torch_function_in_microseconds(
        qlora.linear_nf4, sample_input, qlora_weight
    )
    compiled_qlora_linear_time = nugs.utils.benchmark_torch_function_in_microseconds(
        compiled_qlora_linear, sample_input, qlora_weight
    )
    qlora_triton_linear_time = nugs.utils.benchmark_torch_function_in_microseconds(
        qlora.linear_nf4_trtion, sample_input, qlora_weight
    )

    if bnb_available:
        bnb_linear_time = nugs.utils.benchmark_torch_function_in_microseconds(
            bnb_linear, sample_input
        )
    else:
        bnb_linear_time = -1.0

    return ExperimentResult(
        linear_time,
        qlora_linear_time,
        compiled_qlora_linear_time,
        bnb_linear_time,
        qlora_triton_linear_time,
    )


def mlp_experiment(config: ExperimentConfig) -> ExperimentResult:
    weights = qlora.get_mlp_weights(config.embed_dim, config.device)
    sample_input = qlora.get_sample_inputs(
        config.bsz,
        config.seqlen,
        config.embed_dim,
        config.device,
    )
    mlp = qlora.MLP(*weights)
    nf4_mlp = qlora.NF4MLP(*weights)
    nf4_mlp_triton = qlora.NF4MLPTriton(*weights)
    compiled_qlora_mlp = torch.compile(nf4_mlp, fullgraph=True, dynamic=config.dynamic)
    if bnb_available:
        bnb_mlp = qlora.BnbQloraMLP(*weights, config.device)

    # warmup
    for _ in range(3):
        mlp(sample_input)
        nf4_mlp(sample_input)
        compiled_qlora_mlp(sample_input)
        nf4_mlp_triton(sample_input)
        if bnb_available:
            bnb_mlp(sample_input)

    mlp_time = nugs.utils.benchmark_torch_function_in_microseconds(mlp, sample_input)
    qlora_mlp_time = nugs.utils.benchmark_torch_function_in_microseconds(nf4_mlp, sample_input)
    compiled_qlora_mlp_time = nugs.utils.benchmark_torch_function_in_microseconds(
        compiled_qlora_mlp, sample_input
    )
    qlora_mlp_triton_time = nugs.utils.benchmark_torch_function_in_microseconds(
        nf4_mlp_triton, sample_input
    )
    if bnb_available:
        bnb_mlp_time = nugs.utils.benchmark_torch_function_in_microseconds(bnb_mlp, sample_input)
    else:
        bnb_mlp_time = -1.0

    return ExperimentResult(
        mlp_time,
        qlora_mlp_time,
        compiled_qlora_mlp_time,
        bnb_mlp_time,
        qlora_mlp_triton_time,
    )


experiment_types = {
    "linear": linear_experiment,
    "mlp": mlp_experiment,
}


def gen_configs() -> list[ExperimentConfig]:
    # https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md
    # LLama 7b, 13b, 33b, 65b
    embed_dims = [4096, 5120, 6656, 8192]
    bszs = [8, 16, 32]
    seqlens = [256, 512]
    devices = [torch.device("cuda:0")]
    types = ["linear", "mlp"]
    # NotImplementedError: could not find kernel for aten.__rshift__.Scalar at dispatch key DispatchKey.Meta with dynamic shapes
    # dynamic = [False, True]
    dynamic = [False]
    configs = []
    for item in itertools.product(embed_dims, bszs, seqlens, devices, types, dynamic):
        configs.append(ExperimentConfig(*item))
    return configs


def main(output_path: Path | None, profile_path: Path | None, dynamic: bool):
    """Run experiments and output results to file

    Args:
        output_path (Optional[Path]): Path to write out CSV file for experiment results.
        profile_path (Optional[Path]): Path to write out json chrome trace file for an experiment.
        dynamic (bool): Compile with Dynamic shapes
    """

    results = []
    for experiment_config in tqdm(gen_configs()):
        # Since we are changing between dynamic and not
        import torch._dynamo  # noqa: F402

        torch._dynamo.reset()
        experiment = experiment_types[experiment_config.op]
        experiment_result = experiment(experiment_config)
        merged = asdict(experiment_config) | asdict(experiment_result)
        results.append(merged)

    if output_path is not None:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    else:
        headers = results[0].keys()
        rows = [list(r.values()) for r in results]
        print(tabulate(rows, headers=headers))

    if profile_path is not None:
        profile_experiment = ExperimentConfig(4096, 8, 128, torch.device("cuda:0"), "mlp", dynamic)
        with nugs.utils.print_cuda_memory_usage():
            weights = qlora.get_mlp_weights(
                profile_experiment.embed_dim, profile_experiment.device
            )
        sample_input = qlora.get_sample_inputs(
            profile_experiment.bsz,
            profile_experiment.seqlen,
            profile_experiment.embed_dim,
            profile_experiment.device,
        )

        qlora_mlp = qlora.NF4MLP(*weights)
        compiled_qlora_mlp = torch.compile(qlora_mlp, fullgraph=True, dynamic=dynamic)
        logging.info("Running torch.compile with dynamic = %s", dynamic)
        profile_config = nugs.utils.ProfileConfig(
            str(profile_path), "qlora_mlp", iters=5, warmup_iters=3, sync=True
        )
        nugs.utils.profile_function(
            profile_config,
            compiled_qlora_mlp,
            sample_input,
        )


if __name__ == "__main__":
    """Sample usage:
    # Running sweep
    python benchmarks/qlora.py false --output_path benchmarks/data/qlora_sweep.csv
    """
    CLI(main)
