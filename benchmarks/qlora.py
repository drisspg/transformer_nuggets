import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

import transformer_nuggets as nugs
from transformer_nuggets.quant import QLoRAWeight

bnb_available = False
try:
    import bitsandbytes as bnb

    bnb_available = True
except ImportError:
    raise (
        "Could not import bitsandbytes, make sure you have installed it `pip install bitsandbytes` "
    )


def build_input_weight(input_shape: int, output_shape: int, device: torch.device):
    torch.manual_seed(0)
    input_weight = torch.empty(input_shape, output_shape, device=device, dtype=torch.bfloat16)
    input_weight.normal_(0, 1)
    return input_weight


def build_bitsandbytes_linear(input_weight: torch.Tensor, device: torch.device):
    param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4").cuda(device)
    bnb_linear = bnb.nn.LinearNF4(input_weight.size(0), input_weight.size(1), bias=False)
    bnb_linear.weight = param
    bnb_linear.to(device)
    return bnb_linear


def get_sample_inputs(bsz: int, seqlen: int, n_heads: int, head_dim: int, device: torch.device):
    sample_input = torch.empty(bsz, seqlen, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    sample_input = sample_input.view(bsz * seqlen, n_heads * head_dim)
    return sample_input


@dataclass
class ExperimentConfig:
    input_shape: int
    output_shape: int
    bsz: int
    seqlen: int
    n_heads: int
    head_dim: int
    device: torch.device


@dataclass
class ExperimentResult:
    matmul_time: float
    eager_dequant_time: float
    compiled_dequant_time: float
    bnb_time: float


def run_experiement(config: ExperimentConfig) -> ExperimentResult:
    input_weight = build_input_weight(config.input_shape, config.output_shape, config.device)
    sample_input = get_sample_inputs(
        config.bsz,
        config.seqlen,
        config.n_heads,
        config.head_dim,
        config.device,
    )
    qlora_weight = QLoRAWeight(input_weight)

    def dequant_matmul(lora_weight: QLoRAWeight, input_tensor: torch.Tensor):
        return F.linear(input_tensor, lora_weight.get_original_weight())

    compile_dequant_matmul = torch.compile(dequant_matmul, fullgraph=True)
    eager_dequant_time = nugs.utils.benchmark_torch_function_in_microseconds(
        dequant_matmul, qlora_weight, sample_input
    )
    matmul_time = nugs.utils.benchmark_torch_function_in_microseconds(
        F.linear, sample_input, input_weight
    )
    # warmup
    for _ in range(3):
        compile_dequant_matmul(qlora_weight, sample_input)

    compiled_dequant_time = nugs.utils.benchmark_torch_function_in_microseconds(
        compile_dequant_matmul, qlora_weight, sample_input
    )

    bnb_linear = build_bitsandbytes_linear(input_weight, config.device)
    # warmup
    for _ in range(3):
        bnb_linear(sample_input)
    bnb_time = nugs.utils.benchmark_torch_function_in_microseconds(bnb_linear, sample_input)

    # correctness
    eager_result = dequant_matmul(qlora_weight, sample_input)
    compiled_result = compile_dequant_matmul(qlora_weight, sample_input)
    bnb_result = bnb_linear(sample_input)
    torch.testing.assert_close(eager_result, compiled_result)
    torch.testing.assert_close(compiled_result, bnb_result)
    return ExperimentResult(matmul_time, eager_dequant_time, compiled_dequant_time, bnb_time)


def main(
    path: Optional[Path],
    print_times: int,
):
    # https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md
    configs = [
        # Llama 7b
        ExperimentConfig(4096, 4096, 8, 128, 32, 128, torch.device("cuda:0")),
        # Llama 13b
        ExperimentConfig(5120, 5120, 8, 128, 40, 128, torch.device("cuda:0")),
        # Llama 33b
        ExperimentConfig(6656, 6656, 8, 128, 52, 128, torch.device("cuda:0")),
        # LLama 65b
        ExperimentConfig(8192, 8192, 8, 128, 64, 128, torch.device("cuda:0")),
    ]
    results = []
    for experiment_config in tqdm(configs):
        experiment_result = run_experiement(experiment_config)
        if print_times:
            print(f"Experiment config: {experiment_config}")
            print(f"Time in eager for full matmul: {experiment_result.matmul_time} us")
            print(f"Time in eager for dequant_matmul: {experiment_result.eager_dequant_time} us")
            print(
                f"Time for compiled dequant_matmul : {experiment_result.compiled_dequant_time} us"
            )
            print(f"Time for bnb linear: {experiment_result.bnb_time} us")
        merged = asdict(experiment_config) | asdict(experiment_result)
        results.append(merged)

    if path is not None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments and output results to file")
    parser.add_argument("--print-times", action="store_true", help="print execution times")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Path to write out CSV file for experiment results.",
        default=None,
    )

    args = parser.parse_args()
    path = None
    if args.output_file is not None:
        path = Path(args.output_file)

    main(path, args.print_times)
