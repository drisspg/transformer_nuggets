import argparse
import csv
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
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


def build_input_weight(embed_dim: int, device: torch.device):
    torch.manual_seed(0)
    input_weight = torch.empty(embed_dim, embed_dim, device=device, dtype=torch.bfloat16)
    input_weight.normal_(0, 1)
    return input_weight


def build_bitsandbytes_linear(input_weight: torch.Tensor, device: torch.device):
    param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4").cuda(device)
    bnb_linear = bnb.nn.LinearNF4(input_weight.size(0), input_weight.size(1), bias=False)
    bnb_linear.weight = param
    bnb_linear.to(device)
    return bnb_linear


def get_sample_inputs(bsz: int, seqlen: int, embed_dim: int, device: torch.device):
    sample_input = torch.rand(bsz, seqlen, embed_dim, device=device, dtype=torch.bfloat16)
    sample_input = sample_input.view(bsz * seqlen, embed_dim)
    return sample_input


def get_mlp_weights(
    embed_dim: int, device: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    def find_multiple(n: int, k: int) -> int:
        if n % k == 0:
            return n
        return n + k - (n % k)

    hidden_dim = 4 * embed_dim
    n_hidden = int(2 * hidden_dim / 3)
    n_hidden = find_multiple(n_hidden, 256)
    weight1 = torch.empty((n_hidden, embed_dim), dtype=torch.bfloat16, device=device).normal_(0, 1)
    weight2 = torch.empty((n_hidden, embed_dim), dtype=torch.bfloat16, device=device).normal_(0, 1)
    weight3 = torch.empty((embed_dim, n_hidden), dtype=torch.bfloat16, device=device).normal_(0, 1)

    return weight1, weight2, weight3


class MLP(nn.Module):
    def __init__(self, weight1, weight2, weight3) -> None:
        super().__init__()
        self.w1, self.w2, self.w3 = weight1, weight2, weight3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(F.linear(x, self.w1)) * F.linear(x, self.w2)
        x = F.linear(x, self.w3)
        return x


class QloraMLP(nn.Module):
    def __init__(self, weight1, weight2, weight3) -> None:
        super().__init__()
        self.w1 = QLoRAWeight(weight1)
        self.w2 = QLoRAWeight(weight2)
        self.w3 = QLoRAWeight(weight3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(F.linear(x, self.w1.get_original_weight())) * F.linear(
            x, self.w2.get_original_weight()
        )
        x = F.linear(x, self.w3.get_original_weight())
        return x


class BnbQloraMLP(nn.Module):
    def __init__(self, weight1, weight2, weight3, device) -> None:
        super().__init__()
        self.w1 = build_bitsandbytes_linear(weight1, device)
        self.w2 = build_bitsandbytes_linear(weight2, device)
        self.w3 = build_bitsandbytes_linear(weight3, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        return x


def qlora_linear(
    input_tensor: torch.Tensor,
    lora_weight: QLoRAWeight,
):
    return F.linear(input_tensor, lora_weight.get_original_weight())


@dataclass
class ExperimentConfig:
    embed_dim: int
    bsz: int
    seqlen: int
    device: torch.device
    op: str = "linear"


@dataclass
class ExperimentResult:
    unquantized: float
    eager_qlora: float
    compiled_qlora: float
    bnb_time: float


def linear_experiment(config: ExperimentConfig) -> ExperimentResult:
    input_weight = build_input_weight(config.embed_dim, config.device)
    sample_input = get_sample_inputs(
        config.bsz,
        config.seqlen,
        config.embed_dim,
        config.device,
    )
    qlora_weight = QLoRAWeight(input_weight)
    bnb_linear = build_bitsandbytes_linear(input_weight, config.device)
    compiled_qlora_linear = torch.compile(qlora_linear, fullgraph=True)

    # warmup
    for _ in range(3):
        F.linear(sample_input, input_weight)
        qlora_linear(sample_input, qlora_weight)
        compiled_qlora_linear(sample_input, qlora_weight)
        bnb_linear(sample_input)

    linear_time = nugs.utils.benchmark_torch_function_in_microseconds(
        F.linear, sample_input, input_weight
    )
    qlora_linear_time = nugs.utils.benchmark_torch_function_in_microseconds(
        qlora_linear, sample_input, qlora_weight
    )
    compiled_qlora_linear_time = nugs.utils.benchmark_torch_function_in_microseconds(
        compiled_qlora_linear, sample_input, qlora_weight
    )
    bnb_linear_time = nugs.utils.benchmark_torch_function_in_microseconds(bnb_linear, sample_input)

    # QLoRA and BnB parity
    # compiled_qlora_linear_result = compiled_qlora_linear(sample_input)
    # bnb_linear_result = bnb_linear(sample_input)
    # torch.testing.assert_close(compiled_qlora_linear_result, bnb_linear_result)

    return ExperimentResult(
        linear_time, qlora_linear_time, compiled_qlora_linear_time, bnb_linear_time
    )


def mlp_experiment(config: ExperimentConfig) -> ExperimentResult:
    weights = get_mlp_weights(config.embed_dim, config.device)
    sample_input = get_sample_inputs(
        config.bsz,
        config.seqlen,
        config.embed_dim,
        config.device,
    )
    mlp = MLP(*weights)
    qlora_mlp = QloraMLP(*weights)
    compiled_qlora_mlp = torch.compile(qlora_mlp, fullgraph=True)
    bnb_mlp = BnbQloraMLP(*weights, config.device)

    # warmup
    for _ in range(3):
        mlp(sample_input)
        qlora_mlp(sample_input)
        compiled_qlora_mlp(sample_input)
        bnb_mlp(sample_input)

    mlp_time = nugs.utils.benchmark_torch_function_in_microseconds(mlp, sample_input)
    qlora_mlp_time = nugs.utils.benchmark_torch_function_in_microseconds(qlora_mlp, sample_input)
    compiled_qlora_mlp_time = nugs.utils.benchmark_torch_function_in_microseconds(
        compiled_qlora_mlp, sample_input
    )
    bnb_mlp_time = nugs.utils.benchmark_torch_function_in_microseconds(bnb_mlp, sample_input)

    # QLoRA and BnB parity
    # compiled_qlora_mlp_result = compiled_qlora_mlp(sample_input)
    # bnb_mlp_result = bnb_mlp(sample_input)
    # torch.testing.assert_close(compiled_qlora_mlp_result, bnb_mlp_result)

    return ExperimentResult(mlp_time, qlora_mlp_time, compiled_qlora_mlp_time, bnb_mlp_time)


experiment_types = {
    "linear": linear_experiment,
    "mlp": mlp_experiment,
}


def main(
    path: Optional[Path],
    print_times: int,
):
    # https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md
    configs = [
        # Llama 7b
        ExperimentConfig(4096, 8, 128, torch.device("cuda:0")),
        # Llama 13b
        ExperimentConfig(5120, 8, 128, torch.device("cuda:0")),
        # Llama 33b
        ExperimentConfig(6656, 8, 128, torch.device("cuda:0")),
        # LLama 65b
        ExperimentConfig(8192, 8, 128, torch.device("cuda:0")),
    ]
    results = []
    for experiment_config in tqdm(configs):
        # experiment_result = run_experiment(experiment_config)
        experiment = experiment_types[experiment_config.op]
        experiment_result = experiment(experiment_config)
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
        # force a garbage collection to avoid OOM
        gc.collect()
        torch.cuda.empty_cache()

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
