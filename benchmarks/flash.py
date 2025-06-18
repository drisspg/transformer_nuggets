import argparse
import csv
import enum
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformer_nuggets.utils as utils
from torch.nn.functional import scaled_dot_product_attention
from tqdm import tqdm

from transformer_nuggets.flash import attention, BiasMode, build_alibi_mask
from transformer_nuggets.utils import benchmark_torch_function_in_microseconds

device = torch.device("cuda")


def build_mask(bias_choice, batch, num_heads, seq_len, causal, dtype):
    if bias_choice == BiasMode.rel_pos:
        attn_bias = build_alibi_mask(seq_len, seq_len, num_heads, scale=1, causal=causal)
        attn_bias = attn_bias.expand(batch, num_heads, seq_len, seq_len).to(device).to(dtype)
    elif bias_choice == BiasMode.alibi:
        attn_bias = build_alibi_mask(seq_len, seq_len, num_heads, scale=None, causal=causal)
        attn_bias = attn_bias.expand(batch, num_heads, seq_len, seq_len).to(device).to(dtype)
    elif bias_choice == BiasMode.none:
        attn_bias = None
    return attn_bias


@dataclass
class ExperimentConfig:
    bsz: int
    num_heads: int
    seqlen: int
    head_dim: int
    bias_choice: BiasMode
    causal: bool
    dtype: torch.dtype
    direction: str


@dataclass
class ExperimentResult:
    triton_time: float
    pytorch_time: float


def get_input(
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    q = torch.randn(
        (config.bsz, config.num_heads, config.seqlen, config.head_dim),
        dtype=config.dtype,
        device=device,
        requires_grad=True,
    )
    k = torch.randn(
        (config.bsz, config.num_heads, config.seqlen, config.head_dim),
        dtype=config.dtype,
        device=device,
        requires_grad=True,
    )
    v = torch.randn(
        (config.bsz, config.num_heads, config.seqlen, config.head_dim),
        dtype=config.dtype,
        device=device,
        requires_grad=True,
    )
    if config.bias_choice != BiasMode.none and config.seqlen < 8192:
        mask = build_mask(
            config.bias_choice,
            config.bsz,
            config.num_heads,
            config.seqlen,
            config.causal,
            config.dtype,
        )
        return q, k, v, mask
    else:
        return q, k, v, None


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    q, k, v, mask = get_input(config)
    causal = config.causal
    sm_scale = 1
    bias_choice = config.bias_choice
    is_causal = causal if (bias_choice == BiasMode.none) else False
    if config.direction == "fwd":
        if config.seqlen >= 8192 and config.bias_choice != BiasMode.none:
            # Skip PyTorch for large seq_len because of OOM
            pytorch_time = float("nan")
        else:
            pytorch_time = benchmark_torch_function_in_microseconds(
                scaled_dot_product_attention,
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=mask,
                scale=sm_scale,
            )
        triton_time = benchmark_torch_function_in_microseconds(
            attention, q, k, v, causal, sm_scale, bias_choice
        )

    elif config.direction == "bwd":
        out_triton, _ = attention(q, k, v, causal, sm_scale, bias_choice)
        dOut = torch.randn_like(out_triton)
        triton_time = benchmark_torch_function_in_microseconds(
            out_triton.backward, dOut, retain_graph=True
        )
        if config.seqlen >= 8192 and config.bias_choice != BiasMode.none:
            # Skip PyTorch for large seq_len because of OOM
            pytorch_time = float("nan")
        else:
            out_torch = scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, attn_mask=mask, scale=sm_scale
            )
            pytorch_time = benchmark_torch_function_in_microseconds(
                out_torch.backward, dOut, retain_graph=True
            )
    else:
        raise ValueError("Invalid direction")

    return ExperimentResult(triton_time, pytorch_time)


class KernelChoice(enum.Enum):
    triton = "triton"
    torch = "torch"


def profile_experiment(
    kernel, config: ExperimentConfig, profile_config: utils.ProfileConfig
) -> None:
    q, k, v, mask = get_input(config)
    sm_scale = 1
    causal = config.causal
    bias_choice = config.bias_choice
    is_causal = causal if (bias_choice == BiasMode.none) else False
    dOut = torch.randn_like(q)
    fn = (
        lambda: scaled_dot_product_attention(
            q, k, v, mask, is_causal=is_causal, scale=sm_scale
        ).backward(dOut, retain_graph=True)
        if kernel == KernelChoice.torch
        else lambda: attention(q, k, v, causal, sm_scale, bias_choice).backward(
            dOut, retain_graph=True
        )
    )
    utils.profile_function(profile_config, fn)


def gen_configs() -> list[ExperimentConfig]:
    seqlens = [512, 1024, 2048, 4096, 8192, 16384]
    head_dim = [64, 128]
    bias_choices = [BiasMode.none, BiasMode.rel_pos, BiasMode.alibi]
    causal = [True, False]
    dtypes = [torch.float16]
    directions = ["fwd", "bwd"]
    configs = []

    def get_bsz_num_heads(hidden_dim, seq_len, head_dim, max_tokens=2**14):
        # Default max_tokens = 2**14 = 16384
        assert hidden_dim % head_dim == 0, "hidden_dim must be divisible by head_dim"
        assert max_tokens % seq_len == 0, "max_tokens must be divisible by seq_len"
        num_heads = hidden_dim / head_dim
        batch_size = max_tokens / seq_len
        return int(batch_size), int(num_heads)

    for item in itertools.product(seqlens, head_dim, bias_choices, causal, dtypes, directions):
        # 2048, chosen from FlashV2 Paper
        bsz, num_heads = get_bsz_num_heads(2048, *item[:2])
        configs.append(ExperimentConfig(bsz, num_heads, *item))
    return configs


def main(output_file: Path | None, profile_path: Path | None):
    if output_file is not None:
        configs = gen_configs()
        results = []
        for experiment_config in tqdm(configs, unit="Experiment"):
            experiment_result = run_experiment(experiment_config)
            merged = asdict(experiment_config) | asdict(experiment_result)
            results.append(merged)

        print(f"Writing results to {output_path}")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    else:
        print("No output file specified, skipping experiment!")

    if profile_path is not None:
        if not profile_path.suffix:
            profile_path = profile_path.with_suffix(".json")
        print(f"Writing profile to {profile_path}")
        # Kernel Choice and Experiment Config
        kernel_choice = KernelChoice.triton
        experiment_config = ExperimentConfig(
            4, 32, 4096, 64, BiasMode.none, False, torch.float16, "fwd"
        )

        profile_config = utils.ProfileConfig(
            str(profile_path),
            name=f"sdpa-{kernel_choice.value}",
            iters=5,
            warmup_iters=3,
            sync=True,
        )
        profile_experiment(kernel_choice, experiment_config, profile_config)


if __name__ == "__main__":
    """Sample usage:
    # Running sweep
    python benchmarks/flash.py -o benchmarks/data/flash_attention_sweep.csv

    # only works on post-Ampere GPUs right now
    # bench_flash_attention.run(save_path=None, print_data=True)
    """
    parser = argparse.ArgumentParser(description="Run experiments and output results to file")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Path to write out CSV file for experiment results.",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--profile_path",
        type=str,
        help="Path to write out json chrome trace file for an experiment.",
        default=None,
    )
    args = parser.parse_args()
    output_path = None
    profile_path = None
    if args.output_file is not None:
        output_path = Path(args.output_file)
    if args.profile_path is not None:
        profile_path = Path(args.profile_path)
    if output_path is None and profile_path is None:
        raise ValueError("Must specify at least one of output_file or profile_path")
    main(output_path, profile_path)
