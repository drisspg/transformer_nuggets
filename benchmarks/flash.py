import argparse
import csv
import enum
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import triton
from torch.nn.functional import scaled_dot_product_attention
from tqdm import tqdm

from transformer_nuggets.flash import BiasMode, attention, build_alibi_mask
from transformer_nuggets.utils import benchmark_torch_function_in_microseconds
import transformer_nuggets.utils as utils

device = torch.device("cuda")

BATCH, N_HEADS, D_HEAD = 4, 16, 64
# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 14)],
        line_arg="provider",
        line_vals=["triton"] + (["pytorch"]),
        line_names=["Triton"] + (["Pytorch"]),
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal-{causal}-bias-{bias_choice}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "D_HEAD": D_HEAD,
            "dtype": torch.float16,
            "mode": mode,
            "causal": causal,
            "bias_choice": bias_choice,
        },
    )
    for mode in ["fwd", "bwd"]
    for causal in [False, True]
    for bias_choice in [BiasMode.none, BiasMode.rel_pos, BiasMode.alibi]
]


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


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH,
    H,
    N_CTX,
    D_HEAD,
    mode,
    causal,
    bias_choice,
    provider,
    dtype=torch.float16,
    device=device,
):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    sm_scale = 1
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: attention(q, k, v, causal, sm_scale, bias_choice)
        if mode == "bwd":
            o, mask = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "pytorch":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        # reference implementation
        attn_bias = build_mask(bias_choice, BATCH, H, N_CTX, causal, dtype)
        is_causal = causal if (bias_choice == BiasMode.none) else False
        fn = lambda: scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, attn_mask=attn_bias, scale=sm_scale
        )
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


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
    triton_torchtimer: float
    triton_do_bench: float
    pytorch_torchtimer: float
    pytorch_do_bench: float


def gen_configs() -> List[ExperimentConfig]:
    bszs = [4]
    num_heads = [
        48,
    ]
    seqlens = [
        4096,
    ]
    head_dim = [64]
    bias_choices = [BiasMode.none, BiasMode.rel_pos, BiasMode.alibi]
    causal = [
        True,
    ]
    dtypes = [torch.float16]
    directions = ["fwd", "bwd"]
    configs = []
    for item in itertools.product(
        bszs, num_heads, seqlens, head_dim, bias_choices, causal, dtypes, directions
    ):
        configs.append(ExperimentConfig(*item))
    return configs


def get_input(
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
    if config.bias_choice != BiasMode.none:
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
    warmup = 5
    if config.direction == "fwd":
        for _ in range(warmup):
            attention(q, k, v, causal, sm_scale, bias_choice)
            scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, attn_mask=mask, scale=sm_scale
            )
        triton_torchtimer = benchmark_torch_function_in_microseconds(
            attention, q, k, v, causal, sm_scale, bias_choice
        )
        pytorch_torchtimer = benchmark_torch_function_in_microseconds(
            scaled_dot_product_attention,
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=mask,
            scale=sm_scale,
        )
        fn_triton = lambda: attention(q, k, v, causal, sm_scale, bias_choice)
        # do_bench is in ms
        triton_do_bench = triton.testing.do_bench(fn_triton, warmup=warmup, rep=100) * 1000
        fn_torch = lambda: scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, attn_mask=mask, scale=sm_scale
        )
        pytorch_do_bench = triton.testing.do_bench(fn_torch, warmup=warmup, rep=100) * 1000

    elif config.direction == "bwd":
        out_triton, _ = attention(q, k, v, causal, sm_scale, bias_choice)
        out_torch = scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, attn_mask=mask, scale=sm_scale
        )
        dOut = torch.randn_like(out_triton)
        for _ in range(warmup):
            out_triton.backward(dOut, retain_graph=True)
            out_torch.backward(dOut, retain_graph=True)

        triton_torchtimer = benchmark_torch_function_in_microseconds(
            out_triton.backward, dOut, retain_graph=True
        )
        pytorch_torchtimer = benchmark_torch_function_in_microseconds(
            out_torch.backward, dOut, retain_graph=True
        )
        fn_triton = lambda: out_triton.backward(dOut, retain_graph=True)
        triton_do_bench = triton.testing.do_bench(fn_triton, warmup=warmup, rep=100) * 1000
        fn_torch = lambda: out_torch.backward(dOut, retain_graph=True)
        pytorch_do_bench = triton.testing.do_bench(fn_torch, warmup=warmup, rep=100) * 1000
    else:
        raise ValueError("Invalid direction")

    return ExperimentResult(
        triton_torchtimer, triton_do_bench, pytorch_torchtimer, pytorch_do_bench
    )


class KernelChoice(enum.Enum):
    triton = "triton"
    torch = "torch"


def profile_experiment(
    kernel, config: ExperimentConfig, profile_config: utils.ProfileConfig
) -> ExperimentResult:
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


def main(output_file: Optional[Path], profile_path: Optional[Path]):
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
