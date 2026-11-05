"""Interleaved CUDA-graph comparison of the warpdrive grouped GEMM with ``torch._grouped_mm``
and a loop of ``torch.mm`` per group, on MoE-like shapes with skewed routing.

Run under gpu-run. Compilation, allocation, reference checks and graph capture are untimed;
each graph holds ``CALLS_PER_GRAPH`` calls and the reported microseconds are per call. Shared
GPUs perturb timings, so every variant is replayed in alternating order over several rounds
and the minimum round median is reported. TFLOP/s count ``2 * sum(size_g) * N * K``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from transformer_nuggets.cute.warpdrive import AtomicPersistent, ClusterLaunchControl, CtaPair
from transformer_nuggets.cute.warpdrive.examples.grouped_gemm import (
    GroupedGemm,
    compile_grouped_gemm,
    grouped_gemm_reference,
)
from transformer_nuggets.utils.benchmark import benchmark_cuda_graph_stats

CALLS_PER_GRAPH = 20
# (experts, total tokens, N, K)
SHAPES = [
    (8, 4096, 2048, 7168),
    (8, 16384, 7168, 2048),
    (64, 8192, 4096, 4096),
    (64, 32768, 2048, 7168),
    (128, 16384, 7168, 2048),
    (128, 32768, 4096, 4096),
]


def skewed_sizes(groups: int, total: int, seed: int = 0) -> list[int]:
    """Token counts per expert from a heavy-tailed (Zipf-like) routing, summing to ``total``.
    A few experts get most tokens, some get none, and the rest are not multiples of 128."""
    gen = torch.Generator().manual_seed(seed)
    weights = torch.rand(groups, generator=gen) ** 3  # cube a uniform: a long tail
    weights[torch.randperm(groups, generator=gen)[: groups // 8]] = 0  # idle experts
    sizes = (weights / weights.sum() * total).floor().to(torch.int64)
    sizes[sizes.argmax()] += total - int(sizes.sum())
    return sizes.tolist()


def variants_for(a, b, offsets, sizes, labels: list[str]) -> dict:
    calls = {}
    for label in labels:
        if label == "grouped_mm":
            bt = b.transpose(1, 2)
            calls[label] = lambda bt=bt: torch._grouped_mm(a, bt, offs=offsets)
        elif label == "mm_loop":
            ends = offsets.tolist()
            bounds = [(g, s, e) for g, (s, e) in enumerate(zip([0, *ends[:-1]], ends)) if e > s]
            out = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

            def loop(bounds=bounds, out=out):
                for g, s, e in bounds:
                    torch.mm(a[s:e], b[g].T, out=out[s:e])
                return out

            calls[label] = loop
        else:
            cluster, *options = label.split("_")
            schedules = {"atomic": AtomicPersistent(), "clc": ClusterLaunchControl()}
            op = GroupedGemm(
                cluster=CtaPair() if cluster == "pair" else None,
                schedule=next((schedules[o] for o in options if o in schedules), None),
                tile_n=next((int(o[1:]) for o in options if o.startswith("n")), 256),
            )
            fn = compile_grouped_gemm(op)
            out = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
            workspace, sms = op.work.launch_args(a.device)

            def call(fn=fn, out=out, workspace=workspace, sms=sms):
                fn(a, b, out, offsets, workspace, sms)
                return out

            calls[label] = call
    return calls


def run_shape(groups, tokens, n, k, labels, samples, rounds) -> dict:
    torch.manual_seed(0)
    sizes = skewed_sizes(groups, tokens)
    offsets = torch.tensor(sizes, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(groups, n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    ref = grouped_gemm_reference(a, b, offsets)
    calls = variants_for(a, b, offsets, sizes, labels)
    for label, call in calls.items():
        out = call()
        torch.cuda.synchronize()
        err = (out.float() - ref).abs().max().item()
        if not torch.isfinite(out).all() or err > 2**-6 * ref.abs().max().item() + 1e-2:
            raise AssertionError((label, err))
        print(f"{label}: max abs err {err:.3g}", flush=True)
    graphs = {}
    for label, call in calls.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(CALLS_PER_GRAPH):
                call()
        graphs[label] = graph
        graph.replay()
    torch.cuda.synchronize()
    settle_end = time.monotonic() + 2.0
    while time.monotonic() < settle_end:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    medians = {label: [] for label in calls}
    for round_idx in range(rounds):
        order = list(calls) if round_idx % 2 == 0 else list(reversed(calls))
        for label in order:
            stats = benchmark_cuda_graph_stats(
                graphs[label], num_iters=samples, warmup_iters=2, n_resamples=50
            )
            medians[label].append(stats.median_us / CALLS_PER_GRAPH)
    flops = 2 * sum(sizes) * n * k
    best = {label: min(v) for label, v in medians.items()}
    tflops = {label: flops / us / 1e6 for label, us in best.items()}
    shape = f"G={groups} M={tokens} N={n} K={k}"
    print(
        shape,
        "  ".join(
            f"{label} {us:.1f} us ({tflops[label]:.0f} TFLOP/s)" for label, us in best.items()
        ),
        flush=True,
    )
    return {
        "shape": [groups, tokens, n, k],
        "sizes": sizes,
        "medians_us": medians,
        "min_median_us": best,
        "tflops": tflops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs="*", choices=range(len(SHAPES)))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["grouped_mm", "mm_loop", "single", "pair"],
        help="grouped_mm, mm_loop, or single/pair with optional _atomic/_clc and _n<tile_n>",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # CuTeDSL's argparse must not see the application's arguments
    shapes = SHAPES if not args.shape else [SHAPES[i] for i in args.shape]
    results = [run_shape(*shape, args.variants, args.samples, args.rounds) for shape in shapes]
    if args.output is not None:
        args.output.write_text(
            json.dumps({"gpu": torch.cuda.get_device_name(), "results": results}, indent=2)
        )


if __name__ == "__main__":
    main()
