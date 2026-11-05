"""Interleaved fixed-buffer CUDA-graph comparison of the warpdrive backward against FA4's.

Variants, each one graph of a whole backward call (preprocess, main kernel, dQ postprocess):

* ``warpdrive``: ``fa4_bwd`` (FA4's pre/postprocess around the warpdrive main kernel);
* ``fa4_1cta``: FA4's ``_flash_attn_bwd`` with the 1-CTA configuration the port matches
  (``disable_2cta``: 128 x 128 tiles, Q_stage 2, FA4's register split);
* ``fa4``: FA4's default configuration (2-CTA for head_dim 128 on SM100).

Before timing, dK/dV of ``warpdrive`` must equal ``fa4_1cta`` bitwise and dQ must agree to
the reduce-order rounding. Per round, each graph replays ``--reps`` times under the profiler:
CUPTI gives the main kernel's duration, CUDA events the whole graph. Seven rounds alternate
the variant order; the table reports the minimum round median, in microseconds.

Run under ``gpu-run``:
``python -m transformer_nuggets.cute.warpdrive.examples.bench_fa4_bwd --output bwd.json``.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from flash_attn.cute import flash_bwd_sm100
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

from transformer_nuggets.cute.warpdrive.examples import fa4_bwd as port
from transformer_nuggets.cute.warpdrive.examples.bench_support import (
    provenance,
    replay_timed,
    settle,
)

# (batch, heads, seqlen_q, seqlen_k)
SHAPES = [(2, 16, 1024, 1024), (4, 16, 4096, 4096), (1, 16, 8192, 8192), (4, 16, 3000, 2900)]
MAIN_KERNELS = {"warpdrive": "warpdrive_fa4_bwd", "fa4_1cta": "Backward", "fa4": "Backward"}


def main_kernel(names, variant):
    needle = MAIN_KERNELS[variant]
    found = [n for n in names if needle in n and "rocess" not in n]
    if len(found) != 1:
        raise RuntimeError(f"{variant}: expected one main kernel matching {needle!r}, got {names}")
    return found[0]


def run_shape(b, h, sq, sk, causal, variants, reps, rounds):
    torch.manual_seed(0)
    q = torch.randn(b, sq, h, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(b, sk, h, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    dout = torch.randn_like(q)
    scale = 128**-0.5
    out, lse = _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal, return_lse=True)[:2]
    one_cta, default = (port.fa4_config(q, k, causal, flag) for flag in (True, False))

    def fa4(config):
        return _flash_attn_bwd(q, k, v, out, dout, lse, scale, causal, config=config)[:3]

    calls = {
        "warpdrive": lambda: port.fa4_bwd(q, k, v, out, dout, lse, scale, causal=causal),
        "fa4_1cta": lambda: fa4(one_cta),
        "fa4": lambda: fa4(default),
    }
    calls = {name: calls[name] for name in variants}
    if "warpdrive" in calls:
        ours, ref = calls["warpdrive"](), calls["fa4_1cta"]() if "fa4_1cta" in calls else None
        if ref is not None:
            dq_err = (ours[0].float() - ref[0].float()).abs().max().item()
            if not (torch.equal(ours[1], ref[1]) and torch.equal(ours[2], ref[2])):
                raise AssertionError("dK/dV differ from FA4's 1-CTA backward")
            if dq_err > 2e-3 * ref[0].float().abs().max().item() + 1e-3:
                raise AssertionError(f"dQ differs from FA4's 1-CTA backward by {dq_err}")
            print((b, h, sq, sk, causal), "dK/dV bitwise equal to fa4_1cta; dQ max abs", dq_err)
    graphs = {}
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[name] = graph
    settle(graphs.values(), seconds=3.0)
    medians = {name: {"graph": [], "main": []} for name in calls}
    samples = {name: [] for name in calls}
    for r in range(rounds):
        for name in variants if r % 2 == 0 else list(reversed(variants)):
            graph_us, kernels = replay_timed(graphs[name], reps, warmup=5)
            main = kernels[main_kernel(list(kernels), name)]
            medians[name]["graph"].append(statistics.median(graph_us))
            medians[name]["main"].append(statistics.median(main))
            samples[name].append({"graph_us": graph_us, "kernels_us": kernels})
            print(
                (b, h, sq, sk, causal), r, name, medians[name]["main"][-1], "main us", flush=True
            )
    best = {name: {part: min(values) for part, values in m.items()} for name, m in medians.items()}
    print("minimum round medians (us)", (b, h, sq, sk, causal), best, flush=True)
    return {
        "shape": [b, h, sq, sk],
        "causal": causal,
        "min_median_us": best,
        "medians_us": medians,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=int, nargs="*", help="indices into SHAPES")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(MAIN_KERNELS),
        default=list(MAIN_KERNELS),
    )
    parser.add_argument("--reps", type=int, default=30, help="graph replays per round")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reps < 1 or args.rounds < 1 or len(set(args.variants)) != len(args.variants):
        parser.error("reps and rounds must be positive and variants unique")
    sys.argv = sys.argv[:1]  # CuTeDSL's diagnostic argparse must not see our arguments
    metadata = provenance(
        {
            "warpdrive": Path(port.__file__),
            "benchmark": Path(__file__),
            "flash_attn": Path(flash_bwd_sm100.__file__),
        },
        vars(args) | {"output": str(args.output)},
    )
    print("metadata", metadata, flush=True)
    shapes = SHAPES if not args.shapes else [SHAPES[i] for i in args.shapes]
    results = [
        run_shape(*shape, args.causal, args.variants, args.reps, args.rounds) for shape in shapes
    ]
    if args.output is not None:
        args.output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2))


if __name__ == "__main__":
    main()
