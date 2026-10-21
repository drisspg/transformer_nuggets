"""Autotuning: from the default tile, to a shortlist, to every legal config, to a pinned config.

The epilogue changes what the best tile is, so FlexGEMM tunes the fused kernel itself:

    {"autotune": False}       one config from a measured shape table (no tuning)
    {"autotune": True}        benchmark a measured shortlist (about a dozen)
    + EXHAUSTIVE              benchmark every config QuACK accepts for this epilogue
    {"config": {...}}         pin a config; no tuning at compile time

Widen the search with ``TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE`` or
``torch._inductor.config.patch(max_autotune_gemm_search_space="EXHAUSTIVE")``. It compiles and
times every legal config (158 here vs 12), so compile time grows with it; it is a global
Inductor setting, so scope it to the call you are tuning. Read the winner from the generated code
and pin it so later compiles skip the search. Pick-the-fastest under GPU clock noise can favor a
lucky config, so re-time the result, as below.

The workload is the heavy backward epilogue from 06 (reads h, writes dh and act), where the
default tile is a poor fit: on GB200 the shortlist is ~1.9x faster than the default tile, and
EXHAUSTIVE finds the same winner. Often the shortlist is enough; EXHAUSTIVE tells you for sure.

Run: python examples/flex_gemm/09_autotune.py
"""

import ast
import re
import statistics
import time

import torch
from torch._higher_order_ops import flex_gemm
from torch._inductor.utils import run_and_get_code

from _common import silu
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

T, D, INTER = 8192, 1024, 4096


def swiglu(gate, up):
    return silu(gate) * up


def compile_with(options, h, inductor_config=None):
    def swiglu_backward(g):
        gate, up = h.view(g.shape[0], g.shape[1], 2).unbind(-1)
        act, vjp = torch.func.vjp(swiglu, gate.float(), up.float())
        d_gate, d_up = vjp(g.float())
        dh = torch.stack([d_gate, d_up], dim=-1).view(g.shape[0], 2 * g.shape[1])
        return dh.to(g.dtype), act.to(g.dtype)

    return torch.compile(
        lambda dy, w2: flex_gemm(torch.mm, (dy, w2.t()), swiglu_backward, kernel_options=options),
        fullgraph=True,
        options=inductor_config,
    )


def selected_config(code: str) -> dict:
    """The pinned QuACK config Inductor wrote into the generated wrapper."""
    match = re.search(r"config=(\(\(.*?\)\)),", code)
    return dict(ast.literal_eval(match.group(1)))


def main() -> None:
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    dy = torch.randn(T, D, **kw)
    w2 = torch.randn(INTER, D, **kw) / D**0.5
    h = torch.randn(T, 2 * INTER, **kw)
    quack = {"backend": "QUACK"}

    exhaustive = {"max_autotune_gemm_search_space": "EXHAUSTIVE"}
    variants = {
        "default tile (autotune=False)": compile_with({**quack, "autotune": False}, h),
        "shortlist (autotune=True)": compile_with({**quack, "autotune": True}, h),
        "EXHAUSTIVE": compile_with({**quack, "autotune": True}, h, exhaustive),
    }

    # Compile (and tune) each variant once, recording compile time and the chosen config.
    configs, compile_s = {}, {}
    for name, fn in variants.items():
        start = time.perf_counter()
        _, (code,) = run_and_get_code(fn, dy, w2)
        compile_s[name] = time.perf_counter() - start
        configs[name] = selected_config(code)

    name, pinned = "pinned EXHAUSTIVE winner", configs["EXHAUSTIVE"]
    variants[name] = compile_with({**quack, "config": pinned}, h)
    start = time.perf_counter()
    reference = variants[name](dy, w2)
    compile_s[name], configs[name] = time.perf_counter() - start, pinned
    for other, fn in variants.items():
        torch.testing.assert_close(fn(dy, w2), reference, atol=2e-2, rtol=2e-2, msg=other)

    # Alternate variants over rounds so GPU clock throttling does not favor one of them.
    times = {name: [] for name in variants}
    for _ in range(3):
        for name, fn in variants.items():
            times[name].append(benchmark_cuda_function_in_microseconds(fn, dy, w2))

    for name in variants:
        c = configs[name]
        tile = f"{c['tile_m']}x{c['tile_n']} cluster {c['cluster_m']}x{c['cluster_n']}"
        tile += " swap_ab" if c["swap_ab"] else ""
        print(
            f"{name:<30} {statistics.median(times[name]):8.1f} us   "
            f"compile {compile_s[name]:6.1f} s   {tile}"
        )
    print(f"pin it with kernel_options={{'backend': 'QUACK', 'config': {pinned}}}")


if __name__ == "__main__":
    main()
