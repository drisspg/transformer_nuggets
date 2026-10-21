"""Shared helpers for the FlexGEMM examples: kernel options, checks, and a fair timer."""

import statistics
from collections.abc import Callable

import torch
from torch.utils._pytree import tree_map

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

# QUACK is the CuTeDSL GEMM backend that fuses epilogues; it errors if it cannot.
# "autotune": True benchmarks a measured shortlist of tile configs (see README).
QUACK = {"backend": "QUACK", "autotune": False}
QUACK_AUTOTUNE = {"backend": "QUACK", "autotune": True}


def silu(x):
    """x * sigmoid(x), written as h + h * tanh(h) with h = x / 2: same math, cheaper in an epilogue.

    ``kernel_options={"fast_math": True}`` applies this rewrite to ``F.silu`` automatically (and
    also allows approximate hardware math). Spelling it out keeps the unfused baseline computing
    the same thing.
    """
    half = 0.5 * x
    return half + half * torch.tanh(half)


def assert_close(name: str, got, expected, atol: float = 2e-2, rtol: float = 2e-2) -> None:
    # assert_close compares 8-bit floats bitwise only; widen them to apply the tolerance.
    def widen(x):
        return x.float() if x.dtype.itemsize == 1 else x

    got, expected = tree_map(widen, got), tree_map(widen, expected)
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol, msg=lambda m: f"{name}: {m}")


def assert_rel_error(name: str, got, expected, tol: float = 1e-2) -> None:
    """Norm-wise check, for long reductions where elementwise bf16 noise is large."""
    got = got if isinstance(got, (tuple, list)) else [got]
    expected = expected if isinstance(expected, (tuple, list)) else [expected]
    for i, (g, e) in enumerate(zip(got, expected, strict=True)):
        err = ((g.float() - e.float()).norm() / e.float().norm()).item()
        assert err < tol, f"{name}[{i}]: relative error {err:.4f} >= {tol}"


def report(
    name: str, fused: Callable, baseline: Callable, *args, check: bool = True, **tol
) -> None:
    """Check ``fused(*args)`` against ``baseline(*args)``, then print both median GPU times.

    GPU clocks drop under sustained load, so both callables are timed alternately over several
    rounds after a warm-up round; timing one after the other would favor whichever runs first.
    """
    if check:
        assert_close(name, fused(*args), baseline(*args), **tol)
    times: dict[Callable, list[float]] = {fused: [], baseline: []}
    # Backward launches must stay on the thread that carries the profiler range.
    with torch.autograd.set_multithreading_enabled(False):
        for round_index in range(4):
            for fn in times:
                us = benchmark_cuda_function_in_microseconds(fn, *args)
                if round_index:
                    times[fn].append(us)
    fused_us, baseline_us = (statistics.median(t) for t in times.values())
    print(
        f"{name:<34} flex_gemm {fused_us:8.1f} us   torch.compile {baseline_us:8.1f} us"
        f"   {baseline_us / fused_us:5.2f}x"
    )
