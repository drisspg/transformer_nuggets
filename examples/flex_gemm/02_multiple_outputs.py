"""Multiple outputs: return a tuple and the same kernel writes every element.

Typical uses: keep a pre-activation for backward, or write a quantized copy next to the
result. Extra outputs must have the GEMM's [M, N] shape (shape-changing reductions are in
04_reductions.py) and may have their own dtypes. They are still stored; what fusion removes
is re-reading the GEMM result in a second kernel.

Run: python examples/flex_gemm/02_multiple_outputs.py
"""

import torch
import torch.nn.functional as F
from torch._higher_order_ops import flex_gemm

from _common import QUACK, report

M, N, K = 16384, 4096, 2048
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def main() -> None:
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    a, b = torch.randn(M, K, **kw), torch.randn(K, N, **kw) / K**0.5
    bias = torch.randn(N, **kw)

    def gelu_and_preactivation(acc):
        h = acc + bias
        return F.gelu(h, approximate="tanh"), h

    def bf16_and_fp8(acc):  # e.g. a per-tensor-scaled fp8 copy for a cache
        return acc, (acc.float() * 4.0).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

    for name, epilogue, tol in (
        ("gelu + keep pre-activation", gelu_and_preactivation, {}),
        # fp8 codes can differ by one step (12.5%) where the GEMM's bf16 rounding differs.
        ("bf16 + fp8 copy", bf16_and_fp8, {"atol": 2e-2, "rtol": 0.125}),
    ):

        @torch.compile(fullgraph=True)
        def fused(a, b):
            return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=QUACK)

        report(name, fused, torch.compile(lambda a, b: epilogue(a @ b)), a, b, **tol)


if __name__ == "__main__":
    main()
