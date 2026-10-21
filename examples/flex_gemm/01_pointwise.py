"""Pointwise epilogues: ``flex_gemm(torch.mm, (a, b), epilogue)`` computes ``epilogue(a @ b)``.

``epilogue`` receives the accumulator ``acc`` [M, N] and runs before it is stored. It can
capture tensors from its enclosing scope; their broadcast shape decides how they are loaded:

    scalar  []/[1, 1]   per-row  [M, 1]   per-column  [1, N] or [N]   full tile  [M, N]

Each epilogue below is written once and used twice: inside ``flex_gemm`` and, as the
reference, on ``a @ b`` under plain ``torch.compile`` (cuBLAS plus a fused Triton kernel).

Run: python examples/flex_gemm/01_pointwise.py
"""

import torch
import torch.nn.functional as F
from torch._higher_order_ops import flex_gemm

from _common import QUACK_AUTOTUNE, report, silu

M, N, K = 8192, 4096, 2048


def main() -> None:
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    a, b = torch.randn(M, K, **kw), torch.randn(K, N, **kw) / K**0.5
    bias = torch.randn(N, **kw)
    row_scale, col_scale = torch.rand(M, 1, device="cuda"), torch.rand(1, N, device="cuda")

    def bias_gelu(acc):
        return F.gelu(acc + bias, approximate="tanh")

    def scaled_bias_silu(acc):  # per-row and per-column scales, like an int8/fp8 dequant
        return silu(acc.float() * row_scale * col_scale + bias).to(acc.dtype)

    for name, epilogue in (
        ("bias + gelu", bias_gelu),
        ("row/col scale + bias + silu", scaled_bias_silu),
    ):

        @torch.compile(fullgraph=True)
        def fused(a, b):
            return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=QUACK_AUTOTUNE)

        report(name, fused, torch.compile(lambda a, b: epilogue(a @ b)), a, b)

    # addmm/bmm/baddbmm take their usual arguments; static alpha/beta go in gemm_kwargs.
    @torch.compile(fullgraph=True)
    def fused_addmm_relu(bias, a, b):
        return flex_gemm(
            torch.addmm,
            (bias, a, b),
            F.relu,
            gemm_kwargs={"alpha": 0.5},
            kernel_options=QUACK_AUTOTUNE,
        )

    def addmm_relu(bias, a, b):
        return F.relu(torch.addmm(bias, a, b, alpha=0.5))

    report("addmm(alpha=0.5) + relu", fused_addmm_relu, torch.compile(addmm_relu), bias, a, b)


if __name__ == "__main__":
    main()
