"""SwiGLU: the GEMM computes [M, 2N] but the kernel stores only the [M, N] activation.

The epilogue splits the accumulator into gate and up lanes. Two weight layouts:

    interleaved  b is [K, 2N] with columns gate_0, up_0, gate_1, up_1, ...
                 epilogue: acc.view(M, N, 2) -> lanes [..., 0] and [..., 1]
    chunked      w is [2N, K] with rows [gate; up], passed as w.t()
                 epilogue: acc.chunk(2, dim=-1)

The same epilogue works on an MoE's ``F.grouped_mm``, with ``offs`` (int32 [E] cumulative
end rows of each expert) in ``gemm_kwargs``.

Run: python examples/flex_gemm/03_swiglu.py
"""

import torch
import torch.nn.functional as F
from torch._higher_order_ops import flex_gemm

from _common import QUACK, QUACK_AUTOTUNE, report, silu

M, N, K = 8192, 4096, 2048  # N is the activation width; the GEMM produces 2N columns
NUM_EXPERTS = 8


def swiglu_interleaved(acc):
    lanes = acc.float().view(acc.shape[0], -1, 2)
    return (silu(lanes[..., 0]) * lanes[..., 1]).to(acc.dtype)


def swiglu_chunked(acc):
    gate, up = acc.float().chunk(2, dim=-1)
    return (silu(gate) * up).to(acc.dtype)


def main() -> None:
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    x = torch.randn(M, K, **kw)
    w_chunked = torch.randn(2 * N, K, **kw) / K**0.5
    # The same weights with gate/up rows interleaved, transposed to [K, 2N].
    w_interleaved = w_chunked.view(2, N, K).transpose(0, 1).reshape(2 * N, K).t().contiguous()

    @torch.compile(fullgraph=True)
    def fused_interleaved(x, w):
        return flex_gemm(torch.mm, (x, w), swiglu_interleaved, kernel_options=QUACK)

    @torch.compile(fullgraph=True)
    def fused_chunked(x, w):
        return flex_gemm(torch.mm, (x, w.t()), swiglu_chunked, kernel_options=QUACK)

    def interleaved(x, w):
        return swiglu_interleaved(x @ w)

    def chunked(x, w):
        return swiglu_chunked(x @ w.t())

    report("dense interleaved", fused_interleaved, torch.compile(interleaved), x, w_interleaved)
    report("dense chunked", fused_chunked, torch.compile(chunked), x, w_chunked)

    # MoE: x rows are grouped by expert (equal blocks here); w13 is [E, K, 2I] interleaved.
    tokens, inter = 16384, 2048
    x = torch.randn(tokens, K, **kw)
    w13 = torch.randn(NUM_EXPERTS, K, 2 * inter, **kw) / K**0.5
    rows_per_expert = tokens // NUM_EXPERTS
    offs = torch.arange(1, NUM_EXPERTS + 1, device="cuda", dtype=torch.int32) * rows_per_expert

    @torch.compile(fullgraph=True)
    def fused_grouped(x, w13, offs):
        return flex_gemm(
            F.grouped_mm,
            (x, w13),
            swiglu_interleaved,
            gemm_kwargs={"offs": offs},
            kernel_options=QUACK_AUTOTUNE,
        )

    def grouped(x, w13, offs):
        return swiglu_interleaved(F.grouped_mm(x, w13, offs=offs))

    report("grouped_mm interleaved (E=8)", fused_grouped, torch.compile(grouped), x, w13, offs)


if __name__ == "__main__":
    main()
