"""Grouped reductions: reduce the accumulator over fixed-size column groups inside the kernel.

Spell the group with a ``view`` and reduce its last dimension:

    x = acc.float().view(M, N // G, G)
    x.amax(-1)                       # [M, N // G] extra output
    x / x.sum(-1, keepdim=True)      # or broadcast back into the main output

Supported: sum, mean, prod, amax, amin and logsumexp over one grouped dimension. G dividing
32 always works; a reduction stored as its own output also allows larger multiples of 32. A
whole-row reduction is a grouped one plus a small follow-up over the partials (the logsumexp
below, e.g. for an LM head's cross-entropy).

Run: python examples/flex_gemm/04_reductions.py
"""

import torch
from torch._higher_order_ops import flex_gemm

from _common import QUACK, report

M, N, K = 16384, 4096, 1024
G = 32


def group_rms_norm(acc):  # e.g. a per-head norm after a projection
    x = acc.float().view(acc.shape[0], -1, G)
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).view_as(acc).to(acc.dtype)


def out_and_group_amax(acc):  # e.g. per-block maxima to choose quantization scales
    return acc, acc.float().view(acc.shape[0], -1, G).abs().amax(-1)


def out_and_partial_logsumexp(acc):
    return acc, acc.float().view(acc.shape[0], -1, G).logsumexp(-1)


def main() -> None:
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    a, b = torch.randn(M, K, **kw), torch.randn(K, N, **kw) / K**0.5

    for name, epilogue in (
        ("group RMS norm", group_rms_norm),
        ("out + group amax", out_and_group_amax),
    ):

        @torch.compile(fullgraph=True)
        def fused(a, b):
            return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=QUACK)

        report(f"{name} (G={G})", fused, torch.compile(lambda a, b: epilogue(a @ b)), a, b)

    @torch.compile(fullgraph=True)
    def fused_logsumexp(a, b):
        out, partials = flex_gemm(
            torch.mm, (a, b), out_and_partial_logsumexp, kernel_options=QUACK
        )
        return out, partials.logsumexp(-1)

    def logsumexp(a, b):
        out = a @ b
        return out, out.float().logsumexp(-1)

    report("out + row logsumexp", fused_logsumexp, torch.compile(logsumexp), a, b)


if __name__ == "__main__":
    main()
