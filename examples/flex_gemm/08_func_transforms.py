"""torch.func inside an epilogue: derivatives of an activation, computed in the GEMM.

The epilogue is traced like any PyTorch function, so function transforms compose inside it.
Write only the activation; ``torch.func`` derives the rest, and the derivative math runs in
the GEMM kernel:

    vjp(f, h)(v)           reverse mode: v * f'(h)      (06 uses this for the backward)
    vjp of vjp             second derivative f''(h), e.g. for curvature / exact-Hessian terms
    jvp(f, (h,), (t,))     forward mode: f(h) and the tangent f'(h) * t in one pass

Pass explicit cotangents such as ``torch.ones_like(h)``. FlexGEMM does not lower
``aten.expand`` yet, which ``torch.func.grad`` of a ``.sum()`` emits, and so does ``jvp``
through a Python-scalar multiply (hence ``F.silu`` rather than the tanh form in the jvp case).

Derivatives multiply the epilogue's math, so keep the activation cheap: these SiLU outputs beat
torch.compile, while the same three outputs for Mish (tanh of softplus) are several times slower.

Run: python examples/flex_gemm/08_func_transforms.py
"""

import torch
import torch.nn.functional as F
from torch._higher_order_ops import flex_gemm

from _common import report, silu

M, N, K = 8192, 4096, 2048
# fast_math allows approximate hardware math (vjp case ~1.09x -> ~1.25x on GB200) and rewrites
# F.silu to its tanh form, so in the jvp case the unfused baseline still computes exact sigmoid.
OPTIONS = {"backend": "QUACK", "autotune": True, "fast_math": True}


def derivative(f):
    """x -> f'(x) for an elementwise f, as a vjp with a ones cotangent."""
    return lambda x: torch.func.vjp(f, x)[1](torch.ones_like(x))[0]


def silu_and_derivatives(acc):
    """silu(h), silu'(h), silu''(h) from one accumulator; no derivative written by hand."""
    h = acc.float()
    fns = (silu, derivative(silu), derivative(derivative(silu)))
    return tuple(f(h).to(acc.dtype) for f in fns)


def main() -> None:
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    a, b = torch.randn(M, K, **kw), torch.randn(K, N, **kw) / K**0.5
    tangent = torch.randn(M, N, **kw)  # e.g. da @ b from a forward-mode pass

    def silu_and_tangent(acc):
        out, out_tangent = torch.func.jvp(F.silu, (acc.float(),), (tangent.float(),))
        return out.to(acc.dtype), out_tangent.to(acc.dtype)

    # Independent check of the transforms: autograd on F.silu of the same bf16 GEMM output.
    h = (a @ b).float().requires_grad_()
    (d1,) = torch.autograd.grad(F.silu(h).sum(), h, create_graph=True)
    (d2,) = torch.autograd.grad(d1.sum(), h)
    expected = tuple(x.to(a.dtype) for x in (F.silu(h), d1, d2))
    torch.testing.assert_close(silu_and_derivatives(a @ b), expected, atol=2e-2, rtol=2e-2)

    for name, epilogue in (
        ("silu, silu', silu'' (vjp of vjp)", silu_and_derivatives),
        ("silu + tangent (jvp)", silu_and_tangent),
    ):

        @torch.compile(fullgraph=True)
        def fused(a, b):
            return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=OPTIONS)

        report(name, fused, torch.compile(lambda a, b: epilogue(a @ b)), a, b)


if __name__ == "__main__":
    main()
