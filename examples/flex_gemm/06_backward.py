"""Forward and backward: a SwiGLU MLP whose activation gradient runs inside a GEMM epilogue.

FlexGEMM has no autograd formula: wrap the block in ``torch.autograd.Function`` and write the
backward with more FlexGEMM calls. The stages reverse, so a forward epilogue's VJP can be the
epilogue of the backward GEMM that produces its incoming gradient:

    forward                             backward (dy given)
    h   = x @ w13        [T, 2I]        g   = dy @ w2.T         [T, I]
    act = swiglu(h)      [T, I]         dh  = swiglu_vjp(h, g)  [T, 2I]   <- epilogue
    y   = act @ w2       [T, D]         act = swiglu(h)                   <- same epilogue
                                        dw2 = act.T @ dy;  dx = dh @ w13.T;  dw13 = x.T @ dh

The forward writes act and h (the only saved activation) from one GEMM. The backward epilogue
captures h, calls ``torch.func.vjp`` on the same ``swiglu``, and returns dh plus a recomputed
act, so act is never saved. Change ``swiglu`` and the backward follows.

Only the activation work is fused, so the win shrinks as D grows and the six GEMMs dominate.
The grouped-GEMM MoE version is ``transformer_nuggets/moe/flex_gemm_experts.py``.

Run: python examples/flex_gemm/06_backward.py
"""

import torch
from torch._higher_order_ops import flex_gemm

from _common import assert_rel_error, report, silu

T, D, INTER = 8192, 1024, 4096  # tokens, hidden size, intermediate size I
# The backward epilogue is heavy (it reads h and writes dh and act), so options matter: untuned,
# that GEMM is several times slower than with autotune. fast_math lets CuTeDSL use approximate
# hardware math (~1.08x -> ~1.17x for the whole step on GB200); it slightly changes numerics.
OPTIONS = {"backend": "QUACK", "autotune": True, "fast_math": True}


def swiglu(gate, up):
    return silu(gate) * up


def gate_up(h):
    """Interleaved [T, 2I] (gate_0, up_0, gate_1, ...) -> fp32 gate and up, each [T, I]."""
    lanes = h.float().view(h.shape[0], -1, 2)
    return lanes[..., 0], lanes[..., 1]


@torch.compile(fullgraph=True)
def mlp_forward(x, w13, w2):
    def swiglu_and_keep_h(acc):
        return swiglu(*gate_up(acc)).to(acc.dtype), acc

    act, h = flex_gemm(torch.mm, (x, w13), swiglu_and_keep_h, kernel_options=OPTIONS)
    return act @ w2, h


@torch.compile(fullgraph=True)
def mlp_backward(x, w13, w2, h, dy):
    def swiglu_backward(g):
        # Split a captured tensor's lanes before converting it to fp32.
        gate, up = h.view(g.shape[0], g.shape[1], 2).unbind(-1)
        act, vjp = torch.func.vjp(swiglu, gate.float(), up.float())
        d_gate, d_up = vjp(g.float())
        dh = torch.stack([d_gate, d_up], dim=-1).view(g.shape[0], 2 * g.shape[1])
        return dh.to(g.dtype), act.to(g.dtype)

    dh, act = flex_gemm(torch.mm, (dy, w2.t()), swiglu_backward, kernel_options=OPTIONS)
    return dh @ w13.t(), x.t() @ dh, act.t() @ dy


class SwiGLUMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2):
        y, h = mlp_forward(x, w13, w2)
        ctx.save_for_backward(x, w13, w2, h)
        return y

    @staticmethod
    def backward(ctx, dy):
        return mlp_backward(*ctx.saved_tensors, dy.contiguous())


def reference_mlp(x, w13, w2):
    return swiglu(*gate_up(x @ w13)).to(x.dtype) @ w2


def main() -> None:
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    x = torch.randn(T, D, **kw, requires_grad=True)
    w13 = (torch.randn(D, 2 * INTER, **kw) / D**0.5).requires_grad_()
    w2 = (torch.randn(INTER, D, **kw) / INTER**0.5).requires_grad_()
    dy = torch.randn(T, D, **kw)
    params = (x, w13, w2)

    def fwd_bwd(mlp):
        y = mlp(*params)
        return (y, *torch.autograd.grad(y, params, dy))

    compiled_reference = torch.compile(reference_mlp, fullgraph=True)
    assert_rel_error("y, dx, dw13, dw2", fwd_bwd(SwiGLUMLP.apply), fwd_bwd(reference_mlp))
    report(
        f"SwiGLU MLP fwd+bwd T={T} D={D} I={INTER}",
        lambda: fwd_bwd(SwiGLUMLP.apply),
        lambda: fwd_bwd(compiled_reference),
        check=False,
    )


if __name__ == "__main__":
    main()
