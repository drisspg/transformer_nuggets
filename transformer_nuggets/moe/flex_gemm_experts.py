# SPDX-License-Identifier: Apache-2.0
"""Build fused GEMM epilogues in PyTorch: a worked MoE example.

The MoE supplies the workload; the lesson is how to compose ordinary PyTorch functions
into GEMM kernels with FlexGEMM. Start with ``reference_grouped_glu``, then read
``flex_grouped_gemm`` and V1 -> V2 -> V3. Each version keeps the same six GEMMs (two
forward, four backward) and moves more work into their epilogues. An epilogue runs
before the GEMM stores its result, transforming it or writing auxiliary outputs
without a separate elementwise kernel. All GEMMs here use the QuACK backend.

Inputs and forward math
-----------------------
The caller owns dispatch and combine. This module sees T routed token copies sorted
by expert, not the original unsorted tokens. D is the hidden size, I the intermediate
size, and E the number of experts. ``offs`` is int32 [E] cumulative end offsets:
``[2, 2, 5]`` assigns rows 0:2 to expert 0, no rows to expert 1, and rows 2:5 to expert 2.
``scores`` has one router weight per routed row.

For each expert e's rows (``s = scores[:, None]`` below):

    h   = x @ w13[e].T                 # [T, 2I]: gate_0, up_0, gate_1, up_1, ...
    act = glu(h[:, 0::2], h[:, 1::2])   # [T, I]
    y   = s * (act @ w2[e].T)          # [T, D]

FlexGEMM techniques to take away
-------------------------------
V1: Start from the math.
    ``flex_grouped_gemm`` wraps ``flex_gemm(F.grouped_mm, ...)`` with an identity
    epilogue. GLU and scaling remain outside the GEMMs. The explicit backward makes
    the producer/consumer relationships visible before deciding what to fuse.
V2: Write an epilogue as a PyTorch function.
    ``glu_and_keep_h`` returns both the activation and the pre-activation: a tuple
    becomes multiple kernel outputs, including a value saved for backward.
    ``scale_by_router_score`` captures scores from its enclosing scope. The backward
    is unchanged from V1, isolating what forward epilogue fusion buys.
V3: Compose autograd and reductions inside an epilogue.
    ``glu_backward`` captures saved h and calls ``torch.func.vjp`` on the activation.
    Its traced derivative runs inside the GEMM kernel. The same epilogue returns dh,
    scaled act, and grouped partial sums for the router gradient; only the final sum
    runs outside. Recomputing act here also removes it from the saved intermediates.

Each version has an explicit ``torch.autograd.Function`` and compiled forward/backward.
FlexGEMM does not derive the whole backward: we choose its GEMMs and fusion boundaries;
``torch.func.vjp`` derives the local activation gradient. V1/V2 save h and act, while
V3 saves only h: 66 MiB versus 44 MiB at the shape below, excluding inputs/weights.

Backward math
-------------
Given ``dy = dL/dy``, the common algebra is (omitting dtype conversions):

    g       = dy @ w2[e]             # [T, I], before applying the router score
    ds      = sum_i(act * g)         # [T], one scalar per routed row
    dh      = GLU_vjp(h, s * g)      # [T, 2I], gate/up gradients interleaved
    dx      = dh @ w13[e]            # [T, D]
    dw13[e] = dh_e.T @ x_e           # [2I, D], reduce over expert e's rows
    dw2[e]  = dy_e.T @ (s * act)_e    # [D, I]

The remaining three GEMMs after g share ``input_and_weight_grads``.

Try changing the computation
---------------------------
``swiglu`` and ``situ_glu`` are ordinary PyTorch functions, not named compiler modes.
To try a different GLU, write a deterministic pointwise ``glu(gate, up)`` returning
the same shape and pass it as ``GroupedGLUExperts(..., glu=your_glu)``. The epilogues
call that function and its VJP; no activation-specific backward needs to be written.
Both the forward operations and their derivatives must be supported by the compiler.
Use ``reference_grouped_glu`` to check outputs and gradients, then compare V1/V2/V3:
fusing more work is an experiment, not a guarantee that the most fused version wins.

Recorded timings
----------------
Run ``python -m transformer_nuggets.moe.flex_gemm_experts`` for a fresh GPU-time table.
The following measurements were recorded in commit c0ed3ef on GB200 (T=8192 routed
tokens, D=2048, I=1408, E=8, bf16, ``autotune=True, fast_math=True``).
Fwd+bwd GPU kernel time per step, lower is better:

    impl                     SwiGLU    SiTU-GLU
    eager F.grouped_mm      908 us    1142 us
    torch.compile(eager)    519 us     519 us
    V1                      522 us     566 us
    V2                      473 us     502 us
    V3                      437 us     447 us

Requires an SM100 GPU and a PyTorch build with ``torch._higher_order_ops.flex_gemm``
including pytorch/pytorch#201088 (``autotune`` kernel option); V3 also needs #201086 and
#201087 (interleaved epilogue tensors), and fast SiTU-GLU epilogues need #201085
(constant division as multiply).
"""

import functools
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

Tensor = torch.Tensor
Glu = Callable[[Tensor, Tensor], Tensor]


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up


def situ_glu(gate: Tensor, up: Tensor) -> Tensor:
    """SiTU-GLU: soft-clamped SiLU gate (beta 4) times soft-clamped up (beta 25)."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(up / 25.0))


def interleave_gate_up(w1: Tensor) -> Tensor:
    """[E, 2I, D] stacked ``[gate; up]`` rows -> interleaved ``(gate_i, up_i)`` rows."""
    num_experts, two_i, hidden = w1.shape
    return w1.view(num_experts, 2, two_i // 2, hidden).transpose(1, 2).reshape_as(w1)


def gate_and_up(h: Tensor) -> tuple[Tensor, Tensor]:
    """Split interleaved ``[T, 2I]`` pre-activations into fp32 gate and up ``[T, I]``."""
    pairs = h.float().view(h.shape[0], -1, 2)
    return pairs[..., 0], pairs[..., 1]


def flex_grouped_gemm(
    a: Tensor,
    b: Tensor,
    offs: Tensor,
    epilogue=None,
    *,
    autotune: bool = False,
    fast_math: bool = False,
) -> Tensor:
    """``epilogue(F.grouped_mm(a, b, offs=offs))`` as one fused FlexGEMM/QuACK kernel.

    ``epilogue`` may capture tensors and return one tensor or a tuple of outputs.
    Omit it for an identity epilogue. ``offs`` partitions different axes by operand rank:

    * 2-D ``a`` @ 3-D ``b``: split output rows by expert (varlen-M: forward, dx, g).
    * 2-D ``a`` @ 2-D ``b``: split the reduction by expert (varlen-K: weight gradients).
      ``a`` is a transposed token matrix, so each expert produces its own weight gradient.
    """
    from torch._higher_order_ops import flex_gemm

    return flex_gemm(
        F.grouped_mm,
        (a, b),
        epilogue or (lambda acc: acc),
        gemm_kwargs={"offs": offs},
        kernel_options={"backend": "QUACK", "autotune": autotune, "fast_math": fast_math},
    )


# --------------------------------------------------------------------------------------
# Shared backward tail: input gradient and weight gradients from dh.
# --------------------------------------------------------------------------------------


def input_and_weight_grads(x, w13, offs, dh, dy, scaled_act, opts):
    # Input gradient: varlen-M, dh [T, 2I] @ w13[e] [2I, D].
    dx = flex_grouped_gemm(dh, w13, offs, **opts)
    # Weight gradients: varlen-K (the reduction runs over each expert's tokens), written
    # directly in the parameter layout: dw13[e] = dh_e.T @ x_e, dw2[e] = dy_e.T @ (s * act)_e.
    # Put the score on act, not dy: [T, I] is smaller than [T, D] at the example's I < D.
    dw13 = flex_grouped_gemm(dh.mT, x, offs, **opts)
    dw2 = flex_grouped_gemm(dy.mT, scaled_act, offs, **opts)
    return dx, dw13, dw2


# --------------------------------------------------------------------------------------
# V1: every GEMM is a FlexGEMM grouped GEMM; GLU, scaling and its backward are PyTorch.
# --------------------------------------------------------------------------------------


@torch.compile(fullgraph=True)
def v1_forward(x, w13, w2, offs, scores, glu, opts):
    h = flex_grouped_gemm(x, w13.mT, offs, **opts)
    act = glu(*gate_and_up(h)).to(x.dtype)
    y = flex_grouped_gemm(act, w2.mT, offs, **opts)
    y = (y.float() * scores[:, None]).to(y.dtype)
    return y, h, act


@torch.compile(fullgraph=True)
def v1_backward(x, w13, w2, offs, scores, h, act, dy, glu, opts):
    # g = dy @ w2; dL/dact = scores[:, None] * g. Keep g unscaled for d_scores.
    g = flex_grouped_gemm(dy, w2, offs, **opts).float()
    d_scores = (act.float() * g).sum(-1)
    # The VJP maps dL/dact to (dL/dgate, dL/dup), tracing ordinary PyTorch ops.
    _, glu_vjp = torch.func.vjp(glu, *gate_and_up(h))
    d_gate, d_up = glu_vjp(g * scores[:, None])
    dh = torch.stack([d_gate, d_up], dim=-1).view_as(h).to(h.dtype)
    scaled_act = (act.float() * scores[:, None]).to(act.dtype)
    dx, dw13, dw2 = input_and_weight_grads(x, w13, offs, dh, dy, scaled_act, opts)
    return dx, dw13, dw2, d_scores.to(scores.dtype)


class GroupedGLUV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2, offs, scores, glu, opts):
        y, h, act = v1_forward(x, w13, w2, offs, scores, glu, opts)
        ctx.save_for_backward(x, w13, w2, offs, scores, h, act)
        ctx.glu, ctx.opts = glu, opts
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w13, w2, offs, scores, h, act = ctx.saved_tensors
        dx, dw13, dw2, d_scores = v1_backward(
            x, w13, w2, offs, scores, h, act, dy.contiguous(), ctx.glu, ctx.opts
        )
        return dx, dw13, dw2, None, d_scores, None, None


# --------------------------------------------------------------------------------------
# V2: V1 + elementwise work fused into the forward GEMM epilogues.
# --------------------------------------------------------------------------------------


@torch.compile(fullgraph=True)
def v2_forward(x, w13, w2, offs, scores, glu, opts):
    def glu_and_keep_h(acc):
        # One kernel writes act [T, I] (the GLU of each gate/up pair) and h [T, 2I].
        gate, up = gate_and_up(acc)
        return glu(gate, up).to(acc.dtype), acc

    act, h = flex_grouped_gemm(x, w13.mT, offs, glu_and_keep_h, **opts)

    def scale_by_router_score(acc):
        return (acc.float() * scores[:, None]).to(acc.dtype)

    y = flex_grouped_gemm(act, w2.mT, offs, scale_by_router_score, **opts)
    return y, h, act


class GroupedGLUV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2, offs, scores, glu, opts):
        y, h, act = v2_forward(x, w13, w2, offs, scores, glu, opts)
        ctx.save_for_backward(x, w13, w2, offs, scores, h, act)
        ctx.glu, ctx.opts = glu, opts
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w13, w2, offs, scores, h, act = ctx.saved_tensors
        dx, dw13, dw2, d_scores = v1_backward(
            x, w13, w2, offs, scores, h, act, dy.contiguous(), ctx.glu, ctx.opts
        )
        return dx, dw13, dw2, None, d_scores, None, None


# --------------------------------------------------------------------------------------
# V3: V2 + the GLU backward fused into the dy @ w2 GEMM epilogue.
# --------------------------------------------------------------------------------------


@torch.compile(fullgraph=True)
def v3_forward(x, w13, w2, offs, scores, glu, opts):
    def glu_and_keep_h(acc):
        gate, up = gate_and_up(acc)
        return glu(gate, up).to(acc.dtype), acc

    act, h = flex_grouped_gemm(x, w13.mT, offs, glu_and_keep_h, **opts)

    def scale_by_router_score(acc):
        return (acc.float() * scores[:, None]).to(acc.dtype)

    y = flex_grouped_gemm(act, w2.mT, offs, scale_by_router_score, **opts)
    # Only h is saved: the backward GEMM epilogue rebuilds everything else from it.
    return y, h


# Reduce scores in column groups: I and the GEMM tile N must be divisible by this size.
SCORE_GRAD_GROUP = 32


@torch.compile(fullgraph=True)
def v3_backward(x, w13, w2, offs, scores, h, dy, glu, opts):
    def glu_backward(acc):
        # acc is g = dy @ w2[e]. The VJP also recomputes act from saved h.
        # One GEMM writes everything needed by the remaining backward:
        #   dh          [T, 2I], interleaved gate/up gradients for dx and dw13
        #   scaled_act  [T, I], the right operand of dw2
        #   partials    [T, I / SCORE_GRAD_GROUP], fp32 partial sums for d_scores
        # The epilogue reduces within column groups; the row sum runs after the GEMM.
        gate, up = h.view(acc.shape[0], acc.shape[1], 2).unbind(-1)
        act, glu_vjp = torch.func.vjp(glu, gate.float(), up.float())
        g = acc.float()
        d_gate, d_up = glu_vjp(g * scores[:, None])
        dh = torch.stack([d_gate, d_up], dim=-1).view(acc.shape[0], 2 * acc.shape[1])
        act = act.to(acc.dtype)  # the rounding the forward fed to the down-projection
        scaled_act = (act.float() * scores[:, None]).to(acc.dtype)
        partials = (act.float() * g).view(acc.shape[0], -1, SCORE_GRAD_GROUP).sum(-1)
        return dh.to(acc.dtype), scaled_act, partials

    dh, scaled_act, partials = flex_grouped_gemm(dy, w2, offs, glu_backward, **opts)
    d_scores = partials.sum(-1)
    dx, dw13, dw2 = input_and_weight_grads(x, w13, offs, dh, dy, scaled_act, opts)
    return dx, dw13, dw2, d_scores.to(scores.dtype)


class GroupedGLUV3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2, offs, scores, glu, opts):
        y, h = v3_forward(x, w13, w2, offs, scores, glu, opts)
        ctx.save_for_backward(x, w13, w2, offs, scores, h)
        ctx.glu, ctx.opts = glu, opts
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w13, w2, offs, scores, h = ctx.saved_tensors
        dx, dw13, dw2, d_scores = v3_backward(
            x, w13, w2, offs, scores, h, dy.contiguous(), ctx.glu, ctx.opts
        )
        return dx, dw13, dw2, None, d_scores, None, None


# --------------------------------------------------------------------------------------
# Module and reference.
# --------------------------------------------------------------------------------------


class GroupedGLUExperts(nn.Module):
    """Expert MLPs for expert-sorted tokens: ``s * (glu(x @ w_gate.T, x @ w_up.T) @ w2.T)``.

    Args:
        hidden_size: Token hidden size D.
        intermediate_size: Expert intermediate size I.
        num_experts: Number of experts E.
        glu: Elementwise ``glu(gate, up)``; backward uses ``torch.func.vjp``.
        impl: ``GroupedGLUV1``, ``GroupedGLUV2`` or ``GroupedGLUV3``.
        autotune: Autotune each grouped GEMM's QuACK config.
        fast_math: Allow approximate transcendental math in GEMM epilogues.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        glu: Glu = situ_glu,
        impl: type[torch.autograd.Function] = GroupedGLUV3,
        autotune: bool = False,
        fast_math: bool = False,
    ):
        super().__init__()
        # Gate/up rows interleaved: w13[e, 2i] is gate i, w13[e, 2i + 1] is up i.
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.glu = glu
        self.impl = impl
        self.opts = {"autotune": autotune, "fast_math": fast_math}
        nn.init.normal_(self.w13, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x: Tensor, offs: Tensor, scores: Tensor) -> Tensor:
        """x: [T, D] expert-sorted tokens; offs: int32 [E] end offsets; scores: [T]."""
        return self.impl.apply(x, self.w13, self.w2, offs, scores, self.glu, self.opts)


def reference_grouped_glu(x, w13, w2, offs, scores, glu: Glu = situ_glu) -> Tensor:
    """Eager ``F.grouped_mm`` reference for ``GroupedGLUExperts`` (differentiable)."""
    h = F.grouped_mm(x, w13.mT, offs=offs)
    act = glu(*gate_and_up(h)).to(x.dtype)
    y = F.grouped_mm(act, w2.mT, offs=offs)
    return (y.float() * scores[:, None]).to(y.dtype)


def main():
    from rich.console import Console
    from rich.table import Table
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    num_tokens, hidden, inter, num_experts, topk = 4096, 2048, 1408, 8, 2
    # Stand-in for dispatch: route each token copy to an expert and sort by expert.
    choice = torch.randint(num_experts, (num_tokens * topk,), device="cuda").sort().values
    offs = torch.bincount(choice, minlength=num_experts).cumsum(0).to(torch.int32)
    x = torch.randn(num_tokens * topk, hidden, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_()
    scores = torch.rand(num_tokens * topk, device="cuda")
    dy = torch.randn_like(x)
    impls = {"V1": GroupedGLUV1, "V2": GroupedGLUV2, "V3": GroupedGLUV3}
    baselines = ("compiled", "eager")
    results: dict[tuple[str, str], float] = {}
    for glu in (swiglu, situ_glu):
        for name in (*impls, *baselines):
            experts = GroupedGLUExperts(
                hidden, inter, num_experts, glu, impls.get(name, GroupedGLUV1),
                autotune=True, fast_math=True,
            ).cuda().to(torch.bfloat16)  # fmt: skip
            if name in baselines:
                # "compiled" is the same F.grouped_mm reference under default torch.compile.
                ref = functools.partial(reference_grouped_glu, glu=glu)
                if name == "compiled":
                    ref = torch.compile(ref, fullgraph=True)
                fn = lambda e=experts, ref=ref: ref(x, e.w13, e.w2, offs, scores)
            else:
                fn = lambda e=experts: e(x, offs, scores)

            def step(fn=fn, experts=experts):
                x.grad = experts.w13.grad = experts.w2.grad = None
                fn().backward(dy)

            for _ in range(5):
                step()
            torch.cuda.synchronize()
            steps = 20
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(steps):
                    step()
                torch.cuda.synchronize()
            kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            results[glu.__name__, name] = sum(k.device_time for k in kernels) / steps

    table = Table(
        title=f"Grouped GLU experts fwd+bwd: T={num_tokens}x{topk} D={hidden} "
        f"I={inter} E={num_experts}, GPU kernel time (lower is better)"
    )
    for column in ("glu", "impl", "us/step", *(f"vs {base}" for base in baselines)):
        table.add_column(column, justify="left" if column in ("glu", "impl") else "right")
    for (glu_name, name), us in results.items():
        table.add_row(
            glu_name,
            name,
            f"{us:.1f}",
            *(f"{results[glu_name, base] / us:.2f}x" for base in baselines),
        )
    Console().print(table)


if __name__ == "__main__":
    main()
