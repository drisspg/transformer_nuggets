# SPDX-License-Identifier: Apache-2.0
"""Grouped GLU experts on FlexGEMM (QuACK grouped GEMMs) with fused forward and backward.

The caller owns dispatch ("pool") and combine ("unpool"): tokens arrive sorted by expert
with int32 end offsets, plus each routed token's router score ``s``. Every GEMM goes
through ``flex_grouped_gemm`` (``flex_gemm(F.grouped_mm, ..., backend="QUACK")``) inside
``torch.compile``; its ``epilogue(acc)`` is compiled into the GEMM's store.

``mode="fused"`` (fewest kernels; fastest for both SwiGLU and SiTU-GLU)
    fwd  act, h = glu(x @ w13[e].T), x @ w13[e].T   one kernel: GLU fused, h also stored
         y, y_u = s * (act @ w2[e].T), act @ w2[e].T one kernel
    bwd  dh = vjp(glu)(h)(s * (dy @ w2[e]))          one kernel: torch.func.vjp runs inside
                                                    the GEMM epilogue, reading the saved
                                                    interleaved h and writing interleaved dh
         ds = rowsum(dy * y_u)                       Inductor reduction
``mode="save_h"``
    fwd  h = x @ w13[e].T; act = glu(h) (Inductor); y = s * (act @ w2[e].T)
    bwd  g = dy @ w2[e]; ds = rowsum(act * g); dh = vjp(glu)(s * g) (Inductor)
``mode="recompute"``: like ``save_h`` but saves only ``act`` and recomputes ``h``.

All modes finish with dx = dh @ w13[e] (varlen-M) and the varlen-K weight gradients
dw13 = dh_e.T @ x_e, dw2 = (s * dy)_e.T @ act_e, written in the parameter layout.
The vjp-in-epilogue path is limited by epilogue math throughput, so pass
``fast_math=True`` with it. Requires an SM100 GPU and a PyTorch build with
``torch._higher_order_ops.flex_gemm`` including pytorch/pytorch#201088 (``autotune``
kernel option); ``"fused"`` also needs #201086 and #201087 (interleaved epilogue
tensors), and fast SiTU-GLU epilogues need #201085 (constant division as multiply).
"""

import functools
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

Tensor = torch.Tensor
Glu = Callable[[Tensor, Tensor], Tensor]
Mode = Literal["fused", "save_h", "recompute"]


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up


def situ_glu(gate: Tensor, up: Tensor) -> Tensor:
    """SiTU-GLU: soft-clamped SiLU gate (beta 4) times soft-clamped up (beta 25)."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(up / 25.0))


def interleave_gate_up(w1: Tensor) -> Tensor:
    """[E, 2I, D] stacked ``[gate; up]`` rows -> interleaved ``(gate_i, up_i)`` rows."""
    num_experts, two_i, hidden = w1.shape
    return w1.view(num_experts, 2, two_i // 2, hidden).transpose(1, 2).reshape_as(w1)


def _lanes(h: Tensor) -> tuple[Tensor, Tensor]:
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

    ``epilogue`` receives the fp32 accumulator tile and may read captured tensors; it is
    compiled into the GEMM's store, so its result is the only thing written to memory.
    2-D ``a`` with 3-D ``b`` is varlen-M (MoE forward, dgrad); 2-D ``a`` (m-major) with
    2-D ``b`` is varlen-K (weight gradients).
    """
    from torch._higher_order_ops import flex_gemm

    return flex_gemm(
        F.grouped_mm,
        (a, b),
        epilogue or (lambda acc: acc),
        gemm_kwargs={"offs": offs},
        kernel_options={"backend": "QUACK", "autotune": autotune, "fast_math": fast_math},
    )


def _forward(glu: Glu, mode: Mode, opts: dict, x, w13, w2, offs, scores):
    if mode == "fused":

        def up_proj(acc):
            gate, up = _lanes(acc)
            return glu(gate, up).to(acc.dtype), acc

        act, h = flex_grouped_gemm(x, w13.mT, offs, up_proj, **opts)
        y, y_unscaled = flex_grouped_gemm(
            act,
            w2.mT,
            offs,
            lambda acc: ((acc.float() * scores[:, None]).to(acc.dtype), acc),
            **opts,
        )
        return y, (act, h, y_unscaled)
    if mode == "save_h":
        h = flex_grouped_gemm(x, w13.mT, offs, **opts)
        act = glu(*_lanes(h)).to(x.dtype)
        saved = (act, h)
    else:
        # GLU fused into the up-projection: only act [T, I] is written.
        act = flex_grouped_gemm(
            x, w13.mT, offs, lambda acc: glu(*_lanes(acc)).to(acc.dtype), **opts
        )
        saved = (act,)
    # Router score fused into the down-projection.
    y = flex_grouped_gemm(
        act, w2.mT, offs, lambda acc: (acc.float() * scores[:, None]).to(acc.dtype), **opts
    )
    return y, saved


def _backward(glu: Glu, mode: Mode, opts: dict, x, w13, w2, offs, scores, saved, dy):
    act = saved[0]
    if mode == "fused":
        h, y_unscaled = saved[1:]

        def glu_backward(acc):
            # torch.func.vjp inside the epilogue: gate/up are read from the saved
            # interleaved h in place and dh is stored interleaved.
            gate, up = h.view(acc.shape[0], acc.shape[1], 2).unbind(-1)
            _, glu_vjp = torch.func.vjp(glu, gate.float(), up.float())
            d_gate, d_up = glu_vjp(acc.float() * scores[:, None])
            dh = torch.stack([d_gate, d_up], dim=-1)
            return dh.view(acc.shape[0], 2 * acc.shape[1]).to(acc.dtype)

        dh = flex_grouped_gemm(dy, w2, offs, glu_backward, **opts)
        d_scores = (dy.float() * y_unscaled.float()).sum(-1)
    else:
        g = flex_grouped_gemm(dy, w2, offs, **opts).float()
        d_scores = (act.float() * g).sum(-1)
        h = saved[1] if mode == "save_h" else flex_grouped_gemm(x, w13.mT, offs, **opts)
        _, glu_vjp = torch.func.vjp(glu, *_lanes(h))
        d_gate, d_up = glu_vjp(g * scores[:, None])
        dh = torch.stack([d_gate, d_up], dim=-1).view_as(h).to(h.dtype)
    scaled_dy = (dy.float() * scores[:, None]).to(dy.dtype)
    dx = flex_grouped_gemm(dh, w13, offs, **opts)
    # Weight gradients are varlen-K grouped GEMMs, written in the parameter layout.
    dw13 = flex_grouped_gemm(dh.mT, x, offs, **opts)
    dw2 = flex_grouped_gemm(scaled_dy.mT, act, offs, **opts)
    return dx, dw13, dw2, d_scores.to(scores.dtype)


@functools.cache
def _grouped_glu_function(
    glu: Glu, mode: Mode, autotune: bool, fast_math: bool
) -> type[torch.autograd.Function]:
    opts = {"autotune": autotune, "fast_math": fast_math}
    # Every GLU/mode shares _forward/_backward's code objects; isolate each config's
    # compiled entries so they do not count against one another's recompile limit.
    compile_fn = functools.partial(torch.compile, fullgraph=True, isolate_recompiles=True)
    fwd = compile_fn(functools.partial(_forward, glu, mode, opts))
    bwd = compile_fn(functools.partial(_backward, glu, mode, opts))

    class GroupedGLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w13, w2, offs, scores):
            y, saved = fwd(x, w13, w2, offs, scores)
            ctx.save_for_backward(x, w13, w2, offs, scores, *saved)
            return y

        @staticmethod
        def backward(ctx, dy):
            x, w13, w2, offs, scores, *saved = ctx.saved_tensors
            dx, dw13, dw2, d_scores = bwd(x, w13, w2, offs, scores, tuple(saved), dy.contiguous())
            return dx, dw13, dw2, None, d_scores

    return GroupedGLU


class GroupedGLUExperts(nn.Module):
    """Expert MLPs for expert-sorted tokens: ``s * (glu(x @ w_gate.T, x @ w_up.T) @ w2.T)``.

    Args:
        hidden_size: Token hidden size D.
        intermediate_size: Expert intermediate size I.
        num_experts: Number of experts E.
        glu: Elementwise ``glu(gate, up)``; backward uses ``torch.func.vjp``.
        mode: ``"fused"``, ``"save_h"`` or ``"recompute"`` (see the module docstring).
        autotune: Autotune each grouped GEMM's QuACK config.
        fast_math: Allow approximate transcendental math in GEMM epilogues.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        glu: Glu = situ_glu,
        mode: Mode = "save_h",
        autotune: bool = False,
        fast_math: bool = False,
    ):
        super().__init__()
        # Gate/up rows interleaved: w13[e, 2i] is gate i, w13[e, 2i + 1] is up i.
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.glu = glu
        self.mode = mode
        self.autotune = autotune
        self.fast_math = fast_math
        nn.init.normal_(self.w13, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x: Tensor, offs: Tensor, scores: Tensor | None = None) -> Tensor:
        """x: [T, D] expert-sorted tokens; offs: int32 [E] end offsets; scores: [T] or None."""
        if scores is None:
            scores = x.new_ones(x.shape[0], dtype=torch.float32)
        fn = _grouped_glu_function(self.glu, self.mode, self.autotune, self.fast_math)
        return fn.apply(x, self.w13, self.w2, offs, scores)


def reference_grouped_glu(x, w13, w2, offs, scores=None, glu: Glu = situ_glu) -> Tensor:
    """Eager ``F.grouped_mm`` reference for ``GroupedGLUExperts`` (differentiable)."""
    h = F.grouped_mm(x, w13.mT, offs=offs)
    act = glu(*_lanes(h)).to(x.dtype)
    y = F.grouped_mm(act, w2.mT, offs=offs)
    return y if scores is None else (y.float() * scores[:, None]).to(y.dtype)


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
    baselines = ("compiled", "eager")
    results: dict[tuple[str, str], float] = {}
    for glu in (swiglu, situ_glu):
        for mode in ("fused", "save_h", *baselines):
            experts = GroupedGLUExperts(
                hidden, inter, num_experts, glu, "save_h" if mode in baselines else mode,
                autotune=True, fast_math=True,
            ).cuda().to(torch.bfloat16)  # fmt: skip
            if mode in baselines:
                # "compiled" is the same F.grouped_mm reference under default torch.compile.
                ref = functools.partial(reference_grouped_glu, glu=glu)
                if mode == "compiled":
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
            results[glu.__name__, mode] = sum(k.device_time for k in kernels) / steps

    table = Table(
        title=f"Grouped GLU experts fwd+bwd: T={num_tokens}x{topk} D={hidden} "
        f"I={inter} E={num_experts}, GPU kernel time (lower is better)"
    )
    for column in ("glu", "mode", "us/step", *(f"vs {base}" for base in baselines)):
        table.add_column(column, justify="left" if column in ("glu", "mode") else "right")
    for (glu_name, mode), us in results.items():
        table.add_row(
            glu_name,
            mode,
            f"{us:.1f}",
            *(f"{results[glu_name, base] / us:.2f}x" for base in baselines),
        )
    Console().print(table)


if __name__ == "__main__":
    main()
