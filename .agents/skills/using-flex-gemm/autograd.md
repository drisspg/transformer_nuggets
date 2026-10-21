# FlexGEMM with an explicit backward

FlexGEMM has no built-in autograd formula. Wrap the complete mathematical block in `torch.autograd.Function` and own its backward explicitly. A custom Function's `forward` runs without autograd recording, so the enclosed FlexGEMM call does not require its own formula.

## Reverse the stages

For

```text
x -- P --> p -- GEMM(weight) --> acc -- E --> y
```

the backward is

```text
grad_y -- E' --> grad_acc -- GEMM(weight.T) --> grad_p -- P' --> grad_x
```

- The forward **epilogue** VJP `E'` becomes a backward **prologue** before the gradient GEMMs.
- The forward **prologue** VJP `P'` runs after the input-gradient GEMM, so supported shape-preserving work can become a backward FlexGEMM **epilogue**.

Current FlexGEMM does not automatically absorb `E'` into both gradient GEMMs. Compute `grad_acc` once unless another fused producer naturally creates it.

## Example: scaled input, GEMM, and ReLU

```python
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch._higher_order_ops import flex_gemm

KERNEL_OPTIONS = {"backend": "QUACK", "autotune": True}


class FlexLinearRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, input_scale):
        out = flex_gemm(
            torch.mm,
            (x * input_scale, weight),
            lambda acc: F.relu(acc),
            kernel_options=KERNEL_OPTIONS,
        )
        ctx.save_for_backward(x, weight, out)
        ctx.input_scale = input_scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x, weight, out = ctx.saved_tensors

        # E': the forward ReLU epilogue becomes a backward prologue.
        grad_acc = grad_out * (out > 0)

        grad_x = None
        if ctx.needs_input_grad[0]:
            # P': the forward scale becomes the grad_x GEMM epilogue.
            grad_x = flex_gemm(
                torch.mm,
                (grad_acc, weight.t()),
                lambda acc: acc * ctx.input_scale,
                kernel_options=KERNEL_OPTIONS,
            )

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = (x.t() @ grad_acc) * ctx.input_scale

        return grad_x, grad_weight, None


def linear_relu(x, weight, input_scale=0.5):
    return FlexLinearRelu.apply(x, weight, input_scale)
```

Compile the function that calls `.apply`:

```python
compiled = torch.compile(linear_relu, backend="inductor", fullgraph=True)
out = compiled(x, weight)
out.backward(grad_out)
```

The example is intentionally first-order: `once_differentiable` makes double-backward fail explicitly because the backward itself contains FlexGEMM. If higher-order gradients or `torch.func` transforms are required, design them separately and use the corresponding `setup_context` interface.

## Derive `E'` with `torch.func.vjp`

Instead of writing `E'` by hand, call `torch.func.vjp` on the forward epilogue function inside the backward GEMM's epilogue. The derivative math then runs in the kernel that produces the incoming gradient, and the recomputed forward value comes for free:

```python
def swiglu(gate, up):
    return F.silu(gate) * up


def swiglu_backward(g):
    # h is the captured [T, 2I] pre-activation saved by the forward (interleaved lanes).
    # Split the captured lanes before converting to fp32.
    gate, up = h.view(g.shape[0], g.shape[1], 2).unbind(-1)
    act, vjp = torch.func.vjp(swiglu, gate.float(), up.float())
    d_gate, d_up = vjp(g.float())
    dh = torch.stack([d_gate, d_up], dim=-1).view(g.shape[0], 2 * g.shape[1])
    return dh.to(g.dtype), act.to(g.dtype)


dh, act = flex_gemm(torch.mm, (dy, w2.t()), swiglu_backward, kernel_options=KERNEL_OPTIONS)
```

The complete SwiGLU MLP (forward saves `h` as the physical-accumulator auxiliary, backward recomputes `act` and emits `dh` from one GEMM) is `examples/flex_gemm/06_backward.py`; the grouped-GEMM MoE version is `transformer_nuggets/moe/flex_gemm_experts.py`. Pass explicit cotangents: `torch.func.grad` of a `.sum()` emits `aten.expand`, which FlexGEMM does not lower (see [patterns.md](patterns.md)).

## Design another backward

Given `y = E(P(x) @ weight)`:

1. Derive `grad_acc = vjp_E(saved_values, grad_y)`.
2. Decide which forward values to save and which to recompute.
3. Compute `grad_acc @ weight.T` for the activation gradient.
4. Apply `P'` as that GEMM's epilogue when [supported](constraints.md).
5. Compute `P(x).T @ grad_acc` for the weight gradient.
6. Return `None` for non-differentiable arguments.

For grouped-main epilogues such as SwiGLU, `E'` expands logical `[M, N]` gradients back into physical `[M, group * N]` lanes in the exact forward lane order.

The final output is not always sufficient for `E'`. ReLU can use `out > 0`; SiLU generally requires the accumulator, so save it, return a smaller sufficient auxiliary, or intentionally recompute the forward GEMM.

## Save or recompute

Prefer an already-returned value, then a small auxiliary, then cheap recomputation. Save or recompute the full accumulator only after considering its memory and runtime cost. Avoid adding a full-size auxiliary store solely to simplify backward.

Use `ctx.needs_input_grad` to skip unnecessary gradient GEMMs and reductions.

## Validate

Use independent leaf tensors for the reference:

```python
x_expected = x_actual.detach().clone().requires_grad_()
w_expected = w_actual.detach().clone().requires_grad_()

actual = linear_relu(x_actual, w_actual, scale)
expected = F.relu((x_expected * scale) @ w_expected)
actual.backward(grad)
expected.backward(grad)

torch.testing.assert_close(actual, expected)
torch.testing.assert_close(x_actual.grad, x_expected.grad)
torch.testing.assert_close(w_actual.grad, w_expected.grad)
```

Repeat through `torch.compile(..., fullgraph=True)` and confirm the intended generated routes for both forward and backward.
