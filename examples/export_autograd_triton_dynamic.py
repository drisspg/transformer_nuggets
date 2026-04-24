from __future__ import annotations

from pathlib import Path

import torch

from transformer_nuggets.export_autograd_triton import (
    Specialization,
    export_autograd_triton,
    load_exported_module,
)


def affine_relu(x, w):
    return torch.relu(x @ w)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")

    output_path = Path("agent_space/generated_dynamic_affine_relu.py")
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    w = torch.randn(8, 3, device="cuda", requires_grad=True)

    export_autograd_triton(
        affine_relu,
        specializations=[
            Specialization(
                args=(x, w),
                dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=16)}},
                name="dynamic_batch",
            )
        ],
        out=output_path,
        source_backend="clean_triton",
    )

    generated = load_exported_module(output_path)
    for batch in (1, 4, 16):
        eager_x = torch.randn(batch, 8, device="cuda", requires_grad=True)
        compiled_x = eager_x.detach().clone().requires_grad_()
        eager_w = w.detach().clone().requires_grad_()
        compiled_w = w.detach().clone().requires_grad_()

        eager = affine_relu(eager_x, eager_w)
        compiled = generated.affine_relu_compiled(compiled_x, compiled_w)
        torch.testing.assert_close(compiled, eager)

        eager_grads = torch.autograd.grad(eager.sum(), (eager_x, eager_w))
        compiled_grads = torch.autograd.grad(compiled.sum(), (compiled_x, compiled_w))
        for compiled_grad, eager_grad in zip(compiled_grads, eager_grads, strict=True):
            torch.testing.assert_close(compiled_grad, eager_grad)
        print(f"batch={batch}: {compiled.shape}")

    print(f"generated file: {output_path}")
    print(f"artifact dir: {output_path.with_name(f'{output_path.stem}_artifacts')}")


if __name__ == "__main__":
    main()
