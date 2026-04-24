"""Dynamic export example.

Use source_backend="clean_triton" when the exporter can rewrite every dynamic
Triton launch family in the artifact. Use source_backend="inductor" when an
unsupported dynamic family should keep Inductor's symbolic launch runtime.
"""

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
            )
        ],
        out=output_path,
        source_backend="clean_triton",
    )

    generated = load_exported_module(output_path)
    for batch in (1, 4, 16):
        dynamic_x = torch.randn(batch, 8, device="cuda", requires_grad=True)

        eager = affine_relu(dynamic_x, w)
        compiled = generated.affine_relu_compiled(dynamic_x, w)
        torch.testing.assert_close(compiled, eager)

        eager_grads = torch.autograd.grad(eager.sum(), (dynamic_x, w), retain_graph=True)
        compiled_grads = torch.autograd.grad(compiled.sum(), (dynamic_x, w))
        for compiled_grad, eager_grad in zip(compiled_grads, eager_grads, strict=True):
            torch.testing.assert_close(compiled_grad, eager_grad)
        print(f"batch={batch}: {compiled.shape}")

    print(f"generated file: {output_path}")
    print(f"artifact dir: {output_path.with_name(f'{output_path.stem}_artifacts')}")


if __name__ == "__main__":
    main()
