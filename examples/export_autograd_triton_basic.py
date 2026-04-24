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

    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    w = torch.randn(8, 3, device="cuda", requires_grad=True)
    output_path = Path("generated_affine_relu.py")

    export_autograd_triton(
        affine_relu,
        [Specialization(args=(x, w))],
        output_path,
    )
    generated = load_exported_module(output_path)
    eager = affine_relu(x, w)
    compiled = generated.affine_relu_compiled(x, w)
    torch.testing.assert_close(compiled, eager)

    eager_grads = torch.autograd.grad(eager.sum(), (x, w), retain_graph=True)
    compiled_grads = torch.autograd.grad(compiled.sum(), (x, w))
    for compiled_grad, eager_grad in zip(compiled_grads, eager_grads, strict=True):
        torch.testing.assert_close(compiled_grad, eager_grad)

    print(f"wrote {output_path}")
    print(compiled)


if __name__ == "__main__":
    main()
