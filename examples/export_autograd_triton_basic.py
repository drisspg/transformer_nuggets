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
        [Specialization(args=(x, w), name="static_4x8_8x3")],
        output_path,
    )
    generated = load_exported_module(output_path)
    y = generated.affine_relu_compiled(x, w)
    y.sum().backward()
    print(f"wrote {output_path}")
    print(y)


if __name__ == "__main__":
    main()
