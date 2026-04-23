from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from transformer_nuggets.export_autograd_triton import Specialization, export_autograd_triton


def affine_relu(x, w):
    return torch.relu(x @ w)


def import_generated(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    generated = import_generated(output_path)
    y = generated.affine_relu_compiled(x, w)
    y.sum().backward()
    print(f"wrote {output_path}")
    print(y)


if __name__ == "__main__":
    main()
