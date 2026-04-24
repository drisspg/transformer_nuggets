from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from transformer_nuggets.export_autograd_triton import Specialization, export_autograd_triton


def affine_relu(x, w):
    return torch.relu(x @ w)


def import_python_file(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        source_backend="inductor",
    )

    generated = import_python_file(output_path)
    for batch in (1, 4, 16):
        dynamic_x = torch.randn(batch, 8, device="cuda", requires_grad=True)
        eager = affine_relu(dynamic_x, w)
        compiled = generated.affine_relu_compiled(dynamic_x, w)
        torch.testing.assert_close(compiled, eager)
        print(f"batch={batch}: {compiled.shape}")

    print(f"generated file: {output_path}")
    print(f"artifact dir: {output_path.with_name(f'{output_path.stem}_artifacts')}")


if __name__ == "__main__":
    main()
