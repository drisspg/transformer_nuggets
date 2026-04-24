from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil

import torch

from transformer_nuggets.export_autograd_triton import (
    Specialization,
    export_autograd_triton,
    load_exported_module,
)
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds_triton


HIDDEN_SIZE = 4096
SAMPLE_TOKENS = 128
MAX_TOKENS = 2048
TOKEN_COUNTS = (1, 17, SAMPLE_TOKENS, 512)
DTYPE = torch.bfloat16
EPS = 1e-5


def rms_norm(x, weight, *, eps=EPS):
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")

    sample_x = torch.randn(
        SAMPLE_TOKENS,
        HIDDEN_SIZE,
        device="cuda",
        dtype=DTYPE,
        requires_grad=True,
    )
    weight = torch.ones(HIDDEN_SIZE, device="cuda", dtype=DTYPE, requires_grad=True)

    dynamic_tokens = torch.export.Dim("tokens", min=1, max=MAX_TOKENS)
    exports = {
        "clean_triton": _export_rms_norm(
            Path("agent_space/generated_rms_norm_clean.py"),
            sample_x,
            weight,
            dynamic_tokens,
            max_autotune=False,
        ),
        "clean_triton_max_autotune": _export_rms_norm(
            Path("agent_space/generated_rms_norm_max_autotune.py"),
            sample_x,
            weight,
            dynamic_tokens,
            max_autotune=True,
        ),
    }

    for label, output_path in exports.items():
        generated = load_exported_module(output_path)
        compiled_fn = generated.rms_norm_compiled
        print(f"\n== {label} ==")
        _validate(compiled_fn, weight)
        _benchmark_memory_bandwidth(label, compiled_fn, weight)
        _print_artifact_summary(output_path)


def _export_rms_norm(
    output_path: Path,
    x: torch.Tensor,
    weight: torch.Tensor,
    dynamic_tokens: object,
    *,
    max_autotune: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_path.with_name(f"{output_path.stem}_artifacts")
    output_path.unlink(missing_ok=True)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    export_autograd_triton(
        rms_norm,
        specializations=[
            Specialization(
                args=(x, weight),
                kwargs={"eps": EPS},
                dynamic_shapes={"x": {0: dynamic_tokens}},
            )
        ],
        out=output_path,
        source_backend="clean_triton",
        max_autotune=max_autotune,
    )
    return output_path


def _validate(compiled_fn: Callable, weight: torch.Tensor) -> None:
    for tokens in TOKEN_COUNTS:
        runtime_x = torch.randn(
            tokens,
            HIDDEN_SIZE,
            device="cuda",
            dtype=DTYPE,
            requires_grad=True,
        )

        eager = rms_norm(runtime_x, weight, eps=EPS)
        compiled = compiled_fn(runtime_x, weight, eps=EPS)
        torch.testing.assert_close(compiled, eager, rtol=2e-2, atol=2e-2)

        eager_grads = torch.autograd.grad(
            eager.float().sum(), (runtime_x, weight), retain_graph=True
        )
        compiled_grads = torch.autograd.grad(compiled.float().sum(), (runtime_x, weight))
        for compiled_grad, eager_grad in zip(compiled_grads, eager_grads, strict=True):
            torch.testing.assert_close(compiled_grad, eager_grad, rtol=2e-2, atol=2e-2)
        print(f"tokens={tokens}: {compiled.shape}")


def _benchmark_memory_bandwidth(label: str, compiled_fn: Callable, weight: torch.Tensor) -> None:
    print("forward bandwidth (assumes 2 x reads + 1 weight read + 1 output write):")
    for tokens in TOKEN_COUNTS:
        benchmark_x = torch.randn(
            tokens,
            HIDDEN_SIZE,
            device="cuda",
            dtype=DTYPE,
        )

        def run_forward():
            with torch.no_grad():
                return compiled_fn(benchmark_x, weight, eps=EPS)

        time_us = benchmark_cuda_function_in_microseconds_triton(run_forward)
        bandwidth_gb_s = _forward_memory_bytes(benchmark_x) / (time_us * 1e-6) / 1e9
        print(f"  tokens={tokens}: {time_us:.2f} us, ~{bandwidth_gb_s:.1f} GB/s")


def _forward_memory_bytes(x: torch.Tensor) -> int:
    return 4 * x.numel() * x.element_size()


def _print_artifact_summary(output_path: Path) -> None:
    artifact_dir = output_path.with_name(f"{output_path.stem}_artifacts")
    artifact_source = "\n".join(path.read_text() for path in artifact_dir.glob("*.py"))
    print(f"generated file: {output_path}")
    print(f"artifact dir: {artifact_dir}")
    print(f"inline Triton: {'@triton.jit' in artifact_source}")
    print(f"uses async_compile.triton: {'async_compile.triton' in artifact_source}")


if __name__ == "__main__":
    main()
