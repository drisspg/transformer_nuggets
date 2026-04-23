from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import tree_flatten


@dataclass
class CompiledGraphSource:
    graph_code: str
    source_code: str


@dataclass
class CapturedCompiledSources:
    forward: CompiledGraphSource
    backward: CompiledGraphSource | None
    num_user_outputs: int
    output_kind: str
    needs_autograd: bool


def capture_compiled_autograd_sources(
    fn: Callable[..., Any],
    runtime_tensors: tuple[torch.Tensor, ...],
) -> CapturedCompiledSources:
    try:
        from torch._functorch.aot_autograd import aot_function, make_boxed_compiler
        from torch._inductor.compile_fx import compile_fx
    except ImportError as exc:
        raise RuntimeError(
            "export_autograd_triton requires PyTorch nightly internals: "
            "torch._functorch.aot_autograd and torch._inductor.compile_fx"
        ) from exc

    if not runtime_tensors:
        raise ValueError("At least one tensor runtime argument is required")

    sample_tensors = tuple(_clone_for_capture(tensor) for tensor in runtime_tensors)
    original_outputs = fn(*tuple(_clone_for_capture(tensor) for tensor in runtime_tensors))
    flat_original_outputs, _ = tree_flatten(original_outputs)
    tensor_outputs = [
        output for output in flat_original_outputs if isinstance(output, torch.Tensor)
    ]
    if not tensor_outputs:
        raise ValueError("Exported function must produce at least one tensor output")
    output_kind = _output_kind(original_outputs)
    differentiable_outputs = [
        output for output in tensor_outputs if output.requires_grad and _is_differentiable(output)
    ]
    if not differentiable_outputs:
        return _capture_forward_only(fn, sample_tensors, len(tensor_outputs), output_kind)

    if not any(tensor.requires_grad for tensor in runtime_tensors):
        return _capture_forward_only(fn, sample_tensors, len(tensor_outputs), output_kind)

    records: list[CompiledGraphSource] = []

    def compiler(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[..., Any]:
        compiled = compile_fx(gm, example_inputs)
        records.append(_compiled_graph_source(gm, compiled))
        return compiled

    compiled_fn = aot_function(
        fn,
        fw_compiler=make_boxed_compiler(compiler),
        bw_compiler=make_boxed_compiler(compiler),
    )
    outputs = compiled_fn(*sample_tensors)
    flat_outputs, _ = tree_flatten(outputs)
    compiled_differentiable_outputs = [
        output
        for output in flat_outputs
        if isinstance(output, torch.Tensor) and output.requires_grad and _is_differentiable(output)
    ]
    loss = sum(_real_scalar_loss(output) for output in compiled_differentiable_outputs)
    loss.backward()

    if len(records) < 2:
        raise RuntimeError("AOTAutograd did not compile both forward and backward graphs")

    return CapturedCompiledSources(
        forward=records[0],
        backward=records[1],
        num_user_outputs=len(tensor_outputs),
        output_kind=output_kind,
        needs_autograd=True,
    )


def _capture_forward_only(
    fn: Callable[..., Any],
    sample_tensors: tuple[torch.Tensor, ...],
    num_user_outputs: int,
    output_kind: str,
) -> CapturedCompiledSources:
    try:
        export_result = torch._dynamo.export(fn, aten_graph=True)(*sample_tensors)
    except AttributeError as exc:
        raise RuntimeError("Forward-only export requires torch._dynamo.export") from exc
    from torch._inductor.compile_fx import compile_fx

    compiled = compile_fx(export_result.graph_module, list(sample_tensors))
    return CapturedCompiledSources(
        forward=_compiled_graph_source(export_result.graph_module, compiled),
        backward=None,
        num_user_outputs=num_user_outputs,
        output_kind=output_kind,
        needs_autograd=False,
    )


def _compiled_graph_source(
    gm: torch.fx.GraphModule, compiled: Callable[..., Any]
) -> CompiledGraphSource:
    source_code = getattr(compiled, "source_code", None)
    if source_code is None:
        raise RuntimeError("Inductor did not expose source_code for the compiled FX graph")
    return CompiledGraphSource(graph_code=gm.code, source_code=source_code)


def _clone_for_capture(tensor: torch.Tensor) -> torch.Tensor:
    clone = torch.empty_strided(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    clone.copy_(tensor.detach())
    if _is_differentiable(clone):
        clone.requires_grad_(tensor.requires_grad)
    return clone


def _is_differentiable(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() or tensor.is_complex()


def _real_scalar_loss(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_complex():
        return tensor.real.sum()
    return tensor.sum()


def _output_kind(outputs: Any) -> str:
    if isinstance(outputs, torch.Tensor):
        return "single"
    if isinstance(outputs, tuple) and all(isinstance(output, torch.Tensor) for output in outputs):
        return "tuple"
    if isinstance(outputs, list) and all(isinstance(output, torch.Tensor) for output in outputs):
        return "list"
    raise TypeError(
        "MVP export only supports a Tensor, tuple[Tensor, ...], or list[Tensor] output"
    )
