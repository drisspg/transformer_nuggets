from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import Path
from typing import Any

import torch

from transformer_nuggets.export_autograd_triton.capture import capture_compiled_autograd_sources
from transformer_nuggets.export_autograd_triton.codegen import (
    generate_autograd_source,
    write_autograd_source,
)
from transformer_nuggets.export_autograd_triton.guards import (
    tensor_guard_for,
    validate_static_value,
)
from transformer_nuggets.export_autograd_triton.specs import (
    CapturedSpecialization,
    ExportedAutogradSource,
    Specialization,
)


def export_autograd_triton(
    fn: Callable[..., Any],
    specializations: list[Specialization],
    out: str | Path,
    exported_name: str | None = None,
    source_backend: str = "clean_triton",
    max_autotune: bool = False,
    inductor_config_patches: dict[str, Any] | None = None,
) -> ExportedAutogradSource:
    signature = _validate_signature(fn)
    if source_backend not in {"inductor", "clean_triton"}:
        raise ValueError("source_backend must be 'inductor' or 'clean_triton'")
    if not specializations:
        raise ValueError("At least one specialization is required")
    config_patches = dict(inductor_config_patches or {})
    if max_autotune:
        config_patches["max_autotune"] = True
    captured_specializations = [
        _capture_specialization(fn, signature, specialization, index, config_patches)
        for index, specialization in enumerate(specializations)
    ]
    output_path = Path(out)
    public_name = exported_name or f"{fn.__name__}_compiled"
    source = generate_autograd_source(fn, public_name, captured_specializations)
    write_autograd_source(output_path, source, captured_specializations, source_backend)
    return ExportedAutogradSource(
        output_path=output_path,
        exported_name=public_name,
        specializations=[specialization.name for specialization in captured_specializations],
        source=source,
    )


def _validate_signature(fn: Callable[..., Any]) -> inspect.Signature:
    signature = inspect.signature(fn)
    unsupported = {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
    for parameter in signature.parameters.values():
        if parameter.kind in unsupported:
            raise TypeError("MVP export does not support *args or **kwargs parameters")
    return signature


def _capture_specialization(
    fn: Callable[..., Any],
    signature: inspect.Signature,
    specialization: Specialization,
    index: int,
    inductor_config_patches: dict[str, Any],
) -> CapturedSpecialization:
    if specialization.additional_inputs:
        raise NotImplementedError(
            "additional_inputs dynamic-shape inference is not implemented yet"
        )

    bound = signature.bind(*specialization.args, **(specialization.kwargs or {}))
    bound.apply_defaults()
    runtime_tensor_names = tuple(
        name for name, value in bound.arguments.items() if isinstance(value, torch.Tensor)
    )
    if not runtime_tensor_names:
        raise ValueError("A specialization must include at least one Tensor argument")

    static_args = tuple(
        validate_static_value(name, value)
        for name, value in bound.arguments.items()
        if not isinstance(value, torch.Tensor)
    )
    tensor_guards = tuple(
        tensor_guard_for(
            name,
            bound.arguments[name],
            _dynamic_shape_for(specialization.dynamic_shapes, name),
        )
        for name in runtime_tensor_names
    )

    def tensor_only_fn(*runtime_tensors: torch.Tensor) -> Any:
        if len(runtime_tensors) != len(runtime_tensor_names):
            raise RuntimeError(
                f"Expected {len(runtime_tensor_names)} tensor inputs, got {len(runtime_tensors)}"
            )
        arguments = dict(bound.arguments)
        arguments.update(zip(runtime_tensor_names, runtime_tensors, strict=True))
        return _call_with_bound_arguments(fn, signature, arguments)

    dynamic = specialization.dynamic_shapes is not None
    compiled_sources = capture_compiled_autograd_sources(
        tensor_only_fn,
        tuple(bound.arguments[name] for name in runtime_tensor_names),
        inductor_config_patches=inductor_config_patches,
        dynamic=dynamic,
        dynamic_shapes=_runtime_dynamic_shapes(
            specialization.dynamic_shapes,
            runtime_tensor_names,
        ),
    )
    return CapturedSpecialization(
        name=specialization.name or f"spec_{index}",
        runtime_tensor_names=runtime_tensor_names,
        static_args=static_args,
        tensor_guards=tensor_guards,
        forward_source=compiled_sources.forward.source_code,
        backward_source=(
            compiled_sources.backward.source_code
            if compiled_sources.backward is not None
            else None
        ),
        num_user_outputs=compiled_sources.num_user_outputs,
        output_kind=compiled_sources.output_kind,
        needs_autograd=compiled_sources.needs_autograd,
        dynamic=dynamic,
        differentiable_output_mask=compiled_sources.differentiable_output_mask,
        forward_residual_names=compiled_sources.forward_residual_names,
        backward_saved_input_names=compiled_sources.backward_saved_input_names,
    )


def _dynamic_shape_for(dynamic_shapes: Any | None, name: str) -> Any | None:
    if dynamic_shapes is None:
        return None
    if not isinstance(dynamic_shapes, dict):
        raise TypeError("dynamic_shapes must be a dict keyed by argument name")
    return dynamic_shapes.get(name)


def _runtime_dynamic_shapes(
    dynamic_shapes: Any | None,
    runtime_tensor_names: tuple[str, ...],
) -> Any | None:
    if dynamic_shapes is None:
        return None
    if not isinstance(dynamic_shapes, dict):
        raise TypeError("dynamic_shapes must be a dict keyed by argument name")
    return tuple(dynamic_shapes.get(name) for name in runtime_tensor_names)


def _call_with_bound_arguments(
    fn: Callable[..., Any],
    signature: inspect.Signature,
    arguments: dict[str, Any],
) -> Any:
    args = []
    kwargs = {}
    for parameter in signature.parameters.values():
        value = arguments[parameter.name]
        match parameter.kind:
            case inspect.Parameter.POSITIONAL_ONLY | inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(value)
            case inspect.Parameter.KEYWORD_ONLY:
                kwargs[parameter.name] = value
            case _:
                raise TypeError("Unsupported parameter kind")
    return fn(*args, **kwargs)
