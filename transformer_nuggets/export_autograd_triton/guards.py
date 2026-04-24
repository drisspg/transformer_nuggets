from __future__ import annotations

import ast
from typing import Any

import torch

from transformer_nuggets.export_autograd_triton.specs import StaticArgSpec, TensorGuardSpec


_SUPPORTED_STATIC_TYPES = (str, int, float, bool, type(None))


def tensor_guard_for(
    name: str,
    tensor: torch.Tensor,
    dynamic_shape: Any | None = None,
) -> TensorGuardSpec:
    return TensorGuardSpec(
        name=name,
        rank=tensor.dim(),
        shape=_shape_guard(tensor, dynamic_shape),
        stride=tuple(int(stride) for stride in tensor.stride()),
        dtype=str(tensor.dtype).removeprefix("torch."),
        device_type=tensor.device.type,
        device_index=tensor.device.index,
    )


def _shape_guard(
    tensor: torch.Tensor,
    dynamic_shape: Any | None,
) -> tuple[int | dict[str, Any], ...]:
    shape: list[int | dict[str, Any]] = [int(dim) for dim in tensor.shape]
    if dynamic_shape is None:
        return tuple(shape)
    if not isinstance(dynamic_shape, dict):
        raise TypeError(
            "MVP dynamic_shapes entries must be dicts mapping dim index to torch.export.Dim"
        )
    for dim, dim_spec in dynamic_shape.items():
        if dim != 0:
            raise NotImplementedError("MVP dynamic_shapes only supports dynamic dim 0")
        shape[dim] = _dynamic_dim_guard(dim_spec)
    return tuple(shape)


def _dynamic_dim_guard(dim_spec: Any) -> dict[str, Any]:
    if isinstance(dim_spec, str):
        return {"symbol": dim_spec, "min": None, "max": None}
    name = getattr(dim_spec, "__name__", None)
    if name is None:
        raise TypeError("dynamic_shapes dim specs must be torch.export.Dim or str")
    return {
        "symbol": name,
        "min": _dim_bound(getattr(dim_spec, "min", None)),
        "max": _dim_bound(getattr(dim_spec, "max", None)),
    }


def _dim_bound(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def validate_static_value(name: str, value: Any) -> StaticArgSpec:
    if isinstance(value, _SUPPORTED_STATIC_TYPES):
        return StaticArgSpec(name=name, value=value)
    if isinstance(value, tuple):
        for item in value:
            validate_static_value(name, item)
        ast.literal_eval(repr(value))
        return StaticArgSpec(name=name, value=value)
    raise TypeError(
        f"Static argument {name!r} has unsupported value {value!r}. "
        "MVP static arguments must be str, int, float, bool, None, or tuples of those."
    )


def format_tensor_guard(guard: TensorGuardSpec) -> str:
    return (
        f"{guard.name}: dtype=torch.{guard.dtype}, device={guard.device_type}:{guard.device_index}, "
        f"shape={guard.shape}, stride={guard.stride}"
    )
