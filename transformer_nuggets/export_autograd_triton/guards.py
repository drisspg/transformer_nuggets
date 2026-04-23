from __future__ import annotations

import ast
from typing import Any

import torch

from transformer_nuggets.export_autograd_triton.specs import StaticArgSpec, TensorGuardSpec


_SUPPORTED_STATIC_TYPES = (str, int, float, bool, type(None))


def tensor_guard_for(name: str, tensor: torch.Tensor) -> TensorGuardSpec:
    return TensorGuardSpec(
        name=name,
        rank=tensor.dim(),
        shape=tuple(int(dim) for dim in tensor.shape),
        stride=tuple(int(stride) for stride in tensor.stride()),
        dtype=str(tensor.dtype).removeprefix("torch."),
        device_type=tensor.device.type,
        device_index=tensor.device.index,
    )


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
