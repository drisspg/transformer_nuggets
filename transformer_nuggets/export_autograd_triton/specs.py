from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Specialization:
    args: tuple[Any, ...]
    kwargs: dict[str, Any] | None = None
    additional_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]] | None = None
    dynamic_shapes: Any | None = None
    name: str | None = None


@dataclass
class ExportedAutogradSource:
    output_path: Path
    exported_name: str
    specializations: list[str]
    source: str


@dataclass
class TensorGuardSpec:
    name: str
    rank: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device_type: str
    device_index: int | None


@dataclass
class StaticArgSpec:
    name: str
    value: Any


@dataclass
class CapturedSpecialization:
    name: str
    runtime_tensor_names: tuple[str, ...]
    static_args: tuple[StaticArgSpec, ...]
    tensor_guards: tuple[TensorGuardSpec, ...]
    forward_source: str
    backward_source: str | None
    num_user_outputs: int
    output_kind: str
    needs_autograd: bool
