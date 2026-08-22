"""Shared logical roofline formulas and raw PyTorch trace enrichment.

Formulas are pure shape/dtype models. They can be registered by ATen/custom-op
target or profiler event name and are consumed by both FX analysis and trace-only
fallbacks. Physical traffic and executed instructions remain profiler-counter
measurements rather than registered logical formulas.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import re
from typing import Annotated, Any, Literal

import torch
import typer


__all__ = [
    "RooflineFormulaContext",
    "RooflineSpec",
    "RooflineTensor",
    "RooflineWork",
    "decorate_trace_file",
    "decorate_trace_roofline",
    "get_kernel_roofline_formula",
    "get_roofline_formula",
    "register_kernel_roofline_formula",
    "register_roofline_formula",
]


app = typer.Typer(help="Annotate a PyTorch profiler trace with logical roofline metadata.")

_TRACE_DTYPES = {
    "bool": torch.bool,
    "byte": torch.uint8,
    "char": torch.int8,
    "short": torch.int16,
    "int": torch.int32,
    "long": torch.int64,
    "half": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}
_SEVERITY_STYLE = {
    "high": ("🔴", "terrible"),
    "medium": ("🟡", "yellow"),
    "healthy": ("🟢", "good"),
    "needs_measurement": ("🔵", "cq_build_running"),
    "unknown": ("⚪", "grey"),
}


Confidence = Literal["high", "medium", "low", "unknown"]


@dataclass(frozen=True)
class RooflineSpec:
    """Explicit hardware ceilings used for logical roofline calculations."""

    name: str
    peak_compute_tflops: float
    peak_memory_gbps: float
    launch_latency_us: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_compute_tflops <= 0 or self.peak_memory_gbps <= 0:
            raise ValueError("Roofline compute and memory ceilings must be positive")
        if self.launch_latency_us < 0:
            raise ValueError("Roofline launch latency must be non-negative")


@dataclass(frozen=True)
class RooflineTensor:
    """Shape-only tensor metadata available to logical work formulas."""

    shape: tuple[int, ...]
    dtype: torch.dtype | None
    stride: tuple[int, ...] | None = None
    requires_grad: bool = False

    @property
    def numel(self) -> int:
        """Return the logical number of elements."""
        return math.prod(self.shape)

    @property
    def element_size(self) -> int | None:
        """Return bytes per element when the dtype is known."""
        return None if self.dtype is None else self.dtype.itemsize

    @property
    def nbytes(self) -> int | None:
        """Return logical tensor bytes when the dtype is known."""
        size = self.element_size
        return None if size is None else self.numel * size


@dataclass(frozen=True)
class RooflineFormulaContext:
    """Normalized operation metadata passed to a registered formula."""

    op_name: str
    inputs: tuple[RooflineTensor, ...]
    outputs: tuple[RooflineTensor, ...]
    concrete_inputs: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RooflineWork:
    """Logical useful work and minimal traffic for one operation."""

    logical_flops: int | None
    read_bytes: int | None
    write_bytes: int | None
    model_kind: str
    confidence: Confidence = "high"

    @property
    def logical_bytes(self) -> int | None:
        """Return total logical traffic when both directions are known."""
        if self.read_bytes is None or self.write_bytes is None:
            return None
        return self.read_bytes + self.write_bytes


Formula = Callable[
    [tuple[RooflineTensor, ...], tuple[RooflineTensor, ...], Mapping[str, Any]],
    RooflineWork,
]
_FORMULAS: dict[str, Formula] = {}
_TRACE_INCOMPATIBLE_FORMULAS: set[str] = set()


@dataclass(frozen=True)
class _KernelFormulaRegistration:
    op_names: tuple[str, ...]
    kernel_pattern: re.Pattern[str]
    formula: Formula


_KERNEL_FORMULAS: list[_KernelFormulaRegistration] = []


def _canonical_names(target: Any) -> tuple[str, ...]:
    names = []
    if isinstance(target, torch._ops.OpOverload):
        names.extend((str(target), target._schema.name))
    elif isinstance(target, torch._ops.OpOverloadPacket):
        names.append(str(target).replace(".", "::", 1))
        for overload in target.overloads():
            op = getattr(target, overload)
            names.extend((str(op), op._schema.name))
    else:
        name = str(target)
        names.append(name)
        if name.startswith("aten."):
            parts = name.split(".")
            names.append(f"aten::{parts[1]}")
    return tuple(dict.fromkeys(names))


def register_roofline_formula(
    target: Any,
    *,
    replace: bool = False,
    trace_compatible: bool = True,
) -> Callable[[Formula], Formula]:
    """Register a shape-only logical work formula for an op or trace event name."""

    def register(formula: Formula) -> Formula:
        for name in _canonical_names(target):
            existing = _FORMULAS.get(name)
            if existing is not None and existing is not formula and not replace:
                raise ValueError(f"Roofline formula already registered for {name!r}")
            _FORMULAS[name] = formula
            if trace_compatible:
                _TRACE_INCOMPATIBLE_FORMULAS.discard(name)
            else:
                _TRACE_INCOMPATIBLE_FORMULAS.add(name)
        return formula

    return register


def get_roofline_formula(target: Any, *, for_trace: bool = False) -> Formula | None:
    """Return the first compatible formula matching an op or event name."""
    return next(
        (
            _FORMULAS[name]
            for name in _canonical_names(target)
            if name in _FORMULAS and (not for_trace or name not in _TRACE_INCOMPATIBLE_FORMULAS)
        ),
        None,
    )


def register_kernel_roofline_formula(
    target: Any,
    *,
    kernel_name: str,
) -> Callable[[Formula], Formula]:
    """Register a formula for kernels matching a parent op and name regex."""

    def register(formula: Formula) -> Formula:
        _KERNEL_FORMULAS.append(
            _KernelFormulaRegistration(
                op_names=_canonical_names(target),
                kernel_pattern=re.compile(kernel_name),
                formula=formula,
            )
        )
        return formula

    return register


def get_kernel_roofline_formula(target: Any, kernel_name: str) -> Formula | None:
    """Return the most recently registered matching inner-kernel formula."""
    target_names = set(_canonical_names(target))
    return next(
        (
            registration.formula
            for registration in reversed(_KERNEL_FORMULAS)
            if target_names.intersection(registration.op_names)
            and registration.kernel_pattern.search(kernel_name)
        ),
        None,
    )


def _evaluate_formula(
    formula: Formula,
    inputs: tuple[RooflineTensor, ...],
    outputs: tuple[RooflineTensor, ...],
    kwargs: Mapping[str, Any],
) -> tuple[RooflineWork | None, str | None]:
    """Evaluate one extension formula without allowing it to abort decoration."""
    try:
        work = formula(inputs, outputs, kwargs)
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(work, RooflineWork):
        return None, f"expected RooflineWork, got {type(work).__name__}"
    values = (work.logical_flops, work.read_bytes, work.write_bytes)
    if any(value is not None and value < 0 for value in values):
        return None, "formula returned negative work"
    return work, None


def _dtype_from_trace_name(name: Any) -> torch.dtype | None:
    normalized = str(name).replace("c10::", "").replace("ScalarType::", "").lower()
    return _TRACE_DTYPES.get(normalized)


def _trace_tensors(args: Mapping[str, Any]) -> tuple[RooflineTensor, ...]:
    dims = args.get("Input Dims", ())
    dtypes = args.get("Input type", ())
    strides = args.get("Input Strides", ())
    tensors = []

    def append_shapes(shape: Any, stride: Any, dtype: torch.dtype | None) -> None:
        if not isinstance(shape, Sequence) or isinstance(shape, str):
            return
        if not shape:
            if dtype is not None:
                tensors.append(RooflineTensor(shape=(), dtype=dtype, stride=()))
            return
        if all(not isinstance(dim, Sequence) or isinstance(dim, str) for dim in shape):
            try:
                parsed_shape = tuple(int(dim) for dim in shape)
            except (TypeError, ValueError):
                return
            if any(dim < 0 for dim in parsed_shape):
                return
            parsed_stride = None
            if (
                isinstance(stride, Sequence)
                and not isinstance(stride, str)
                and all(
                    not isinstance(value, Sequence) or isinstance(value, str) for value in stride
                )
            ):
                try:
                    parsed_stride = tuple(int(value) for value in stride)
                except (TypeError, ValueError):
                    parsed_stride = None
            tensors.append(
                RooflineTensor(
                    shape=parsed_shape,
                    dtype=dtype,
                    stride=parsed_stride,
                )
            )
            return
        for index, child_shape in enumerate(shape):
            child_stride = (
                stride[index]
                if isinstance(stride, Sequence)
                and not isinstance(stride, str)
                and index < len(stride)
                else None
            )
            append_shapes(child_shape, child_stride, dtype)

    dims = dims if isinstance(dims, Sequence) and not isinstance(dims, str) else ()
    dtypes = dtypes if isinstance(dtypes, Sequence) and not isinstance(dtypes, str) else ()
    strides = strides if isinstance(strides, Sequence) and not isinstance(strides, str) else ()
    for index, shape in enumerate(dims):
        dtype = _dtype_from_trace_name(dtypes[index] if index < len(dtypes) else "")
        stride = strides[index] if index < len(strides) else None
        append_shapes(shape, stride, dtype)
    return tuple(tensors)


def _generic_trace_work(context: RooflineFormulaContext) -> RooflineWork:
    tensor_bytes = [tensor.nbytes for tensor in context.inputs]
    read_bytes = None if any(value is None for value in tensor_bytes) else sum(tensor_bytes)
    return RooflineWork(
        logical_flops=None,
        read_bytes=read_bytes,
        write_bytes=None,
        model_kind="generic_trace_io",
        confidence="unknown",
    )


def _sum_tensor_bytes(tensors: Sequence[RooflineTensor]) -> int | None:
    """Sum tensor bytes while preserving unknown dtype information."""
    values = [tensor.nbytes for tensor in tensors]
    return None if any(value is None for value in values) else sum(values)


@register_roofline_formula("aten::mm")
def _aten_mm_work(
    inputs: tuple[RooflineTensor, ...],
    outputs: tuple[RooflineTensor, ...],
    kwargs: Mapping[str, Any],
) -> RooflineWork:
    """Model ordinary dense matrix multiplication."""
    del outputs, kwargs
    left, right = inputs[-2:]
    m, k = left.shape
    n = right.shape[-1]
    return RooflineWork(
        logical_flops=2 * m * n * k,
        read_bytes=_sum_tensor_bytes((left, right)),
        write_bytes=(None if left.element_size is None else left.element_size * m * n),
        model_kind="gemm",
        confidence="high",
    )


@register_roofline_formula("aten::mul")
def _aten_mul_work(
    inputs: tuple[RooflineTensor, ...],
    outputs: tuple[RooflineTensor, ...],
    kwargs: Mapping[str, Any],
) -> RooflineWork:
    """Model broadcast multiplication."""
    del outputs, kwargs
    output_shape = torch.broadcast_shapes(*(tensor.shape for tensor in inputs))
    output_elements = math.prod(output_shape)
    return RooflineWork(
        logical_flops=output_elements,
        read_bytes=_sum_tensor_bytes(inputs),
        write_bytes=(
            None if inputs[0].element_size is None else output_elements * inputs[0].element_size
        ),
        model_kind="pointwise_mul",
        confidence="high",
    )


@register_roofline_formula("aten::sigmoid")
def _aten_sigmoid_work(
    inputs: tuple[RooflineTensor, ...],
    outputs: tuple[RooflineTensor, ...],
    kwargs: Mapping[str, Any],
) -> RooflineWork:
    """Count one semantic sigmoid per element without weighting the transcendental."""
    del outputs, kwargs
    x = inputs[0]
    return RooflineWork(
        logical_flops=x.numel,
        read_bytes=x.nbytes,
        write_bytes=x.nbytes,
        model_kind="sigmoid",
        confidence="medium",
    )


@register_roofline_formula("aten::copy_")
def _aten_copy_work(
    inputs: tuple[RooflineTensor, ...],
    outputs: tuple[RooflineTensor, ...],
    kwargs: Mapping[str, Any],
) -> RooflineWork:
    """Model source reads and destination writes for copy_."""
    del outputs, kwargs
    destination, source = inputs[:2]
    return RooflineWork(
        logical_flops=0,
        read_bytes=source.nbytes,
        write_bytes=destination.nbytes,
        model_kind="copy",
        confidence="high",
    )


def _rate(value: int | None, duration_us: float, scale: float) -> float | None:
    """Return a per-second logical rate scaled to the requested unit."""
    return None if value is None else value / max(duration_us, 1e-12) / scale


def _runtime_metrics(
    work: RooflineWork,
    duration_us: float,
    roofline_spec: RooflineSpec | None,
    *,
    launch_count: int = 1,
) -> dict[str, Any]:
    logical_bytes = work.logical_bytes
    metrics: dict[str, Any] = {
        "logical_flops": work.logical_flops,
        "logical_read_bytes": work.read_bytes,
        "logical_write_bytes": work.write_bytes,
        "logical_bytes": logical_bytes,
        "model_kind": work.model_kind,
        "model_confidence": work.confidence,
        "duration_us": duration_us,
        "arithmetic_intensity_flops_per_byte": (
            None
            if work.logical_flops is None or logical_bytes is None
            else work.logical_flops / max(logical_bytes, 1)
        ),
        "achieved_logical_tflops": _rate(work.logical_flops, duration_us, 1e6),
        "achieved_logical_read_gbps": _rate(work.read_bytes, duration_us, 1e3),
        "achieved_logical_read_tbps": _rate(work.read_bytes, duration_us, 1e6),
        "achieved_logical_write_gbps": _rate(work.write_bytes, duration_us, 1e3),
        "achieved_logical_write_tbps": _rate(work.write_bytes, duration_us, 1e6),
        "achieved_logical_gbps": _rate(logical_bytes, duration_us, 1e3),
        "achieved_logical_tbps": _rate(logical_bytes, duration_us, 1e6),
    }
    if roofline_spec is None or logical_bytes is None:
        return metrics

    memory_floor_us = logical_bytes / (roofline_spec.peak_memory_gbps * 1e3)
    compute_floor_us = (
        None
        if work.logical_flops is None
        else work.logical_flops / (roofline_spec.peak_compute_tflops * 1e6)
    )
    roofline_floor_us = (
        memory_floor_us if compute_floor_us is None else max(memory_floor_us, compute_floor_us)
    )
    launch_floor_us = launch_count * roofline_spec.launch_latency_us
    effective_floor_us = max(roofline_floor_us, launch_floor_us)
    metrics.update(
        {
            "roofline_spec": roofline_spec.name,
            "memory_floor_us": memory_floor_us,
            "compute_floor_us": compute_floor_us,
            "roofline_floor_us": roofline_floor_us,
            "launch_floor_us": launch_floor_us,
            "effective_floor_us": effective_floor_us,
            "predicted_bound": (
                "launch"
                if launch_floor_us >= roofline_floor_us and launch_floor_us > 0
                else "unknown_compute"
                if compute_floor_us is None
                else "compute"
                if compute_floor_us >= memory_floor_us
                else "memory"
            ),
            "achieved_memory_roofline_percent": 100 * memory_floor_us / max(duration_us, 1e-12),
            "achieved_roofline_percent": 100 * effective_floor_us / max(duration_us, 1e-12),
        }
    )
    return metrics


_ROOFLINE_ARG_PREFIXES = (
    "trace_roofline",
    "transformer_nuggets.trace_roofline",
    "stage_",
    "logical_",
    "achieved_",
    "arithmetic_intensity_",
    "model_kind",
    "model_confidence",
    "memory_floor_us",
    "compute_floor_us",
    "roofline_floor_us",
    "launch_floor_us",
    "effective_floor_us",
    "predicted_bound",
    "observed_",
    "inner_kernel_",
    "formula_error",
)


def _clear_roofline_args(args: dict[str, Any]) -> None:
    """Remove prior enrichment fields before idempotent regeneration."""
    for key in list(args):
        if key.startswith(_ROOFLINE_ARG_PREFIXES):
            args.pop(key)


def _trace_identity(value: Any) -> int | str | None:
    """Return a hashable Kineto identity or None for malformed values."""
    return value if isinstance(value, int | str) and not isinstance(value, bool) else None


def _safe_float(value: Any) -> float | None:
    """Return a finite float or None for malformed trace values."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _trace_interval(event: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return a finite non-negative duration interval."""
    start = _safe_float(event.get("ts", 0))
    duration = _safe_float(event.get("dur", 0))
    if start is None or duration is None or duration < 0:
        return None
    return start, start + duration


def _interval_union_duration(intervals: Sequence[tuple[float, float]]) -> float:
    """Return wall time covered by the union of intervals."""
    if not intervals:
        return 0.0
    total = 0.0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def decorate_trace_roofline(
    trace: Mapping[str, Any],
    *,
    roofline_spec: RooflineSpec | None = None,
    slow_kernel_threshold_us: float = 50.0,
    slow_roofline_percent: float = 60.0,
) -> dict[str, Any]:
    """Annotate raw profiler kernels using matching CPU-op metadata and formulas.

    Kernels are joined to CPU operators through Kineto's ``External id``. Every
    matched kernel receives at least generic input-byte metadata. Registered
    formulas add output bytes and FLOPs, enabling achieved roofline metrics.
    """
    events = []
    cpu_ops: dict[int | str, dict[str, Any] | None] = {}
    for raw_event in trace.get("traceEvents", ()):
        if isinstance(raw_event, Mapping) and raw_event.get("cat") == "gpu_roofline_annotation":
            continue
        if not isinstance(raw_event, Mapping):
            events.append(raw_event)
            continue
        event = dict(raw_event)
        event_args = event.get("args")
        event["args"] = dict(event_args) if isinstance(event_args, Mapping) else {}
        _clear_roofline_args(event["args"])
        events.append(event)
        external_id = _trace_identity(event["args"].get("External id"))
        if event.get("cat") == "cpu_op" and external_id is not None:
            cpu_ops[external_id] = event if external_id not in cpu_ops else None

    gpu_events_by_external: dict[Any, list[dict[str, Any]]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("cat") not in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        }:
            continue
        external_id = _trace_identity(event["args"].get("External id"))
        if external_id is not None:
            gpu_events_by_external.setdefault(external_id, []).append(event)

    op_metrics: dict[Any, dict[str, Any]] = {}
    for external_id, gpu_events in gpu_events_by_external.items():
        gpu_events = [event for event in gpu_events if _trace_interval(event) is not None]
        if not gpu_events:
            continue
        cpu_event = cpu_ops.get(external_id)
        if cpu_event is None:
            for event in gpu_events:
                event["args"]["trace_roofline_attribution_error"] = "ambiguous_external_id"
            continue
        cpu_args = cpu_event.get("args", {})
        context = RooflineFormulaContext(
            op_name=str(cpu_event.get("name", "unknown")),
            inputs=_trace_tensors(cpu_args),
            outputs=(),
            concrete_inputs=tuple(cpu_args.get("Concrete Inputs", ())),
        )
        formula = get_roofline_formula(context.op_name, for_trace=True)
        formula_error = None
        if formula is None:
            work = _generic_trace_work(context)
        else:
            work, formula_error = _evaluate_formula(
                formula,
                context.inputs,
                context.outputs,
                {"_concrete_inputs": context.concrete_inputs},
            )
            if work is None:
                work = _generic_trace_work(context)
        intervals = [_trace_interval(event) for event in gpu_events]
        intervals = [interval for interval in intervals if interval is not None]
        wall_us = max(end for _, end in intervals) - min(start for start, _ in intervals)
        busy_us = _interval_union_duration(intervals)
        gpu_time_us = sum(end - start for start, end in intervals)
        sorted_gpu_events = sorted(
            gpu_events,
            key=lambda item: float(item.get("ts", 0) or 0),
        )
        kernel_works: list[RooflineWork | None] = []
        for event in sorted_gpu_events:
            kernel_formula = get_kernel_roofline_formula(
                context.op_name,
                str(event.get("name", "")),
            )
            if kernel_formula is None:
                kernel_works.append(None)
                continue
            kernel_work, _ = _evaluate_formula(
                kernel_formula,
                context.inputs,
                context.outputs,
                {
                    "_concrete_inputs": context.concrete_inputs,
                    "kernel_name": str(event.get("name", "")),
                    "grid": event.get("args", {}).get("grid"),
                    "block": event.get("args", {}).get("block"),
                    "parent_op": context.op_name,
                },
            )
            kernel_works.append(kernel_work)
        known_kernel_flops = [
            kernel_work.logical_flops
            for kernel_work in kernel_works
            if kernel_work is not None and kernel_work.logical_flops is not None
        ]
        known_kernel_bytes = [
            kernel_work.logical_bytes
            for kernel_work in kernel_works
            if kernel_work is not None and kernel_work.logical_bytes is not None
        ]
        metrics = {
            "trace_roofline_cpu_op": context.op_name,
            "observed_kernel_count": sum(event.get("cat") == "kernel" for event in gpu_events),
            "observed_gpu_launch_count": len(gpu_events),
            "observed_op_wall_us": wall_us,
            "observed_op_gpu_busy_us": busy_us,
            "observed_op_gpu_time_us": gpu_time_us,
            "observed_op_gap_us": max(wall_us - busy_us, 0.0),
            "inner_kernel_formula_count": sum(
                kernel_work is not None for kernel_work in kernel_works
            ),
            "inner_kernel_known_flops": sum(known_kernel_flops),
            "inner_kernel_known_bytes": sum(known_kernel_bytes),
            "inner_kernel_flop_coverage_ratio": (
                None
                if work.logical_flops in {None, 0}
                else sum(known_kernel_flops) / work.logical_flops
            ),
            "inner_kernel_byte_coverage_ratio": (
                None
                if work.logical_bytes in {None, 0}
                else sum(known_kernel_bytes) / work.logical_bytes
            ),
            "formula_error": formula_error,
            **_runtime_metrics(
                work,
                wall_us,
                roofline_spec,
                launch_count=len(gpu_events),
            ),
        }
        op_metrics[external_id] = metrics
        cpu_event["args"].update({"transformer_nuggets.trace_roofline_op": True, **metrics})
        kernel_index = 0
        for gpu_op_index, (event, kernel_work) in enumerate(
            zip(sorted_gpu_events, kernel_works),
            start=1,
        ):
            is_kernel = event.get("cat") == "kernel"
            if is_kernel:
                kernel_index += 1
            duration_us = float(event.get("dur", 0) or 0)
            if kernel_work is not None:
                work_scope = "kernel_specific"
                event_metrics = {
                    **_runtime_metrics(kernel_work, duration_us, roofline_spec, launch_count=1),
                    "trace_roofline_parent_op_logical_flops": metrics["logical_flops"],
                    "trace_roofline_parent_op_logical_bytes": metrics["logical_bytes"],
                    "trace_roofline_parent_op_wall_us": wall_us,
                }
            elif len(gpu_events) == 1:
                work_scope = "single_kernel_parent_op"
                event_metrics = metrics
            else:
                work_scope = "parent_op_shared"
                event_metrics = {
                    "trace_roofline_parent_op_logical_flops": metrics["logical_flops"],
                    "trace_roofline_parent_op_logical_bytes": metrics["logical_bytes"],
                    "trace_roofline_parent_op_wall_us": wall_us,
                    "trace_roofline_parent_op_model_kind": metrics["model_kind"],
                    "trace_roofline_parent_op_model_confidence": metrics["model_confidence"],
                    "trace_roofline_parent_op_achieved_tflops": metrics["achieved_logical_tflops"],
                    "trace_roofline_parent_op_achieved_tbps": metrics["achieved_logical_tbps"],
                    "trace_roofline_parent_op_roofline_percent": metrics.get(
                        "achieved_roofline_percent"
                    ),
                }
            event["args"].update(
                {
                    "transformer_nuggets.trace_roofline": True,
                    "trace_roofline_work_scope": work_scope,
                    "trace_roofline_cpu_op": context.op_name,
                    "trace_roofline_gpu_op_index": gpu_op_index,
                    "trace_roofline_kernel_index": kernel_index if is_kernel else 0,
                    "trace_roofline_kernel_duration_us": duration_us,
                    **event_metrics,
                }
            )

    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "user_annotation":
            continue
        stage_interval = _trace_interval(event)
        if stage_interval is None:
            continue
        start, end = stage_interval
        pid, tid = event.get("pid"), event.get("tid")

        def contained_cpu_ops(*, same_thread: bool) -> set[int | str]:
            result = set()
            for external_id, cpu_event in cpu_ops.items():
                if cpu_event is None or external_id not in op_metrics:
                    continue
                cpu_interval = _trace_interval(cpu_event)
                if cpu_interval is None or cpu_event.get("pid") != pid:
                    continue
                if same_thread and cpu_event.get("tid") != tid:
                    continue
                cpu_start, cpu_end = cpu_interval
                if start <= cpu_start and cpu_end <= end:
                    result.add(external_id)
            return result

        if "backward" in str(event.get("name", "")).lower():
            child_ids = contained_cpu_ops(same_thread=False)
            attribution_scope = "cross_thread_backward_inclusive"
        else:
            child_ids = contained_cpu_ops(same_thread=True)
            attribution_scope = "same_thread_inclusive"
        if not child_ids:
            continue
        child_metrics = [op_metrics[external_id] for external_id in child_ids]
        stage_gpu_events = [
            gpu_event
            for external_id in child_ids
            for gpu_event in gpu_events_by_external[external_id]
        ]
        stage_intervals = [
            interval
            for gpu_event in stage_gpu_events
            if (interval := _trace_interval(gpu_event)) is not None
        ]
        if not stage_intervals:
            continue
        gpu_wall_us = max(end for _, end in stage_intervals) - min(
            start for start, _ in stage_intervals
        )
        gpu_busy_us = _interval_union_duration(stage_intervals)
        gpu_time_us = sum(end - start for start, end in stage_intervals)
        trusted_metrics = [
            metric for metric in child_metrics if metric["model_confidence"] in {"high", "medium"}
        ]
        known_flops = [
            metric["logical_flops"]
            for metric in trusted_metrics
            if metric["logical_flops"] is not None
        ]
        known_bytes = [
            metric["logical_bytes"]
            for metric in trusted_metrics
            if metric["logical_bytes"] is not None
        ]
        event["args"].update(
            {
                "transformer_nuggets.trace_roofline_stage": True,
                "stage_operation_count": len(child_ids),
                "stage_kernel_count": sum(
                    metric["observed_kernel_count"] for metric in child_metrics
                ),
                "stage_gpu_launch_count": sum(
                    metric["observed_gpu_launch_count"] for metric in child_metrics
                ),
                "stage_attribution_scope": attribution_scope,
                "stage_aggregation": "inclusive",
                "stage_gpu_wall_us": gpu_wall_us,
                "stage_gpu_busy_us": gpu_busy_us,
                "stage_gpu_time_us": gpu_time_us,
                "stage_known_logical_flops": sum(known_flops),
                "stage_low_confidence_operation_count": len(child_metrics) - len(trusted_metrics),
                "stage_unknown_flop_operation_count": len(child_metrics) - len(known_flops),
                "stage_known_logical_bytes": sum(known_bytes),
                "stage_unknown_byte_operation_count": len(child_metrics) - len(known_bytes),
                "stage_achieved_known_tflops": sum(known_flops) / max(gpu_wall_us, 1e-12) / 1e6,
                "stage_achieved_known_gbps": sum(known_bytes) / max(gpu_wall_us, 1e-12) / 1e3,
                "stage_achieved_known_tbps": sum(known_bytes) / max(gpu_wall_us, 1e-12) / 1e6,
            }
        )

    synthetic_events = []
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "kernel":
            continue
        args = event.get("args", {})
        if not args.get("transformer_nuggets.trace_roofline"):
            continue
        duration_us = float(event.get("dur", 0) or 0)
        tflops = args.get("achieved_logical_tflops")
        tbps = args.get("achieved_logical_tbps")
        roofline = args.get("achieved_roofline_percent")
        memory_roofline = args.get("achieved_memory_roofline_percent")
        displayed_roofline = roofline if roofline is not None else memory_roofline
        tflops_label = "? TF/s" if tflops is None else f"{float(tflops):.1f} TF/s"
        tbps_label = "? TB/s" if tbps is None else f"{float(tbps):.2f} TB/s"
        roofline_label = (
            "? roofline"
            if displayed_roofline is None
            else f"{float(displayed_roofline):.0f}% roofline"
        )
        confidence = args.get("model_confidence", "unknown")
        is_slow = (
            confidence in {"high", "medium"}
            and displayed_roofline is not None
            and duration_us >= slow_kernel_threshold_us
            and float(displayed_roofline) < slow_roofline_percent
        )
        needs_measurement = confidence == "low" and duration_us >= slow_kernel_threshold_us
        if is_slow:
            severity = (
                "high" if duration_us >= 100 and float(displayed_roofline) < 30 else "medium"
            )
        elif needs_measurement:
            severity = "needs_measurement"
        elif displayed_roofline is None:
            severity = "unknown"
        else:
            severity = "healthy"
        icon, cname = _SEVERITY_STYLE[severity]
        model_label = args.get("model_kind", args.get("trace_roofline_cpu_op", "unknown"))
        synthetic_events.append(
            {
                "ph": "X",
                "cat": "gpu_roofline_annotation",
                "cname": cname,
                "name": (
                    f"{icon} {duration_us:.0f} us · {model_label} · "
                    f"{tflops_label} · {tbps_label} · {roofline_label}"
                ),
                "pid": event.get("pid", 0),
                "tid": event.get("tid", 0),
                "ts": event.get("ts", 0),
                "dur": duration_us,
                "args": {
                    **args,
                    "transformer_nuggets.roofline_annotation": True,
                    "is_optimization_finding": is_slow,
                    "needs_measurement": needs_measurement,
                    "severity": severity,
                    "kernel_name": event.get("name", ""),
                    "slow_kernel_threshold_us": slow_kernel_threshold_us,
                    "slow_roofline_percent": slow_roofline_percent,
                },
            }
        )

    output = dict(trace)
    output["traceEvents"] = [*events, *synthetic_events]
    return output


def decorate_trace_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    roofline_spec: RooflineSpec | None = None,
    slow_kernel_threshold_us: float = 50.0,
    slow_roofline_percent: float = 60.0,
) -> Path:
    """Read, enrich, and write a Chrome/Kineto JSON trace."""
    from transformer_nuggets.utils.perfetto import (
        read_trace,
        write_perfetto_trace,
        write_trace,
    )

    input_path = Path(input_path)
    if input_path.suffix not in {".json", ".gz"}:
        raise ValueError("Trace-only roofline decoration currently requires JSON or JSON.GZ input")
    if output_path is None:
        name = input_path.name
        if name.endswith(".json.gz"):
            name = name[: -len(".json.gz")]
        elif name.endswith(".json"):
            name = name[: -len(".json")]
        output_path = input_path.with_name(f"{name}.roofline.json.gz")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace = decorate_trace_roofline(
        read_trace(input_path),
        roofline_spec=roofline_spec,
        slow_kernel_threshold_us=slow_kernel_threshold_us,
        slow_roofline_percent=slow_roofline_percent,
    )
    if output_path.suffix in {".pftrace", ".perfetto-trace"}:
        write_perfetto_trace(output_path, trace, trace_format="track_event")
    else:
        write_trace(output_path, trace)
    return output_path


@app.command()
def annotate_trace(
    input_path: Annotated[Path, typer.Argument(help="Input Kineto JSON or JSON.GZ trace")],
    output_path: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output JSON, JSON.GZ, or PFTRACE"),
    ] = None,
    peak_compute_tflops: Annotated[
        float | None,
        typer.Option(help="Optional logical compute ceiling in TFLOP/s"),
    ] = None,
    peak_memory_gbps: Annotated[
        float | None,
        typer.Option(help="Optional memory ceiling in GB/s"),
    ] = None,
    roofline_name: Annotated[
        str,
        typer.Option(help="Label stored with supplied hardware ceilings"),
    ] = "user-supplied",
    launch_latency_us: Annotated[
        float,
        typer.Option(min=0.0, help="Optional per-launch floor in microseconds"),
    ] = 0.0,
    formula_module: Annotated[
        list[str] | None,
        typer.Option(
            "--formula-module", help="Import a module that registers formulas; repeatable"
        ),
    ] = None,
    slow_kernel_threshold_us: Annotated[
        float,
        typer.Option(min=0.0, help="Minimum duration for red/yellow roofline coloring"),
    ] = 50.0,
    slow_roofline_percent: Annotated[
        float,
        typer.Option(min=0.0, help="Maximum roofline efficiency flagged as slow"),
    ] = 60.0,
) -> None:
    """Annotate one raw PyTorch profiler trace."""
    for module_name in formula_module or ():
        importlib.import_module(module_name)
    if (peak_compute_tflops is None) != (peak_memory_gbps is None):
        raise typer.BadParameter(
            "--peak-compute-tflops and --peak-memory-gbps must be supplied together"
        )
    spec = (
        None
        if peak_compute_tflops is None
        else RooflineSpec(
            name=roofline_name,
            peak_compute_tflops=peak_compute_tflops,
            peak_memory_gbps=peak_memory_gbps,
            launch_latency_us=launch_latency_us,
        )
    )
    path = decorate_trace_file(
        input_path,
        output_path,
        roofline_spec=spec,
        slow_kernel_threshold_us=slow_kernel_threshold_us,
        slow_roofline_percent=slow_roofline_percent,
    )
    typer.echo(f"Wrote {path}")
