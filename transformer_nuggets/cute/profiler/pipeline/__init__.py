"""Declarative CuTeDSL pipeline plans, measurements, analysis, and Perfetto export."""

from __future__ import annotations

from importlib import import_module

from transformer_nuggets.cute.profiler.pipeline.annotations import (
    PipelineAnnotations,
    extract_plan,
)
from transformer_nuggets.cute.profiler.pipeline.plan import (
    Dependency,
    PipelinePlan,
    Region,
    Resource,
    Role,
    SourceLocation,
    Timeline,
)

_LAZY_EXPORTS = {
    "MeasuredCapture": ("iket", "MeasuredCapture"),
    "MeasuredRegion": ("iket", "MeasuredRegion"),
    "PipelineAnalysis": ("analysis", "PipelineAnalysis"),
    "analyze_pipeline": ("analysis", "analyze_pipeline"),
    "load_iket_capture": ("iket", "load_iket_capture"),
    "write_pipeline_perfetto": ("perfetto", "write_pipeline_perfetto"),
}


def __getattr__(name: str):
    """Load measurement, analysis, and visualization helpers on first use."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(
        import_module(f"transformer_nuggets.cute.profiler.pipeline.{module_name}"),
        attribute,
    )
    globals()[name] = value
    return value


__all__ = [
    "Dependency",
    "MeasuredCapture",
    "MeasuredRegion",
    "PipelineAnalysis",
    "PipelineAnnotations",
    "PipelinePlan",
    "Region",
    "Resource",
    "Role",
    "SourceLocation",
    "Timeline",
    "analyze_pipeline",
    "extract_plan",
    "load_iket_capture",
    "write_pipeline_perfetto",
]
