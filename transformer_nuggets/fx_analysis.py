"""Analyze eager FX forward/backward graphs for fusion opportunities and annotate traces.

The analyzer uses PyTorch's generic FX primitives rather than Inductor. It captures
an ATen graph with ``make_fx``, identifies connected pointwise/reduction regions, estimates
the minimum traffic eliminated by fusion, and profiles the graph eagerly with one
``record_function`` scope per FX node. A postprocessor then combines those node
annotations into region-level boxes on the GPU timeline.

The partition and effective-byte model follow Elias Ellison's
``https://github.com/eellison/better-benchmark`` at commit
``91bd49f012b50205a352b6ae4ba97525c04c2e55``. The repository owner explicitly
approved reuse; this implementation preserves that provenance.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
import traceback
from typing import Any

import torch
from torch.fx import GraphModule, Interpreter, Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.module_tracker import ModuleTracker

from transformer_nuggets.roofline import (
    RooflineFormulaContext,
    RooflineSpec,
    RooflineTensor,
    RooflineWork,
    get_roofline_formula,
    register_roofline_formula,
)
from transformer_nuggets.utils.perfetto import (
    TraceFormat,
    add_cuda_graph_annotation_boxes,
    perfetto_trace_path,
    read_trace,
    write_perfetto_trace,
)


__all__ = [
    "FusionRegion",
    "RooflineSpec",
    "RooflineWork",
    "register_roofline_formula",
    "FxFusionAnalysis",
    "FxNodeRoofline",
    "FxFusionInterpreter",
    "FxFusionProfile",
    "FxTrainingAnalysis",
    "FxTrainingProfile",
    "add_fx_fusion_region_boxes",
    "analyze_aot_training",
    "analyze_fx_fusion",
    "analyze_fx_graph",
    "profile_aot_training",
    "profile_fx_fusion",
    "rank_trace_findings",
]


FX_ANNOTATION_PREFIX = "fx_fusion"
FX_OP_ANNOTATION_PREFIX = "fx_op"
FX_FUSION_MARKER = "transformer_nuggets.fx_fusion_region"
FX_FUSION_KERNEL_MARKER = "transformer_nuggets.fx_fusion_kernel"
GPU_WORK_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
FUSIBLE_COPY_OPS = frozenset({torch.ops.aten._to_copy.default})
# NOTE [Logical FLOP Approximation]
# Pointwise nodes count one useful scalar operation per output element and
# reductions count one per input element. Transcendentals are not weighted by
# instruction cost. These values support arithmetic-intensity triage; they are
# not executed-instruction counts.

TRANSPARENT_OPS = frozenset(
    {
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    }
)


@dataclass(frozen=True)
class FusionRegion:
    """One connected eager FX region that could plausibly become one fused kernel."""

    region_id: str
    node_names: tuple[str, ...]
    op_names: tuple[str, ...]
    input_bytes: int
    output_bytes: int
    avoidable_read_bytes: int
    avoidable_write_bytes: int
    logical_flops: int
    phase: str
    module_fqn: str | None
    source_locations: tuple[str, ...]
    pattern_label: str

    @property
    def minimum_avoidable_bytes(self) -> int:
        """Return intermediate reads and writes eliminated by ideal fusion."""
        return self.avoidable_read_bytes + self.avoidable_write_bytes

    @property
    def ideal_bytes(self) -> int:
        """Return the external reads plus escaping writes of an ideal fused kernel."""
        return self.input_bytes + self.output_bytes

    @property
    def eager_minimum_bytes(self) -> int:
        """Return ideal traffic plus the minimum known intermediate round trips."""
        return self.ideal_bytes + self.minimum_avoidable_bytes


@dataclass(frozen=True)
class FxNodeRoofline:
    """Static logical work model for one FX call-function node."""

    node_id: str
    graph_id: str
    node_name: str
    op_name: str
    region_id: str | None
    model_kind: str
    model_confidence: str
    input_bytes: int
    output_bytes: int
    logical_flops: int | None
    phase: str
    module_fqn: str | None
    source_locations: tuple[str, ...]
    pattern_label: str

    @property
    def logical_bytes(self) -> int:
        """Return modeled input reads plus output writes."""
        return self.input_bytes + self.output_bytes

    @property
    def arithmetic_intensity(self) -> float | None:
        """Return logical FLOPs per byte when a FLOP model exists."""
        if self.logical_flops is None:
            return None
        return self.logical_flops / max(self.logical_bytes, 1)


@dataclass(frozen=True)
class FxFusionAnalysis:
    """Captured FX graph and its node/region roofline models."""

    graph_module: GraphModule
    regions: tuple[FusionRegion, ...]
    nodes: tuple[FxNodeRoofline, ...]


@dataclass(frozen=True)
class FxFusionProfile:
    """Result of an eager FX profiling run."""

    analysis: FxFusionAnalysis
    trace_path: Path
    findings_path: Path


@dataclass(frozen=True)
class FxTrainingAnalysis:
    """Forward and backward FX analyses captured by AOTAutograd."""

    forward: tuple[FxFusionAnalysis, ...]
    backward: tuple[FxFusionAnalysis, ...]

    @property
    def regions(self) -> tuple[FusionRegion, ...]:
        """Return all forward and backward candidate regions."""
        return tuple(
            region for analysis in (*self.forward, *self.backward) for region in analysis.regions
        )

    @property
    def nodes(self) -> tuple[FxNodeRoofline, ...]:
        """Return all forward and backward node roofline models."""
        return tuple(
            node for analysis in (*self.forward, *self.backward) for node in analysis.nodes
        )


@dataclass(frozen=True)
class FxTrainingProfile:
    """Result of an eager AOTAutograd forward/backward profiling run."""

    analysis: FxTrainingAnalysis
    trace_path: Path
    findings_path: Path


@dataclass(frozen=True)
class _ProvenanceRecord:
    """One eager dispatcher event used to restore module/source attribution."""

    op_name: str
    phase: str
    output_signature: tuple[Any, ...]
    module_fqn: str | None
    source_locations: tuple[str, ...]


class _ProvenanceMode(TorchDispatchMode):
    """Record eager operator provenance during an isolated training iteration."""

    def __init__(self, tracker: ModuleTracker) -> None:
        super().__init__()
        self.tracker = tracker
        self.records: list[_ProvenanceRecord] = []

    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        output = func(*args, **({} if kwargs is None else kwargs))
        parents = [parent for parent in self.tracker.parents if parent != "Global"]
        self.records.append(
            _ProvenanceRecord(
                op_name=str(func),
                phase="backward" if self.tracker.is_bw else "forward",
                output_signature=_value_signature(output),
                module_fqn=max(parents, key=lambda value: value.count("."), default=None),
                source_locations=_current_source_locations(),
            )
        )
        return output


def _value_signature(value: Any) -> tuple[Any, ...]:
    """Return a compact tensor shape/dtype signature for provenance matching."""
    signature = []
    for leaf in _pytree.tree_leaves(value):
        if isinstance(leaf, torch.Tensor):
            signature.append(
                (
                    tuple(int(dim) for dim in leaf.shape),
                    str(leaf.dtype),
                    tuple(leaf.stride()),
                    str(leaf.device),
                )
            )
    return tuple(signature)


def _current_source_locations() -> tuple[str, ...]:
    """Return the nearest non-PyTorch Python frames for one eager operation."""
    locations = []
    for frame in traceback.extract_stack(limit=40)[:-2]:
        filename = frame.filename.replace("\\", "/")
        if "/torch/" in filename or "/transformer_nuggets/" in filename:
            continue
        if frame.name in {"wrapper", "inner"}:
            continue
        location = f"{frame.filename}:{frame.lineno} in {frame.name}"
        if location not in locations:
            locations.append(location)
    return tuple(locations[-4:])


def _node_source_locations(node: Node) -> tuple[str, ...]:
    """Return source locations restored by sidecar matching or FX stack traces."""
    restored = node.meta.get("roofline_source_locations")
    if restored:
        return tuple(restored)
    stack_trace = node.meta.get("stack_trace")
    if not stack_trace:
        return ()
    return tuple(
        line.strip() for line in str(stack_trace).splitlines() if line.strip().startswith("File ")
    )[-4:]


def _apply_provenance(
    graph_module: GraphModule,
    records: Sequence[_ProvenanceRecord],
    phase: str,
) -> None:
    """Greedily match eager dispatcher provenance onto an AOT FX graph."""
    candidates = [record for record in records if record.phase == phase]
    cursor = 0
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.meta.get("original_aten", node.target)
        if not isinstance(target, torch._ops.OpOverload):
            continue
        op_name = str(target)
        signature = _value_signature(node.meta.get("val"))
        match_index = None
        for index in range(cursor, len(candidates)):
            candidate = candidates[index]
            if candidate.op_name == op_name and (
                not signature or candidate.output_signature == signature
            ):
                match_index = index
                break
        if match_index is None:
            continue
        record = candidates[match_index]
        cursor = match_index + 1
        node.meta["roofline_module_fqn"] = record.module_fqn
        node.meta["roofline_source_locations"] = record.source_locations
        node.meta["roofline_phase"] = phase


class EagerFusionSupport(OperatorSupportBase):
    """Accept functional pointwise/reduction operations and transparent views."""

    def is_node_supported(
        self,
        submodules: Mapping[str, torch.nn.Module],
        node: Node,
    ) -> bool:
        del submodules
        if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
            return False
        if node.target in TRANSPARENT_OPS or node.target in FUSIBLE_COPY_OPS:
            return True
        return not node.target._schema.is_mutable and any(
            tag in node.target.tags for tag in (torch.Tag.pointwise, torch.Tag.reduction)
        )


def _is_fusion_compute(node: Node) -> bool:
    """Return whether a node is a materializing pointwise or reduction operation."""
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target not in TRANSPARENT_OPS
        and (
            node.target in FUSIBLE_COPY_OPS
            or any(tag in node.target.tags for tag in (torch.Tag.pointwise, torch.Tag.reduction))
        )
    )


def _connected_components(nodes: Sequence[Node]) -> list[list[Node]]:
    """Split a proposed FX partition into producer-consumer components."""
    node_set = set(nodes)
    adjacency = {node: set() for node in nodes}
    for node in nodes:
        for input_node in node.all_input_nodes:
            if input_node in node_set:
                adjacency[node].add(input_node)
                adjacency[input_node].add(node)

    components = []
    visited = set()
    for start in nodes:
        if start in visited:
            continue
        component = []
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _tensor_numel(tensor: torch.Tensor) -> int:
    """Return a tensor's static logical element count."""
    try:
        return math.prod(int(dim) for dim in tensor.shape)
    except (TypeError, ValueError, GuardOnDataDependentSymNode) as exc:
        raise ValueError("FX fusion analysis currently requires static tensor shapes") from exc


def _tensor_numel_total(value: Any) -> int:
    """Count unique logical tensor elements in a nested value."""
    seen = set()
    total = 0
    for leaf in _pytree.tree_leaves(value):
        if not isinstance(leaf, torch.Tensor):
            continue
        key = _tensor_alias_key(leaf)
        if key in seen:
            continue
        seen.add(key)
        total += _tensor_numel(leaf)
    return total


def _tensor_alias_key(tensor: torch.Tensor) -> tuple[Any, int, int, int]:
    """Return an exact-view key suitable for de-duplicating fake aliases."""
    try:
        storage = tensor.untyped_storage()._cdata
    except RuntimeError:
        storage = id(tensor)
    return storage, tensor.storage_offset(), _tensor_numel(tensor), tensor.element_size()


def _tensor_nbytes(value: Any) -> int:
    """Count unique logical tensor bytes in a possibly nested result."""
    seen = set()
    total = 0
    for leaf in _pytree.tree_leaves(value):
        if not isinstance(leaf, torch.Tensor):
            continue
        key = _tensor_alias_key(leaf)
        if key in seen:
            continue
        seen.add(key)
        total += _tensor_numel(leaf) * leaf.element_size()
    return total


def _materialized_source(node: Node) -> Node:
    """Return the compute/input node underlying transparent FX aliases."""
    while (
        node.op == "call_function"
        and node.target in TRANSPARENT_OPS
        and len(node.all_input_nodes) == 1
    ):
        node = node.all_input_nodes[0]
    return node


def _unique_node_nbytes(nodes: Sequence[Node] | set[Node]) -> int:
    """Count logical bytes across FX values while de-duplicating exact aliases."""
    seen = set()
    total = 0
    for node in nodes:
        for leaf in _pytree.tree_leaves(node.meta.get("val")):
            if not isinstance(leaf, torch.Tensor):
                continue
            key = _tensor_alias_key(leaf)
            if key in seen:
                continue
            seen.add(key)
            total += _tensor_numel(leaf) * leaf.element_size()
    return total


def _node_nbytes(node: Node) -> int:
    """Return logical bytes for the fake value attached to an FX node."""
    return _tensor_nbytes(node.meta.get("val"))


def _node_logical_flops(node: Node) -> int:
    """Estimate useful scalar FLOPs; see NOTE [Logical FLOP Approximation]."""
    if not isinstance(node.target, torch._ops.OpOverload) or node.target in FUSIBLE_COPY_OPS:
        return 0
    if torch.Tag.reduction in node.target.tags:
        return (
            _tensor_numel_total(node.all_input_nodes[0].meta.get("val"))
            if node.all_input_nodes
            else 0
        )
    if torch.Tag.pointwise in node.target.tags:
        return _tensor_numel_total(node.meta.get("val"))
    return 0


def _matrix_logical_flops(node: Node) -> int | None:
    """Return standard GEMM FLOPs for supported matrix operators."""
    inputs = node.all_input_nodes
    if node.target in {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}:
        matrix_inputs = inputs[-2:]
        if len(matrix_inputs) != 2:
            return None
        left, right = (item.meta.get("val") for item in matrix_inputs)
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return None
        m, k = left.shape
        _, n = right.shape
        return 2 * int(m) * int(n) * int(k)
    if node.target in {torch.ops.aten.bmm.default, torch.ops.aten.baddbmm.default}:
        matrix_inputs = inputs[-2:]
        if len(matrix_inputs) != 2:
            return None
        left, right = (item.meta.get("val") for item in matrix_inputs)
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return None
        batch, m, k = left.shape
        _, _, n = right.shape
        return 2 * int(batch) * int(m) * int(n) * int(k)
    return None


def _fused_rms_norm_logical_flops(node: Node) -> int | None:
    """Estimate useful RMSNorm FLOPs for PyTorch's fused eager operator."""
    if node.target is not torch.ops.aten._fused_rms_norm.default or not node.all_input_nodes:
        return None
    input_value = node.all_input_nodes[0].meta.get("val")
    if not isinstance(input_value, torch.Tensor) or not input_value.shape:
        return None
    elements = _tensor_numel(input_value)
    rows = elements // int(input_value.shape[-1])
    return 4 * elements + 2 * rows


def _roofline_tensors(value: Any) -> tuple[RooflineTensor, ...]:
    """Convert nested FX fake values into formula-facing tensor metadata."""
    return tuple(
        RooflineTensor(
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=tensor.dtype,
            stride=tuple(tensor.stride()),
            requires_grad=tensor.requires_grad,
        )
        for tensor in _pytree.tree_leaves(value)
        if isinstance(tensor, torch.Tensor)
    )


def _fx_formula_context(node: Node) -> RooflineFormulaContext:
    """Build normalized registered-formula inputs for one FX node."""
    concrete_inputs = tuple(
        value
        for value in _pytree.tree_leaves(node.args)
        if not isinstance(value, Node | torch.Tensor)
    )
    kwargs = {
        name: value
        for name, value in node.kwargs.items()
        if not isinstance(value, Node | torch.Tensor)
    }
    return RooflineFormulaContext(
        op_name=str(node.target),
        inputs=tuple(
            tensor
            for input_node in node.all_input_nodes
            for tensor in _roofline_tensors(input_node.meta.get("val"))
        ),
        outputs=_roofline_tensors(node.meta.get("val")),
        concrete_inputs=concrete_inputs,
        kwargs=kwargs,
    )


def _build_node_roofline(node: Node, graph_id: str) -> FxNodeRoofline:
    """Build the best available logical work model for one FX node."""
    region_id = node.meta.get("fusion_region_id")
    node_id = f"{region_id}:{node.name}" if region_id is not None else f"{graph_id}:{node.name}"
    if node.op != "call_function":
        return FxNodeRoofline(
            node_id=node_id,
            graph_id=graph_id,
            node_name=node.name,
            op_name=str(node.target),
            region_id=region_id,
            model_kind="metadata",
            model_confidence="high",
            input_bytes=0,
            output_bytes=0,
            logical_flops=0,
            phase=str(node.meta.get("roofline_phase", _graph_phase(graph_id))),
            module_fqn=node.meta.get("roofline_module_fqn"),
            source_locations=_node_source_locations(node),
            pattern_label="metadata",
        )

    input_bytes = _unique_node_nbytes(set(node.all_input_nodes))
    output_bytes = _node_nbytes(node)
    formula = get_roofline_formula(node.target)
    if formula is not None:
        context = _fx_formula_context(node)
        work = formula(
            context.inputs,
            context.outputs,
            {**(context.kwargs or {}), "_concrete_inputs": context.concrete_inputs},
        )
        model_kind = work.model_kind
        model_confidence = work.confidence
        logical_flops = work.logical_flops
        input_bytes = input_bytes if work.read_bytes is None else work.read_bytes
        output_bytes = output_bytes if work.write_bytes is None else work.write_bytes
    elif not isinstance(node.target, torch._ops.OpOverload):
        model_kind = "python"
        model_confidence = "high"
        logical_flops = 0
        input_bytes = 0
        output_bytes = 0
    elif node.target in TRANSPARENT_OPS:
        model_kind = "view"
        model_confidence = "high"
        logical_flops = 0
        input_bytes = 0
        output_bytes = 0
    elif node.target in FUSIBLE_COPY_OPS:
        model_kind = "copy"
        model_confidence = "high"
        logical_flops = 0
    elif (matrix_flops := _matrix_logical_flops(node)) is not None:
        model_kind = "matrix"
        model_confidence = "high"
        logical_flops = matrix_flops
    elif (rms_flops := _fused_rms_norm_logical_flops(node)) is not None:
        model_kind = "rms_norm"
        model_confidence = "high"
        logical_flops = rms_flops
    elif torch.Tag.reduction in node.target.tags:
        model_kind = "reduction"
        model_confidence = "medium"
        logical_flops = _node_logical_flops(node)
    elif torch.Tag.pointwise in node.target.tags:
        model_kind = "pointwise"
        model_confidence = "medium"
        logical_flops = _node_logical_flops(node)
    else:
        model_kind = "generic_io"
        model_confidence = "unknown"
        logical_flops = None

    op_name = str(node.target)
    return FxNodeRoofline(
        node_id=node_id,
        graph_id=graph_id,
        node_name=node.name,
        op_name=op_name,
        region_id=region_id,
        model_kind=model_kind,
        model_confidence=model_confidence,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        logical_flops=logical_flops,
        phase=str(node.meta.get("roofline_phase", _graph_phase(graph_id))),
        module_fqn=node.meta.get("roofline_module_fqn"),
        source_locations=_node_source_locations(node),
        pattern_label=_node_pattern_label(model_kind, op_name),
    )


def _graph_phase(graph_id: str) -> str:
    """Infer eager/forward/backward phase from a graph identifier."""
    if graph_id.startswith("backward"):
        return "backward"
    if graph_id.startswith("forward"):
        return "forward"
    return "eager"


def _node_pattern_label(model_kind: str, op_name: str) -> str:
    """Return a concise recognizable label for one operation model."""
    if model_kind == "rms_norm":
        return "fused RMSNorm"
    if model_kind == "matrix":
        return "matrix multiplication"
    if model_kind == "reduction":
        return "reduction"
    if model_kind == "copy":
        return "dtype/layout copy"
    if model_kind == "pointwise":
        return op_name.removeprefix("aten.").split(".")[0]
    return model_kind.replace("_", " ")


def _region_pattern_label(op_names: Sequence[str]) -> str:
    """Recognize common fusion-opportunity shapes from ordered ATen operations."""
    lowered = " ".join(op_names).lower()
    if "mean" in lowered and "rsqrt" in lowered and "mul" in lowered:
        return "RMSNorm-like decomposition"
    if any(name in lowered for name in ("sigmoid", "silu")) and "mul" in lowered:
        return "gated activation"
    if any(name in lowered for name in ("sum", "mean", "amax", "var")):
        return "reduction epilogue"
    return "pointwise chain"


def _consumer_reads_and_outputs(node: Node, node_set: set[Node]) -> tuple[int, set[Node]]:
    """Follow transparent views to consumer reads and escaping region values."""
    read_bytes = 0
    outputs = set()
    queue = deque((node, user) for user in node.users)
    visited = set()
    while queue:
        value_node, user = queue.popleft()
        edge = (value_node, user)
        if edge in visited:
            continue
        visited.add(edge)
        if user not in node_set:
            outputs.add(value_node)
        elif _is_fusion_compute(user):
            read_bytes += _node_nbytes(value_node)
        else:
            queue.extend((user, next_user) for next_user in user.users)
    return read_bytes, outputs


def _build_region(
    region_id: str, nodes: Sequence[Node], graph_order: dict[Node, int]
) -> FusionRegion:
    """Build traffic metadata for one connected fusion region."""
    ordered_nodes = sorted(nodes, key=graph_order.__getitem__)
    node_set = set(ordered_nodes)
    compute_nodes = [node for node in ordered_nodes if _is_fusion_compute(node)]
    external_inputs = {
        input_node
        for node in ordered_nodes
        for input_node in node.all_input_nodes
        if input_node not in node_set
    }
    uses = {node: _consumer_reads_and_outputs(node, node_set) for node in compute_nodes}
    escaping_outputs = set().union(*(outputs for _, outputs in uses.values()))
    avoidable_read_bytes = sum(read_bytes for read_bytes, _ in uses.values())
    avoidable_write_bytes = sum(
        _node_nbytes(node)
        for node, (read_bytes, outputs) in uses.items()
        if read_bytes and not outputs
    )

    op_names = tuple(str(node.target) for node in compute_nodes)
    modules = [node.meta.get("roofline_module_fqn") for node in compute_nodes]
    modules = [module for module in modules if module]
    module_fqn = (
        next(
            module
            for module in modules
            if modules.count(module) == max(map(modules.count, modules))
        )
        if modules
        else None
    )
    source_locations = tuple(
        dict.fromkeys(
            location for node in compute_nodes for location in _node_source_locations(node)
        )
    )[:6]
    phase = str(
        next(
            (
                node.meta["roofline_phase"]
                for node in compute_nodes
                if node.meta.get("roofline_phase")
            ),
            "eager",
        )
    )
    return FusionRegion(
        region_id=region_id,
        node_names=tuple(node.name for node in ordered_nodes),
        op_names=op_names,
        input_bytes=_unique_node_nbytes(external_inputs),
        output_bytes=_unique_node_nbytes(
            {_materialized_source(node) for node in escaping_outputs}
        ),
        avoidable_read_bytes=avoidable_read_bytes,
        avoidable_write_bytes=avoidable_write_bytes,
        logical_flops=sum(_node_logical_flops(node) for node in compute_nodes),
        phase=phase,
        module_fqn=module_fqn,
        source_locations=source_locations,
        pattern_label=_region_pattern_label(op_names),
    )


def analyze_fx_graph(
    graph_module: GraphModule,
    *,
    region_prefix: str = "region",
) -> FxFusionAnalysis:
    """Identify connected multi-op fusion regions in an existing FX graph."""
    graph_order = {node: index for index, node in enumerate(graph_module.graph.nodes)}
    for node in graph_module.graph.nodes:
        node.meta["analysis_graph_id"] = region_prefix
    proposed = CapabilityBasedPartitioner(
        graph_module,
        EagerFusionSupport(),
        allows_single_node_partition=True,
    ).propose_partitions()

    components = []
    for partition in proposed:
        components.extend(_connected_components(list(partition.nodes)))
    components = [
        component for component in components if sum(map(_is_fusion_compute, component)) >= 2
    ]
    components.sort(key=lambda component: min(graph_order[node] for node in component))

    regions = []
    for index, component in enumerate(components):
        region_id = f"{region_prefix}_{index:03d}"
        for node in component:
            node.meta["fusion_region_id"] = region_id
        regions.append(_build_region(region_id, component, graph_order))
    return FxFusionAnalysis(
        graph_module=graph_module,
        regions=tuple(regions),
        nodes=tuple(
            _build_node_roofline(node, region_prefix) for node in graph_module.graph.nodes
        ),
    )


def _capture_forward_provenance(
    function: Callable[..., Any],
    args: tuple[Any, ...],
) -> tuple[_ProvenanceRecord, ...]:
    """Collect forward module/source attribution on cloned representative inputs."""
    cloned_args = _clone_warmup_values(args)
    module = function if isinstance(function, torch.nn.Module) else torch.nn.Module()
    module_state = (
        (
            tuple(function.parameters()),
            tuple(function.buffers()),
        )
        if isinstance(function, torch.nn.Module)
        else ((), ())
    )
    cuda_device = _cuda_device((args, module_state))
    mode = _ProvenanceMode(ModuleTracker())
    with _preserve_runtime_state(module, args, cuda_device):
        with mode.tracker, mode:
            function(*cloned_args)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
    return tuple(mode.records)


def analyze_fx_fusion(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    functionalize: bool = True,
) -> FxFusionAnalysis:
    """Capture an eager function and identify connected multi-op fusion regions.

    Args:
        function: Callable to capture with ``make_fx``.
        args: Representative positional arguments. Static tensor shapes are required.
        functionalize: Rewrite mutations into functional operations before capture.

    Returns:
        The captured graph and candidate fusion regions.
    """
    provenance_records = _capture_forward_provenance(function, args)
    captured_function = (
        torch.func.functionalize(function, remove="mutations") if functionalize else function
    )
    capture_args = _clone_warmup_values(args)
    module = function if isinstance(function, torch.nn.Module) else torch.nn.Module()
    module_state = (
        (
            tuple(function.parameters()),
            tuple(function.buffers()),
        )
        if isinstance(function, torch.nn.Module)
        else ((), ())
    )
    cuda_device = _cuda_device((args, module_state))
    with _preserve_runtime_state(module, args, cuda_device):
        graph_module = make_fx(
            captured_function,
            tracing_mode="real",
            record_stack_traces=True,
        )(*capture_args)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
    _apply_provenance(graph_module, provenance_records, "forward")
    return analyze_fx_graph(graph_module)


class FxFusionInterpreter(Interpreter):
    """Execute an FX graph while labeling eager or CUDA Graph operations."""

    def __init__(self, module: GraphModule, *, cuda_graph_annotations: bool = False) -> None:
        super().__init__(module)
        self.cuda_graph_annotations = cuda_graph_annotations

    def run_node(self, node: Node) -> Any:
        region_id = node.meta.get("fusion_region_id")
        if region_id is not None:
            label = f"{FX_ANNOTATION_PREFIX}::{region_id}::{node.name}"
        elif node.op == "call_function":
            graph_id = node.meta.get("analysis_graph_id", "graph")
            label = f"{FX_OP_ANNOTATION_PREFIX}::{graph_id}::{node.name}::{node.target}"
        else:
            label = None
        if label is None:
            scope = nullcontext()
        elif self.cuda_graph_annotations:
            from torch.cuda.graph_annotations import mark_kernels

            scope = mark_kernels(label, backward=False)
        else:
            scope = record_function(label)
        with scope:
            return super().run_node(node)


def _duration_interval(event: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return a finite positive interval for one Chrome duration event."""
    if event.get("ph") != "X":
        return None
    try:
        start = float(event.get("ts", 0))
        duration = float(event.get("dur", 0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
        return None
    return start, start + duration


def _annotation_identity(event: Mapping[str, Any]) -> tuple[str, str | None] | None:
    """Extract a graph-qualified node ID and optional region from a GPU annotation."""
    if event.get("cat") != "gpu_user_annotation":
        return None
    name = str(event.get("name", ""))
    if name.startswith(f"{FX_ANNOTATION_PREFIX}::"):
        parts = name.split("::", 2)
        if len(parts) == 3:
            region_id, node_name = parts[1:]
            return f"{region_id}:{node_name}", region_id
    if name.startswith(f"{FX_OP_ANNOTATION_PREFIX}::"):
        parts = name.split("::", 3)
        if len(parts) == 4:
            graph_id, node_name = parts[1:3]
            return f"{graph_id}:{node_name}", None
    return None


def _finding_severity(region: FusionRegion, observed_launch_count: int) -> str:
    """Classify a finding by traffic reduction and GPU launch count."""
    reduction = region.minimum_avoidable_bytes / max(region.eager_minimum_bytes, 1)
    if region.minimum_avoidable_bytes >= 64 * 2**20 or (
        observed_launch_count >= 4 and reduction >= 0.5
    ):
        return "high"
    if region.minimum_avoidable_bytes >= 2**20 or observed_launch_count >= 3:
        return "medium"
    return "low"


def _finding_name(region: FusionRegion, observed_launch_count: int, severity: str) -> str:
    """Build a compact severity-coded Perfetto label for a fusion candidate."""
    icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[severity]
    mib = region.minimum_avoidable_bytes / 2**20
    reduction = 100 * region.minimum_avoidable_bytes / max(region.eager_minimum_bytes, 1)
    owner = f" · {region.module_fqn}" if region.module_fqn else ""
    return (
        f"{icon} {region.pattern_label}{owner} [{region.region_id}]: "
        f"{observed_launch_count}→1 launches · {mib:.1f} MiB saved · "
        f"{reduction:.0f}% traffic"
    )


def _region_runtime_metrics(
    region: FusionRegion,
    observed_wall_us: float,
    observed_busy_us: float,
    observed_launch_count: int,
    roofline_spec: RooflineSpec | None,
) -> dict[str, Any]:
    """Derive logical achieved rates and optional hardware-relative roofline metrics."""
    current_bytes = region.eager_minimum_bytes
    metrics: dict[str, Any] = {
        "phase": region.phase,
        "module_fqn": region.module_fqn,
        "source_locations": list(region.source_locations),
        "pattern_label": region.pattern_label,
        "logical_flops": region.logical_flops,
        "current_logical_bytes": current_bytes,
        "observed_region_wall_us": observed_wall_us,
        "observed_gpu_busy_us": observed_busy_us,
        "observed_inter_kernel_gap_us": max(observed_wall_us - observed_busy_us, 0.0),
        "current_arithmetic_intensity_flops_per_byte": region.logical_flops
        / max(current_bytes, 1),
        "fused_arithmetic_intensity_flops_per_byte": region.logical_flops
        / max(region.ideal_bytes, 1),
        "achieved_logical_tflops": region.logical_flops / max(observed_wall_us, 1e-12) / 1e6,
        "achieved_logical_gbps": current_bytes / max(observed_wall_us, 1e-12) / 1e3,
        "achieved_busy_logical_tflops": region.logical_flops / max(observed_busy_us, 1e-12) / 1e6,
        "achieved_busy_logical_gbps": current_bytes / max(observed_busy_us, 1e-12) / 1e3,
    }
    if roofline_spec is None:
        return metrics

    compute_floor_us = region.logical_flops / (roofline_spec.peak_compute_tflops * 1e6)
    current_memory_floor_us = current_bytes / (roofline_spec.peak_memory_gbps * 1e3)
    fused_memory_floor_us = region.ideal_bytes / (roofline_spec.peak_memory_gbps * 1e3)
    current_roofline_floor_us = max(compute_floor_us, current_memory_floor_us)
    fused_roofline_floor_us = max(
        compute_floor_us,
        fused_memory_floor_us,
        roofline_spec.launch_latency_us,
    )
    metrics.update(
        {
            "roofline_spec": roofline_spec.name,
            "peak_compute_tflops": roofline_spec.peak_compute_tflops,
            "peak_memory_gbps": roofline_spec.peak_memory_gbps,
            "assumed_launch_latency_us": roofline_spec.launch_latency_us,
            "predicted_bound": (
                "compute" if compute_floor_us >= current_memory_floor_us else "memory"
            ),
            "compute_floor_us": compute_floor_us,
            "current_memory_floor_us": current_memory_floor_us,
            "current_roofline_floor_us": current_roofline_floor_us,
            "achieved_roofline_percent": 100
            * current_roofline_floor_us
            / max(observed_wall_us, 1e-12),
            "fused_memory_floor_us": fused_memory_floor_us,
            "fused_roofline_floor_us": fused_roofline_floor_us,
            "idealized_recoverable_us": max(observed_wall_us - fused_roofline_floor_us, 0.0),
            "idealized_fused_speedup_upper_bound": observed_wall_us
            / max(fused_roofline_floor_us, 1e-12),
            "current_launch_floor_us": observed_launch_count * roofline_spec.launch_latency_us,
        }
    )
    return metrics


def _node_runtime_metrics(
    node: FxNodeRoofline,
    observed_wall_us: float,
    observed_busy_us: float,
    roofline_spec: RooflineSpec | None,
) -> dict[str, Any]:
    """Derive achieved logical rates for one FX node and its GPU operations."""
    logical_bytes = node.logical_bytes
    achieved_tflops = (
        None
        if node.logical_flops is None
        else node.logical_flops / max(observed_wall_us, 1e-12) / 1e6
    )
    metrics: dict[str, Any] = {
        "node_id": node.node_id,
        "graph_id": node.graph_id,
        "phase": node.phase,
        "module_fqn": node.module_fqn,
        "source_locations": list(node.source_locations),
        "pattern_label": node.pattern_label,
        "node_name": node.node_name,
        "op_name": node.op_name,
        "region_id": node.region_id,
        "model_kind": node.model_kind,
        "model_confidence": node.model_confidence,
        "logical_flops": node.logical_flops,
        "logical_read_bytes": node.input_bytes,
        "logical_write_bytes": node.output_bytes,
        "logical_bytes": logical_bytes,
        "arithmetic_intensity_flops_per_byte": node.arithmetic_intensity,
        "observed_node_wall_us": observed_wall_us,
        "observed_node_gpu_busy_us": observed_busy_us,
        "observed_node_gap_us": max(observed_wall_us - observed_busy_us, 0.0),
        "achieved_logical_tflops": achieved_tflops,
        "achieved_logical_gbps": logical_bytes / max(observed_wall_us, 1e-12) / 1e3,
        "achieved_logical_tbps": logical_bytes / max(observed_wall_us, 1e-12) / 1e6,
        "achieved_busy_logical_tflops": (
            None
            if node.logical_flops is None
            else node.logical_flops / max(observed_busy_us, 1e-12) / 1e6
        ),
        "achieved_busy_logical_gbps": logical_bytes / max(observed_busy_us, 1e-12) / 1e3,
        "achieved_busy_logical_tbps": logical_bytes / max(observed_busy_us, 1e-12) / 1e6,
    }
    if roofline_spec is None:
        return metrics

    memory_floor_us = logical_bytes / (roofline_spec.peak_memory_gbps * 1e3)
    compute_floor_us = (
        None
        if node.logical_flops is None
        else node.logical_flops / (roofline_spec.peak_compute_tflops * 1e6)
    )
    roofline_floor_us = (
        memory_floor_us if compute_floor_us is None else max(compute_floor_us, memory_floor_us)
    )
    metrics.update(
        {
            "roofline_spec": roofline_spec.name,
            "peak_compute_tflops": roofline_spec.peak_compute_tflops,
            "peak_memory_gbps": roofline_spec.peak_memory_gbps,
            "predicted_bound": (
                "unknown_compute"
                if compute_floor_us is None
                else "compute"
                if compute_floor_us >= memory_floor_us
                else "memory"
            ),
            "compute_floor_us": compute_floor_us,
            "memory_floor_us": memory_floor_us,
            "roofline_floor_us": roofline_floor_us,
            "achieved_memory_roofline_percent": 100
            * memory_floor_us
            / max(observed_wall_us, 1e-12),
            "achieved_roofline_percent": (
                None
                if compute_floor_us is None
                else 100 * roofline_floor_us / max(observed_wall_us, 1e-12)
            ),
        }
    )
    return metrics


def _node_finding_name(node: FxNodeRoofline, metrics: Mapping[str, Any], launches: int) -> str:
    """Build a compact node-level achieved-rate label."""
    tflops = metrics["achieved_logical_tflops"]
    tflops_label = "? TF/s" if tflops is None else f"{tflops:.2f} TF/s"
    owner = f" · {node.module_fqn}" if node.module_fqn else ""
    return (
        f"Node {node.pattern_label}{owner}: "
        f"{launches} launch{'es' if launches != 1 else ''} · "
        f"{tflops_label} · {metrics['achieved_logical_tbps']:.2f} TB/s"
    )


def add_fx_fusion_region_boxes(
    trace: dict[str, Any],
    regions: Sequence[FusionRegion],
    *,
    nodes: Sequence[FxNodeRoofline] = (),
    keep_node_annotations: bool = True,
    roofline_spec: RooflineSpec | None = None,
) -> dict[str, Any]:
    """Combine per-node FX GPU annotations into candidate-region boxes."""
    raw_events = trace.get("traceEvents", ())
    if any(
        isinstance(event, Mapping)
        and isinstance(event.get("args"), Mapping)
        and event["args"].get(FX_FUSION_MARKER)
        for event in raw_events
    ):
        return trace.copy()

    events = []
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            events.append(raw_event)
            continue
        event = dict(raw_event)
        if isinstance(event.get("args"), Mapping):
            event["args"] = dict(event["args"])
        events.append(event)
    regions_by_id = {region.region_id: region for region in regions}
    nodes_by_id = {node.node_id: node for node in nodes}
    annotations_by_track: dict[tuple[Any, Any], list[tuple[float, float, str, str | None]]] = (
        defaultdict(list)
    )
    work_by_track: dict[tuple[Any, Any], list[tuple[int, float, float, str]]] = defaultdict(list)
    fx_annotation_indices = set()
    annotation_node_ids: dict[int, str] = {}

    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        interval = _duration_interval(event)
        if interval is None:
            continue
        track = (event.get("pid", 0), event.get("tid", 0))
        identity = _annotation_identity(event)
        if identity is not None:
            node_id, region_id = identity
            annotations_by_track[track].append((*interval, node_id, region_id))
            fx_annotation_indices.add(index)
            annotation_node_ids[index] = node_id
        elif event.get("cat") in GPU_WORK_CATEGORIES:
            work_by_track[track].append((index, *interval, str(event.get("cat"))))

    boxes = []
    node_event_indices: dict[str, list[int]] = defaultdict(list)
    node_busy_us: dict[str, float] = defaultdict(float)
    node_starts: dict[str, float] = {}
    node_ends: dict[str, float] = {}
    region_event_indices: dict[str, list[int]] = defaultdict(list)
    region_busy_us: dict[str, float] = defaultdict(float)
    region_starts: dict[str, float] = {}
    region_ends: dict[str, float] = {}
    for track, work_events in work_by_track.items():
        annotations = annotations_by_track.get(track, ())
        active: tuple[str, float, float, int, int] | None = None
        for event_index, start, end, category in sorted(
            work_events, key=lambda item: (item[1], item[2], item[0])
        ):
            matching = [
                (annotation_end - annotation_start, node_id, region_id)
                for annotation_start, annotation_end, node_id, region_id in annotations
                if annotation_start <= start and end <= annotation_end
            ]
            if not matching:
                if active is not None:
                    boxes.append((track, *active))
                    active = None
                continue
            _, node_id, region_id = min(matching)
            node_event_indices[node_id].append(event_index)
            node_busy_us[node_id] += end - start
            node_starts[node_id] = min(node_starts.get(node_id, start), start)
            node_ends[node_id] = max(node_ends.get(node_id, end), end)

            if region_id not in regions_by_id:
                if active is not None:
                    boxes.append((track, *active))
                    active = None
                continue
            region_event_indices[region_id].append(event_index)
            region_busy_us[region_id] += end - start
            region_starts[region_id] = min(region_starts.get(region_id, start), start)
            region_ends[region_id] = max(region_ends.get(region_id, end), end)
            if active is None or active[0] != region_id:
                if active is not None:
                    boxes.append((track, *active))
                active = (region_id, start, end, int(category == "kernel"), 1)
            else:
                active = (
                    region_id,
                    active[1],
                    max(active[2], end),
                    active[3] + int(category == "kernel"),
                    active[4] + 1,
                )
        if active is not None:
            boxes.append((track, *active))

    node_kernel_counts: dict[str, int] = defaultdict(int)
    node_launch_counts: dict[str, int] = defaultdict(int)
    for node_id, event_indices in node_event_indices.items():
        for event_index in event_indices:
            node_launch_counts[node_id] += 1
            node_kernel_counts[node_id] += int(events[event_index].get("cat") == "kernel")
    node_runtime_metrics = {
        node_id: _node_runtime_metrics(
            nodes_by_id[node_id],
            node_ends[node_id] - node_starts[node_id],
            node_busy_us[node_id],
            roofline_spec,
        )
        for node_id in node_event_indices
        if node_id in nodes_by_id
    }

    segment_counts: dict[str, int] = defaultdict(int)
    region_kernel_counts: dict[str, int] = defaultdict(int)
    region_launch_counts: dict[str, int] = defaultdict(int)
    for _, region_id, _, _, kernel_count, launch_count in boxes:
        segment_counts[region_id] += 1
        region_kernel_counts[region_id] += kernel_count
        region_launch_counts[region_id] += launch_count
    segment_indices: dict[str, int] = defaultdict(int)
    runtime_metrics = {
        region_id: _region_runtime_metrics(
            regions_by_id[region_id],
            region_ends[region_id] - region_starts[region_id],
            region_busy_us[region_id],
            region_launch_counts[region_id],
            roofline_spec,
        )
        for region_id in region_event_indices
    }

    generated = []
    for (pid, tid), region_id, start, end, segment_kernel_count, segment_launch_count in boxes:
        region = regions_by_id[region_id]
        observed_kernel_count = region_kernel_counts[region_id]
        observed_launch_count = region_launch_counts[region_id]
        severity = _finding_severity(region, observed_launch_count)
        segment_indices[region_id] += 1
        traffic_reduction = region.minimum_avoidable_bytes / max(region.eager_minimum_bytes, 1)
        idealized_bandwidth_speedup = region.eager_minimum_bytes / max(region.ideal_bytes, 1)
        generated.append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "cname": {"high": "terrible", "medium": "yellow", "low": "good"}[severity],
                "name": _finding_name(region, observed_launch_count, severity),
                "pid": pid,
                "tid": tid,
                "ts": start,
                "dur": end - start,
                "args": {
                    FX_FUSION_MARKER: True,
                    "region_id": region.region_id,
                    "severity": severity,
                    "operations": list(region.op_names),
                    "operation_count": len(region.op_names),
                    "observed_kernel_count": observed_kernel_count,
                    "observed_gpu_launch_count": observed_launch_count,
                    "expected_kernel_count": 1,
                    "expected_gpu_launch_count": 1,
                    "expected_kernel_savings": max(observed_kernel_count - 1, 0),
                    "expected_gpu_launch_savings": max(observed_launch_count - 1, 0),
                    "segment_index": segment_indices[region_id],
                    "segment_count": segment_counts[region_id],
                    "segment_kernel_count": segment_kernel_count,
                    "segment_gpu_launch_count": segment_launch_count,
                    "segment_metrics": "region_total",
                    "expected_read_bytes": region.input_bytes,
                    "expected_write_bytes": region.output_bytes,
                    "avoidable_intermediate_read_bytes": region.avoidable_read_bytes,
                    "avoidable_intermediate_write_bytes": region.avoidable_write_bytes,
                    "ideal_bytes": region.ideal_bytes,
                    "eager_minimum_bytes": region.eager_minimum_bytes,
                    "minimum_avoidable_bytes": region.minimum_avoidable_bytes,
                    "traffic_reduction_percent": 100 * traffic_reduction,
                    "idealized_bandwidth_speedup": idealized_bandwidth_speedup,
                    **runtime_metrics[region_id],
                },
            }
        )

    for annotation_index, node_id in annotation_node_ids.items():
        if node_id not in node_runtime_metrics:
            continue
        node = nodes_by_id[node_id]
        metrics = node_runtime_metrics[node_id]
        annotation = events[annotation_index]
        annotation["name"] = _node_finding_name(
            node,
            metrics,
            node_launch_counts[node_id],
        )
        annotation.setdefault("args", {}).update(
            {
                **metrics,
                "observed_kernel_count": node_kernel_counts[node_id],
                "observed_gpu_launch_count": node_launch_counts[node_id],
            }
        )

    for node_id, event_indices in node_event_indices.items():
        if node_id not in node_runtime_metrics:
            continue
        metrics = node_runtime_metrics[node_id]
        kernel_index = 0
        for gpu_op_index, event_index in enumerate(
            sorted(event_indices, key=lambda index: float(events[index].get("ts", 0))),
            start=1,
        ):
            event = events[event_index]
            args = event.setdefault("args", {})
            is_kernel = event.get("cat") == "kernel"
            if is_kernel:
                kernel_index += 1
            args.update(
                {
                    "transformer_nuggets.fx_node_roofline": True,
                    "fx_node_id": node_id,
                    "fx_node_gpu_op_index": gpu_op_index,
                    "fx_node_kernel_index": kernel_index if is_kernel else 0,
                    "fx_node_kernel_duration_us": float(event.get("dur", 0)),
                    "fx_node_op_name": metrics["op_name"],
                    "fx_node_model_kind": metrics["model_kind"],
                    "fx_node_model_confidence": metrics["model_confidence"],
                    "fx_node_phase": metrics["phase"],
                    "fx_node_module_fqn": metrics["module_fqn"],
                    "fx_node_source_locations": metrics["source_locations"],
                    "fx_node_pattern_label": metrics["pattern_label"],
                    "fx_node_logical_flops": metrics["logical_flops"],
                    "fx_node_logical_bytes": metrics["logical_bytes"],
                    "fx_node_arithmetic_intensity": metrics["arithmetic_intensity_flops_per_byte"],
                    "fx_node_achieved_logical_tflops": metrics["achieved_logical_tflops"],
                    "fx_node_achieved_logical_gbps": metrics["achieved_logical_gbps"],
                    "fx_node_achieved_logical_tbps": metrics["achieved_logical_tbps"],
                }
            )
            if roofline_spec is not None:
                args.update(
                    {
                        "fx_node_predicted_bound": metrics["predicted_bound"],
                        "fx_node_achieved_memory_roofline_percent": metrics[
                            "achieved_memory_roofline_percent"
                        ],
                        "fx_node_achieved_roofline_percent": metrics["achieved_roofline_percent"],
                    }
                )

    for region_id, event_indices in region_event_indices.items():
        metrics = runtime_metrics[region_id]
        severity = _finding_severity(regions_by_id[region_id], region_launch_counts[region_id])
        kernel_index = 0
        for gpu_op_index, event_index in enumerate(
            sorted(event_indices, key=lambda index: float(events[index].get("ts", 0))),
            start=1,
        ):
            event = events[event_index]
            args = event.setdefault("args", {})
            is_kernel = event.get("cat") == "kernel"
            if is_kernel:
                kernel_index += 1
            args.update(
                {
                    FX_FUSION_KERNEL_MARKER: True,
                    "fx_fusion_region_id": region_id,
                    "fx_fusion_region_severity": severity,
                    "fx_fusion_region_phase": metrics["phase"],
                    "fx_fusion_region_module_fqn": metrics["module_fqn"],
                    "fx_fusion_region_source_locations": metrics["source_locations"],
                    "fx_fusion_region_pattern_label": metrics["pattern_label"],
                    "fx_fusion_gpu_op_index": gpu_op_index,
                    "fx_fusion_kernel_index": kernel_index if is_kernel else 0,
                    "fx_fusion_kernel_duration_us": float(event.get("dur", 0)),
                    "fx_fusion_region_logical_flops": metrics["logical_flops"],
                    "fx_fusion_region_logical_bytes": metrics["current_logical_bytes"],
                    "fx_fusion_region_wall_us": metrics["observed_region_wall_us"],
                    "fx_fusion_region_achieved_logical_tflops": metrics["achieved_logical_tflops"],
                    "fx_fusion_region_achieved_logical_gbps": metrics["achieved_logical_gbps"],
                }
            )
            if roofline_spec is not None:
                args.update(
                    {
                        "fx_fusion_region_predicted_bound": metrics["predicted_bound"],
                        "fx_fusion_region_achieved_roofline_percent": metrics[
                            "achieved_roofline_percent"
                        ],
                    }
                )

    output = trace.copy()
    output["traceEvents"] = [
        event
        for index, event in enumerate(events)
        if keep_node_annotations or index not in fx_annotation_indices
    ] + generated
    return output


def rank_trace_findings(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Build a step-normalized ranked report from an enriched Chrome trace."""

    def numeric(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return result if math.isfinite(result) and result >= 0 else default

    events = [event for event in trace.get("traceEvents", ()) if isinstance(event, Mapping)]
    gpu_intervals = [
        interval
        for event in events
        if event.get("cat") in GPU_WORK_CATEGORIES
        if (interval := _duration_interval(event)) is not None
    ]
    step_start = min((start for start, _ in gpu_intervals), default=0.0)
    step_end = max((end for _, end in gpu_intervals), default=step_start)
    step_wall_us = step_end - step_start

    region_rows: dict[str, dict[str, Any]] = {}
    node_rows: dict[str, dict[str, Any]] = {}
    for event in events:
        args = event.get("args")
        if event.get("cat") != "gpu_user_annotation" or not isinstance(args, Mapping):
            continue
        if args.get(FX_FUSION_MARKER):
            if not args.get("region_id"):
                continue
            region_id = str(args["region_id"])
            if region_id in region_rows:
                continue
            row = dict(args)
            observed_wall_us = numeric(row.get("observed_region_wall_us"))
            if "idealized_recoverable_us" in row:
                recoverable_us = numeric(row.get("idealized_recoverable_us"))
                priority_basis = "supplied_roofline_fused_floor"
            else:
                reduction = numeric(row.get("traffic_reduction_percent")) / 100
                recoverable_us = observed_wall_us * reduction
                priority_basis = "traffic_reduction_proxy"
            row.update(
                {
                    "observed_step_percent": 100 * observed_wall_us / max(step_wall_us, 1e-12),
                    "priority_recoverable_us": recoverable_us,
                    "priority_step_percent": 100 * recoverable_us / max(step_wall_us, 1e-12),
                    "priority_basis": priority_basis,
                }
            )
            region_rows[region_id] = row
        elif args.get("node_id"):
            node_id = str(args["node_id"])
            if node_id in node_rows:
                continue
            row = dict(args)
            observed_wall_us = numeric(row.get("observed_node_wall_us"))
            row["observed_step_percent"] = 100 * observed_wall_us / max(step_wall_us, 1e-12)
            node_rows[node_id] = row

    regions = sorted(
        region_rows.values(),
        key=lambda row: (
            -numeric(row.get("priority_recoverable_us")),
            -numeric(row.get("observed_region_wall_us")),
            str(row["region_id"]),
        ),
    )
    nodes = sorted(
        node_rows.values(),
        key=lambda row: (
            -numeric(row.get("observed_node_wall_us")),
            str(row["node_id"]),
        ),
    )
    for rank, row in enumerate(regions, start=1):
        row["rank"] = rank
    for rank, row in enumerate(nodes, start=1):
        row["rank"] = rank
    return {
        "summary": {
            "gpu_step_wall_us": step_wall_us,
            "region_finding_count": len(regions),
            "attributed_node_count": len(nodes),
        },
        "regions": regions,
        "nodes": nodes,
    }


def _node_value_description(node: Node) -> str:
    """Return a compact shape/dtype label for an FX value."""
    tensors = [
        leaf
        for leaf in _pytree.tree_leaves(node.meta.get("val"))
        if isinstance(leaf, torch.Tensor)
    ]
    if not tensors:
        return node.name
    tensor = tensors[0]
    shape = "×".join(str(int(dim)) for dim in tensor.shape) or "scalar"
    return f"{node.name}\\n{shape} {str(tensor.dtype).removeprefix('torch.')}"


def _region_followup_markdown(
    analysis: FxFusionAnalysis,
    region: FusionRegion,
    finding: Mapping[str, Any],
) -> str:
    """Render one fusion region and its boundaries as a Mermaid report."""
    graph_nodes = {node.name: node for node in analysis.graph_module.graph.nodes}
    region_nodes = [graph_nodes[name] for name in region.node_names]
    region_set = set(region_nodes)
    boundary_inputs = {
        input_node
        for node in region_nodes
        for input_node in node.all_input_nodes
        if input_node not in region_set
    }
    escaping = {
        node for node in region_nodes if any(user not in region_set for user in node.users)
    }

    lines = ["flowchart LR"]
    for node in sorted(boundary_inputs, key=lambda item: item.name):
        lines.append(f'    I_{node.name}["{_node_value_description(node)}"]')
    for node in region_nodes:
        label = str(node.target).replace('"', "'")
        lines.append(f'    N_{node.name}["{node.name}\\n{label}"]')
    for node in sorted(escaping, key=lambda item: item.name):
        lines.append(f'    O_{node.name}["output {node.name}"]')
    for node in region_nodes:
        for input_node in node.all_input_nodes:
            source = f"N_{input_node.name}" if input_node in region_set else f"I_{input_node.name}"
            lines.append(f"    {source} --> N_{node.name}")
        if node in escaping:
            lines.append(f"    N_{node.name} --> O_{node.name}")
    lines.append("    classDef candidate fill:#fff3cd,stroke:#d39e00,color:#111")
    lines.append("    classDef boundary fill:#e8f1ff,stroke:#3974c6,color:#111")
    if region_nodes:
        lines.append(
            "    class " + ",".join(f"N_{node.name}" for node in region_nodes) + " candidate"
        )
    boundary_ids = [
        *(f"I_{node.name}" for node in boundary_inputs),
        *(f"O_{node.name}" for node in escaping),
    ]
    if boundary_ids:
        lines.append("    class " + ",".join(boundary_ids) + " boundary")

    return f"""# {region.pattern_label}: {region.region_id}

- Phase: `{region.phase}`
- Module: `{region.module_fqn or "unknown"}`
- Rank: {finding["rank"]}
- Observed wall time: {float(finding["observed_region_wall_us"]):.2f} us
- Priority recoverable time: {float(finding["priority_recoverable_us"]):.2f} us
- Step impact: {float(finding["priority_step_percent"]):.2f}%
- Launches: {finding["observed_gpu_launch_count"]} → {finding["expected_gpu_launch_count"]}
- Logical traffic: {int(finding["eager_minimum_bytes"])} → {int(finding["ideal_bytes"])} bytes

```mermaid
{chr(10).join(lines)}
```
"""


def _extract_region_graph(
    analysis: FxFusionAnalysis,
    region: FusionRegion,
) -> tuple[GraphModule, list[Node]]:
    """Extract a candidate region into a standalone eager FX GraphModule."""
    graph_order = {node: index for index, node in enumerate(analysis.graph_module.graph.nodes)}
    graph_nodes = {node.name: node for node in analysis.graph_module.graph.nodes}
    region_nodes = [graph_nodes[name] for name in region.node_names]
    region_set = set(region_nodes)
    external_inputs = sorted(
        {
            input_node
            for node in region_nodes
            for input_node in node.all_input_nodes
            if input_node not in region_set
        },
        key=graph_order.__getitem__,
    )
    escaping_outputs = sorted(
        {node for node in region_nodes if any(user not in region_set for user in node.users)},
        key=graph_order.__getitem__,
    )

    graph = torch.fx.Graph()
    environment = {node: graph.placeholder(node.name) for node in external_inputs}
    for node in sorted(region_nodes, key=graph_order.__getitem__):
        environment[node] = graph.node_copy(node, lambda value: environment[value])
        environment[node].meta.clear()
    outputs = tuple(environment[node] for node in escaping_outputs)
    graph.output(outputs[0] if len(outputs) == 1 else outputs)
    graph_module = GraphModule(analysis.graph_module, graph)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module, external_inputs


def _region_replay_manifest(
    analysis: FxFusionAnalysis,
    region: FusionRegion,
    external_inputs: Sequence[Node],
) -> dict[str, Any]:
    """Build static tensor/operation metadata for an executable replay bundle."""
    graph_nodes = {node.name: node for node in analysis.graph_module.graph.nodes}
    region_nodes = [graph_nodes[name] for name in region.node_names]

    storage_groups: dict[tuple[Any, torch.dtype], int] = {}

    def specs(node: Node) -> list[dict[str, Any]]:
        result = []
        for tensor in _pytree.tree_leaves(node.meta.get("val")):
            if not isinstance(tensor, torch.Tensor):
                continue
            try:
                storage = tensor.untyped_storage()
                storage_key = (storage._cdata, tensor.dtype)
                storage_size = storage.nbytes() // tensor.element_size()
            except RuntimeError:
                storage_key = (id(tensor), tensor.dtype)
                storage_size = tensor.numel()
            storage_group = storage_groups.setdefault(storage_key, len(storage_groups))
            result.append(
                {
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                    "stride": list(tensor.stride()),
                    "storage_offset": tensor.storage_offset(),
                    "storage_size": storage_size,
                    "storage_group": storage_group,
                    "requires_grad": tensor.requires_grad,
                }
            )
        return result

    return {
        "region_id": region.region_id,
        "phase": region.phase,
        "module_fqn": region.module_fqn,
        "source_locations": list(region.source_locations),
        "pattern_label": region.pattern_label,
        "inputs": [{"node": node.name, "tensors": specs(node)} for node in external_inputs],
        "nodes": [
            {
                "node": node.name,
                "target": str(node.target),
                "inputs": [input_node.name for input_node in node.all_input_nodes],
                "outputs": specs(node),
            }
            for node in region_nodes
        ],
    }


def _replay_script() -> str:
    """Return the standalone eager region replay driver source."""
    return """#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def make_tensor(spec, device, backings):
    dtype_name = spec["dtype"].removeprefix("torch.")
    dtype = getattr(torch, dtype_name)
    group = spec["storage_group"]
    backing = backings.get(group)
    if backing is None:
        backing = torch.empty(spec["storage_size"], dtype=dtype, device=device)
        if backing.is_floating_point() or backing.is_complex():
            backing.normal_()
        else:
            backing.zero_()
        backings[group] = backing
    tensor = backing.as_strided(
        spec["shape"],
        spec["stride"],
        spec["storage_offset"],
    )
    if spec["requires_grad"] and (tensor.is_floating_point() or tensor.is_complex()):
        tensor.requires_grad_(True)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads(next(root.glob("*.replay.json")).read_text())
    module = torch.load(root / "region.pt", weights_only=False)
    inputs = []
    backings = {}
    for entry in manifest["inputs"]:
        tensors = [make_tensor(spec, args.device, backings) for spec in entry["tensors"]]
        if not tensors:
            raise RuntimeError(f"Cannot reconstruct non-tensor input {entry['node']}")
        inputs.append(tensors[0] if len(tensors) == 1 else tuple(tensors))
    with torch.no_grad():
        output = module(*inputs)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    leaves = [value for value in torch.utils._pytree.tree_leaves(output) if isinstance(value, torch.Tensor)]
    print(manifest["region_id"], [(tuple(value.shape), str(value.dtype)) for value in leaves])


if __name__ == "__main__":
    main()
"""


def _write_investigation_followups(
    analyses: Sequence[FxFusionAnalysis],
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write diagrams, replay manifests, and bounded NCU templates per finding."""
    analysis_by_region = {
        region.region_id: analysis for analysis in analyses for region in analysis.regions
    }
    region_by_id = {
        region.region_id: region for analysis in analyses for region in analysis.regions
    }
    followup_dir = output_path.with_name(f"{output_path.name}.followups")
    followup_dir.mkdir(parents=True, exist_ok=True)
    for finding in report["regions"]:
        region_id = str(finding["region_id"])
        analysis = analysis_by_region.get(region_id)
        region = region_by_id.get(region_id)
        if analysis is None or region is None:
            continue
        prefix = f"{int(finding['rank']):02d}_{region_id}"
        region_dir = followup_dir / prefix
        region_dir.mkdir(parents=True, exist_ok=True)
        diagram_path = region_dir / "diagram.md"
        replay_path = region_dir / "manifest.replay.json"
        region_path = region_dir / "region.pt"
        replay_script_path = region_dir / "replay.py"
        ncu_path = region_dir / "ncu.sh"
        region_module, external_inputs = _extract_region_graph(analysis, region)
        diagram_path.write_text(_region_followup_markdown(analysis, region, finding))
        replay_path.write_text(
            json.dumps(
                _region_replay_manifest(analysis, region, external_inputs),
                indent=2,
            )
        )
        torch.save(region_module, region_path)
        replay_script_path.write_text(_replay_script())
        replay_script_path.chmod(0o755)
        ncu_path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            f"# Region: {region_id} ({region.pattern_label})\n"
            f"# Expected launches: {finding['observed_gpu_launch_count']} current, "
            f"{finding['expected_gpu_launch_count']} ideal.\n"
            "# Replace KERNEL_REGEX after the replay identifies the representative launch.\n"
            "ncu --set roofline --kernel-name-base demangled "
            "--kernel-name 'regex:<KERNEL_REGEX>' --launch-count 1 -- "
            "python replay.py --device cuda\n"
        )
        ncu_path.chmod(0o755)
        finding["followups"] = {
            "diagram": str(diagram_path),
            "replay_manifest": str(replay_path),
            "region_module": str(region_path),
            "replay_script": str(replay_script_path),
            "ncu_template": str(ncu_path),
        }


def _cuda_device(values: Any) -> torch.device | None:
    """Return the single CUDA device used by nested tensor arguments."""
    devices = {
        value.device
        for value in _pytree.tree_leaves(values)
        if isinstance(value, torch.Tensor) and value.is_cuda
    }
    if len(devices) > 1:
        raise ValueError("FX fusion profiling currently requires a single CUDA device")
    return next(iter(devices), None)


def _write_annotated_trace(
    torch_profiler: Any,
    path: str | Path,
    analyses: Sequence[FxFusionAnalysis],
    trace_format: TraceFormat,
    roofline_spec: RooflineSpec | None,
    graph_annotations: Mapping[int, Sequence[Any]] | None = None,
) -> tuple[Path, Path]:
    """Export one profiler result plus ranked JSON findings."""
    regions = tuple(region for analysis in analyses for region in analysis.regions)
    nodes = tuple(node for analysis in analyses for node in analysis.nodes)
    output_path = perfetto_trace_path(path, trace_format=trace_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_trace_path = Path(temp_dir) / "fx_fusion.json"
        torch_profiler.export_chrome_trace(str(raw_trace_path))
        trace = read_trace(raw_trace_path)
        if graph_annotations:
            trace = add_cuda_graph_annotation_boxes(trace, graph_annotations)
        trace = add_fx_fusion_region_boxes(
            trace,
            regions,
            nodes=nodes,
            roofline_spec=roofline_spec,
        )
        write_perfetto_trace(output_path, trace, trace_format=trace_format)
        findings_path = output_path.with_name(f"{output_path.name}.findings.json")
        report = rank_trace_findings(trace)
        _write_investigation_followups(analyses, report, output_path)
        findings_path.write_text(json.dumps(report, indent=2))
    return output_path, findings_path


def profile_fx_fusion(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    path: str | Path,
    *,
    functionalize: bool = True,
    trace_format: TraceFormat = "track_event",
    roofline_spec: RooflineSpec | None = None,
    cuda_graph: bool = False,
) -> FxFusionProfile:
    """Profile an FX graph eagerly and write an annotated Perfetto trace.

    The MVP accepts positional arguments with static shapes and profiles at most
    one CUDA device. It does not invoke Inductor or another code generator.
    """
    analysis = analyze_fx_fusion(function, args, functionalize=functionalize)
    module = function if isinstance(function, torch.nn.Module) else torch.nn.Module()
    module_state = (
        (
            tuple(function.parameters()),
            tuple(function.buffers()),
        )
        if isinstance(function, torch.nn.Module)
        else ((), ())
    )
    cuda_device = _cuda_device((args, module_state))
    activities = [ProfilerActivity.CPU]
    if cuda_device is not None:
        activities.append(ProfilerActivity.CUDA)

    graph_annotations = None
    if cuda_graph:
        if cuda_device is None:
            raise ValueError("CUDA Graph profiling requires CUDA tensor arguments")
        interpreter = FxFusionInterpreter(
            analysis.graph_module,
            cuda_graph_annotations=True,
        )
        capture_args = _clone_warmup_values(args)
        capture_stream = torch.cuda.Stream(device=cuda_device)
        capture_stream.wait_stream(torch.cuda.current_stream(cuda_device))
        with _preserve_runtime_state(module, args, cuda_device):
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    interpreter.run(*capture_args)
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                stream=capture_stream,
                enable_annotations=True,
            ):
                captured_output = interpreter.run(*capture_args)
        with profile(activities=activities, record_shapes=True) as torch_profiler:
            graph.replay()
            torch.cuda.synchronize(cuda_device)
        from torch.cuda.graph_annotations import get_kernel_annotations

        graph_annotations = get_kernel_annotations()
        del captured_output
    else:
        with profile(activities=activities, record_shapes=True) as torch_profiler:
            FxFusionInterpreter(analysis.graph_module).run(*args)
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)

    trace_path, findings_path = _write_annotated_trace(
        torch_profiler,
        path,
        (analysis,),
        trace_format,
        roofline_spec,
        graph_annotations,
    )
    return FxFusionProfile(
        analysis=analysis,
        trace_path=trace_path,
        findings_path=findings_path,
    )


class _AotGraphCapture:
    """Capture and eagerly execute one AOTAutograd graph phase."""

    def __init__(
        self,
        phase: str,
        provenance_records: Sequence[_ProvenanceRecord] = (),
        provenance_state: dict[int, tuple[str | None, tuple[str, ...]]] | None = None,
    ) -> None:
        self.phase = phase
        self.provenance_records = provenance_records
        self.provenance_state = {} if provenance_state is None else provenance_state
        self.analyses: list[FxFusionAnalysis] = []

    def __call__(self, graph_module: GraphModule, example_inputs: Sequence[Any]) -> Callable:
        from torch._functorch._aot_autograd.utils import make_boxed_func

        del example_inputs
        _apply_provenance(graph_module, self.provenance_records, self.phase)
        if self.phase == "forward":
            for node in graph_module.graph.nodes:
                sequence = node.meta.get("seq_nr")
                if sequence is None or sequence in self.provenance_state:
                    continue
                self.provenance_state[int(sequence)] = (
                    node.meta.get("roofline_module_fqn"),
                    _node_source_locations(node),
                )
        else:
            for node in graph_module.graph.nodes:
                sequence = node.meta.get("seq_nr")
                provenance = (
                    self.provenance_state.get(int(sequence)) if sequence is not None else None
                )
                if provenance is None:
                    continue
                module_fqn, source_locations = provenance
                node.meta["roofline_module_fqn"] = module_fqn
                node.meta["roofline_source_locations"] = source_locations
                node.meta["roofline_phase"] = "backward"
        suffix = "" if not self.analyses else f"_{len(self.analyses)}"
        analysis = analyze_fx_graph(
            graph_module,
            region_prefix=f"{self.phase}{suffix}_region",
        )
        self.analyses.append(analysis)
        return make_boxed_func(FxFusionInterpreter(graph_module).run)


def _clear_gradients(module: torch.nn.Module, values: Any) -> None:
    """Clear parameter and leaf-input gradients between AOTAutograd iterations."""
    module.zero_grad(set_to_none=True)
    for value in _pytree.tree_leaves(values):
        if isinstance(value, torch.Tensor) and value.is_leaf and value.grad is not None:
            value.grad = None


def _clone_warmup_values(values: Any) -> Any:
    """Clone representative tensors while preserving duplicate/view aliasing."""
    flat_values, tree_spec = _pytree.tree_flatten(values)
    storage_clones: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}
    object_clones: dict[int, torch.Tensor] = {}
    cloned_values = []
    for value in flat_values:
        if not isinstance(value, torch.Tensor):
            cloned_values.append(value)
            continue
        if id(value) in object_clones:
            cloned_values.append(object_clones[id(value)])
            continue
        if value.layout != torch.strided:
            clone = value.detach().clone()
        else:
            storage = value.untyped_storage()
            storage_key = (storage.data_ptr(), value.dtype, value.device)
            backing = storage_clones.get(storage_key)
            if backing is None:
                storage_elements = storage.nbytes() // value.element_size()
                storage_view = value.as_strided((storage_elements,), (1,), 0)
                backing = storage_view.detach().clone()
                storage_clones[storage_key] = backing
            clone = backing.as_strided(value.shape, value.stride(), value.storage_offset())
        clone.requires_grad_(value.requires_grad)
        object_clones[id(value)] = clone
        cloned_values.append(clone)
    return _pytree.tree_unflatten(cloned_values, tree_spec)


def _gradient_tensors(module: torch.nn.Module, values: Any) -> tuple[torch.Tensor, ...]:
    """Return unique parameters and leaf inputs whose gradients may accumulate."""
    tensors = [*module.parameters()]
    tensors.extend(
        value
        for value in _pytree.tree_leaves(values)
        if isinstance(value, torch.Tensor) and value.is_leaf and value.requires_grad
    )
    return tuple(dict.fromkeys(tensors))


@contextmanager
def _preserve_runtime_state(
    module: torch.nn.Module,
    values: Any,
    cuda_device: torch.device | None,
) -> Iterator[None]:
    """Restore RNG, mutable buffers, and existing gradients after exploratory runs."""
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
    buffers = dict(module.named_buffers())
    buffer_snapshots = {name: buffer.detach().clone() for name, buffer in buffers.items()}
    gradient_snapshots = {
        tensor: None if tensor.grad is None else tensor.grad.detach().clone()
        for tensor in _gradient_tensors(module, values)
    }
    try:
        yield
    finally:
        with torch.no_grad():
            for name, snapshot in buffer_snapshots.items():
                buffers[name].copy_(snapshot)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_device is not None and cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, cuda_device)
        for tensor, gradient in gradient_snapshots.items():
            tensor.grad = None if gradient is None else gradient.clone()


def _capture_training_provenance(
    loss_module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    loss_selector: Callable[[Any], torch.Tensor] | None,
) -> tuple[_ProvenanceRecord, ...]:
    """Run one isolated eager iteration to collect module/source dispatcher records."""
    with ModuleTracker() as tracker, _ProvenanceMode(tracker) as mode:
        output = loss_module(*args, **kwargs)
        loss = loss_selector(output) if loss_selector is not None else output
        _scalar_loss(loss).backward()
    return tuple(mode.records)


def _scalar_loss(output: Any) -> torch.Tensor:
    """Validate that a training wrapper returns one scalar loss tensor."""
    if not isinstance(output, torch.Tensor) or output.numel() != 1:
        raise ValueError("AOT training profiling requires the module to return one scalar loss")
    return output


@dataclass(frozen=True)
class _PreparedAotTraining:
    """Captured AOTAutograd graphs plus the measured eager loss callable."""

    analysis: FxTrainingAnalysis
    run_loss: Callable[[], torch.Tensor]
    cuda_device: torch.device | None


def _prepare_aot_training(
    loss_module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None,
    loss_selector: Callable[[Any], torch.Tensor] | None,
) -> _PreparedAotTraining:
    """Capture AOTAutograd graphs while isolating warmup state from measurement."""
    from torch._functorch.aot_autograd import aot_function

    runtime_kwargs = {} if kwargs is None else dict(kwargs)
    named_parameters = dict(loss_module.named_parameters())
    named_buffers = dict(loss_module.named_buffers())
    runtime_values = (args, runtime_kwargs)
    cuda_device = _cuda_device(
        (runtime_values, tuple(named_parameters.values()), tuple(named_buffers.values()))
    )
    provenance_args, provenance_kwargs = _clone_warmup_values(runtime_values)
    with _preserve_runtime_state(loss_module, runtime_values, cuda_device):
        _clear_gradients(loss_module, (provenance_args, provenance_kwargs))
        provenance_records = _capture_training_provenance(
            loss_module,
            provenance_args,
            provenance_kwargs,
            loss_selector,
        )
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)

    provenance_state: dict[int, tuple[str | None, tuple[str, ...]]] = {}
    forward_capture = _AotGraphCapture(
        "forward",
        provenance_records,
        provenance_state,
    )
    backward_capture = _AotGraphCapture(
        "backward",
        provenance_records,
        provenance_state,
    )

    def functional_call(
        parameters: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        *runtime_args: Any,
        **call_kwargs: Any,
    ) -> Any:
        output = torch.func.functional_call(
            loss_module,
            (parameters, buffers),
            runtime_args,
            call_kwargs,
        )
        return loss_selector(output) if loss_selector is not None else output

    aot_loss = aot_function(
        functional_call,
        fw_compiler=forward_capture,
        bw_compiler=backward_capture,
        num_params_buffers=len(named_parameters) + len(named_buffers),
    )

    def run_loss(call_args: tuple[Any, ...], call_kwargs: Mapping[str, Any]) -> torch.Tensor:
        return _scalar_loss(
            aot_loss(
                named_parameters,
                named_buffers,
                *call_args,
                **call_kwargs,
            )
        )

    warmup_args, warmup_kwargs = _clone_warmup_values(runtime_values)
    with _preserve_runtime_state(loss_module, runtime_values, cuda_device):
        _clear_gradients(loss_module, (warmup_args, warmup_kwargs))
        run_loss(warmup_args, warmup_kwargs).backward()
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)

    analysis = FxTrainingAnalysis(
        forward=tuple(forward_capture.analyses),
        backward=tuple(backward_capture.analyses),
    )
    if not analysis.forward or not analysis.backward:
        raise RuntimeError("AOTAutograd did not capture both forward and backward graphs")
    return _PreparedAotTraining(
        analysis=analysis,
        run_loss=lambda: run_loss(args, runtime_kwargs),
        cuda_device=cuda_device,
    )


def analyze_aot_training(
    loss_module: torch.nn.Module,
    args: tuple[Any, ...],
    *,
    kwargs: Mapping[str, Any] | None = None,
    loss_selector: Callable[[Any], torch.Tensor] | None = None,
) -> FxTrainingAnalysis:
    """Capture and execute eager forward/backward FX graphs without profiling."""
    prepared = _prepare_aot_training(loss_module, args, kwargs, loss_selector)
    prepared.run_loss().backward()
    if prepared.cuda_device is not None:
        torch.cuda.synchronize(prepared.cuda_device)
    return prepared.analysis


def profile_aot_training(
    loss_module: torch.nn.Module,
    args: tuple[Any, ...],
    path: str | Path,
    *,
    kwargs: Mapping[str, Any] | None = None,
    loss_selector: Callable[[Any], torch.Tensor] | None = None,
    trace_format: TraceFormat = "track_event",
    roofline_spec: RooflineSpec | None = None,
) -> FxTrainingProfile:
    """Profile eager AOTAutograd forward/backward graphs without Inductor.

    ``loss_module`` must return one scalar loss tensor, or ``loss_selector`` must
    select one from its output. A warm-up iteration captures and caches the
    AOTAutograd graphs outside the profiler; the measured iteration then executes
    the captured FX graphs eagerly with node annotations. Optimizer execution is
    intentionally outside this API.
    """
    prepared = _prepare_aot_training(loss_module, args, kwargs, loss_selector)
    activities = [ProfilerActivity.CPU]
    if prepared.cuda_device is not None:
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True) as torch_profiler:
        prepared.run_loss().backward()
        if prepared.cuda_device is not None:
            torch.cuda.synchronize(prepared.cuda_device)

    trace_path, findings_path = _write_annotated_trace(
        torch_profiler,
        path,
        (*prepared.analysis.forward, *prepared.analysis.backward),
        trace_format,
        roofline_spec,
    )
    return FxTrainingProfile(
        analysis=prepared.analysis,
        trace_path=trace_path,
        findings_path=findings_path,
    )
