import copy
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from transformer_nuggets.fx_analysis import (
    FX_FUSION_KERNEL_MARKER,
    FX_FUSION_MARKER,
    FxFusionInterpreter,
    RooflineSpec,
    RooflineWork,
    add_fx_fusion_region_boxes,
    analyze_aot_training,
    analyze_fx_fusion,
    analyze_fx_graph,
    profile_aot_training,
    profile_fx_fusion,
    rank_trace_findings,
    register_roofline_formula,
)


def pointwise_chain(x, y):
    product = x * y
    gate = torch.sigmoid(product)
    return product * gate


@torch.library.custom_op("transformer_nuggets_test::opaque_sin", mutates_args=())
def opaque_sin(x: torch.Tensor) -> torch.Tensor:
    """Test-only opaque operation backed by an eager PyTorch kernel."""
    return torch.sin(x)


@opaque_sin.register_fake
def opaque_sin_fake(x: torch.Tensor) -> torch.Tensor:
    """Return fake metadata for the test-only opaque operation."""
    return torch.empty_like(x)


def test_analyze_fx_fusion_finds_connected_pointwise_region():
    result = analyze_fx_fusion(pointwise_chain, (torch.randn(4), torch.randn(4)))

    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.node_names == ("mul", "sigmoid", "mul_1")
    assert region.op_names == (
        "aten.mul.Tensor",
        "aten.sigmoid.default",
        "aten.mul.Tensor",
    )
    assert region.input_bytes == 32
    assert region.output_bytes == 16
    assert region.minimum_avoidable_bytes == 80
    assert region.ideal_bytes == 48
    assert region.eager_minimum_bytes == 128
    call_nodes = [node for node in result.nodes if node.model_kind != "metadata"]
    assert [node.logical_flops for node in call_nodes] == [4, 4, 4]
    assert [node.logical_bytes for node in call_nodes] == [48, 32, 48]
    assert all(node.arithmetic_intensity is not None for node in call_nodes)
    assert all(node.phase == "forward" for node in call_nodes)
    assert any(node.source_locations for node in call_nodes)
    assert all(
        node.meta.get("fusion_region_id") == region.region_id
        for node in result.graph_module.graph.nodes
        if node.name in region.node_names
    )


def test_analyze_fx_fusion_keeps_opaque_ops_as_attributed_barriers():
    def with_opaque_op(x):
        opaque = opaque_sin(x)
        return opaque * torch.sigmoid(opaque)

    result = analyze_fx_fusion(with_opaque_op, (torch.randn(4),))

    opaque_node = next(node for node in result.nodes if "opaque_sin" in node.op_name)
    assert opaque_node.model_kind == "generic_io"
    assert opaque_node.logical_flops is None
    assert opaque_node.logical_bytes == 32
    assert len(result.regions) == 1
    assert result.regions[0].op_names == (
        "aten.sigmoid.default",
        "aten.mul.Tensor",
    )

    @register_roofline_formula(torch.ops.transformer_nuggets_test.opaque_sin.default)
    def opaque_sin_work(inputs, output, kwargs):
        del kwargs
        result = output[0]
        return RooflineWork(
            logical_flops=4 * result.numel,
            read_bytes=inputs[0].nbytes,
            write_bytes=result.nbytes,
            model_kind="opaque_sin",
            confidence="high",
        )

    modeled = analyze_fx_fusion(with_opaque_op, (torch.randn(4),))
    opaque_node = next(node for node in modeled.nodes if "opaque_sin" in node.op_name)
    assert opaque_node.model_kind == "opaque_sin"
    assert opaque_node.model_confidence == "high"
    assert opaque_node.logical_flops == 16
    assert opaque_node.logical_bytes == 32


def test_analyze_fx_fusion_includes_reduction_regions():
    def rms_norm_core(x):
        variance = x.square().mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + 1e-6)

    result = analyze_fx_fusion(rms_norm_core, (torch.randn(2, 4),))

    assert len(result.regions) == 1
    assert result.regions[0].op_names == (
        "aten.pow.Tensor_Scalar",
        "aten.mean.dim",
        "aten.add.Tensor",
        "aten.rsqrt.default",
        "aten.mul.Tensor",
    )


def test_analyze_fx_fusion_keeps_size_preserving_views_in_region():
    def with_view(x):
        activated = torch.sigmoid(x)
        return activated.view(2, 2) * 2

    result = analyze_fx_fusion(with_view, (torch.randn(4),))

    assert len(result.regions) == 1
    assert result.regions[0].op_names == (
        "aten.sigmoid.default",
        "aten.mul.Tensor",
    )
    assert "view" in result.regions[0].node_names


def test_analyze_fx_fusion_deduplicates_alias_outputs():
    def with_alias_outputs(x):
        activated = torch.sigmoid(x)
        return activated, activated.detach(), activated * 2

    result = analyze_fx_fusion(with_alias_outputs, (torch.randn(4),))

    region = result.regions[0]
    assert region.output_bytes == 32
    assert region.minimum_avoidable_bytes == 16


def test_analyze_fx_fusion_tracks_expand_consumer_bytes():
    def with_expand(x):
        activated = torch.sigmoid(x)
        return activated.expand(2, 4) * 2

    result = analyze_fx_fusion(with_expand, (torch.randn(1, 4),))

    region = result.regions[0]
    assert "expand" in region.node_names
    assert region.minimum_avoidable_bytes == 48


def test_analyze_fx_fusion_supports_module_tensor_state():
    class GatedScale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.randn(()))

        def forward(self, x):
            product = x * self.scale
            return product * torch.sigmoid(product)

    result = analyze_fx_fusion(GatedScale(), (torch.randn(4),))

    assert len(result.regions) == 1
    assert len(result.regions[0].op_names) == 3
    assert result.regions[0].module_fqn == "GatedScale"
    assert result.regions[0].source_locations


def test_analyze_fx_fusion_counts_reads_of_escaping_intermediate():
    def with_escaping_intermediate(x, y):
        product = x * y
        return product, torch.sigmoid(product)

    result = analyze_fx_fusion(
        with_escaping_intermediate,
        (torch.randn(4), torch.randn(4)),
    )

    region = result.regions[0]
    assert region.output_bytes == 32
    assert region.minimum_avoidable_bytes == 16


def test_analyze_fx_fusion_does_not_horizontally_merge_independent_branch():
    def with_independent_branch(x, y):
        product = x * y
        gated = product * torch.sigmoid(product)
        independent = x + 1
        return gated, independent

    result = analyze_fx_fusion(with_independent_branch, (torch.randn(4), torch.randn(4)))

    assert len(result.regions) == 1
    assert result.regions[0].node_names == ("mul", "sigmoid", "mul_1")


def test_analyze_fx_graph_reports_unbacked_dynamic_shapes_clearly():
    def data_dependent(x):
        indices = torch.nonzero(x)
        return indices * 2

    graph = make_fx(data_dependent, tracing_mode="symbolic")(torch.tensor([0, 1, 0, 1]))

    with pytest.raises(ValueError, match="requires static tensor shapes"):
        analyze_fx_graph(graph)


def test_analyze_fx_fusion_random_pointwise_dag_invariants():
    for seed in range(10):
        choices = [
            random.Random(seed).choice(("add", "mul", "sigmoid", "transpose")) for _ in range(8)
        ]

        def random_dag(x, y):
            value = x
            for choice in choices:
                if choice == "add":
                    value = value + y
                elif choice == "mul":
                    value = value * y
                elif choice == "sigmoid":
                    value = torch.sigmoid(value)
                else:
                    value = value.transpose(0, 1).transpose(0, 1)
            return value

        result = analyze_fx_fusion(random_dag, (torch.randn(4, 8), torch.randn(4, 8)))

        assert all(region.input_bytes >= 0 for region in result.regions)
        assert all(region.output_bytes >= 0 for region in result.regions)
        assert all(region.minimum_avoidable_bytes >= 0 for region in result.regions)
        assert all(region.eager_minimum_bytes >= region.ideal_bytes for region in result.regions)
        assert all(node.logical_bytes >= 0 for node in result.nodes)


def test_analyze_fx_fusion_assigns_separate_connected_regions():
    def two_chains(x, y):
        product = x * y
        gated = product * torch.sigmoid(product)
        shifted = x + y
        activated = shifted * torch.relu(shifted)
        return gated, activated

    result = analyze_fx_fusion(two_chains, (torch.randn(4), torch.randn(4)))

    assert [region.region_id for region in result.regions] == ["region_000", "region_001"]
    assert [len(region.op_names) for region in result.regions] == [3, 3]


def test_add_fx_fusion_region_boxes_combines_consecutive_region_kernels():
    analysis = analyze_fx_fusion(pointwise_chain, (torch.randn(4), torch.randn(4)))
    region = analysis.regions[0]
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "mul_kernel",
                "pid": 0,
                "tid": 7,
                "ts": 10,
                "dur": 2,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "fx_fusion::region_000::mul",
                "pid": 0,
                "tid": 7,
                "ts": 10,
                "dur": 2,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "sigmoid_kernel",
                "pid": 0,
                "tid": 7,
                "ts": 20,
                "dur": 3,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "fx_fusion::region_000::sigmoid",
                "pid": 0,
                "tid": 7,
                "ts": 20,
                "dur": 3,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "final_kernel",
                "pid": 0,
                "tid": 7,
                "ts": 30,
                "dur": 4,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "fx_fusion::region_000::mul_1",
                "pid": 0,
                "tid": 7,
                "ts": 30,
                "dur": 4,
                "args": {},
            },
        ]
    }

    processed = add_fx_fusion_region_boxes(
        trace,
        [region],
        nodes=analysis.nodes,
        roofline_spec=RooflineSpec(
            name="test-gpu",
            peak_compute_tflops=1.0,
            peak_memory_gbps=1.0,
            launch_latency_us=1.0,
        ),
    )
    boxes = [
        event for event in processed["traceEvents"] if event.get("args", {}).get(FX_FUSION_MARKER)
    ]
    kernels = [
        event
        for event in processed["traceEvents"]
        if event.get("args", {}).get(FX_FUSION_KERNEL_MARKER)
    ]
    node_boxes = [
        event
        for event in processed["traceEvents"]
        if event.get("cat") == "gpu_user_annotation" and event.get("args", {}).get("node_id")
    ]

    assert len(boxes) == 1
    assert len(kernels) == 3
    assert len(node_boxes) == 3
    assert boxes[0]["ts"] == 10
    assert boxes[0]["dur"] == 24
    assert boxes[0]["name"] == (
        "🟡 gated activation [region_000]: 3→1 launches · 0.0 MiB saved · 62% traffic"
    )
    assert boxes[0]["cname"] == "yellow"
    assert boxes[0]["args"]["severity"] == "medium"
    assert boxes[0]["args"]["operation_count"] == 3
    assert boxes[0]["args"]["observed_kernel_count"] == 3
    assert boxes[0]["args"]["observed_gpu_launch_count"] == 3
    assert boxes[0]["args"]["expected_kernel_count"] == 1
    assert boxes[0]["args"]["expected_gpu_launch_count"] == 1
    assert boxes[0]["args"]["expected_kernel_savings"] == 2
    assert boxes[0]["args"]["expected_gpu_launch_savings"] == 2
    assert boxes[0]["args"]["segment_index"] == 1
    assert boxes[0]["args"]["segment_count"] == 1
    assert boxes[0]["args"]["segment_kernel_count"] == 3
    assert boxes[0]["args"]["segment_gpu_launch_count"] == 3
    assert boxes[0]["args"]["segment_metrics"] == "region_total"
    assert boxes[0]["args"]["expected_read_bytes"] == 32
    assert boxes[0]["args"]["expected_write_bytes"] == 16
    assert boxes[0]["args"]["avoidable_intermediate_read_bytes"] == 48
    assert boxes[0]["args"]["avoidable_intermediate_write_bytes"] == 32
    assert boxes[0]["args"]["minimum_avoidable_bytes"] == 80
    assert boxes[0]["args"]["traffic_reduction_percent"] == 62.5
    assert boxes[0]["args"]["idealized_bandwidth_speedup"] == pytest.approx(128 / 48)
    assert boxes[0]["args"]["logical_flops"] == 12
    assert boxes[0]["args"]["observed_region_wall_us"] == 24
    assert boxes[0]["args"]["observed_gpu_busy_us"] == 9
    assert boxes[0]["args"]["observed_inter_kernel_gap_us"] == 15
    assert boxes[0]["args"]["predicted_bound"] == "memory"
    assert boxes[0]["args"]["achieved_roofline_percent"] == pytest.approx(100 * 0.128 / 24)
    assert boxes[0]["args"]["fused_roofline_floor_us"] == 1
    assert boxes[0]["args"]["idealized_recoverable_us"] == 23
    assert boxes[0]["args"]["idealized_fused_speedup_upper_bound"] == 24
    assert all(kernel["args"]["fx_fusion_region_id"] == "region_000" for kernel in kernels)
    assert [kernel["args"]["fx_fusion_kernel_index"] for kernel in kernels] == [1, 2, 3]
    assert [kernel["args"]["fx_node_logical_flops"] for kernel in kernels] == [4, 4, 4]
    assert all(kernel["args"]["fx_node_achieved_logical_tbps"] > 0 for kernel in kernels)
    assert all("TF/s" in node_box["name"] for node_box in node_boxes)
    assert all(
        kernel["args"]["fx_fusion_region_predicted_bound"] == "memory" for kernel in kernels
    )
    assert len(trace["traceEvents"]) == 6
    assert not any(
        event.get("args", {}).get(FX_FUSION_KERNEL_MARKER) for event in trace["traceEvents"]
    )
    report = rank_trace_findings(processed)
    assert report["summary"] == {
        "gpu_step_wall_us": 24,
        "region_finding_count": 1,
        "attributed_node_count": 3,
    }
    assert report["regions"][0]["rank"] == 1
    assert report["regions"][0]["priority_recoverable_us"] == 23
    assert report["regions"][0]["priority_basis"] == "supplied_roofline_fused_floor"
    assert report["nodes"][0]["rank"] == 1
    assert add_fx_fusion_region_boxes(processed, [region]) == processed


def test_add_fx_fusion_region_boxes_splits_on_unannotated_gpu_work():
    region = analyze_fx_fusion(pointwise_chain, (torch.randn(4), torch.randn(4))).regions[0]
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 10,
                "dur": 2,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "fx_fusion::region_000::mul",
                "pid": 0,
                "tid": 7,
                "ts": 10,
                "dur": 2,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 20,
                "dur": 2,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 30,
                "dur": 2,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "fx_fusion::region_000::sigmoid",
                "pid": 0,
                "tid": 7,
                "ts": 30,
                "dur": 2,
                "args": {},
            },
        ]
    }

    processed = add_fx_fusion_region_boxes(trace, [region])
    boxes = [
        event for event in processed["traceEvents"] if event.get("args", {}).get(FX_FUSION_MARKER)
    ]

    assert [(box["ts"], box["dur"]) for box in boxes] == [(10, 2), (30, 2)]
    assert [box["args"]["segment_index"] for box in boxes] == [1, 2]
    assert all(box["args"]["segment_count"] == 2 for box in boxes)
    assert all(box["args"]["observed_gpu_launch_count"] == 2 for box in boxes)
    assert all(box["args"]["segment_gpu_launch_count"] == 1 for box in boxes)


def test_analyze_aot_training_captures_forward_and_backward():
    class LossModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))

        def forward(self, x):
            projected = x @ self.weight
            return (projected * torch.sigmoid(projected)).sum()

    module = LossModule()
    reference = copy.deepcopy(module)
    x = torch.randn(2, 4)
    reference(x).backward()

    result = analyze_aot_training(module, (x,))

    assert len(result.forward) == 1
    assert len(result.backward) == 1
    assert result.forward[0].regions
    assert result.backward[0].regions
    assert any(node.module_fqn for node in result.forward[0].nodes)
    assert any(node.source_locations for node in result.forward[0].nodes)
    assert all(node.phase == "backward" for node in result.backward[0].nodes)
    assert all(
        region.region_id.startswith("forward_region_") for region in result.forward[0].regions
    )
    assert all(
        region.region_id.startswith("backward_region_") for region in result.backward[0].regions
    )
    torch.testing.assert_close(module.weight.grad, reference.weight.grad)


def test_analyze_aot_training_preserves_aliased_inputs():
    class AliasedLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

        def forward(self, left, right):
            left.add_(1)
            return ((left + right) * self.weight).sum()

    reference = AliasedLoss()
    reference_input = torch.tensor([0.0, 1.0, 2.0])
    reference(reference_input, reference_input).backward()

    module = AliasedLoss()
    aliased_input = torch.tensor([0.0, 1.0, 2.0])
    analyze_aot_training(module, (aliased_input, aliased_input))

    torch.testing.assert_close(module.weight.grad, reference.weight.grad)


def test_analyze_aot_training_accumulates_existing_gradients():
    class LossModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))

        def forward(self, x):
            return (x * self.weight).sum()

    module = LossModule()
    x = torch.ones(4, requires_grad=True)
    module.weight.grad = torch.full_like(module.weight, 7)
    x.grad = torch.full_like(x, 11)

    analyze_aot_training(module, (x,))

    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 8))
    torch.testing.assert_close(x.grad, torch.full_like(x, 12))


def test_analyze_aot_training_restores_state_on_failure():
    class InvalidLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("steps", torch.zeros(()))

        def forward(self, x):
            self.steps.add_(1)
            return x * self.weight + torch.rand_like(x)

    module = InvalidLoss()
    module.weight.grad = torch.full_like(module.weight, 5)
    rng_state = torch.random.get_rng_state()

    with pytest.raises(ValueError, match="one scalar loss"):
        analyze_aot_training(module, (torch.ones(2),))

    assert module.steps.item() == 0
    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 5))
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_analyze_aot_training_supports_nonleaf_inputs():
    class LossModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            product = x * self.weight
            return (product * torch.sigmoid(product)).sum()

    base = torch.randn(2, 4, requires_grad=True)
    result = analyze_aot_training(LossModule(), (base * 2,))

    assert result.backward
    assert base.grad is not None


def test_analyze_aot_training_restores_warmup_buffer_state():
    class BufferedLossModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4))
            self.register_buffer("steps", torch.zeros(()))

        def forward(self, x):
            self.steps.add_(1)
            product = x * self.weight
            return (product * torch.sigmoid(product)).sum()

    module = BufferedLossModule()
    analyze_aot_training(module, (torch.randn(2, 4),))

    assert module.steps.item() == 1


def test_analyze_aot_training_supports_tied_parameters():
    class TiedLossModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.projection = torch.nn.Linear(4, 16, bias=False)
            self.projection.weight = self.embedding.weight

        def forward(self, token_ids):
            hidden = self.embedding(token_ids)
            return self.projection(hidden).square().mean()

    module = TiedLossModule()
    result = analyze_aot_training(module, (torch.randint(0, 16, (2, 3)),))

    assert result.forward
    assert result.backward
    assert module.embedding.weight is module.projection.weight
    assert module.embedding.weight.grad is not None


def test_analyze_aot_training_supports_kwargs_and_loss_selector():
    class StructuredLossModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4))

        def forward(self, x, *, scale):
            product = x * self.weight * scale
            return {"loss": (product * torch.sigmoid(product)).sum()}

    result = analyze_aot_training(
        StructuredLossModule(),
        (torch.randn(2, 4),),
        kwargs={"scale": torch.tensor(0.5)},
        loss_selector=lambda output: output["loss"],
    )

    assert result.forward
    assert result.backward


def test_fx_fusion_interpreter_preserves_eager_result():
    args = (torch.randn(4), torch.randn(4))
    analysis = analyze_fx_fusion(pointwise_chain, args)

    expected = pointwise_chain(*args)
    actual = FxFusionInterpreter(analysis.graph_module).run(*args)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for profiler integration"
)
def test_profile_aot_training_writes_forward_and_backward_boxes(tmp_path):
    class LossModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(32, device="cuda"))

        def forward(self, x):
            product = x * self.weight
            return (product * torch.sigmoid(product)).sum()

    result = profile_aot_training(
        LossModule(),
        (torch.randn(8, 32, device="cuda"),),
        tmp_path / "training.pftrace",
    )

    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace

    trace = Trace()
    trace.ParseFromString(result.trace_path.read_bytes())
    names = [
        packet.track_event.name
        for packet in trace.packet
        if packet.HasField("track_event")
        and any(
            annotation.name == FX_FUSION_MARKER and annotation.bool_value
            for annotation in packet.track_event.debug_annotations
        )
    ]
    assert any("forward_region_" in name for name in names)
    assert any("backward_region_" in name for name in names)
    assert result.findings_path.exists()
    report = json.loads(result.findings_path.read_text())
    assert all(
        all(Path(path).exists() for path in finding["followups"].values())
        for finding in report["regions"]
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for profiler integration"
)
def test_profile_fx_fusion_writes_annotated_trace(tmp_path):
    args = (
        torch.randn(1024, device="cuda"),
        torch.randn(1024, device="cuda"),
    )

    result = profile_fx_fusion(
        pointwise_chain,
        args,
        tmp_path / "pointwise.pftrace",
        roofline_spec=RooflineSpec(
            name="test-gpu",
            peak_compute_tflops=1.0,
            peak_memory_gbps=1.0,
        ),
    )

    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace

    trace = Trace()
    trace.ParseFromString(result.trace_path.read_bytes())
    boxes = [
        packet.track_event
        for packet in trace.packet
        if packet.HasField("track_event")
        and any(
            annotation.name == FX_FUSION_MARKER and annotation.bool_value
            for annotation in packet.track_event.debug_annotations
        )
    ]
    assert len(result.analysis.regions) == 1
    assert len(boxes) == 1
    annotations = {annotation.name: annotation for annotation in boxes[0].debug_annotations}
    assert annotations["severity"].string_value == "medium"
    assert annotations["operation_count"].int_value == 3
    assert annotations["observed_kernel_count"].int_value == 3
    assert annotations["observed_gpu_launch_count"].int_value == 3
    assert annotations["expected_kernel_count"].int_value == 1
    assert annotations["roofline_spec"].string_value == "test-gpu"
    assert annotations["achieved_roofline_percent"].double_value > 0
    assert annotations[FX_FUSION_MARKER].bool_value
    report = json.loads(result.findings_path.read_text())
    assert report["summary"]["region_finding_count"] == 1
    assert report["summary"]["attributed_node_count"] == 3
    followups = report["regions"][0]["followups"]
    assert all(Path(path).exists() for path in followups.values())
    replay = subprocess.run(
        [sys.executable, followups["replay_script"], "--device", "cpu"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert replay.returncode == 0, replay.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for profiling")
def test_followup_replay_preserves_strided_offset_input(tmp_path):
    def strided_chain(x):
        view = x[:, 1::2]
        product = view * 2
        return product * torch.sigmoid(product)

    result = profile_fx_fusion(
        strided_chain,
        (torch.randn(8, 16, device="cuda"),),
        tmp_path / "strided.pftrace",
    )
    report = json.loads(result.findings_path.read_text())
    replay_script = report["regions"][0]["followups"]["replay_script"]
    replay = subprocess.run(
        [sys.executable, replay_script, "--device", "cpu"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert replay.returncode == 0, replay.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for profiling")
def test_profile_fx_fusion_commits_module_state_once(tmp_path):
    class StatefulPointwise(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("steps", torch.zeros((), device="cuda"))

        def forward(self, x, y):
            self.steps.add_(1)
            product = x * y
            return product * torch.sigmoid(product)

    module = StatefulPointwise()
    profile_fx_fusion(
        module,
        (torch.randn(32, device="cuda"), torch.randn(32, device="cuda")),
        tmp_path / "stateful.pftrace",
    )

    assert module.steps.item() == 1


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for CUDA Graph profiling"
)
def test_profile_fx_fusion_supports_cuda_graph_replay(tmp_path):
    from torch.cuda.graph_annotations import is_available

    if not is_available():
        pytest.skip("CUDA Graph kernel annotations are unavailable")

    class StatefulPointwise(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("steps", torch.zeros((), device="cuda"))

        def forward(self, x, y):
            self.steps.add_(1)
            product = x * y
            return product * torch.sigmoid(product)

    module = StatefulPointwise()
    args = (
        torch.randn(1024, device="cuda"),
        torch.randn(1024, device="cuda"),
    )

    result = profile_fx_fusion(
        module,
        args,
        tmp_path / "pointwise_graph.pftrace",
        cuda_graph=True,
    )

    report = json.loads(result.findings_path.read_text())
    assert report["summary"]["region_finding_count"] == 1
    assert report["summary"]["attributed_node_count"] == 4
    assert module.steps.item() == 1
