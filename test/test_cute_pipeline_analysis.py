"""Host-only tests for declarative CuTeDSL pipeline analysis and Perfetto export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

import transformer_nuggets.cute.profiler.pipeline.annotations as pipeline_annotations
from transformer_nuggets.cute.profiler.pipeline import (
    Dependency,
    MeasuredCapture,
    MeasuredRegion,
    PipelineAnnotations,
    PipelinePlan,
    Region,
    Resource,
    Role,
    analyze_pipeline,
    extract_plan,
    load_iket_capture,
    write_pipeline_perfetto,
)
from transformer_nuggets.utils.merge_traces import merge_traces


ANNOTATED_SOURCE = """
PIPELINE = PipelineAnnotations("demo", iteration_name="chunk")
PIPELINE.resource(
    "x", label="X ring", depth=1, storage="SMEM", description="staged value"
)
PIPELINE.resource(
    "state", label="State", depth=1, storage="registers",
    description="recurrent state", kind="state"
)

class Kernel:
    @PIPELINE.role("load", label="Load", warp_start=0, warp_end=1, color="#00f")
    def run_load(self):
        PIPELINE.region(
            "load.x", label="Load X", weight=1, description="load", produces=("x",)
        )
        PIPELINE.iteration_end()

    @PIPELINE.role("compute", label="Compute", warp_start=1, warp_end=2, color="#0f0")
    def run_compute(self):
        PIPELINE.region(
            "compute.x", label="Consume X", weight=3, description="compute",
            consumes=("x", "state"), produces=("state@1",), releases=("x",)
        )
        PIPELINE.iteration_end()
"""


def toy_plan() -> PipelinePlan:
    """Build a two-role depth-one recurrent pipeline with a known cycle ratio."""
    plan = PipelinePlan("toy", iteration_name="chunk")
    plan.add_role(Role("load", "Load", 0, 1, "#00f"))
    plan.add_role(Role("compute", "Compute", 1, 2, "#0f0"))
    plan.add_resource(Resource("x", "X ring", 1, "SMEM", "staged value"))
    plan.add_region(Region("load.x", "load", "Load X", 0, 1, "load"))
    plan.add_region(Region("compute.x", "compute", "Consume X", 0, 3, "compute"))
    plan.add_dependency(Dependency("load.x", "compute.x", "x"))
    plan.add_dependency(Dependency("compute.x", "load.x", "x", distance=1, kind="reuse"))
    return plan


def logical_capture(timeline, *, tick_ns: int = 100) -> MeasuredCapture:
    """Create synthetic measurements for host-only scheduling tests."""
    regions = tuple(
        MeasuredRegion(
            item.region.name,
            item.region.role,
            item.iteration,
            item.start * tick_ns,
            item.end * tick_ns,
            float((item.end - item.start) * tick_ns),
            1,
        )
        for item in timeline.regions
    )
    return MeasuredCapture(
        "logical test capture",
        timeline.plan_name,
        (0, 0, 0),
        0,
        timeline.logical_duration * tick_ns,
        regions,
    )


def test_annotations_extract_data_state_and_reuse(tmp_path: Path) -> None:
    source = tmp_path / "annotated.py"
    source.write_text(ANNOTATED_SOURCE)

    plan = extract_plan(source)

    assert list(plan.roles) == ["load", "compute"]
    assert {(edge.resource, edge.kind, edge.distance) for edge in plan.dependencies} == {
        ("x", "data", 0),
        ("state", "state", 1),
        ("x", "reuse", 1),
    }
    assert plan.schedule(iterations=3).logical_duration > 0


def test_runtime_annotations_are_noops_until_iket_is_enabled() -> None:
    annotations = PipelineAnnotations("test", iteration_name="tile")

    @annotations.role("load", label="Load", warp_start=0, warp_end=1, color="#fff")
    def body(value: int) -> int:
        annotations.region("work", label="Work", weight=1, description="work")
        return value + 1

    assert body(2) == 3
    assert (
        annotations.resource("x", label="X", depth=1, storage="SMEM", description="value") is None
    )


def test_iket_unavailable_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    annotations = PipelineAnnotations("test", iteration_name="tile")
    annotations.enable_iket()

    def unavailable():
        raise RuntimeError("IKET pipeline ranges require a supported CuTeDSL release")

    monkeypatch.setattr(pipeline_annotations, "load_iket", unavailable)

    @annotations.role("load", label="Load", warp_start=0, warp_end=1, color="#fff")
    def body() -> None:
        annotations.region("work", label="Work", weight=1, description="work")

    with pytest.raises(RuntimeError, match="IKET pipeline ranges require"):
        body()


def test_scheduler_rejects_same_iteration_cycle() -> None:
    plan = PipelinePlan("cycle", iteration_name="tile")
    plan.add_role(Role("a", "A", 0, 1, "#fff"))
    plan.add_role(Role("b", "B", 1, 2, "#fff"))
    plan.add_resource(Resource("x", "X", 1, "register", "value"))
    plan.add_region(Region("a.work", "a", "A", 0, 1, "work"))
    plan.add_region(Region("b.work", "b", "B", 0, 1, "work"))
    plan.add_dependency(Dependency("a.work", "b.work", "x"))
    plan.add_dependency(Dependency("b.work", "a.work", "x"))

    with pytest.raises(ValueError, match="cycle"):
        plan.schedule(iterations=1)


def test_semantic_edges_survive_role_edge_deduplication() -> None:
    plan = PipelinePlan("same-role", iteration_name="tile")
    plan.add_role(Role("role", "Role", 0, 1, "#fff"))
    plan.add_resource(Resource("x", "X", 1, "SMEM", "x"))
    plan.add_resource(Resource("y", "Y", 1, "SMEM", "y"))
    plan.add_region(Region("a", "role", "A", 0, 1, "a"))
    plan.add_region(Region("b", "role", "B", 1, 1, "b"))
    plan.add_dependency(Dependency("a", "b", "x"))
    plan.add_dependency(Dependency("a", "b", "y"))

    timeline = plan.schedule(iterations=1)

    assert [item.dependency.resource for item in timeline.dependencies] == ["x", "y"]


def test_iket_json_join(tmp_path: Path) -> None:
    timeline = toy_plan().schedule(iterations=2)
    names = ["load.x", "compute.x"]
    ranges = []
    locations = [
        {"ctaId": [0, 0, 0], "warpId": 0},
        {"ctaId": [0, 0, 0], "warpId": 1},
    ]
    for item in timeline.regions:
        ranges.append(
            {
                "startTs": 1_000 + item.start * 10,
                "endTs": 1_000 + item.end * 10,
                "rangeNameIdx": names.index(item.region.name),
                "warpLocIdxs": [0 if item.region.role == "load" else 1] * 2,
            }
        )
    path = tmp_path / "iket.json"
    path.write_text(
        json.dumps(
            {
                "stringTable": names,
                "locationTable": locations,
                "launches": [{"kernelName": "toy", "ranges": ranges}],
            }
        )
    )

    measured = load_iket_capture(path, timeline)

    assert len(measured.regions) == len(timeline.regions)
    assert measured.duration_ns == timeline.logical_duration * 10


def test_analysis_and_perfetto_direct_flows(tmp_path: Path) -> None:
    timeline = toy_plan().schedule(iterations=4)
    measured = logical_capture(timeline)
    analysis = analyze_pipeline(timeline, measured)

    assert analysis.logical_critical_path.makespan == timeline.logical_duration
    assert analysis.logical_critical_cycle.initiation_interval == 4
    assert analysis.logical_critical_cycle.resource_initiation_interval == 3
    ring = next(item for item in analysis.rings if item.resource == "x")
    assert ring.peak_occupancy <= ring.depth
    assert ring.final_occupancy == 0

    output = write_pipeline_perfetto(
        tmp_path / "pipeline.pftrace",
        timeline,
        measured,
        analysis=analysis,
    )
    trace = Trace()
    trace.ParseFromString(output.read_bytes())
    begins = [
        packet.track_event
        for packet in trace.packet
        if packet.HasField("track_event")
        and packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN
    ]
    outgoing = [flow_id for event in begins for flow_id in event.flow_ids]
    incoming = [flow_id for event in begins for flow_id in event.terminating_flow_ids]
    assert len(begins) == len(timeline.regions)
    assert sorted(outgoing) == sorted(incoming)
    assert len(outgoing) == len(timeline.dependencies)
    assert any(event.flow_ids and event.terminating_flow_ids for event in begins)
    assert any(
        packet.HasField("track_event") and packet.track_event.type == TrackEvent.TYPE_COUNTER
        for packet in trace.packet
    )

    depth_by_track: dict[int, int] = {}
    for packet in trace.packet:
        if not packet.HasField("track_event"):
            continue
        event = packet.track_event
        if event.type == TrackEvent.TYPE_SLICE_BEGIN:
            depth_by_track[event.track_uuid] = depth_by_track.get(event.track_uuid, 0) + 1
            assert depth_by_track[event.track_uuid] == 1
        elif event.type == TrackEvent.TYPE_SLICE_END:
            assert depth_by_track.get(event.track_uuid, 0) == 1
            depth_by_track[event.track_uuid] = 0
    assert not any(depth_by_track.values())

    merged_path = tmp_path / "merged.pftrace"
    merge_traces(
        [str(output), str(output)],
        str(merged_path),
        labels=["first", "second"],
    )
    merged = Trace()
    merged.ParseFromString(merged_path.read_bytes())
    process_names = {
        packet.track_descriptor.process.process_name
        for packet in merged.packet
        if packet.HasField("track_descriptor") and packet.track_descriptor.HasField("process")
    }
    assert any(name.startswith("first ·") for name in process_names)
    assert any(name.startswith("second ·") for name in process_names)


def test_perfetto_splits_crossing_role_envelopes_and_normalizes_suffix(
    tmp_path: Path,
) -> None:
    plan = PipelinePlan("crossing", iteration_name="tile")
    plan.add_role(Role("role", "Role", 0, 1, "#fff"))
    plan.add_region(Region("a", "role", "A", 0, 1, "a"))
    plan.add_region(Region("b", "role", "B", 1, 1, "b"))
    timeline = plan.schedule(iterations=1)
    measured = MeasuredCapture(
        "measured",
        "crossing",
        (0, 0, 0),
        0,
        15,
        (
            MeasuredRegion("a", "role", 0, 0, 10, 10.0, 1),
            MeasuredRegion("b", "role", 0, 5, 15, 10.0, 1),
        ),
    )

    output = write_pipeline_perfetto(tmp_path / "crossing.json", timeline, measured)
    trace = Trace()
    trace.ParseFromString(output.read_bytes())
    descriptors = {
        packet.track_descriptor.uuid: packet.track_descriptor
        for packet in trace.packet
        if packet.HasField("track_descriptor")
    }
    begins = [
        packet.track_event
        for packet in trace.packet
        if packet.HasField("track_event")
        and packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN
    ]

    assert output.suffix == ".pftrace"
    assert len({event.track_uuid for event in begins}) == 2
    assert all(descriptors[event.track_uuid].sibling_merge_behavior for event in begins)
    assert any(descriptor.HasField("process") for descriptor in descriptors.values())
