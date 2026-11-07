"""Host-only tests for declarative CuTeDSL pipeline analysis and Perfetto export."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")  # transformer_nuggets.cute imports CuTeDSL
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
    report,
    write_pipeline_perfetto,
)
from transformer_nuggets.cute.profiler.pipeline.analysis import ring_occupancy_samples
from transformer_nuggets.utils.merge_traces import merge_traces


ANNOTATED_SOURCE = """
PIPELINE = PipelineAnnotations("demo", iteration_name="chunk")
PIPELINE.resource(
    "x", label="X ring", depth=1
)
PIPELINE.resource(
    "state", label="State", depth=1, kind="state"
)

class Kernel:
    @PIPELINE.role("load", label="Load", warp_start=0, warp_end=1)
    def run_load(self):
        PIPELINE.region(
            "load.x", label="Load X", weight=1, produces=("x",)
        )
        PIPELINE.iteration_end()

    @PIPELINE.role("compute", label="Compute", warp_start=1, warp_end=2)
    def run_compute(self):
        PIPELINE.region(
            "compute.x", label="Consume X", weight=3,
            consumes=("x", "state"), produces=("state@1",), releases=("x",)
        )
        PIPELINE.iteration_end()
"""


def toy_plan() -> PipelinePlan:
    """Build a two-role depth-one recurrent pipeline with a known cycle ratio."""
    plan = PipelinePlan("toy", iteration_name="chunk")
    plan.add_role(Role("load", "Load", 0, 1))
    plan.add_role(Role("compute", "Compute", 1, 2))
    plan.add_resource(Resource("x", "X ring", 1))
    plan.add_region(Region("load.x", "load", "Load X", 0, 1))
    plan.add_region(Region("compute.x", "compute", "Consume X", 0, 3))
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

    @annotations.role("load", label="Load", warp_start=0, warp_end=1)
    def body(value: int) -> int:
        annotations.region("work", label="Work", weight=1)
        return value + 1

    assert body(2) == 3
    assert annotations.resource("x", label="X", depth=1) is None


def test_scheduler_rejects_same_iteration_cycle() -> None:
    plan = PipelinePlan("cycle", iteration_name="tile")
    plan.add_role(Role("a", "A", 0, 1))
    plan.add_role(Role("b", "B", 1, 2))
    plan.add_resource(Resource("x", "X", 1))
    plan.add_region(Region("a.work", "a", "A", 0, 1))
    plan.add_region(Region("b.work", "b", "B", 0, 1))
    plan.add_dependency(Dependency("a.work", "b.work", "x"))
    plan.add_dependency(Dependency("b.work", "a.work", "x"))

    with pytest.raises(ValueError, match="cycle"):
        plan.schedule(iterations=1)


def test_semantic_edges_survive_role_edge_deduplication() -> None:
    plan = PipelinePlan("same-role", iteration_name="tile")
    plan.add_role(Role("role", "Role", 0, 1))
    plan.add_resource(Resource("x", "X", 1))
    plan.add_resource(Resource("y", "Y", 1))
    plan.add_region(Region("a", "role", "A", 0, 1))
    plan.add_region(Region("b", "role", "B", 1, 1))
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


def write_iket(path: Path, ranges: list[tuple[str, int, int, int]]) -> Path:
    """Write an IKET JSON capture of CTA (0, 0, 0) from ``(name, warp, start, end)`` ranges."""
    names = sorted({name for name, *_ in ranges})
    warps = sorted({warp for _, warp, *_ in ranges})
    path.write_text(
        json.dumps(
            {
                "stringTable": names,
                "locationTable": [{"ctaId": [0, 0, 0], "warpId": warp} for warp in warps],
                "launches": [
                    {
                        "kernelName": "toy",
                        "ranges": [
                            {
                                "startTs": start,
                                "endTs": end,
                                "rangeNameIdx": names.index(name),
                                "warpLocIdxs": [warps.index(warp)],
                            }
                            for name, warp, start, end in ranges
                        ],
                    }
                ],
            }
        )
    )
    return path


def two_warp_plan() -> PipelinePlan:
    """One region on a two-warp role."""
    plan = PipelinePlan("two-warp", iteration_name="tile")
    plan.add_role(Role("role", "Role", 0, 2))
    plan.add_region(Region("work", "role", "Work", 0, 1))
    return plan


def test_iket_join_rejects_extra_occurrences(tmp_path: Path) -> None:
    """A persistent CTA that ran three items does not silently analyze the first two."""
    timeline = two_warp_plan().schedule(iterations=2)
    path = write_iket(
        tmp_path / "extra.json",
        [("work", warp, 100 * i, 100 * i + 50) for warp in (0, 1) for i in range(3)],
    )
    with pytest.raises(ValueError, match=r"warp 0 recorded 'work' 3 times.*iterations=3"):
        load_iket_capture(path, timeline)


def test_iket_join_rejects_a_warp_missing_an_occurrence(tmp_path: Path) -> None:
    """Warp 1 missing its middle range would shift its last one onto iteration 1, while warp 0
    alone satisfies every (region, iteration)."""
    timeline = two_warp_plan().schedule(iterations=3)
    ranges = [("work", 0, 100 * i, 100 * i + 50) for i in range(3)]
    ranges += [("work", 1, 100 * i, 100 * i + 50) for i in (0, 2)]
    path = write_iket(tmp_path / "missing.json", ranges)
    with pytest.raises(ValueError, match=r"warp 1 recorded 'work' 2 times"):
        load_iket_capture(path, timeline)


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
    # merge_traces folds each input into one process named after its label.
    assert process_names == {"first", "second"}


def test_steady_state_analysis_does_not_depend_on_the_window() -> None:
    """A reuse distance at least the window length still bounds the recurrence and is a ring."""
    results = []
    for iterations in (1, 2, 4):
        timeline = toy_plan().schedule(iterations=iterations)
        analysis = analyze_pipeline(timeline, logical_capture(timeline))
        cycle = analysis.logical_critical_cycle
        results.append((cycle.initiation_interval, [ring.resource for ring in analysis.rings]))
    assert results == [(4, ["x"])] * 3


def fanout_plan(first: str) -> PipelinePlan:
    """``load`` fills one slot of ``x`` that ``fast`` and ``slow`` both read and release."""
    plan = PipelinePlan("fanout", iteration_name="tile")
    for warp, name in enumerate(("load", "fast", "slow")):
        plan.add_role(Role(name, name, warp, warp + 1))
        plan.add_region(Region(name, name, name, 0, 1))
    plan.add_resource(Resource("x", "X", 1))
    consumers = ("fast", "slow") if first == "fast" else ("slow", "fast")
    for consumer in consumers:
        plan.add_dependency(Dependency("load", consumer, "x"))
    for consumer in consumers:
        plan.add_dependency(Dependency(consumer, "load", "x", distance=1, kind="reuse"))
    return plan


@pytest.mark.parametrize("first", ["fast", "slow"])
def test_ring_slot_is_free_after_its_last_release(first: str) -> None:
    """With two readers the slot drains when the slow one releases it, whatever the order of
    the plan's dependencies."""
    timeline = fanout_plan(first).schedule(iterations=2)
    regions = [
        (name, i, 600 * i + start, 600 * i + end)
        for i in range(2)
        for name, start, end in (("load", 0, 100), ("fast", 100, 300), ("slow", 100, 600))
    ]
    measured = MeasuredCapture(
        "measured",
        "fanout",
        (0, 0, 0),
        0,
        1200,
        tuple(MeasuredRegion(n, n, i, s, e, float(e - s), 1) for n, i, s, e in regions),
    )
    analysis = analyze_pipeline(timeline, measured)
    assert [(r.resource, r.peak_occupancy, r.final_occupancy) for r in analysis.rings] == [
        ("x", 1, 0)
    ]
    assert ring_occupancy_samples(timeline, measured) == {
        "x": ((0, 0), (100, 1), (600, 0), (700, 1), (1200, 0))
    }
    assert [d.resource for d in analysis.depth_counterfactuals] == ["x"]


def test_perfetto_splits_crossing_role_envelopes_and_normalizes_suffix(
    tmp_path: Path,
) -> None:
    plan = PipelinePlan("crossing", iteration_name="tile")
    plan.add_role(Role("role", "Role", 0, 1))
    plan.add_region(Region("a", "role", "A", 0, 1))
    plan.add_region(Region("b", "role", "B", 1, 1))
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


def test_perfetto_zero_duration_slice_begins_before_it_ends(tmp_path: Path) -> None:
    """An empty region at the time the next one starts on its lane opens and closes before it."""
    plan = PipelinePlan("empty", iteration_name="tile")
    plan.add_role(Role("role", "Role", 0, 1))
    plan.add_region(Region("a", "role", "A", 0, 1))
    plan.add_region(Region("b", "role", "B", 1, 1))
    timeline = plan.schedule(iterations=1)
    regions = (
        MeasuredRegion("a", "role", 0, 5, 5, 0.0, 1),
        MeasuredRegion("b", "role", 0, 5, 9, 4.0, 1),
    )
    measured = MeasuredCapture("measured", "empty", (0, 0, 0), 5, 4, regions)

    trace = Trace()
    trace.ParseFromString(
        write_pipeline_perfetto(tmp_path / "empty.pftrace", timeline, measured).read_bytes()
    )

    open_slices: dict[int, int] = {}
    for packet in trace.packet:
        event = packet.track_event
        if event.type == TrackEvent.TYPE_SLICE_BEGIN:
            open_slices[event.track_uuid] = open_slices.get(event.track_uuid, 0) + 1
        elif event.type == TrackEvent.TYPE_SLICE_END:
            assert open_slices.get(event.track_uuid, 0) > 0, "slice end without a begin"
            open_slices[event.track_uuid] -= 1
    assert not any(open_slices.values())


INDEXED_SOURCE = """
PIPELINE = PipelineAnnotations("indexed", iteration_name="chunk")
PIPELINE.resource("x_0", label="X hi", depth=1)
PIPELINE.resource("x_1", label="X lo", depth=1)
PIPELINE.resource("acc", label="Acc", depth=1)

class Kernel:
    @PIPELINE.role("state", label="State", warp_start=0, warp_end=4)
    def run_state(self):
        PIPELINE.region(
            "state.split", label="Split", weight=2,
            produces=("x_0", "x_1"),
        )
        PIPELINE.region(
            "state.wait", label="Wait", weight=1,
            consumes=("acc",), releases=("acc",),
        )
        PIPELINE.iteration_end()

    @PIPELINE.role("mma", label="MMA", warp_start=4, warp_end=5)
    def run_mma(self):
        for split in cutlass.range_constexpr(2):
            PIPELINE.region(
                "mma.pass", label="Pass", weight=2,
                consumes=("x_{index}",), releases=("x_{index}",), produces=("acc",),
                count=2, index=split,
            )
        PIPELINE.iteration_end()
"""


def test_indexed_regions_expand_with_per_copy_tokens(tmp_path: Path) -> None:
    source = tmp_path / "indexed.py"
    source.write_text(INDEXED_SOURCE)

    plan = extract_plan(source)

    assert [region.name for region in plan.regions.values() if region.role == "mma"] == [
        "mma.pass[0]",
        "mma.pass[1]",
    ]
    edges = {(edge.source, edge.target, edge.resource, edge.kind) for edge in plan.dependencies}
    assert ("state.split", "mma.pass[0]", "x_0", "data") in edges
    assert ("state.split", "mma.pass[1]", "x_1", "data") in edges
    # Plain produces attach to the last copy only.
    assert ("mma.pass[1]", "state.wait", "acc", "data") in edges
    assert not any(
        edge.source == "mma.pass[0]" and edge.resource == "acc" for edge in plan.dependencies
    )
    assert plan.schedule(iterations=2).logical_duration > 0


def test_indexed_region_requires_count_and_index_together(tmp_path: Path) -> None:
    source = tmp_path / "bad.py"
    source.write_text(INDEXED_SOURCE.replace("count=2, index=split,", "count=2,"))
    with pytest.raises(ValueError, match="count>1 and index"):
        extract_plan(source)


def test_runtime_indexed_region_names_and_rejects_dynamic_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations = PipelineAnnotations("test", iteration_name="tile")
    annotations.enable_iket()
    pushed: list[str] = []
    fake_iket = SimpleNamespace(range_push=pushed.append, range_pop=lambda: None)
    monkeypatch.setattr(pipeline_annotations, "iket", fake_iket)

    @annotations.role("mma", label="MMA", warp_start=0, warp_end=1)
    def body(index) -> None:
        annotations.region("pass", label="Pass", weight=1, count=2, index=index)
        annotations.iteration_end()

    body(1)
    assert pushed == ["pass[1]"]
    with pytest.raises(TypeError, match="compile-time int"):
        body(SimpleNamespace())
    with pytest.raises(ValueError, match="outside count"):
        body(2)


def test_report_lists_regions_and_scheduler_findings() -> None:
    timeline = toy_plan().schedule(iterations=4)
    measured = logical_capture(timeline)
    analysis = analyze_pipeline(timeline, measured)

    text = report(timeline, measured, analysis, unprofiled_iteration_ns=100.0)

    assert "per iteration ~" in text and "instrumentation +" in text
    assert "load     load.x" in text and "compute  compute.x" in text
    assert "iteration 2 timeline" in text
    assert "RecMII" in text and "critical cycle: " in text


def test_pipeline_report_is_the_function_after_importing_every_submodule() -> None:
    """No submodule shadows the package's ``report`` export, whatever was imported first."""
    script = (
        "import pkgutil, importlib\n"
        "import transformer_nuggets.cute.profiler.pipeline as pipeline\n"
        "for info in pkgutil.iter_modules(pipeline.__path__):\n"
        "    importlib.import_module(f'{pipeline.__name__}.{info.name}')\n"
        "from transformer_nuggets.cute.profiler.pipeline import report\n"
        "assert callable(report), report\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_capture_cli_reanalyzes_an_existing_trace(tmp_path: Path, capsys) -> None:
    from transformer_nuggets.cute.profiler.pipeline.capture import main

    source = tmp_path / "annotated.py"
    source.write_text(ANNOTATED_SOURCE)
    timeline = extract_plan(source).schedule(iterations=3)
    names = ["load.x", "compute.x"]
    locations = [{"ctaId": [0, 0, 0], "warpId": 0}, {"ctaId": [0, 0, 0], "warpId": 1}]
    ranges = [
        {
            "startTs": 1_000 + item.start * 10,
            "endTs": 1_000 + item.end * 10,
            "rangeNameIdx": names.index(item.region.name),
            "warpLocIdxs": [0 if item.region.role == "load" else 1] * 2,
        }
        for item in timeline.regions
    ]
    trace = tmp_path / "iket.trace.json"
    trace.write_text(
        json.dumps(
            {
                "stringTable": names,
                "locationTable": locations,
                "launches": [{"kernelName": "demo", "ranges": ranges}],
            }
        )
    )

    rc = main(
        [
            "--tag",
            "demo.v1",
            "--annotated",
            str(source),
            "--iterations",
            "3",
            "--out",
            str(tmp_path / "traces"),
            "--from-trace",
            str(trace),
        ]
    )

    assert rc == 0
    # A dotted tag keeps its dot: demo.v1.report.txt, not demo.report.txt.
    assert (tmp_path / "traces" / "demo.v1.enriched.pftrace").exists()
    assert "RecMII" in (tmp_path / "traces" / "demo.v1.report.txt").read_text()
    assert "iteration 1 timeline" in capsys.readouterr().out
