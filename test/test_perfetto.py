import gzip
import json

import pytest

from transformer_nuggets.utils.perfetto import (
    default_trace_path,
    default_track_event_path,
    read_trace,
    split_overlapping_slices,
    write_trace,
)
from transformer_nuggets.utils.track_event import chrome_trace_to_track_event_trace


def _duration_events(trace):
    return [event for event in trace["traceEvents"] if event.get("ph") == "X"]


def test_split_overlapping_slices_creates_adjacent_lanes():
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 0,
                "tid": 7,
                "args": {"name": "stream 7"},
            },
            {"ph": "X", "name": "a", "pid": 0, "tid": 7, "ts": 0, "dur": 10},
            {"ph": "X", "name": "b", "pid": 0, "tid": 7, "ts": 5, "dur": 10},
            {"ph": "X", "name": "c", "pid": 0, "tid": 7, "ts": 10, "dur": 1},
        ]
    }

    fixed = split_overlapping_slices(trace, track_pattern="stream.*")
    durations = _duration_events(fixed)

    assert [event["tid"] for event in durations] == [700, 701, 700]
    thread_names = {
        event["tid"]: event["args"]["name"]
        for event in fixed["traceEvents"]
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    assert thread_names[700] == "stream 7 #0"
    assert thread_names[701] == "stream 7 #1"


def test_split_overlapping_slices_leaves_non_overlapping_tracks_unchanged():
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 0,
                "tid": 3,
                "args": {"name": "stream 3"},
            },
            {"ph": "X", "name": "a", "pid": 0, "tid": 3, "ts": 0, "dur": 10},
            {"ph": "X", "name": "b", "pid": 0, "tid": 3, "ts": 10, "dur": 10},
        ]
    }

    fixed = split_overlapping_slices(trace, track_pattern="stream.*")

    assert fixed == trace


def test_split_overlapping_slices_remaps_flow_by_correlation_and_timestamp():
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 0,
                "tid": 7,
                "args": {"name": "stream 7"},
            },
            {
                "ph": "X",
                "name": "a",
                "pid": 0,
                "tid": 7,
                "ts": 0,
                "dur": 10,
                "args": {"correlation": 42},
            },
            {
                "ph": "X",
                "name": "b",
                "pid": 0,
                "tid": 7,
                "ts": 5,
                "dur": 10,
                "args": {"correlation": 42},
            },
            {"ph": "f", "pid": 0, "tid": 7, "ts": 0, "id": 42},
            {"ph": "f", "pid": 0, "tid": 7, "ts": 5, "id": 42},
        ]
    }

    fixed = split_overlapping_slices(trace, track_pattern="stream.*")
    flow_tids = [event["tid"] for event in fixed["traceEvents"] if event.get("ph") == "f"]

    assert flow_tids == [700, 701]


def test_split_overlapping_slices_keeps_same_tid_in_different_pids_separate():
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 0,
                "tid": 7,
                "args": {"name": "stream 7"},
            },
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 1,
                "tid": 7,
                "args": {"name": "stream 7"},
            },
            {"ph": "X", "name": "a", "pid": 0, "tid": 7, "ts": 0, "dur": 10},
            {"ph": "X", "name": "b", "pid": 0, "tid": 7, "ts": 5, "dur": 10},
            {"ph": "X", "name": "c", "pid": 1, "tid": 7, "ts": 0, "dur": 10},
            {"ph": "X", "name": "d", "pid": 1, "tid": 7, "ts": 5, "dur": 10},
        ]
    }

    fixed = split_overlapping_slices(trace, track_pattern="stream.*")
    by_pid = {}
    for event in _duration_events(fixed):
        by_pid.setdefault(event["pid"], []).append(event["tid"])

    assert by_pid == {0: [700, 701], 1: [700, 701]}


def test_gzip_trace_roundtrip(tmp_path):
    path = tmp_path / "trace.json.gz"
    trace = {"traceEvents": [{"ph": "X", "name": "a", "pid": 0, "tid": 0}]}

    write_trace(path, trace)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert json.load(f) == trace
    assert read_trace(path) == trace


def test_default_trace_path_prefers_gzip_for_stems_and_respects_explicit_gzip():
    assert default_trace_path("foo").as_posix() == "foo.json.gz"
    assert default_trace_path("foo.json").as_posix() == "foo.json"
    assert default_trace_path("foo.json.gz").as_posix() == "foo.json.gz"
    assert default_trace_path("foo", gzip_by_default=False).as_posix() == "foo.json"


def test_default_track_event_path_uses_native_perfetto_suffix():
    assert default_track_event_path("foo").as_posix() == "foo.pftrace"
    assert default_track_event_path("foo.json").as_posix() == "foo.pftrace"
    assert default_track_event_path("foo.json.gz").as_posix() == "foo.pftrace"
    assert default_track_event_path("foo.pftrace").as_posix() == "foo.pftrace"


def test_track_event_conversion_preserves_instants_counters_and_warns_on_unsupported():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent, Trace

    trace = {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 1, "args": {"name": "worker"}},
            {"ph": "i", "name": "marker", "pid": 0, "tid": 1, "ts": 1},
            {"ph": "C", "name": "memory", "pid": 0, "tid": 1, "ts": 2, "args": {"bytes": 42}},
            {"ph": "B", "name": "unsupported", "pid": 0, "tid": 1, "ts": 3},
        ]
    }

    with pytest.warns(RuntimeWarning, match="unsupported Chrome trace phases: B"):
        payload = chrome_trace_to_track_event_trace(trace)

    parsed = Trace()
    parsed.ParseFromString(payload)
    event_types = [
        packet.track_event.type for packet in parsed.packet if packet.HasField("track_event")
    ]
    assert TrackEvent.TYPE_INSTANT in event_types
    assert TrackEvent.TYPE_COUNTER in event_types


def test_track_event_conversion_puts_gpu_annotations_on_separate_track():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent, Trace

    trace = {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 7, "args": {"name": "stream 7"}},
            {"ph": "X", "cat": "kernel", "name": "kernel", "pid": 0, "tid": 7, "ts": 0, "dur": 10},
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "burst_0",
                "pid": 0,
                "tid": 7,
                "ts": 0,
                "dur": 10,
            },
        ]
    }

    parsed = Trace()
    parsed.ParseFromString(chrome_trace_to_track_event_trace(trace))
    names_by_uuid = {
        packet.track_descriptor.uuid: packet.track_descriptor.name
        for packet in parsed.packet
        if packet.HasField("track_descriptor") and packet.track_descriptor.name
    }
    event_tracks = {
        packet.track_event.name: names_by_uuid[packet.track_event.track_uuid]
        for packet in parsed.packet
        if packet.HasField("track_event")
        and packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN
    }

    assert event_tracks["kernel"] == "stream 7"
    assert event_tracks["burst_0"] == "GPU annotations stream 7"


def test_track_event_conversion_attaches_paired_flows_to_slices():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent, Trace

    trace = {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 1, "args": {"name": "cpu"}},
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 2, "args": {"name": "gpu"}},
            {"ph": "X", "name": "cudaLaunchKernel", "pid": 0, "tid": 1, "ts": 0, "dur": 10},
            {"ph": "X", "name": "kernel", "pid": 0, "tid": 2, "ts": 20, "dur": 5},
            {"ph": "s", "name": "ac2g", "pid": 0, "tid": 1, "ts": 1, "id": 99},
            {"ph": "f", "name": "ac2g", "pid": 0, "tid": 2, "ts": 20, "id": 99},
            {"ph": "f", "name": "single-ended-noise", "pid": 0, "tid": 2, "ts": 22, "id": 100},
        ]
    }

    parsed = Trace()
    parsed.ParseFromString(chrome_trace_to_track_event_trace(trace))
    begins = [
        packet.track_event
        for packet in parsed.packet
        if packet.HasField("track_event")
        and packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN
    ]
    flow_ids_by_name = {event.name: tuple(event.flow_ids) for event in begins}

    assert flow_ids_by_name["cudaLaunchKernel"] == (99,)
    assert flow_ids_by_name["kernel"] == (99,)
    assert all(100 not in flow_ids for flow_ids in flow_ids_by_name.values())


def test_track_event_conversion_splits_crossing_slices_and_keeps_nested_slices():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import (
        TrackDescriptor,
        TrackEvent,
        Trace,
    )

    trace = {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 1, "args": {"name": "stream 1"}},
            {"ph": "X", "name": "outer", "pid": 0, "tid": 1, "ts": 0, "dur": 10},
            {"ph": "X", "name": "inner", "pid": 0, "tid": 1, "ts": 2, "dur": 2},
            {"ph": "X", "name": "crossing", "pid": 0, "tid": 1, "ts": 5, "dur": 10},
        ]
    }

    parsed = Trace()
    parsed.ParseFromString(chrome_trace_to_track_event_trace(trace, track_pattern="stream.*"))

    descriptors_by_uuid = {
        packet.track_descriptor.uuid: packet.track_descriptor
        for packet in parsed.packet
        if packet.HasField("track_descriptor") and packet.track_descriptor.name
    }
    names_by_uuid = {uuid: descriptor.name for uuid, descriptor in descriptors_by_uuid.items()}
    begin_events = [
        packet.track_event
        for packet in parsed.packet
        if packet.HasField("track_event")
        and packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN
    ]
    event_tracks = {event.name: names_by_uuid[event.track_uuid] for event in begin_events}
    event_track_uuids = {event.name: event.track_uuid for event in begin_events}

    assert set(names_by_uuid.values()) >= {"stream 1"}
    assert event_tracks["outer"] == event_tracks["inner"] == event_tracks["crossing"] == "stream 1"
    assert event_track_uuids["outer"] == event_track_uuids["inner"]
    assert event_track_uuids["crossing"] != event_track_uuids["outer"]

    outer_desc = descriptors_by_uuid[event_track_uuids["outer"]]
    crossing_desc = descriptors_by_uuid[event_track_uuids["crossing"]]
    assert outer_desc.sibling_merge_behavior == (
        TrackDescriptor.SIBLING_MERGE_BEHAVIOR_BY_SIBLING_MERGE_KEY
    )
    assert crossing_desc.sibling_merge_behavior == outer_desc.sibling_merge_behavior
    assert crossing_desc.sibling_merge_key == outer_desc.sibling_merge_key


def test_track_event_conversion_keeps_back_to_back_slices_separate():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent, Trace

    trace = {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 7, "args": {"name": "stream 7"}},
            {"ph": "X", "name": "a", "pid": 0, "tid": 7, "ts": 10, "dur": 10},
            {"ph": "X", "name": "b", "pid": 0, "tid": 7, "ts": 20, "dur": 10},
        ]
    }

    parsed = Trace()
    parsed.ParseFromString(chrome_trace_to_track_event_trace(trace))

    open_stacks = {}
    rendered = {}
    for packet in parsed.packet:
        if not packet.HasField("track_event"):
            continue
        event = packet.track_event
        stack = open_stacks.setdefault(event.track_uuid, [])
        if event.type == TrackEvent.TYPE_SLICE_BEGIN:
            stack.append((event.name, packet.timestamp))
        elif event.type == TrackEvent.TYPE_SLICE_END:
            name, begin_ts = stack.pop()
            rendered[name] = (begin_ts, packet.timestamp)

    assert all(not stack for stack in open_stacks.values())
    assert rendered == {"a": (10_000, 20_000), "b": (20_000, 30_000)}


def test_merge_traces_writes_native_pftrace(tmp_path):
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace

    from transformer_nuggets.utils.merge_traces import merge_traces

    inputs = []
    for idx in range(2):
        path = tmp_path / f"rank{idx}.json"
        events = [{"ph": "X", "name": f"op{idx}", "pid": 7, "tid": 3, "ts": 100 + idx, "dur": 5}]
        path.write_text(json.dumps({"traceEvents": events}))
        inputs.append(str(path))

    output = tmp_path / "merged.pftrace"
    merge_traces(inputs, str(output), labels=["impl a", "impl b"], align_timestamps=True)

    trace = Trace()
    trace.ParseFromString(output.read_bytes())
    process_names = {
        p.track_descriptor.process.process_name
        for p in trace.packet
        if p.HasField("track_descriptor") and p.track_descriptor.HasField("process")
    }
    assert {"impl a", "impl b"} <= process_names
    slice_names = {
        p.track_event.name
        for p in trace.packet
        if p.HasField("track_event") and p.track_event.name
    }
    assert {"op0", "op1"} <= slice_names
