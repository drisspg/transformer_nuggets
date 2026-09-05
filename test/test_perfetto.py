import gzip
import json

import pytest

from transformer_nuggets.utils.perfetto import (
    add_cuda_graph_annotation_boxes,
    add_cuda_graph_annotation_tracks_to_native_trace,
    default_trace_path,
    default_track_event_path,
    read_trace,
    separate_gpu_annotation_slices,
    split_overlapping_slices,
    write_trace,
)
from transformer_nuggets.utils.track_event import chrome_trace_to_track_event_trace


def _duration_events(trace):
    return [event for event in trace["traceEvents"] if event.get("ph") == "X"]


def _annotation_boxes(trace):
    return [
        event
        for event in trace["traceEvents"]
        if isinstance(event, dict) and event.get("cat") == "gpu_user_annotation"
    ]


def test_cuda_graph_annotations_become_contiguous_gpu_boxes():
    graph_id = 2
    annotations = {
        (graph_id << 32) | 0: [{"name": "attention"}],
        (graph_id << 32) | 2: [{"name": "attention"}],
        (graph_id << 32) | 3: [{"name": "loss"}],
    }
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "kernel_a",
                "pid": 0,
                "tid": 7,
                "ts": 10,
                "dur": 3,
                "args": {"graph id": graph_id, "graph node id": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "kernel_b",
                "pid": 0,
                "tid": 7,
                "ts": 13,
                "dur": 5,
                "args": {"graph id": graph_id, "graph node id": 2},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "kernel_c",
                "pid": 0,
                "tid": 7,
                "ts": 18,
                "dur": 2,
                "args": {"graph id": graph_id, "graph node id": 3},
            },
        ]
    }

    processed = add_cuda_graph_annotation_boxes(trace, annotations)
    boxes = [
        event for event in processed["traceEvents"] if event.get("cat") == "gpu_user_annotation"
    ]

    assert [(box["name"], box["ts"], box["dur"]) for box in boxes] == [
        ("attention", 10.0, 8.0),
        ("loss", 18.0, 2.0),
    ]
    assert all(box["tid"] == 7 for box in boxes)
    assert len(trace["traceEvents"]) == 3


def test_cuda_graph_annotation_boxes_label_backward_phase():
    graph_id = 2
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 10,
                "dur": 3,
                "args": {"graph id": graph_id, "graph node id": 1},
            }
        ]
    }
    annotations = {(graph_id << 32) | 1: [{"name": "attention", "autograd_phase": "backward"}]}

    processed = add_cuda_graph_annotation_boxes(trace, annotations)

    assert processed["traceEvents"][-1]["name"] == "attention backward"


def test_cuda_graph_annotation_boxes_accept_monitor_embedded_metadata():
    graph_id = 2
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 10,
                "dur": 3,
                "args": {
                    "graph id": graph_id,
                    "graph node id": 1,
                    "annotation": '[{"name": "attention"}]',
                },
            }
        ]
    }

    processed = add_cuda_graph_annotation_boxes(
        trace,
        {(graph_id << 32) | 1: [{"name": "registry label"}]},
    )

    assert processed["traceEvents"][-1]["name"] == "attention"
    assert processed["traceEvents"][-1]["cat"] == "gpu_user_annotation"


def test_cuda_graph_annotation_boxes_split_on_unannotated_gpu_work():
    graph_id = 2
    annotations = {(graph_id << 32) | 1: [{"name": "attention"}]}
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 0,
                "dur": 2,
                "args": {"graph id": graph_id, "graph node id": 1, "correlation": 4},
            },
            {"ph": "X", "cat": "kernel", "pid": 0, "tid": 7, "ts": 2, "dur": 2, "args": {}},
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 4,
                "dur": 2,
                "args": {"graph id": graph_id, "graph node id": 1, "correlation": 4},
            },
        ]
    }

    processed = add_cuda_graph_annotation_boxes(trace, annotations)

    assert [(box["ts"], box["dur"]) for box in _annotation_boxes(processed)] == [
        (0.0, 2.0),
        (4.0, 2.0),
    ]


@pytest.mark.parametrize("with_correlation", [False, True])
def test_cuda_graph_annotation_boxes_split_at_replay_boundaries(with_correlation):
    graph_id = 2
    annotations = {(graph_id << 32) | 0: [{"name": "attention"}]}
    args = {"graph id": graph_id, "graph node id": 0}
    if with_correlation:
        first_args = {**args, "correlation": 4}
        second_args = {**args, "correlation": 5}
    else:
        first_args = second_args = args
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 0,
                "dur": 2,
                "args": first_args,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 2,
                "dur": 2,
                "args": second_args,
            },
        ]
    }

    processed = add_cuda_graph_annotation_boxes(trace, annotations)

    assert [(box["ts"], box["dur"]) for box in _annotation_boxes(processed)] == [
        (0.0, 2.0),
        (2.0, 2.0),
    ]


def test_cuda_graph_annotation_boxes_keep_devices_separate():
    graph_id = 2
    annotations = {(graph_id << 32) | 0: [{"name": "attention"}]}
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 0,
                "dur": 2,
                "args": {"graph id": graph_id, "graph node id": 0, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 2,
                "dur": 2,
                "args": {"graph id": graph_id, "graph node id": 0, "device": 1},
            },
        ]
    }

    processed = add_cuda_graph_annotation_boxes(trace, annotations)

    assert [(box["ts"], box["dur"]) for box in _annotation_boxes(processed)] == [
        (0.0, 2.0),
        (2.0, 2.0),
    ]


def test_cuda_graph_annotation_boxes_skip_malformed_events():
    trace = {
        "traceEvents": [
            "not an event",
            {"ph": "X", "cat": "kernel", "args": "not an args mapping"},
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": "not a timestamp",
                "dur": 2,
                "args": {"graph id": 2, "graph node id": 0},
            },
        ]
    }

    assert add_cuda_graph_annotation_boxes(trace, {(2 << 32): [{"name": "attention"}]}) == trace


def test_native_cuda_graph_annotation_tracks_preserve_gpu_packets():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

    trace = Trace()
    process_packet = trace.packet.add()
    process_packet.track_descriptor.uuid = 999
    process_packet.track_descriptor.process.pid = 123
    process_packet.track_descriptor.process.process_name = "python"
    for timestamp, annotation in [(100, "attention"), (110, "attention"), (120, None)]:
        packet = trace.packet.add()
        packet.timestamp = timestamp
        packet.timestamp_clock_id = 6
        event = packet.gpu_render_stage_event
        event.duration = 10
        event.event_id = timestamp
        event.gpu_id = 0
        event.hw_queue_iid = 7
        metadata = {
            "process_id": "123",
            "device id": "0",
            "stream id": "7",
            "graph id": "2",
            "correlation": "4",
        }
        if annotation is not None:
            metadata["name"] = annotation
        for name, value in metadata.items():
            item = event.extra_data.add()
            item.name = name
            item.value = value

    original_packets = [packet.SerializeToString() for packet in trace.packet]
    output = add_cuda_graph_annotation_tracks_to_native_trace(
        gzip.compress(trace.SerializeToString())
    )
    processed = Trace()
    processed.ParseFromString(gzip.decompress(output))

    assert [
        packet.SerializeToString() for packet in processed.packet[: len(original_packets)]
    ] == (original_packets)
    assert sum(packet.HasField("gpu_render_stage_event") for packet in processed.packet) == 3
    annotation_descriptors = [
        packet.track_descriptor
        for packet in processed.packet
        if packet.HasField("track_descriptor") and packet.track_descriptor.name == "annotations"
    ]
    assert len(annotation_descriptors) == 1
    assert annotation_descriptors[0].parent_uuid == 999
    annotation_events = [
        packet.track_event for packet in processed.packet if packet.HasField("track_event")
    ]
    assert [event.type for event in annotation_events] == [
        TrackEvent.TYPE_SLICE_BEGIN,
        TrackEvent.TYPE_SLICE_END,
    ]
    assert annotation_events[0].name == "attention"


def test_native_annotation_tracks_merge_streams_and_split_overlaps():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

    trace = Trace()
    process_packet = trace.packet.add()
    process_packet.track_descriptor.uuid = 999
    process_packet.track_descriptor.process.pid = 123
    for timestamp, duration, stream, name in [
        (100, 20, "7", "attention"),
        (105, 10, "8", "optimizer"),
    ]:
        packet = trace.packet.add()
        packet.timestamp = timestamp
        packet.timestamp_clock_id = 6
        event = packet.gpu_render_stage_event
        event.duration = duration
        event.event_id = timestamp
        event.gpu_id = 0
        event.hw_queue_iid = int(stream)
        for key, value in {
            "process_id": "123",
            "device id": "0",
            "stream id": stream,
            "graph id": "2",
            "correlation": stream,
            "name": name,
        }.items():
            item = event.extra_data.add()
            item.name = key
            item.value = value

    processed = Trace()
    processed.ParseFromString(
        add_cuda_graph_annotation_tracks_to_native_trace(trace.SerializeToString())
    )
    names_by_uuid = {
        packet.track_descriptor.uuid: packet.track_descriptor.name
        for packet in processed.packet
        if packet.HasField("track_descriptor") and packet.track_descriptor.name
    }
    annotation_tracks = {
        packet.track_event.name: names_by_uuid[packet.track_event.track_uuid]
        for packet in processed.packet
        if packet.HasField("track_event")
        and packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN
    }

    assert annotation_tracks == {
        "attention": "annotations",
        "optimizer": "annotations overlap",
    }


def test_native_annotation_tracks_require_graph_identity_and_label_backward():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

    trace = Trace()
    for timestamp, metadata in [
        (100, {"name": "unrelated eager metadata", "stream id": "7"}),
        (
            110,
            {
                "name": "attention",
                "autograd_phase": "backward",
                "stream id": "7",
                "graph id": "2",
                "correlation": "4",
            },
        ),
    ]:
        packet = trace.packet.add()
        packet.timestamp = timestamp
        event = packet.gpu_render_stage_event
        event.duration = 10
        event.event_id = timestamp
        event.gpu_id = 0
        event.hw_queue_iid = 7
        for name, value in metadata.items():
            item = event.extra_data.add()
            item.name = name
            item.value = value

    processed = Trace()
    processed.ParseFromString(
        add_cuda_graph_annotation_tracks_to_native_trace(trace.SerializeToString())
    )
    annotation_begins = [
        packet.track_event
        for packet in processed.packet
        if packet.HasField("track_event")
        and packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN
    ]

    assert [event.name for event in annotation_begins] == ["attention backward"]


def test_native_annotation_tracks_skip_truncated_gzip():
    payload = b"\x1f\x8btruncated"
    assert add_cuda_graph_annotation_tracks_to_native_trace(payload) == payload


def test_cuda_graph_annotation_boxes_skip_overflowing_timestamps():
    event = {
        "ph": "X",
        "cat": "kernel",
        "pid": 0,
        "tid": 7,
        "ts": 10**10_000,
        "dur": 2,
        "args": {"graph id": 2, "graph node id": 0},
    }

    assert add_cuda_graph_annotation_boxes(
        {"traceEvents": [event]}, {(2 << 32): [{"name": "attention"}]}
    ) == {"traceEvents": [event]}


def test_legacy_cuda_graph_annotation_box_remains_idempotent():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 0,
                "dur": 2,
                "args": {"graph id": 2, "graph node id": 0},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "attention",
                "pid": 0,
                "tid": 7,
                "ts": 0,
                "dur": 2,
                "args": {"transformer_nuggets.graph_annotation": True},
            },
        ]
    }

    assert add_cuda_graph_annotation_boxes(trace, {(2 << 32): [{"name": "attention"}]}) == trace


def test_cuda_graph_annotation_box_idempotency_is_per_replay_source():
    graph_id = 2
    annotations = {(graph_id << 32) | 0: [{"name": "attention"}]}
    first = {
        "ph": "X",
        "cat": "kernel",
        "pid": 0,
        "tid": 7,
        "ts": 0,
        "dur": 2,
        "args": {"graph id": graph_id, "graph node id": 0, "correlation": 4},
    }
    second = {**first, "ts": 2, "args": {**first["args"], "correlation": 5}}

    once = add_cuda_graph_annotation_boxes({"traceEvents": [first]}, annotations)
    twice = add_cuda_graph_annotation_boxes(
        {"traceEvents": [*once["traceEvents"], second]}, annotations
    )

    assert [(box["ts"], box["dur"]) for box in _annotation_boxes(twice)] == [
        (0.0, 2.0),
        (2.0, 2.0),
    ]


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
    assert thread_names[700] == "stream 7"
    assert thread_names[701] == "stream 7 overlap"


def test_split_overlapping_slices_ignores_gpu_annotation_spans():
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 0,
                "tid": 7,
                "args": {"name": "stream 7"},
            },
            {"ph": "X", "cat": "kernel", "name": "a", "pid": 0, "tid": 7, "ts": 0, "dur": 10},
            {"ph": "X", "cat": "kernel", "name": "b", "pid": 0, "tid": 7, "ts": 5, "dur": 10},
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "attention",
                "pid": 0,
                "tid": 7,
                "ts": 0,
                "dur": 15,
            },
        ]
    }

    fixed = split_overlapping_slices(trace, track_pattern="stream.*")
    thread_names = {
        event["tid"]: event["args"]["name"]
        for event in fixed["traceEvents"]
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    annotations = [
        event for event in fixed["traceEvents"] if event.get("cat") == "gpu_user_annotation"
    ]

    assert set(thread_names.values()) == {"stream 7", "stream 7 overlap"}
    assert annotations[0]["tid"] == 700


def test_gpu_annotations_are_aggregated_without_dropping_source_metadata():
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
                "pid": 0,
                "tid": 8,
                "args": {"name": "stream 8"},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "attention",
                "pid": 0,
                "tid": 7,
                "ts": 0,
                "dur": 10,
                "args": {"detail": "left"},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "optimizer",
                "pid": 0,
                "tid": 8,
                "ts": 5,
                "dur": 10,
                "args": {"detail": "right"},
            },
        ]
    }

    processed = separate_gpu_annotation_slices(trace)
    thread_names = {
        event["tid"]: event["args"]["name"]
        for event in processed["traceEvents"]
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    annotations = [event for event in processed["traceEvents"] if event.get("cat") == "annotation"]

    assert {thread_names[event["tid"]] for event in annotations} == {
        "annotations",
        "annotations overlap",
    }
    assert {event["name"] for event in annotations} == {"attention", "optimizer"}
    assert {event["args"]["detail"] for event in annotations} == {"left", "right"}
    assert {
        event["args"]["transformer_nuggets.annotation_source_tid"] for event in annotations
    } == {
        7,
        8,
    }


def test_overlap_lane_sort_indices_stay_adjacent():
    trace = {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 7, "args": {"name": "stream 7"}},
            {
                "ph": "M",
                "name": "thread_sort_index",
                "pid": 0,
                "tid": 7,
                "args": {"sort_index": 10},
            },
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 8, "args": {"name": "stream 8"}},
            {
                "ph": "M",
                "name": "thread_sort_index",
                "pid": 0,
                "tid": 8,
                "args": {"sort_index": 11},
            },
            {"ph": "X", "name": "a", "pid": 0, "tid": 7, "ts": 0, "dur": 10},
            {"ph": "X", "name": "b", "pid": 0, "tid": 7, "ts": 5, "dur": 10},
            {"ph": "X", "name": "c", "pid": 0, "tid": 8, "ts": 0, "dur": 10},
        ]
    }

    fixed = split_overlapping_slices(trace, track_pattern="stream.*")
    names = {
        event["tid"]: event["args"]["name"]
        for event in fixed["traceEvents"]
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    sort_indices = {
        event["tid"]: event["args"]["sort_index"]
        for event in fixed["traceEvents"]
        if event.get("ph") == "M" and event.get("name") == "thread_sort_index"
    }
    ordered_names = [
        name for tid, name in sorted(names.items(), key=lambda item: sort_indices[item[0]])
    ]

    assert ordered_names == ["stream 7", "stream 7 overlap", "stream 8"]


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
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

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


def test_track_event_conversion_splits_annotation_overlaps_from_gpu_work():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

    trace = {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 7, "args": {"name": "stream 7"}},
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 8, "args": {"name": "stream 8"}},
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
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "burst_1",
                "pid": 0,
                "tid": 8,
                "ts": 5,
                "dur": 10,
            },
            {
                "ph": "X",
                "cat": "gpu_roofline_annotation",
                "name": "roofline_0",
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
    assert event_tracks["burst_0"] == "annotations"
    assert event_tracks["burst_1"] == "annotations overlap"
    assert event_tracks["roofline_0"] == "Roofline stream 7"


def test_track_event_conversion_attaches_paired_flows_to_slices():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

    trace = {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 1, "args": {"name": "cpu"}},
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 2, "args": {"name": "gpu"}},
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 3, "args": {"name": "cpu2"}},
            {"ph": "X", "name": "cudaLaunchKernel", "pid": 0, "tid": 1, "ts": 0, "dur": 10},
            {"ph": "X", "name": "kernel", "pid": 0, "tid": 2, "ts": 20, "dur": 5},
            {"ph": "X", "name": "consume", "pid": 0, "tid": 3, "ts": 30, "dur": 5},
            {"ph": "s", "name": "ac2g", "pid": 0, "tid": 1, "ts": 1, "id": 99},
            {"ph": "f", "name": "ac2g", "pid": 0, "tid": 2, "ts": 20, "id": 99},
            {"ph": "s", "name": "g2c", "pid": 0, "tid": 2, "ts": 22, "id": 101},
            {"ph": "f", "name": "g2c", "pid": 0, "tid": 3, "ts": 30, "id": 101},
            {"ph": "f", "name": "single-ended-noise", "pid": 0, "tid": 2, "ts": 23, "id": 100},
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
    terminating_flow_ids_by_name = {
        event.name: tuple(event.terminating_flow_ids) for event in begins
    }

    assert flow_ids_by_name["cudaLaunchKernel"] == (99,)
    assert terminating_flow_ids_by_name["cudaLaunchKernel"] == ()
    assert flow_ids_by_name["kernel"] == (101,)
    assert terminating_flow_ids_by_name["kernel"] == (99,)
    assert flow_ids_by_name["consume"] == ()
    assert terminating_flow_ids_by_name["consume"] == (101,)
    assert all(100 not in flow_ids for flow_ids in flow_ids_by_name.values())
    assert all(100 not in flow_ids for flow_ids in terminating_flow_ids_by_name.values())


def test_track_event_conversion_splits_crossing_slices_and_keeps_nested_slices():
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import (
        Trace,
        TrackDescriptor,
        TrackEvent,
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

    assert set(names_by_uuid.values()) >= {"stream 1", "stream 1 overlap"}
    assert event_tracks["outer"] == event_tracks["inner"] == "stream 1"
    assert event_tracks["crossing"] == "stream 1 overlap"
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
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

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
        events = [
            {
                "ph": "M",
                "name": "process_sort_index",
                "pid": 7,
                "tid": 0,
                "args": {"sort_index": 100 + idx},
            },
            {"ph": "X", "name": f"op{idx}", "pid": 7, "tid": 3, "ts": 100 + idx, "dur": 5},
        ]
        path.write_text(json.dumps({"traceEvents": events}))
        inputs.append(str(path))

    output = tmp_path / "merged.pftrace"
    merge_traces(inputs, str(output), labels=["impl a", "impl b"], align_timestamps=True)

    trace = Trace()
    trace.ParseFromString(output.read_bytes())
    process_ranks = {
        p.track_descriptor.process.process_name: p.track_descriptor.sibling_order_rank
        for p in trace.packet
        if p.HasField("track_descriptor") and p.track_descriptor.HasField("process")
    }
    assert process_ranks["impl a"] < process_ranks["impl b"]
    slice_names = {
        p.track_event.name
        for p in trace.packet
        if p.HasField("track_event") and p.track_event.name
    }
    assert {"op0", "op1"} <= slice_names


def test_merge_traces_accepts_native_pftrace_inputs(tmp_path):
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace

    from transformer_nuggets.utils.merge_traces import merge_traces
    from transformer_nuggets.utils.track_event import write_track_event_trace

    inputs = []
    for index in range(2):
        path = tmp_path / f"rank{index}.pftrace"
        write_track_event_trace(
            path,
            {
                "traceEvents": [
                    {
                        "ph": "M",
                        "name": "process_name",
                        "pid": 7,
                        "tid": 0,
                        "args": {"name": "worker"},
                    },
                    {
                        "ph": "X",
                        "name": f"op{index}",
                        "pid": 7,
                        "tid": 3,
                        "ts": 100 + index,
                        "dur": 5,
                    },
                ]
            },
        )
        inputs.append(str(path))

    output = tmp_path / "merged.pftrace"
    merge_traces(inputs, str(output), labels=["rank 0", "rank 1"])

    trace = Trace()
    trace.ParseFromString(output.read_bytes())
    descriptor_uuids = [
        packet.track_descriptor.uuid
        for packet in trace.packet
        if packet.HasField("track_descriptor")
    ]
    assert len(descriptor_uuids) == len(set(descriptor_uuids))
    process_names = {
        packet.track_descriptor.process.process_name
        for packet in trace.packet
        if packet.HasField("track_descriptor") and packet.track_descriptor.HasField("process")
    }
    assert any(name.startswith("rank 0") for name in process_names)
    assert any(name.startswith("rank 1") for name in process_names)
    slice_names = {
        packet.track_event.name
        for packet in trace.packet
        if packet.HasField("track_event") and packet.track_event.name
    }
    assert {"op0", "op1"} <= slice_names
    sequence_ids = {
        packet.trusted_packet_sequence_id
        for packet in trace.packet
        if packet.HasField("track_event")
    }
    assert len(sequence_ids) == 2


def test_merge_traces_rejects_native_input_with_json_output(tmp_path):
    from transformer_nuggets.utils.merge_traces import merge_traces
    from transformer_nuggets.utils.track_event import write_track_event_trace

    input_path = tmp_path / "rank0.pftrace"
    write_track_event_trace(input_path, {"traceEvents": []})

    with pytest.raises(ValueError, match="native Perfetto inputs require"):
        merge_traces([str(input_path)], str(tmp_path / "merged.json"))


def test_merge_traces_folds_each_rank_into_one_process(tmp_path):
    """Kineto emits a python process plus one 'python' process per CUDA device
    with small pids that collide across ranks; the merge must give each rank a
    single uniquely-numbered process with GPU and global tracks under it."""
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace

    from transformer_nuggets.utils.merge_traces import merge_traces

    def kineto_like_trace(path, pid):
        trace = Trace()
        for uuid, parent, name, rank, process_pid in (
            (11, 0, "", pid, pid),  # real python process
            (12, 0, "", 5_000_000, 0),  # device 0 process, same pid on every rank
            (13, 0, "", 5_000_001, 1),  # empty device 1 process
            (21, 11, "thread 1 (python)", 100, None),
            (22, 12, "stream 7", 700, None),
            (23, 12, "annotations", 690, None),
            (24, 0, "Kineto events", 2**31 - 1, None),
        ):
            descriptor = trace.packet.add().track_descriptor
            descriptor.uuid, descriptor.parent_uuid, descriptor.name = uuid, parent, name
            descriptor.sibling_order_rank = rank
            if process_pid is not None:
                descriptor.process.pid = process_pid
                descriptor.process.process_name = "python"
        for track_uuid, name in ((21, "launch"), (22, "kernel")):
            packet = trace.packet.add()
            packet.timestamp = 10
            packet.trusted_packet_sequence_id = 1
            packet.track_event.track_uuid = track_uuid
            packet.track_event.name = name
        path.write_bytes(trace.SerializeToString())
        return str(path)

    inputs = [kineto_like_trace(tmp_path / f"rank{i}.pftrace", pid=4000 + i) for i in range(2)]
    output = tmp_path / "merged.pftrace"
    merge_traces(inputs, str(output), labels=["Rank 0", "Rank 1"])

    trace = Trace()
    trace.ParseFromString(output.read_bytes())
    descriptors = {
        p.track_descriptor.uuid: p.track_descriptor
        for p in trace.packet
        if p.HasField("track_descriptor")
    }
    processes = [d for d in descriptors.values() if d.HasField("process")]
    assert [d.process.process_name for d in processes] == ["Rank 0", "Rank 1"]
    assert len({d.process.pid for d in processes}) == 2
    assert [d.sibling_order_rank for d in processes] == [0, 1]
    for process in processes:
        children = sorted(
            (d for d in descriptors.values() if d.parent_uuid == process.uuid),
            key=lambda d: d.sibling_order_rank,
        )
        assert [d.name for d in children] == [
            "thread 1 (python)",
            "annotations",
            "stream 7",
            "Kineto events",
        ]
    assert all(d.parent_uuid or d.HasField("process") for d in descriptors.values())
    event_tracks = {p.track_event.track_uuid for p in trace.packet if p.HasField("track_event")}
    assert event_tracks <= set(descriptors)
