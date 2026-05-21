import gzip
import json

from transformer_nuggets.utils.perfetto import (
    default_trace_path,
    read_trace,
    split_overlapping_slices,
    write_trace,
)


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
    flow_tids = [
        event["tid"] for event in fixed["traceEvents"] if event.get("ph") == "f"
    ]

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
    assert default_trace_path("foo.json.gz").as_posix() == "foo.json.gz"
    assert default_trace_path("foo", gzip_by_default=False).as_posix() == "foo.json"
