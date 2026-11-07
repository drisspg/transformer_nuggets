import pytest

pytest.importorskip("cutlass")  # transformer_nuggets.cute imports CuTeDSL

import torch

from transformer_nuggets.cute.profiler import (
    Event,
    TagTable,
    allocate_profile_buffer,
    decode_events,
    dependency_gaps,
    link_dependency_flow,
    link_payload_flow,
    overlap_between,
    summarize_by_tag,
    validate_event_counts,
)
from transformer_nuggets.cute.profiler.ops import PAYLOAD_FLAG


def make_event(
    start_ns: int,
    dur_ns: int,
    tag_id: int,
    tag_name: str,
    event_idx: int,
    unit_id: int = 0,
) -> Event:
    """Construct one decoded event for host-side analysis tests."""
    return Event(
        start_ns=start_ns,
        dur_ns=dur_ns,
        tag_id=tag_id,
        tag_name=tag_name,
        tid=0,
        unit_id=unit_id,
        event_idx=event_idx,
    )


def test_summarize_by_tag_reports_duration_statistics():
    events = [
        make_event(0, 10, 0, "load", 0),
        make_event(20, 20, 0, "load", 1),
        make_event(50, 30, 0, "load", 2),
        make_event(90, 5, 1, "store", 3),
    ]

    summaries = summarize_by_tag(events)

    assert summaries["load"].count == 3
    assert summaries["load"].total_ns == 60
    assert summaries["load"].mean_ns == 20
    assert summaries["load"].min_ns == 10
    assert summaries["load"].p50_ns == 20
    assert summaries["load"].p95_ns == 29
    assert summaries["load"].max_ns == 30
    assert summaries["store"].p50_ns == 5


def test_overlap_between_merges_intervals_and_isolates_units():
    events = [
        make_event(0, 10, 0, "load", 0),
        make_event(20, 10, 0, "load", 1),
        make_event(5, 10, 1, "compute", 2),
        make_event(25, 10, 1, "compute", 3),
        make_event(0, 100, 1, "compute", 0, unit_id=1),
    ]

    overlap = overlap_between(events, "load", "compute")

    assert overlap.overlap_ns == 10
    assert overlap.left_total_ns == 20
    assert overlap.right_total_ns == 120
    assert overlap.left_fraction == 0.5
    assert overlap.right_fraction == 10 / 120

    global_overlap = overlap_between(events, "load", "compute", same_unit=False)
    assert global_overlap.overlap_ns == 20
    assert global_overlap.left_fraction == 1.0


def test_validate_event_counts_reports_too_few_and_too_many():
    events = [
        make_event(0, 10, 0, "load", 0, unit_id=0),
        make_event(20, 10, 0, "load", 1, unit_id=0),
        make_event(0, 10, 0, "load", 0, unit_id=1),
    ]

    with pytest.raises(RuntimeError, match="Expected 2 events.*unit 1: 1"):
        validate_event_counts(events, num_units=2, expected_events_per_unit=2)
    with pytest.raises(RuntimeError, match="Expected 1 events.*unit 0: 2"):
        validate_event_counts(events, num_units=2, expected_events_per_unit=1)


def test_dependency_gaps_pair_by_event_order_and_support_pipeline_offset():
    events = [
        make_event(0, 10, 0, "produce", 1),
        make_event(20, 10, 0, "produce", 5),
        make_event(15, 5, 1, "consume", 3),
        make_event(35, 5, 1, "consume", 7),
        make_event(55, 5, 1, "consume", 11),
    ]

    direct = dependency_gaps(events, "produce", "consume")
    offset = dependency_gaps(events, "produce", "consume", successor_offset=1)

    assert direct.gaps_ns == (5, 5)
    assert direct.count == 2
    assert direct.p50_ns == 5
    assert offset.gaps_ns == (25, 25)


def test_dependency_gaps_empty_and_invalid_offset():
    assert dependency_gaps([], "produce", "consume").count == 0

    with pytest.raises(ValueError, match="successor_offset must be non-negative"):
        dependency_gaps([], "produce", "consume", successor_offset=-1)


def test_link_dependency_flow_pairs_static_slots_with_pipeline_offset():
    def trace_slice(name: str, event_idx: int, ts: float, tid: int) -> dict:
        return {
            "name": name,
            "ph": "X",
            "ts": ts,
            "dur": 1.0,
            "pid": 0,
            "tid": tid,
            "args": {"unit_id": 0, "event_idx": event_idx},
        }

    first_compute = trace_slice("compute", 3, 3.0, 1)
    second_compute = trace_slice("compute", 7, 7.0, 1)
    first_acquire = trace_slice("acquire", 0, 0.0, 0)
    second_acquire = trace_slice("acquire", 4, 4.0, 0)
    third_acquire = trace_slice("acquire", 8, 8.0, 0)
    trace = {
        "traceEvents": [
            second_compute,
            first_acquire,
            third_acquire,
            first_compute,
            second_acquire,
        ]
    }

    processed = link_dependency_flow(
        "compute",
        "acquire",
        successor_offset=1,
        flow_name="release_to_reacquire",
    )(trace, None)

    flows = [event for event in processed["traceEvents"] if event.get("cat") == "pipeline"]
    assert [(event["ph"], event["id"], event["name"]) for event in flows] == [
        ("s", 1, "release_to_reacquire"),
        ("f", 1, "release_to_reacquire"),
        ("s", 2, "release_to_reacquire"),
        ("f", 2, "release_to_reacquire"),
    ]
    assert first_compute["args"]["unblocks"] == "acquire[1]"
    assert second_acquire["args"]["depends_on"] == "compute[0]"
    assert second_compute["args"]["unblocks"] == "acquire[2]"
    assert third_acquire["args"]["depends_on"] == "compute[1]"


def test_link_dependency_flow_rejects_negative_offset():
    with pytest.raises(ValueError, match="successor_offset must be non-negative"):
        link_dependency_flow("produce", "consume", successor_offset=-1)


def test_decode_legacy_payload_records():
    """A tag word with PAYLOAD_FLAG carries a signed payload in its high half; one without
    decodes as before (payload None)."""
    prof = allocate_profile_buffer(2, 2, device="cpu")
    buf = prof.tensor.view(2, prof.slice_size)
    buf[0, 1:5] = torch.tensor([100, 5, 1, 3])  # plain record: tag 1, tid 3
    word = 2 | PAYLOAD_FLAG | (-7 << 32)
    buf[1, 5:9] = torch.tensor([200, 6, word, 4])  # slot 1 of unit 1: tag 2, payload -7
    events = decode_events(prof, TagTable(["a", "plain", "item"]))
    assert [(e.unit_id, e.event_idx, e.tag_name, e.tid, e.payload) for e in events] == [
        (0, 0, "plain", 3, None),
        (1, 1, "item", 4, -7),
    ]


def compact_record(ts_lo: int, dur_ns: int, tag: int) -> int:
    """Pack one compact record: ``tag(8) | dur_ns(24) | ts_lo32(32)``, as a signed int64."""
    word = ts_lo | dur_ns << 32 | tag << 56
    return word - (1 << 64) if word >= 1 << 63 else word


def compact_buffer(units: list[list[int]], max_events: int):
    """A CPU compact profile buffer: each unit's anchor followed by its records."""
    prof = allocate_profile_buffer(max_events, len(units), device="cpu", compact=True)
    grid = prof.tensor.view(len(units), prof.slice_size)
    for unit, words in enumerate(units):
        grid[unit, : len(words)] = torch.tensor(words)
    return prof


def test_compact_decode_reconstructs_starts_around_the_anchor():
    """Another warp can read its start before warp 0 writes the anchor: a start just before the
    anchor decodes there, and one past a 2^32 ns boundary after it decodes after it."""
    anchor = (8 << 32) + 1_000
    wrapped_anchor = (8 << 32) - 50  # unit 1's events start past the 2^32 ns boundary
    prof = compact_buffer(
        [
            [anchor, compact_record(900, 20, 0), compact_record(1_500, 5, 1)],
            [wrapped_anchor, compact_record(30, 10, 1)],
        ],
        max_events=2,
    )

    events = decode_events(prof, TagTable(["early", "late"]))

    assert [(e.unit_id, e.start_ns, e.dur_ns, e.tag_name) for e in events] == [
        (0, anchor - 100, 20, "early"),
        (0, anchor + 500, 5, "late"),
        (1, wrapped_anchor + 80, 10, "late"),
    ]


def test_compact_decode_rejects_records_without_an_anchor():
    prof = compact_buffer([[1 << 40, compact_record(1, 2, 0)], [0, compact_record(1, 2, 0)]], 1)
    with pytest.raises(RuntimeError, match=r"units \[1\] hold 1 record\(s\) but no anchor"):
        decode_events(prof, TagTable(["t"]))


def test_link_payload_flow_links_slices_across_units():
    def trace_slice(name: str, payload: int | None, ts: float, pid: int) -> dict:
        args = {"unit_id": pid} if payload is None else {"unit_id": pid, "payload": payload}
        return {"name": name, "ph": "X", "ts": ts, "dur": 2.0, "pid": pid, "tid": 0, "args": args}

    producer = trace_slice("row", 10, 0.0, 0)
    consumer = trace_slice("row", 11, 5.0, 1)
    other = trace_slice("poll", 11, 4.0, 1)  # another tag with the same payload
    plain = trace_slice("row", None, 1.0, 2)
    trace = {"traceEvents": [producer, consumer, other, plain]}

    processed = link_payload_flow([(10, 11), (10, 99)], "row", flow_name="dep")(trace, None)

    flows = [event for event in processed["traceEvents"] if event.get("cat") == "pipeline"]
    assert [(e["ph"], e["id"], e["pid"], e["ts"]) for e in flows] == [
        ("s", 1, 0, 1.0),
        ("f", 1, 1, 6.0),
    ]  # (10, 99) has no target slice
    assert producer["args"]["unblocks"] == "row[11]"
    assert consumer["args"]["depends_on"] == "row[10]"
    assert "depends_on" not in other["args"]

    duplicate = {"traceEvents": [producer, dict(producer, args=dict(producer["args"]))]}
    with pytest.raises(ValueError, match="payload 10"):
        link_payload_flow([(10, 10)], "row")(duplicate, None)
