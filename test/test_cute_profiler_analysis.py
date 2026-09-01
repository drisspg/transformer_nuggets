import pytest

from transformer_nuggets.cute.profiler import (
    Event,
    dependency_gaps,
    link_dependency_flow,
    overlap_between,
    summarize_by_tag,
    validate_event_counts,
)


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
