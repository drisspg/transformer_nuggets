"""Host-side summaries for decoded CuTeDSL intra-kernel profile events."""

from __future__ import annotations

from dataclasses import dataclass
import math

from transformer_nuggets.cute.profiler.host import Event


__all__ = [
    "EventSummary",
    "OverlapSummary",
    "DependencyGapSummary",
    "summarize_by_tag",
    "overlap_between",
    "dependency_gaps",
]


@dataclass(frozen=True)
class EventSummary:
    """Aggregate durations for one profile tag."""

    count: int
    total_ns: int
    mean_ns: float
    min_ns: int
    p50_ns: float
    p95_ns: float
    max_ns: int


@dataclass(frozen=True)
class OverlapSummary:
    """Intersection of two tagged event streams."""

    overlap_ns: int
    left_total_ns: int
    right_total_ns: int
    left_fraction: float
    right_fraction: float


@dataclass(frozen=True)
class DependencyGapSummary:
    """Time from predecessor completion to matched successor start."""

    gaps_ns: tuple[int, ...]
    count: int
    min_ns: int | None
    p50_ns: float | None
    p95_ns: float | None
    max_ns: int | None


def percentile(values: list[int], quantile: float) -> float:
    """Return a linearly interpolated percentile for non-empty integer values."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if not values:
        raise ValueError("percentile requires at least one value")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_by_tag(events: list[Event]) -> dict[str, EventSummary]:
    """Summarize event durations by human-readable tag name."""
    durations_by_tag: dict[str, list[int]] = {}
    for event in events:
        durations_by_tag.setdefault(event.tag_name, []).append(event.dur_ns)

    return {
        tag_name: EventSummary(
            count=len(durations),
            total_ns=sum(durations),
            mean_ns=sum(durations) / len(durations),
            min_ns=min(durations),
            p50_ns=percentile(durations, 0.50),
            p95_ns=percentile(durations, 0.95),
            max_ns=max(durations),
        )
        for tag_name, durations in durations_by_tag.items()
    }


def matches_tag(event: Event, tag: int | str) -> bool:
    """Return whether an event matches a numeric ID or human-readable name."""
    if isinstance(tag, int):
        return event.tag_id == tag
    return event.tag_name == tag


def merge_intervals(events: list[Event]) -> list[tuple[int, int]]:
    """Merge overlapping event intervals into a sorted disjoint list."""
    intervals = sorted((event.start_ns, event.start_ns + event.dur_ns) for event in events)
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def interval_total(intervals: list[tuple[int, int]]) -> int:
    """Return the total duration of disjoint intervals."""
    return sum(end - start for start, end in intervals)


def interval_overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    """Return the intersection duration of two sorted disjoint interval lists."""
    left_idx = 0
    right_idx = 0
    overlap_ns = 0
    while left_idx < len(left) and right_idx < len(right):
        left_start, left_end = left[left_idx]
        right_start, right_end = right[right_idx]
        overlap_ns += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_idx += 1
        else:
            right_idx += 1
    return overlap_ns


def overlap_between(
    events: list[Event],
    left_tag: int | str,
    right_tag: int | str,
    *,
    same_unit: bool = True,
) -> OverlapSummary:
    """Measure time where two tagged event streams are simultaneously active.

    By default, intersections are computed independently within each ``unit_id``
    so unrelated CTAs cannot create artificial overlap. Returned durations and
    fractions aggregate those per-unit interval unions; inspect units separately
    when CTA-to-CTA variance matters.
    """
    if same_unit:
        event_groups = [
            [event for event in events if event.unit_id == unit_id]
            for unit_id in sorted({event.unit_id for event in events})
        ]
    else:
        event_groups = [events]

    overlap_ns = 0
    left_total_ns = 0
    right_total_ns = 0

    for unit_events in event_groups:
        left = merge_intervals([event for event in unit_events if matches_tag(event, left_tag)])
        right = merge_intervals([event for event in unit_events if matches_tag(event, right_tag)])
        left_total_ns += interval_total(left)
        right_total_ns += interval_total(right)
        overlap_ns += interval_overlap(left, right)

    return OverlapSummary(
        overlap_ns=overlap_ns,
        left_total_ns=left_total_ns,
        right_total_ns=right_total_ns,
        left_fraction=overlap_ns / left_total_ns if left_total_ns else 0.0,
        right_fraction=overlap_ns / right_total_ns if right_total_ns else 0.0,
    )


def event_order(event: Event) -> tuple[bool, int, int]:
    """Order static-slot events first by slot and synthesized events by time."""
    return (
        event.event_idx is None,
        event.event_idx if event.event_idx is not None else 0,
        event.start_ns,
    )


def dependency_gaps(
    events: list[Event],
    predecessor_tag: int | str,
    successor_tag: int | str,
    *,
    successor_offset: int = 0,
) -> DependencyGapSummary:
    """Summarize predecessor-end to successor-start gaps within each unit.

    Repeated tags are ordered by ``event_idx`` when available. ``successor_offset``
    pairs predecessor ``i`` with successor ``i + successor_offset``, which covers
    fixed-depth pipelines such as refill ``k`` enabling wait ``k + depth``.
    Pairing is positional after sorting, so it assumes complete event streams;
    use ``expected_events_per_unit`` during capture to catch dropped records.
    Legacy atomic slots represent allocation order and are only suitable when
    that order is deterministic for the selected tags. Negative gaps mean the
    selected regions overlap and therefore do not form a completed-before-start
    dependency under the chosen pairing.
    """
    if successor_offset < 0:
        raise ValueError("successor_offset must be non-negative")

    gaps: list[int] = []
    for unit_id in sorted({event.unit_id for event in events}):
        unit_events = [event for event in events if event.unit_id == unit_id]
        predecessors = sorted(
            (event for event in unit_events if matches_tag(event, predecessor_tag)),
            key=event_order,
        )
        successors = sorted(
            (event for event in unit_events if matches_tag(event, successor_tag)),
            key=event_order,
        )
        pair_count = min(len(predecessors), max(0, len(successors) - successor_offset))
        for pair_idx in range(pair_count):
            predecessor = predecessors[pair_idx]
            successor = successors[pair_idx + successor_offset]
            gaps.append(successor.start_ns - (predecessor.start_ns + predecessor.dur_ns))

    if not gaps:
        return DependencyGapSummary((), 0, None, None, None, None)

    return DependencyGapSummary(
        gaps_ns=tuple(gaps),
        count=len(gaps),
        min_ns=min(gaps),
        p50_ns=percentile(gaps, 0.50),
        p95_ns=percentile(gaps, 0.95),
        max_ns=max(gaps),
    )
