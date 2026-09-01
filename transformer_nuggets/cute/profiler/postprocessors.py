"""Built-in post-processors for profiler trace customization.

Quick start:
    from transformer_nuggets.cute.profiler.postprocessors import group_by_unit
    with profile_session(..., post_process_events=group_by_unit) as (prof, tags):
        ...
"""

from __future__ import annotations

import re

from transformer_nuggets.cute.profiler.host import Event, PostProcessContext


__all__ = [
    "group_by_unit",
    "group_by_tag",
    "strip_tid_suffix",
    "prefix_tag_with_unit",
    "filter_by_tag",
    "compose",
    "rename_processes",
    "rename_threads",
    "link_dependency_flow",
]


def group_by_unit(events: list[Event], ctx: PostProcessContext) -> list[Event]:
    """Group events by unit_id: each unit becomes a Perfetto process with tag-based threads.

    This is the standard "Nsight-style" view where:
    - Each unit_id becomes a separate pid (process row in Perfetto)
    - Process names are "{unit_name} {unit_id}" (e.g., "Block 0", "Block 1")
    - Within each process, events are grouped into threads by tag_id
    - Thread names are the tag_name (e.g., "compute", "store")

    Note: Perfetto displays threads as "{thread_name} {tid}", so avoid putting
    the tid in your tag names (e.g., use "producer" not "producer_0").

    Perfetto will render:
        Block 0 (process)
          ├─ compute 0 (thread)
          ├─ store 1 (thread)
        Block 1 (process)
          ├─ compute 0 (thread)
          ├─ store 1 (thread)

    Args:
        events: List of decoded events.
        ctx: Post-processing context with unit_name.

    Returns:
        Modified events with pid/tid set for unit-based grouping.
    """
    for e in events:
        e.pid = e.unit_id
        e.tid = e.tag_id
    return events


def strip_tid_suffix(events: list[Event], ctx: PostProcessContext) -> list[Event]:
    """Strip trailing tid numbers from tag names to avoid Perfetto duplication.

    Perfetto displays threads as "{thread_name} {tid}". If your tag names already
    contain the tid (e.g., "warp_0", "consumer_1"), this results in redundant
    display like "warp_0 0".

    This post-processor strips trailing "_N" or "N" suffixes that match the tid:
    - "consumer_warp1" with tid=1 → "consumer_warp"
    - "warp_0" with tid=0 → "warp"
    - "compute" with tid=2 → "compute" (unchanged)

    Use after group_by_unit:
        post_process_events=compose(group_by_unit, strip_tid_suffix)

    Args:
        events: List of decoded events.
        ctx: Post-processing context.

    Returns:
        Events with cleaned tag_names.
    """
    for e in events:
        tid_str = str(e.tid)
        for pattern in (rf"_0*{tid_str}$", rf"(?<=[a-zA-Z])0*{tid_str}$"):
            new_name = re.sub(pattern, "", e.tag_name)
            if new_name != e.tag_name:
                e.tag_name = new_name
                break
    return events


def group_by_tag(events: list[Event], ctx: PostProcessContext) -> list[Event]:
    """Group events by tag: each tag becomes a Perfetto process with unit-based threads.

    Inverse of group_by_unit. Useful when you want to compare the same operation
    across different units (e.g., see all "compute" events together).

    Perfetto will render:
        compute (process)
          ├─ Block 0 (thread)
          ├─ Block 1 (thread)
        store (process)
          ├─ Block 0 (thread)
          ├─ Block 1 (thread)

    Args:
        events: List of decoded events.
        ctx: Post-processing context.

    Returns:
        Modified events with pid/tid swapped for tag-based grouping.
    """
    for e in events:
        e.pid = e.tag_id
        e.tid = e.unit_id
    return events


def prefix_tag_with_unit(events: list[Event], ctx: PostProcessContext) -> list[Event]:
    """Prefix each event's tag_name with its unit_id.

    Transforms: "compute" → "Block 0: compute"

    Useful when viewing all events in a single flat list but wanting
    to distinguish which unit each event came from.

    Args:
        events: List of decoded events.
        ctx: Post-processing context with unit_name.

    Returns:
        Events with modified tag_names.
    """
    for e in events:
        e.tag_name = f"{ctx.unit_name} {e.unit_id}: {e.tag_name}"
    return events


def filter_by_tag(
    tag_names: list[str] | None = None,
    tag_ids: list[int] | None = None,
):
    """Create a post-processor that filters events to only specified tags.

    Use this factory to create a filtering post-processor:

        with profile_session(
            ...,
            post_process_events=filter_by_tag(tag_names=["compute"]),
        ) as (prof, tags):
            ...

    Args:
        tag_names: List of tag names to keep. If None, uses tag_ids.
        tag_ids: List of tag IDs to keep. If None, uses tag_names.

    Returns:
        A post-processor function that filters events.
    """

    def _filter(events: list[Event], ctx: PostProcessContext) -> list[Event]:
        keep_ids = set(tag_ids or ())
        if tag_names is not None:
            keep_ids.update(ctx.tag_table.id(name) for name in tag_names)
        return [e for e in events if e.tag_id in keep_ids]

    return _filter


def compose(*processors):
    """Compose multiple post-processors into one.

    Processors are applied left-to-right:
        compose(a, b, c) applies a, then b, then c

    Example:
        with profile_session(
            ...,
            post_process_events=compose(
                filter_by_tag(tag_names=["compute", "store"]),
                group_by_unit,
            ),
        ) as (prof, tags):
            ...

    Args:
        *processors: Post-processor functions to compose.

    Returns:
        A single post-processor that applies all in sequence.
    """

    def _composed(events: list[Event], ctx: PostProcessContext) -> list[Event]:
        for proc in processors:
            events = proc(events, ctx)
        return events

    return _composed


def rename_processes(name_map: dict[int, str]):
    """Create a trace post-processor that renames processes.

    Use with post_process_trace to customize process names in Perfetto.

    Example:
        with profile_session(
            ...,
            post_process_events=group_by_unit,
            post_process_trace=rename_processes({0: "Producer CTA", 1: "Consumer CTA"}),
        ) as (prof, tags):
            ...

    Args:
        name_map: Dict mapping pid -> custom name.

    Returns:
        A trace post-processor function.
    """

    def _rename(trace: dict, ctx: PostProcessContext) -> dict:
        for event in trace["traceEvents"]:
            if event.get("ph") == "M" and event.get("name") == "process_name":
                pid = event.get("pid")
                if pid in name_map:
                    event["args"]["name"] = name_map[pid]
        return trace

    return _rename


def append_slice_relation(event: dict, key: str, label: str) -> None:
    """Append one readable dependency label to a duration slice."""
    args = event.setdefault("args", {})
    args[key] = f"{args[key]}, {label}" if key in args else label


def trace_event_order(event: dict) -> tuple[bool, int, float]:
    """Order profiler slices by static event slot, falling back to timestamp."""
    event_idx = event.get("args", {}).get("event_idx")
    return event_idx is None, event_idx if event_idx is not None else 0, event["ts"]


def next_flow_id(trace: dict) -> int:
    """Return an unused positive flow ID for a Chrome trace dictionary."""
    flow_ids = [
        event["id"]
        for event in trace["traceEvents"]
        if event.get("ph") in {"s", "t", "f"} and isinstance(event.get("id"), int)
    ]
    return max(flow_ids, default=0) + 1


def link_dependency_flow(
    predecessor_tag: str,
    successor_tag: str,
    *,
    successor_offset: int = 0,
    flow_name: str | None = None,
):
    """Draw one repeated producer-to-consumer dependency in Perfetto.

    Slices are paired independently within each profiling unit after ordering by
    ``event_idx``. ``successor_offset=N`` links predecessor ``i`` to successor
    ``i + N``, matching :func:`dependency_gaps` for fixed-depth pipelines.

    Args:
        predecessor_tag: Name of the source duration slices.
        successor_tag: Name of the destination duration slices.
        successor_offset: Positional offset applied to the successor stream.
        flow_name: Optional name for the emitted Perfetto flow.

    Returns:
        Trace post-processor that adds direct dependency arrows and readable
        ``unblocks``/``depends_on`` slice arguments.
    """
    if successor_offset < 0:
        raise ValueError("successor_offset must be non-negative")
    relation_name = flow_name or f"{predecessor_tag}_to_{successor_tag}"

    def _link(trace: dict, ctx: PostProcessContext) -> dict:
        del ctx
        by_unit: dict[int, dict[str, list[dict]]] = {}
        for event in trace["traceEvents"]:
            if event.get("ph") != "X" or event.get("name") not in {
                predecessor_tag,
                successor_tag,
            }:
                continue
            unit_id = event.get("args", {}).get("unit_id")
            if unit_id is None:
                continue
            by_unit.setdefault(unit_id, {}).setdefault(event["name"], []).append(event)

        flow_id = next_flow_id(trace)
        for tags in by_unit.values():
            predecessors = sorted(tags.get(predecessor_tag, ()), key=trace_event_order)
            successors = sorted(tags.get(successor_tag, ()), key=trace_event_order)
            pair_count = min(len(predecessors), max(0, len(successors) - successor_offset))
            for pair_idx in range(pair_count):
                predecessor = predecessors[pair_idx]
                successor_idx = pair_idx + successor_offset
                successor = successors[successor_idx]
                predecessor_label = f"{predecessor_tag}[{pair_idx}]"
                successor_label = f"{successor_tag}[{successor_idx}]"
                append_slice_relation(predecessor, "unblocks", successor_label)
                append_slice_relation(successor, "depends_on", predecessor_label)
                trace["traceEvents"].extend(
                    (
                        {
                            "name": relation_name,
                            "cat": "pipeline",
                            "ph": "s",
                            "ts": predecessor["ts"] + predecessor["dur"] / 2,
                            "pid": predecessor["pid"],
                            "tid": predecessor["tid"],
                            "id": flow_id,
                        },
                        {
                            "name": relation_name,
                            "cat": "pipeline",
                            "ph": "f",
                            "ts": successor["ts"] + successor["dur"] / 2,
                            "pid": successor["pid"],
                            "tid": successor["tid"],
                            "id": flow_id,
                        },
                    )
                )
                flow_id += 1
        return trace

    return _link


def rename_threads(name_map: dict[tuple[int, int] | int, str]):
    """Create a trace post-processor that renames threads.

    Use with post_process_trace to customize thread names in Perfetto.

    Example:
        # Rename by (pid, tid) tuple
        with profile_session(
            ...,
            post_process_trace=rename_threads({(0, 0): "Main Warp", (0, 1): "Helper Warp"}),
        ) as (prof, tags):
            ...

        # Or just by tid (applies to all pids)
        with profile_session(
            ...,
            post_process_trace=rename_threads({0: "Warp 0", 1: "Warp 1"}),
        ) as (prof, tags):
            ...

    Args:
        name_map: Dict mapping (pid, tid) or tid -> custom name.

    Returns:
        A trace post-processor function.
    """

    def _rename(trace: dict, ctx: PostProcessContext) -> dict:
        for event in trace["traceEvents"]:
            if event.get("ph") == "M" and event.get("name") == "thread_name":
                pid = event.get("pid")
                tid = event.get("tid")
                if (pid, tid) in name_map:
                    event["args"]["name"] = name_map[(pid, tid)]
                elif tid in name_map:
                    event["args"]["name"] = name_map[tid]
        return trace

    return _rename
