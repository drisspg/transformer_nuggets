"""Built-in post-processors for profiler trace customization.

Quick start:
    from transformer_nuggets.cute.profiler.postprocessors import group_by_unit
    with profile_session(..., post_process_events=group_by_unit) as (prof, tags):
        ...
"""

from __future__ import annotations

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
    import re

    for e in events:
        tid_str = str(e.tid)
        patterns = [
            rf"_0*{tid_str}$",
            rf"(?<=[a-zA-Z])0*{tid_str}$",
        ]
        for pattern in patterns:
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
        keep_ids = set()
        if tag_names is not None:
            for name in tag_names:
                keep_ids.add(ctx.tag_table.id(name))
        if tag_ids is not None:
            keep_ids.update(tag_ids)
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


def rename_threads(name_map: dict[tuple[int, int], str] | dict[int, str]):
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
