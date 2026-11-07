"""Built-in post-processors for profiler trace customization.

Quick start:
    from transformer_nuggets.cute.profiler.postprocessors import group_by_unit
    with profile_session(..., post_process_events=group_by_unit) as session:
        ...
"""

from __future__ import annotations

from collections.abc import Iterable

from transformer_nuggets.cute.profiler.host import Event, PostProcessContext


__all__ = [
    "group_by_unit",
    "compose",
    "rename_processes",
    "rename_threads",
    "link_dependency_flow",
    "link_payload_flow",
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


def compose(*processors):
    """Compose multiple post-processors into one.

    Processors are applied left-to-right:
        compose(a, b, c) applies a, then b, then c

    Example:
        with profile_session(
            ...,
            post_process_events=compose(group_by_unit, add_warp_args),
        ) as session:
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
        ) as session:
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


def append_flow(trace: dict, flow_id: int, name: str, source: dict, target: dict) -> None:
    """Append one Perfetto flow arrow from the middle of ``source`` to that of ``target``."""
    trace["traceEvents"].extend(
        {
            "name": name,
            "cat": "pipeline",
            "ph": phase,
            "ts": event["ts"] + event["dur"] / 2,
            "pid": event["pid"],
            "tid": event["tid"],
            "id": flow_id,
        }
        for phase, event in (("s", source), ("f", target))
    )


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
                append_flow(trace, flow_id, relation_name, predecessor, successor)
                flow_id += 1
        return trace

    return _link


def link_payload_flow(
    edges: Iterable[tuple[int, int]],
    predecessor_tag: str,
    successor_tag: str | None = None,
    *,
    flow_name: str | None = None,
):
    """Draw arrows between slices identified by their payload, across profiling units.

    For each ``(source, target)`` pair, links the ``predecessor_tag`` slice recorded with
    payload ``source`` to the ``successor_tag`` slice (default: ``predecessor_tag``) recorded
    with payload ``target``, whichever units recorded them: for example an instruction-table
    row and a row that consumes its output on another CTA. Pairs with a missing slice are
    skipped.

    Args:
        edges: ``(source payload, target payload)`` pairs.
        predecessor_tag: Name of the source duration slices.
        successor_tag: Name of the destination duration slices.
        flow_name: Optional name for the emitted Perfetto flow.

    Returns:
        Trace post-processor that adds the arrows and readable ``unblocks``/``depends_on``
        slice arguments.

    Raises:
        ValueError: When two slices of one tag carry the same payload.
    """
    successor_tag = successor_tag or predecessor_tag
    relation_name = flow_name or f"{predecessor_tag}_to_{successor_tag}"
    edges = list(edges)

    def _link(trace: dict, ctx: PostProcessContext) -> dict:
        del ctx
        slices: dict[tuple[str, int], dict] = {}
        for event in trace["traceEvents"]:
            payload = event.get("args", {}).get("payload")
            if event.get("ph") != "X" or payload is None:
                continue
            if event["name"] not in (predecessor_tag, successor_tag):
                continue
            key = (event["name"], payload)
            if key in slices:
                raise ValueError(f"two {event['name']!r} slices carry payload {payload}")
            slices[key] = event

        flow_id = next_flow_id(trace)
        for source, target in edges:
            predecessor = slices.get((predecessor_tag, source))
            successor = slices.get((successor_tag, target))
            if predecessor is None or successor is None:
                continue
            append_slice_relation(predecessor, "unblocks", f"{successor_tag}[{target}]")
            append_slice_relation(successor, "depends_on", f"{predecessor_tag}[{source}]")
            append_flow(trace, flow_id, relation_name, predecessor, successor)
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
        ) as session:
            ...

        # Or just by tid (applies to all pids)
        with profile_session(
            ...,
            post_process_trace=rename_threads({0: "Warp 0", 1: "Warp 1"}),
        ) as session:
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
