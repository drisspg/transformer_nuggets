"""Host/Python Utilities for NVIDIA Intra-Kernel Profiling.

This module provides Python-side utilities for allocating profile buffers,
managing tag tables, decoding events, and exporting to Perfetto trace format.

Static Allocation Buffer Layout (all int64):
    For each unit u (0 <= u < num_units):
        buf[u * slice_size + 0] = event_count for unit u
        buf[u * slice_size + 1 + 4*i + 0] = start_ns
        buf[u * slice_size + 1 + 4*i + 1] = dur_ns
        buf[u * slice_size + 1 + 4*i + 2] = tag_id
        buf[u * slice_size + 1 + 4*i + 3] = tid

    slice_size = 1 + 4 * max_events_per_unit

Usage:
    tag_table = TagTable(["produce", "consume", "wait_lock"])
    produce_tag = tag_table.id("produce")
    consume_tag = tag_table.id("consume")

    prof = allocate_profile_buffer(max_events_per_unit=64, num_units=4, device="cuda")
    self.kernel(x_c, prof.to_cute(), prof.max_events_per_unit, produce_tag, consume_tag).launch(...)

    events = decode_events(prof, tag_table)
    events_to_perfetto(events, "trace.json", pid=0)

Or use the context manager:
    with profile_session(64, num_units=4, tag_names=["produce", "consume"], trace_path="trace.json") as (prof, tags):
        produce_tag = tags.id("produce")
        self.kernel(x_c, prof.to_cute(), prof.max_events_per_unit, produce_tag).launch(...)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Callable, Iterator

import torch

try:
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTE = True
except ImportError:
    HAS_CUTE = False


__all__ = [
    "ProfileBuf",
    "TagTable",
    "Event",
    "PostProcessContext",
    "allocate_profile_buffer",
    "decode_events",
    "events_to_perfetto",
    "profile_session",
]


@dataclass
class Event:
    """A decoded profiling event.

    Attributes:
        start_ns: Event start time in nanoseconds.
        dur_ns: Event duration in nanoseconds.
        tag_id: Integer tag ID from the kernel.
        tag_name: Human-readable tag name from TagTable.
        tid: Thread/lane ID (used as Perfetto tid).
        unit_id: Profiling unit ID (e.g., block index).
        pid: Process ID for grouping in Perfetto (e.g., CTA/block grouping).
        extra_args: Optional dict of extra args to include in Perfetto trace.
    """

    start_ns: int
    dur_ns: int
    tag_id: int
    tag_name: str
    tid: int
    unit_id: int = 0
    pid: int | None = None
    extra_args: dict | None = None


@dataclass
class ProfileBuf:
    """Profile buffer wrapper with helper methods.

    Attributes:
        tensor: The underlying torch.int64 tensor.
        max_events_per_unit: Maximum number of events per profiling unit.
        num_units: Number of profiling units (e.g., blocks, warps).
        unit_name: Name for units in trace output (e.g., "Block", "Warp").
    """

    tensor: torch.Tensor
    max_events_per_unit: int
    num_units: int
    unit_name: str = "Unit"

    @property
    def slice_size(self) -> int:
        """Size of each unit's buffer slice in int64 elements."""
        return 1 + 4 * self.max_events_per_unit

    def to_cute(self) -> cute.Tensor:
        """Convert to CUTE tensor for kernel arguments.

        Requires cutlass to be installed.
        """
        return from_dlpack(self.tensor)

    def reset(self) -> None:
        """Reset the buffer (zero all event counts and data)."""
        self.tensor.zero_()


class TagTable:
    """Mapping between tag names and integer IDs.

    Use this to define named events on the host and get the corresponding
    integer IDs to pass to kernels.

    Example:
        tag_table = TagTable(["produce", "consume", "wait_lock"])
        produce_tag = tag_table.id("produce")
        consume_tag = tag_table.id("consume")
    """

    def __init__(self, tag_names: list[str]):
        """Initialize the tag table.

        Args:
            tag_names: List of tag names. IDs are assigned in order (0, 1, 2, ...).
        """
        self._names = list(tag_names)
        self._name_to_id = {name: i for i, name in enumerate(self._names)}

    def id(self, name: str) -> int:
        """Get the integer ID for a tag name.

        Args:
            name: Tag name.

        Returns:
            Integer tag ID.

        Raises:
            KeyError: If name is not in the tag table.
        """
        return self._name_to_id[name]

    def name(self, tag_id: int) -> str:
        """Get the tag name for an integer ID.

        Args:
            tag_id: Integer tag ID.

        Returns:
            Tag name.

        Raises:
            IndexError: If tag_id is out of range.
        """
        return self._names[tag_id]

    @property
    def names(self) -> list[str]:
        """List of all tag names in order."""
        return list(self._names)

    def __len__(self) -> int:
        return len(self._names)


@dataclass
class PostProcessContext:
    """Context passed to post-processing callbacks.

    Attributes:
        tag_table: TagTable for mapping tag IDs to names.
        prof_buf: The ProfileBuf containing raw event data.
        unit_name: Name for units in trace output (e.g., "Block", "Warp").
    """

    tag_table: TagTable
    prof_buf: ProfileBuf
    unit_name: str


def allocate_profile_buffer(
    max_events_per_unit: int,
    num_units: int | tuple[int, str],
    device: torch.device | str | None = None,
    stream: torch.cuda.Stream | None = None,
) -> ProfileBuf:
    """Allocate a profile buffer for intra-kernel profiling.

    Uses static allocation: each profiling unit (block, warp, etc.) gets its own
    pre-allocated buffer slice. This eliminates atomics from the profiling path.

    Args:
        max_events_per_unit: Maximum events each unit can record.
        num_units: Number of profiling units. Can be:
            - int: Just the count (uses "Unit" as name in traces)
            - tuple[int, str]: (count, name) for nicer trace labels (e.g., (4, "Block"))
        device: Device to allocate on. Defaults to current CUDA device.
        stream: CUDA stream for allocation (optional).

    Returns:
        ProfileBuf with the allocated tensor.

    Buffer size: num_units * (1 + 4 * max_events_per_unit) int64s.
    """
    if isinstance(num_units, tuple):
        num_units_count, unit_name = num_units
    else:
        num_units_count = num_units
        unit_name = "Unit"

    if device is None:
        device = torch.device("cuda")
    elif isinstance(device, str):
        device = torch.device(device)

    slice_size = 1 + 4 * max_events_per_unit
    total_size = num_units_count * slice_size

    if stream is not None:
        with torch.cuda.stream(stream):
            tensor = torch.zeros(total_size, dtype=torch.int64, device=device)
    else:
        tensor = torch.zeros(total_size, dtype=torch.int64, device=device)

    return ProfileBuf(
        tensor=tensor,
        max_events_per_unit=max_events_per_unit,
        num_units=num_units_count,
        unit_name=unit_name,
    )


def decode_events(
    buf: ProfileBuf,
    tag_table: TagTable,
    tid_base: int = 0,
) -> list[Event]:
    """Decode profiling events from the buffer.

    Scans all event slots in each unit's slice, skipping empty slots.
    This works with both atomic mode (counter auto-incremented) and
    static mode (explicit event indices).

    Args:
        buf: Profile buffer (ProfileBuf).
        tag_table: TagTable for mapping tag IDs to names.
        tid_base: Base offset for thread IDs.

    Returns:
        List of Event objects from all units.
    """
    cpu_buf = buf.tensor.cpu().numpy()

    slice_size = buf.slice_size
    events = []

    for unit_id in range(buf.num_units):
        unit_offset = unit_id * slice_size

        for i in range(buf.max_events_per_unit):
            base = unit_offset + 1 + 4 * i
            start_ns = int(cpu_buf[base + 0])
            dur_ns = int(cpu_buf[base + 1])
            tag_id = int(cpu_buf[base + 2])
            tid = int(cpu_buf[base + 3])

            if start_ns == 0 and dur_ns == 0:
                continue

            if 0 <= tag_id < len(tag_table):
                tag_name = tag_table.name(tag_id)
            else:
                tag_name = f"unknown_{tag_id}"

            events.append(
                Event(
                    start_ns=start_ns,
                    dur_ns=dur_ns,
                    tag_id=tag_id,
                    tag_name=tag_name,
                    tid=tid + tid_base,
                    unit_id=unit_id,
                )
            )

    return events


def events_to_perfetto(
    events: list[Event],
    trace_path: str | None = None,
    pid: int = 0,
    tid_prefix: int | None = None,
    unit_name: str = "Unit",
) -> dict:
    """Export events to Perfetto-compatible Chrome trace JSON format.

    The output file can be viewed at https://ui.perfetto.dev/

    Timestamps are normalized relative to the first event and converted to
    microseconds (Chrome trace format default).

    If an event has a custom `pid` set, it will be used instead of the default pid,
    enabling grouping (e.g., one process per CTA/block with warps as threads).

    Args:
        events: List of Event objects from decode_events.
        trace_path: Path to write the trace JSON file. If None, just returns the trace dict.
        pid: Default process ID (e.g., GPU rank). Used as 'pid' in trace unless event.pid is set.
        tid_prefix: Optional prefix for thread IDs. If provided, tid = tid_prefix + event.tid.
                    This can satisfy Perfetto's pid/tid quirk where tid should start with pid.
        unit_name: Name for units in trace (e.g., "Block", "Warp"). Defaults to "Unit".

    Returns:
        The trace dict (can be further modified before writing).
    """
    if not events:
        trace = {"traceEvents": []}
        if trace_path is not None:
            with open(trace_path, "w") as f:
                json.dump(trace, f)
        return trace

    trace_events = []

    min_start_ns = min(e.start_ns for e in events)

    has_custom_pids = any(e.pid is not None for e in events)

    unique_pids = sorted({e.pid if e.pid is not None else pid for e in events})

    pid_tid_to_name: dict[tuple[int, int], str] = {}
    for e in events:
        event_pid = e.pid if e.pid is not None else pid
        key = (event_pid, e.tid)
        if key in pid_tid_to_name:
            continue
        thread_name = e.tag_name if e.pid is not None else f"{unit_name} {e.tid}"
        pid_tid_to_name[key] = thread_name

    pid_tid_pairs = sorted(pid_tid_to_name.keys())

    for p in unique_pids:
        process_name = f"{unit_name} {p}" if has_custom_pids or p != pid else f"GPU {p}"
        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": p,
                "args": {"name": process_name},
            }
        )

    for p, t in pid_tid_pairs:
        actual_tid = t if tid_prefix is None else tid_prefix + t
        thread_name = pid_tid_to_name[(p, t)]
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": p,
                "tid": actual_tid,
                "args": {"name": thread_name},
            }
        )

    for event in events:
        event_pid = event.pid if event.pid is not None else pid
        tid = event.tid
        if tid_prefix is not None:
            tid = tid_prefix + event.tid

        ts_us = (event.start_ns - min_start_ns) / 1000.0
        dur_us = event.dur_ns / 1000.0

        args = {
            "tag_id": event.tag_id,
            "start_ns": event.start_ns,
            "dur_ns": event.dur_ns,
            "unit_id": event.unit_id,
        }
        if event.extra_args:
            args.update(event.extra_args)

        trace_events.append(
            {
                "name": event.tag_name,
                "cat": "profile",
                "ph": "X",
                "ts": ts_us,
                "dur": dur_us,
                "pid": event_pid,
                "tid": tid,
                "args": args,
            }
        )

    trace = {
        "traceEvents": trace_events,
    }

    if trace_path is not None:
        with open(trace_path, "w") as f:
            json.dump(trace, f, indent=2)

    return trace


@contextmanager
def profile_session(
    max_events_per_unit: int,
    num_units: int | tuple[int, str],
    tag_names: list[str],
    trace_path: str | None = None,
    device: torch.device | str | None = None,
    stream: torch.cuda.Stream | None = None,
    pid: int = 0,
    post_process_events: Callable[[list[Event], PostProcessContext], list[Event]] | None = None,
    post_process_trace: Callable[[dict, PostProcessContext], dict] | None = None,
) -> Iterator[tuple[ProfileBuf, TagTable]]:
    """Context manager for profiling a kernel session.

    Allocates a profile buffer, yields it for use in kernel launches,
    then decodes events and optionally writes to Perfetto trace.

    Example:
        with profile_session(
            max_events_per_unit=64,
            num_units=(num_blocks, "Block"),  # Named units for nicer traces
            tag_names=["produce", "consume"],
            trace_path="trace.json"
        ) as (prof, tags):
            TAG_PRODUCE = tags.id("produce")
            TAG_CONSUME = tags.id("consume")
            self.kernel(x_c, prof.to_cute(), prof.max_events_per_unit, TAG_PRODUCE, TAG_CONSUME).launch(...)

    Post-processing example (group by block/warp):
        def group_by_block_warp(events, ctx):
            warps_per_block = 4
            for e in events:
                block_id = e.unit_id
                warp_id = e.tid
                e.pid = block_id  # group rows under the CTA/block
                e.tid = warp_id   # one lane per warp
                e.tag_name = f"warp {warp_id}: {e.tag_name}"
            return events

        with profile_session(..., post_process_events=group_by_block_warp) as (prof, tags):
            ...

    Args:
        max_events_per_unit: Maximum events each unit can record.
        num_units: Number of profiling units. Can be:
            - int: Just the count (uses "Unit" as name in traces)
            - tuple[int, str]: (count, name) for nicer trace labels (e.g., (4, "Block"))
        tag_names: List of tag names for this session.
        trace_path: Optional path to write Perfetto trace JSON. If None, no file is written.
        device: Device to allocate on.
        stream: CUDA stream for allocation.
        pid: Process ID for trace (e.g., GPU rank).
        post_process_events: Optional callback to mutate events before Perfetto conversion.
            Signature: (events, context) -> events. Can rename, filter, or regroup events.
        post_process_trace: Optional callback to mutate the Perfetto trace dict before writing.
            Signature: (trace_dict, context) -> trace_dict. Can add flow events, counters, etc.

    Yields:
        Tuple of (ProfileBuf, TagTable).
    """
    prof = allocate_profile_buffer(
        max_events_per_unit=max_events_per_unit,
        num_units=num_units,
        device=device,
        stream=stream,
    )
    tag_table = TagTable(tag_names)

    yield prof, tag_table

    if stream is not None:
        stream.synchronize()
    else:
        torch.cuda.synchronize()

    events = decode_events(prof, tag_table)

    ctx = PostProcessContext(
        tag_table=tag_table,
        prof_buf=prof,
        unit_name=prof.unit_name,
    )

    if post_process_events is not None:
        events = post_process_events(events, ctx)

    if trace_path is not None:
        trace = events_to_perfetto(events, trace_path=None, pid=pid, unit_name=prof.unit_name)

        if post_process_trace is not None:
            trace = post_process_trace(trace, ctx)

        with open(trace_path, "w") as f:
            json.dump(trace, f, indent=2)
