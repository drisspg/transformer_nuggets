"""Host/Python Utilities for NVIDIA Intra-Kernel Profiling.

This module provides Python-side utilities for allocating profile buffers,
managing tag tables, decoding events, and exporting to Perfetto trace format.

Buffer Layout (all int64):
    buf[0] = event_count
    For each event i (0 <= i < max_events):
        buf[1 + 4*i + 0] = start_ns
        buf[1 + 4*i + 1] = dur_ns
        buf[1 + 4*i + 2] = tag_id
        buf[1 + 4*i + 3] = tid

Usage:
    # Setup
    tag_table = TagTable(["produce", "consume", "wait_lock"])
    TAG_PRODUCE = tag_table.id("produce")  # 0
    TAG_CONSUME = tag_table.id("consume")  # 1

    # Allocate buffer
    prof = allocate_profile_buffer(256, device=x.device)

    # Launch kernel with profiling
    self.kernel(x_c, prof.to_cute(), prof.max_events, TAG_PRODUCE, TAG_CONSUME).launch(...)

    # Decode and export
    events, overflow = decode_events(prof.tensor, tag_table, pid=rank)
    events_to_perfetto(events, "trace.json", pid=rank)

Or use the context manager:
    with profile_session(256, ["produce", "consume"], trace_path="trace.json") as (prof, tags):
        TAG_PRODUCE = tags.id("produce")
        self.kernel(x_c, prof.to_cute(), prof.max_events, TAG_PRODUCE).launch(...)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator

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
    "allocate_profile_buffer",
    "decode_events",
    "events_to_perfetto",
    "profile_session",
]


@dataclass
class Event:
    """A decoded profiling event."""

    start_ns: int
    dur_ns: int
    tag_id: int
    tag_name: str
    tid: int


@dataclass
class ProfileBuf:
    """Profile buffer wrapper with helper methods.

    Attributes:
        tensor: The underlying torch.int64 tensor (length 1 + 4*max_events).
        max_events: Maximum number of events this buffer can hold.
    """

    tensor: torch.Tensor
    max_events: int

    def to_cute(self) -> cute.Tensor:
        """Convert to CUTE tensor for kernel arguments.

        Requires cutlass to be installed.
        """
        if not HAS_CUTE:
            raise ImportError("cutlass is required for to_cute()")
        return from_dlpack(self.tensor)

    def reset(self) -> None:
        """Reset the buffer (zero the event count and all events)."""
        self.tensor.zero_()


class TagTable:
    """Mapping between tag names and integer IDs.

    Use this to define named events on the host and get the corresponding
    integer IDs to pass to kernels.

    Example:
        tag_table = TagTable(["produce", "consume", "wait_lock"])
        TAG_PRODUCE = tag_table.id("produce")  # 0
        TAG_CONSUME = tag_table.id("consume")  # 1
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


def allocate_profile_buffer(
    max_events: int,
    device: torch.device | str | None = None,
    stream: torch.cuda.Stream | None = None,
) -> ProfileBuf:
    """Allocate a profile buffer for intra-kernel profiling.

    Args:
        max_events: Maximum number of events to record. Buffer size is 1 + 4*max_events int64s.
        device: Device to allocate on. Defaults to current CUDA device.
        stream: CUDA stream for allocation (optional).

    Returns:
        ProfileBuf with the allocated tensor and max_events.
    """
    if device is None:
        device = torch.device("cuda")
    elif isinstance(device, str):
        device = torch.device(device)

    # Buffer layout: [count] + [start, dur, tag, tid] * max_events
    size = 1 + 4 * max_events

    if stream is not None:
        with torch.cuda.stream(stream):
            tensor = torch.zeros(size, dtype=torch.int64, device=device)
    else:
        tensor = torch.zeros(size, dtype=torch.int64, device=device)

    return ProfileBuf(tensor=tensor, max_events=max_events)


def decode_events(
    buf: torch.Tensor | ProfileBuf,
    tag_table: TagTable,
    pid: int = 0,
    tid_base: int = 0,
    drop_overflow: bool = True,
) -> tuple[list[Event], int]:
    """Decode profiling events from the buffer.

    Args:
        buf: Profile buffer tensor or ProfileBuf.
        tag_table: TagTable for mapping tag IDs to names.
        pid: Process ID for events (e.g., GPU rank).
        tid_base: Base offset for thread IDs.
        drop_overflow: If True, only return events up to max_events.

    Returns:
        Tuple of (list of Event objects, overflow_count).
        overflow_count is the number of events that exceeded max_events.
    """
    if isinstance(buf, ProfileBuf):
        tensor = buf.tensor
        max_events = buf.max_events
    else:
        tensor = buf
        # Infer max_events from buffer size: size = 1 + 4*max_events
        max_events = (tensor.numel() - 1) // 4

    # Copy to CPU for decoding
    cpu_buf = tensor.cpu().numpy()

    event_count = int(cpu_buf[0])
    overflow_count = max(0, event_count - max_events)

    if drop_overflow:
        num_events = min(event_count, max_events)
    else:
        num_events = event_count

    events = []
    for i in range(num_events):
        if i >= max_events:
            break

        base = 1 + 4 * i
        start_ns = int(cpu_buf[base + 0])
        dur_ns = int(cpu_buf[base + 1])
        tag_id = int(cpu_buf[base + 2])
        tid = int(cpu_buf[base + 3])

        # Skip empty events (start_ns == 0 usually means unwritten)
        if start_ns == 0 and dur_ns == 0:
            continue

        # Map tag_id to name (handle unknown tags)
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
            )
        )

    return events, overflow_count


def events_to_perfetto(
    events: list[Event],
    trace_path: str,
    pid: int = 0,
    tid_prefix: int | None = None,
) -> None:
    """Export events to Perfetto-compatible Chrome trace JSON format.

    The output file can be viewed at https://ui.perfetto.dev/

    Timestamps are normalized relative to the first event and converted to
    microseconds (Chrome trace format default).

    Args:
        events: List of Event objects from decode_events.
        trace_path: Path to write the trace JSON file.
        pid: Process ID (e.g., GPU rank). Used as 'pid' in trace.
        tid_prefix: Optional prefix for thread IDs. If provided, tid = tid_prefix + event.tid.
                    This can satisfy Perfetto's pid/tid quirk where tid should start with pid.
    """
    if not events:
        # Write empty trace
        with open(trace_path, "w") as f:
            json.dump({"traceEvents": []}, f)
        return

    trace_events = []

    # Find minimum start time to normalize timestamps
    min_start_ns = min(e.start_ns for e in events)

    # Add metadata event for process name
    trace_events.append(
        {
            "name": "process_name",
            "ph": "M",  # Metadata
            "pid": pid,
            "args": {"name": f"GPU {pid}"},
        }
    )

    # Add thread name metadata for each unique tid
    unique_tids = sorted({e.tid for e in events})
    for tid in unique_tids:
        actual_tid = tid if tid_prefix is None else tid_prefix + tid
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": actual_tid,
                "args": {"name": f"Block {tid}"},
            }
        )

    for event in events:
        tid = event.tid
        if tid_prefix is not None:
            tid = tid_prefix + event.tid

        # Chrome trace format uses microseconds
        # Normalize timestamps relative to trace start
        ts_us = (event.start_ns - min_start_ns) / 1000.0
        dur_us = event.dur_ns / 1000.0

        # Ensure minimum duration is visible (at least 0.1 us)
        dur_us = max(dur_us, 0.1)

        trace_events.append(
            {
                "name": event.tag_name,
                "cat": "profile",
                "ph": "X",  # Complete event (duration)
                "ts": ts_us,
                "dur": dur_us,
                "pid": pid,
                "tid": tid,
                "args": {
                    "tag_id": event.tag_id,
                    "start_ns": event.start_ns,
                    "dur_ns": event.dur_ns,
                },
            }
        )

    trace = {
        "traceEvents": trace_events,
    }

    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)


@contextmanager
def profile_session(
    max_events: int,
    tag_names: list[str],
    trace_path: str | None = None,
    device: torch.device | str | None = None,
    stream: torch.cuda.Stream | None = None,
    pid: int = 0,
) -> Iterator[tuple[ProfileBuf, TagTable]]:
    """Context manager for profiling a kernel session.

    Allocates a profile buffer, yields it for use in kernel launches,
    then decodes events and optionally writes to Perfetto trace.

    Example:
        with profile_session(256, ["produce", "consume"], trace_path="trace.json") as (prof, tags):
            TAG_PRODUCE = tags.id("produce")
            TAG_CONSUME = tags.id("consume")
            self.kernel(x_c, prof.to_cute(), prof.max_events, TAG_PRODUCE, TAG_CONSUME).launch(...)

    Args:
        max_events: Maximum number of events to record.
        tag_names: List of tag names for this session.
        trace_path: Optional path to write Perfetto trace JSON. If None, no file is written.
        device: Device to allocate on.
        stream: CUDA stream for allocation.
        pid: Process ID for trace (e.g., GPU rank).

    Yields:
        Tuple of (ProfileBuf, TagTable).
    """
    prof = allocate_profile_buffer(max_events, device=device, stream=stream)
    tag_table = TagTable(tag_names)

    yield prof, tag_table

    # Sync to ensure kernel has completed
    if stream is not None:
        stream.synchronize()
    else:
        torch.cuda.synchronize()

    # Decode and optionally write trace
    events, overflow = decode_events(prof, tag_table, pid=pid)

    if overflow > 0:
        import warnings

        warnings.warn(
            f"Profile buffer overflow: {overflow} events dropped. "
            f"Consider increasing max_events (current: {max_events}).",
            stacklevel=2,
        )

    if trace_path is not None:
        events_to_perfetto(events, trace_path, pid=pid)
