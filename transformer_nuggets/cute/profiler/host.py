"""Host/Python Utilities for NVIDIA Intra-Kernel Profiling.

This module provides Python-side utilities for allocating profile buffers,
managing tag tables, decoding events, and exporting to native Perfetto TrackEvent traces.

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
    events_to_perfetto(events, "trace.pftrace", pid=0)

Or use the context manager:
    with profile_session(
        64,
        num_units=4,
        tag_names=["produce", "consume"],
        trace_path="trace.pftrace",
    ) as (prof, tags):
        produce_tag = tags.id("produce")
        self.kernel(x_c, prof.to_cute(), prof.max_events_per_unit, produce_tag).launch(...)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Callable, Iterator
from typing import Literal

import torch

from transformer_nuggets.utils.perfetto import write_perfetto_trace

try:
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
except ImportError:
    pass


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
        compact: ``True`` if events use the 1xi64 packed format
            ``[tag(8) | dur_ns(24) | ts_lo32(32)]`` with slot 0 holding a
            64-bit anchor; ``False`` for the legacy 4xi64 record format.
    """

    tensor: torch.Tensor
    max_events_per_unit: int
    num_units: int
    unit_name: str = "Unit"
    compact: bool = False

    @property
    def slice_size(self) -> int:
        """Size of each unit's buffer slice in int64 elements."""
        if self.compact:
            return 1 + self.max_events_per_unit
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
    compact: bool = False,
) -> ProfileBuf:
    """Allocate a profile buffer for intra-kernel profiling.

    Each profiling unit (block, warp, ...) owns a pre-allocated slice of the
    buffer; no atomics are needed on the recording path.

    Args:
        max_events_per_unit: Maximum events each unit can record.
        num_units: ``int`` (count, uses ``"Unit"`` as label) or
            ``(count, name)`` for nicer trace labels (e.g. ``(4, "Block")``).
        device: Device to allocate on. Defaults to current CUDA device.
        stream: Optional CUDA stream for allocation.
        compact: ``True`` selects the 1xi64 packed record format used by
            :func:`compact_event_stop` (slice size = ``1 + max_events``).
            ``False`` keeps the legacy 4xi64 records (slice size =
            ``1 + 4 * max_events``).

    Returns:
        ProfileBuf with the allocated tensor.
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

    slice_size = (1 + max_events_per_unit) if compact else (1 + 4 * max_events_per_unit)
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
        compact=compact,
    )


def decode_events(
    buf: ProfileBuf,
    tag_table: TagTable,
    tid_base: int = 0,
) -> list[Event]:
    """Decode profiling events from the buffer.

    Dispatches on ``buf.compact``: legacy 4xi64 records or the 1xi64 packed
    format produced by :func:`compact_event_stop`. Empty slots are skipped.

    Args:
        buf: Profile buffer (ProfileBuf).
        tag_table: TagTable for mapping tag IDs to names.
        tid_base: Base offset added to ``Event.tid``.

    Returns:
        List of Event objects from all units.
    """
    if buf.compact:
        return _decode_events_compact(buf, tag_table, tid_base)
    return _decode_events_legacy(buf, tag_table, tid_base)


def _tag_name_or_unknown(tag_table: TagTable, tag_id: int) -> str:
    if 0 <= tag_id < len(tag_table):
        return tag_table.name(tag_id)
    return f"unknown_{tag_id}"


def _decode_events_legacy(buf: ProfileBuf, tag_table: TagTable, tid_base: int) -> list[Event]:
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

            events.append(
                Event(
                    start_ns=start_ns,
                    dur_ns=dur_ns,
                    tag_id=tag_id,
                    tag_name=_tag_name_or_unknown(tag_table, tag_id),
                    tid=tid + tid_base,
                    unit_id=unit_id,
                )
            )

    return events


def _decode_events_compact(buf: ProfileBuf, tag_table: TagTable, tid_base: int) -> list[Event]:
    """Decode the 1xi64 packed records, reconstructing 64-bit ts from each unit's anchor.

    Bit-unpacks every slot in one shot with torch, masks out empty / unitless
    rows, and only iterates Python over the surviving valid events.
    """
    mask32 = (1 << 32) - 1
    mask24 = (1 << 24) - 1

    grid = buf.tensor.cpu().to(torch.int64).view(buf.num_units, buf.slice_size)
    anchors = grid[:, 0]
    records = grid[:, 1:]

    # CTAs in the same launch start within microseconds, so all real anchors
    # should agree to within a small window. An anchor diverging from the
    # median by >1s almost certainly means a caller event_idx overflowed and
    # clobbered the next unit's anchor slot.
    valid_anchor = anchors != 0
    if valid_anchor.any():
        median = anchors[valid_anchor].median()
        suspect = valid_anchor & ((anchors - median).abs() > 1_000_000_000)
        if suspect.any():
            bad = suspect.nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"Compact-mode anchors for units {bad} diverge from the median by >1s; "
                "likely an event_idx >= max_events_per_unit overflow clobbered the next "
                "unit's anchor. Bump max_events_per_unit or fix caller slot allocation."
            )

    valid = (records != 0) & valid_anchor.unsqueeze(1)
    if not valid.any():
        return []

    ts_lo = records & mask32
    dur_ns = (records >> 32) & mask24
    tag_ids = (records >> 56) & 0xFF
    anchor_lo = anchors & mask32
    anchor_hi = (anchors >> 32) << 32
    wrap = (ts_lo < anchor_lo.unsqueeze(1)).to(torch.int64) << 32
    start_ns = anchor_hi.unsqueeze(1) + wrap + ts_lo

    starts = start_ns[valid].tolist()
    durs = dur_ns[valid].tolist()
    tags = tag_ids[valid].tolist()
    units = torch.nonzero(valid, as_tuple=True)[0].tolist()

    events = []
    for s, d, t, u in zip(starts, durs, tags, units):
        events.append(
            Event(
                start_ns=s,
                dur_ns=d,
                tag_id=t,
                tag_name=_tag_name_or_unknown(tag_table, t),
                tid=u + tid_base,
                unit_id=u,
            )
        )
    return events


def events_to_perfetto(
    events: list[Event],
    trace_path: str | None = None,
    pid: int = 0,
    tid_prefix: int | None = None,
    unit_name: str = "Unit",
    split_overlaps: bool = True,
    trace_format: Literal["chrome_json", "track_event"] = "track_event",
) -> dict:
    """Export events to a Perfetto trace.

    Native TrackEvent ``.pftrace`` output is the default. Pass
    ``trace_format="chrome_json"`` to write legacy Chrome JSON/JSON.GZ.

    Timestamps are normalized relative to the first event and converted to
    microseconds in the intermediate trace dict.

    If an event has a custom `pid` set, it will be used instead of the default pid,
    enabling grouping (e.g., one process per CTA/block with warps as threads).

    Args:
        events: List of Event objects from decode_events.
        trace_path: Path to write the trace file. If None, just returns the
            intermediate trace dict.
        pid: Default process ID (e.g., GPU rank). Used as 'pid' in trace unless event.pid is set.
        tid_prefix: Optional prefix for thread IDs. If provided, tid = tid_prefix + event.tid.
                    This can satisfy Perfetto's pid/tid quirk where tid should start with pid.
        unit_name: Name for units in trace (e.g., "Block", "Warp"). Defaults to "Unit".
        split_overlaps: If true, split overlapping duration slices on the same
            track into sibling lanes/tracks before writing the trace.
        trace_format: ``"track_event"`` writes native Perfetto ``.pftrace`` output;
            ``"chrome_json"`` writes Chrome JSON/JSON.GZ output.

    Returns:
        The trace dict (can be further modified before writing).
    """
    if not events:
        trace = {"traceEvents": []}
        if trace_path is not None:
            write_perfetto_trace(trace_path, trace, trace_format=trace_format)
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
        write_perfetto_trace(
            trace_path,
            trace,
            trace_format=trace_format,
            split_overlaps=split_overlaps,
        )

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
    split_overlaps: bool = True,
    trace_format: Literal["chrome_json", "track_event"] = "track_event",
    compact: bool = False,
) -> Iterator[tuple[ProfileBuf, TagTable]]:
    """Context manager for profiling a kernel session.

    Allocates a profile buffer, yields it for use in kernel launches,
    then decodes events and optionally writes to Perfetto trace.

    Example:
        with profile_session(
            max_events_per_unit=64,
            num_units=(num_blocks, "Block"),  # Named units for nicer traces
            tag_names=["produce", "consume"],
            trace_path="trace.pftrace"
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
        split_overlaps: If true, split overlapping duration slices on the same
            Perfetto track into sibling lanes/tracks before writing the trace.
        trace_format: ``"track_event"`` writes native Perfetto ``.pftrace`` output;
            ``"chrome_json"`` writes Chrome JSON/JSON.GZ output.
        compact: Use the 1xi64 packed record format (paired with
            ``compact_event_stop`` / ``compact_anchor_init`` on the device
            side). Slot 0 of each unit's slice is the 64-bit anchor; the
            decoder reconstructs full timestamps from the anchor + each
            record's low-32-bit timer reading.

    Yields:
        Tuple of (ProfileBuf, TagTable).
    """
    prof = allocate_profile_buffer(
        max_events_per_unit=max_events_per_unit,
        num_units=num_units,
        device=device,
        stream=stream,
        compact=compact,
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
        trace = events_to_perfetto(
            events,
            trace_path=None,
            pid=pid,
            unit_name=prof.unit_name,
            split_overlaps=False,
        )

        if post_process_trace is not None:
            trace = post_process_trace(trace, ctx)

        write_perfetto_trace(
            trace_path,
            trace,
            trace_format=trace_format,
            split_overlaps=split_overlaps,
        )
