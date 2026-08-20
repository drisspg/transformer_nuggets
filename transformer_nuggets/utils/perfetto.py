"""Perfetto/Chrome trace helpers.

Supported output formats:

- ``track_event``: native Perfetto protobuf traces, written as ``.pftrace`` or
  ``.perfetto-trace``. This is the default for programmatically generated
  traces. It uses explicit TrackEvent descriptors and supports merged backing
  tracks for crossing overlaps on one logical timeline.
- ``chrome_json``: legacy Chrome JSON/JSON.GZ traces. This remains useful for
  compatibility with tools that only consume Chrome JSON, but Perfetto treats
  this format as best-effort and requires duration events on a track to nest.

The Chrome JSON helpers intentionally operate on plain trace dictionaries so
both torch.profiler output and transformer-nuggets' lightweight CuTe profiler
can reuse them.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import zlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Any, Literal

from transformer_nuggets.utils.track_event import (
    default_track_event_path,
    write_track_event_trace,
)

TraceFormat = Literal["chrome_json", "track_event"]
"""Perfetto-compatible output format selector."""


@contextmanager
def open_trace(path: str | Path, mode: str) -> Iterator[Any]:
    """Open a trace file as text, transparently handling ``.gz`` paths."""
    path = Path(path)
    text_mode = mode if "t" in mode else f"{mode}t"
    if path.suffix == ".gz":
        with gzip.open(path, text_mode, encoding="utf-8") as f:
            yield f
    else:
        with open(path, text_mode, encoding="utf-8") as f:
            yield f


def read_trace(path: str | Path) -> dict[str, Any]:
    """Read a Chrome/Perfetto JSON trace from a plain or gzipped file."""
    with open_trace(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {"traceEvents": data}


def write_trace(path: str | Path, trace: dict[str, Any], *, indent: int | None = 2) -> None:
    """Write a Chrome/Perfetto JSON trace to a plain or gzipped file."""
    with open_trace(path, "w") as f:
        json.dump(trace, f, indent=indent)


def _annotation_label(entries: Sequence[Any] | None) -> str | None:
    metadata: dict[str, Any] = {}
    for annotation in entries or ():
        if isinstance(annotation, dict):
            metadata.update(annotation)
        elif isinstance(annotation, str):
            metadata["name"] = annotation

    name = metadata.get("name")
    if name is None:
        return None
    name = str(name)
    autograd_phase = metadata.get("autograd_phase")
    if autograd_phase == "backward" and name != "backward":
        return f"{name} backward"
    return name


_GPU_EVENT_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
_GRAPH_ANNOTATION_MARKER = "transformer_nuggets.graph_annotation"
_GRAPH_ANNOTATION_SOURCE = "transformer_nuggets.graph_annotation_source"


@dataclass(frozen=True)
class _GpuTraceEvent:
    """Validated GPU work event used to construct graph annotation spans."""

    pid: int | str | None
    tid: int | str | None
    device: int | str | None
    graph_id: int | None
    graph_node_id: int | None
    correlation: int | str | None
    name: str | None
    start: float
    end: float
    index: int


def _graph_annotation_box(
    name: str,
    pid: Any,
    tid: Any,
    start: float,
    end: float,
    source: str,
) -> dict[str, Any]:
    """Build a marked annotation box for one graph replay source."""
    return {
        "ph": "X",
        "cat": "gpu_user_annotation",
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": start,
        "dur": end - start,
        "args": {
            _GRAPH_ANNOTATION_MARKER: True,
            _GRAPH_ANNOTATION_SOURCE: source,
        },
    }


def _trace_identity(value: Any) -> int | str | None:
    """Validate a Chrome trace identity field used for grouping GPU work."""
    if value is None or (isinstance(value, (int, str)) and not isinstance(value, bool)):
        return value
    return None


def _finite_float(value: Any) -> float | None:
    """Return a finite trace timestamp or duration, skipping malformed values."""
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _embedded_annotation(args: Mapping[str, Any]) -> Sequence[Any] | None:
    """Read an event-local annotation list, which overrides the global registry."""
    if "annotation" not in args:
        return None
    embedded = args["annotation"]
    if isinstance(embedded, str):
        try:
            embedded = json.loads(embedded)
        except json.JSONDecodeError:
            return ()
    return embedded if isinstance(embedded, list) else ()


def _gpu_trace_event(
    event: Any,
    index: int,
    annotations: Mapping[int, Sequence[Any]] | None,
) -> _GpuTraceEvent | None:
    """Normalize one GPU duration event, returning ``None`` for malformed input."""
    if not isinstance(event, Mapping):
        return None
    if event.get("ph") != "X" or event.get("cat") not in _GPU_EVENT_CATEGORIES:
        return None
    args = event.get("args", {})
    if not isinstance(args, Mapping):
        return None

    start = _finite_float(event.get("ts"))
    duration = _finite_float(event.get("dur", 0.0))
    if start is None or duration is None or duration < 0:
        return None

    pid = _trace_identity(event.get("pid"))
    tid = _trace_identity(event.get("tid"))
    if (event.get("pid") is not None and pid is None) or (
        event.get("tid") is not None and tid is None
    ):
        return None
    device_value = next(
        (args[name] for name in ("device", "device id", "device_id") if name in args), None
    )
    device = _trace_identity(device_value)
    if device_value is not None and device is None:
        return None

    graph_id = args.get("graph id")
    graph_node_id = args.get("graph node id")
    if graph_id is None or graph_node_id is None:
        return _GpuTraceEvent(
            pid, tid, device, None, None, None, None, start, start + duration, index
        )
    if isinstance(graph_id, bool) or isinstance(graph_node_id, bool):
        return None
    try:
        graph_id = int(graph_id)
        graph_node_id = int(graph_node_id)
    except (TypeError, ValueError):
        return None

    correlation = _trace_identity(args.get("correlation"))
    if args.get("correlation") is not None and correlation is None:
        return None
    entries = _embedded_annotation(args)
    if entries is None and annotations is not None:
        entries = annotations.get((graph_id << 32) | graph_node_id)
    return _GpuTraceEvent(
        pid,
        tid,
        device,
        graph_id,
        graph_node_id,
        correlation,
        _annotation_label(entries),
        start,
        start + duration,
        index,
    )


def _annotation_source(
    event: _GpuTraceEvent,
    replay: tuple[str, int | str],
) -> str:
    """Serialize the physical track and replay identity that generated a box."""
    return json.dumps(
        [event.pid, event.tid, event.device, event.graph_id, *replay],
        separators=(",", ":"),
    )


def _generated_annotation_sources(events: Sequence[Any]) -> set[str]:
    """Return sources with annotation boxes already synthesized in this trace."""
    sources = set()
    for event in events:
        if not isinstance(event, Mapping):
            continue
        args = event.get("args")
        if not isinstance(args, Mapping) or not args.get(_GRAPH_ANNOTATION_MARKER):
            continue
        source = args.get(_GRAPH_ANNOTATION_SOURCE)
        if isinstance(source, str):
            sources.add(source)
    return sources


def add_cuda_graph_annotation_boxes(
    trace: dict[str, Any],
    annotations: Mapping[int, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Add GPU annotation spans for contiguous CUDA Graph regions.

    Graph annotation metadata is joined by ``(graph id, graph node id)`` when an
    annotation registry is supplied. Event-local ``annotation`` metadata takes
    precedence. Boxes never cross unannotated GPU work, physical-device tracks,
    graph/replay boundaries, or a repeated graph node when correlation is absent.
    """
    raw_events = trace.get("traceEvents", ())
    events = list(raw_events) if isinstance(raw_events, Sequence) else []
    if any(
        isinstance(event, Mapping)
        and isinstance(event.get("args"), Mapping)
        and event["args"].get(_GRAPH_ANNOTATION_MARKER)
        and _GRAPH_ANNOTATION_SOURCE not in event["args"]
        for event in events
    ):
        return trace.copy()
    generated_sources = _generated_annotation_sources(events)
    work_by_track: dict[
        tuple[int | str | None, int | str | None, int | str | None], list[_GpuTraceEvent]
    ] = defaultdict(list)
    for index, event in enumerate(events):
        gpu_event = _gpu_trace_event(event, index, annotations)
        if gpu_event is not None:
            work_by_track[(gpu_event.pid, gpu_event.tid, gpu_event.device)].append(gpu_event)

    annotation_boxes: list[dict[str, Any]] = []
    for stream_events in work_by_track.values():
        active: tuple[str, str, float, float] | None = None
        previous_graph_id = None
        replay_index = 0
        seen_nodes: set[int] = set()
        for event in sorted(stream_events, key=lambda item: (item.start, item.end, item.index)):
            if event.graph_id is None or event.graph_node_id is None:
                if active is not None:
                    name, source, start, end = active
                    annotation_boxes.append(
                        _graph_annotation_box(name, event.pid, event.tid, start, end, source)
                    )
                    active = None
                continue

            if event.graph_id != previous_graph_id:
                if previous_graph_id is not None:
                    replay_index += 1
                previous_graph_id = event.graph_id
                seen_nodes.clear()
            if event.correlation is None:
                if event.graph_node_id in seen_nodes:
                    replay_index += 1
                    seen_nodes.clear()
                seen_nodes.add(event.graph_node_id)
                replay = ("nodes", replay_index)
            else:
                replay = ("correlation", event.correlation)
            source = _annotation_source(event, replay)

            if event.name is None or source in generated_sources:
                if active is not None:
                    name, active_source, start, end = active
                    annotation_boxes.append(
                        _graph_annotation_box(
                            name, event.pid, event.tid, start, end, active_source
                        )
                    )
                    active = None
                continue
            if active is None or (event.name, source) != active[:2]:
                if active is not None:
                    name, active_source, start, end = active
                    annotation_boxes.append(
                        _graph_annotation_box(
                            name, event.pid, event.tid, start, end, active_source
                        )
                    )
                active = (event.name, source, event.start, event.end)
            else:
                active = (active[0], active[1], active[2], max(active[3], event.end))
        if active is not None:
            name, source, start, end = active
            annotation_boxes.append(
                _graph_annotation_box(name, event.pid, event.tid, start, end, source)
            )

    processed = trace.copy()
    processed["traceEvents"] = [*events, *annotation_boxes]
    return processed


@dataclass(frozen=True)
class _NativeGpuTraceEvent:
    """GPU render-stage event with optional embedded graph annotation metadata."""

    track: tuple[str, str, str, str]
    graph_id: str | None
    correlation: str | None
    node_id: str
    name: str | None
    timestamp: int
    duration: int
    clock_id: int


def _native_track_uuid(track: tuple[str, str, str, str]) -> int:
    """Return a stable nonzero TrackEvent UUID for a native GPU annotation track."""
    value = int.from_bytes(hashlib.blake2b(repr(track).encode(), digest_size=8).digest(), "little")
    return (value & ((1 << 63) - 1)) or 1


def _native_gpu_event(packet: Any) -> _NativeGpuTraceEvent | None:
    """Extract annotation metadata from one native CUPTI GPU render-stage packet."""
    if not packet.HasField("gpu_render_stage_event"):
        return None
    event = packet.gpu_render_stage_event
    if event.duration <= 0:
        return None
    metadata = {item.name: item.value for item in event.extra_data}
    track = (
        metadata.get("process_id", "unknown"),
        metadata.get("device id", str(event.gpu_id)),
        metadata.get("stream id", str(event.hw_queue_iid)),
        str(event.hw_queue_iid),
    )
    return _NativeGpuTraceEvent(
        track=track,
        graph_id=metadata.get("graph id"),
        correlation=metadata.get("correlation"),
        node_id=metadata.get("graph node id", str(event.event_id)),
        name=_annotation_label([metadata]) if metadata.get("graph id") is not None else None,
        timestamp=int(packet.timestamp),
        duration=int(event.duration),
        clock_id=int(packet.timestamp_clock_id),
    )


def add_cuda_graph_annotation_tracks_to_native_trace(payload: bytes) -> bytes:
    """Append TrackEvent annotation slices while preserving native CUPTI GPU packets."""
    compressed = payload.startswith(b"\x1f\x8b")
    try:
        raw_payload = gzip.decompress(payload) if compressed else payload
    except (EOFError, OSError):
        return payload
    try:
        from google.protobuf.message import DecodeError
        from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent
    except ImportError:
        return payload

    trace = Trace()
    try:
        trace.ParseFromString(raw_payload)
    except DecodeError:
        return payload
    if any(
        packet.HasField("track_descriptor")
        and packet.track_descriptor.name.startswith("GPU annotations stream ")
        for packet in trace.packet
    ):
        return payload

    work_by_track: dict[tuple[str, str, str, str], list[_NativeGpuTraceEvent]] = defaultdict(list)
    for packet in trace.packet:
        event = _native_gpu_event(packet)
        if event is not None:
            work_by_track[event.track].append(event)

    slices: list[tuple[tuple[str, str, str, str], str, int, int, int]] = []
    for track, events in work_by_track.items():
        active: tuple[str, str | None, str | None, int, int, int] | None = None
        seen_nodes: set[str] = set()
        replay_index = 0
        for event in sorted(events, key=lambda item: item.timestamp):
            if event.correlation is None:
                if event.node_id in seen_nodes:
                    replay_index += 1
                    seen_nodes.clear()
                seen_nodes.add(event.node_id)
                replay = f"nodes:{replay_index}"
            else:
                replay = f"correlation:{event.correlation}"
            identity = (event.graph_id, replay)
            if event.name is None:
                if active is not None:
                    name, _graph_id, _replay, start, end, clock_id = active
                    slices.append((track, name, start, end, clock_id))
                    active = None
                continue
            if active is None or (event.name, *identity) != active[:3]:
                if active is not None:
                    name, _graph_id, _replay, start, end, clock_id = active
                    slices.append((track, name, start, end, clock_id))
                active = (
                    event.name,
                    event.graph_id,
                    replay,
                    event.timestamp,
                    event.timestamp + event.duration,
                    event.clock_id,
                )
            else:
                active = (*active[:4], max(active[4], event.timestamp + event.duration), active[5])
        if active is not None:
            name, _graph_id, _replay, start, end, clock_id = active
            slices.append((track, name, start, end, clock_id))

    if not slices:
        return payload

    for track in sorted({track for track, *_ in slices}):
        descriptor_packet = trace.packet.add()
        descriptor_packet.timestamp = 0
        descriptor = descriptor_packet.track_descriptor
        descriptor.uuid = _native_track_uuid(track)
        descriptor.name = f"GPU annotations stream {track[2]}"
        descriptor.sibling_order_rank = -10

    markers = []
    for index, (track, name, start, end, clock_id) in enumerate(slices):
        markers.append((start, 1, index, track, name, clock_id))
        markers.append((end, 0, index, track, name, clock_id))
    for timestamp, begin_rank, _index, track, name, clock_id in sorted(markers):
        packet = trace.packet.add()
        packet.timestamp = timestamp
        if clock_id:
            packet.timestamp_clock_id = clock_id
        packet.trusted_packet_sequence_id = 2001
        event = packet.track_event
        event.track_uuid = _native_track_uuid(track)
        if begin_rank:
            event.type = TrackEvent.TYPE_SLICE_BEGIN
            event.name = name
        else:
            event.type = TrackEvent.TYPE_SLICE_END

    output = trace.SerializeToString()
    return gzip.compress(output) if compressed else output


def default_trace_path(file_path: str | Path, *, gzip_by_default: bool = True) -> Path:
    """Return a Chrome JSON trace path, treating suffix-less paths as stems.

    Examples:
        ``"foo"`` -> ``"foo.json.gz"`` when ``gzip_by_default`` is true.
        ``"foo"`` -> ``"foo.json"`` when false.
        Explicit ``.json``/``.json.gz`` paths are respected.
    """
    path = Path(file_path)
    suffixes = path.suffixes
    if suffixes[-2:] == [".json", ".gz"] or path.suffix in {".json", ".gz"}:
        return path
    if path.suffix:
        return Path(f"{path}.json.gz") if gzip_by_default else Path(f"{path}.json")
    return path.with_suffix(".json.gz" if gzip_by_default else ".json")


@dataclass(frozen=True)
class _Slice:
    event: dict[str, Any]
    index: int
    ts: float
    dur: float

    @property
    def end_ts(self) -> float:
        return self.ts + self.dur


def _assign_lanes(slices: list[_Slice]) -> dict[int, int]:
    """Greedily assign non-overlapping slices to display lanes."""
    lane_end_times: list[float] = []
    assignments: dict[int, int] = {}

    for slc in slices:
        lane = None
        for lane_idx, lane_end in enumerate(lane_end_times):
            if slc.ts >= lane_end:
                lane = lane_idx
                lane_end_times[lane_idx] = slc.end_ts
                break
        if lane is None:
            lane = len(lane_end_times)
            lane_end_times.append(slc.end_ts)
        assignments[slc.index] = lane

    return assignments


def _stable_string_sort_index(value: Any, lane: int) -> int:
    encoded = str(value).encode("utf-8", errors="replace")
    return (zlib.crc32(encoded) % 1_000_000) * 100 + lane


def _make_tid_allocator(events: list[dict[str, Any]]):
    existing_by_pid: dict[Any, set[Any]] = defaultdict(set)
    max_numeric_by_pid: dict[Any, int] = defaultdict(int)
    for event in events:
        if "pid" not in event or "tid" not in event:
            continue
        pid = event.get("pid", 0)
        tid = event.get("tid", 0)
        existing_by_pid[pid].add(tid)
        if isinstance(tid, int) and not isinstance(tid, bool):
            max_numeric_by_pid[pid] = max(max_numeric_by_pid[pid], tid)

    reserved_by_pid = {pid: set(tids) for pid, tids in existing_by_pid.items()}

    def allocate(pid: Any, original_tid: Any, lane_count: int) -> dict[int, Any]:
        if lane_count <= 1:
            return {0: original_tid}

        if isinstance(original_tid, str):
            lane_tids: dict[int, Any] = {}
            for lane in range(lane_count):
                candidate = f"{original_tid}#{lane}"
                if candidate in reserved_by_pid[pid] and candidate != original_tid:
                    suffix = 1
                    while f"{candidate}_{suffix}" in reserved_by_pid[pid]:
                        suffix += 1
                    candidate = f"{candidate}_{suffix}"
                lane_tids[lane] = candidate
                reserved_by_pid[pid].add(candidate)
            return lane_tids

        original_int = int(original_tid)
        base = original_int * 100
        candidates = [base + lane for lane in range(lane_count)]
        conflicts = [
            tid for tid in candidates if tid in reserved_by_pid[pid] and tid != original_tid
        ]
        if conflicts:
            base = max_numeric_by_pid[pid] + 1
            if base % 100:
                base += 100 - (base % 100)
            candidates = [base + lane for lane in range(lane_count)]
            while any(tid in reserved_by_pid[pid] for tid in candidates):
                base += 100
                candidates = [base + lane for lane in range(lane_count)]

        lane_tids = {lane: tid for lane, tid in enumerate(candidates)}
        for tid in candidates:
            reserved_by_pid[pid].add(tid)
            max_numeric_by_pid[pid] = max(max_numeric_by_pid[pid], tid)
        return lane_tids

    return allocate


def _compile_pattern(pattern: str | Pattern[str] | None) -> Pattern[str] | None:
    if pattern is None or hasattr(pattern, "search"):
        return pattern
    return re.compile(pattern)


def split_overlapping_slices(
    trace: dict[str, Any],
    *,
    track_pattern: str | Pattern[str] | None = None,
) -> dict[str, Any]:
    """Move overlapping ``ph='X'`` slices on the same track to extra lanes.

    Perfetto does not render overlapping duration slices on one track reliably;
    for CUDA streams this can produce confusing empty-looking rows or hidden
    slices. This helper keeps non-overlapping tracks unchanged and, only for
    tracks that actually overlap, remaps their duration slices to adjacent tids
    named ``"<original name> #0"``, ``"<original name> #1"``, ...

    Args:
        trace: Chrome/Perfetto trace dictionary.
        track_pattern: Optional regex matched against thread names. ``None``
            processes every track; e.g. ``"stream.*"`` limits rewriting to CUDA
            stream tracks in torch.profiler traces.

    Returns:
        A shallow-copied trace dict with a new ``traceEvents`` list.
    """
    pattern = _compile_pattern(track_pattern)
    events = list(trace.get("traceEvents", []))

    thread_names: dict[tuple[Any, Any], str] = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            key = (event.get("pid", 0), event.get("tid", 0))
            thread_names[key] = event.get("args", {}).get("name", f"track {key[1]}")

    track_slices: dict[tuple[Any, Any], list[_Slice]] = defaultdict(list)
    for idx, event in enumerate(events):
        if event.get("ph") != "X":
            continue
        dur = float(event.get("dur", 0) or 0)
        if dur <= 0:
            continue
        key = (event.get("pid", 0), event.get("tid", 0))
        track_name = thread_names.get(key, f"track {key[1]}")
        if pattern is None or pattern.search(track_name):
            track_slices[key].append(
                _Slice(event=event, index=idx, ts=float(event.get("ts", 0) or 0), dur=dur)
            )

    allocate_tids = _make_tid_allocator(events)
    tid_mappings: dict[tuple[Any, Any, int], Any] = {}
    event_lane_assignments: dict[int, tuple[Any, Any, int]] = {}
    split_tracks: set[tuple[Any, Any]] = set()

    for (pid, original_tid), slices in track_slices.items():
        slices.sort(key=lambda slc: (slc.ts, slc.end_ts, slc.index))
        assignments = _assign_lanes(slices)
        lane_count = max(assignments.values(), default=0) + 1
        lane_tids = allocate_tids(pid, original_tid, lane_count)
        for lane, new_tid in lane_tids.items():
            tid_mappings[(pid, original_tid, lane)] = new_tid
        if lane_count > 1:
            split_tracks.add((pid, original_tid))
        for slc in slices:
            event_lane_assignments[slc.index] = (pid, original_tid, assignments[slc.index])

    if not split_tracks:
        return trace.copy()

    correlation_tid_map: dict[tuple[Any, Any, Any, Any], Any] = {}
    metadata_needed: dict[tuple[Any, Any, int], dict[str, Any]] = {}
    original_names = {key: name for key, name in thread_names.items()}

    for idx, event in enumerate(events):
        if idx not in event_lane_assignments:
            continue
        pid, original_tid, lane = event_lane_assignments[idx]
        new_tid = tid_mappings[(pid, original_tid, lane)]
        args = event.get("args", {})
        correlation = args.get("correlation") if isinstance(args, dict) else None
        if correlation is not None:
            correlation_tid_map[(pid, original_tid, correlation, event.get("ts", 0))] = new_tid

        original_name = original_names.get((pid, original_tid), f"track {original_tid}")
        metadata_needed[(pid, original_tid, lane)] = {
            "pid": pid,
            "tid": new_tid,
            "name": f"{original_name.rstrip()} #{lane}",
            "sort_index": _sort_index(original_tid, lane),
            "ts": event.get("ts", 0),
        }

    lane0_remap = {
        (pid, original_tid): tid_mappings[(pid, original_tid, 0)]
        for pid, original_tid in split_tracks
        if tid_mappings[(pid, original_tid, 0)] != original_tid
    }

    new_events: list[dict[str, Any]] = []
    for idx, event in enumerate(events):
        new_event = event.copy()
        if "args" in new_event and isinstance(new_event["args"], dict):
            new_event["args"] = new_event["args"].copy()

        if idx in event_lane_assignments:
            pid, original_tid, lane = event_lane_assignments[idx]
            new_event["tid"] = tid_mappings[(pid, original_tid, lane)]
        elif event.get("ph") == "M" and event.get("name") in {
            "thread_name",
            "thread_sort_index",
        }:
            pid = event.get("pid", 0)
            tid = event.get("tid", 0)
            if (pid, tid) in split_tracks:
                new_event["tid"] = tid_mappings[(pid, tid, 0)]
                if event.get("name") == "thread_name":
                    original_name = original_names.get((pid, tid), f"track {tid}")
                    new_event["args"] = {"name": f"{original_name.rstrip()} #0"}
        else:
            pid = event.get("pid", 0)
            tid = event.get("tid", 0)
            if event.get("ph") == "f":
                flow_id = event.get("id")
                key = (pid, tid, flow_id, event.get("ts", 0))
                if flow_id is not None and key in correlation_tid_map:
                    new_event["tid"] = correlation_tid_map[key]
                elif (pid, tid) in lane0_remap:
                    new_event["tid"] = lane0_remap[(pid, tid)]
            elif (pid, tid) in lane0_remap:
                new_event["tid"] = lane0_remap[(pid, tid)]

        new_events.append(new_event)

    existing_thread_names: set[tuple[Any, Any]] = set()
    existing_sort_indices: set[tuple[Any, Any]] = set()
    for event in new_events:
        if event.get("ph") != "M":
            continue
        key = (event.get("pid", 0), event.get("tid", 0))
        if event.get("name") == "thread_name":
            existing_thread_names.add(key)
        elif event.get("name") == "thread_sort_index":
            existing_sort_indices.add(key)

    for metadata in metadata_needed.values():
        key = (metadata["pid"], metadata["tid"])
        if key not in existing_thread_names:
            new_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "ts": metadata["ts"],
                    "pid": metadata["pid"],
                    "tid": metadata["tid"],
                    "args": {"name": metadata["name"]},
                }
            )
        if key not in existing_sort_indices:
            new_events.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "ts": metadata["ts"],
                    "pid": metadata["pid"],
                    "tid": metadata["tid"],
                    "args": {"sort_index": metadata["sort_index"]},
                }
            )

    _reassign_sort_indices(new_events, pattern)

    new_trace = trace.copy()
    new_trace["traceEvents"] = new_events
    return new_trace


def _sort_index(original_tid: Any, lane: int) -> int:
    if isinstance(original_tid, str):
        return _stable_string_sort_index(original_tid, lane)
    return int(original_tid) * 100 + lane


def _reassign_sort_indices(
    events: list[dict[str, Any]], track_pattern: Pattern[str] | None = None
) -> None:
    thread_names: dict[tuple[Any, Any], str] = {}
    sort_indices: dict[tuple[Any, Any], int] = {}

    for event in events:
        if event.get("ph") != "M":
            continue
        key = (event.get("pid", 0), event.get("tid", 0))
        if event.get("name") == "thread_name":
            thread_names[key] = event.get("args", {}).get("name", "")
        elif event.get("name") == "thread_sort_index":
            sort_indices[key] = int(event.get("args", {}).get("sort_index", 0) or 0)

    lane_re = re.compile(r"^(.*?)\s+#(\d+)$")
    groups: dict[str, list[tuple[tuple[Any, Any], int, int]]] = defaultdict(list)
    singles: list[tuple[tuple[Any, Any], int]] = []

    for key, name in thread_names.items():
        match = lane_re.match(name)
        if match:
            base = f"{key[0]}:{match.group(1)}"
            lane = int(match.group(2))
            groups[base].append((key, lane, sort_indices.get(key, 0)))
        elif track_pattern is None or track_pattern.search(name):
            singles.append((key, sort_indices.get(key, 0)))

    if not groups:
        return

    entries: list[tuple[int, list[tuple[Any, Any]]]] = []
    for key, sort_index in singles:
        entries.append((sort_index, [key]))
    for lanes in groups.values():
        lanes.sort(key=lambda item: item[1])
        leader_sort_index = lanes[0][2]
        entries.append((leader_sort_index, [item[0] for item in lanes]))

    entries.sort(key=lambda item: item[0])
    new_sort: dict[tuple[Any, Any], int] = {}
    next_sort = 0
    for _, keys in entries:
        for key in keys:
            new_sort[key] = next_sort
            next_sort += 1

    updated: set[tuple[Any, Any]] = set()
    for event in events:
        if event.get("ph") != "M" or event.get("name") != "thread_sort_index":
            continue
        key = (event.get("pid", 0), event.get("tid", 0))
        if key in new_sort and key not in updated:
            event["args"] = {"sort_index": new_sort[key]}
            updated.add(key)

    for key, sort_index in new_sort.items():
        if key in updated:
            continue
        events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "ts": 0,
                "pid": key[0],
                "tid": key[1],
                "args": {"sort_index": sort_index},
            }
        )


def perfetto_trace_path(
    path: str | Path,
    *,
    trace_format: TraceFormat = "track_event",
    gzip_trace: bool = False,
) -> Path:
    """Normalize a requested output path for the selected trace format."""
    if trace_format == "track_event":
        return default_track_event_path(path)
    if trace_format == "chrome_json":
        return default_trace_path(path, gzip_by_default=gzip_trace)
    raise ValueError(f"Unsupported trace_format: {trace_format!r}")


def write_perfetto_trace(
    path: str | Path,
    trace: dict[str, Any],
    *,
    trace_format: TraceFormat = "track_event",
    split_overlaps: bool = True,
    track_pattern: str | Pattern[str] | None = None,
    gzip_trace: bool = False,
) -> Path:
    """Write a trace in the requested Perfetto-compatible format.

    ``track_event`` writes native Perfetto protobuf ``.pftrace`` files.
    ``chrome_json`` writes legacy Chrome JSON/JSON.GZ and optionally applies
    JSON-only overlap lane splitting.
    """
    if trace_format == "track_event":
        return write_track_event_trace(
            path,
            trace,
            track_pattern=track_pattern,
            split_overlaps=split_overlaps,
        )

    if trace_format != "chrome_json":
        raise ValueError(f"Unsupported trace_format: {trace_format!r}")

    path = default_trace_path(path, gzip_by_default=gzip_trace)
    if split_overlaps:
        trace = split_overlapping_slices(trace, track_pattern=track_pattern)
    write_trace(path, trace)
    return path
