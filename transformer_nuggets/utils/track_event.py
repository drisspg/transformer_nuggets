"""Native Perfetto TrackEvent protobuf conversion helpers."""

from __future__ import annotations

import functools
import hashlib
import json
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterable
from re import Pattern


TraceDict = dict[str, Any]
"""Loose Chrome/Perfetto JSON trace dictionary."""

TrackKey = tuple[Any, Any]
"""Chrome JSON track identity: ``(pid, tid)``."""

_TRAILING_TRACK_LANE_RE = re.compile(r"\s+#\d+$")


@dataclass(frozen=True, slots=True)
class ChromeTrack:
    """A logical Chrome JSON track derived from ``pid``/``tid`` metadata.

    Chrome JSON represents a timeline row with process/thread-ish IDs and
    optional metadata events. Native TrackEvent needs explicit track
    descriptors, so this is the normalized track model used between parsing and
    protobuf emission.
    """

    pid: Any
    tid: Any
    name: str
    sort_index: int = 0
    key: TrackKey = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", (self.pid, self.tid))


@dataclass(frozen=True, slots=True)
class ChromeMetadata:
    """Process and track metadata parsed from Chrome JSON ``ph='M'`` events."""

    process_names: dict[Any, str]
    process_sort_indices: dict[Any, int]
    tracks: dict[TrackKey, ChromeTrack]

    def track_for(self, pid: Any, tid: Any) -> ChromeTrack:
        key = (pid, tid)
        track = self.tracks.get(key)
        if track is not None:
            return track
        return ChromeTrack(pid=pid, tid=tid, name=_clean_track_name(pid, tid))


@dataclass(frozen=True, slots=True)
class DurationSlice:
    """A Chrome JSON ``ph='X'`` duration event with normalized timing.

    ``lane`` is a backing TrackEvent track index, not a user-visible stream or
    thread. Crossing intervals on the same logical track need distinct backing
    tracks so Perfetto can keep each TrackEvent stack valid. Those backing
    tracks are emitted with the same merge key so the UI can display one logical
    row.
    """

    event: TraceDict
    index: int
    track: ChromeTrack
    ts_us: float
    dur_us: float
    lane: int = 0
    flow_ids: tuple[int, ...] = ()
    flow_latencies_us: tuple[tuple[int, float], ...] = ()

    @property
    def end_us(self) -> float:
        return self.ts_us + self.dur_us


@dataclass(frozen=True, slots=True)
class InstantEvent:
    """A Chrome JSON instant event, ``ph='i'`` or ``ph='I'``."""

    event: TraceDict
    index: int
    track: ChromeTrack
    ts_us: float


@dataclass(frozen=True, slots=True)
class CounterSample:
    """A numeric sample parsed from a Chrome JSON counter event, ``ph='C'``."""

    event: TraceDict
    index: int
    track: ChromeTrack
    ts_us: float
    counter_name: str
    value: int | float


@dataclass(frozen=True, slots=True)
class FlowInstant:
    """A Chrome JSON flow marker, ``ph='s'``/``'t'``/``'f'``.

    These are not emitted as standalone events. Paired launch/completion flows
    are attached to the real CPU launch and GPU kernel duration slices instead.
    """

    event: TraceDict
    index: int
    track: ChromeTrack
    ts_us: float


@dataclass(frozen=True, slots=True)
class ParsedChromeTrace:
    """Typed subset of a Chrome JSON trace before TrackEvent lane assignment.

    Supported input phases are:
    - ``M`` metadata
    - ``X`` complete duration slices
    - ``i``/``I`` instants
    - ``C`` counters
    - ``s``/``t``/``f`` flow markers

    Unsupported phases are recorded so the caller gets a warning rather than
    silent data loss.
    """

    metadata: ChromeMetadata
    duration_slices: list[DurationSlice]
    instants: list[InstantEvent]
    counters: list[CounterSample]
    flows: list[FlowInstant]
    unsupported_phases: frozenset[str]


@dataclass(frozen=True, slots=True)
class AssignedTrace:
    """Parsed Chrome trace after TrackEvent-compatible lane assignment."""

    metadata: ChromeMetadata
    duration_slices: list[DurationSlice]
    instants: list[InstantEvent]
    counters: list[CounterSample]
    flows: list[FlowInstant]
    unsupported_phases: frozenset[str]


def _is_numeric_process_id(pid: Any) -> bool:
    return isinstance(pid, int) and not isinstance(pid, bool) and pid >= 0


@dataclass(frozen=True, slots=True)
class TrackEventProtos:
    """Perfetto protobuf classes loaded lazily from the ``perfetto`` package."""

    TraceProtoBuilder: Any
    TrackDescriptor: Any
    TrackEvent: Any


@dataclass(frozen=True, slots=True)
class TrackIds:
    """Stable protobuf UUID mappings for emitted TrackEvent descriptors."""

    process_uuids: dict[Any, int]
    duration_track_uuids: dict[tuple[TrackKey, int], int]
    instant_track_uuids: dict[TrackKey, int]
    counter_track_uuids: dict[tuple[TrackKey, str], int]


Marker = tuple[int, int, float, int, int, bool, DurationSlice]
"""Sortable begin/end packet marker for a TrackEvent duration slice."""


def default_track_event_path(file_path: str | Path) -> Path:
    """Return a native Perfetto TrackEvent trace path.

    Perfetto's native protobuf trace format is conventionally written as
    ``.pftrace`` or ``.perfetto-trace``. Chrome JSON suffixes are treated as
    format requests and rewritten to ``.pftrace``.
    """
    path = Path(file_path)
    suffixes = path.suffixes
    if path.suffix in {".pftrace", ".perfetto-trace"}:
        return path
    if suffixes[-2:] == [".json", ".gz"]:
        return path.with_suffix("").with_suffix(".pftrace")
    if path.suffix == ".json":
        return path.with_suffix(".pftrace")
    if path.suffix:
        return Path(f"{path}.pftrace")
    return path.with_suffix(".pftrace")


def _compile_pattern(pattern: str | Pattern[str] | None) -> Pattern[str] | None:
    if pattern is None or hasattr(pattern, "search"):
        return pattern
    return re.compile(pattern)


def _stable_uuid(*parts: Any) -> int:
    payload = repr(parts).encode("utf-8", errors="replace")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return (value & ((1 << 63) - 1)) or 1


def _timestamp_us_to_ns(value: Any) -> int:
    return int(round(float(value or 0) * 1000.0))


@functools.cache
def _load_perfetto_protos() -> TrackEventProtos:
    try:
        from perfetto.trace_builder.proto_builder import TraceProtoBuilder
        from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import (
            TrackDescriptor,
            TrackEvent,
        )
    except ImportError as exc:
        raise ImportError(
            "Native Perfetto TrackEvent traces require the `perfetto` Python package. "
            "Install transformer-nuggets with updated dependencies or run `pip install perfetto`."
        ) from exc
    return TrackEventProtos(TraceProtoBuilder, TrackDescriptor, TrackEvent)


def _event_args(event: TraceDict) -> dict[str, Any]:
    args = event.get("args", {})
    return args if isinstance(args, dict) else {}


def _event_track(
    metadata: ChromeMetadata,
    event: TraceDict,
    annotation_track_cache: dict[TrackKey, ChromeTrack],
) -> ChromeTrack:
    pid = event.get("pid", 0)
    tid = event.get("tid", 0)
    base_track = metadata.track_for(pid, tid)
    if event.get("cat") != "gpu_user_annotation":
        return base_track

    key = base_track.key
    annotation_track = annotation_track_cache.get(key)
    if annotation_track is not None:
        return annotation_track
    annotation_track = ChromeTrack(
        pid=pid,
        tid=f"{tid}:gpu_user_annotation",
        name=f"GPU annotations {base_track.name}",
        sort_index=base_track.sort_index,
    )
    annotation_track_cache[key] = annotation_track
    return annotation_track


def _clean_process_name(pid: Any, name: Any | None = None) -> str:
    if pid == -1:
        return "Kineto events"
    if name is not None:
        return str(name)
    return f"process {pid}" if _is_numeric_process_id(pid) else str(pid)


def _clean_track_name(pid: Any, tid: Any, name: Any | None = None) -> str:
    if pid == -1:
        return "Kineto events"
    text = str(name if name is not None else f"track {tid}").rstrip()
    text = _TRAILING_TRACK_LANE_RE.sub("", text)
    if text.startswith("track ") and text != "track 0":
        text = text[len("track ") :]
    return text


def _hide_chrome_event(event: TraceDict) -> bool:
    # Empty pid/tid events are global record-window markers. PyTorch also emits
    # profiler bookkeeping groups under string pids "Spans" and "Traces". They
    # are useful to Kineto, but add visual noise for the native Perfetto view.
    return (event.get("pid") == "" and event.get("tid") == "") or event.get("pid") in {
        "Spans",
        "Traces",
    }


def _parse_metadata(events: Iterable[TraceDict]) -> ChromeMetadata:
    process_names: dict[Any, str] = {}
    process_sort_indices: dict[Any, int] = {}
    thread_names: dict[TrackKey, str] = {}
    sort_indices: dict[TrackKey, int] = {}

    for event in events:
        if event.get("ph") != "M" or _hide_chrome_event(event):
            continue
        pid = event.get("pid", 0)
        tid = event.get("tid", 0)
        args = _event_args(event)
        if event.get("name") == "process_name":
            process_names[pid] = _clean_process_name(pid, args.get("name"))
        elif event.get("name") == "process_sort_index":
            process_sort_indices[pid] = int(args.get("sort_index", 0) or 0)
        elif event.get("name") == "thread_name":
            thread_names[(pid, tid)] = _clean_track_name(pid, tid, args.get("name"))
        elif event.get("name") == "thread_sort_index":
            sort_indices[(pid, tid)] = int(args.get("sort_index", 0) or 0)

    track_keys = set(thread_names) | set(sort_indices)
    tracks = {
        key: ChromeTrack(
            pid=key[0],
            tid=key[1],
            name=thread_names.get(key, _clean_track_name(key[0], key[1])),
            sort_index=sort_indices.get(key, 0),
        )
        for key in track_keys
    }
    return ChromeMetadata(
        process_names=process_names,
        process_sort_indices=process_sort_indices,
        tracks=tracks,
    )


def _parse_counter_samples(
    event: TraceDict,
    index: int,
    track: ChromeTrack,
) -> list[CounterSample]:
    args = _event_args(event)
    samples: list[CounterSample] = []
    for name, value in args.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        counter_name = str(name if len(args) > 1 else event.get("name", name))
        samples.append(
            CounterSample(
                event=event,
                index=index,
                track=track,
                ts_us=float(event.get("ts", 0) or 0),
                counter_name=counter_name,
                value=value,
            )
        )
    return samples


def parse_chrome_trace(trace: TraceDict) -> ParsedChromeTrace:
    """Parse loose Chrome JSON into explicit internal event models."""
    raw_events = trace.get("traceEvents", [])
    events = raw_events if isinstance(raw_events, list) else list(raw_events)
    metadata = _parse_metadata(events)
    duration_slices: list[DurationSlice] = []
    instants: list[InstantEvent] = []
    counters: list[CounterSample] = []
    flows: list[FlowInstant] = []
    unsupported: set[str] = set()
    annotation_track_cache: dict[TrackKey, ChromeTrack] = {}

    for idx, event in enumerate(events):
        if _hide_chrome_event(event):
            continue
        ph = event.get("ph")
        if ph == "M":
            continue
        track = _event_track(metadata, event, annotation_track_cache)
        if ph == "X":
            dur_us = float(event.get("dur", 0) or 0)
            if dur_us > 0:
                duration_slices.append(
                    DurationSlice(
                        event=event,
                        index=idx,
                        track=track,
                        ts_us=float(event.get("ts", 0) or 0),
                        dur_us=dur_us,
                    )
                )
        elif ph in {"i", "I"}:
            instants.append(
                InstantEvent(
                    event=event,
                    index=idx,
                    track=track,
                    ts_us=float(event.get("ts", 0) or 0),
                )
            )
        elif ph == "C":
            counters.extend(_parse_counter_samples(event, idx, track))
        elif ph in {"s", "t", "f"}:
            flows.append(
                FlowInstant(
                    event=event,
                    index=idx,
                    track=track,
                    ts_us=float(event.get("ts", 0) or 0),
                )
            )
        elif ph in {"B", "E"}:
            unsupported.add(str(ph))
        else:
            unsupported.add(str(ph))

    return ParsedChromeTrace(
        metadata=metadata,
        duration_slices=duration_slices,
        instants=instants,
        counters=counters,
        flows=flows,
        unsupported_phases=frozenset(unsupported),
    )


def _assign_nesting_lanes(slices: list[DurationSlice]) -> dict[int, int]:
    """Assign slices to lanes where each lane can be emitted as nested TrackEvents.

    TrackEvent begin/end packets support properly nested slices on one track,
    but not crossing intervals. Since slices are processed by start time with
    longer equal-start slices first, a lane is valid when the new slice either
    starts after the active stack or is contained by the current innermost
    active slice. This keeps assignment close to O(number of slices * lanes)
    instead of checking every earlier slice in the lane.
    """
    lane_end_stacks: list[list[float]] = []
    assignments: dict[int, int] = {}

    for slc in sorted(slices, key=lambda s: (s.ts_us, -s.end_us, s.index)):
        for lane_idx, end_stack in enumerate(lane_end_stacks):
            while end_stack and end_stack[-1] <= slc.ts_us:
                end_stack.pop()
            if not end_stack or slc.end_us <= end_stack[-1]:
                end_stack.append(slc.end_us)
                assignments[slc.index] = lane_idx
                break
        else:
            assignments[slc.index] = len(lane_end_stacks)
            lane_end_stacks.append([slc.end_us])

    return assignments


def _flow_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value) & ((1 << 64) - 1)
    except (TypeError, ValueError):
        return None


def _slice_correlation_id(slc: DurationSlice) -> int | None:
    return _flow_id(_event_args(slc.event).get("correlation"))


def _is_cuda_launch_slice(slc: DurationSlice) -> bool:
    name = str(slc.event.get("name", ""))
    return name.startswith(("cudaLaunch", "cuLaunch")) or "LaunchKernel" in name


def _is_gpu_kernel_slice(slc: DurationSlice) -> bool:
    args = _event_args(slc.event)
    return "stream" in args and "device" in args and not _is_cuda_launch_slice(slc)


def _smallest_containing_slice(
    slices: list[DurationSlice],
    flow: FlowInstant,
) -> DurationSlice | None:
    containing = [
        slc
        for slc in slices
        if slc.track.key == flow.track.key and slc.ts_us <= flow.ts_us <= slc.end_us
    ]
    if containing:
        return min(
            containing,
            key=lambda slc: (slc.dur_us, abs(slc.ts_us - flow.ts_us), slc.index),
        )

    same_track = [slc for slc in slices if slc.track.key == flow.track.key]
    if not same_track:
        return None
    return min(same_track, key=lambda slc: (abs(slc.ts_us - flow.ts_us), slc.dur_us, slc.index))


def _paired_flow_ids_by_slice(
    slices: list[DurationSlice],
    flows: list[FlowInstant],
) -> tuple[dict[int, set[int]], dict[int, dict[int, float]]]:
    flows_by_id: dict[int, list[FlowInstant]] = defaultdict(list)
    for flow in flows:
        flow_id = _flow_id(flow.event.get("id"))
        if flow_id is not None:
            flows_by_id[flow_id].append(flow)

    slices_by_correlation: dict[int, list[DurationSlice]] = defaultdict(list)
    for slc in slices:
        correlation_id = _slice_correlation_id(slc)
        if correlation_id is not None:
            slices_by_correlation[correlation_id].append(slc)

    flow_ids_by_slice: dict[int, set[int]] = defaultdict(set)
    latencies_by_slice: dict[int, dict[int, float]] = defaultdict(dict)
    for flow_id, markers in flows_by_id.items():
        sources = [marker for marker in markers if marker.event.get("ph") in {"s", "t"}]
        destinations = [marker for marker in markers if marker.event.get("ph") == "f"]
        if not sources or not destinations:
            continue

        correlated_slices = slices_by_correlation.get(flow_id, [])
        launch_slices = [slc for slc in correlated_slices if _is_cuda_launch_slice(slc)]
        kernel_slices = [slc for slc in correlated_slices if _is_gpu_kernel_slice(slc)]

        source = min(sources, key=lambda marker: marker.ts_us)
        source_slice = min(launch_slices, key=lambda slc: slc.ts_us) if launch_slices else None
        if source_slice is None:
            source_slice = _smallest_containing_slice(slices, source)
        if source_slice is not None:
            flow_ids_by_slice[source_slice.index].add(flow_id)

        for destination in destinations:
            destination_slice = min(
                kernel_slices,
                key=lambda slc: abs(slc.ts_us - destination.ts_us),
            ) if kernel_slices else None
            if destination_slice is None:
                destination_slice = _smallest_containing_slice(slices, destination)
            if destination_slice is None:
                continue
            flow_ids_by_slice[destination_slice.index].add(flow_id)
            launch_ts = source_slice.ts_us if source_slice is not None else source.ts_us
            latencies_by_slice[destination_slice.index][flow_id] = (
                destination_slice.ts_us - launch_ts
            )

    return flow_ids_by_slice, latencies_by_slice


def assign_trackevent_lanes(
    parsed: ParsedChromeTrace,
    *,
    track_pattern: str | Pattern[str] | None = None,
    split_overlaps: bool = True,
    include_flows: bool = True,
) -> AssignedTrace:
    pattern = _compile_pattern(track_pattern)
    slices_by_track: dict[TrackKey, list[DurationSlice]] = defaultdict(list)
    for slc in parsed.duration_slices:
        slices_by_track[slc.track.key].append(slc)

    lane_by_index: dict[int, int] = {}
    for _track_key, slices in slices_by_track.items():
        track_name = slices[0].track.name
        should_split = split_overlaps and (pattern is None or pattern.search(track_name))
        assignments = (
            _assign_nesting_lanes(slices) if should_split else {slc.index: 0 for slc in slices}
        )
        lane_by_index.update(assignments)

    if include_flows:
        flow_ids_by_slice, latencies_by_slice = _paired_flow_ids_by_slice(
            parsed.duration_slices,
            parsed.flows,
        )
    else:
        flow_ids_by_slice, latencies_by_slice = {}, {}

    return AssignedTrace(
        metadata=parsed.metadata,
        duration_slices=[
            DurationSlice(
                event=slc.event,
                index=slc.index,
                track=slc.track,
                ts_us=slc.ts_us,
                dur_us=slc.dur_us,
                lane=lane_by_index[slc.index],
                flow_ids=tuple(sorted(flow_ids_by_slice.get(slc.index, ()))),
                flow_latencies_us=tuple(sorted(latencies_by_slice.get(slc.index, {}).items())),
            )
            for slc in parsed.duration_slices
        ],
        instants=parsed.instants,
        counters=parsed.counters,
        flows=parsed.flows,
        unsupported_phases=parsed.unsupported_phases,
    )


def _add_debug_annotation(track_event: Any, name: str, value: Any) -> None:
    annotation = track_event.debug_annotations.add()
    annotation.name = str(name)
    if isinstance(value, bool):
        annotation.bool_value = value
    elif isinstance(value, int) and not isinstance(value, bool):
        annotation.int_value = value
    elif isinstance(value, float):
        annotation.double_value = value
    elif value is None:
        annotation.string_value = "null"
    elif isinstance(value, str):
        annotation.string_value = value
    else:
        annotation.legacy_json_value = json.dumps(value, default=str)


def _copy_event_payload(
    track_event: Any,
    event: TraceDict,
    args: dict[str, Any] | None = None,
) -> None:
    if "cat" in event:
        track_event.categories.append(str(event["cat"]))
    event_args = _event_args(event) if args is None else args
    for name, value in event_args.items():
        _add_debug_annotation(track_event, str(name), value)


def _add_correlation_id(track_event: Any, value: Any) -> None:
    """Attach a non-causal correlation ID without drawing a flow arrow."""
    if value is None:
        return
    try:
        track_event.correlation_id = int(value) & ((1 << 64) - 1)
    except (TypeError, ValueError):
        track_event.correlation_id_str = str(value)


def _warn_for_unsupported_phases(phases: frozenset[str]) -> None:
    if not phases:
        return
    warnings.warn(
        "TrackEvent conversion skipped unsupported Chrome trace phases: "
        f"{', '.join(sorted(phases))}",
        RuntimeWarning,
        stacklevel=3,
    )


def _sorted_tracks(trace: AssignedTrace) -> list[ChromeTrack]:
    tracks: dict[TrackKey, ChromeTrack] = {}
    for slc in trace.duration_slices:
        tracks[slc.track.key] = slc.track
    for instant in trace.instants:
        tracks[instant.track.key] = instant.track
    for counter in trace.counters:
        tracks[counter.track.key] = counter.track
    return sorted(
        tracks.values(),
        key=lambda track: (str(track.pid), track.sort_index, str(track.tid)),
    )


def _process_sort_rank(trace: AssignedTrace, pid: Any) -> int:
    if pid == -1:
        return 2**31 - 1
    return trace.metadata.process_sort_indices.get(pid, 0)


def _process_ids(trace: AssignedTrace, tracks: list[ChromeTrack]) -> list[Any]:
    pids = {track.pid for track in tracks} | set(trace.metadata.process_names)
    return sorted(pids, key=lambda pid: (_process_sort_rank(trace, pid), str(pid)))


def _max_lane_by_track(trace: AssignedTrace) -> dict[TrackKey, int]:
    max_lane: dict[TrackKey, int] = defaultdict(int)
    for slc in trace.duration_slices:
        max_lane[slc.track.key] = max(max_lane[slc.track.key], slc.lane)
    return max_lane


def _define_process_tracks(
    builder: Any,
    trace: AssignedTrace,
    protos: TrackEventProtos,
    tracks: list[ChromeTrack],
) -> dict[Any, int]:
    process_uuids: dict[Any, int] = {}
    for pid in _process_ids(trace, tracks):
        process_uuid = _stable_uuid("process", pid)
        process_uuids[pid] = process_uuid
        packet = builder.add_packet()
        packet.timestamp = 0
        desc = packet.track_descriptor
        desc.uuid = process_uuid
        if _is_numeric_process_id(pid):
            desc.process.pid = int(pid)
            desc.process.process_name = str(
                trace.metadata.process_names.get(pid, f"process {pid}")
            )
        else:
            # torch.profiler uses string pids like "Spans" and "Traces" for
            # synthetic groups. Keep those as generic TrackDescriptor groups;
            # otherwise Perfetto shows misleading labels like "process Spans 0".
            desc.name = _clean_process_name(pid, trace.metadata.process_names.get(pid))
        desc.child_ordering = protos.TrackDescriptor.EXPLICIT
        desc.sibling_order_rank = _process_sort_rank(trace, pid)
    return process_uuids


def _define_duration_tracks(
    builder: Any,
    trace: AssignedTrace,
    process_uuids: dict[Any, int],
    protos: TrackEventProtos,
    tracks: list[ChromeTrack],
) -> dict[tuple[TrackKey, int], int]:
    max_lane_by_track = _max_lane_by_track(trace)
    track_uuids: dict[tuple[TrackKey, int], int] = {}
    for track in tracks:
        lane_count = max_lane_by_track.get(track.key, 0) + 1
        for lane in range(lane_count):
            track_uuid = _stable_uuid("track", track.pid, track.tid, lane)
            track_uuids[(track.key, lane)] = track_uuid
            packet = builder.add_packet()
            packet.timestamp = 0
            desc = packet.track_descriptor
            desc.uuid = track_uuid
            desc.parent_uuid = process_uuids[track.pid]
            desc.name = track.name
            annotation_bias = -10 if str(track.tid).endswith(":gpu_user_annotation") else 0
            desc.sibling_order_rank = track.sort_index * 100 + annotation_bias + lane
            if lane_count > 1:
                desc.sibling_merge_behavior = (
                    protos.TrackDescriptor.SIBLING_MERGE_BEHAVIOR_BY_SIBLING_MERGE_KEY
                )
                desc.sibling_merge_key = f"{track.pid}:{track.tid}:{track.name}"
    return track_uuids


def _define_counter_tracks(
    builder: Any,
    trace: AssignedTrace,
    process_uuids: dict[Any, int],
) -> dict[tuple[TrackKey, str], int]:
    counter_names = sorted(
        {(sample.track, sample.counter_name) for sample in trace.counters},
        key=lambda item: (str(item[0].pid), item[0].sort_index, str(item[0].tid), item[1]),
    )
    counter_uuids: dict[tuple[TrackKey, str], int] = {}
    for track, counter_name in counter_names:
        uuid = _stable_uuid("counter", track.pid, track.tid, counter_name)
        counter_uuids[(track.key, counter_name)] = uuid
        packet = builder.add_packet()
        packet.timestamp = 0
        desc = packet.track_descriptor
        desc.uuid = uuid
        desc.parent_uuid = process_uuids[track.pid]
        desc.name = counter_name
        desc.counter.SetInParent()
    return counter_uuids


def _define_track_ids(builder: Any, trace: AssignedTrace, protos: TrackEventProtos) -> TrackIds:
    tracks = _sorted_tracks(trace)
    process_uuids = _define_process_tracks(builder, trace, protos, tracks)
    return TrackIds(
        process_uuids=process_uuids,
        duration_track_uuids=_define_duration_tracks(
            builder,
            trace,
            process_uuids,
            protos,
            tracks,
        ),
        instant_track_uuids={
            track.key: _stable_uuid("track", track.pid, track.tid, 0) for track in tracks
        },
        counter_track_uuids=_define_counter_tracks(builder, trace, process_uuids),
    )


def _duration_markers(trace: AssignedTrace, track_ids: TrackIds) -> list[Marker]:
    markers: list[Marker] = []
    for slc in trace.duration_slices:
        track_uuid = track_ids.duration_track_uuids[(slc.track.key, slc.lane)]
        markers.append(
            (
                _timestamp_us_to_ns(slc.ts_us),
                0,
                -slc.dur_us,
                track_uuid,
                slc.index,
                True,
                slc,
            )
        )
        markers.append(
            (
                _timestamp_us_to_ns(slc.end_us),
                1,
                slc.dur_us,
                track_uuid,
                slc.index,
                False,
                slc,
            )
        )
    markers.sort()
    return markers


def _emit_duration_markers(
    builder: Any,
    trace: AssignedTrace,
    track_ids: TrackIds,
    protos: TrackEventProtos,
    trusted_packet_sequence_id: int,
) -> None:
    add_packet = builder.add_packet
    slice_begin = protos.TrackEvent.TYPE_SLICE_BEGIN
    slice_end = protos.TrackEvent.TYPE_SLICE_END
    for ts_ns, _begin_order, _duration_key, track_uuid, _slice_index, is_begin, slc in (
        _duration_markers(trace, track_ids)
    ):
        packet = add_packet()
        packet.timestamp = ts_ns
        packet.trusted_packet_sequence_id = trusted_packet_sequence_id
        track_event = packet.track_event
        track_event.track_uuid = track_uuid
        if is_begin:
            event = slc.event
            event_args = _event_args(event)
            track_event.type = slice_begin
            track_event.name = str(event.get("name", "slice"))
            _copy_event_payload(track_event, event, event_args)
            _add_correlation_id(track_event, event_args.get("correlation"))
            track_event.flow_ids.extend(slc.flow_ids)
            if slc.flow_latencies_us:
                for flow_id, latency_us in slc.flow_latencies_us:
                    suffix = "" if len(slc.flow_latencies_us) == 1 else f"[{flow_id}]"
                    _add_debug_annotation(
                        track_event,
                        f"launch_latency_us{suffix}",
                        latency_us,
                    )
                    _add_debug_annotation(track_event, f"launch_flow_id{suffix}", flow_id)
        else:
            track_event.type = slice_end


def _emit_instants(
    builder: Any,
    trace: AssignedTrace,
    track_ids: TrackIds,
    protos: TrackEventProtos,
    trusted_packet_sequence_id: int,
) -> None:
    for instant in sorted(trace.instants, key=lambda item: (item.ts_us, item.index)):
        packet = builder.add_packet()
        packet.timestamp = _timestamp_us_to_ns(instant.ts_us)
        packet.trusted_packet_sequence_id = trusted_packet_sequence_id
        track_event = packet.track_event
        track_event.type = protos.TrackEvent.TYPE_INSTANT
        track_event.track_uuid = track_ids.instant_track_uuids[instant.track.key]
        track_event.name = str(instant.event.get("name", "instant"))
        _copy_event_payload(track_event, instant.event)


def _emit_counters(
    builder: Any,
    trace: AssignedTrace,
    track_ids: TrackIds,
    protos: TrackEventProtos,
    trusted_packet_sequence_id: int,
) -> None:
    for sample in sorted(
        trace.counters,
        key=lambda item: (item.ts_us, item.index, item.counter_name),
    ):
        packet = builder.add_packet()
        packet.timestamp = _timestamp_us_to_ns(sample.ts_us)
        packet.trusted_packet_sequence_id = trusted_packet_sequence_id
        track_event = packet.track_event
        track_event.type = protos.TrackEvent.TYPE_COUNTER
        track_event.track_uuid = track_ids.counter_track_uuids[
            (sample.track.key, sample.counter_name)
        ]
        if isinstance(sample.value, int) and not isinstance(sample.value, bool):
            track_event.counter_value = sample.value
        else:
            track_event.double_counter_value = float(sample.value)


def build_track_event_trace(
    trace: AssignedTrace,
    *,
    trusted_packet_sequence_id: int = 1001,
) -> bytes:
    """Build a native Perfetto TrackEvent protobuf trace from typed events."""
    _warn_for_unsupported_phases(trace.unsupported_phases)
    protos = _load_perfetto_protos()
    builder = protos.TraceProtoBuilder()
    track_ids = _define_track_ids(builder, trace, protos)
    _emit_duration_markers(builder, trace, track_ids, protos, trusted_packet_sequence_id)
    _emit_instants(builder, trace, track_ids, protos, trusted_packet_sequence_id)
    _emit_counters(builder, trace, track_ids, protos, trusted_packet_sequence_id)
    return builder.serialize()


def chrome_trace_to_track_event_trace(
    trace: TraceDict,
    *,
    track_pattern: str | Pattern[str] | None = None,
    split_overlaps: bool = True,
    trusted_packet_sequence_id: int = 1001,
    include_flows: bool = True,
) -> bytes:
    """Convert Chrome JSON trace events to native Perfetto TrackEvents.

    Supported Chrome phases are parsed explicitly: duration slices (``X``),
    instants (``i``/``I``), counters (``C``), flow markers (``s``/``t``/``f``),
    and metadata (``M``). Legacy Chrome flow markers are emitted only when they
    form paired launch/completion links, and are attached to the real enclosing
    slices instead of standalone marker events. Pass ``include_flows=False`` to
    suppress those causal arrows. Unsupported phases produce a warning instead
    of being silently dropped.
    """
    parsed = parse_chrome_trace(trace)
    assigned = assign_trackevent_lanes(
        parsed,
        track_pattern=track_pattern,
        split_overlaps=split_overlaps,
        include_flows=include_flows,
    )
    return build_track_event_trace(
        assigned,
        trusted_packet_sequence_id=trusted_packet_sequence_id,
    )


def write_track_event_trace(
    path: str | Path,
    trace: TraceDict,
    *,
    track_pattern: str | Pattern[str] | None = None,
    split_overlaps: bool = True,
    include_flows: bool = True,
) -> Path:
    """Write a native Perfetto TrackEvent protobuf trace from Chrome JSON input."""
    path = default_track_event_path(path)
    payload = chrome_trace_to_track_event_trace(
        trace,
        track_pattern=track_pattern,
        split_overlaps=split_overlaps,
        include_flows=include_flows,
    )
    with open(path, "wb") as f:
        f.write(payload)
    return path
