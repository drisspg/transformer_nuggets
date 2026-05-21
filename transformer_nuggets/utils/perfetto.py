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
import json
import re
import zlib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Iterator
from re import Pattern


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


from transformer_nuggets.utils.track_event import (
    chrome_trace_to_track_event_trace,
    default_track_event_path,
    write_track_event_trace,
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
