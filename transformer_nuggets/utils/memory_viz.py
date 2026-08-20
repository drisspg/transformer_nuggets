import html
import json
from functools import cache
from pathlib import Path

_SKIP_NAMES = {
    "torch::unwind::unwind()",
    "torch::CapturedTraceback::gather(bool, bool, bool)",
}

_CPYTHON_MARKERS = (
    "/usr/local/src/conda/python",
    "/conda-bld/python",
    "/cpython/",
)

_BARE_NOISE_PREFIXES = (
    "_Py",
    "Py_",
    "PyEval_",
    "PyObject_",
    "PyRun_",
    "pyrun",
    "pymain",
    "run_mod",
    "slot_tp_",
    "cfunction_",
    "vectorcall",
    "__libc_",
    "_start",
)


def _is_cpython_c_frame(fn: str, name: str) -> bool:
    if any(m in fn for m in _CPYTHON_MARKERS):
        return True
    return fn.endswith(".c") and name.startswith(("_Py", "Py", "pyrun", "pymain", "run_"))


def _is_bare_noise(name: str) -> bool:
    return name.startswith(_BARE_NOISE_PREFIXES) or ".llvm." in name


def _shorten_path(path: str) -> str:
    markers = ["/site-packages/", "/lib/python"]
    for marker in markers:
        idx = path.find(marker)
        if idx >= 0:
            return path[idx + len(marker) :]
    return path


def _extract_frames(frames: list[dict]) -> list[str]:
    result = []
    for f in frames:
        fn = f.get("filename", "")
        name = f.get("name", "")
        line = f.get("line", 0)
        if not name or name in _SKIP_NAMES:
            continue
        if _is_cpython_c_frame(fn, name):
            continue
        if fn and fn != "??" and fn != "":
            result.append(f"{_shorten_path(fn)}:{line} {name}")
        elif name and not _is_bare_noise(name):
            result.append(name)
    return result


def _normalize_pool_id(value: object) -> tuple[int, int] | None:
    """Normalize allocator pool IDs while accepting older snapshots without them."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def _is_private_pool(pool_id: tuple[int, int] | None) -> bool:
    """Return whether a pool ID identifies a non-default allocator pool."""
    return pool_id is not None and pool_id != (0, 0)


def _block_address(block: dict, segment: dict, offset: int = 0) -> int:
    """Read a block address or derive it from its position in the segment."""
    return int(block.get("address", block.get("addr", int(segment.get("address", 0)) + offset)))


def _block_requested_size(block: dict) -> int:
    """Return user-requested allocation bytes rather than rounded block bytes."""
    return int(block.get("requested_size", block.get("size", 0)))


def _find_pool_id(segments: list[dict], addr: int) -> tuple[int, int] | None:
    """Resolve an address by binary-searching segments sorted by start address."""
    left = 0
    right = len(segments) - 1
    while left <= right:
        middle = (left + right) // 2
        segment = segments[middle]
        start = int(segment.get("address", 0))
        if addr < start:
            right = middle - 1
        elif addr >= start + int(segment.get("total_size", 0)):
            left = middle + 1
        else:
            return _normalize_pool_id(segment.get("segment_pool_id"))
    return None


def _fx_metadata(raw_frames: list[dict]) -> list[dict]:
    """Preserve FX source metadata that is not represented in display frame strings."""
    keys = ("fx_node_op", "fx_node_name", "fx_node_target", "fx_original_trace")
    result = []
    for frame in raw_frames:
        metadata = {key: frame[key] for key in keys if frame.get(key)}
        if metadata:
            result.append(metadata)
    return result


def _allocation_record(
    *,
    addr: int,
    size: int,
    stack_idx: int,
    stream: int,
    pool_id: tuple[int, int] | None,
    time_us: float | None,
    compile_context: object = "",
    user_metadata: object = "",
    category: object = None,
    fx: list[dict] | None = None,
    ghost: bool = False,
    origin: str = "trace",
    block_size: int | None = None,
) -> dict:
    """Create the serialized allocation record consumed by both D3 views."""
    return {
        "si": stack_idx,
        "s": size,
        "block_size": size if block_size is None else block_size,
        "ts": [],
        "offsets": [],
        "addr": f"0x{addr:x}",
        "addr_int": addr,
        "stream": stream,
        "pool": pool_id,
        "time_us": time_us,
        "ctx": compile_context or "",
        "metadata": user_metadata or "",
        "annotations": [],
        "category": category,
        "fx": fx or [],
        "ghost": ghost,
        "origin": origin,
        "free_requested": False,
    }


def process_snapshot(
    snapshot: dict, device: int = 0
) -> tuple[list[dict], list[dict], list[str], list[list[int]], int, int]:
    """Reconcile allocator history with current segment state into D3 timeline data."""
    traces = snapshot.get("device_traces", [])
    if device < 0 or device >= len(traces):
        return [], [], [], [], 0, 0

    device_segments = sorted(
        (
            segment
            for segment in snapshot.get("segments", [])
            if segment.get("device", 0) == device
        ),
        key=lambda segment: int(segment.get("address", 0)),
    )
    device_trace = traces[device]

    frame_to_idx: dict[str, int] = {}
    frames: list[str] = []
    stack_to_idx: dict[tuple[int, ...], int] = {}
    stacks: list[list[int]] = []
    content_cache: dict[tuple, int] = {}
    stack_identity_cache: dict[int, tuple[list[dict], int]] = {}
    fx_identity_cache: dict[int, tuple[list[dict], list[dict]]] = {}

    def get_stack_idx(raw_frames: list[dict]) -> int:
        identity = stack_identity_cache.get(id(raw_frames))
        if identity is not None and identity[0] is raw_frames:
            return identity[1]
        content_key = tuple(
            (frame.get("filename", ""), frame.get("name", ""), frame.get("line", 0))
            for frame in raw_frames
        )
        if content_key in content_cache:
            stack_idx = content_cache[content_key]
            stack_identity_cache[id(raw_frames)] = (raw_frames, stack_idx)
            return stack_idx
        frame_indices = []
        for frame in _extract_frames(raw_frames):
            if frame not in frame_to_idx:
                frame_to_idx[frame] = len(frames)
                frames.append(frame)
            frame_indices.append(frame_to_idx[frame])
        key = tuple(frame_indices)
        if key not in stack_to_idx:
            stack_to_idx[key] = len(stacks)
            stacks.append(frame_indices)
        stack_idx = stack_to_idx[key]
        content_cache[content_key] = stack_idx
        stack_identity_cache[id(raw_frames)] = (raw_frames, stack_idx)
        return stack_idx

    def get_fx_metadata(raw_frames: list[dict]) -> list[dict]:
        identity = fx_identity_cache.get(id(raw_frames))
        if identity is not None and identity[0] is raw_frames:
            return identity[1]
        metadata = _fx_metadata(raw_frames)
        fx_identity_cache[id(raw_frames)] = (raw_frames, metadata)
        return metadata

    event_stack_indices = [get_stack_idx(entry.get("frames", [])) for entry in device_trace]
    alloc_polys: list[dict] = []
    live_by_addr: dict[int, int] = {}
    annotations_by_addr: dict[int, list[object]] = {}
    initially_allocated: list[int] = []
    event_actions: dict[int, tuple[str, int]] = {}

    for event_idx, entry in enumerate(device_trace):
        action = entry.get("action", "")
        addr = int(entry.get("addr", 0))
        raw_frames = entry.get("frames", [])
        match action:
            case "alloc":
                annotations_by_addr.pop(addr, None)
                pool_id = _normalize_pool_id(entry.get("pool_id")) or _find_pool_id(
                    device_segments, addr
                )
                alloc_id = len(alloc_polys)
                alloc_polys.append(
                    _allocation_record(
                        addr=addr,
                        size=int(entry.get("size", 0)),
                        stack_idx=event_stack_indices[event_idx],
                        stream=int(entry.get("stream", 0)),
                        pool_id=pool_id,
                        time_us=entry.get("time_us"),
                        compile_context=entry.get("compile_context"),
                        user_metadata=entry.get("user_metadata"),
                        category=entry.get("category"),
                        fx=get_fx_metadata(raw_frames),
                    )
                )
                live_by_addr[addr] = alloc_id
                event_actions[event_idx] = ("alloc", alloc_id)
            case "free_requested":
                alloc_id = live_by_addr.get(addr)
                if alloc_id is not None:
                    alloc_polys[alloc_id]["free_requested"] = True
            case "free_completed" | "free":
                annotations_by_addr.pop(addr, None)
                alloc_id = live_by_addr.pop(addr, None)
                if alloc_id is None:
                    pool_id = _normalize_pool_id(entry.get("pool_id")) or _find_pool_id(
                        device_segments, addr
                    )
                    alloc_id = len(alloc_polys)
                    alloc_polys.append(
                        _allocation_record(
                            addr=addr,
                            size=int(entry.get("size", 0)),
                            stack_idx=event_stack_indices[event_idx],
                            stream=int(entry.get("stream", 0)),
                            pool_id=pool_id,
                            time_us=None,
                            compile_context=entry.get("compile_context"),
                            user_metadata=entry.get("user_metadata"),
                            category=entry.get("category"),
                            fx=get_fx_metadata(raw_frames),
                            ghost=True,
                            origin="unmatched_free",
                        )
                    )
                    initially_allocated.append(alloc_id)
                event_actions[event_idx] = ("free", alloc_id)
            case "annotate":
                annotation = entry.get("user_metadata", "")
                annotations_by_addr.setdefault(addr, []).append(annotation)
                if addr in live_by_addr:
                    alloc_polys[live_by_addr[addr]]["annotations"].append(annotation)
            case _:
                pass

    for segment in device_segments:
        pool_id = _normalize_pool_id(segment.get("segment_pool_id"))
        block_offset = 0
        for block in segment.get("blocks", []):
            addr = _block_address(block, segment, block_offset)
            block_offset += int(block.get("size", 0))
            if block.get("state") not in {
                "active_allocated",
                "active_pending_free",
                "active_awaiting_free",
            }:
                continue
            if addr in live_by_addr:
                continue
            raw_frames = block.get("frames", [])
            if not raw_frames and block.get("history"):
                raw_frames = block["history"][0].get("frames", [])
            alloc_id = len(alloc_polys)
            alloc_polys.append(
                _allocation_record(
                    addr=addr,
                    size=_block_requested_size(block),
                    block_size=int(block.get("size", 0)),
                    stack_idx=get_stack_idx(raw_frames),
                    stream=int(segment.get("stream", 0)),
                    pool_id=pool_id,
                    time_us=None,
                    user_metadata=block.get("user_metadata"),
                    category=block.get("category"),
                    fx=get_fx_metadata(raw_frames),
                    ghost=True,
                    origin="snapshot",
                )
            )
            alloc_polys[alloc_id]["annotations"] = annotations_by_addr.get(addr, []).copy()
            alloc_polys[alloc_id]["free_requested"] = block.get("state") != "active_allocated"
            initially_allocated.append(alloc_id)
            live_by_addr[addr] = alloc_id

    current_stack: list[int] = []
    stack_pos: dict[int, int] = {}
    allocated = 0
    hwm = 0
    hwm_at_timestep = 0
    timeline: list[dict] = []

    def add_allocation(alloc_id: int, timestep: int) -> None:
        nonlocal allocated
        poly = alloc_polys[alloc_id]
        poly["ts"].append(timestep)
        poly["offsets"].append(allocated)
        stack_pos[alloc_id] = len(current_stack)
        current_stack.append(alloc_id)
        allocated += poly["s"]

    def free_allocation(alloc_id: int, timestep: int) -> None:
        nonlocal allocated
        idx_in_stack = stack_pos.pop(alloc_id, None)
        if idx_in_stack is None:
            return
        poly = alloc_polys[alloc_id]
        poly["ts"].append(timestep)
        poly["offsets"].append(poly["offsets"][-1])
        current_stack.pop(idx_in_stack)
        allocated -= poly["s"]
        for stack_idx in range(idx_in_stack, len(current_stack)):
            above_id = current_stack[stack_idx]
            stack_pos[above_id] = stack_idx
            above = alloc_polys[above_id]
            above["ts"].extend((timestep, timestep))
            above["offsets"].extend((above["offsets"][-1], above["offsets"][-1] - poly["s"]))

    # Reverse unmatched frees so the earliest freed reconstructed block is stacked highest.
    initially_allocated.reverse()
    for alloc_id in initially_allocated:
        add_allocation(alloc_id, 0)

    final_reserved = sum(int(segment.get("total_size", 0)) for segment in device_segments)
    reserved_delta = sum(
        int(entry.get("size", 0))
        * (1 if entry.get("action") in {"segment_alloc", "segment_map"} else -1)
        for entry in device_trace
        if entry.get("action") in {"segment_alloc", "segment_map", "segment_free", "segment_unmap"}
    )
    reserved = max(0, final_reserved - reserved_delta)
    timestep_base = 1 if initially_allocated else 0
    if initially_allocated:
        hwm = allocated
        timeline.append(
            {
                "step": 0,
                "t": None,
                "a": allocated,
                "r": reserved,
                "h": hwm,
                "act": "preexisting",
                "s": allocated,
                "si": 0,
                "addr": None,
                "pool": None,
                "device_free": None,
                "metadata": "",
            }
        )

    for event_idx, entry in enumerate(device_trace):
        timestep = timestep_base + event_idx
        event_action = event_actions.get(event_idx)
        if event_action is not None:
            action, alloc_id = event_action
            if action == "alloc":
                add_allocation(alloc_id, timestep)
            else:
                free_allocation(alloc_id, timestep)

        action = entry.get("action", "")
        if action in {"segment_alloc", "segment_map"}:
            reserved += int(entry.get("size", 0))
        elif action in {"segment_free", "segment_unmap"}:
            reserved = max(0, reserved - int(entry.get("size", 0)))

        if allocated > hwm:
            hwm = allocated
            hwm_at_timestep = timestep
        timeline.append(
            {
                "step": timestep,
                "t": entry.get("time_us"),
                "a": allocated,
                "r": reserved,
                "h": hwm,
                "act": action,
                "s": int(entry.get("size", 0)),
                "si": event_stack_indices[event_idx],
                "addr": f"0x{int(entry['addr']):x}" if "addr" in entry else None,
                "pool": _normalize_pool_id(entry.get("pool_id"))
                or _find_pool_id(device_segments, int(entry.get("addr", 0))),
                "device_free": entry.get("device_free"),
                "metadata": entry.get("user_metadata", ""),
            }
        )

    max_timestep = timestep_base + len(device_trace)
    for alloc_id in current_stack:
        poly = alloc_polys[alloc_id]
        poly["ts"].append(max_timestep)
        poly["offsets"].append(poly["offsets"][-1])

    return timeline, alloc_polys, frames, stacks, max_timestep, hwm_at_timestep


def _json_for_html(data: object) -> str:
    return json.dumps(data).replace("<", r"\u003c")


@cache
def _d3_source() -> str:
    """Load the vendored D3 runtime for self-contained offline visualizations."""
    source = Path(__file__).with_name("static").joinpath("d3.v7.min.js").read_text()
    return source.replace("</script", r"<\/script")


def _timeline_values(timeline: list[dict], max_timestep: int, key: str) -> list[int]:
    """Expand sparse event samples into one value for every chart timestep."""
    values = [0] * (max_timestep + 1)
    entries_by_step = {entry["step"]: int(entry[key]) for entry in timeline}
    current = 0
    for timestep in range(max_timestep + 1):
        current = entries_by_step.get(timestep, current)
        values[timestep] = current
    return values


def _segment_data(snapshot: dict, device: int) -> list[dict]:
    """Serialize current segment and block state for allocator-state inspection."""
    result = []
    for segment in snapshot.get("segments", []):
        if segment.get("device", 0) != device:
            continue
        blocks = []
        block_offset = 0
        for block in segment.get("blocks", []):
            blocks.append(
                {
                    "address": f"0x{_block_address(block, segment, block_offset):x}",
                    "size": int(block.get("size", 0)),
                    "requested_size": _block_requested_size(block),
                    "state": block.get("state", "unknown"),
                    "metadata": block.get("user_metadata", ""),
                }
            )
            block_offset += int(block.get("size", 0))
        result.append(
            {
                "address": f"0x{int(segment.get('address', 0)):x}",
                "total_size": int(segment.get("total_size", 0)),
                "allocated_size": int(segment.get("allocated_size", 0)),
                "active_size": int(segment.get("active_size", 0)),
                "stream": int(segment.get("stream", 0)),
                "pool": _normalize_pool_id(segment.get("segment_pool_id")),
                "segment_type": segment.get("segment_type", "unknown"),
                "expandable": bool(segment.get("is_expandable", False)),
                "metadata": segment.get("user_metadata", ""),
                "blocks": blocks,
            }
        )
    return result


def _private_pool_data(
    snapshot: dict, device: int, max_timestep: int, trace_step_offset: int
) -> list[dict]:
    """Summarize CUDA Graph and MemPool reserved and active memory over time."""
    segments = sorted(
        (
            segment
            for segment in snapshot.get("segments", [])
            if segment.get("device", 0) == device
        ),
        key=lambda segment: int(segment.get("address", 0)),
    )
    trace = snapshot.get("device_traces", [])[device]
    pools: dict[tuple[tuple[int, int], int], dict] = {}

    def get_pool(pool_id: tuple[int, int], stream: int) -> dict:
        key = (pool_id, stream)
        if key not in pools:
            pools[key] = {
                "id": pool_id,
                "stream": stream,
                "reserved_bytes": 0,
                "active_bytes": 0,
                "allocated_bytes": 0,
                "inactive_bytes": 0,
                "num_segments": 0,
                "num_blocks": 0,
                "net_trace_delta": 0,
                "events": [],
            }
        return pools[key]

    for segment in segments:
        pool_id = _normalize_pool_id(segment.get("segment_pool_id"))
        if not _is_private_pool(pool_id):
            continue
        pool = get_pool(pool_id, int(segment.get("stream", 0)))
        pool["reserved_bytes"] += int(segment.get("total_size", 0))
        pool["active_bytes"] += int(segment.get("active_size", 0))
        pool["allocated_bytes"] += int(segment.get("allocated_size", 0))
        pool["num_segments"] += 1
        for block in segment.get("blocks", []):
            pool["num_blocks"] += 1
            if block.get("state") == "inactive":
                pool["inactive_bytes"] += int(block.get("size", 0))

    for event_idx, entry in enumerate(trace):
        action = entry.get("action")
        if action not in {"segment_alloc", "segment_map", "segment_free", "segment_unmap"}:
            continue
        addr = int(entry.get("addr", 0))
        pool_id = _normalize_pool_id(entry.get("pool_id")) or _find_pool_id(segments, addr)
        if not _is_private_pool(pool_id):
            continue
        delta = int(entry.get("size", 0))
        if action in {"segment_free", "segment_unmap"}:
            delta = -delta
        pool = get_pool(pool_id, int(entry.get("stream", 0)))
        pool["net_trace_delta"] += delta
        pool["events"].append((event_idx, delta))

    for pool in pools.values():
        current = max(0, pool["reserved_bytes"] - pool.pop("net_trace_delta"))
        peak = current
        points = [{"step": 0, "reserved": current}]
        for event_idx, delta in pool.pop("events"):
            current = max(0, current + delta)
            peak = max(peak, current)
            points.append(
                {
                    "step": min(max_timestep, trace_step_offset + event_idx),
                    "reserved": current,
                }
            )
        if points[-1]["step"] < max_timestep:
            points.append({"step": max_timestep, "reserved": current})
        pool["peak_reserved_bytes"] = peak
        pool["timeline"] = points

    return sorted(pools.values(), key=lambda pool: pool["reserved_bytes"], reverse=True)


def _build_memory_viz_data(snapshot: dict, device: int, title: str) -> dict:
    timeline, alloc_polys, frames, stacks, max_ts, hwm_timestep = process_snapshot(
        snapshot, device
    )
    allocated_timeline = _timeline_values(timeline, max_ts, "a")
    reserved_timeline = _timeline_values(timeline, max_ts, "r")
    trace_step_offset = int(bool(timeline and timeline[0]["act"] == "preexisting"))
    private_pools = (
        _private_pool_data(snapshot, device, max_ts, trace_step_offset)
        if 0 <= device < len(snapshot.get("device_traces", []))
        else []
    )
    hwm = max(allocated_timeline, default=0)
    reserved_hwm = max(reserved_timeline, default=0)
    return {
        "timeline": allocated_timeline,
        "reserved_timeline": reserved_timeline,
        "allocs": alloc_polys,
        "frames": frames,
        "stacks": stacks,
        "events": timeline,
        "segments": _segment_data(snapshot, device),
        "private_pools": private_pools,
        "allocator_settings": snapshot.get("allocator_settings", {}),
        "meta": {
            "title": title,
            "device": device,
            "num_events": len(timeline),
            "num_allocs": len(alloc_polys),
            "high_water_mark_bytes": hwm,
            "reserved_high_water_mark_bytes": reserved_hwm,
            "current_reserved_bytes": reserved_timeline[-1] if reserved_timeline else 0,
            "private_pool_reserved_bytes": sum(pool["reserved_bytes"] for pool in private_pools),
            "num_private_pools": len(private_pools),
            "hwm_timestep": hwm_timestep,
            "max_timestep": max_ts,
        },
    }


def generate_memory_html(
    snapshot: dict,
    device: int = 0,
    title: str = "Memory Timeline",
) -> str:
    return (
        _MEMORY_VIZ_TEMPLATE.replace("__DOCUMENT_TITLE__", html.escape(title))
        .replace("__VISIBLE_TITLE__", html.escape(title))
        .replace("__BOOTSTRAP__", _json_for_html(_build_memory_viz_data(snapshot, device, title)))
        .replace("__D3_SOURCE__", _d3_source())
    )


def generate_memory_comparison_html(
    snapshot_left: dict,
    snapshot_right: dict,
    device: int = 0,
    device_left: int | None = None,
    device_right: int | None = None,
    title_left: str = "Left",
    title_right: str = "Right",
) -> str:
    if device_left is None:
        device_left = device
    if device_right is None:
        device_right = device
    doc_title = f"{title_left} vs {title_right}"
    return (
        _MEMORY_COMPARISON_TEMPLATE.replace("__DOCUMENT_TITLE__", html.escape(doc_title))
        .replace("__TITLE_LEFT__", html.escape(title_left))
        .replace("__TITLE_RIGHT__", html.escape(title_right))
        .replace(
            "__BOOTSTRAP_LEFT__",
            _json_for_html(_build_memory_viz_data(snapshot_left, device_left, title_left)),
        )
        .replace(
            "__BOOTSTRAP_RIGHT__",
            _json_for_html(_build_memory_viz_data(snapshot_right, device_right, title_right)),
        )
        .replace("__D3_SOURCE__", _d3_source())
    )


_MEMORY_VIZ_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__DOCUMENT_TITLE__</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');
  :root {
    --bg: #0E0E0E;
    --surface: #1a1a1a;
    --border: rgba(255, 255, 255, 0.10);
    --text: rgba(255, 255, 255, 0.92);
    --text-muted: rgba(255, 255, 255, 0.50);
    --accent: #3E93CC;
    --accent-light: rgba(62, 147, 204, 0.12);
    --accent-stroke: rgba(62, 147, 204, 0.7);
    --hwm-color: rgba(255, 255, 255, 0.60);
    --grid: rgba(255, 255, 255, 0.03);
    --tooltip-bg: #1f1f1f;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'IBM Plex Mono', 'Fira Mono', monospace;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  #header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    position: relative;
    z-index: 60;
  }

  #header h1 { font-size: 14px; font-weight: 500; font-family: var(--mono); letter-spacing: 0.03em; text-transform: uppercase; flex-shrink: 0; }

  #header-mid {
    display: flex;
    gap: 12px;
    align-items: center;
    flex: 1;
    justify-content: center;
  }

  #help-dropdown {
    display: none;
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    margin-top: 6px;
    background: var(--tooltip-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 14px;
    white-space: nowrap;
    z-index: 50;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    font-family: var(--mono);
    font-size: 11px;
    line-height: 2;
    color: var(--text-muted);
  }

  #help-dropdown kbd {
    display: inline-block;
    padding: 1px 5px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text);
    min-width: 18px;
    text-align: center;
  }

  #help-trigger:hover #help-dropdown { display: block; }

  #settings-trigger {
    cursor: pointer;
    position: relative;
    font-size: 14px;
    opacity: 0.6;
    transition: opacity 0.15s;
    user-select: none;
  }
  #settings-trigger:hover { opacity: 1; }
  #settings-dropdown {
    display: none;
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 6px;
    background: var(--tooltip-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    white-space: nowrap;
    z-index: 50;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
  }
  #settings-trigger.open #settings-dropdown { display: block; }
  #settings-dropdown label { display: flex; align-items: center; gap: 6px; }
  #settings-dropdown select {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 2px 4px;
    font-size: 11px;
    font-family: var(--mono);
    cursor: pointer;
  }

  #controls-shell {
    position: relative;
    flex-shrink: 0;
    margin-left: auto;
  }

  #controls-toggle {
    width: 26px;
    height: 26px;
    display: grid;
    place-items: center;
    background: rgba(255,255,255,0.04);
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
  }
  #controls-toggle:hover { color: var(--text); background: rgba(255,255,255,0.08); }

  #controls {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    width: max-content;
    max-width: min(860px, calc(100vw - 24px));
    padding: 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 5px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.55);
    z-index: 80;
  }
  #controls.collapsed { display: none; }

  @media (max-width: 1050px) {
    #header-mid { display: none; }
    #header { padding: 10px 14px; }
  }

  .toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
  }

  .toggle input[type="checkbox"] {
    accent-color: var(--accent);
    width: 14px;
    height: 14px;
  }

  .toggle:hover { color: var(--text); }

  .stat {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--text-muted);
    padding: 4px 10px;
    background: rgba(255,255,255,0.04);
    border-radius: 3px;
    border: 1px solid var(--border);
  }

  .stat strong { color: var(--text); font-weight: 500; }

  #main {
    display: flex;
    flex: 1;
    min-height: 0;
  }

  #chart-container {
    flex: 1;
    padding: 0;
    min-height: 0;
    position: relative;
  }

  #alloc-canvas {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
  }

  #chart-container > svg {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 1;
  }

  #detail-panel {
    width: 480px;
    border-left: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
    position: relative;
    transition: width 0.15s, min-width 0.15s;
    min-width: 480px;
  }

  #detail-panel.collapsed {
    width: 0 !important;
    min-width: 0 !important;
    border-left: none;
    overflow: hidden;
  }

  #panel-toggle {
    width: 22px;
    height: 48px;
    align-self: center;
    flex-shrink: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px 0 0 4px;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
  }
  #panel-toggle:hover { color: var(--text); background: rgba(255,255,255,0.06); }

  #resize-handle {
    position: absolute;
    left: 0;
    top: 0;
    width: 4px;
    height: 100%;
    cursor: col-resize;
    z-index: 11;
  }
  #resize-handle:hover, #resize-handle.dragging { background: var(--accent); }

  #detail-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    font-weight: 500;
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }

  #detail-header .detail-stats {
    font-weight: 400;
    color: var(--text-muted);
    font-size: 12px;
  }

  #detail-header .detail-actions {
    display: flex;
    gap: 4px;
    align-items: center;
  }


  #detail-body {
    flex: 1;
    overflow-y: auto;
    padding: 0;
  }

  #detail-body::-webkit-scrollbar { width: 6px; }
  #detail-body::-webkit-scrollbar-track { background: transparent; }
  #detail-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .stack-frame {
    padding: 3px 16px;
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.5;
    cursor: pointer;
    overflow: hidden;
    border-left: 2px solid transparent;
  }

  .stack-frame .frame-text {
    white-space: pre-wrap;
    word-break: break-all;
    display: block;
  }

  .stack-frame:hover { background: rgba(255,255,255,0.04); }

  .stack-frame.frame-user {
    color: var(--text);
    border-left-color: #49C963;
    background: rgba(73, 201, 99, 0.04);
  }
  .stack-frame.frame-user .frame-func { color: #49C963; font-weight: 500; }
  .stack-frame.frame-user .frame-basename { color: var(--text); }
  .stack-frame.frame-user .frame-file { color: rgba(255,255,255,0.4); }

  .stack-frame.frame-library {
    color: rgba(255,255,255,0.6);
    border-left-color: #3E93CC;
  }
  .stack-frame.frame-library .frame-func { color: #5BA8D9; }
  .stack-frame.frame-library .frame-basename { color: rgba(255,255,255,0.7); }
  .stack-frame.frame-library .frame-file { color: rgba(255,255,255,0.25); }

  .stack-frame.frame-native {
    color: rgba(255,255,255,0.35);
  }
  .stack-frame.frame-native .frame-cpp { color: rgba(189, 147, 249, 0.5); }
  .stack-frame.frame-native .frame-basename { color: rgba(255,255,255,0.5); }

  .stack-frame.frame-noise {
    color: rgba(255,255,255,0.18);
    font-size: 10px;
  }

  .frame-noise { display: none; }

  .alloc-details {
    padding: 12px 16px;
    font-family: var(--mono);
    font-size: 12px;
  }

  .alloc-details table {
    width: 100%;
    border-collapse: collapse;
  }

  .alloc-details td {
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    vertical-align: top;
  }

  .alloc-details td:first-child {
    color: var(--text-muted);
    width: 100px;
    padding-right: 12px;
  }

  .alloc-details td:last-child {
    color: var(--text);
    word-break: break-all;
  }

  .empty-detail {
    padding: 24px 16px;
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
  }

  .axis text { fill: var(--text-muted); font-size: 11px; font-family: var(--font); }
  .axis line, .axis path { stroke: var(--border); }
  .grid line { stroke: var(--grid); }
  .grid path { stroke: none; }

  .hwm-line { stroke: var(--hwm-color); stroke-width: 0.75; stroke-dasharray: 8 4; }
  .hwm-label { fill: var(--hwm-color); font-size: 11px; font-family: var(--mono); font-weight: 500; letter-spacing: 0.02em; }
  .reserved-line { fill: none; stroke: #C97049; stroke-width: 1.25; stroke-dasharray: 4 3; }
  .pool-reserved-line { fill: none; stroke: rgba(255,255,255,0.45); stroke-width: 1; stroke-dasharray: 2 2; }
  .event-marker { stroke-width: 1; stroke-dasharray: 2 3; opacity: 0.8; }
  .event-marker.oom { stroke: #e74c3c; }
  .event-marker.snapshot { stroke: #f1c40f; }
  .detail-json { padding: 12px 16px; white-space: pre-wrap; word-break: break-word; font: 11px/1.5 var(--mono); color: var(--text-muted); }

  #search-input {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 4px 10px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    width: 180px;
    outline: none;
  }

  #search-input:focus { border-color: var(--accent); }
  #search-input::placeholder { color: rgba(255,255,255,0.25); }

  #regex-toggle {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-left: none;
    border-radius: 0 3px 3px 0;
    padding: 4px 8px;
    color: var(--text-muted);
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    height: 100%;
  }

  #regex-toggle:hover { color: var(--text); }
  #regex-toggle.active { background: var(--accent); color: white; border-color: var(--accent); }

  #search-input { border-radius: 3px 0 0 3px; }

  #minimap {
    height: 40px;
    padding: 0 16px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }

  #minimap svg { width: 100%; height: 100%; }

  .minimap-area { fill: rgba(62, 147, 204, 0.3); }
  .minimap-viewport {
    fill: rgba(255,255,255,0.06);
    stroke: rgba(255,255,255,0.3);
    stroke-width: 1;
    cursor: grab;
  }
  .minimap-viewport:active { cursor: grabbing; }

  .detail-tabs {
    display: flex;
    gap: 0;
  }

  .detail-tab {
    padding: 4px 12px;
    font-size: 10px;
    font-family: var(--mono);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
    cursor: pointer;
  }

  .detail-tab:first-child { border-radius: 3px 0 0 3px; }
  .detail-tab:last-child { border-radius: 0 3px 3px 0; }
  .detail-tab + .detail-tab { border-left: none; }
  .detail-tab.active { background: var(--accent); color: white; border-color: var(--accent); }
  .detail-tabs { flex-wrap: wrap; }

  .breakdown-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 16px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    border-bottom: 1px solid rgba(255,255,255,0.03);
    cursor: pointer;
  }

  .breakdown-row:hover { background: rgba(255,255,255,0.03); color: var(--text); }

  .breakdown-row .bd-size {
    min-width: 70px;
    text-align: right;
    color: var(--text);
    font-weight: 500;
  }

  .breakdown-row .bd-count {
    min-width: 30px;
    text-align: right;
    color: rgba(255,255,255,0.3);
    font-size: 10px;
  }

  .breakdown-row .bd-pct {
    min-width: 40px;
    text-align: right;
    color: var(--accent);
    font-size: 10px;
  }

  .breakdown-row .bd-bar {
    width: 60px;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
    flex-shrink: 0;
  }

  .breakdown-row .bd-bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
  }

  .breakdown-row .bd-bar-fill.leak-bar {
    background: #e74c3c;
  }

  .breakdown-row .bd-frame {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .peak-label {
    padding: 8px 16px;
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    background: rgba(255,255,255,0.02);
  }

  #tooltip {
    position: fixed; display: none;
    background: #1f1f1f; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px; padding: 10px 14px;
    font-size: 12px; line-height: 1.6;
    pointer-events: none; z-index: 100; max-width: 500px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    font-family: var(--mono);
  }

  #tooltip .tt-label { color: var(--text-muted); margin-right: 4px; }
  #tooltip .tt-value { color: var(--text); font-weight: 500; font-family: var(--mono); }
  #tooltip .tt-row { white-space: nowrap; }
  #tooltip .tt-hint {
    margin-top: 4px; padding-top: 4px; border-top: 1px solid var(--border);
    color: var(--text-muted); font-size: 10px; font-style: italic;
  }
  #tooltip .tt-api { color: #78BBE3; font-size: 11px; font-weight: 500; }
  #tooltip .tt-user { color: #49C963; font-size: 10px; }


  #perf-display {
    position: fixed;
    bottom: 8px;
    left: 8px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-muted);
    background: rgba(0,0,0,0.7);
    padding: 3px 8px;
    border-radius: 3px;
    z-index: 200;
    pointer-events: none;
  }

  #shortcut-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 6px 24px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    font-size: 11px;
    color: var(--text-muted);
    flex-shrink: 0;
  }

  #shortcut-bar kbd {
    display: inline-block;
    padding: 1px 5px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text);
    min-width: 18px;
    text-align: center;
  }

  #shortcut-bar .sep {
    width: 1px;
    height: 14px;
    background: var(--border);
  }

  #speed-indicator {
    color: var(--accent);
    font-weight: 600;
  }
</style>
</head>
<body>
<div id="header">
  <h1>__VISIBLE_TITLE__</h1>
  <div id="header-mid">
    <span class="stat">Peak: <strong id="peak-stat"></strong></span>
    <span class="stat">Reserved: <strong id="reserved-stat"></strong></span>
    <span class="stat">Pools: <strong id="pools-stat"></strong></span>
    <span class="stat">Allocs: <strong id="allocs-stat"></strong></span>
    <span class="stat">Events: <strong id="events-stat"></strong></span>
    <span id="help-trigger" class="stat" style="cursor:help;position:relative;">
      ? controls
      <div id="help-dropdown">
        <div><kbd>scroll</kbd> zoom X</div>
        <div><kbd>drag</kbd> pan X</div>
        <div><kbd>shift+drag</kbd> box zoom (X+Y)</div>
        <div><kbd>dbl-click</kbd> reset view</div>
        <div><kbd>click</kbd> inspect allocation stack</div>
        <div><kbd>A</kbd><kbd>D</kbd> pan &nbsp; <kbd>W</kbd><kbd>S</kbd> zoom</div>
        <div><kbd>[</kbd><kbd>]</kbd> change speed</div>
        <div><kbd>/</kbd> search &nbsp; <kbd>esc</kbd> clear</div>
      </div>
    </span>
  </div>
  <div id="controls-shell">
    <button id="controls-toggle" title="Show controls" aria-expanded="false">&#9664;</button>
    <div id="controls" class="collapsed">
    <div style="display:flex;align-items:center;gap:0;">
      <input type="text" id="search-input" placeholder="/ search allocations...">
      <button id="regex-toggle" title="Toggle regex mode">.*</button>
    </div>
    <label class="toggle">
      <input type="checkbox" id="autofit-toggle">
      Auto-fit Y
    </label>
    <label class="toggle">
      <input type="checkbox" id="hwm-toggle" checked>
      High Water Mark
    </label>
    <label class="toggle" title="Show cached allocator memory, including CUDA Graph private pools">
      <input type="checkbox" id="reserved-toggle">
      Reserved memory
    </label>
    <label class="toggle" title="Hide allocations that were never freed during recording (weights, buffers, etc.) and zoom to dynamic range">
      <input type="checkbox" id="dim-persistent-toggle">
      Hide never-freed
    </label>
    <span id="settings-trigger" title="Settings">&#9881;
      <div id="settings-dropdown">
        <label>Color by
          <select id="color-mode">
            <option value="stack">stack</option>
            <option value="size">size</option>
            <option value="category">category</option>
            <option value="order">order</option>
          </select>
        </label>
      </div>
    </span>
    </div>
  </div>
</div>
<div id="main">
  <div id="chart-container"></div>
  <button id="panel-toggle" title="Show stack/details" aria-expanded="false">&#9654;</button>
  <div id="detail-panel" class="collapsed">
    <div id="resize-handle"></div>
    <div id="detail-header">
      <div class="detail-tabs"></div>
      <div class="detail-actions">
        <span class="detail-stats" id="detail-stats"></span>
      </div>
    </div>
    <div id="detail-body">
      <div class="empty-detail">Click an allocation to inspect its stack trace</div>
    </div>
  </div>
</div>
<div id="minimap"></div>
<div id="shortcut-bar" style="display:none">
  <span><kbd>A</kbd><kbd>D</kbd> pan</span>
  <span><kbd>W</kbd><kbd>S</kbd> zoom</span>
  <div class="sep"></div>
  <span><kbd>[</kbd><kbd>]</kbd> speed: <span id="speed-indicator">3</span>/5</span>
  <div class="sep"></div>
  <span><kbd>/</kbd> search</span>
  <div class="sep"></div>
  <span><kbd>?</kbd> toggle shortcuts</span>
</div>
<div id="tooltip"></div>
<div id="perf-display"></div>

<script id="memory-viz-bootstrap" type="application/json">__BOOTSTRAP__</script>
<script>__D3_SOURCE__</script>
<script>
const BOOTSTRAP = JSON.parse(document.getElementById('memory-viz-bootstrap').textContent);
const {
  timeline: TIMELINE,
  reserved_timeline: RESERVED_TIMELINE,
  allocs: ALLOCS,
  frames: FRAMES,
  stacks: STACKS,
  events: EVENTS,
  segments: SEGMENTS,
  private_pools: PRIVATE_POOLS,
  allocator_settings: ALLOCATOR_SETTINGS,
  meta: META,
} = BOOTSTRAP;
const MAX_TS = Math.max(1, META.max_timestep);

function resolveStack(stackIdx) {
  const indices = STACKS[stackIdx] || [];
  return indices.map(i => FRAMES[i]);
}

function formatBytes(b) {
  if (Math.abs(b) >= 1024**3) return (b / 1024**3).toFixed(2) + ' GiB';
  if (Math.abs(b) >= 1024**2) return (b / 1024**2).toFixed(1) + ' MiB';
  if (Math.abs(b) >= 1024)    return (b / 1024).toFixed(0) + ' KiB';
  return b + ' B';
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

function formatPool(pool) {
  return pool ? `(${pool[0]}, ${pool[1]})` : 'default/unknown';
}

document.getElementById('peak-stat').textContent = formatBytes(META.high_water_mark_bytes);
document.getElementById('reserved-stat').textContent = formatBytes(META.current_reserved_bytes);
document.getElementById('pools-stat').textContent = `${META.num_private_pools} / ${formatBytes(META.private_pool_reserved_bytes)}`;
document.getElementById('allocs-stat').textContent = META.num_allocs.toLocaleString();
document.getElementById('events-stat').textContent = META.num_events.toLocaleString();

function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const k = (n + h / 30) % 12;
    const c = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * c).toString(16).padStart(2, '0');
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

const PALETTE = Array.from({length: 128}, (_, i) =>
  hslToHex((i * 137.508) % 360, 42, 52)
);

function getColor(stackIdx) {
  return PALETTE[stackIdx % PALETTE.length];
}

const SIZE_PALETTE = Array.from({length: 32}, (_, i) =>
  hslToHex((i * 137.508) % 360, 35, 48)
);

const allocSizes = ALLOCS.map(a => a.s);
const sortedSizes = [...new Set(allocSizes)].sort((a, b) => a - b);
const sizeToColorIdx = new Map();
sortedSizes.forEach((s, i) => sizeToColorIdx.set(s, i % SIZE_PALETTE.length));

function getSizeColor(allocIdx) {
  return SIZE_PALETTE[sizeToColorIdx.get(ALLOCS[allocIdx].s)];
}

const categoryValues = [...new Set(ALLOCS.map(a => a.category ?? 'unknown'))];
const categoryToColorIdx = new Map(categoryValues.map((category, i) => [category, i]));

function getCategoryColor(allocIdx) {
  return PALETTE[categoryToColorIdx.get(ALLOCS[allocIdx].category ?? 'unknown') % PALETTE.length];
}

let colorMode = 'stack';
let showReserved = false;

function recolorAllocs() {
  let pIdx = 0;
  for (let i = 0; i < ALLOCS.length; i++) {
    const isPersistent = allocPersistent[i];
    switch (colorMode) {
      case 'size': allocColors[i] = getSizeColor(i); break;
      case 'category': allocColors[i] = getCategoryColor(i); break;
      case 'order': allocColors[i] = PALETTE[i % PALETTE.length]; break;
      default: allocColors[i] = getColor(ALLOCS[i].si); break;
    }
    allocAlphas[i] = isPersistent
      ? PERSISTENT_ALPHAS[pIdx++ % PERSISTENT_ALPHAS.length]
      : 0.85;
  }
}

const PERSISTENT_ALPHAS = [0.55, 0.62, 0.70];

const tooltipEl = document.getElementById('tooltip');
const detailBody = document.getElementById('detail-body');
const detailStats = document.getElementById('detail-stats');
const detailTabs = document.querySelector('.detail-tabs');
const EMPTY_STACK_DETAIL = '<div class="empty-detail">Click an allocation to inspect its stack trace</div>';
const uiState = {
  activeDetailView: 'stack',
  selectedAlloc: null,
  selectedStackIdx: -1,
  selectedStackLabel: '',
};

function showTooltip(event, html) {
  tooltipEl.innerHTML = html;
  tooltipEl.style.display = 'block';
  const tw = tooltipEl.offsetWidth, th = tooltipEl.offsetHeight;
  tooltipEl.style.left = (event.pageX + 16 + tw > window.innerWidth ? event.pageX - tw - 12 : event.pageX + 16) + 'px';
  tooltipEl.style.top = (event.pageY + 16 + th > window.innerHeight ? event.pageY - th - 12 : event.pageY + 16) + 'px';
}

function hideTooltip() { tooltipEl.style.display = 'none'; }

function selectStack(stackIdx, label) {
  uiState.selectedStackIdx = stackIdx;
  uiState.selectedStackLabel = label;
}

function selectAlloc(alloc) {
  uiState.selectedAlloc = alloc;
  selectStack(alloc.si, formatBytes(alloc.s));
}

function classifyFrame(frame) {
  if (frame.includes('::')) return 'native';
  if (frame.includes('.cpp:') || frame.includes('.c:')) return 'native';
  if (!frame.includes('/') && !frame.includes('.py')) return 'noise';
  if (frame.includes('/site-packages/') || frame.includes('/torch/')) return 'library';
  if (frame.includes('/lib/python') || frame.includes('/conda/') || frame.includes('lib/python')) return 'library';
  return 'user';
}

const NOISE_FRAMES = new Set([
  'cfunction_call', '_PyEval_EvalFrameDefault', 'PyEval_EvalCode',
  '_PyObject_Call_Prepend', 'slot_tp_call', 'PyObject_Call',
  '_PyObject_MakeTpCall', '_PyFunction_Vectorcall', 'pymain_run_file',
  'pyrun_file', '_PyRun_SimpleFileObject', '_PyRun_AnyFileObject',
  'Py_RunMain', 'pymain_run_file_obj', 'pymain_run_module',
  '_start', '__libc_start_main', '__libc_init_first', 'main',
]);

function frameFunc(frame) {
  if (frame.includes('::')) return frame.split('::').pop();
  const sp = frame.indexOf(' ', frame.lastIndexOf(':'));
  return sp > 0 ? frame.substring(sp + 1) : frame;
}

function isNoiseFrame(frame) {
  const fn = frameFunc(frame);
  return NOISE_FRAMES.has(fn) || fn.startsWith('_Py') || fn.startsWith('Py_')
    || fn.startsWith('pymain_') || fn.startsWith('pyrun_')
    || /^run_mod\.llvm\.|^pymain_main\.llvm\./.test(fn);
}

function bestFrame(stackIdx) {
  const stack = resolveStack(stackIdx);
  for (const f of stack) {
    if (classifyFrame(f) === 'user') return f;
  }
  for (const f of stack) {
    if (f.includes('.py') && !isNoiseFrame(f)) return f;
  }
  for (const f of stack) {
    if (f.includes('::') && !isNoiseFrame(f)) return f;
  }
  for (const f of stack) {
    if (!isNoiseFrame(f)) return f;
  }
  return stack[0] || '';
}

function tooltipFrameInfo(stackIdx) {
  const stack = resolveStack(stackIdx);
  let userFrame = null, apiFrame = null;
  for (const f of stack) {
    if (classifyFrame(f) === 'user') { userFrame = f; break; }
  }
  for (const f of stack) {
    if (f.includes('.py') && !isNoiseFrame(f) && classifyFrame(f) === 'library') {
      apiFrame = f; break;
    }
  }
  if (!apiFrame) {
    for (const f of stack) {
      if (f.includes('::') && !isNoiseFrame(f)) { apiFrame = f; break; }
    }
  }
  return { userFrame, apiFrame };
}

function renderFrame(frame) {
  const hasColon = frame.includes(':');
  const isCpp = !hasColon && frame.includes('::');
  if (isCpp) {
    const parts = frame.split('::');
    const funcName = parts[parts.length - 1];
    const ns = parts.slice(0, -1).join('::');
    return `<span class="frame-cpp">${escapeHtml(ns)}::</span><span class="frame-basename">${escapeHtml(funcName)}</span>`;
  }
  if (hasColon) {
    const sp = frame.indexOf(' ', frame.lastIndexOf(':'));
    if (sp > 0) {
      const filePart = frame.substring(0, sp);
      const funcPart = frame.substring(sp + 1);
      const lastSlash = filePart.lastIndexOf('/');
      const basename = lastSlash >= 0 ? filePart.substring(lastSlash + 1) : filePart;
      const dir = lastSlash >= 0 ? filePart.substring(0, lastSlash + 1) : '';
      return `<span class="frame-file">${escapeHtml(dir)}</span><span class="frame-basename">${escapeHtml(basename)}</span> <span class="frame-func">${escapeHtml(funcPart)}</span>`;
    }
    return `<span class="frame-file">${escapeHtml(frame)}</span>`;
  }
  return escapeHtml(frame);
}

function renderStack(stackIdx, label) {
  const stack = resolveStack(stackIdx);
  detailStats.textContent = label;
  if (!stack.length) {
    detailBody.innerHTML = '<div class="empty-detail">No frames recorded</div>';
    return;
  }

  detailBody.innerHTML = stack.map(f => {
    const cls = classifyFrame(f);
    return `<div class="stack-frame frame-${cls}"><span class="frame-text">${renderFrame(f)}</span></div>`;
  }).join('');
}

function renderStackSelection() {
  if (uiState.selectedStackIdx >= 0) {
    renderStack(uiState.selectedStackIdx, uiState.selectedStackLabel);
    return;
  }
  detailBody.innerHTML = EMPTY_STACK_DETAIL;
}

// --- Chart setup ---
const container = document.getElementById('chart-container');
let containerRect = container.getBoundingClientRect();
const margin = { top: 20, right: 60, bottom: 40, left: 80 };
let width = Math.max(1, containerRect.width - margin.left - margin.right);
let height = Math.max(1, containerRect.height - margin.top - margin.bottom);

// Canvas for allocation polygons (behind SVG)
const canvas = document.createElement('canvas');
canvas.id = 'alloc-canvas';
canvas.width = Math.max(1, containerRect.width * devicePixelRatio);
canvas.height = Math.max(1, containerRect.height * devicePixelRatio);
canvas.style.width = containerRect.width + 'px';
canvas.style.height = containerRect.height + 'px';
container.insertBefore(canvas, container.firstChild);
const ctx = canvas.getContext('2d');
ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

// SVG for axes, grid, HWM, zoom overlay
const svg = d3.select('#chart-container').append('svg')
  .attr('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`)
  .attr('preserveAspectRatio', 'none');

const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

const clipRect = svg.append('defs').append('clipPath').attr('id', 'clip')
  .append('rect').attr('width', width).attr('height', height);

const xScale = d3.scaleLinear().domain([0, MAX_TS]).range([0, width]);
const yScale = d3.scaleLinear().domain([0, Math.max(1, META.high_water_mark_bytes) * 1.05]).range([height, 0]);

const xAxis = d3.axisBottom(xScale).ticks(10);
const yAxisFn = d3.axisLeft(yScale).ticks(8).tickFormat(d => formatBytes(d));

const gridG = g.append('g').attr('class', 'grid')
  .call(d3.axisLeft(yScale).ticks(8).tickSize(-width).tickFormat(''));

const xAxisG = g.append('g').attr('class', 'axis x-axis')
  .attr('transform', `translate(0,${height})`).call(xAxis);

const yAxisG = g.append('g').attr('class', 'axis y-axis').call(yAxisFn);

const chartArea = g.append('g').attr('clip-path', 'url(#clip)');

// HWM (clickable for peak breakdown)
const hwmG = chartArea.append('g').attr('class', 'hwm-group').style('cursor', 'pointer');
hwmG.append('line').attr('class', 'hwm-line')
  .attr('x1', 0).attr('x2', width)
  .attr('y1', yScale(META.high_water_mark_bytes)).attr('y2', yScale(META.high_water_mark_bytes));
hwmG.append('text').attr('class', 'hwm-label')
  .attr('x', width - 4).attr('y', yScale(META.high_water_mark_bytes) - 6)
  .attr('text-anchor', 'end')
  .text('HWM: ' + formatBytes(META.high_water_mark_bytes));

const reservedPath = chartArea.append('path').attr('class', 'reserved-line').style('display', 'none');
const poolLineG = chartArea.append('g').attr('class', 'pool-reserved-lines');
const markerData = EVENTS.filter(event => event.act === 'oom' || event.act === 'snapshot');
const markerG = chartArea.append('g').attr('class', 'event-markers');

// --- Canvas rendering ---
let currentTransform = d3.zoomIdentity;
let searchMatcher = null;
let hoveredAlloc = null;

// Precompute colors and start/end arrays for fast access
const allocStarts = new Float64Array(ALLOCS.length);
const allocEnds = new Float64Array(ALLOCS.length);
for (let i = 0; i < ALLOCS.length; i++) {
  allocStarts[i] = ALLOCS[i].ts[0];
  allocEnds[i] = ALLOCS[i].ts[ALLOCS[i].ts.length - 1];
}

const allocPersistent = new Uint8Array(ALLOCS.length);
const allocColors = new Array(ALLOCS.length);
const allocAlphas = new Float64Array(ALLOCS.length);
for (let i = 0; i < ALLOCS.length; i++) {
  allocPersistent[i] = allocEnds[i] >= META.max_timestep ? 1 : 0;
}

function buildDerivedData() {
  const stackFrameLabels = Array.from({ length: STACKS.length }, (_, stackIdx) => bestFrame(stackIdx));
  const peakAllocIndices = [];
  let peakTotalBytes = 0;
  const leakAllocIndices = [];
  let leakTotalBytes = 0;
  const leakGroupsByFrame = new Map();
  const peakTs = META.hwm_timestep;
  const maxTs = META.max_timestep;
  const earlyThreshold = maxTs * 0.05;

  for (let ai = 0; ai < ALLOCS.length; ai++) {
    const alloc = ALLOCS[ai];
    const firstTs = allocStarts[ai];
    const lastTs = allocEnds[ai];

    if (firstTs <= peakTs && lastTs >= peakTs) {
      peakAllocIndices.push(ai);
      peakTotalBytes += alloc.s;
    }

    if (lastTs >= maxTs && firstTs > earlyThreshold) {
      leakAllocIndices.push(ai);
      leakTotalBytes += alloc.s;

      const frame = stackFrameLabels[alloc.si];
      let group = leakGroupsByFrame.get(frame);
      if (!group) {
        group = { frame, si: alloc.si, count: 0, totalBytes: 0 };
        leakGroupsByFrame.set(frame, group);
      }
      group.count += 1;
      group.totalBytes += alloc.s;
    }
  }

  peakAllocIndices.sort((left, right) => ALLOCS[right].s - ALLOCS[left].s);

  return {
    stackFrameLabels,
    peakAllocIndices,
    peakTotalBytes,
    leakAllocIndices,
    leakTotalBytes,
    leakGroups: Array.from(leakGroupsByFrame.values()).sort(
      (left, right) => right.totalBytes - left.totalBytes
    ),
  };
}

const derivedData = buildDerivedData();
recolorAllocs();
let dimPersistent = false;

// Bucket index for O(bucket_size) hit testing instead of O(n)
const NUM_HIT_BUCKETS = Math.max(1, Math.min(2000, MAX_TS));
const hitBucketSize = MAX_TS / NUM_HIT_BUCKETS;
const hitBuckets = new Array(NUM_HIT_BUCKETS + 1);
for (let b = 0; b <= NUM_HIT_BUCKETS; b++) hitBuckets[b] = [];
for (let ai = 0; ai < ALLOCS.length; ai++) {
  const b0 = Math.max(0, Math.floor(allocStarts[ai] / hitBucketSize));
  const b1 = Math.min(NUM_HIT_BUCKETS, Math.floor(allocEnds[ai] / hitBucketSize));
  for (let b = b0; b <= b1; b++) hitBuckets[b].push(ai);
}

// Search match cache: precompute on search change instead of per-frame
let searchMatchSet = null;
function updateSearchCache() {
  if (!searchMatcher) { searchMatchSet = null; return; }
  searchMatchSet = new Set();
  for (let ai = 0; ai < ALLOCS.length; ai++) {
    const stack = resolveStack(ALLOCS[ai].si);
    if (stack.some(f => searchMatcher.test(f))) searchMatchSet.add(ai);
  }
}

function tracePoly(ai, newX) {
  const d = ALLOCS[ai];
  const ts = d.ts, offsets = d.offsets, size = d.s;
  ctx.moveTo(newX(ts[0]), yScale(offsets[0]));
  for (let i = 1; i < ts.length; i++) {
    ctx.lineTo(newX(ts[i]), yScale(offsets[i]));
  }
  for (let i = ts.length - 1; i >= 0; i--) {
    ctx.lineTo(newX(ts[i]), yScale(offsets[i] + size));
  }
  ctx.closePath();
}

const perfEl = document.getElementById('perf-display');
let perfFrames = 0, perfSum = 0, perfLastUpdate = performance.now();

function drawCanvas() {
  const t0 = performance.now();
  const newX = currentTransform.rescaleX(xScale);
  const [d0, d1] = newX.domain();

  ctx.clearRect(0, 0, containerRect.width, containerRect.height);
  ctx.save();
  ctx.translate(margin.left, margin.top);
  ctx.beginPath();
  ctx.rect(0, 0, width, height);
  ctx.clip();

  const pxPerTs = width / (d1 - d0);
  const minVisPx = 0.5;

  // Batch visible allocs by color+alpha to minimize Canvas state changes
  const batches = {};
  let hoveredIdx = -1;

  for (let ai = 0; ai < ALLOCS.length; ai++) {
    if (allocEnds[ai] < d0 || allocStarts[ai] > d1) continue;

    const visW = (Math.min(allocEnds[ai], d1) - Math.max(allocStarts[ai], d0)) * pxPerTs;
    const visH = yScale(0) - yScale(ALLOCS[ai].s);
    if (visW < minVisPx && visH < minVisPx) continue;

    if (dimPersistent && allocPersistent[ai]) continue;

    if (ALLOCS[ai] === hoveredAlloc) { hoveredIdx = ai; continue; }

    let alpha = allocAlphas[ai];
    if (searchMatchSet !== null) {
      alpha = searchMatchSet.has(ai) ? 0.9 : 0.06;
    }

    const key = allocColors[ai] + alpha;
    if (!batches[key]) batches[key] = { color: allocColors[ai], alpha, indices: [] };
    batches[key].indices.push(ai);
  }

  for (const batch of Object.values(batches)) {
    ctx.beginPath();
    for (const ai of batch.indices) tracePoly(ai, newX);
    ctx.globalAlpha = batch.alpha;
    ctx.fillStyle = batch.color;
    ctx.fill();
    ctx.globalAlpha = Math.min(batch.alpha, 0.3);
    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  if (hoveredIdx >= 0) {
    ctx.beginPath();
    tracePoly(hoveredIdx, newX);
    ctx.globalAlpha = 1.0;
    ctx.fillStyle = allocColors[hoveredIdx];
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  ctx.restore();

  const elapsed = performance.now() - t0;
  perfFrames++; perfSum += elapsed;
  const now = performance.now();
  if (now - perfLastUpdate > 500) {
    const avg = perfSum / perfFrames;
    perfEl.textContent = `draw: ${avg.toFixed(1)}ms (${perfFrames} frames)`;
    perfFrames = 0; perfSum = 0; perfLastUpdate = now;
  }
}

// Hit testing: bucket lookup instead of full scan
function hitTest(mx, my) {
  const newX = currentTransform.rescaleX(xScale);
  const dataX = newX.invert(mx - margin.left);
  const dataY = yScale.invert(my - margin.top);

  const bi = Math.max(0, Math.min(NUM_HIT_BUCKETS, Math.floor(dataX / hitBucketSize)));
  const candidates = hitBuckets[bi];

  let best = null;
  let bestSize = Infinity;

  for (const ai of candidates) {
    if (dataX < allocStarts[ai] || dataX > allocEnds[ai]) continue;

    const d = ALLOCS[ai];
    let offset = d.offsets[0];
    for (let i = 1; i < d.ts.length; i++) {
      if (d.ts[i] > dataX) break;
      offset = d.offsets[i];
    }

    if (dataY >= offset && dataY <= offset + d.s && d.s < bestSize) {
      best = d;
      bestSize = d.s;
    }
  }
  return best;
}

let yMode = 'fixed';
let customYDomain = null;

function getBaseYDomain(d0, d1) {
  const fixedPeak = Math.max(
    1,
    META.high_water_mark_bytes,
    showReserved ? META.reserved_high_water_mark_bytes : 0,
  );
  if (yMode !== 'autofit' && !dimPersistent) return [0, fixedPeak * 1.05];

  let minY = Infinity, maxY = 0;
  for (let ai = 0; ai < ALLOCS.length; ai++) {
    if (allocEnds[ai] < d0 || allocStarts[ai] > d1) continue;
    if (dimPersistent && allocPersistent[ai]) continue;
    const d = ALLOCS[ai];
    let offset = d.offsets[0];
    for (let i = 0; i < d.ts.length; i++) {
      if (d.ts[i] <= d0) offset = d.offsets[i];
      if (d.ts[i] >= d0 && d.ts[i] <= d1) {
        minY = Math.min(minY, d.offsets[i]);
        maxY = Math.max(maxY, d.offsets[i] + d.s);
      }
    }
    minY = Math.min(minY, offset);
    maxY = Math.max(maxY, offset + d.s);
  }
  if (showReserved) {
    const start = Math.max(0, Math.floor(d0));
    const end = Math.min(RESERVED_TIMELINE.length, Math.ceil(d1) + 1);
    for (let timestep = start; timestep < end; timestep++) {
      maxY = Math.max(maxY, RESERVED_TIMELINE[timestep]);
    }
  }
  if (maxY === 0) { minY = 0; maxY = fixedPeak; }
  if (minY === Infinity) minY = 0;
  const pad = Math.max(1, (maxY - minY) * 0.05);
  return [Math.max(0, minY - pad), maxY + pad];
}

function updateChart(transform) {
  currentTransform = transform;
  const newX = transform.rescaleX(xScale);
  const [d0, d1] = newX.domain();

  yScale.domain(customYDomain || getBaseYDomain(d0, d1));

  xAxisG.call(xAxis.scale(newX));
  xAxisG.selectAll('text').attr('fill', 'var(--text-muted)');
  xAxisG.selectAll('line, path').attr('stroke', 'var(--border)');

  yAxisG.call(yAxisFn);
  yAxisG.selectAll('text').attr('fill', 'var(--text-muted)');
  yAxisG.selectAll('line, path').attr('stroke', 'var(--border)');

  gridG.call(d3.axisLeft(yScale).ticks(8).tickSize(-width).tickFormat(''));
  gridG.selectAll('line').attr('stroke', 'var(--grid)');
  gridG.selectAll('path').attr('stroke', 'none');

  const hwmY = yScale(META.high_water_mark_bytes);
  hwmG.select('.hwm-line').attr('x2', width).attr('y1', hwmY).attr('y2', hwmY);
  hwmG.select('.hwm-label').attr('x', width - 4).attr('y', hwmY - 6);

  if (showReserved) {
    reservedPath
      .style('display', null)
      .datum(RESERVED_TIMELINE)
      .attr('d', d3.line().x((d, i) => newX(i)).y(d => yScale(d)));
    poolLineG.selectAll('path').data(PRIVATE_POOLS).join('path')
      .attr('class', 'pool-reserved-line')
      .attr('d', pool => d3.line()
        .x(point => newX(point.step))
        .y(point => yScale(point.reserved))(pool.timeline));
  } else {
    reservedPath.style('display', 'none').attr('d', null);
    poolLineG.selectAll('path').remove();
  }
  markerG.selectAll('line').data(markerData).join('line')
    .attr('class', event => `event-marker ${event.act}`)
    .attr('x1', event => newX(event.step)).attr('x2', event => newX(event.step))
    .attr('y1', 0).attr('y2', height);

  drawCanvas();
  for (const hook of chartUpdateHooks) hook();
}

function transformForDomain(d0, d1) {
  const range = Math.max(1e-9, d1 - d0);
  return d3.zoomIdentity.translate(-d0 * width / range, 0).scale(MAX_TS / range);
}

const chartUpdateHooks = [];

const zoom = d3.zoom()
  .scaleExtent([1, 2000])
  .filter(event => !event.shiftKey)
  .translateExtent([[0, 0], [width, height]])
  .extent([[0, 0], [width, height]])
  .on('zoom', (event) => updateChart(event.transform));

const zoomRect = chartArea.append('rect')
  .attr('width', width).attr('height', height)
  .attr('fill', 'none').attr('pointer-events', 'all')
  .call(zoom);

// Box zoom: shift+drag to select a region
let boxStart = null;
const boxRect = chartArea.append('rect')
  .attr('fill', 'rgba(62, 147, 204, 0.15)')
  .attr('stroke', 'var(--accent)')
  .attr('stroke-width', 1)
  .attr('stroke-dasharray', '4 2')
  .style('display', 'none')
  .attr('pointer-events', 'none');

svg.node().addEventListener('pointerdown', function(event) {
  if (!event.shiftKey || event.button !== 0) return;
  event.preventDefault();
  const [mx, my] = d3.pointer(event, g.node());
  boxStart = { x: Math.max(0, Math.min(width, mx)), y: Math.max(0, Math.min(height, my)) };
  boxRect.style('display', null).attr('width', 0).attr('height', 0);
  svg.node().setPointerCapture(event.pointerId);
});

svg.node().addEventListener('pointermove', function(event) {
  if (!boxStart) return;
  const [mx, my] = d3.pointer(event, g.node());
  const cx = Math.max(0, Math.min(width, mx));
  const cy = Math.max(0, Math.min(height, my));
  boxRect
    .attr('x', Math.min(boxStart.x, cx))
    .attr('y', Math.min(boxStart.y, cy))
    .attr('width', Math.abs(cx - boxStart.x))
    .attr('height', Math.abs(cy - boxStart.y));
});

svg.node().addEventListener('pointerup', function(event) {
  if (!boxStart) return;
  const [mx, my] = d3.pointer(event, g.node());
  const x0 = Math.max(0, Math.min(boxStart.x, mx));
  const x1 = Math.min(width, Math.max(boxStart.x, mx));
  const y0 = Math.max(0, Math.min(boxStart.y, my));
  const y1 = Math.min(height, Math.max(boxStart.y, my));
  boxStart = null;
  boxRect.style('display', 'none');

  if (x1 - x0 < 5 || y1 - y0 < 5) return;

  const newX = currentTransform.rescaleX(xScale);
  const dataX0 = newX.invert(x0);
  const dataX1 = newX.invert(x1);
  customYDomain = [yScale.invert(y1), yScale.invert(y0)];
  zoomRect.transition().duration(300).call(zoom.transform, transformForDomain(dataX0, dataX1));
});

zoomRect.on('mousemove', function(event) {
  const [mx, my] = d3.pointer(event, svg.node());
  const hit = hitTest(mx, my);
  if (hit !== hoveredAlloc) {
    hoveredAlloc = hit;
    drawCanvas();
  }
  if (hit) {
    const info = tooltipFrameInfo(hit.si);
    const primary = info.userFrame || info.apiFrame;
    const secondary = info.userFrame && info.apiFrame ? info.apiFrame : null;
    const lines = [`<div class="tt-row"><span class="tt-label">Size:</span><span class="tt-value">${formatBytes(hit.s)}</span></div>`];
    lines.push(`<div class="tt-row"><span class="tt-label">Pool:</span><span class="tt-value">${escapeHtml(formatPool(hit.pool))}</span></div>`);
    if (primary) lines.push(`<div class="tt-${info.userFrame ? 'user' : 'api'}">${escapeHtml(primary)}</div>`);
    if (secondary) lines.push(`<div class="tt-api">${escapeHtml(secondary)}</div>`);
    showTooltip(event, lines.join(''));
  } else {
    hideTooltip();
  }
});

zoomRect.on('mouseleave', function() {
  if (hoveredAlloc) { hoveredAlloc = null; drawCanvas(); }
  hideTooltip();
});

zoomRect.on('click', function(event) {
  const [mx, my] = d3.pointer(event, svg.node());
  const hit = hitTest(mx, my);
  if (hit) {
    selectAlloc(hit);
    handleAllocationSelection();
  }
});

zoomRect.on('dblclick.zoom', null);
zoomRect.on('dblclick', function() {
  customYDomain = null;
  zoomRect.transition().duration(300).call(zoom.transform, d3.zoomIdentity);
});

drawCanvas();

// WASD / arrow key navigation (Perfetto-style, smooth)
const activeKeys = new Set();
let animating = false;
const SPEEDS = [
  { pan: 0.005, zoom: 1.01 },
  { pan: 0.01,  zoom: 1.025 },
  { pan: 0.02,  zoom: 1.05 },
  { pan: 0.04,  zoom: 1.08 },
  { pan: 0.08,  zoom: 1.12 },
];
let speedIdx = 2;

function navTarget() {
  let t = currentTransform;
  const panPx = width * SPEEDS[speedIdx].pan;
  const zoomFactor = SPEEDS[speedIdx].zoom;

  if (activeKeys.has('a') || activeKeys.has('arrowleft'))
    t = t.translate(panPx / t.k, 0);
  if (activeKeys.has('d') || activeKeys.has('arrowright'))
    t = t.translate(-panPx / t.k, 0);
  if (activeKeys.has('w') || activeKeys.has('arrowup')) {
    const cx = (width / 2 - t.x) / t.k;
    t = t.translate(cx, 0).scale(zoomFactor).translate(-cx, 0);
  }
  if (activeKeys.has('s') || activeKeys.has('arrowdown')) {
    const cx = (width / 2 - t.x) / t.k;
    t = t.translate(cx, 0).scale(1 / zoomFactor).translate(-cx, 0);
  }
  const [domainStart, domainEnd] = t.rescaleX(xScale).domain();
  const range = Math.min(MAX_TS, Math.max(MAX_TS / 2000, domainEnd - domainStart));
  const start = Math.max(0, Math.min(MAX_TS - range, domainStart));
  return transformForDomain(start, start + range);
}

function navLoop() {
  if (activeKeys.size === 0) { animating = false; return; }
  zoomRect.call(zoom.transform, navTarget());
  requestAnimationFrame(navLoop);
}

const shortcutBar = document.getElementById('shortcut-bar');
const speedIndicator = document.getElementById('speed-indicator');

document.addEventListener('keydown', function(event) {
  if (event.target.tagName === 'INPUT') return;
  const k = event.key.toLowerCase();

  if (k === '?') {
    shortcutBar.style.display = shortcutBar.style.display === 'none' ? 'flex' : 'none';
    return;
  }
  if (k === '[' || k === ']') {
    speedIdx = Math.max(0, Math.min(SPEEDS.length - 1, speedIdx + (k === ']' ? 1 : -1)));
    speedIndicator.textContent = speedIdx + 1;
    shortcutBar.style.display = 'flex';
    return;
  }

  if (!['a','d','w','s','arrowleft','arrowright','arrowup','arrowdown'].includes(k)) return;
  event.preventDefault();
  activeKeys.add(k);
  if (!animating) { animating = true; navLoop(); }
});

document.addEventListener('keyup', function(event) {
  activeKeys.delete(event.key.toLowerCase());
});

const controlsShell = document.getElementById('controls-shell');
const controls = document.getElementById('controls');
const controlsToggle = document.getElementById('controls-toggle');

function setControlsCollapsed(collapsed) {
  controls.classList.toggle('collapsed', collapsed);
  controlsToggle.textContent = collapsed ? '◀' : '▶';
  controlsToggle.title = collapsed ? 'Show controls' : 'Hide controls';
  controlsToggle.setAttribute('aria-expanded', String(!collapsed));
}

controlsToggle.addEventListener('click', event => {
  event.stopPropagation();
  setControlsCollapsed(!controls.classList.contains('collapsed'));
});
document.addEventListener('pointerdown', event => {
  if (!controlsShell.contains(event.target)) setControlsCollapsed(true);
});

const settingsTrigger = document.getElementById('settings-trigger');
settingsTrigger.addEventListener('click', function(e) {
  if (e.target.closest('#settings-dropdown')) return;
  this.classList.toggle('open');
});

document.getElementById('hwm-toggle').onchange = function() {
  hwmG.style('display', this.checked ? null : 'none');
};

document.getElementById('reserved-toggle').onchange = function() {
  showReserved = this.checked;
  customYDomain = null;
  updateChart(currentTransform);
};

document.getElementById('autofit-toggle').onchange = function() {
  yMode = this.checked ? 'autofit' : 'fixed';
  customYDomain = null;
  updateChart(currentTransform);
};

document.getElementById('dim-persistent-toggle').onchange = function() {
  dimPersistent = this.checked;
  customYDomain = null;
  updateChart(currentTransform);
};

document.getElementById('color-mode').onchange = function() {
  colorMode = this.value;
  recolorAllocs();
  drawCanvas();
};

// --- Feature 1: Search & Filter ---
const searchInput = document.getElementById('search-input');
const regexToggle = document.getElementById('regex-toggle');
let useRegex = false;

regexToggle.addEventListener('click', () => {
  useRegex = !useRegex;
  regexToggle.classList.toggle('active', useRegex);
  applySearch(searchInput.value);
});

function applySearch(query) {
  searchInput.value = query;
  if (!query) {
    searchMatcher = null;
  } else if (useRegex) {
    try { searchMatcher = new RegExp(query, 'i'); } catch(e) { searchMatcher = null; }
  } else {
    const q = query.toLowerCase();
    searchMatcher = { test: (s) => s.toLowerCase().includes(q) };
  }
  updateSearchCache();
  drawCanvas();
}

searchInput.addEventListener('input', (e) => applySearch(e.target.value));

document.addEventListener('keydown', function(event) {
  if (event.key === '/' && event.target.tagName !== 'INPUT') {
    event.preventDefault();
    searchInput.focus();
  }
  if (event.key === 'Escape' && event.target === searchInput) {
    searchInput.value = '';
    applySearch('');
    searchInput.blur();
  }
});

// --- Allocation Details ---
function showDetails() {
  const d = uiState.selectedAlloc;
  if (!d) {
    detailBody.innerHTML = '<div class="empty-detail">Click an allocation to see its details</div>';
    return;
  }
  const ts = d.time_us === null || d.time_us === undefined
    ? 'N/A'
    : new Date(d.time_us / 1000).toLocaleString();
  const lifetimeEnd = d.ts[d.ts.length - 1];
  detailStats.textContent = formatBytes(d.s);
  detailBody.innerHTML = `<div class="alloc-details"><table>
    <tr><td>Requested</td><td>${formatBytes(d.s)} (${d.s.toLocaleString()} bytes)</td></tr>
    <tr><td>Block size</td><td>${formatBytes(d.block_size)}</td></tr>
    <tr><td>Address</td><td>${escapeHtml(d.addr || 'N/A')}</td></tr>
    <tr><td>Stream</td><td>${escapeHtml(d.stream ?? 'N/A')}</td></tr>
    <tr><td>Pool</td><td>${escapeHtml(formatPool(d.pool))}</td></tr>
    <tr><td>Origin</td><td>${escapeHtml(d.origin)}${d.ghost ? ' (reconstructed)' : ''}</td></tr>
    <tr><td>Category</td><td>${escapeHtml(d.category ?? 'unknown')}</td></tr>
    <tr><td>Timestamp</td><td>${escapeHtml(ts)}</td></tr>
    <tr><td>Compile ctx</td><td>${escapeHtml(d.ctx || 'None')}</td></tr>
    <tr><td>Metadata</td><td><pre>${escapeHtml(JSON.stringify(d.metadata || {}, null, 2))}</pre></td></tr>
    <tr><td>Annotations</td><td><pre>${escapeHtml(JSON.stringify(d.annotations || [], null, 2))}</pre></td></tr>
    <tr><td>FX</td><td><pre>${escapeHtml(JSON.stringify(d.fx || [], null, 2))}</pre></td></tr>
    <tr><td>Lifetime</td><td>ts ${d.ts[0]} \u2192 ${lifetimeEnd}${lifetimeEnd >= META.max_timestep ? ' (never freed)' : ''}</td></tr>
  </table></div>`;
}

// --- Feature 2: What's at Peak ---
function showPeakBreakdown() {
  const alive = derivedData.peakAllocIndices;
  const total = derivedData.peakTotalBytes;
  const maxAliveSize = alive[0] === undefined ? 1 : ALLOCS[alive[0]].s;

  detailStats.textContent = `${alive.length} allocs, ${formatBytes(total)}`;

  let html = `<div class="peak-label">Allocations alive at peak (${formatBytes(META.high_water_mark_bytes)})</div>`;
  html += alive.map(ai => {
    const d = ALLOCS[ai];
    const pct = (d.s / Math.max(1, META.high_water_mark_bytes) * 100).toFixed(1);
    const barW = (d.s / maxAliveSize * 100).toFixed(0);
    return `<div class="breakdown-row" data-action="show-stack" data-stack-idx="${d.si}" data-label="${encodeURIComponent(formatBytes(d.s))}">
      <span class="bd-size">${formatBytes(d.s)}</span>
      <span class="bd-pct">${pct}%</span>
      <span class="bd-bar"><span class="bd-bar-fill" style="width:${barW}%"></span></span>
      <span class="bd-frame">${escapeHtml(derivedData.stackFrameLabels[d.si])}</span>
    </div>`;
  }).join('');
  detailBody.innerHTML = html;
}

hwmG.on('click', function() {
  activateDetailView('peak');
});

// --- Feature 5: Leak Detection (never-freed allocations) ---
function showLeaks() {
  const candidates = derivedData.leakAllocIndices;

  if (candidates.length === 0) {
    detailStats.textContent = 'No potential leaks';
    detailBody.innerHTML = '<div class="empty-detail">No potential memory leaks detected.<br>All allocations born after the setup phase were freed.</div>';
    searchMatchSet = null;
    drawCanvas();
    return;
  }

  const groups = derivedData.leakGroups;
  const maxBytes = groups[0]?.totalBytes || 1;

  detailStats.textContent = `${candidates.length} allocs, ${formatBytes(derivedData.leakTotalBytes)}`;

  searchMatchSet = new Set(candidates);
  drawCanvas();

  let html = '<div class="peak-label">Never-freed allocations (excluding setup phase)</div>';
  html += groups.map(g => {
    const pct = (g.totalBytes / derivedData.leakTotalBytes * 100).toFixed(1);
    const barW = (g.totalBytes / maxBytes * 100).toFixed(0);
    return `<div class="breakdown-row" data-action="apply-search" data-query="${encodeURIComponent(g.frame)}">
      <span class="bd-size">${formatBytes(g.totalBytes)}</span>
      <span class="bd-count">\u00d7${g.count}</span>
      <span class="bd-pct">${pct}%</span>
      <span class="bd-bar"><span class="bd-bar-fill leak-bar" style="width:${barW}%"></span></span>
      <span class="bd-frame">${escapeHtml(g.frame)}</span>
    </div>`;
  }).join('');

  detailBody.innerHTML = html;
}

function showPools() {
  detailStats.textContent = `${PRIVATE_POOLS.length} private pools`;
  if (!PRIVATE_POOLS.length) {
    detailBody.innerHTML = '<div class="empty-detail">No live CUDA Graph or MemPool segments were present in this snapshot.</div>';
    return;
  }
  const maxReserved = PRIVATE_POOLS.reduce(
    (maximum, pool) => Math.max(maximum, pool.reserved_bytes),
    1,
  );
  detailBody.innerHTML = '<div class="peak-label">Private pool reserved memory</div>' + PRIVATE_POOLS.map(pool => {
    const width = (pool.reserved_bytes / maxReserved * 100).toFixed(0);
    return `<div class="breakdown-row">
      <span class="bd-size">${formatBytes(pool.reserved_bytes)}</span>
      <span class="bd-count">${pool.num_segments} seg</span>
      <span class="bd-pct">${formatBytes(pool.active_bytes)}</span>
      <span class="bd-bar"><span class="bd-bar-fill" style="width:${width}%"></span></span>
      <span class="bd-frame">pool ${escapeHtml(formatPool(pool.id))}, stream ${escapeHtml(pool.stream)}, peak ${formatBytes(pool.peak_reserved_bytes)}, inactive ${formatBytes(pool.inactive_bytes)}</span>
    </div>`;
  }).join('');
}

function showEvents() {
  const visibleEvents = EVENTS.slice(-2000).reverse();
  detailStats.textContent = `${EVENTS.length} events`;
  if (!visibleEvents.length) {
    detailBody.innerHTML = '<div class="empty-detail">No allocator history events were recorded.</div>';
    return;
  }
  const overflowWarning = ALLOCATOR_SETTINGS.trace_alloc_overflowed
    ? '<div class="peak-label" style="color:#e74c3c">Warning: allocator history overflowed; older events were overwritten.</div>'
    : '';
  detailBody.innerHTML = overflowWarning + '<div class="peak-label">Allocator state history (newest first)</div>' + visibleEvents.map(event => {
    const oom = event.act === 'oom' ? `, device free ${formatBytes(event.device_free ?? 0)}` : '';
    return `<div class="breakdown-row" data-action="show-event-stack" data-stack-idx="${event.si}" data-label="${encodeURIComponent(event.act)}">
      <span class="bd-size">${formatBytes(event.s)}</span>
      <span class="bd-count">t${event.step}</span>
      <span class="bd-pct">${formatBytes(event.a)}</span>
      <span class="bd-frame">${escapeHtml(event.act)} ${escapeHtml(event.addr ?? '')}, reserved ${formatBytes(event.r)}${oom}</span>
    </div>`;
  }).join('');
}

function showSegments() {
  detailStats.textContent = `${SEGMENTS.length} segments`;
  if (!SEGMENTS.length) {
    detailBody.innerHTML = '<div class="empty-detail">No current allocator segments.</div>';
    return;
  }
  detailBody.innerHTML = '<div class="peak-label">Current cached segment state</div>' + SEGMENTS.map(segment => {
    const activeBlocks = segment.blocks.filter(block => block.state !== 'inactive').length;
    return `<div class="breakdown-row">
      <span class="bd-size">${formatBytes(segment.total_size)}</span>
      <span class="bd-count">${activeBlocks}/${segment.blocks.length}</span>
      <span class="bd-pct">${formatBytes(segment.allocated_size)}</span>
      <span class="bd-frame">${escapeHtml(segment.address)}, pool ${escapeHtml(formatPool(segment.pool))}, stream ${escapeHtml(segment.stream)}, ${escapeHtml(segment.segment_type)}${segment.expandable ? ', expandable' : ''}</span>
    </div>`;
  }).join('');
}

function showSettings() {
  detailStats.textContent = 'allocator settings';
  detailBody.innerHTML = `<pre class="detail-json">${escapeHtml(JSON.stringify(ALLOCATOR_SETTINGS, null, 2))}</pre>`;
}

// --- Panel toggle & resize ---
const detailPanel = document.getElementById('detail-panel');
const panelToggle = document.getElementById('panel-toggle');
const resizeHandle = document.getElementById('resize-handle');

function setDetailPanelCollapsed(collapsed) {
  detailPanel.classList.toggle('collapsed', collapsed);
  panelToggle.textContent = collapsed ? '▶' : '◀';
  panelToggle.title = collapsed ? 'Show stack/details' : 'Hide stack/details';
  panelToggle.setAttribute('aria-expanded', String(!collapsed));
  setTimeout(resizeChart, 200);
}

panelToggle.addEventListener('click', () => {
  setDetailPanelCollapsed(!detailPanel.classList.contains('collapsed'));
});

let resizing = false;
resizeHandle.addEventListener('pointerdown', (e) => {
  resizing = true;
  resizeHandle.classList.add('dragging');
  resizeHandle.setPointerCapture(e.pointerId);
  e.preventDefault();
});
document.addEventListener('pointermove', (e) => {
  if (!resizing) return;
  const newW = Math.max(200, window.innerWidth - e.clientX);
  detailPanel.style.width = newW + 'px';
  detailPanel.style.minWidth = newW + 'px';
});
document.addEventListener('pointerup', () => {
  if (!resizing) return;
  resizing = false;
  resizeHandle.classList.remove('dragging');
  resizeChart();
});

// --- Detail panel tabs ---
const detailViews = [
  { id: 'stack', label: 'Stack', render: renderStackSelection, selectionViewId: 'stack' },
  { id: 'details', label: 'Details', render: showDetails, selectionViewId: 'details' },
  { id: 'peak', label: 'Peak', render: showPeakBreakdown, selectionViewId: 'stack' },
  { id: 'leaks', label: 'Leaks', render: showLeaks, selectionViewId: 'stack' },
  { id: 'pools', label: 'Pools', render: showPools, selectionViewId: 'stack' },
  { id: 'events', label: 'Events', render: showEvents, selectionViewId: 'stack' },
  { id: 'segments', label: 'Segments', render: showSegments, selectionViewId: 'stack' },
  { id: 'settings', label: 'Settings', render: showSettings, selectionViewId: 'stack' },
];
const detailViewById = Object.fromEntries(detailViews.map(view => [view.id, view]));

function updateActiveDetailTab() {
  detailTabs.querySelectorAll('.detail-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.tab === uiState.activeDetailView);
  });
}

function renderDetailTabs() {
  detailTabs.innerHTML = detailViews
    .map(view => `<button class="detail-tab" data-tab="${view.id}">${view.label}</button>`)
    .join('');
  updateActiveDetailTab();
}

function renderActiveDetailView() {
  detailViewById[uiState.activeDetailView].render();
}

function activateDetailView(viewId, { resetSearch = false } = {}) {
  uiState.activeDetailView = viewId;
  updateActiveDetailTab();
  if (resetSearch) {
    searchInput.value = '';
    applySearch('');
  } else if (viewId !== 'leaks' && searchMatcher === null && searchMatchSet !== null) {
    searchMatchSet = null;
    drawCanvas();
  }
  renderActiveDetailView();
}

function handleAllocationSelection() {
  setDetailPanelCollapsed(false);
  activateDetailView(detailViewById[uiState.activeDetailView].selectionViewId);
}

renderDetailTabs();

detailTabs.addEventListener('click', function(event) {
  const tab = event.target.closest('.detail-tab');
  if (!tab) return;
  activateDetailView(tab.dataset.tab, { resetSearch: tab.dataset.tab === 'stack' });
});

detailBody.addEventListener('click', function(event) {
  const row = event.target.closest('.breakdown-row');
  if (!row) return;
  if (row.dataset.action === 'show-stack' || row.dataset.action === 'show-event-stack') {
    selectStack(Number(row.dataset.stackIdx), decodeURIComponent(row.dataset.label));
    activateDetailView('stack');
    return;
  }
  if (row.dataset.action === 'apply-search') {
    if (useRegex) {
      useRegex = false;
      regexToggle.classList.remove('active');
    }
    applySearch(decodeURIComponent(row.dataset.query));
  }
});

// --- Feature 4: Minimap ---
const minimapContainer = document.getElementById('minimap');
let minimapRect = minimapContainer.getBoundingClientRect();
let minimapW = Math.max(1, minimapRect.width - 32);
const minimapH = 32;
const miniMargin = { left: 16, top: 4 };

const miniSvg = d3.select('#minimap').append('svg')
  .attr('viewBox', `0 0 ${minimapW + 32} ${minimapH + 8}`);

const miniG = miniSvg.append('g')
  .attr('transform', `translate(${miniMargin.left},${miniMargin.top})`);

const miniX = d3.scaleLinear().domain([0, MAX_TS]).range([0, minimapW]);
const miniY = d3.scaleLinear()
  .domain([0, Math.max(1, META.high_water_mark_bytes, META.reserved_high_water_mark_bytes) * 1.05])
  .range([minimapH, 0]);

const miniArea = miniG.append('path')
  .datum(TIMELINE)
  .attr('class', 'minimap-area');

function minimapAreaPath() {
  return d3.area()
    .x((d, i) => i * minimapW / Math.max(1, TIMELINE.length - 1))
    .y0(minimapH)
    .y1(d => miniY(d))(TIMELINE);
}
miniArea.attr('d', minimapAreaPath());

const viewportRect = miniG.append('rect')
  .attr('class', 'minimap-viewport')
  .attr('y', 0)
  .attr('height', minimapH);

function updateMinimap() {
  miniY.domain([
    0,
    Math.max(
      1,
      META.high_water_mark_bytes,
      showReserved ? META.reserved_high_water_mark_bytes : 0,
    ) * 1.05,
  ]);
  miniArea.attr('d', minimapAreaPath());
  const newX = currentTransform.rescaleX(xScale);
  const [d0, d1] = newX.domain();
  const x0 = miniX(Math.max(0, d0));
  const x1 = miniX(Math.min(MAX_TS, d1));
  viewportRect.attr('x', x0).attr('width', Math.max(2, x1 - x0));
}

updateMinimap();
chartUpdateHooks.push(updateMinimap);

const miniDrag = d3.drag()
  .on('drag', function(event) {
    const domainPerPx = MAX_TS / minimapW;
    const shift = event.dx * domainPerPx;
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    const range = d1 - d0;
    const newD0 = Math.max(0, Math.min(MAX_TS - range, d0 + shift));
    zoomRect.call(zoom.transform, transformForDomain(newD0, newD0 + range));
  });

viewportRect.call(miniDrag);

miniSvg.on('click', function(event) {
  const [mx] = d3.pointer(event, miniG.node());
  const clickTs = miniX.invert(mx);
  const newX = currentTransform.rescaleX(xScale);
  const [d0, d1] = newX.domain();
  const range = d1 - d0;
  const newD0 = Math.max(0, Math.min(MAX_TS - range, clickTs - range / 2));
  zoomRect.transition().duration(300).call(zoom.transform, transformForDomain(newD0, newD0 + range));
});

function resizeChart() {
  const [domainStart, domainEnd] = currentTransform.rescaleX(xScale).domain();
  containerRect = container.getBoundingClientRect();
  width = Math.max(1, containerRect.width - margin.left - margin.right);
  height = Math.max(1, containerRect.height - margin.top - margin.bottom);

  canvas.width = Math.max(1, containerRect.width * devicePixelRatio);
  canvas.height = Math.max(1, containerRect.height * devicePixelRatio);
  canvas.style.width = `${containerRect.width}px`;
  canvas.style.height = `${containerRect.height}px`;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

  svg.attr('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`);
  clipRect.attr('width', width).attr('height', height);
  xScale.range([0, width]);
  yScale.range([height, 0]);
  xAxisG.attr('transform', `translate(0,${height})`);
  zoom.translateExtent([[0, 0], [width, height]]).extent([[0, 0], [width, height]]);
  zoomRect.attr('width', width).attr('height', height);

  minimapRect = minimapContainer.getBoundingClientRect();
  minimapW = Math.max(1, minimapRect.width - 32);
  miniSvg.attr('viewBox', `0 0 ${minimapW + 32} ${minimapH + 8}`);
  miniX.range([0, minimapW]);
  miniArea.attr('d', minimapAreaPath());

  const start = Math.max(0, Math.min(MAX_TS, domainStart));
  const end = Math.max(start + 1e-9, Math.min(MAX_TS, domainEnd));
  zoomRect.call(zoom.transform, transformForDomain(start, end));
}

window.addEventListener('resize', resizeChart);
</script>
</body>
</html>
"""


_MEMORY_COMPARISON_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__DOCUMENT_TITLE__</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');
  :root {
    --bg: #0E0E0E;
    --surface: #1a1a1a;
    --border: rgba(255, 255, 255, 0.10);
    --text: rgba(255, 255, 255, 0.92);
    --text-muted: rgba(255, 255, 255, 0.50);
    --accent: #3E93CC;
    --accent-light: rgba(62, 147, 204, 0.12);
    --accent-stroke: rgba(62, 147, 204, 0.7);
    --hwm-color: rgba(255, 255, 255, 0.60);
    --grid: rgba(255, 255, 255, 0.03);
    --tooltip-bg: #1f1f1f;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'IBM Plex Mono', 'Fira Mono', monospace;
    --left-accent: #3E93CC;
    --right-accent: #C97049;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  #header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    position: relative;
    z-index: 60;
  }

  #header h1 { font-size: 14px; font-weight: 500; font-family: var(--mono); letter-spacing: 0.03em; text-transform: uppercase; flex-shrink: 0; }
  #header h1 .vs { color: var(--text-muted); margin: 0 8px; }
  #header h1 .title-left { color: var(--left-accent); }
  #header h1 .title-right { color: var(--right-accent); }

  #header-mid {
    display: flex;
    gap: 12px;
    align-items: center;
    flex: 1;
    justify-content: center;
  }

  #help-dropdown {
    display: none;
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    margin-top: 6px;
    background: var(--tooltip-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 14px;
    white-space: nowrap;
    z-index: 50;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    font-family: var(--mono);
    font-size: 11px;
    line-height: 2;
    color: var(--text-muted);
  }

  #help-dropdown kbd {
    display: inline-block;
    padding: 1px 5px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text);
    min-width: 18px;
    text-align: center;
  }

  #help-trigger:hover #help-dropdown { display: block; }

  #settings-trigger {
    cursor: pointer;
    position: relative;
    font-size: 14px;
    opacity: 0.6;
    transition: opacity 0.15s;
    user-select: none;
  }
  #settings-trigger:hover { opacity: 1; }
  #settings-dropdown {
    display: none;
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 6px;
    background: var(--tooltip-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    white-space: nowrap;
    z-index: 50;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
  }
  #settings-trigger.open #settings-dropdown { display: block; }
  #settings-dropdown label { display: flex; align-items: center; gap: 6px; }
  #settings-dropdown select {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 2px 4px;
    font-size: 11px;
    font-family: var(--mono);
    cursor: pointer;
  }

  #controls-shell {
    position: relative;
    flex-shrink: 0;
    margin-left: auto;
  }

  #controls-toggle {
    width: 26px;
    height: 26px;
    display: grid;
    place-items: center;
    background: rgba(255,255,255,0.04);
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
  }
  #controls-toggle:hover { color: var(--text); background: rgba(255,255,255,0.08); }

  #controls {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    width: max-content;
    max-width: min(900px, calc(100vw - 24px));
    padding: 9px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 5px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.55);
    z-index: 80;
  }
  #controls.collapsed { display: none; }

  @media (max-width: 1180px) {
    #header-mid { display: none; }
    #header { padding: 10px 14px; }
  }

  #trace-toggle-group {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .trace-toggle-btn {
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: rgba(255,255,255,0.04);
    color: var(--text-muted);
    font-family: var(--mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: background 0.15s, color 0.15s, border-color 0.15s, opacity 0.15s;
  }
  .trace-toggle-btn:hover { color: var(--text); }
  .trace-toggle-btn.active-left {
    background: rgba(62, 147, 204, 0.14);
    color: var(--left-accent);
    border-color: rgba(62, 147, 204, 0.55);
  }
  .trace-toggle-btn.active-right {
    background: rgba(201, 112, 73, 0.14);
    color: var(--right-accent);
    border-color: rgba(201, 112, 73, 0.55);
  }
  .trace-toggle-btn.inactive {
    opacity: 0.5;
  }

  .toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
  }

  .toggle input[type="checkbox"] {
    accent-color: var(--accent);
    width: 14px;
    height: 14px;
  }

  .toggle:hover { color: var(--text); }

  .stat {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--text-muted);
    padding: 4px 10px;
    background: rgba(255,255,255,0.04);
    border-radius: 3px;
    border: 1px solid var(--border);
  }
  .stat strong { color: var(--text); font-weight: 500; }
  .stat.stat-left strong { color: var(--left-accent); }
  .stat.stat-right strong { color: var(--right-accent); }

  #main {
    display: flex;
    flex: 1;
    min-height: 0;
  }

  #charts-wrapper {
    display: flex;
    flex: 1;
    min-width: 0;
    position: relative;
  }

  .chart-pane {
    flex: 1;
    min-width: 0;
    position: relative;
    overflow: hidden;
  }

  .chart-pane .pane-label {
    position: absolute;
    top: 8px;
    left: 12px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    z-index: 5;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(0,0,0,0.5);
  }
  .chart-pane.pane-left .pane-label { color: var(--left-accent); }
  .chart-pane.pane-right .pane-label { color: var(--right-accent); }

  .chart-pane canvas {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
  }

  .chart-pane > svg {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 1;
  }

  #detail-panel {
    width: 420px;
    border-left: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
    position: relative;
    transition: width 0.15s, min-width 0.15s;
    min-width: 420px;
  }

  #detail-panel.collapsed {
    width: 0 !important;
    min-width: 0 !important;
    border-left: none;
    overflow: hidden;
  }

  #panel-toggle {
    width: 22px;
    height: 48px;
    align-self: center;
    flex-shrink: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px 0 0 4px;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
  }
  #panel-toggle:hover { color: var(--text); background: rgba(255,255,255,0.06); }

  #resize-handle {
    position: absolute;
    left: 0;
    top: 0;
    width: 4px;
    height: 100%;
    cursor: col-resize;
    z-index: 11;
  }
  #resize-handle:hover, #resize-handle.dragging { background: var(--accent); }

  #detail-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    font-weight: 500;
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }

  #detail-header .detail-stats {
    font-weight: 400;
    color: var(--text-muted);
    font-size: 12px;
  }

  #detail-header .detail-actions {
    display: flex;
    gap: 4px;
    align-items: center;
  }

  #detail-body {
    flex: 1;
    overflow-y: auto;
    padding: 0;
  }

  #detail-body::-webkit-scrollbar { width: 6px; }
  #detail-body::-webkit-scrollbar-track { background: transparent; }
  #detail-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .stack-frame {
    padding: 3px 16px;
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.5;
    cursor: pointer;
    overflow: hidden;
    border-left: 2px solid transparent;
  }

  .stack-frame .frame-text {
    white-space: pre-wrap;
    word-break: break-all;
    display: block;
  }

  .stack-frame:hover { background: rgba(255,255,255,0.04); }

  .stack-frame.frame-user {
    color: var(--text);
    border-left-color: #49C963;
    background: rgba(73, 201, 99, 0.04);
  }
  .stack-frame.frame-user .frame-func { color: #49C963; font-weight: 500; }
  .stack-frame.frame-user .frame-basename { color: var(--text); }
  .stack-frame.frame-user .frame-file { color: rgba(255,255,255,0.4); }

  .stack-frame.frame-library {
    color: rgba(255,255,255,0.6);
    border-left-color: #3E93CC;
  }
  .stack-frame.frame-library .frame-func { color: #5BA8D9; }
  .stack-frame.frame-library .frame-basename { color: rgba(255,255,255,0.7); }
  .stack-frame.frame-library .frame-file { color: rgba(255,255,255,0.25); }

  .stack-frame.frame-native {
    color: rgba(255,255,255,0.35);
  }
  .stack-frame.frame-native .frame-cpp { color: rgba(189, 147, 249, 0.5); }
  .stack-frame.frame-native .frame-basename { color: rgba(255,255,255,0.5); }

  .stack-frame.frame-noise {
    color: rgba(255,255,255,0.18);
    font-size: 10px;
  }

  .frame-noise { display: none; }

  .alloc-details {
    padding: 12px 16px;
    font-family: var(--mono);
    font-size: 12px;
  }

  .alloc-details table {
    width: 100%;
    border-collapse: collapse;
  }

  .alloc-details td {
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    vertical-align: top;
  }

  .alloc-details td:first-child {
    color: var(--text-muted);
    width: 100px;
    padding-right: 12px;
  }

  .alloc-details td:last-child {
    color: var(--text);
    word-break: break-all;
  }

  .empty-detail {
    padding: 24px 16px;
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
  }

  .axis text { fill: var(--text-muted); font-size: 11px; font-family: var(--font); }
  .axis line, .axis path { stroke: var(--border); }
  .grid line { stroke: var(--grid); }
  .grid path { stroke: none; }

  .hwm-line { stroke: var(--hwm-color); stroke-width: 0.75; stroke-dasharray: 8 4; }
  .hwm-label { fill: var(--hwm-color); font-size: 11px; font-family: var(--mono); font-weight: 500; letter-spacing: 0.02em; }
  .reserved-line { fill: none; stroke: #C97049; stroke-width: 1.25; stroke-dasharray: 4 3; }
  .pool-reserved-line { fill: none; stroke: rgba(255,255,255,0.45); stroke-width: 1; stroke-dasharray: 2 2; }
  .event-marker { stroke-width: 1; stroke-dasharray: 2 3; opacity: 0.8; }
  .event-marker.oom { stroke: #e74c3c; }
  .event-marker.snapshot { stroke: #f1c40f; }
  .detail-json { padding: 12px 16px; white-space: pre-wrap; word-break: break-word; font: 11px/1.5 var(--mono); color: var(--text-muted); }

  #search-input {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 3px 0 0 3px;
    padding: 4px 10px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    width: 180px;
    outline: none;
  }

  #search-input:focus { border-color: var(--accent); }
  #search-input::placeholder { color: rgba(255,255,255,0.25); }

  #regex-toggle {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-left: none;
    border-radius: 0 3px 3px 0;
    padding: 4px 8px;
    color: var(--text-muted);
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    height: 100%;
  }
  #regex-toggle:hover { color: var(--text); }
  #regex-toggle.active { background: var(--accent); color: white; border-color: var(--accent); }

  #minimaps {
    display: flex;
    height: 40px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }

  .minimap-pane {
    flex: 1;
    padding: 0 8px;
    min-width: 0;
  }
  .minimap-pane + .minimap-pane { border-left: 1px solid var(--border); }
  .minimap-pane svg { width: 100%; height: 100%; }

  .minimap-area { fill: rgba(62, 147, 204, 0.3); }
  .minimap-viewport {
    fill: rgba(255,255,255,0.06);
    stroke: rgba(255,255,255,0.3);
    stroke-width: 1;
    cursor: grab;
  }
  .minimap-viewport:active { cursor: grabbing; }

  .detail-tabs {
    display: flex;
    gap: 0;
  }

  .detail-tab {
    padding: 4px 12px;
    font-size: 10px;
    font-family: var(--mono);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
    cursor: pointer;
  }

  .detail-tab:first-child { border-radius: 3px 0 0 3px; }
  .detail-tab:last-child { border-radius: 0 3px 3px 0; }
  .detail-tab + .detail-tab { border-left: none; }
  .detail-tab.active { background: var(--accent); color: white; border-color: var(--accent); }
  .detail-tabs { flex-wrap: wrap; }

  .lr-toggle {
    display: flex;
    gap: 0;
    margin-left: 8px;
  }
  .lr-btn {
    padding: 3px 10px;
    font-size: 10px;
    font-family: var(--mono);
    font-weight: 600;
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
    cursor: pointer;
    text-transform: uppercase;
  }
  .lr-btn:first-child { border-radius: 3px 0 0 3px; }
  .lr-btn:last-child { border-radius: 0 3px 3px 0; border-left: none; }
  .lr-btn.active-left { background: var(--left-accent); color: white; border-color: var(--left-accent); }
  .lr-btn.active-right { background: var(--right-accent); color: white; border-color: var(--right-accent); }

  .breakdown-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 16px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    border-bottom: 1px solid rgba(255,255,255,0.03);
    cursor: pointer;
  }

  .breakdown-row:hover { background: rgba(255,255,255,0.03); color: var(--text); }

  .breakdown-row .bd-size {
    min-width: 70px;
    text-align: right;
    color: var(--text);
    font-weight: 500;
  }

  .breakdown-row .bd-count {
    min-width: 30px;
    text-align: right;
    color: rgba(255,255,255,0.3);
    font-size: 10px;
  }

  .breakdown-row .bd-pct {
    min-width: 40px;
    text-align: right;
    color: var(--accent);
    font-size: 10px;
  }

  .breakdown-row .bd-bar {
    width: 60px;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
    flex-shrink: 0;
  }

  .breakdown-row .bd-bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
  }

  .breakdown-row .bd-bar-fill.leak-bar { background: #e74c3c; }

  .breakdown-row .bd-frame {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .peak-label {
    padding: 8px 16px;
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    background: rgba(255,255,255,0.02);
  }

  #tooltip {
    position: fixed; display: none;
    background: #1f1f1f; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px; padding: 10px 14px;
    font-size: 12px; line-height: 1.6;
    pointer-events: none; z-index: 100; max-width: 500px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    font-family: var(--mono);
  }

  #tooltip .tt-label { color: var(--text-muted); margin-right: 4px; }
  #tooltip .tt-value { color: var(--text); font-weight: 500; font-family: var(--mono); }
  #tooltip .tt-row { white-space: nowrap; }
  #tooltip .tt-api { color: #78BBE3; font-size: 11px; font-weight: 500; }
  #tooltip .tt-user { color: #49C963; font-size: 10px; }

  #shortcut-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 6px 24px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    font-size: 11px;
    color: var(--text-muted);
    flex-shrink: 0;
  }

  #shortcut-bar kbd {
    display: inline-block;
    padding: 1px 5px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text);
    min-width: 18px;
    text-align: center;
  }

  #shortcut-bar .sep {
    width: 1px;
    height: 14px;
    background: var(--border);
  }

  #speed-indicator {
    color: var(--accent);
    font-weight: 600;
  }
</style>
</head>
<body>
<div id="header">
  <h1><span class="title-left">__TITLE_LEFT__</span><span class="vs">vs</span><span class="title-right">__TITLE_RIGHT__</span></h1>
  <div id="header-mid">
    <span class="stat stat-left">L <strong id="peak-stat-left"></strong> peak · <strong id="allocs-stat-left"></strong> allocs</span>
    <span class="stat stat-right">R <strong id="peak-stat-right"></strong> peak · <strong id="allocs-stat-right"></strong> allocs</span>
    <span id="help-trigger" class="stat" style="cursor:help;position:relative;">
      ?
      <div id="help-dropdown">
        <div><kbd>scroll</kbd> zoom X</div>
        <div><kbd>drag</kbd> pan X</div>
        <div><kbd>shift+drag</kbd> box zoom (X+Y)</div>
        <div><kbd>dbl-click</kbd> reset view</div>
        <div><kbd>click</kbd> inspect allocation stack</div>
        <div><kbd>A</kbd><kbd>D</kbd> pan &nbsp; <kbd>W</kbd><kbd>S</kbd> zoom</div>
        <div><kbd>[</kbd><kbd>]</kbd> change speed</div>
        <div><kbd>/</kbd> search &nbsp; <kbd>esc</kbd> clear</div>
      </div>
    </span>
  </div>
  <div id="controls-shell">
    <button id="controls-toggle" title="Show controls" aria-expanded="false">&#9664;</button>
    <div id="controls" class="collapsed">
    <div id="trace-toggle-group">
      <button class="trace-toggle-btn active-left" data-trace-side="left">__TITLE_LEFT__</button>
      <button class="trace-toggle-btn active-right" data-trace-side="right">__TITLE_RIGHT__</button>
    </div>
    <div style="display:flex;align-items:center;gap:0;">
      <input type="text" id="search-input" placeholder="/ search allocations...">
      <button id="regex-toggle" title="Toggle regex mode">.*</button>
    </div>
    <label class="toggle">
      <input type="checkbox" id="autofit-toggle">
      Auto-fit Y
    </label>
    <label class="toggle">
      <input type="checkbox" id="hwm-toggle" checked>
      HWM
    </label>
    <label class="toggle" title="Show cached allocator memory, including CUDA Graph private pools">
      <input type="checkbox" id="reserved-toggle">
      Reserved
    </label>
    <label class="toggle" title="Hide allocations that were never freed during recording">
      <input type="checkbox" id="dim-persistent-toggle">
      Hide never-freed
    </label>
    <span id="settings-trigger" title="Settings">&#9881;
      <div id="settings-dropdown">
        <label>Color by
          <select id="color-mode">
            <option value="stack">stack</option>
            <option value="size">size</option>
            <option value="category">category</option>
            <option value="order">order</option>
          </select>
        </label>
      </div>
    </span>
    </div>
  </div>
</div>
<div id="main">
  <div id="charts-wrapper">
    <div class="chart-pane pane-left" id="chart-left">
      <span class="pane-label">__TITLE_LEFT__</span>
    </div>
    <div class="chart-pane pane-right" id="chart-right">
      <span class="pane-label">__TITLE_RIGHT__</span>
    </div>
  </div>
  <button id="panel-toggle" title="Show stack/details" aria-expanded="false">&#9654;</button>
  <div id="detail-panel" class="collapsed">
    <div id="resize-handle"></div>
    <div id="detail-header">
      <div style="display:flex;align-items:center;">
        <div class="detail-tabs"></div>
        <div class="lr-toggle" id="lr-toggle" style="display:none;">
          <button class="lr-btn active-left" data-side="left">L</button>
          <button class="lr-btn" data-side="right">R</button>
        </div>
      </div>
      <div class="detail-actions">
        <span class="detail-stats" id="detail-stats"></span>
      </div>
    </div>
    <div id="detail-body">
      <div class="empty-detail">Click an allocation to inspect its stack trace</div>
    </div>
  </div>
</div>
<div id="minimaps">
  <div class="minimap-pane" id="minimap-left"></div>
  <div class="minimap-pane" id="minimap-right"></div>
</div>
<div id="shortcut-bar" style="display:none">
  <span><kbd>A</kbd><kbd>D</kbd> pan</span>
  <span><kbd>W</kbd><kbd>S</kbd> zoom</span>
  <div class="sep"></div>
  <span><kbd>[</kbd><kbd>]</kbd> speed: <span id="speed-indicator">3</span>/5</span>
  <div class="sep"></div>
  <span><kbd>/</kbd> search</span>
  <div class="sep"></div>
  <span><kbd>?</kbd> toggle shortcuts</span>
</div>
<div id="tooltip"></div>

<script id="bootstrap-left" type="application/json">__BOOTSTRAP_LEFT__</script>
<script id="bootstrap-right" type="application/json">__BOOTSTRAP_RIGHT__</script>
<script>__D3_SOURCE__</script>
<script>
const BOOTSTRAP_LEFT = JSON.parse(document.getElementById('bootstrap-left').textContent);
const BOOTSTRAP_RIGHT = JSON.parse(document.getElementById('bootstrap-right').textContent);

function formatBytes(b) {
  if (Math.abs(b) >= 1024**3) return (b / 1024**3).toFixed(2) + ' GiB';
  if (Math.abs(b) >= 1024**2) return (b / 1024**2).toFixed(1) + ' MiB';
  if (Math.abs(b) >= 1024)    return (b / 1024).toFixed(0) + ' KiB';
  return b + ' B';
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

function formatPool(pool) {
  return pool ? `(${pool[0]}, ${pool[1]})` : 'default/unknown';
}

function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const k = (n + h / 30) % 12;
    const c = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * c).toString(16).padStart(2, '0');
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

const PALETTE = Array.from({length: 128}, (_, i) => hslToHex((i * 137.508) % 360, 42, 52));
const SIZE_PALETTE = Array.from({length: 32}, (_, i) => hslToHex((i * 137.508) % 360, 35, 48));
const PERSISTENT_ALPHAS = [0.55, 0.62, 0.70];

function classifyFrame(frame) {
  if (frame.includes('::')) return 'native';
  if (frame.includes('.cpp:') || frame.includes('.c:')) return 'native';
  if (!frame.includes('/') && !frame.includes('.py')) return 'noise';
  if (frame.includes('/site-packages/') || frame.includes('/torch/')) return 'library';
  if (frame.includes('/lib/python') || frame.includes('/conda/') || frame.includes('lib/python')) return 'library';
  return 'user';
}

const NOISE_FRAMES = new Set([
  'cfunction_call', '_PyEval_EvalFrameDefault', 'PyEval_EvalCode',
  '_PyObject_Call_Prepend', 'slot_tp_call', 'PyObject_Call',
  '_PyObject_MakeTpCall', '_PyFunction_Vectorcall', 'pymain_run_file',
  'pyrun_file', '_PyRun_SimpleFileObject', '_PyRun_AnyFileObject',
  'Py_RunMain', 'pymain_run_file_obj', 'pymain_run_module',
  '_start', '__libc_start_main', '__libc_init_first', 'main',
]);

function frameFunc(frame) {
  if (frame.includes('::')) return frame.split('::').pop();
  const sp = frame.indexOf(' ', frame.lastIndexOf(':'));
  return sp > 0 ? frame.substring(sp + 1) : frame;
}

function isNoiseFrame(frame) {
  const fn = frameFunc(frame);
  return NOISE_FRAMES.has(fn) || fn.startsWith('_Py') || fn.startsWith('Py_')
    || fn.startsWith('pymain_') || fn.startsWith('pyrun_')
    || /^run_mod\.llvm\.|^pymain_main\.llvm\./.test(fn);
}

function renderFrame(frame) {
  const hasColon = frame.includes(':');
  const isCpp = !hasColon && frame.includes('::');
  if (isCpp) {
    const parts = frame.split('::');
    const funcName = parts[parts.length - 1];
    const ns = parts.slice(0, -1).join('::');
    return `<span class="frame-cpp">${escapeHtml(ns)}::</span><span class="frame-basename">${escapeHtml(funcName)}</span>`;
  }
  if (hasColon) {
    const sp = frame.indexOf(' ', frame.lastIndexOf(':'));
    if (sp > 0) {
      const filePart = frame.substring(0, sp);
      const funcPart = frame.substring(sp + 1);
      const lastSlash = filePart.lastIndexOf('/');
      const basename = lastSlash >= 0 ? filePart.substring(lastSlash + 1) : filePart;
      const dir = lastSlash >= 0 ? filePart.substring(0, lastSlash + 1) : '';
      return `<span class="frame-file">${escapeHtml(dir)}</span><span class="frame-basename">${escapeHtml(basename)}</span> <span class="frame-func">${escapeHtml(funcPart)}</span>`;
    }
    return `<span class="frame-file">${escapeHtml(frame)}</span>`;
  }
  return escapeHtml(frame);
}

// --- Shared tooltip ---
const tooltipEl = document.getElementById('tooltip');
function showTooltip(event, html) {
  tooltipEl.innerHTML = html;
  tooltipEl.style.display = 'block';
  const tw = tooltipEl.offsetWidth, th = tooltipEl.offsetHeight;
  tooltipEl.style.left = (event.pageX + 16 + tw > window.innerWidth ? event.pageX - tw - 12 : event.pageX + 16) + 'px';
  tooltipEl.style.top = (event.pageY + 16 + th > window.innerHeight ? event.pageY - th - 12 : event.pageY + 16) + 'px';
}
function hideTooltip() { tooltipEl.style.display = 'none'; }

// --- Header stats ---
document.getElementById('peak-stat-left').textContent = formatBytes(BOOTSTRAP_LEFT.meta.high_water_mark_bytes);
document.getElementById('allocs-stat-left').textContent = BOOTSTRAP_LEFT.meta.num_allocs.toLocaleString();
document.getElementById('peak-stat-right').textContent = formatBytes(BOOTSTRAP_RIGHT.meta.high_water_mark_bytes);
document.getElementById('allocs-stat-right').textContent = BOOTSTRAP_RIGHT.meta.num_allocs.toLocaleString();

// --- Shared UI state ---
const detailBody = document.getElementById('detail-body');
const detailStats = document.getElementById('detail-stats');
const detailTabs = document.querySelector('.detail-tabs');
const lrToggle = document.getElementById('lr-toggle');
const EMPTY_STACK_DETAIL = '<div class="empty-detail">Click an allocation to inspect its stack trace</div>';

let colorMode = 'stack';
let dimPersistent = false;
let showReserved = false;
let searchMatcher = null;
let useRegex = false;
let activeSide = 'left';

const uiState = {
  activeDetailView: 'stack',
  selectedAlloc: null,
  selectedSide: null,
  selectedStackIdx: -1,
  selectedStackLabel: '',
};

// ============================================================
// Chart pane factory
// ============================================================
function createChartPane(bootstrap, containerEl, minimapEl, paneId) {
  const {
    timeline: TIMELINE,
    reserved_timeline: RESERVED_TIMELINE,
    allocs: ALLOCS,
    frames: FRAMES,
    stacks: STACKS,
    events: EVENTS,
    segments: SEGMENTS,
    private_pools: PRIVATE_POOLS,
    allocator_settings: ALLOCATOR_SETTINGS,
    meta: META,
  } = bootstrap;
  const MAX_TS = Math.max(1, META.max_timestep);

  function resolveStack(stackIdx) {
    return (STACKS[stackIdx] || []).map(i => FRAMES[i]);
  }

  function bestFrame(stackIdx) {
    const stack = resolveStack(stackIdx);
    for (const f of stack) { if (classifyFrame(f) === 'user') return f; }
    for (const f of stack) { if (f.includes('.py') && !isNoiseFrame(f)) return f; }
    for (const f of stack) { if (f.includes('::') && !isNoiseFrame(f)) return f; }
    for (const f of stack) { if (!isNoiseFrame(f)) return f; }
    return stack[0] || '';
  }

  function tooltipFrameInfo(stackIdx) {
    const stack = resolveStack(stackIdx);
    let userFrame = null, apiFrame = null;
    for (const f of stack) { if (classifyFrame(f) === 'user') { userFrame = f; break; } }
    for (const f of stack) {
      if (f.includes('.py') && !isNoiseFrame(f) && classifyFrame(f) === 'library') { apiFrame = f; break; }
    }
    if (!apiFrame) {
      for (const f of stack) { if (f.includes('::') && !isNoiseFrame(f)) { apiFrame = f; break; } }
    }
    return { userFrame, apiFrame };
  }

  // Color + alpha arrays
  const allocSizes = ALLOCS.map(a => a.s);
  const sortedSizes = [...new Set(allocSizes)].sort((a, b) => a - b);
  const sizeToColorIdx = new Map();
  sortedSizes.forEach((s, i) => sizeToColorIdx.set(s, i % SIZE_PALETTE.length));
  const categoryValues = [...new Set(ALLOCS.map(a => a.category ?? 'unknown'))];
  const categoryToColorIdx = new Map(categoryValues.map((category, i) => [category, i]));

  const allocStarts = new Float64Array(ALLOCS.length);
  const allocEnds = new Float64Array(ALLOCS.length);
  for (let i = 0; i < ALLOCS.length; i++) {
    allocStarts[i] = ALLOCS[i].ts[0];
    allocEnds[i] = ALLOCS[i].ts[ALLOCS[i].ts.length - 1];
  }

  const allocPersistent = new Uint8Array(ALLOCS.length);
  const allocColors = new Array(ALLOCS.length);
  const allocAlphas = new Float64Array(ALLOCS.length);
  for (let i = 0; i < ALLOCS.length; i++) {
    allocPersistent[i] = allocEnds[i] >= META.max_timestep ? 1 : 0;
  }

  function recolorAllocs() {
    let pIdx = 0;
    for (let i = 0; i < ALLOCS.length; i++) {
      switch (colorMode) {
        case 'size': allocColors[i] = SIZE_PALETTE[sizeToColorIdx.get(ALLOCS[i].s)]; break;
        case 'category': allocColors[i] = PALETTE[categoryToColorIdx.get(ALLOCS[i].category ?? 'unknown') % PALETTE.length]; break;
        case 'order': allocColors[i] = PALETTE[i % PALETTE.length]; break;
        default: allocColors[i] = PALETTE[ALLOCS[i].si % PALETTE.length]; break;
      }
      allocAlphas[i] = allocPersistent[i] ? PERSISTENT_ALPHAS[pIdx++ % PERSISTENT_ALPHAS.length] : 0.85;
    }
  }

  // Derived data
  function buildDerivedData() {
    const stackFrameLabels = Array.from({ length: STACKS.length }, (_, si) => bestFrame(si));
    const peakAllocIndices = [];
    let peakTotalBytes = 0;
    const leakAllocIndices = [];
    let leakTotalBytes = 0;
    const leakGroupsByFrame = new Map();
    const peakTs = META.hwm_timestep;
    const maxTs = META.max_timestep;
    const earlyThreshold = maxTs * 0.05;

    for (let ai = 0; ai < ALLOCS.length; ai++) {
      const firstTs = allocStarts[ai], lastTs = allocEnds[ai];
      if (firstTs <= peakTs && lastTs >= peakTs) {
        peakAllocIndices.push(ai);
        peakTotalBytes += ALLOCS[ai].s;
      }
      if (lastTs >= maxTs && firstTs > earlyThreshold) {
        leakAllocIndices.push(ai);
        leakTotalBytes += ALLOCS[ai].s;
        const frame = stackFrameLabels[ALLOCS[ai].si];
        let group = leakGroupsByFrame.get(frame);
        if (!group) {
          group = { frame, si: ALLOCS[ai].si, count: 0, totalBytes: 0 };
          leakGroupsByFrame.set(frame, group);
        }
        group.count += 1;
        group.totalBytes += ALLOCS[ai].s;
      }
    }
    peakAllocIndices.sort((a, b) => ALLOCS[b].s - ALLOCS[a].s);

    return {
      stackFrameLabels,
      peakAllocIndices,
      peakTotalBytes,
      leakAllocIndices,
      leakTotalBytes,
      leakGroups: Array.from(leakGroupsByFrame.values()).sort((a, b) => b.totalBytes - a.totalBytes),
    };
  }

  const derivedData = buildDerivedData();
  recolorAllocs();

  // Hit bucket index
  const NUM_HIT_BUCKETS = Math.max(1, Math.min(2000, MAX_TS));
  const hitBucketSize = MAX_TS / NUM_HIT_BUCKETS;
  const hitBuckets = new Array(NUM_HIT_BUCKETS + 1);
  for (let b = 0; b <= NUM_HIT_BUCKETS; b++) hitBuckets[b] = [];
  for (let ai = 0; ai < ALLOCS.length; ai++) {
    const b0 = Math.max(0, Math.floor(allocStarts[ai] / hitBucketSize));
    const b1 = Math.min(NUM_HIT_BUCKETS, Math.floor(allocEnds[ai] / hitBucketSize));
    for (let b = b0; b <= b1; b++) hitBuckets[b].push(ai);
  }

  // Search match cache
  let searchMatchSet = null;
  function updateSearchCache() {
    if (!searchMatcher) { searchMatchSet = null; return; }
    searchMatchSet = new Set();
    for (let ai = 0; ai < ALLOCS.length; ai++) {
      const stack = resolveStack(ALLOCS[ai].si);
      if (stack.some(f => searchMatcher.test(f))) searchMatchSet.add(ai);
    }
  }

  // --- Chart setup ---
  let containerRect = containerEl.getBoundingClientRect();
  const margin = { top: 20, right: 20, bottom: 40, left: 70 };
  let chartWidth = Math.max(1, containerRect.width - margin.left - margin.right);
  let chartHeight = Math.max(1, containerRect.height - margin.top - margin.bottom);

  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, containerRect.width * devicePixelRatio);
  canvas.height = Math.max(1, containerRect.height * devicePixelRatio);
  canvas.style.width = containerRect.width + 'px';
  canvas.style.height = containerRect.height + 'px';
  containerEl.insertBefore(canvas, containerEl.firstChild);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

  const svg = d3.select(containerEl).append('svg')
    .attr('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`)
    .attr('preserveAspectRatio', 'none');

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const clipRect = svg.append('defs').append('clipPath').attr('id', `clip-${paneId}`)
    .append('rect').attr('width', chartWidth).attr('height', chartHeight);

  const xScale = d3.scaleLinear().domain([0, MAX_TS]).range([0, chartWidth]);
  const yScale = d3.scaleLinear().domain([0, Math.max(1, META.high_water_mark_bytes) * 1.05]).range([chartHeight, 0]);

  const xAxis = d3.axisBottom(xScale).ticks(6);
  const yAxisFn = d3.axisLeft(yScale).ticks(6).tickFormat(d => formatBytes(d));

  const gridG = g.append('g').attr('class', 'grid')
    .call(d3.axisLeft(yScale).ticks(6).tickSize(-chartWidth).tickFormat(''));
  const xAxisG = g.append('g').attr('class', 'axis x-axis')
    .attr('transform', `translate(0,${chartHeight})`).call(xAxis);
  const yAxisG = g.append('g').attr('class', 'axis y-axis').call(yAxisFn);

  const chartArea = g.append('g').attr('clip-path', `url(#clip-${paneId})`);

  const hwmG = chartArea.append('g').attr('class', 'hwm-group').style('cursor', 'pointer');
  hwmG.append('line').attr('class', 'hwm-line')
    .attr('x1', 0).attr('x2', chartWidth)
    .attr('y1', yScale(META.high_water_mark_bytes)).attr('y2', yScale(META.high_water_mark_bytes));
  hwmG.append('text').attr('class', 'hwm-label')
    .attr('x', chartWidth - 4).attr('y', yScale(META.high_water_mark_bytes) - 6)
    .attr('text-anchor', 'end')
    .text('HWM: ' + formatBytes(META.high_water_mark_bytes));
  const reservedPath = chartArea.append('path').attr('class', 'reserved-line').style('display', 'none');
  const poolLineG = chartArea.append('g').attr('class', 'pool-reserved-lines');
  const markerData = EVENTS.filter(event => event.act === 'oom' || event.act === 'snapshot');
  const markerG = chartArea.append('g').attr('class', 'event-markers');

  let currentTransform = d3.zoomIdentity;
  let hoveredAlloc = null;
  let customYDomain = null;
  let yMode = 'fixed';

  function tracePoly(ai, newX) {
    const d = ALLOCS[ai];
    const ts = d.ts, offsets = d.offsets, size = d.s;
    ctx.moveTo(newX(ts[0]), yScale(offsets[0]));
    for (let i = 1; i < ts.length; i++) ctx.lineTo(newX(ts[i]), yScale(offsets[i]));
    for (let i = ts.length - 1; i >= 0; i--) ctx.lineTo(newX(ts[i]), yScale(offsets[i] + size));
    ctx.closePath();
  }

  function drawCanvas() {
    const cw = containerEl.offsetWidth, ch = containerEl.offsetHeight;
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();

    ctx.clearRect(0, 0, cw, ch);
    ctx.save();
    ctx.translate(margin.left, margin.top);
    ctx.beginPath();
    ctx.rect(0, 0, chartWidth, chartHeight);
    ctx.clip();

    const pxPerTs = chartWidth / (d1 - d0);
    const minVisPx = 0.5;
    const batches = {};
    let hoveredIdx = -1;

    for (let ai = 0; ai < ALLOCS.length; ai++) {
      if (allocEnds[ai] < d0 || allocStarts[ai] > d1) continue;
      const visW = (Math.min(allocEnds[ai], d1) - Math.max(allocStarts[ai], d0)) * pxPerTs;
      const visH = yScale(0) - yScale(ALLOCS[ai].s);
      if (visW < minVisPx && visH < minVisPx) continue;
      if (dimPersistent && allocPersistent[ai]) continue;
      if (ALLOCS[ai] === hoveredAlloc) { hoveredIdx = ai; continue; }

      let alpha = allocAlphas[ai];
      if (searchMatchSet !== null) alpha = searchMatchSet.has(ai) ? 0.9 : 0.06;

      const key = allocColors[ai] + alpha;
      if (!batches[key]) batches[key] = { color: allocColors[ai], alpha, indices: [] };
      batches[key].indices.push(ai);
    }

    for (const batch of Object.values(batches)) {
      ctx.beginPath();
      for (const ai of batch.indices) tracePoly(ai, newX);
      ctx.globalAlpha = batch.alpha;
      ctx.fillStyle = batch.color;
      ctx.fill();
      ctx.globalAlpha = Math.min(batch.alpha, 0.3);
      ctx.strokeStyle = 'rgba(0,0,0,0.5)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    if (hoveredIdx >= 0) {
      ctx.beginPath();
      tracePoly(hoveredIdx, newX);
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = allocColors[hoveredIdx];
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    ctx.restore();
  }

  function hitTest(mx, my) {
    const newX = currentTransform.rescaleX(xScale);
    const dataX = newX.invert(mx - margin.left);
    const dataY = yScale.invert(my - margin.top);
    const bi = Math.max(0, Math.min(NUM_HIT_BUCKETS, Math.floor(dataX / hitBucketSize)));
    const candidates = hitBuckets[bi];
    let best = null, bestSize = Infinity;
    for (const ai of candidates) {
      if (dataX < allocStarts[ai] || dataX > allocEnds[ai]) continue;
      const d = ALLOCS[ai];
      let offset = d.offsets[0];
      for (let i = 1; i < d.ts.length; i++) { if (d.ts[i] > dataX) break; offset = d.offsets[i]; }
      if (dataY >= offset && dataY <= offset + d.s && d.s < bestSize) { best = d; bestSize = d.s; }
    }
    return best;
  }

  function getBaseYDomain(d0, d1) {
    const fixedPeak = Math.max(
      1,
      META.high_water_mark_bytes,
      showReserved ? META.reserved_high_water_mark_bytes : 0,
    );
    if (yMode !== 'autofit' && !dimPersistent) return [0, fixedPeak * 1.05];

    let minY = Infinity, maxY = 0;
    for (let ai = 0; ai < ALLOCS.length; ai++) {
      if (allocEnds[ai] < d0 || allocStarts[ai] > d1) continue;
      if (dimPersistent && allocPersistent[ai]) continue;
      const d = ALLOCS[ai];
      let offset = d.offsets[0];
      for (let i = 0; i < d.ts.length; i++) {
        if (d.ts[i] <= d0) offset = d.offsets[i];
        if (d.ts[i] >= d0 && d.ts[i] <= d1) {
          minY = Math.min(minY, d.offsets[i]);
          maxY = Math.max(maxY, d.offsets[i] + d.s);
        }
      }
      minY = Math.min(minY, offset);
      maxY = Math.max(maxY, offset + d.s);
    }
    if (showReserved) {
      const start = Math.max(0, Math.floor(d0));
      const end = Math.min(RESERVED_TIMELINE.length, Math.ceil(d1) + 1);
      for (let timestep = start; timestep < end; timestep++) {
        maxY = Math.max(maxY, RESERVED_TIMELINE[timestep]);
      }
    }
    if (maxY === 0) { minY = 0; maxY = fixedPeak; }
    if (minY === Infinity) minY = 0;
    const pad = Math.max(1, (maxY - minY) * 0.05);
    return [Math.max(0, minY - pad), maxY + pad];
  }

  const chartUpdateHooks = [];

  function updateChart(transform) {
    currentTransform = transform;
    const newX = transform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    yScale.domain(customYDomain || getBaseYDomain(d0, d1));

    xAxisG.call(xAxis.scale(newX));
    xAxisG.selectAll('text').attr('fill', 'var(--text-muted)');
    xAxisG.selectAll('line, path').attr('stroke', 'var(--border)');
    yAxisG.call(yAxisFn);
    yAxisG.selectAll('text').attr('fill', 'var(--text-muted)');
    yAxisG.selectAll('line, path').attr('stroke', 'var(--border)');
    gridG.call(d3.axisLeft(yScale).ticks(6).tickSize(-chartWidth).tickFormat(''));
    gridG.selectAll('line').attr('stroke', 'var(--grid)');
    gridG.selectAll('path').attr('stroke', 'none');

    const hwmY = yScale(META.high_water_mark_bytes);
    hwmG.select('.hwm-line').attr('x2', chartWidth).attr('y1', hwmY).attr('y2', hwmY);
    hwmG.select('.hwm-label').attr('x', chartWidth - 4).attr('y', hwmY - 6);
    if (showReserved) {
      reservedPath
        .style('display', null)
        .datum(RESERVED_TIMELINE)
        .attr('d', d3.line().x((d, i) => newX(i)).y(d => yScale(d)));
      poolLineG.selectAll('path').data(PRIVATE_POOLS).join('path')
        .attr('class', 'pool-reserved-line')
        .attr('d', pool => d3.line()
          .x(point => newX(point.step))
          .y(point => yScale(point.reserved))(pool.timeline));
    } else {
      reservedPath.style('display', 'none').attr('d', null);
      poolLineG.selectAll('path').remove();
    }
    markerG.selectAll('line').data(markerData).join('line')
      .attr('class', event => `event-marker ${event.act}`)
      .attr('x1', event => newX(event.step)).attr('x2', event => newX(event.step))
      .attr('y1', 0).attr('y2', chartHeight);

    drawCanvas();
    for (const hook of chartUpdateHooks) hook();
  }

  function transformForDomain(d0, d1) {
    const range = Math.max(1e-9, d1 - d0);
    return d3.zoomIdentity.translate(-d0 * chartWidth / range, 0).scale(MAX_TS / range);
  }

  // Zoom behavior
  let onZoomCallback = null;
  let syncing = false;

  const zoom = d3.zoom()
    .scaleExtent([1, 2000])
    .filter(event => !event.shiftKey)
    .translateExtent([[0, 0], [chartWidth, chartHeight]])
    .extent([[0, 0], [chartWidth, chartHeight]])
    .on('zoom', (event) => {
      updateChart(event.transform);
      if (!syncing && onZoomCallback) {
        const newX = event.transform.rescaleX(xScale);
        const [d0, d1] = newX.domain();
        onZoomCallback(d0 / MAX_TS, d1 / MAX_TS);
      }
    });

  const zoomRect = chartArea.append('rect')
    .attr('width', chartWidth).attr('height', chartHeight)
    .attr('fill', 'none').attr('pointer-events', 'all')
    .call(zoom);

  // Box zoom
  let boxStart = null;
  const boxRect = chartArea.append('rect')
    .attr('fill', 'rgba(62, 147, 204, 0.15)')
    .attr('stroke', 'var(--accent)')
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '4 2')
    .style('display', 'none')
    .attr('pointer-events', 'none');

  svg.node().addEventListener('pointerdown', function(event) {
    if (!event.shiftKey || event.button !== 0) return;
    event.preventDefault();
    const [mx, my] = d3.pointer(event, g.node());
    boxStart = { x: Math.max(0, Math.min(chartWidth, mx)), y: Math.max(0, Math.min(chartHeight, my)) };
    boxRect.style('display', null).attr('width', 0).attr('height', 0);
    svg.node().setPointerCapture(event.pointerId);
  });

  svg.node().addEventListener('pointermove', function(event) {
    if (!boxStart) return;
    const [mx, my] = d3.pointer(event, g.node());
    const cx = Math.max(0, Math.min(chartWidth, mx));
    const cy = Math.max(0, Math.min(chartHeight, my));
    boxRect
      .attr('x', Math.min(boxStart.x, cx))
      .attr('y', Math.min(boxStart.y, cy))
      .attr('width', Math.abs(cx - boxStart.x))
      .attr('height', Math.abs(cy - boxStart.y));
  });

  svg.node().addEventListener('pointerup', function(event) {
    if (!boxStart) return;
    const [mx, my] = d3.pointer(event, g.node());
    const x0 = Math.max(0, Math.min(boxStart.x, mx));
    const x1 = Math.min(chartWidth, Math.max(boxStart.x, mx));
    const y0 = Math.max(0, Math.min(boxStart.y, my));
    const y1 = Math.min(chartHeight, Math.max(boxStart.y, my));
    boxStart = null;
    boxRect.style('display', 'none');
    if (x1 - x0 < 5 || y1 - y0 < 5) return;
    const newX = currentTransform.rescaleX(xScale);
    customYDomain = [yScale.invert(y1), yScale.invert(y0)];
    zoomRect.transition().duration(300).call(zoom.transform, transformForDomain(newX.invert(x0), newX.invert(x1)));
  });

  // Mouse interaction
  zoomRect.on('mousemove', function(event) {
    const [mx, my] = d3.pointer(event, svg.node());
    const hit = hitTest(mx, my);
    if (hit !== hoveredAlloc) { hoveredAlloc = hit; drawCanvas(); }
    if (hit) {
      const info = tooltipFrameInfo(hit.si);
      const primary = info.userFrame || info.apiFrame;
      const secondary = info.userFrame && info.apiFrame ? info.apiFrame : null;
      const lines = [`<div class="tt-row"><span class="tt-label">Size:</span><span class="tt-value">${formatBytes(hit.s)}</span></div>`];
      lines.push(`<div class="tt-row"><span class="tt-label">Pool:</span><span class="tt-value">${escapeHtml(formatPool(hit.pool))}</span></div>`);
      if (primary) lines.push(`<div class="tt-${info.userFrame ? 'user' : 'api'}">${escapeHtml(primary)}</div>`);
      if (secondary) lines.push(`<div class="tt-api">${escapeHtml(secondary)}</div>`);
      showTooltip(event, lines.join(''));
    } else {
      hideTooltip();
    }
  });

  zoomRect.on('mouseleave', function() {
    if (hoveredAlloc) { hoveredAlloc = null; drawCanvas(); }
    hideTooltip();
  });

  zoomRect.on('dblclick.zoom', null);
  zoomRect.on('dblclick', function() {
    customYDomain = null;
    zoomRect.transition().duration(300).call(zoom.transform, d3.zoomIdentity);
  });

  // Minimap
  let mmRect = minimapEl.getBoundingClientRect();
  let mmW = Math.max(1, mmRect.width - 16);
  const mmH = 32;
  const miniSvg = d3.select(minimapEl).append('svg')
    .attr('viewBox', `0 0 ${mmW + 16} ${mmH + 8}`);
  const miniG = miniSvg.append('g').attr('transform', 'translate(8,4)');
  const miniX = d3.scaleLinear().domain([0, MAX_TS]).range([0, mmW]);
  const miniY = d3.scaleLinear()
    .domain([0, Math.max(1, META.high_water_mark_bytes, META.reserved_high_water_mark_bytes) * 1.05])
    .range([mmH, 0]);

  miniG.append('path')
    .datum(TIMELINE)
    .attr('class', 'minimap-area')
    .attr('d', d3.area().x((d, i) => i * mmW / Math.max(1, TIMELINE.length - 1)).y0(mmH).y1(d => miniY(d)));

  const viewportRect = miniG.append('rect')
    .attr('class', 'minimap-viewport').attr('y', 0).attr('height', mmH);

  function updateMinimap() {
    miniY.domain([
      0,
      Math.max(
        1,
        META.high_water_mark_bytes,
        showReserved ? META.reserved_high_water_mark_bytes : 0,
      ) * 1.05,
    ]);
    miniG.select('.minimap-area').attr(
      'd',
      d3.area()
        .x((d, i) => i * mmW / Math.max(1, TIMELINE.length - 1))
        .y0(mmH)
        .y1(d => miniY(d)),
    );
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    const x0 = miniX(Math.max(0, d0));
    const x1 = miniX(Math.min(MAX_TS, d1));
    viewportRect.attr('x', x0).attr('width', Math.max(2, x1 - x0));
  }

  function resize() {
    const [domainStart, domainEnd] = currentTransform.rescaleX(xScale).domain();
    containerRect = containerEl.getBoundingClientRect();
    chartWidth = Math.max(1, containerRect.width - margin.left - margin.right);
    chartHeight = Math.max(1, containerRect.height - margin.top - margin.bottom);

    canvas.width = Math.max(1, containerRect.width * devicePixelRatio);
    canvas.height = Math.max(1, containerRect.height * devicePixelRatio);
    canvas.style.width = `${containerRect.width}px`;
    canvas.style.height = `${containerRect.height}px`;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

    svg.attr('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`);
    clipRect.attr('width', chartWidth).attr('height', chartHeight);

    xScale.range([0, chartWidth]);
    yScale.range([chartHeight, 0]);

    xAxisG.attr('transform', `translate(0,${chartHeight})`);
    zoom.translateExtent([[0, 0], [chartWidth, chartHeight]]);
    zoom.extent([[0, 0], [chartWidth, chartHeight]]);
    zoomRect.attr('width', chartWidth).attr('height', chartHeight);

    hwmG.select('.hwm-line').attr('x2', chartWidth);
    hwmG.select('.hwm-label').attr('x', chartWidth - 4);

    mmRect = minimapEl.getBoundingClientRect();
    mmW = Math.max(1, mmRect.width - 16);
    miniSvg.attr('viewBox', `0 0 ${mmW + 16} ${mmH + 8}`);
    miniX.range([0, mmW]);
    miniG.select('.minimap-area').attr(
      'd',
      d3.area().x((d, i) => i * mmW / Math.max(1, TIMELINE.length - 1)).y0(mmH).y1(d => miniY(d))
    );

    const start = Math.max(0, Math.min(MAX_TS, domainStart));
    const end = Math.max(start + 1e-9, Math.min(MAX_TS, domainEnd));
    zoomRect.call(zoom.transform, transformForDomain(start, end));
  }

  updateMinimap();
  chartUpdateHooks.push(updateMinimap);

  const miniDrag = d3.drag().on('drag', function(event) {
    const domainPerPx = MAX_TS / mmW;
    const shift = event.dx * domainPerPx;
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    const range = d1 - d0;
    const newD0 = Math.max(0, Math.min(MAX_TS - range, d0 + shift));
    zoomRect.call(zoom.transform, transformForDomain(newD0, newD0 + range));
  });
  viewportRect.call(miniDrag);

  miniSvg.on('click', function(event) {
    const [mx] = d3.pointer(event, miniG.node());
    const clickTs = miniX.invert(mx);
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    const range = d1 - d0;
    const newD0 = Math.max(0, Math.min(MAX_TS - range, clickTs - range / 2));
    zoomRect.transition().duration(300).call(zoom.transform, transformForDomain(newD0, newD0 + range));
  });

  drawCanvas();

  // HWM click -> peak breakdown
  hwmG.on('click', function() {
    activeSide = paneId;
    activateDetailView('peak');
  });

  return {
    META,
    ALLOCS,
    FRAMES,
    STACKS,
    EVENTS,
    SEGMENTS,
    PRIVATE_POOLS,
    ALLOCATOR_SETTINGS,
    derivedData,
    resolveStack,
    recolorAllocs,
    updateSearchCache,
    drawCanvas,
    updateChart,
    paneId,
    setYMode(mode) { yMode = mode; customYDomain = null; updateChart(currentTransform); },
    setDimPersistent() { customYDomain = null; updateChart(currentTransform); },
    setHwmVisible(v) { hwmG.style('display', v ? null : 'none'); },
    setReservedVisible() { customYDomain = null; updateChart(currentTransform); },
    syncZoom(fracStart, fracEnd) {
      syncing = true;
      const d0 = fracStart * MAX_TS;
      const d1 = fracEnd * MAX_TS;
      zoomRect.call(zoom.transform, transformForDomain(d0, d1));
      syncing = false;
    },
    resetZoom() {
      customYDomain = null;
      zoomRect.transition().duration(300).call(zoom.transform, d3.zoomIdentity);
    },
    onZoom(cb) { onZoomCallback = cb; },
    onClick(cb) {
      zoomRect.on('click', function(event) {
        const [mx, my] = d3.pointer(event, svg.node());
        const hit = hitTest(mx, my);
        if (hit) cb(hit, paneId);
      });
    },
    resize,
    getTransformFrac() {
      const newX = currentTransform.rescaleX(xScale);
      const [d0, d1] = newX.domain();
      return [d0 / MAX_TS, d1 / MAX_TS];
    },
  };
}

// ============================================================
// Instantiate two panes
// ============================================================
const leftPane = createChartPane(BOOTSTRAP_LEFT, document.getElementById('chart-left'), document.getElementById('minimap-left'), 'left');
const rightPane = createChartPane(BOOTSTRAP_RIGHT, document.getElementById('chart-right'), document.getElementById('minimap-right'), 'right');
const panes = { left: leftPane, right: rightPane };
const visiblePanes = { left: true, right: true };

// Zoom linking
leftPane.onZoom((f0, f1) => rightPane.syncZoom(f0, f1));
rightPane.onZoom((f0, f1) => leftPane.syncZoom(f0, f1));

// Click handling -> shared detail panel
function handleAllocClick(alloc, side) {
  uiState.selectedAlloc = alloc;
  uiState.selectedSide = side;
  uiState.selectedStackIdx = alloc.si;
  uiState.selectedStackLabel = `${side === 'left' ? 'L' : 'R'}: ${formatBytes(alloc.s)}`;
  handleAllocationSelection();
}

leftPane.onClick(handleAllocClick);
rightPane.onClick(handleAllocClick);

// ============================================================
// Detail panel rendering (shared)
// ============================================================
function resolveStackForSide(side, stackIdx) {
  const pane = panes[side];
  return (pane.STACKS[stackIdx] || []).map(i => pane.FRAMES[i]);
}

function renderStack(side, stackIdx, label) {
  const stack = resolveStackForSide(side, stackIdx);
  detailStats.textContent = label;
  if (!stack.length) {
    detailBody.innerHTML = '<div class="empty-detail">No frames recorded</div>';
    return;
  }
  detailBody.innerHTML = stack.map(f => {
    const cls = classifyFrame(f);
    return `<div class="stack-frame frame-${cls}"><span class="frame-text">${renderFrame(f)}</span></div>`;
  }).join('');
}

function renderStackSelection() {
  if (uiState.selectedStackIdx >= 0 && uiState.selectedSide) {
    renderStack(uiState.selectedSide, uiState.selectedStackIdx, uiState.selectedStackLabel);
    return;
  }
  detailBody.innerHTML = EMPTY_STACK_DETAIL;
}

function showDetails() {
  const d = uiState.selectedAlloc;
  if (!d) {
    detailBody.innerHTML = '<div class="empty-detail">Click an allocation to see its details</div>';
    return;
  }
  const ts = d.time_us === null || d.time_us === undefined
    ? 'N/A'
    : new Date(d.time_us / 1000).toLocaleString();
  const pane = panes[uiState.selectedSide];
  const lifetimeEnd = d.ts[d.ts.length - 1];
  detailStats.textContent = `${uiState.selectedSide === 'left' ? 'L' : 'R'}: ${formatBytes(d.s)}`;
  detailBody.innerHTML = `<div class="alloc-details"><table>
    <tr><td>Pane</td><td>${uiState.selectedSide === 'left' ? 'Left' : 'Right'}</td></tr>
    <tr><td>Requested</td><td>${formatBytes(d.s)} (${d.s.toLocaleString()} bytes)</td></tr>
    <tr><td>Block size</td><td>${formatBytes(d.block_size)}</td></tr>
    <tr><td>Address</td><td>${escapeHtml(d.addr || 'N/A')}</td></tr>
    <tr><td>Stream</td><td>${escapeHtml(d.stream ?? 'N/A')}</td></tr>
    <tr><td>Pool</td><td>${escapeHtml(formatPool(d.pool))}</td></tr>
    <tr><td>Origin</td><td>${escapeHtml(d.origin)}${d.ghost ? ' (reconstructed)' : ''}</td></tr>
    <tr><td>Timestamp</td><td>${escapeHtml(ts)}</td></tr>
    <tr><td>Compile ctx</td><td>${escapeHtml(d.ctx || 'None')}</td></tr>
    <tr><td>Metadata</td><td><pre>${escapeHtml(JSON.stringify(d.metadata || {}, null, 2))}</pre></td></tr>
    <tr><td>Annotations</td><td><pre>${escapeHtml(JSON.stringify(d.annotations || [], null, 2))}</pre></td></tr>
    <tr><td>FX</td><td><pre>${escapeHtml(JSON.stringify(d.fx || [], null, 2))}</pre></td></tr>
    <tr><td>Lifetime</td><td>ts ${d.ts[0]} \u2192 ${lifetimeEnd}${lifetimeEnd >= pane.META.max_timestep ? ' (never freed)' : ''}</td></tr>
  </table></div>`;
}

function showPeakBreakdown() {
  const pane = panes[activeSide];
  const alive = pane.derivedData.peakAllocIndices;
  const total = pane.derivedData.peakTotalBytes;
  const maxAliveSize = alive[0] === undefined ? 1 : pane.ALLOCS[alive[0]].s;

  detailStats.textContent = `${alive.length} allocs, ${formatBytes(total)}`;
  let html = `<div class="peak-label">${activeSide === 'left' ? 'Left' : 'Right'}: Allocations alive at peak (${formatBytes(pane.META.high_water_mark_bytes)})</div>`;
  html += alive.map(ai => {
    const d = pane.ALLOCS[ai];
    const pct = (d.s / Math.max(1, pane.META.high_water_mark_bytes) * 100).toFixed(1);
    const barW = (d.s / maxAliveSize * 100).toFixed(0);
    return `<div class="breakdown-row" data-action="show-stack" data-side="${activeSide}" data-stack-idx="${d.si}" data-label="${encodeURIComponent(formatBytes(d.s))}">
      <span class="bd-size">${formatBytes(d.s)}</span>
      <span class="bd-pct">${pct}%</span>
      <span class="bd-bar"><span class="bd-bar-fill" style="width:${barW}%"></span></span>
      <span class="bd-frame">${escapeHtml(pane.derivedData.stackFrameLabels[d.si])}</span>
    </div>`;
  }).join('');
  detailBody.innerHTML = html;
}

function showLeaks() {
  const pane = panes[activeSide];
  const candidates = pane.derivedData.leakAllocIndices;

  if (candidates.length === 0) {
    detailStats.textContent = 'No potential leaks';
    detailBody.innerHTML = '<div class="empty-detail">No potential memory leaks detected.</div>';
    return;
  }

  const groups = pane.derivedData.leakGroups;
  const maxBytes = groups[0]?.totalBytes || 1;

  detailStats.textContent = `${candidates.length} allocs, ${formatBytes(pane.derivedData.leakTotalBytes)}`;

  let html = `<div class="peak-label">${activeSide === 'left' ? 'Left' : 'Right'}: Never-freed allocations (excluding setup phase)</div>`;
  html += groups.map(g => {
    const pct = (g.totalBytes / pane.derivedData.leakTotalBytes * 100).toFixed(1);
    const barW = (g.totalBytes / maxBytes * 100).toFixed(0);
    return `<div class="breakdown-row" data-action="apply-search" data-query="${encodeURIComponent(g.frame)}">
      <span class="bd-size">${formatBytes(g.totalBytes)}</span>
      <span class="bd-count">\u00d7${g.count}</span>
      <span class="bd-pct">${pct}%</span>
      <span class="bd-bar"><span class="bd-bar-fill leak-bar" style="width:${barW}%"></span></span>
      <span class="bd-frame">${escapeHtml(g.frame)}</span>
    </div>`;
  }).join('');
  detailBody.innerHTML = html;
}

function showPools() {
  const pane = panes[activeSide];
  detailStats.textContent = `${pane.PRIVATE_POOLS.length} private pools`;
  if (!pane.PRIVATE_POOLS.length) {
    detailBody.innerHTML = '<div class="empty-detail">No live CUDA Graph or MemPool segments in this snapshot.</div>';
    return;
  }
  const maxReserved = pane.PRIVATE_POOLS.reduce(
    (maximum, pool) => Math.max(maximum, pool.reserved_bytes),
    1,
  );
  detailBody.innerHTML = '<div class="peak-label">Private pool reserved memory</div>' + pane.PRIVATE_POOLS.map(pool => {
    const width = (pool.reserved_bytes / maxReserved * 100).toFixed(0);
    return `<div class="breakdown-row">
      <span class="bd-size">${formatBytes(pool.reserved_bytes)}</span>
      <span class="bd-count">${pool.num_segments} seg</span>
      <span class="bd-pct">${formatBytes(pool.active_bytes)}</span>
      <span class="bd-bar"><span class="bd-bar-fill" style="width:${width}%"></span></span>
      <span class="bd-frame">pool ${escapeHtml(formatPool(pool.id))}, stream ${escapeHtml(pool.stream)}, peak ${formatBytes(pool.peak_reserved_bytes)}, inactive ${formatBytes(pool.inactive_bytes)}</span>
    </div>`;
  }).join('');
}

function showEvents() {
  const pane = panes[activeSide];
  const events = pane.EVENTS.slice(-2000).reverse();
  detailStats.textContent = `${pane.EVENTS.length} events`;
  if (!events.length) {
    detailBody.innerHTML = '<div class="empty-detail">No allocator history events were recorded.</div>';
    return;
  }
  const overflowWarning = pane.ALLOCATOR_SETTINGS.trace_alloc_overflowed
    ? '<div class="peak-label" style="color:#e74c3c">Warning: allocator history overflowed; older events were overwritten.</div>'
    : '';
  detailBody.innerHTML = overflowWarning + '<div class="peak-label">Allocator state history (newest first)</div>' + events.map(event => {
    const oom = event.act === 'oom' ? `, device free ${formatBytes(event.device_free ?? 0)}` : '';
    return `<div class="breakdown-row" data-action="show-event-stack" data-side="${activeSide}" data-stack-idx="${event.si}" data-label="${encodeURIComponent(event.act)}">
      <span class="bd-size">${formatBytes(event.s)}</span>
      <span class="bd-count">t${event.step}</span>
      <span class="bd-pct">${formatBytes(event.a)}</span>
      <span class="bd-frame">${escapeHtml(event.act)} ${escapeHtml(event.addr ?? '')}, reserved ${formatBytes(event.r)}${oom}</span>
    </div>`;
  }).join('');
}

function showSegments() {
  const pane = panes[activeSide];
  detailStats.textContent = `${pane.SEGMENTS.length} segments`;
  if (!pane.SEGMENTS.length) {
    detailBody.innerHTML = '<div class="empty-detail">No current allocator segments.</div>';
    return;
  }
  detailBody.innerHTML = '<div class="peak-label">Current cached segment state</div>' + pane.SEGMENTS.map(segment => {
    const activeBlocks = segment.blocks.filter(block => block.state !== 'inactive').length;
    return `<div class="breakdown-row">
      <span class="bd-size">${formatBytes(segment.total_size)}</span>
      <span class="bd-count">${activeBlocks}/${segment.blocks.length}</span>
      <span class="bd-pct">${formatBytes(segment.allocated_size)}</span>
      <span class="bd-frame">${escapeHtml(segment.address)}, pool ${escapeHtml(formatPool(segment.pool))}, stream ${escapeHtml(segment.stream)}, ${escapeHtml(segment.segment_type)}${segment.expandable ? ', expandable' : ''}</span>
    </div>`;
  }).join('');
}

function showSettings() {
  const pane = panes[activeSide];
  detailStats.textContent = 'allocator settings';
  detailBody.innerHTML = `<pre class="detail-json">${escapeHtml(JSON.stringify(pane.ALLOCATOR_SETTINGS, null, 2))}</pre>`;
}

// Detail view system
const detailViews = [
  { id: 'stack', label: 'Stack', render: renderStackSelection, hasLR: false },
  { id: 'details', label: 'Details', render: showDetails, hasLR: false },
  { id: 'peak', label: 'Peak', render: showPeakBreakdown, hasLR: true },
  { id: 'leaks', label: 'Leaks', render: showLeaks, hasLR: true },
  { id: 'pools', label: 'Pools', render: showPools, hasLR: true },
  { id: 'events', label: 'Events', render: showEvents, hasLR: true },
  { id: 'segments', label: 'Segments', render: showSegments, hasLR: true },
  { id: 'settings', label: 'Settings', render: showSettings, hasLR: true },
];
const detailViewById = Object.fromEntries(detailViews.map(v => [v.id, v]));

function updateActiveDetailTab() {
  detailTabs.querySelectorAll('.detail-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.tab === uiState.activeDetailView);
  });
  const view = detailViewById[uiState.activeDetailView];
  lrToggle.style.display = view.hasLR ? 'flex' : 'none';
  lrToggle.querySelectorAll('.lr-btn').forEach(btn => {
    btn.classList.remove('active-left', 'active-right');
    if (btn.dataset.side === activeSide) btn.classList.add(activeSide === 'left' ? 'active-left' : 'active-right');
  });
}

function renderDetailTabs() {
  detailTabs.innerHTML = detailViews
    .map(v => `<button class="detail-tab" data-tab="${v.id}">${v.label}</button>`)
    .join('');
  updateActiveDetailTab();
}

function renderActiveDetailView() {
  detailViewById[uiState.activeDetailView].render();
}

function activateDetailView(viewId, { resetSearch = false } = {}) {
  uiState.activeDetailView = viewId;
  updateActiveDetailTab();
  if (resetSearch) applySearch('');
  renderActiveDetailView();
}

function handleAllocationSelection() {
  setDetailPanelCollapsed(false);
  activateDetailView(uiState.activeDetailView === 'details' ? 'details' : 'stack');
}

renderDetailTabs();

detailTabs.addEventListener('click', function(event) {
  const tab = event.target.closest('.detail-tab');
  if (!tab) return;
  activateDetailView(tab.dataset.tab, { resetSearch: tab.dataset.tab === 'stack' });
});

lrToggle.addEventListener('click', function(event) {
  const btn = event.target.closest('.lr-btn');
  if (!btn) return;
  activeSide = btn.dataset.side;
  updateActiveDetailTab();
  renderActiveDetailView();
});

detailBody.addEventListener('click', function(event) {
  const row = event.target.closest('.breakdown-row');
  if (!row) return;
  if (row.dataset.action === 'show-stack' || row.dataset.action === 'show-event-stack') {
    const side = row.dataset.side || activeSide;
    uiState.selectedSide = side;
    uiState.selectedStackIdx = Number(row.dataset.stackIdx);
    uiState.selectedStackLabel = `${side === 'left' ? 'L' : 'R'}: ${decodeURIComponent(row.dataset.label)}`;
    activateDetailView('stack');
    return;
  }
  if (row.dataset.action === 'apply-search') {
    if (useRegex) {
      useRegex = false;
      regexToggleEl.classList.remove('active');
    }
    applySearch(decodeURIComponent(row.dataset.query));
  }
});

// --- Panel toggle & resize ---
const detailPanel = document.getElementById('detail-panel');
const panelToggle = document.getElementById('panel-toggle');
const resizeHandle = document.getElementById('resize-handle');
const traceToggleGroup = document.getElementById('trace-toggle-group');
const leftEl = document.getElementById('chart-left');
const rightEl = document.getElementById('chart-right');
const leftMinimapEl = document.getElementById('minimap-left');
const rightMinimapEl = document.getElementById('minimap-right');

function setDetailPanelCollapsed(collapsed) {
  detailPanel.classList.toggle('collapsed', collapsed);
  panelToggle.textContent = collapsed ? '\u25B6' : '\u25C0';
  panelToggle.title = collapsed ? 'Show stack/details' : 'Hide stack/details';
  panelToggle.setAttribute('aria-expanded', String(!collapsed));
  setTimeout(applyPaneLayout, 200);
}

panelToggle.addEventListener('click', () => {
  setDetailPanelCollapsed(!detailPanel.classList.contains('collapsed'));
});

let panelResizing = false;
resizeHandle.addEventListener('pointerdown', (e) => {
  panelResizing = true;
  resizeHandle.classList.add('dragging');
  resizeHandle.setPointerCapture(e.pointerId);
  e.preventDefault();
});
document.addEventListener('pointermove', (e) => {
  if (!panelResizing) return;
  const newW = Math.max(200, window.innerWidth - e.clientX);
  detailPanel.style.width = newW + 'px';
  detailPanel.style.minWidth = newW + 'px';
});
document.addEventListener('pointerup', () => {
  if (!panelResizing) return;
  panelResizing = false;
  resizeHandle.classList.remove('dragging');
  applyPaneLayout();
});

function syncTraceToggleButtons() {
  traceToggleGroup.querySelectorAll('.trace-toggle-btn').forEach(btn => {
    const side = btn.dataset.traceSide;
    const isVisible = visiblePanes[side];
    btn.classList.toggle('inactive', !isVisible);
    btn.classList.toggle('active-left', side === 'left' && isVisible);
    btn.classList.toggle('active-right', side === 'right' && isVisible);
  });
}

function applyPaneLayout() {
  const showLeft = visiblePanes.left;
  const showRight = visiblePanes.right;

  leftEl.style.display = showLeft ? '' : 'none';
  rightEl.style.display = showRight ? '' : 'none';
  leftMinimapEl.style.display = showLeft ? '' : 'none';
  rightMinimapEl.style.display = showRight ? '' : 'none';

  if (showLeft && showRight) {
    leftEl.style.flex = '1';
    rightEl.style.flex = '1';
  } else if (showLeft) {
    leftEl.style.flex = '1';
  } else {
    rightEl.style.flex = '1';
  }

  syncTraceToggleButtons();

  if (!visiblePanes[activeSide]) {
    activeSide = visiblePanes.left ? 'left' : 'right';
    updateActiveDetailTab();
  }

  if (visiblePanes.left) leftPane.resize();
  if (visiblePanes.right) rightPane.resize();
  requestAnimationFrame(() => {
    if (visiblePanes.left) leftPane.resize();
    if (visiblePanes.right) rightPane.resize();
  });
}

traceToggleGroup.addEventListener('click', function(event) {
  const btn = event.target.closest('.trace-toggle-btn');
  if (!btn) return;
  const side = btn.dataset.traceSide;
  if (visiblePanes.left && visiblePanes.right) {
    visiblePanes[side] = false;
  } else {
    visiblePanes.left = true;
    visiblePanes.right = true;
  }
  applyPaneLayout();
});

applyPaneLayout();
window.addEventListener('resize', applyPaneLayout);

// --- Search ---
const searchInput = document.getElementById('search-input');
const regexToggleEl = document.getElementById('regex-toggle');

regexToggleEl.addEventListener('click', () => {
  useRegex = !useRegex;
  regexToggleEl.classList.toggle('active', useRegex);
  applySearch(searchInput.value);
});

function applySearch(query) {
  searchInput.value = query;
  if (!query) {
    searchMatcher = null;
  } else if (useRegex) {
    try { searchMatcher = new RegExp(query, 'i'); } catch(e) { searchMatcher = null; }
  } else {
    const q = query.toLowerCase();
    searchMatcher = { test: (s) => s.toLowerCase().includes(q) };
  }
  leftPane.updateSearchCache();
  rightPane.updateSearchCache();
  leftPane.drawCanvas();
  rightPane.drawCanvas();
}

searchInput.addEventListener('input', (e) => applySearch(e.target.value));

document.addEventListener('keydown', function(event) {
  if (event.key === '/' && event.target.tagName !== 'INPUT') {
    event.preventDefault();
    searchInput.focus();
  }
  if (event.key === 'Escape' && event.target === searchInput) {
    searchInput.value = '';
    applySearch('');
    searchInput.blur();
  }
});

// --- Shared controls ---
const controlsShell = document.getElementById('controls-shell');
const controls = document.getElementById('controls');
const controlsToggle = document.getElementById('controls-toggle');

function setControlsCollapsed(collapsed) {
  controls.classList.toggle('collapsed', collapsed);
  controlsToggle.textContent = collapsed ? '◀' : '▶';
  controlsToggle.title = collapsed ? 'Show controls' : 'Hide controls';
  controlsToggle.setAttribute('aria-expanded', String(!collapsed));
}

controlsToggle.addEventListener('click', event => {
  event.stopPropagation();
  setControlsCollapsed(!controls.classList.contains('collapsed'));
});
document.addEventListener('pointerdown', event => {
  if (!controlsShell.contains(event.target)) setControlsCollapsed(true);
});

document.getElementById('hwm-toggle').onchange = function() {
  leftPane.setHwmVisible(this.checked);
  rightPane.setHwmVisible(this.checked);
};

document.getElementById('reserved-toggle').onchange = function() {
  showReserved = this.checked;
  leftPane.setReservedVisible();
  rightPane.setReservedVisible();
};

document.getElementById('autofit-toggle').onchange = function() {
  const mode = this.checked ? 'autofit' : 'fixed';
  leftPane.setYMode(mode);
  rightPane.setYMode(mode);
};

document.getElementById('dim-persistent-toggle').onchange = function() {
  dimPersistent = this.checked;
  leftPane.setDimPersistent(dimPersistent);
  rightPane.setDimPersistent(dimPersistent);
};

document.getElementById('color-mode').onchange = function() {
  colorMode = this.value;
  leftPane.recolorAllocs();
  rightPane.recolorAllocs();
  leftPane.drawCanvas();
  rightPane.drawCanvas();
};

document.getElementById('settings-trigger').addEventListener('click', function(e) {
  if (e.target.closest('#settings-dropdown')) return;
  this.classList.toggle('open');
});

// --- WASD navigation ---
const activeKeys = new Set();
let animating = false;
const SPEEDS = [
  { pan: 0.005, zoom: 1.01 },
  { pan: 0.01,  zoom: 1.025 },
  { pan: 0.02,  zoom: 1.05 },
  { pan: 0.04,  zoom: 1.08 },
  { pan: 0.08,  zoom: 1.12 },
];
let speedIdx = 2;

function navTick() {
  if (activeKeys.size === 0) { animating = false; return; }
  const [f0, f1] = leftPane.getTransformFrac();
  const range = f1 - f0;
  let newF0 = f0, newF1 = f1;
  const panAmt = SPEEDS[speedIdx].pan;
  const zoomFactor = SPEEDS[speedIdx].zoom;

  if (activeKeys.has('a') || activeKeys.has('arrowleft')) { newF0 -= panAmt; newF1 -= panAmt; }
  if (activeKeys.has('d') || activeKeys.has('arrowright')) { newF0 += panAmt; newF1 += panAmt; }
  if (activeKeys.has('w') || activeKeys.has('arrowup')) {
    const center = (newF0 + newF1) / 2;
    const half = range / 2 / zoomFactor;
    newF0 = center - half; newF1 = center + half;
  }
  if (activeKeys.has('s') || activeKeys.has('arrowdown')) {
    const center = (newF0 + newF1) / 2;
    const half = range / 2 * zoomFactor;
    newF0 = center - half; newF1 = center + half;
  }
  const newRange = Math.min(1, Math.max(1 / 2000, newF1 - newF0));
  if (newF0 < 0) { newF0 = 0; newF1 = newRange; }
  if (newF1 > 1) { newF1 = 1; newF0 = 1 - newRange; }
  leftPane.syncZoom(newF0, newF1);
  rightPane.syncZoom(newF0, newF1);
  requestAnimationFrame(navTick);
}

const shortcutBar = document.getElementById('shortcut-bar');
const speedIndicator = document.getElementById('speed-indicator');

document.addEventListener('keydown', function(event) {
  if (event.target.tagName === 'INPUT') return;
  const k = event.key.toLowerCase();

  if (k === '?') {
    shortcutBar.style.display = shortcutBar.style.display === 'none' ? 'flex' : 'none';
    return;
  }
  if (k === '[' || k === ']') {
    speedIdx = Math.max(0, Math.min(SPEEDS.length - 1, speedIdx + (k === ']' ? 1 : -1)));
    speedIndicator.textContent = speedIdx + 1;
    shortcutBar.style.display = 'flex';
    return;
  }

  if (!['a','d','w','s','arrowleft','arrowright','arrowup','arrowdown'].includes(k)) return;
  event.preventDefault();
  activeKeys.add(k);
  if (!animating) { animating = true; navTick(); }
});

document.addEventListener('keyup', function(event) {
  activeKeys.delete(event.key.toLowerCase());
});
</script>
</body>
</html>
"""
