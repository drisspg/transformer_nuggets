import html
import json


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
    if fn.endswith(".c") and name.startswith(("_Py", "Py", "pyrun", "pymain", "run_")):
        return True
    return False


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


def process_snapshot(
    snapshot: dict, device: int = 0
) -> tuple[list[dict], list[dict], list[str], list[list[int]], int, int]:
    traces = snapshot.get("device_traces", [])
    if device >= len(traces):
        return [], [], [], [], 0, 0

    frame_to_idx: dict[str, int] = {}
    frames: list[str] = []
    stack_to_idx: dict[tuple[int, ...], int] = {}
    stacks: list[list[int]] = []

    def intern_frame(f: str) -> int:
        if f not in frame_to_idx:
            frame_to_idx[f] = len(frames)
            frames.append(f)
        return frame_to_idx[f]

    id_cache: dict[int, int] = {}
    content_cache: dict[tuple, int] = {}

    def get_stack_idx(raw_frames: list[dict]) -> int:
        raw_id = id(raw_frames)
        if raw_id in id_cache:
            return id_cache[raw_id]
        content_key = tuple(
            (f.get("filename", ""), f.get("name", ""), f.get("line", 0)) for f in raw_frames
        )
        if content_key in content_cache:
            result = content_cache[content_key]
            id_cache[raw_id] = result
            return result
        extracted = _extract_frames(raw_frames)
        frame_indices = [intern_frame(f) for f in extracted]
        key = tuple(frame_indices)
        if key not in stack_to_idx:
            stack_to_idx[key] = len(stacks)
            stacks.append(frame_indices)
        result = stack_to_idx[key]
        id_cache[raw_id] = result
        content_cache[content_key] = result
        return result

    allocated = 0
    reserved = 0
    hwm = 0
    hwm_at_timestep = 0
    timeline: list[dict] = []

    current_stack: list[int] = []
    stack_pos: dict[int, int] = {}
    alloc_id_by_addr: dict[int, int] = {}

    alloc_polys: list[dict] = []
    timestep = 0

    for seg in snapshot.get("segments", []):
        if seg.get("device", 0) != device:
            continue
        for block in seg.get("blocks", []):
            if block.get("state", "") not in ("active_allocated", "active_pending_free"):
                continue
            size = block.get("size", 0)
            addr = block.get("addr", seg.get("address", 0))
            raw_frames = block.get("frames", [])
            if not raw_frames and "history" in block and block["history"]:
                raw_frames = block["history"][0].get("frames", [])
            si = get_stack_idx(raw_frames)
            offset = allocated
            allocated += size
            alloc_id = len(alloc_polys)
            alloc_polys.append(
                {
                    "si": si,
                    "s": size,
                    "ts": [timestep],
                    "offsets": [offset],
                    "addr": f"0x{addr:x}",
                    "stream": seg.get("stream", 0),
                    "time_us": 0,
                    "ctx": "",
                }
            )
            stack_pos[alloc_id] = len(current_stack)
            current_stack.append(alloc_id)
            alloc_id_by_addr[addr] = alloc_id
    if allocated > 0:
        timestep += 1
        if allocated > hwm:
            hwm = allocated
            hwm_at_timestep = 0
        timeline.append(
            {
                "t": 0,
                "a": allocated,
                "r": 0,
                "h": hwm,
                "act": "preexisting",
                "s": allocated,
                "si": 0,
            }
        )

    for i, entry in enumerate(traces[device]):
        action = entry.get("action", "")
        addr = entry.get("addr", 0)
        size = entry.get("size", 0)
        time_us = entry.get("time_us", i)
        si = get_stack_idx(entry.get("frames", []))

        match action:
            case "alloc":
                allocated += size
                offset = allocated - size
                alloc_id = len(alloc_polys)
                alloc_polys.append(
                    {
                        "si": si,
                        "s": size,
                        "ts": [timestep],
                        "offsets": [offset],
                        "addr": f"0x{addr:x}",
                        "stream": entry.get("stream", 0),
                        "time_us": time_us,
                        "ctx": entry.get("compile_context") or "",
                    }
                )
                stack_pos[alloc_id] = len(current_stack)
                current_stack.append(alloc_id)
                alloc_id_by_addr[addr] = alloc_id
                timestep += 1

            case "free_completed" | "free_requested":
                if addr not in alloc_id_by_addr:
                    continue
                allocated -= size
                freed_id = alloc_id_by_addr.pop(addr)
                poly = alloc_polys[freed_id]
                poly["ts"].append(timestep)
                poly["offsets"].append(poly["offsets"][-1])

                idx_in_stack = stack_pos.pop(freed_id, None)
                if idx_in_stack is not None:
                    current_stack.pop(idx_in_stack)
                    for j in range(idx_in_stack, len(current_stack)):
                        stack_pos[current_stack[j]] = j
                    for above_id in current_stack[idx_in_stack:]:
                        above = alloc_polys[above_id]
                        above["ts"].append(timestep)
                        above["offsets"].append(above["offsets"][-1])
                        above["ts"].append(timestep + 1)
                        above["offsets"].append(above["offsets"][-1] - size)

                timestep += 2

            case "segment_alloc":
                reserved += size
            case "segment_free":
                reserved -= size
            case _:
                pass

        if allocated > hwm:
            hwm = allocated
            hwm_at_timestep = timestep
        timeline.append(
            {
                "t": time_us,
                "a": allocated,
                "r": reserved,
                "h": hwm,
                "act": action,
                "s": size,
                "si": si,
            }
        )

    for alloc_id in current_stack:
        poly = alloc_polys[alloc_id]
        poly["ts"].append(timestep)
        poly["offsets"].append(poly["offsets"][-1])

    return timeline, alloc_polys, frames, stacks, timestep, hwm_at_timestep


def _json_for_html(data: object) -> str:
    return json.dumps(data).replace("<", r"\u003c")


def _build_memory_viz_data(snapshot: dict, device: int, title: str) -> dict:
    timeline, alloc_polys, frames, stacks, max_ts, hwm_timestep = process_snapshot(
        snapshot, device
    )
    max_at_time = [entry["a"] for entry in timeline]
    hwm = max((entry["h"] for entry in timeline), default=0)
    return {
        "timeline": max_at_time,
        "allocs": alloc_polys,
        "frames": frames,
        "stacks": stacks,
        "meta": {
            "title": title,
            "device": device,
            "num_events": len(timeline),
            "num_allocs": len(alloc_polys),
            "high_water_mark_bytes": hwm,
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
    )


def generate_memory_comparison_html(
    snapshot_left: dict,
    snapshot_right: dict,
    device: int = 0,
    title_left: str = "Left",
    title_right: str = "Right",
) -> str:
    doc_title = f"{title_left} vs {title_right}"
    return (
        _MEMORY_COMPARISON_TEMPLATE.replace("__DOCUMENT_TITLE__", html.escape(doc_title))
        .replace("__TITLE_LEFT__", html.escape(title_left))
        .replace("__TITLE_RIGHT__", html.escape(title_right))
        .replace(
            "__BOOTSTRAP_LEFT__",
            _json_for_html(_build_memory_viz_data(snapshot_left, device, title_left)),
        )
        .replace(
            "__BOOTSTRAP_RIGHT__",
            _json_for_html(_build_memory_viz_data(snapshot_right, device, title_right)),
        )
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

  #controls {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
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
    position: absolute;
    left: -24px;
    top: 50%;
    transform: translateY(-50%);
    width: 24px;
    height: 48px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-right: none;
    border-radius: 4px 0 0 4px;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 12px;
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
  <div id="controls">
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
            <option value="order">order</option>
          </select>
        </label>
      </div>
    </span>
  </div>
</div>
<div id="main">
  <div id="chart-container"></div>
  <div id="detail-panel">
    <div id="panel-toggle" title="Toggle detail panel">◀</div>
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
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const BOOTSTRAP = JSON.parse(document.getElementById('memory-viz-bootstrap').textContent);
const { timeline: TIMELINE, allocs: ALLOCS, frames: FRAMES, stacks: STACKS, meta: META } = BOOTSTRAP;

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

document.getElementById('peak-stat').textContent = formatBytes(META.high_water_mark_bytes);
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

let colorMode = 'stack';

function recolorAllocs() {
  let pIdx = 0;
  for (let i = 0; i < ALLOCS.length; i++) {
    const isPersistent = allocPersistent[i];
    switch (colorMode) {
      case 'size': allocColors[i] = getSizeColor(i); break;
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
    return `<span class="frame-cpp">${ns}::</span><span class="frame-basename">${funcName}</span>`;
  }
  if (hasColon) {
    const sp = frame.indexOf(' ', frame.lastIndexOf(':'));
    if (sp > 0) {
      const filePart = frame.substring(0, sp);
      const funcPart = frame.substring(sp + 1);
      const lastSlash = filePart.lastIndexOf('/');
      const basename = lastSlash >= 0 ? filePart.substring(lastSlash + 1) : filePart;
      const dir = lastSlash >= 0 ? filePart.substring(0, lastSlash + 1) : '';
      return `<span class="frame-file">${dir}</span><span class="frame-basename">${basename}</span> <span class="frame-func">${funcPart}</span>`;
    }
    return `<span class="frame-file">${frame}</span>`;
  }
  return frame;
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
const containerRect = container.getBoundingClientRect();
const margin = { top: 20, right: 60, bottom: 40, left: 80 };
const width = containerRect.width - margin.left - margin.right;
const height = containerRect.height - margin.top - margin.bottom;

// Canvas for allocation polygons (behind SVG)
const canvas = document.createElement('canvas');
canvas.id = 'alloc-canvas';
canvas.width = containerRect.width * devicePixelRatio;
canvas.height = containerRect.height * devicePixelRatio;
canvas.style.width = containerRect.width + 'px';
canvas.style.height = containerRect.height + 'px';
container.insertBefore(canvas, container.firstChild);
const ctx = canvas.getContext('2d');
ctx.scale(devicePixelRatio, devicePixelRatio);

// SVG for axes, grid, HWM, zoom overlay
const svg = d3.select('#chart-container').append('svg')
  .attr('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`)
  .attr('preserveAspectRatio', 'none');

const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

svg.append('defs').append('clipPath').attr('id', 'clip')
  .append('rect').attr('width', width).attr('height', height);

const xScale = d3.scaleLinear().domain([0, META.max_timestep]).range([0, width]);
const yScale = d3.scaleLinear().domain([0, META.high_water_mark_bytes * 1.05]).range([height, 0]);

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
  const allocIndicesByStack = Array.from({ length: STACKS.length }, () => []);
  const stackSummariesById = stackFrameLabels.map((frame, stackIdx) => ({
    si: stackIdx,
    frame,
    count: 0,
    totalBytes: 0,
    firstTs: Infinity,
    lastTs: -Infinity,
  }));
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
    const summary = stackSummariesById[alloc.si];

    allocIndicesByStack[alloc.si].push(ai);
    summary.count += 1;
    summary.totalBytes += alloc.s;
    summary.firstTs = Math.min(summary.firstTs, firstTs);
    summary.lastTs = Math.max(summary.lastTs, lastTs);

    if (firstTs <= peakTs && lastTs >= peakTs) {
      peakAllocIndices.push(ai);
      peakTotalBytes += alloc.s;
    }

    if (lastTs >= maxTs && firstTs > earlyThreshold) {
      leakAllocIndices.push(ai);
      leakTotalBytes += alloc.s;

      let group = leakGroupsByFrame.get(summary.frame);
      if (!group) {
        group = { frame: summary.frame, si: alloc.si, count: 0, totalBytes: 0 };
        leakGroupsByFrame.set(summary.frame, group);
      }
      group.count += 1;
      group.totalBytes += alloc.s;
    }
  }

  peakAllocIndices.sort((left, right) => ALLOCS[right].s - ALLOCS[left].s);

  return {
    stackFrameLabels,
    allocIndicesByStack,
    stackSummariesById,
    stackSummaries: stackSummariesById
      .filter(summary => summary.count > 0)
      .sort((left, right) => right.totalBytes - left.totalBytes),
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
const NUM_HIT_BUCKETS = Math.max(1, Math.min(2000, META.max_timestep));
const hitBucketSize = META.max_timestep / NUM_HIT_BUCKETS;
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
const fullYDomain = [0, META.high_water_mark_bytes * 1.05];
let customYDomain = null;

function getBaseYDomain(d0, d1) {
  if (yMode === 'autofit' || dimPersistent) {
    let minY = Infinity, maxY = 0;
    for (let ai = 0; ai < ALLOCS.length; ai++) {
      if (allocEnds[ai] < d0 || allocStarts[ai] > d1) continue;
      if (dimPersistent && allocPersistent[ai]) continue;
      const d = ALLOCS[ai];
      for (let i = 0; i < d.ts.length; i++) {
        if (d.ts[i] >= d0 && d.ts[i] <= d1) {
          minY = Math.min(minY, d.offsets[i]);
          maxY = Math.max(maxY, d.offsets[i] + d.s);
        }
      }
    }
    if (maxY === 0) { minY = 0; maxY = META.high_water_mark_bytes; }
    if (minY === Infinity) minY = 0;
    const pad = (maxY - minY) * 0.05;
    return [Math.max(0, minY - pad), maxY + pad];
  }
  return fullYDomain;
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
  hwmG.select('.hwm-line').attr('y1', hwmY).attr('y2', hwmY);
  hwmG.select('.hwm-label').attr('y', hwmY - 6);

  drawCanvas();
  for (const hook of chartUpdateHooks) hook();
}

function transformForDomain(d0, d1) {
  const range = d1 - d0;
  return d3.zoomIdentity.translate(-d0 * width / range, 0).scale(META.max_timestep / range);
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
    if (primary) lines.push(`<div class="tt-${info.userFrame ? 'user' : 'api'}">${primary}</div>`);
    if (secondary) lines.push(`<div class="tt-api">${secondary}</div>`);
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
  return t;
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

const settingsTrigger = document.getElementById('settings-trigger');
settingsTrigger.addEventListener('click', function(e) {
  if (e.target.closest('#settings-dropdown')) return;
  this.classList.toggle('open');
});

document.getElementById('hwm-toggle').onchange = function() {
  hwmG.style('display', this.checked ? null : 'none');
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
  const ts = d.time_us ? new Date(d.time_us / 1000).toLocaleString() : 'N/A';
  detailStats.textContent = formatBytes(d.s);
  detailBody.innerHTML = `<div class="alloc-details"><table>
    <tr><td>Size</td><td>${formatBytes(d.s)} (${d.s.toLocaleString()} bytes)</td></tr>
    <tr><td>Address</td><td>${d.addr || 'N/A'}</td></tr>
    <tr><td>Stream</td><td>${d.stream ?? 'N/A'}</td></tr>
    <tr><td>Timestamp</td><td>${ts}</td></tr>
    <tr><td>Compile ctx</td><td>${d.ctx || 'None'}</td></tr>
    <tr><td>Lifetime</td><td>ts ${d.ts[0]} \u2192 ${d.ts[d.ts.length - 1]}${d.ts[d.ts.length - 1] >= META.max_timestep ? ' (never freed)' : ''}</td></tr>
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
    const pct = (d.s / META.high_water_mark_bytes * 100).toFixed(1);
    const barW = (d.s / maxAliveSize * 100).toFixed(0);
    return `<div class="breakdown-row" data-action="show-stack" data-stack-idx="${d.si}" data-label="${encodeURIComponent(formatBytes(d.s))}">
      <span class="bd-size">${formatBytes(d.s)}</span>
      <span class="bd-pct">${pct}%</span>
      <span class="bd-bar"><span class="bd-bar-fill" style="width:${barW}%"></span></span>
      <span class="bd-frame">${derivedData.stackFrameLabels[d.si]}</span>
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
      <span class="bd-frame">${g.frame}</span>
    </div>`;
  }).join('');

  detailBody.innerHTML = html;
}



// --- Panel toggle & resize ---
const detailPanel = document.getElementById('detail-panel');
const panelToggle = document.getElementById('panel-toggle');
const resizeHandle = document.getElementById('resize-handle');

panelToggle.addEventListener('click', () => {
  const collapsed = detailPanel.classList.toggle('collapsed');
  panelToggle.textContent = collapsed ? '▶' : '◀';
  setTimeout(() => { updateChart(currentTransform); }, 200);
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
  updateChart(currentTransform);
});

// --- Detail panel tabs ---
const detailViews = [
  { id: 'stack', label: 'Stack Trace', render: renderStackSelection, selectionViewId: 'stack' },
  { id: 'details', label: 'Details', render: showDetails, selectionViewId: 'details' },
  { id: 'peak', label: 'At Peak', render: showPeakBreakdown, selectionViewId: 'stack' },
  { id: 'leaks', label: 'Leaks', render: showLeaks, selectionViewId: 'stack' },
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
  }
  renderActiveDetailView();
}

function handleAllocationSelection() {
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
  if (row.dataset.action === 'show-stack') {
    selectStack(Number(row.dataset.stackIdx), decodeURIComponent(row.dataset.label));
    activateDetailView('stack');
    return;
  }
  if (row.dataset.action === 'apply-search') {
    applySearch(decodeURIComponent(row.dataset.query));
  }
});

// --- Feature 4: Minimap ---
const minimapContainer = document.getElementById('minimap');
const minimapRect = minimapContainer.getBoundingClientRect();
const minimapW = minimapRect.width - 32;
const minimapH = 32;
const miniMargin = { left: 16, top: 4 };

const miniSvg = d3.select('#minimap').append('svg')
  .attr('viewBox', `0 0 ${minimapW + 32} ${minimapH + 8}`);

const miniG = miniSvg.append('g')
  .attr('transform', `translate(${miniMargin.left},${miniMargin.top})`);

const miniX = d3.scaleLinear().domain([0, META.max_timestep]).range([0, minimapW]);
const miniY = d3.scaleLinear().domain([0, META.high_water_mark_bytes * 1.05]).range([minimapH, 0]);

miniG.append('path')
  .datum(TIMELINE)
  .attr('class', 'minimap-area')
  .attr('d', d3.area()
    .x((d, i) => i * minimapW / TIMELINE.length)
    .y0(minimapH)
    .y1(d => miniY(d))
  );

const viewportRect = miniG.append('rect')
  .attr('class', 'minimap-viewport')
  .attr('y', 0)
  .attr('height', minimapH);

function updateMinimap() {
  const newX = currentTransform.rescaleX(xScale);
  const [d0, d1] = newX.domain();
  const x0 = miniX(Math.max(0, d0));
  const x1 = miniX(Math.min(META.max_timestep, d1));
  viewportRect.attr('x', x0).attr('width', Math.max(2, x1 - x0));
}

updateMinimap();
chartUpdateHooks.push(updateMinimap);

const miniDrag = d3.drag()
  .on('drag', function(event) {
    const dx = event.dx;
    const domainPerPx = META.max_timestep / minimapW;
    const shift = dx * domainPerPx;
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    const range = d1 - d0;
    const newD0 = Math.max(0, Math.min(META.max_timestep - range, d0 + shift));
    zoomRect.call(zoom.transform, transformForDomain(newD0, newD0 + range));
  });

viewportRect.call(miniDrag);

miniSvg.on('click', function(event) {
  const [mx] = d3.pointer(event, miniG.node());
  const clickTs = miniX.invert(mx);
  const newX = currentTransform.rescaleX(xScale);
  const [d0, d1] = newX.domain();
  const range = d1 - d0;
  const newD0 = Math.max(0, Math.min(META.max_timestep - range, clickTs - range / 2));
  zoomRect.transition().duration(300).call(zoom.transform, transformForDomain(newD0, newD0 + range));
});
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

  #controls {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
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
    position: absolute;
    left: -24px;
    top: 50%;
    transform: translateY(-50%);
    width: 24px;
    height: 48px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-right: none;
    border-radius: 4px 0 0 4px;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 12px;
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
    <span class="stat stat-left">Peak: <strong id="peak-stat-left"></strong></span>
    <span class="stat stat-left">Allocs: <strong id="allocs-stat-left"></strong></span>
    <span class="stat stat-right">Peak: <strong id="peak-stat-right"></strong></span>
    <span class="stat stat-right">Allocs: <strong id="allocs-stat-right"></strong></span>
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
  <div id="controls">
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
            <option value="order">order</option>
          </select>
        </label>
      </div>
    </span>
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
  <div id="detail-panel">
    <div id="panel-toggle" title="Toggle detail panel">&#9664;</div>
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
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const BOOTSTRAP_LEFT = JSON.parse(document.getElementById('bootstrap-left').textContent);
const BOOTSTRAP_RIGHT = JSON.parse(document.getElementById('bootstrap-right').textContent);

function formatBytes(b) {
  if (Math.abs(b) >= 1024**3) return (b / 1024**3).toFixed(2) + ' GiB';
  if (Math.abs(b) >= 1024**2) return (b / 1024**2).toFixed(1) + ' MiB';
  if (Math.abs(b) >= 1024)    return (b / 1024).toFixed(0) + ' KiB';
  return b + ' B';
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
    return `<span class="frame-cpp">${ns}::</span><span class="frame-basename">${funcName}</span>`;
  }
  if (hasColon) {
    const sp = frame.indexOf(' ', frame.lastIndexOf(':'));
    if (sp > 0) {
      const filePart = frame.substring(0, sp);
      const funcPart = frame.substring(sp + 1);
      const lastSlash = filePart.lastIndexOf('/');
      const basename = lastSlash >= 0 ? filePart.substring(lastSlash + 1) : filePart;
      const dir = lastSlash >= 0 ? filePart.substring(0, lastSlash + 1) : '';
      return `<span class="frame-file">${dir}</span><span class="frame-basename">${basename}</span> <span class="frame-func">${funcPart}</span>`;
    }
    return `<span class="frame-file">${frame}</span>`;
  }
  return frame;
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
let searchMatcher = null;
let useRegex = false;
let activeSide = 'left';

const uiState = {
  activeDetailView: 'stack',
  selectedAlloc: null,
  selectedSide: null,
  selectedStackIdx: -1,
  selectedStackLabel: '',
  selectedFrames: null,
  selectedStacks: null,
};

// ============================================================
// Chart pane factory
// ============================================================
function createChartPane(bootstrap, containerEl, minimapEl, paneId) {
  const { timeline: TIMELINE, allocs: ALLOCS, frames: FRAMES, stacks: STACKS, meta: META } = bootstrap;

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
  const NUM_HIT_BUCKETS = Math.max(1, Math.min(2000, META.max_timestep));
  const hitBucketSize = META.max_timestep / NUM_HIT_BUCKETS;
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
  let chartWidth = containerRect.width - margin.left - margin.right;
  let chartHeight = containerRect.height - margin.top - margin.bottom;

  const canvas = document.createElement('canvas');
  canvas.width = containerRect.width * devicePixelRatio;
  canvas.height = containerRect.height * devicePixelRatio;
  canvas.style.width = containerRect.width + 'px';
  canvas.style.height = containerRect.height + 'px';
  containerEl.insertBefore(canvas, containerEl.firstChild);
  const ctx = canvas.getContext('2d');
  ctx.scale(devicePixelRatio, devicePixelRatio);

  const svg = d3.select(containerEl).append('svg')
    .attr('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`)
    .attr('preserveAspectRatio', 'none');

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const clipRect = svg.append('defs').append('clipPath').attr('id', `clip-${paneId}`)
    .append('rect').attr('width', chartWidth).attr('height', chartHeight);

  const xScale = d3.scaleLinear().domain([0, META.max_timestep]).range([0, chartWidth]);
  const yScale = d3.scaleLinear().domain([0, META.high_water_mark_bytes * 1.05]).range([chartHeight, 0]);

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

  let currentTransform = d3.zoomIdentity;
  let hoveredAlloc = null;
  const fullYDomain = [0, META.high_water_mark_bytes * 1.05];
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
    if (yMode === 'autofit' || dimPersistent) {
      let minY = Infinity, maxY = 0;
      for (let ai = 0; ai < ALLOCS.length; ai++) {
        if (allocEnds[ai] < d0 || allocStarts[ai] > d1) continue;
        if (dimPersistent && allocPersistent[ai]) continue;
        const d = ALLOCS[ai];
        for (let i = 0; i < d.ts.length; i++) {
          if (d.ts[i] >= d0 && d.ts[i] <= d1) {
            minY = Math.min(minY, d.offsets[i]);
            maxY = Math.max(maxY, d.offsets[i] + d.s);
          }
        }
      }
      if (maxY === 0) { minY = 0; maxY = META.high_water_mark_bytes; }
      if (minY === Infinity) minY = 0;
      const pad = (maxY - minY) * 0.05;
      return [Math.max(0, minY - pad), maxY + pad];
    }
    return fullYDomain;
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
    hwmG.select('.hwm-line').attr('y1', hwmY).attr('y2', hwmY);
    hwmG.select('.hwm-label').attr('y', hwmY - 6);

    drawCanvas();
    for (const hook of chartUpdateHooks) hook();
  }

  function transformForDomain(d0, d1) {
    const range = d1 - d0;
    return d3.zoomIdentity.translate(-d0 * chartWidth / range, 0).scale(META.max_timestep / range);
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
        onZoomCallback(d0 / META.max_timestep, d1 / META.max_timestep);
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
      if (primary) lines.push(`<div class="tt-${info.userFrame ? 'user' : 'api'}">${primary}</div>`);
      if (secondary) lines.push(`<div class="tt-api">${secondary}</div>`);
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
  let mmW = mmRect.width - 16;
  const mmH = 32;
  const miniSvg = d3.select(minimapEl).append('svg')
    .attr('viewBox', `0 0 ${mmW + 16} ${mmH + 8}`);
  const miniG = miniSvg.append('g').attr('transform', 'translate(8,4)');
  const miniX = d3.scaleLinear().domain([0, META.max_timestep]).range([0, mmW]);
  const miniY = d3.scaleLinear().domain([0, META.high_water_mark_bytes * 1.05]).range([mmH, 0]);

  miniG.append('path')
    .datum(TIMELINE)
    .attr('class', 'minimap-area')
    .attr('d', d3.area().x((d, i) => i * mmW / TIMELINE.length).y0(mmH).y1(d => miniY(d)));

  const viewportRect = miniG.append('rect')
    .attr('class', 'minimap-viewport').attr('y', 0).attr('height', mmH);

  function updateMinimap() {
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    const x0 = miniX(Math.max(0, d0));
    const x1 = miniX(Math.min(META.max_timestep, d1));
    viewportRect.attr('x', x0).attr('width', Math.max(2, x1 - x0));
  }

  function resize() {
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
      d3.area().x((d, i) => i * mmW / TIMELINE.length).y0(mmH).y1(d => miniY(d))
    );

    updateChart(currentTransform);
  }

  updateMinimap();
  chartUpdateHooks.push(updateMinimap);

  const miniDrag = d3.drag().on('drag', function(event) {
    const domainPerPx = META.max_timestep / mmW;
    const shift = event.dx * domainPerPx;
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    const range = d1 - d0;
    const newD0 = Math.max(0, Math.min(META.max_timestep - range, d0 + shift));
    zoomRect.call(zoom.transform, transformForDomain(newD0, newD0 + range));
  });
  viewportRect.call(miniDrag);

  miniSvg.on('click', function(event) {
    const [mx] = d3.pointer(event, miniG.node());
    const clickTs = miniX.invert(mx);
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    const range = d1 - d0;
    const newD0 = Math.max(0, Math.min(META.max_timestep - range, clickTs - range / 2));
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
    derivedData,
    resolveStack,
    recolorAllocs,
    updateSearchCache,
    drawCanvas,
    updateChart,
    paneId,
    setYMode(mode) { yMode = mode; customYDomain = null; updateChart(currentTransform); },
    setDimPersistent(v) { customYDomain = null; updateChart(currentTransform); },
    setHwmVisible(v) { hwmG.style('display', v ? null : 'none'); },
    syncZoom(fracStart, fracEnd) {
      syncing = true;
      const d0 = fracStart * META.max_timestep;
      const d1 = fracEnd * META.max_timestep;
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
      return [d0 / META.max_timestep, d1 / META.max_timestep];
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
  const pane = panes[side];
  uiState.selectedAlloc = alloc;
  uiState.selectedSide = side;
  uiState.selectedStackIdx = alloc.si;
  uiState.selectedStackLabel = `${side === 'left' ? 'L' : 'R'}: ${formatBytes(alloc.s)}`;
  uiState.selectedFrames = pane.FRAMES;
  uiState.selectedStacks = pane.STACKS;
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
  const ts = d.time_us ? new Date(d.time_us / 1000).toLocaleString() : 'N/A';
  const pane = panes[uiState.selectedSide];
  detailStats.textContent = `${uiState.selectedSide === 'left' ? 'L' : 'R'}: ${formatBytes(d.s)}`;
  detailBody.innerHTML = `<div class="alloc-details"><table>
    <tr><td>Pane</td><td>${uiState.selectedSide === 'left' ? 'Left' : 'Right'}</td></tr>
    <tr><td>Size</td><td>${formatBytes(d.s)} (${d.s.toLocaleString()} bytes)</td></tr>
    <tr><td>Address</td><td>${d.addr || 'N/A'}</td></tr>
    <tr><td>Stream</td><td>${d.stream ?? 'N/A'}</td></tr>
    <tr><td>Timestamp</td><td>${ts}</td></tr>
    <tr><td>Lifetime</td><td>ts ${d.ts[0]} \u2192 ${d.ts[d.ts.length - 1]}${d.ts[d.ts.length - 1] >= pane.META.max_timestep ? ' (never freed)' : ''}</td></tr>
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
    const pct = (d.s / pane.META.high_water_mark_bytes * 100).toFixed(1);
    const barW = (d.s / maxAliveSize * 100).toFixed(0);
    return `<div class="breakdown-row" data-action="show-stack" data-side="${activeSide}" data-stack-idx="${d.si}" data-label="${encodeURIComponent(formatBytes(d.s))}">
      <span class="bd-size">${formatBytes(d.s)}</span>
      <span class="bd-pct">${pct}%</span>
      <span class="bd-bar"><span class="bd-bar-fill" style="width:${barW}%"></span></span>
      <span class="bd-frame">${pane.derivedData.stackFrameLabels[d.si]}</span>
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
      <span class="bd-frame">${g.frame}</span>
    </div>`;
  }).join('');
  detailBody.innerHTML = html;
}

// Detail view system
const detailViews = [
  { id: 'stack', label: 'Stack Trace', render: renderStackSelection, hasLR: false },
  { id: 'details', label: 'Details', render: showDetails, hasLR: false },
  { id: 'peak', label: 'At Peak', render: showPeakBreakdown, hasLR: true },
  { id: 'leaks', label: 'Leaks', render: showLeaks, hasLR: true },
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
  const viewId = uiState.activeDetailView;
  const sel = detailViewById[viewId];
  activateDetailView(sel.hasLR ? viewId : (viewId === 'details' ? 'details' : 'stack'));
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
  if (row.dataset.action === 'show-stack') {
    const side = row.dataset.side || activeSide;
    uiState.selectedSide = side;
    uiState.selectedStackIdx = Number(row.dataset.stackIdx);
    uiState.selectedStackLabel = `${side === 'left' ? 'L' : 'R'}: ${decodeURIComponent(row.dataset.label)}`;
    activateDetailView('stack');
    return;
  }
  if (row.dataset.action === 'apply-search') {
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

panelToggle.addEventListener('click', () => {
  const collapsed = detailPanel.classList.toggle('collapsed');
  panelToggle.textContent = collapsed ? '\u25B6' : '\u25C0';
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
document.getElementById('hwm-toggle').onchange = function() {
  leftPane.setHwmVisible(this.checked);
  rightPane.setHwmVisible(this.checked);
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
  newF0 = Math.max(0, newF0);
  newF1 = Math.min(1, newF1);
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
