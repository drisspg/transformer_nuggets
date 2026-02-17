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


def _is_cpython_c_frame(fn: str, name: str) -> bool:
    if any(m in fn for m in _CPYTHON_MARKERS):
        return True
    if fn.endswith(".c") and name.startswith(("_Py", "Py", "pyrun", "pymain", "run_")):
        return True
    return False


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
        elif name:
            result.append(name)
    return result


def process_snapshot(
    snapshot: dict, device: int = 0
) -> tuple[list[dict], list[dict], list[str], list[list[int]], int]:
    traces = snapshot.get("device_traces", [])
    if device >= len(traces):
        return [], [], [], [], 0

    frame_to_idx: dict[str, int] = {}
    frames: list[str] = []
    stack_to_idx: dict[tuple[int, ...], int] = {}
    stacks: list[list[int]] = []

    def intern_frame(f: str) -> int:
        if f not in frame_to_idx:
            frame_to_idx[f] = len(frames)
            frames.append(f)
        return frame_to_idx[f]

    def get_stack_idx(raw_frames: list[dict]) -> int:
        extracted = _extract_frames(raw_frames)
        frame_indices = [intern_frame(f) for f in extracted]
        key = tuple(frame_indices)
        if key not in stack_to_idx:
            stack_to_idx[key] = len(stacks)
            stacks.append(frame_indices)
        return stack_to_idx[key]

    allocated = 0
    reserved = 0
    hwm = 0
    timeline: list[dict] = []
    last_time = 0

    current_stack: list[int] = []
    alloc_id_by_addr: dict[int, int] = {}

    alloc_polys: list[dict] = []
    timestep = 0

    for i, entry in enumerate(traces[device]):
        action = entry.get("action", "")
        addr = entry.get("addr", 0)
        size = entry.get("size", 0)
        time_us = entry.get("time_us", i)
        last_time = max(last_time, time_us)
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
                    }
                )
                current_stack.append(alloc_id)
                alloc_id_by_addr[addr] = alloc_id
                timestep += 1

            case "free_completed":
                allocated -= size
                if addr in alloc_id_by_addr:
                    freed_id = alloc_id_by_addr.pop(addr)
                    poly = alloc_polys[freed_id]
                    poly["ts"].append(timestep)
                    poly["offsets"].append(poly["offsets"][-1])

                    if freed_id in current_stack:
                        idx_in_stack = current_stack.index(freed_id)
                        current_stack.pop(idx_in_stack)
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
            case "snapshot":
                continue
            case _:
                pass

        hwm = max(hwm, allocated)
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

    return timeline, alloc_polys, frames, stacks, timestep


def generate_memory_html(
    snapshot: dict,
    device: int = 0,
    title: str = "Memory Timeline",
) -> str:
    timeline, alloc_polys, frames, stacks, max_ts = process_snapshot(snapshot, device)
    hwm = max((p["h"] for p in timeline), default=0)
    hwm_timestep = next((i for i, p in enumerate(timeline) if p["a"] == hwm), 0)
    max_at_time = [e["a"] for e in timeline]

    meta = {
        "title": title,
        "device": device,
        "num_events": len(timeline),
        "num_allocs": len(alloc_polys),
        "high_water_mark_bytes": hwm,
        "hwm_timestep": hwm_timestep,
        "max_timestep": max_ts,
    }

    return (
        _MEMORY_VIZ_TEMPLATE.replace("__TITLE__", title)
        .replace("__TIMELINE__", json.dumps(max_at_time))
        .replace("__ALLOCS__", json.dumps(alloc_polys))
        .replace("__FRAMES__", json.dumps(frames))
        .replace("__STACKS__", json.dumps(stacks))
        .replace("__META__", json.dumps(meta))
    )


_MEMORY_VIZ_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
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

  svg { width: 100%; height: 100%; position: relative; z-index: 1; }

  #detail-panel {
    width: 480px;
    border-left: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
  }

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

  #detail-body {
    flex: 1;
    overflow-y: auto;
    padding: 0;
  }

  #detail-body::-webkit-scrollbar { width: 6px; }
  #detail-body::-webkit-scrollbar-track { background: transparent; }
  #detail-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .stack-group {
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }

  .stack-group-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
    background: rgba(255,255,255,0.02);
  }

  .stack-group-header:hover { background: rgba(255,255,255,0.04); }

  .stack-group-header .chevron {
    display: inline-block;
    width: 12px;
    font-size: 8px;
    transition: transform 0.15s;
    color: rgba(255,255,255,0.3);
  }

  .stack-group.collapsed .chevron { transform: rotate(-90deg); }
  .stack-group.collapsed .stack-group-frames { display: none; }

  .stack-group-header .group-count {
    margin-left: auto;
    font-size: 9px;
    color: rgba(255,255,255,0.25);
  }

  .stack-group-header .group-tag {
    padding: 1px 6px;
    border-radius: 2px;
    font-size: 9px;
  }

  .group-tag.user { background: rgba(73, 201, 99, 0.15); color: #49C963; }
  .group-tag.internal { background: rgba(255, 255, 255, 0.06); color: var(--text-muted); }

  .stack-frame {
    padding: 3px 16px 3px 34px;
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.5;
    color: var(--text-muted);
    cursor: pointer;
    overflow: hidden;
  }

  .stack-frame .frame-text {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
  }

  .stack-frame.expanded .frame-text {
    white-space: pre-wrap;
    word-break: break-all;
  }

  .stack-frame:hover { background: rgba(255,255,255,0.03); color: var(--text); }

  .stack-frame .frame-cpp { color: #bd93f9; }
  .stack-frame .frame-file { color: #3E93CC; }
  .stack-frame .frame-func { color: #49C963; }
  .stack-frame .frame-basename { color: var(--text); }

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

  .zoom-hint { font-size: 11px; color: var(--text-muted); opacity: 0.5; }

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
  <h1>__TITLE__</h1>
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
  </div>
</div>
<div id="main">
  <div id="chart-container"></div>
  <div id="detail-panel">
    <div id="detail-header">
      <div class="detail-tabs">
        <button class="detail-tab active" data-tab="stack">Stack Trace</button>
        <button class="detail-tab" data-tab="breakdown">Breakdown</button>
        <button class="detail-tab" data-tab="peak">At Peak</button>
      </div>
      <span class="detail-stats" id="detail-stats"></span>
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

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const TIMELINE = __TIMELINE__;
const ALLOCS = __ALLOCS__;
const FRAMES = __FRAMES__;
const STACKS = __STACKS__;
const META = __META__;

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

const PALETTE = [
  '#3E93CC', '#2E7DB5', '#5BA8D9', '#78BBE3',
  '#49C963', '#3AA852', '#6DD883', '#8DE49D',
  '#bd93f9', '#a06eed', '#d4b5ff', '#9054e0',
  '#CC6B3E', '#E08A5B', '#B55A30', '#F0A478',
  '#3ECCC1', '#5BD9D0', '#2EB5AC', '#78E3DC',
  '#C9CC3E', '#D9DB5B', '#B5B72E', '#E3E478',
];

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function getColor(stackIdx) {
  const frame = bestFrame(stackIdx);
  return PALETTE[hashStr(frame) % PALETTE.length];
}

const tooltipEl = document.getElementById('tooltip');
const detailBody = document.getElementById('detail-body');
const detailStats = document.getElementById('detail-stats');

function showTooltip(event, html) {
  tooltipEl.innerHTML = html;
  tooltipEl.style.display = 'block';
  const tw = tooltipEl.offsetWidth, th = tooltipEl.offsetHeight;
  tooltipEl.style.left = (event.pageX + 16 + tw > window.innerWidth ? event.pageX - tw - 12 : event.pageX + 16) + 'px';
  tooltipEl.style.top = (event.pageY + 16 + th > window.innerHeight ? event.pageY - th - 12 : event.pageY + 16) + 'px';
}

function hideTooltip() { tooltipEl.style.display = 'none'; }

let lastStackIdx = -1;
let lastStackLabel = '';

function classifyFrame(frame) {
  if (frame.includes('::')) return 'internal';
  if (frame.includes('/site-packages/') || frame.includes('/torch/')) return 'internal';
  if (frame.includes('/lib/python') || frame.includes('/conda/') || frame.includes('lib/python')) return 'internal';
  if (frame.includes('.cpp:') || frame.includes('.c:')) return 'internal';
  return 'user';
}

function bestFrame(stackIdx) {
  const stack = resolveStack(stackIdx);
  for (const f of stack) {
    if (classifyFrame(f) === 'user') return f;
  }
  for (const f of stack) {
    if (f.includes('.py')) return f;
  }
  return stack[stack.length - 1] || '';
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

const GROUP_LABELS = { user: 'Your Code', internal: 'Internals' };

function renderStack(stackIdx, label) {
  lastStackIdx = stackIdx;
  lastStackLabel = label;
  const stack = resolveStack(stackIdx);
  detailStats.textContent = label;
  if (!stack.length) {
    detailBody.innerHTML = '<div class="empty-detail">No frames recorded</div>';
    return;
  }

  const groups = [];
  let cur = null;
  for (const frame of stack) {
    const cls = classifyFrame(frame);
    if (!cur || cur.cls !== cls) {
      cur = { cls, frames: [] };
      groups.push(cur);
    }
    cur.frames.push(frame);
  }

  detailBody.innerHTML = groups.map(g => {
    const collapsed = g.cls !== 'user' ? ' collapsed' : '';
    const framesHtml = g.frames.map(f =>
      `<div class="stack-frame" onclick="event.stopPropagation();this.classList.toggle('expanded')"><span class="frame-text">${renderFrame(f)}</span></div>`
    ).join('');
    return `<div class="stack-group${collapsed}">
      <div class="stack-group-header" onclick="this.parentElement.classList.toggle('collapsed')">
        <span class="chevron">▼</span>
        <span class="group-tag ${g.cls}">${GROUP_LABELS[g.cls]}</span>
        <span class="group-count">${g.frames.length}</span>
      </div>
      <div class="stack-group-frames">${framesHtml}</div>
    </div>`;
  }).join('');
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
  .attr('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`);

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

// Precompute colors for each alloc
const allocColors = ALLOCS.map(d => getColor(d.si));

function drawCanvas() {
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

  for (let ai = 0; ai < ALLOCS.length; ai++) {
    const d = ALLOCS[ai];
    const tStart = d.ts[0];
    const tEnd = d.ts[d.ts.length - 1];
    if (tEnd < d0 || tStart > d1) continue;

    const visW = (Math.min(tEnd, d1) - Math.max(tStart, d0)) * pxPerTs;
    const visH = yScale(0) - yScale(d.s);
    if (visW < minVisPx && visH < minVisPx) continue;

    const ts = d.ts;
    const offsets = d.offsets;
    const size = d.s;

    ctx.beginPath();
    // Bottom edge left to right
    for (let i = 0; i < ts.length; i++) {
      const x = newX(ts[i]);
      const y = yScale(offsets[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    // Top edge right to left
    for (let i = ts.length - 1; i >= 0; i--) {
      ctx.lineTo(newX(ts[i]), yScale(offsets[i] + size));
    }
    ctx.closePath();

    let alpha = 0.85;
    if (searchMatcher) {
      const stack = resolveStack(d.si);
      alpha = stack.some(f => searchMatcher.test(f)) ? 0.9 : 0.06;
    }
    if (d === hoveredAlloc) alpha = 1.0;

    ctx.globalAlpha = alpha;
    ctx.fillStyle = allocColors[ai];
    ctx.fill();
    ctx.globalAlpha = d === hoveredAlloc ? 1.0 : 0.3;
    ctx.strokeStyle = d === hoveredAlloc ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.5)';
    ctx.lineWidth = d === hoveredAlloc ? 1.5 : 0.5;
    ctx.stroke();
  }

  ctx.restore();
}

// Hit testing: find allocation under mouse
function hitTest(mx, my) {
  const newX = currentTransform.rescaleX(xScale);
  const dataX = newX.invert(mx - margin.left);
  const dataY = yScale.invert(my - margin.top);

  let best = null;
  let bestSize = Infinity;

  for (const d of ALLOCS) {
    const tStart = d.ts[0];
    const tEnd = d.ts[d.ts.length - 1];
    if (dataX < tStart || dataX > tEnd) continue;

    // Find the offset at this timestep via linear search of keyframes
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
  if (yMode === 'autofit') {
    let maxY = 0;
    for (const d of ALLOCS) {
      const tStart = d.ts[0];
      const tEnd = d.ts[d.ts.length - 1];
      if (tEnd < d0 || tStart > d1) continue;
      for (let i = 0; i < d.ts.length; i++) {
        if (d.ts[i] >= d0 && d.ts[i] <= d1) {
          maxY = Math.max(maxY, d.offsets[i] + d.s);
        }
      }
    }
    if (maxY === 0) maxY = META.high_water_mark_bytes;
    return [0, maxY * 1.1];
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
}

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

  const newK = META.max_timestep / (dataX1 - dataX0);
  const newTx = -dataX0 * width / (dataX1 - dataX0);
  const t = d3.zoomIdentity.translate(newTx, 0).scale(newK);
  zoomRect.transition().duration(300).call(zoom.transform, t);
});

zoomRect.on('mousemove', function(event) {
  const [mx, my] = d3.pointer(event, svg.node());
  const hit = hitTest(mx, my);
  if (hit !== hoveredAlloc) {
    hoveredAlloc = hit;
    drawCanvas();
  }
  if (hit) {
    const bf = bestFrame(hit.si);
    showTooltip(event, [
      `<div class="tt-row"><span class="tt-label">Size:</span><span class="tt-value">${formatBytes(hit.s)}</span></div>`,
      bf ? `<div class="tt-hint">${bf}</div>` : '',
    ].join(''));
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
  if (hit) renderStack(hit.si, formatBytes(hit.s));
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

document.getElementById('hwm-toggle').onchange = function() {
  hwmG.style('display', this.checked ? null : 'none');
};

document.getElementById('autofit-toggle').onchange = function() {
  yMode = this.checked ? 'autofit' : 'fixed';
  customYDomain = null;
  updateChart(currentTransform);
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

// --- Feature 2: What's at Peak ---
function showPeakBreakdown() {
  const peakTs = META.hwm_timestep;
  const alive = ALLOCS.filter(d => d.ts[0] <= peakTs && d.ts[d.ts.length - 1] >= peakTs);
  alive.sort((a, b) => b.s - a.s);
  const total = alive.reduce((s, d) => s + d.s, 0);

  detailStats.textContent = `${alive.length} allocs, ${formatBytes(total)}`;

  let html = `<div class="peak-label">Allocations alive at peak (${formatBytes(META.high_water_mark_bytes)})</div>`;
  html += alive.map(d => {
    const pct = (d.s / META.high_water_mark_bytes * 100).toFixed(1);
    const bf = bestFrame(d.si);
    const barW = (d.s / alive[0].s * 100).toFixed(0);
    return `<div class="breakdown-row" onclick="renderStack(${d.si}, '${formatBytes(d.s)}'); setActiveTab('stack');">
      <span class="bd-size">${formatBytes(d.s)}</span>
      <span class="bd-pct">${pct}%</span>
      <span class="bd-bar"><span class="bd-bar-fill" style="width:${barW}%"></span></span>
      <span class="bd-frame">${bf}</span>
    </div>`;
  }).join('');
  detailBody.innerHTML = html;
}

hwmG.on('click', function() {
  setActiveTab('peak');
  showPeakBreakdown();
});

// --- Feature 3: Memory Breakdown ---
function showBreakdown() {
  const byFrame = {};
  for (const d of ALLOCS) {
    const f = bestFrame(d.si);
    if (!byFrame[f]) byFrame[f] = { frame: f, totalBytes: 0, count: 0, si: d.si };
    byFrame[f].totalBytes += d.s;
    byFrame[f].count += 1;
  }
  const rows = Object.values(byFrame).sort((a, b) => b.totalBytes - a.totalBytes);
  const maxBytes = rows[0]?.totalBytes || 1;
  const totalBytes = rows.reduce((s, r) => s + r.totalBytes, 0);

  detailStats.textContent = `${rows.length} call sites`;

  detailBody.innerHTML = rows.slice(0, 30).map(r => {
    const pct = (r.totalBytes / totalBytes * 100).toFixed(1);
    const barW = (r.totalBytes / maxBytes * 100).toFixed(0);
    return `<div class="breakdown-row" onclick="applySearch('${r.frame.replace(/'/g, "\\'")}')">
      <span class="bd-size">${formatBytes(r.totalBytes)}</span>
      <span class="bd-count">×${r.count}</span>
      <span class="bd-pct">${pct}%</span>
      <span class="bd-bar"><span class="bd-bar-fill" style="width:${barW}%"></span></span>
      <span class="bd-frame">${r.frame}</span>
    </div>`;
  }).join('');
}

// --- Detail panel tabs ---
function setActiveTab(tab) {
  document.querySelectorAll('.detail-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
}

document.querySelectorAll('.detail-tab').forEach(tab => {
  tab.addEventListener('click', function() {
    setActiveTab(this.dataset.tab);
    if (this.dataset.tab === 'breakdown') showBreakdown();
    else if (this.dataset.tab === 'peak') showPeakBreakdown();
    else {
      searchInput.value = '';
      applySearch('');
      if (lastStackIdx >= 0) renderStack(lastStackIdx, lastStackLabel);
      else detailBody.innerHTML = '<div class="empty-detail">Click an allocation to inspect its stack trace</div>';
    }
  });
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

const origUpdateChart = updateChart;
updateChart = function(transform) {
  origUpdateChart(transform);
  updateMinimap();
};

zoom.on('zoom', (event) => updateChart(event.transform));

const miniDrag = d3.drag()
  .on('drag', function(event) {
    const dx = event.dx;
    const domainPerPx = META.max_timestep / minimapW;
    const shift = dx * domainPerPx;
    const newX = currentTransform.rescaleX(xScale);
    const [d0, d1] = newX.domain();
    const range = d1 - d0;
    const newD0 = Math.max(0, Math.min(META.max_timestep - range, d0 + shift));
    const newK = META.max_timestep / range;
    const newTx = -newD0 * width / range;
    const t = d3.zoomIdentity.translate(newTx, 0).scale(newK);
    zoomRect.call(zoom.transform, t);
  });

viewportRect.call(miniDrag);

miniSvg.on('click', function(event) {
  const [mx] = d3.pointer(event, miniG.node());
  const clickTs = miniX.invert(mx);
  const newX = currentTransform.rescaleX(xScale);
  const [d0, d1] = newX.domain();
  const range = d1 - d0;
  const newD0 = Math.max(0, Math.min(META.max_timestep - range, clickTs - range / 2));
  const newK = META.max_timestep / range;
  const newTx = -newD0 * width / range;
  const t = d3.zoomIdentity.translate(newTx, 0).scale(newK);
  zoomRect.transition().duration(300).call(zoom.transform, t);
});
</script>
</body>
</html>
"""
