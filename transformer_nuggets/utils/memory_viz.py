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


def _categorize_stack(frames: list[str]) -> str:
    for frame in frames:
        if "/site-packages/" not in frame and "lib/python" not in frame and "::" not in frame:
            return frame
    for frame in frames:
        if "/site-packages/" not in frame and "lib/python" not in frame:
            return frame
    return frames[0] if frames else "unknown"


def process_snapshot(
    snapshot: dict, device: int = 0
) -> tuple[list[dict], list[dict], list[list[str]], list[str], int]:
    traces = snapshot.get("device_traces", [])
    if device >= len(traces):
        return [], [], [], [], 0

    stack_to_idx: dict[tuple[str, ...], int] = {}
    stacks: list[list[str]] = []

    def get_stack_idx(frames: list[dict]) -> int:
        extracted = _extract_frames(frames)
        key = tuple(extracted)
        if key not in stack_to_idx:
            stack_to_idx[key] = len(stacks)
            stacks.append(extracted)
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

    categories = [_categorize_stack(stack) for stack in stacks]
    return timeline, alloc_polys, stacks, categories, timestep


def generate_memory_html(
    snapshot: dict,
    device: int = 0,
    title: str = "Memory Timeline",
) -> str:
    timeline, alloc_polys, stacks, categories, max_ts = process_snapshot(snapshot, device)
    hwm = max((p["h"] for p in timeline), default=0)
    hwm_timestep = next((i for i, p in enumerate(timeline) if p["a"] == hwm), 0)

    cat_to_idx: dict[str, int] = {}
    for cat in categories:
        if cat not in cat_to_idx:
            cat_to_idx[cat] = len(cat_to_idx)
    cat_indices = [cat_to_idx.get(c, 0) for c in categories]

    meta = {
        "title": title,
        "device": device,
        "num_events": len(timeline),
        "num_allocs": len(alloc_polys),
        "high_water_mark_bytes": hwm,
        "hwm_timestep": hwm_timestep,
        "num_categories": len(cat_to_idx),
        "max_timestep": max_ts,
    }

    return (
        _MEMORY_VIZ_TEMPLATE.replace("__TITLE__", title)
        .replace("__TIMELINE__", json.dumps(timeline))
        .replace("__ALLOCS__", json.dumps(alloc_polys))
        .replace("__STACKS__", json.dumps(stacks))
        .replace("__CATEGORIES__", json.dumps(cat_indices))
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

  #header h1 { font-size: 14px; font-weight: 500; font-family: var(--mono); letter-spacing: 0.03em; text-transform: uppercase; }

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
    padding: 8px 16px 16px;
    min-height: 0;
  }

  svg { width: 100%; height: 100%; }

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

  .alloc-poly { stroke: rgba(0,0,0,0.5); stroke-width: 0.5; cursor: pointer; transition: opacity 0.15s; }
  .alloc-poly:hover { stroke: rgba(255,255,255,0.8); stroke-width: 1; }
  .alloc-poly.dimmed { opacity: 0.08 !important; }
  .alloc-poly.highlighted { stroke: rgba(255,255,255,0.9); stroke-width: 1.5; }

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
  <div id="controls">
    <div style="display:flex;align-items:center;gap:0;">
      <input type="text" id="search-input" placeholder="/ search allocations...">
      <button id="regex-toggle" title="Toggle regex mode">.*</button>
    </div>
    <label class="toggle">
      <input type="checkbox" id="hwm-toggle" checked>
      High Water Mark
    </label>
    <span class="stat">Peak: <strong id="peak-stat"></strong></span>
    <span class="stat">Allocs: <strong id="allocs-stat"></strong></span>
    <span class="zoom-hint">scroll to zoom, drag to pan, dbl-click to reset, <kbd>?</kbd> shortcuts</span>
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
const STACKS = __STACKS__;
const CATEGORIES = __CATEGORIES__;
const META = __META__;

function formatBytes(b) {
  if (Math.abs(b) >= 1024**3) return (b / 1024**3).toFixed(2) + ' GiB';
  if (Math.abs(b) >= 1024**2) return (b / 1024**2).toFixed(1) + ' MiB';
  if (Math.abs(b) >= 1024)    return (b / 1024).toFixed(0) + ' KiB';
  return b + ' B';
}

document.getElementById('peak-stat').textContent = formatBytes(META.high_water_mark_bytes);
document.getElementById('allocs-stat').textContent = META.num_allocs.toLocaleString();

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
  const stack = STACKS[stackIdx] || [];
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
  const stack = STACKS[stackIdx] || [];
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

// Build polygon points for each allocation
// Each alloc has ts[] (timesteps) and offsets[] (bottom y at each timestep)
// Polygon: bottom-left → bottom-right → top-right → top-left
function buildPolygonPoints(d, xScale, yScale) {
  const ts = d.ts;
  const offsets = d.offsets;
  const size = d.s;
  const points = [];
  // Bottom edge: left to right
  for (let i = 0; i < ts.length; i++) {
    points.push([xScale(ts[i]), yScale(offsets[i])]);
  }
  // Top edge: right to left
  for (let i = ts.length - 1; i >= 0; i--) {
    points.push([xScale(ts[i]), yScale(offsets[i] + size)]);
  }
  return points.map(p => p.join(',')).join(' ');
}

// --- Chart setup ---
const container = document.getElementById('chart-container');
const rect = container.getBoundingClientRect();
const margin = { top: 20, right: 60, bottom: 40, left: 80 };
const width = rect.width - margin.left - margin.right;
const height = rect.height - margin.top - margin.bottom;

const svg = d3.select('#chart-container').append('svg')
  .attr('viewBox', `0 0 ${rect.width} ${rect.height}`);

const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

svg.append('defs').append('clipPath').attr('id', 'clip')
  .append('rect').attr('width', width).attr('height', height);

const xScale = d3.scaleLinear().domain([0, META.max_timestep]).range([0, width]);
const yScale = d3.scaleLinear().domain([0, META.high_water_mark_bytes * 1.05]).range([height, 0]);

const xAxis = d3.axisBottom(xScale).ticks(10);
const yAxisFn = d3.axisLeft(yScale).ticks(8).tickFormat(d => formatBytes(d));

g.append('g').attr('class', 'grid')
  .call(d3.axisLeft(yScale).ticks(8).tickSize(-width).tickFormat(''));

const xAxisG = g.append('g').attr('class', 'axis x-axis')
  .attr('transform', `translate(0,${height})`).call(xAxis);

g.append('g').attr('class', 'axis y-axis').call(yAxisFn);

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

// Draw allocation polygons
const polysG = chartArea.append('g');

polysG.selectAll('.alloc-poly')
  .data(ALLOCS)
  .join('polygon')
  .attr('class', 'alloc-poly')
  .attr('points', d => buildPolygonPoints(d, xScale, yScale))
  .attr('fill', d => getColor(d.si))
  .attr('opacity', 0.85)
  .on('mousemove', function(event, d) {
    const bf = bestFrame(d.si);
    showTooltip(event, [
      `<div class="tt-row"><span class="tt-label">Size:</span><span class="tt-value">${formatBytes(d.s)}</span></div>`,
      bf ? `<div class="tt-hint">${bf}</div>` : '',
    ].join(''));
  })
  .on('mouseleave', hideTooltip)
  .on('click', function(event, d) {
    renderStack(d.si, formatBytes(d.s));
  });

// Zoom
let currentTransform = d3.zoomIdentity;

function updateChart(transform) {
  currentTransform = transform;
  const newX = transform.rescaleX(xScale);
  xAxisG.call(xAxis.scale(newX));
  xAxisG.selectAll('text').attr('fill', 'var(--text-muted)');
  xAxisG.selectAll('line, path').attr('stroke', 'var(--border)');

  polysG.selectAll('.alloc-poly')
    .attr('points', d => buildPolygonPoints(d, newX, yScale));
}

const zoom = d3.zoom()
  .scaleExtent([1, 2000])
  .translateExtent([[0, 0], [width, height]])
  .extent([[0, 0], [width, height]])
  .on('zoom', (event) => updateChart(event.transform));

const zoomRect = chartArea.append('rect')
  .attr('width', width).attr('height', height)
  .attr('fill', 'none').attr('pointer-events', 'all')
  .call(zoom);

// Raise polys above the zoom rect
polysG.raise();

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
  let matcher;
  if (!query) {
    matcher = null;
  } else if (useRegex) {
    try { matcher = new RegExp(query, 'i'); } catch(e) { matcher = null; }
  } else {
    const q = query.toLowerCase();
    matcher = { test: (s) => s.toLowerCase().includes(q) };
  }

  polysG.selectAll('.alloc-poly').each(function(d) {
    const el = d3.select(this);
    if (!matcher) {
      el.classed('dimmed', false).classed('highlighted', false);
      return;
    }
    const stack = STACKS[d.si] || [];
    const match = stack.some(f => matcher.test(f));
    el.classed('dimmed', !match).classed('highlighted', match);
  });
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

const miniArea = d3.area()
  .x((d, i) => miniX(i))
  .y0(minimapH)
  .y1(d => miniY(d));

const allocatedAtTimestep = new Float64Array(META.max_timestep + 1);
for (const t of TIMELINE) {
  if (t.act === 'alloc' || t.act === 'free_completed') {
    const ts = ALLOCS.length > 0 ? Math.round(miniX.invert(miniX(0))) : 0;
  }
}

let runningAlloc = 0;
let tsIdx = 0;
for (const t of TIMELINE) {
  if (t.act === 'alloc' || t.act === 'free_completed' || t.act === 'segment_alloc' || t.act === 'segment_free') {
    allocatedAtTimestep[tsIdx] = t.a;
  } else {
    allocatedAtTimestep[tsIdx] = tsIdx > 0 ? allocatedAtTimestep[tsIdx - 1] : 0;
  }
  tsIdx++;
}

const miniData = [];
const step = Math.max(1, Math.floor(tsIdx / minimapW));
for (let i = 0; i < tsIdx; i += step) {
  miniData.push(allocatedAtTimestep[i]);
}

miniG.append('path')
  .datum(miniData)
  .attr('class', 'minimap-area')
  .attr('d', d3.area()
    .x((d, i) => i * minimapW / miniData.length)
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
