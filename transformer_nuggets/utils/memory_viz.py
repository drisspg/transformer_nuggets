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
  .group-tag.torch { background: rgba(62, 147, 204, 0.15); color: #3E93CC; }
  .group-tag.cpp { background: rgba(189, 147, 249, 0.15); color: #bd93f9; }
  .group-tag.python { background: rgba(255, 255, 255, 0.06); color: var(--text-muted); }

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

  .alloc-poly { stroke: rgba(0,0,0,0.5); stroke-width: 0.5; cursor: pointer; transition: opacity 0.1s; }
  .alloc-poly:hover { stroke: rgba(255,255,255,0.8); stroke-width: 1; }

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
      <span>Stack Trace</span>
      <span class="detail-stats" id="detail-stats"></span>
    </div>
    <div id="detail-body">
      <div class="empty-detail">Click an allocation to inspect its stack trace</div>
    </div>
  </div>
</div>
<div id="shortcut-bar" style="display:none">
  <span><kbd>A</kbd><kbd>D</kbd> pan</span>
  <span><kbd>W</kbd><kbd>S</kbd> zoom</span>
  <div class="sep"></div>
  <span><kbd>[</kbd><kbd>]</kbd> speed: <span id="speed-indicator">3</span>/5</span>
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

function getColor(stackIdx) {
  return PALETTE[(CATEGORIES[stackIdx] || 0) % PALETTE.length];
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

function classifyFrame(frame) {
  if (!frame.includes(':') && frame.includes('::')) return 'cpp';
  if (frame.includes('/site-packages/torch/') || frame.includes('/torch/')) return 'torch';
  if (frame.includes('/lib/python') || frame.includes('/conda/') || frame.includes('lib/python')) return 'torch';
  return 'user';
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

const GROUP_LABELS = { user: 'Your Code', torch: 'PyTorch / Python', cpp: 'C++ Runtime' };

function renderStack(stackIdx, label) {
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

// HWM
const hwmG = chartArea.append('g').attr('class', 'hwm-group').style('pointer-events', 'none');
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
    showTooltip(event, [
      `<div class="tt-row"><span class="tt-label">Size:</span><span class="tt-value">${formatBytes(d.s)}</span></div>`,
      (STACKS[d.si]||[])[0] ? `<div class="tt-hint">${STACKS[d.si][0]}</div>` : '',
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
    t = t.translate(panPx, 0);
  if (activeKeys.has('d') || activeKeys.has('arrowright'))
    t = t.translate(-panPx, 0);
  if (activeKeys.has('w') || activeKeys.has('arrowup')) {
    const cx = width / 2;
    t = t.translate(cx, 0).scale(zoomFactor).translate(-cx, 0);
  }
  if (activeKeys.has('s') || activeKeys.has('arrowdown')) {
    const cx = width / 2;
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
</script>
</body>
</html>
"""
