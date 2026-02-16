import json


def _extract_frames(frames: list[dict]) -> list[str]:
    result = []
    for f in frames:
        fn = f.get("filename", "")
        name = f.get("name", "")
        line = f.get("line", 0)
        if not name or name in (
            "torch::unwind::unwind()",
            "torch::CapturedTraceback::gather(bool, bool, bool)",
        ):
            continue
        if fn and fn != "??" and fn != "":
            result.append(f"{fn}:{line} {name}")
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
  :root {
    --bg: #0f1117;
    --surface: #1a1d2e;
    --border: #2a2d3e;
    --text: #e2e4e9;
    --text-muted: #8b8fa3;
    --accent: #6366f1;
    --accent-light: rgba(99, 102, 241, 0.15);
    --accent-stroke: rgba(99, 102, 241, 0.8);
    --hwm-color: #f59e0b;
    --grid: rgba(255, 255, 255, 0.04);
    --tooltip-bg: #1e2235;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    --mono: 'SF Mono', 'Fira Code', 'Consolas', monospace;
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

  #header h1 { font-size: 16px; font-weight: 600; }

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
    font-size: 12px;
    color: var(--text-muted);
    padding: 4px 10px;
    background: var(--surface);
    border-radius: 4px;
    border: 1px solid var(--border);
  }

  .stat strong { color: var(--text); font-weight: 600; }

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
    font-size: 13px;
    font-weight: 600;
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

  .stack-frame {
    padding: 4px 16px;
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.6;
    color: var(--text-muted);
    border-bottom: 1px solid rgba(255,255,255,0.02);
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

  .stack-frame .frame-idx {
    display: inline-block; width: 24px;
    color: rgba(255,255,255,0.15); text-align: right;
    margin-right: 8px; font-size: 10px;
    vertical-align: top;
  }

  .stack-frame .frame-cpp { color: #f97316; }
  .stack-frame .frame-file { color: var(--accent); }
  .stack-frame .frame-func { color: var(--text); }

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

  .hwm-line { stroke: var(--hwm-color); stroke-width: 1; stroke-dasharray: 6 4; }
  .hwm-label { fill: var(--hwm-color); font-size: 11px; font-family: var(--font); font-weight: 600; }

  .alloc-poly { stroke: rgba(0,0,0,0.3); stroke-width: 0.5; cursor: pointer; }
  .alloc-poly:hover { stroke: white; stroke-width: 1.5; }

  #tooltip {
    position: fixed; display: none;
    background: var(--tooltip-bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px 12px;
    font-size: 12px; line-height: 1.5;
    pointer-events: none; z-index: 100; max-width: 500px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
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
  '#6366f1', '#8b5cf6', '#a78bfa', '#c084fc',
  '#ec4899', '#f43f5e', '#fb7185', '#f97316',
  '#f59e0b', '#eab308', '#84cc16', '#22c55e',
  '#14b8a6', '#06b6d4', '#0ea5e9', '#3b82f6',
  '#818cf8', '#a5b4fc', '#c4b5fd', '#e879f9',
  '#f472b6', '#fb923c', '#fbbf24', '#a3e635',
  '#34d399', '#2dd4bf', '#22d3ee', '#38bdf8',
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

function renderStack(stackIdx, label) {
  const stack = STACKS[stackIdx] || [];
  detailStats.textContent = label;
  if (!stack.length) {
    detailBody.innerHTML = '<div class="empty-detail">No frames recorded</div>';
    return;
  }
  detailBody.innerHTML = stack.map((frame, i) => {
    const hasColon = frame.includes(':');
    const isCpp = !hasColon && frame.includes('::');
    let inner;
    if (isCpp) {
      inner = `<span class="frame-cpp">${frame}</span>`;
    } else if (hasColon) {
      const sp = frame.indexOf(' ', frame.lastIndexOf(':'));
      if (sp > 0) {
        inner = `<span class="frame-file">${frame.substring(0, sp)}</span> <span class="frame-func">${frame.substring(sp + 1)}</span>`;
      } else {
        inner = `<span class="frame-file">${frame}</span>`;
      }
    } else {
      inner = frame;
    }
    return `<div class="stack-frame" onclick="this.classList.toggle('expanded')"><span class="frame-idx">${i}</span><span class="frame-text">${inner}</span></div>`;
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
