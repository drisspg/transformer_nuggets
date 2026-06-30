# Intra-Kernel Profiling for CUTE DSL

Profile code regions **inside** GPU kernels and export to [Perfetto](https://ui.perfetto.dev/).

> Inspired by [gau-nernst's intra-kernel profiling for AMD MI300X](https://gau-nernst.github.io/amd-a2a/#intra-kernel-profiling)

## Quick Start

```python
from transformer_nuggets.cute.profiler import profile_session, profile_region

@cute.kernel
def my_kernel(output, prof_buf, max_events):
    bidx, _, _ = cute.arch.block_idx()
    for i in cutlass.range(4):
        with profile_region(prof_buf, max_events, TAG_COMPUTE, bidx):
            compute_something()

with profile_session(
    max_events_per_unit=64,
    num_units=(num_blocks, "Block"),
    tag_names=["compute"],
    trace_path="trace.pftrace",
) as (prof, _):
    my_kernel(output, prof.tensor, prof.max_events_per_unit)
```

## Three Modes

### Atomic mode (default)

Omit `event_idx` and indices are allocated via atomics at runtime:

```python
with profile_region(prof_buf, max_events, TAG, unit_id):
    do_work()
```

Works in loops without manual bookkeeping. The tradeoff: **nested regions cause timing skew**. The inner region's atomic runs while the outer region's timer is still running, inflating the outer duration.

### Static mode (accurate for nesting)

Pass an explicit `event_idx` to avoid atomics entirely:

```python
with profile_region(..., event_idx=Int32(0)):
    with profile_region(..., event_idx=Int32(1)):
        do_work()
```

For loops, compute indices from the loop variable:

```python
for i in cutlass.range(4):
    with profile_region(..., event_idx=Int32(0) + i * Int32(2)):
        ...
    with profile_region(..., event_idx=Int32(1) + i * Int32(2)):
        ...
```

You set `max_events_per_unit` to something larger than you need; the decoder scans all slots and skips empty ones.

### Token mode (explicit pairing across scopes)

`with profile_region(...)` already pairs start and end structurally, but it can't bridge two Python scopes or open in iteration `i` and close in `i+1`. For those cases, use `region_start` / `region_end` and pass the returned `RegionToken` explicitly:

```python
from transformer_nuggets.cute.profiler import region_start, region_end, RegionToken

outer = region_start(prof_buf, unit_id, max_events)
for i in cutlass.range(N):
    inner = region_start(prof_buf, unit_id, max_events)
    do_work(i)
    region_end(prof_buf, TAG_INNER, inner, max_events)
region_end(prof_buf, TAG_OUTER, outer, max_events)
```

`RegionToken` captures `(unit_id, event_idx, start_ns, target_warp)`, so `region_end` only needs the token plus the tag, and the start/end pair can't drift on those fields. It's a `NamedTuple`, so the DSL can thread it through `cutlass.range` as a loop-carried value if you need to keep a region open across iterations.

This is the closest analogue to NVIDIA IKET's `iket.range_start` / `iket.range_end` SSA-token pairing in `cutlass.cute.experimental.iket`. The shape is similar but the mechanism is different: IKET emits MLIR ops and lowers via the proprietary `iket` dialect, while this profiler emits inline PTX (`%globaltimer`, `st.global.cs.u64`) directly.

## API

### Host (`host.py`)

| Function | Description |
|----------|-------------|
| `profile_session(...)` | Context manager: allocate, yield, decode, write trace |
| `allocate_profile_buffer(max_events_per_unit, num_units, device)` | Allocate buffer |
| `decode_events(buf, tag_table)` | Decode to `Event` list |
| `events_to_perfetto(events, path)` | Write native Perfetto TrackEvent `.pftrace` by default, or Chrome JSON/JSON.GZ with `trace_format="chrome_json"` |
| `TagTable(names)` | Map tag names ↔ integer IDs |
| `PostProcessContext` | Context passed to post-processing callbacks |

### Device (`ops.py`)

| Function | Description |
|----------|-------------|
| `profile_region(buf, max_events, tag, unit_id, target_warp=None, event_idx=None, tid=None)` | Context-manager API |
| `region_start(buf, unit_id, max_events, target_warp=None, event_idx=None) -> RegionToken` | Open a region, return a pairing token |
| `region_end(buf, tag, token, max_events_per_unit, tid=None)` | Close a region using its token |
| `warp_start/warp_stop(...)` | Low-level start/stop (lane 0 of target_warp) |
| `warp_atomic_alloc(...)` | Allocate event index atomically |

## Trace Formats

Two output formats are supported:

- `track_event` (default): native Perfetto protobuf (`.pftrace` / `.perfetto-trace`). This is the preferred format for programmatically generated traces. Crossing overlaps on one logical track are encoded as multiple backing TrackEvent tracks with the same merge key, so Perfetto can display them as one logical row.
- `chrome_json`: legacy Chrome JSON/JSON.GZ. Use this only for compatibility with tools that require Chrome JSON. Perfetto handles this format on a best-effort basis and requires duration events on a track to nest cleanly.

Pass `trace_format="chrome_json"` to `profile_session` or `events_to_perfetto` to opt into legacy JSON output. Pass `split_overlaps=False` to keep raw tracks.

## Post-Processing


You can pass callbacks to `profile_session` to mutate events or the Perfetto trace before writing:

- `post_process_events(events, ctx) -> events`: Rename, filter, or regroup events.
- `post_process_trace(trace_dict, ctx) -> trace_dict`: Add flow events, counters, etc.

The `PostProcessContext` provides:
- `tag_table`: TagTable for looking up tag names/IDs.
- `prof_buf`: The ProfileBuf with raw event data.
- `unit_name`: Name for units in trace (e.g., "Block", "Warp").

### Example: Dynamic Naming

```python
def add_unit_prefix(events, ctx):
    for e in events:
        e.tag_name = f"{ctx.unit_name} {e.unit_id}: {e.tag_name}"
    return events

with profile_session(
    ...,
    post_process_events=add_unit_prefix,
) as (prof, tags):
    ...
```

### Example: Group by Block/Warp (Nsight-style view)

To get separate Perfetto rows for each warp within each block (like Nsight's CTA view), set `pid` per block and `tid` per warp:

```python
def group_by_block_warp(events, ctx):
    warps_per_block = 4
    for e in events:
        block_id = e.unit_id
        warp_id = e.tid % warps_per_block
        e.pid = block_id              # Each block becomes a Perfetto "process"
        e.tid = warp_id               # Each warp becomes a lane under that process
        e.tag_name = f"warp {warp_id}: {e.tag_name}"
    return events

with profile_session(
    ...,
    post_process_events=group_by_block_warp,
) as (prof, tags):
    ...
```

In Perfetto, this renders as:
- `CTA 0` (process)
  - `warp 0` (thread lane)
  - `warp 1`
  - ...
- `CTA 1` (process)
  - `warp 0`
  - ...

### Example: Add Extra Metadata

```python
def add_tile_coords(events, ctx):
    tile_map = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    for e in events:
        tile_m, tile_n = tile_map.get(e.unit_id, (-1, -1))
        e.extra_args = {"tile_m": tile_m, "tile_n": tile_n}
    return events
```

The `extra_args` dict is merged into each event's `args` in the Perfetto trace.

## Buffer Layout

Each unit owns `1 + 4 * max_events_per_unit` int64s:

```
For each unit u (0 <= u < num_units):
  buf[u * slice_size + 0]           = event_count (atomic mode) or unused
  buf[u * slice_size + 1 + 4*i + 0] = start_ns
  buf[u * slice_size + 1 + 4*i + 1] = dur_ns
  buf[u * slice_size + 1 + 4*i + 2] = tag_id
  buf[u * slice_size + 1 + 4*i + 3] = tid

slice_size = 1 + 4 * max_events_per_unit
```

## Example

```
python -m transformer_nuggets.cute.profiler.example
```

Open the trace in https://ui.perfetto.dev/
