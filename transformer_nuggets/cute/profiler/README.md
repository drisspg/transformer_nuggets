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
    trace_path="trace.json",
) as (prof, _):
    my_kernel(output, prof.tensor, prof.max_events_per_unit)
```

## Two Modes

### Atomic Mode (simple)

Omit `event_idx` and indices are allocated via atomics at runtime:

```python
with profile_region(prof_buf, max_events, TAG, tid):
    do_work()
```

Works in loops without manual bookkeeping. The tradeoff: **nested regions cause timing skew**. The inner region's atomic runs while the outer region's timer is still running, inflating the outer duration.

### Manual Mode (accurate for nesting)

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

## API

### Host (`host.py`)

| Function | Description |
|----------|-------------|
| `profile_session(...)` | Context manager: allocate, yield, decode, write trace |
| `allocate_profile_buffer(max_events_per_unit, num_units, device)` | Allocate buffer |
| `decode_events(buf, tag_table)` | Decode to `Event` list |
| `events_to_perfetto(events, path)` | Write Chrome trace JSON |
| `TagTable(names)` | Map tag names ↔ integer IDs |

### Device (`ops.py`)

| Function | Description |
|----------|-------------|
| `profile_region(buf, max_events, tag, tid, event_idx=None)` | Context manager |
| `warp_start/warp_stop(...)` | Low-level start/stop (lane 0 of target_warp) |
| `warp_atomic_alloc(...)` | Allocate event index atomically |

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
