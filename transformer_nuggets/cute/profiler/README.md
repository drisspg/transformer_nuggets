# NVIDIA Intra-Kernel Profiling for CUTE DSL

This package provides utilities for profiling code regions **inside** GPU kernels, generating [Perfetto](https://ui.perfetto.dev/)-compatible traces.

## Overview

Unlike traditional GPU profilers (Nsight, NCU) that profile at the kernel level, this utility enables **intra-kernel** profiling — measuring execution time of specific code regions within a single kernel launch.

```
┌────────────────────────────────────────────────────────────┐
│ Kernel Launch                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Region A   │  │   Region B   │  │   Region C   │     │
│  │  (profiled)  │  │  (profiled)  │  │  (profiled)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
```

## Package Structure

```
transformer_nuggets/cute/profiler/
├── __init__.py      # Package exports
├── ops.py           # Device-side profiling operations (CUTE DSL)
├── host.py          # Host-side utilities (buffer, decode, Perfetto)
├── example.py       # Complete working example
└── README.md        # This file
```

## Quick Start

### 1. Define Tags

Tags are named identifiers for profiled regions. Define them on the host:

```python
from transformer_nuggets.cute.profiler import TagTable

# Create tag table (order determines IDs: 0, 1, 2, ...)
tag_table = TagTable(["load", "compute", "store"])

# Use as constants in kernel
TAG_LOAD = 0
TAG_COMPUTE = 1
TAG_STORE = 2
```

### 2. Allocate Profile Buffer

```python
from transformer_nuggets.cute.profiler import allocate_profile_buffer

# Allocate buffer for up to 256 events
prof = allocate_profile_buffer(max_events=256, device="cuda")

# Buffer layout: [event_count, event0..., event1..., ...]
# Each event: [start_ns, dur_ns, tag_id, tid] (4 x int64)
```

### 3. Instrument Your Kernel

**Method 1: Context Manager (Recommended)**

```python
from transformer_nuggets.cute.profiler import profile_region

@cute.kernel
def my_kernel(output, prof_buf, max_events):
    bidx, _, _ = cute.arch.block_idx()

    # Profile a region with `with` statement
    with profile_region(prof_buf, max_events, Int32(TAG_COMPUTE), bidx):
        # ... code to profile ...
        compute_something()
```

**Method 2: Explicit Start/Stop**

```python
from transformer_nuggets.cute.profiler import lane0_warp0_start, lane0_warp0_stop

@cute.kernel
def my_kernel(output, prof_buf, max_events):
    bidx, _, _ = cute.arch.block_idx()

    # Start profiling (only warp 0, lane 0 records)
    eid = lane0_warp0_start(prof_buf, max_events)

    # ... code to profile ...
    compute_something()

    # Stop profiling
    lane0_warp0_stop(prof_buf, eid, Int32(TAG_COMPUTE), bidx, max_events)
```

### 4. Decode and Export

```python
from transformer_nuggets.cute.profiler import decode_events, events_to_perfetto

# Sync GPU
torch.cuda.synchronize()

# Decode events
events, overflow = decode_events(prof, tag_table)

# Export to Perfetto trace
events_to_perfetto(events, "trace.json")
print("View at: https://ui.perfetto.dev/")
```

### 5. View in Perfetto

1. Open https://ui.perfetto.dev/
2. Drag and drop `trace.json`
3. Explore the timeline!

## API Reference

### Host-Side (`host.py`)

| Function | Description |
|----------|-------------|
| `TagTable(names)` | Create mapping of tag names → integer IDs |
| `allocate_profile_buffer(max_events, device)` | Allocate GPU buffer for events |
| `decode_events(buf, tag_table)` | Decode raw buffer → list of `Event` objects |
| `events_to_perfetto(events, path)` | Export to Chrome trace JSON |
| `profile_session(...)` | Context manager combining all of the above |

### Device-Side (`ops.py`)

| Function | Description |
|----------|-------------|
| `profile_region(buf, max_events, tag, tid)` | Context manager for `with` statement |
| `lane0_warp0_start(buf, max_events)` | Start, only warp 0 lane 0 records |
| `lane0_warp0_stop(buf, eid, tag, tid, max_events)` | Stop, only warp 0 lane 0 records |
| `warp_start(buf, max_events, target_warp)` | Start, specific warp only |
| `warp_stop(buf, eid, tag, tid, max_events, target_warp)` | Stop, specific warp only |
| `elected_start(buf, max_events)` | Start, one lane per warp (all warps) |
| `elected_stop(buf, eid, tag, tid, max_events)` | Stop, one lane per warp |
| `profile_start(buf, max_events)` | Low-level start (all threads record!) |
| `profile_stop(buf, eid, tag, tid, max_events)` | Low-level stop |
| `read_globaltimer()` | Read GPU nanosecond timer |

## Profiling Strategies

### Minimal Overhead (Default)

Use `lane0_warp0_*` or `profile_region` — only **one thread per block** records events:

```python
# Only warp 0, lane 0 profiles (1 thread per block)
with profile_region(prof_buf, max_events, tag, bidx):
    ...
```

### Per-Warp Profiling

Use `elected_*` for one event per warp (useful for comparing warp performance):

```python
eid = elected_start(prof_buf, max_events)
...
elected_stop(prof_buf, eid, tag, warp_id, max_events)
```

### Warp-Specialized Kernels

Use `warp_*` to profile specific warps (e.g., TMA producer vs consumer):

```python
# Only warp 0 (producer) profiles
eid = warp_start(prof_buf, max_events, target_warp=Int32(0))
...
warp_stop(prof_buf, eid, TAG_PRODUCER, bidx, max_events, target_warp=Int32(0))
```

## Buffer Layout

The profile buffer is a `torch.int64` tensor with this layout:

```
buf[0]           = event_count (atomic counter)
buf[1 + 4*i + 0] = start_ns    (nanoseconds, from %globaltimer)
buf[1 + 4*i + 1] = dur_ns      (end_ns - start_ns)
buf[1 + 4*i + 2] = tag_id      (integer from TagTable)
buf[1 + 4*i + 3] = tid         (thread identifier, e.g., block index)
```

Total size: `1 + 4 * max_events` int64s.

## Complete Example

```python
import torch
from cutlass import Int32
from transformer_nuggets.cute.profiler import (
    TagTable, allocate_profile_buffer, decode_events, events_to_perfetto,
    profile_region, lane0_warp0_start, lane0_warp0_stop,
)

# Tags
TAG_OUTER = 0
TAG_INNER = 1

class MyProfiledKernel(CuteOp):
    @cute.kernel
    def kernel(self, output, prof_buf, max_events):
        bidx, _, _ = cute.arch.block_idx()

        # Profile outer loop
        eid = lane0_warp0_start(prof_buf, max_events)

        for i in cutlass.range(4):
            # Profile inner work with context manager
            with profile_region(prof_buf, max_events, Int32(TAG_INNER), bidx):
                do_work()

        lane0_warp0_stop(prof_buf, eid, Int32(TAG_OUTER), bidx, max_events)

    @cute.jit()
    def __call__(self, output, prof_buf, max_events):
        self.kernel(output, prof_buf, max_events).launch(grid=(4,1,1), block=(128,1,1))

    def interface(self, output, prof_buf, max_events):
        self.__call__(from_dlpack(output), from_dlpack(prof_buf), Int32(max_events))

# Run
tag_table = TagTable(["outer", "inner"])
prof = allocate_profile_buffer(256, device="cuda")
kernel = MyProfiledKernel()
kernel.interface(output, prof.tensor, 256)
torch.cuda.synchronize()

events, _ = decode_events(prof, tag_table)
events_to_perfetto(events, "trace.json")
```

## Running the Example

```bash
python -m transformer_nuggets.cute.profiler.example
```

This generates `profiler_example_trace.json` — open it in Perfetto to see the timeline.

## Requirements

- PyTorch with CUDA
- NVIDIA GPU (Hopper/Blackwell recommended)
- `cutlass` Python package (for CUTE DSL)

## Known Limitations

### ⚠️ Context Manager Variable Scoping (TODO)

The `profile_region` context manager currently has a limitation: **variables defined outside the `with` block are not visible inside**. This is due to how CUTE DSL generates IR — the context manager's `__enter__`/`__exit__` execute at IR generation time, creating a new scope.

```python
# ❌ This does NOT work today:
val = output[idx]  # defined outside
with profile_region(prof_buf, max_events, tag, tid):
    output[idx] = val + 1.0  # ERROR: 'val' not visible inside!

# ✅ Workaround: use explicit start/stop instead:
val = output[idx]
eid = lane0_warp0_start(prof_buf, max_events)
output[idx] = val + 1.0  # works fine
lane0_warp0_stop(prof_buf, eid, tag, tid, max_events)

# ✅ Or: define variables inside the context:
with profile_region(prof_buf, max_events, tag, tid):
    val = output[idx]  # defined inside - works!
    output[idx] = val + 1.0
```

**TODO**: Investigate CUTE DSL internals to enable proper variable capture across context manager boundaries.

## Notes

- **Timer Resolution**: Uses `%globaltimer` PTX register (nanosecond resolution on sm70+).
- **Overhead**: Each profiled region adds ~2 global memory accesses. Use sparingly in hot paths.
- **Thread Safety**: Event allocation uses atomic increment. Buffer overflow is handled gracefully.
- **Perfetto**: Timestamps are normalized to microseconds. Original nanoseconds preserved in event args.
