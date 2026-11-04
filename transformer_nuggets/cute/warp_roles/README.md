# warp_roles

A small authoring layer for warp-specialized Blackwell (SM100/SM103) kernels in CuTeDSL.

You write each warp role as an ordinary `@cute.jit` method and declare the handoffs between
roles. The layer derives the synchronization bookkeeping that has one correct answer and checks
the protocol at compile time. Everything it derives can be printed, and every role body stays
plain CuTeDSL, so you can drop to `cute.arch` / `cutlass.pipeline` code whenever you need to.

It is meant as authoring code for people and coding agents iterating on kernel designs:
tiling, schedules, stage counts and which syncs to skip stay in your hands; the layer makes each
of those a cheap, checked experiment.

> Status: experimental. Requires `nvidia-cutlass-dsl` 4.8 and an SM100/SM103 GPU. The API is
> still moving; see [Roadmap](#roadmap).

## Design principles

- **Sugar, not a wall.** The layer wraps existing CuTeDSL/CUTLASS primitives (mbarriers,
  `PipelineState`, TMEM allocators, `setmaxnreg`, CLC). Role bodies are plain CuTeDSL.
- **Hide only what has one correct answer.** You supply the design inputs (stage counts, per-stage
  TMA bytes via `Tma(tx_bytes)`, register budgets via `@role(regs=...)`, warp placement). The
  layer derives the bookkeeping from them: arrival counts, phases, barrier init and teardown,
  TMEM allocate/free, warpgroup register padding, scheduler plumbing, and cross-CTA targets,
  masks and byte scaling.
- **Make mistakes compile-time errors.** Unreleased or double-released handles (reported with
  the source line), an instance-bound role touching another instance's slot, or a channel used by
  an undeclared role raise `ProtocolError` at compile time instead of hanging on the GPU.
- **Derived values are inspectable.** `print(kernel.describe())` shows roles, warps, register
  budgets, per-channel arrival counts, multicast masks and the TMEM plan.
- **Thin at runtime.** At a matched schedule, the FA4-style example timed within about 1% of
  hand-written FA4 (see [Examples](#examples)).

## A first kernel

A BF16 GEMM (`D = A @ B^T`) with a TMA load warp, an MMA warp and four epilogue warps
(abridged from [`examples/gemm.py`](examples/gemm.py)):

```python
from transformer_nuggets.cute.warp_roles import (
    Consumer, Producer, Ring, Threads, Tma, Tmem, TmemPlan, Umma, WarpSpecialized, role,
)

class Gemm(WarpSpecialized):
    def __init__(self, ab_stages=3):
        # A/B tiles: filled by TMA (with this many bytes per stage), drained by tcgen05 MMA.
        self.ab = Ring(ab_stages, Tma((128 + 256) * 64 * 2), Umma)
        # Accumulator: filled by MMA, drained by the epilogue threads.
        self.acc = Ring(1, Umma, Threads)
        self.tmem_plan = TmemPlan().region("acc", 256, 1)

    @role(warps=4, regs=40)
    @cute.jit
    def load(self, ab: Producer, c):
        for k in cutlass.range(c.num_k_tiles, unroll=1):
            with ab.produce() as s:  # acquire an empty stage ... commit
                cute.copy(c.tma_a, c.tAgA[None, k], c.tAsA[None, s.index], tma_bar_ptr=s.barrier)
                cute.copy(c.tma_b, c.tBgB[None, k], c.tBsB[None, s.index], tma_bar_ptr=s.barrier)

    @role(warps=5, regs=40)
    @cute.jit
    def mma(self, ab: Consumer, acc: Producer, tmem: Tmem, c):
        tAcc = cute.make_tensor(tmem.ptr("acc"), c.acc_layout)
        with acc.produce() as a:
            for k in cutlass.range(c.num_k_tiles, unroll=1):
                with ab.consume() as s:  # wait for a full stage ... release it
                    ...  # cute.gemm(tiled_mma, tAcc[..., a.index], c.tCrA[..., s.index], ...)

    @role(warps=range(4), regs=160)
    @cute.jit
    def epilogue(self, acc: Consumer, tmem: Tmem, c):
        with acc.consume() as a:
            ...  # TMEM -> registers -> global

    @cute.kernel
    def kernel(self, ...):
        smem = self.begin_roles()   # layer: barrier/payload storage + init, TMEM allocation
        ...                         # your prologue: SMEM tensors, TMA partitions, layouts
        self.run_roles(ctx)         # layer: init sync, role dispatch, TMEM teardown
```

There is no `if warp_idx == ...` in user code, no barrier initialization, no arrival counts, no
phase variables, and no TMEM allocation code. `begin_roles` and `run_roles` emit all of that
(for CTA pairs, TMEM is allocated in `run_roles` after the cluster sync).

## How roles are called

`run_roles(ctx)` initializes the barriers, then branches once per register budget on a
warpgroup-uniform condition, applies `setmaxnreg` there, and dispatches each role on its own
warps. Each `@role` method is traced once per role instance; its arguments are injected from the
parameter annotations. Channel, turnstile and tile parameters are matched to the kernel attribute
with the same name:

| Annotation | Receives | Matched to |
|---|---|---|
| `ab: Producer` / `ab: Consumer` | that channel's end | `self.ab` (`Ring` or `Slots`) |
| `s: Producer[Instance]` / `s: Consumer[Instance]` | a `Slots` end bound to this instance's slot | `self.s` |
| `tmem: Tmem` | TMEM view: `tmem.ptr("acc", j)`, `tmem.cols(...)` | the kernel's `tmem_plan` |
| `tiles: Tiles` | work cursor: `tiles.valid`, `tiles.work`, `tiles.advance()` | `self.tiles` (`TileStream`) |
| `stage: Instance` | this instance's index (a Python int) | |
| `mufu: Turn` | a turnstile seat: `with mufu.turn():` | `self.mufu` (`Turnstile`) |
| anything else | the context object passed to `run_roles(ctx)` | |

The signature is the dependency declaration: the layer learns each channel's producer and
consumer roles from it and derives arrival counts from their warps. A misspelled channel name is
a compile-time error. Note that today **any unrecognized annotation** (including a typo such as
`Consumr`) silently receives the context; stricter injection is on the roadmap.

`@role` arguments:

- `warps`: a warp **index** (`warps=4` is warp 4, not four warps), a `range` (`range(4)` is
  warps 0-3), or a list of ranges (one role instance per range, e.g.
  `[range(0, 4), range(4, 8)]` for two softmax warpgroups).
- `regs`: the role's `setmaxnreg` budget (default `None`: no adjustment). Every warp of a
  warpgroup must share one budget; unowned warps are padded to their warpgroup's budget.
- `cta`: `"all"` (default) or `"leader"` (only the leader CTA of a `CtaPair`, e.g. the MMA
  issuer; a no-op for single-CTA kernels).

## Concepts

### Channels: `Ring` and `Slots`

A channel is the handoff protocol for a staged resource: a full and an empty mbarrier per stage,
plus phase bits. The memory itself (SMEM tiles, TMEM regions) is allocated separately; a
handle's `index` selects the stage's buffer.

- **`Ring(n, producer, consumer)`: the next buffer.** Stages are used in order 0..n-1 and wrap;
  one cursor per end. Use for streams: K/V tiles, GEMM A/B, accumulator double-buffering.
- **`Slots(n, producer, consumer)`: that buffer.** The caller names the slot
  (`acquire(j)`/`wait(j)`); each slot keeps its own phase and advances independently. Use when
  a buffer belongs to a logical index or a role instance, e.g. FA4's `Q[j]` and `S/P/O[j]`.
  Slot indices are Python ints (from `Instance` or `cutlass.range_constexpr`), at most 31 slots.
  An instance-bound end (`Consumer[Instance]`) still takes the slot explicitly
  (`s.wait(stage)`) and rejects any other slot; a plain `Slots` end may use any slot.

Each side is signalled by a **mechanism**:

| Mechanism | Signal | Typical side |
|---|---|---|
| `Tma(tx_bytes)` | TMA transaction bytes complete the barrier | producer of SMEM tiles |
| `Umma` | one elected `tcgen05.commit` | MMA producer or consumer |
| `Threads` | every thread of the bound role(s) arrives | epilogue, softmax, correction |

A `Tma` or `Umma` side signals once, so it must belong to exactly one single-warp role. A
`Threads` side may have several roles (FA4's S/P/O return edge needs both softmax and
correction); the layer sums their arrivals. Optional channel arguments: `payload=(dtype, shape)` for layer-owned per-stage SMEM
(`handle.data`; CTA-local channels only), `commit_fence`/`release_fence` (`"tmem_store"` or
`"smem_async"`; TMA producers take no `commit_fence`), and `remote_scope` for CTA pairs
(below).

### Handles

`with ab.produce() as s:` / `with ab.consume() as s:` pair the acquire/commit or wait/release by
construction. For lifetimes that don't nest (FA4 holds Q across the whole KV loop and releases
V(i-1) after issuing the next QK), use explicit handles:

```python
v = kv.wait()
...
v.release()
```

The layer checks at trace time that every handle is closed exactly once, in the same dynamic
region where it was opened.

### `Turnstile`

A round-robin critical section between the instances of one role (FA4's two softmax warpgroups
taking turns on the exp2 unit): `with mufu.turn():`. `Turnstile(enabled=False)` keeps the API
and emits nothing, for A/B runs.

### `TmemPlan`

Named TMEM regions, optionally with slots and aliases:

```python
self.tmem_plan = (
    TmemPlan()
    .region("S", 128, 2)                     # 2 slots of 128 columns
    .region("O", 128, 2)
    .alias("P", of="S", offset=64, cols=64)  # bf16 P written into S's columns
)
```

The layer allocates the smallest power-of-two column count (32 to 512) covering the plan, 2-CTA
aware, and frees it after every TMEM user has finished. In a role, `tmem.cols(name, slot)` is a
column offset and `tmem.ptr(name, slot)` an fp32 TMEM pointer; reinterpreting a region (for
example as packed bf16 P) is explicit in your code. Capacity and alias bounds are checked;
ordering between aliased regions is your responsibility (document it next to the code that
relies on it).

### Scheduling: `TileStream`

```python
self.tiles = TileStream(order=Raster(swizzle=8), policy=Clc(stages=2))

@role(warps=4, regs=40)
@cute.jit
def load(self, ab: Producer, tiles: Tiles, c):
    while tiles.valid:
        m, n, _ = tiles.work
        ...
        tiles.advance()
```

`TileStream()` defaults to `Raster(along="m", swizzle=1)` and `OneShot()`. Policies are a
one-line switch:

| Policy | How a CTA gets its next work item |
|---|---|
| `OneShot()` | one item per CTA (grid = number of items) |
| `StaticPersistent()` | persistent grid; each CTA strides by the grid size |
| `DynamicPersistent()` | persistent grid; claims items from a global atomic counter |
| `Clc(stages=2)` | Blackwell cluster launch control (hardware work stealing) |

One-shot and static policies decode work locally in every role, with no synchronization. Dynamic
and CLC add a layer-managed scheduler warp that claims each item and broadcasts it to all roles
over a ring. `Raster(along=..., swizzle=...)` maps a linear index to tile coordinates;
`TileStream(ready=PdlWait())` waits for a programmatic dependent launch before the first item.
Tile counts, batch and SM count are runtime values, so one compile serves every problem size.

Rules: loops are `while tiles.valid` because CuTeDSL only stages `for` loops over `range`; each
role has exactly one `advance()`, directly in that loop body; under persistent policies every
role must take and drain the stream.

Host side (see [`examples/persistent_gemm.py`](examples/persistent_gemm.py)):

```python
workspace, max_ctas = op.tiles.launch_args(device)   # outside the compiled entry point
# inside the @cute.jit entry point:
work = self.tiles.problem((m_tiles, n_tiles, 1), workspace, max_ctas)
self.kernel(..., work).launch(**self.launch_config(work), stream=stream)
# and in the kernel: smem = self.begin_roles(tiles=work)
```

Static and dynamic policies need `max_ctas`. The dynamic policy's int32 counter workspace
starts zeroed and resets itself after each launch; don't share one workspace between
concurrent launches.

### CTA pairs: `CtaPair`

```python
self.cluster = CtaPair()                 # cluster (2, 1, 1), tcgen05 CtaGroup.TWO

@role(warps=5, regs=40, cta="leader")    # MMA issued by the leader CTA only
```

Switching a kernel to 2-CTA is that declaration and `cta="leader"` on the MMA role, plus the
usual 2-CTA kernel changes in your code: a 2x MMA tile M, MMA and TMA atoms built with
`self.cluster.cta_group` and `self.cluster.layout_vmnk()`, and rank-based partitions
(`self.cluster.rank()`). The persistent GEMM and FA4 examples show both versions. From role
placement the layer derives which channels span the pair, where each barrier
lives, per-CTA arrival counts, remote arrivals, `tcgen05.commit` multicast masks, TMA bytes for
both halves, producer tails, 2-CTA TMEM allocation, and the exit cluster barrier. Work items are
per cluster; dynamic and CLC policies claim on the leader and publish to both CTAs.

`remote_scope` defaults to `"cta"` for channels without a payload, which is what CUTLASS and FA4
pipelines use: correct for TMA/MMA/TMEM handoffs ordered by their own fences. **A pair-scoped
channel whose signal publishes ordinary memory (DSMEM or global) that you write yourself must set
`remote_scope="cluster"`.** The layer cannot detect this case. In our measurements the cluster
scope made FA4's hot S channel about 30% slower, which is why it is not the default.
Layer-owned payloads are per-CTA SMEM, so pair-scoped channels cannot have one.

## Writing roles: CuTeDSL rules

- Put `@role(...)` directly above `@cute.jit`.
- Helpers that take channel ends (or the tile cursor) must return them, and the caller
  reassigns: `s, stats = step(s, stats)`. Otherwise loops don't carry their phases. This is the
  same rule as for `PipelineState`.
- The context is read-only: write into tensors it holds (`c.mO[...] = x`), never rebind its
  fields.
- Define kernels in source files; CuTeDSL re-reads source for AST preprocessing.

## Inspecting and drilling down

`print(kernel.describe())` shows what the layer derived. For the 2-CTA persistent GEMM with CLC:

```
8 warps (256 threads); CTA pair, cluster (2, 1, 1), tcgen05 CtaGroup.TWO, leader = rank 0; ...
  role epilogue    warps  0-3  regs=160
  role load        warps  4-4  regs=40
  role mma         warps  5-5  regs=40 cta=leader
  role tiles.scheduler warps  6-6  regs=40 cta=leader
  padding          warps  7-7  regs=40
  chan ab     Ring(6)  load -> mma; pair-scoped: full leader [1, 1, 1, 1, 1, 1]; ... tx=65536 release_mask=0b11 tail
  chan acc    Ring(2)  mma -> epilogue; pair-scoped: ... empty leader [256, 256] commit_mask=0b11 remote_scope=cta
  ...
  tmem acc [0, 256), [256, 512)
```

Set `trace_protocol = True` on the kernel before its first compile to also record each role's
acquire/commit/wait/release sequence with source lines (as traced, not a runtime history);
`describe()` after the compile includes it.

The underlying objects are always reachable:

- Role bodies are plain CuTeDSL: any `cute.arch`, `cute.copy`, named barrier or fence works.
- Handles: `s.index`, `s.barrier` (full mbarrier), `s.empty_barrier`, `s.data` (payload).
- Channel ends: `end.state` (a `PipelineState` for rings, an Int32 phase-bit mask for slots),
  `end.scope` (derived counts, masks, bytes).
- `self.cluster.rank()`, `is_leader()`, `layout_vmnk()`; `tmem.base`, `tmem.ptr(...)`.
- `begin_roles()` returns the ordinary `SmemAllocator`; your own pipelines and barriers can live
  next to the layer's. Named barrier IDs 1 and 2 are reserved by the layer.

## Examples

| Example | What it shows |
|---|---|
| [`examples/gemm.py`](examples/gemm.py) | minimal 1-CTA GEMM: three roles, two rings |
| [`examples/persistent_gemm.py`](examples/persistent_gemm.py) | persistent GEMM: all four policies, TMEM accumulator double-buffering, TMA-store epilogue, `cluster=CtaPair()`, `tile_n` |
| [`examples/fa4_fwd.py`](examples/fa4_fwd.py) | FlashAttention-4-style forward (bf16, head dim 128): two softmax instances, correction, fixed slots, fan-out/join, non-nested handles, turnstile, S/P TMEM aliasing, split-P, persistent and 2-CTA |

Input limits: GEMMs take row-major bf16 `a: (M, K)`, `b: (N, K)` with K and N divisible by 8.
FA4 takes contiguous, 16-byte-aligned bf16 `(batch, seqlen, heads, 128)` tensors with the key
length a positive multiple of 128, and imports `flash_attn.cute` and `quack` helpers.

Observed on one GB300 (bf16, CUDA-graph replay medians, November 2026; nvidia-cutlass-dsl 4.8.0,
PyTorch 2.15 nightly cu132, FlashAttention-4 at 849b5daa). Indicative, not a tuned benchmark:

| Workload | Reference | warp_roles |
|---|---:|---:|
| GEMM 8192^3, 2-CTA persistent | `torch.mm` 694 us | 705 us |
| GEMM 4096^3, 2-CTA, `tile_n=224` | `torch.mm` 89.9 us | 89.7 us |
| FA4 forward 4x16x4096, same 1-CTA non-persistent schedule as FA4 | FA4 375 us | 373 us |
| FA4 forward 4x16x4096, 2-CTA CLC + split-P | FA4 default 337 us | 346 us |

The FA4 example and GEMMs are teaching kernels, not production replacements: masking, varlen,
GQA packing, other head dims and tuning are out of scope.

## Tests

```bash
pip install -e ".[cute,dev]" pytest-xdist
pytest -n 6 test/test_cute_warp_roles*.py
```

FA4 tests skip when FlashAttention-4 (`flash_attn.cute`) is not installed.

Covers numerical correctness (including ragged shapes and non-uniform attention data that
catches slot mixups), bitwise determinism across repeated launches, compile reuse across shapes,
every policy and CTA-pair configuration, and the compile-time protocol errors.

## Roadmap

- Back `Ring`/`Slots` with CUTLASS pipeline objects where semantics match, exposed for drill-down.
- Explicit role returns for drained producer ends; stricter parameter injection.
- A small public scheduler interface for swapping in custom or external schedulers.
- Unmanaged roles and barriers, per-site check opt-outs, and an automatic TMA atom/byte check.
- Optional storage binding on channels (`buffer=...`, `handle.data` as the stage view).
- Named IKET tracing driven by role and channel names.
- Larger clusters with TMA multicast, round-robin role instances, cp.async producers.
