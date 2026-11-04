# Tracing

[Back to warpdrive](../README.md)

## Inspect the kernel

```python
from transformer_nuggets.cute.warpdrive import ClusterLaunchControl, CtaPair
from transformer_nuggets.cute.warpdrive.examples.persistent_gemm import (
    PersistentGemm,
    compile_persistent_gemm,
)

op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair())
print(op.describe())
compile_persistent_gemm(op)
print(op.describe())
```

For the 2-CTA persistent GEMM with CLC, after the compile (output at `2491252`; print yours for the
current version):

```
PersistentGemm: 8 warps, 256 threads
  cluster  CTA pair (2, 1, 1), tcgen05 CtaGroup.TWO, leader = rank 0
  params   PersistentGemmParams

Roles
  warps  role            regs  cta     raw
  0-3    epilogue         160  all
  4      load              40  all
  5      mma               40  leader
  6      work.scheduler    40  leader
  7      (padding)         40

Channels
  ab         Ring(6)  load -> mma
             signal    Tma(32768 B) -> Umma
             full      leader 1 x6, tx 65536 B (leader-armed)
             empty     leader 1 x6, peer 1 x6
             scope     pair-scoped, release_mask 0b11, tail
             storage   buffer A, B
             tma       verified CtaGroup.TWO, 32768 B/stage per CTA @ persistent_gemm.py:156,157
  acc        Ring(2)  mma -> epilogue
             signal    Umma -> Threads
             full      leader 1 x2, peer 1 x2
             empty     leader 256 x2
             scope     pair-scoped, commit_mask 0b11, remote_scope cta
  work_ring  Ring(2)  work.scheduler -> epilogue, load, mma, work.scheduler
             signal    Tma(16 B) -> Threads
             full      leader 1 x2, peer 1 x2, tx 16 B (leader-armed)
             empty     leader 384 x2
             scope     pair-scoped, remote_scope cluster
             storage   payload Int32(4,)
  work_clc   Slots(1) work.scheduler -> work.scheduler
             signal    Tma(16 B) -> Threads
             full      1, tx 16 B
             empty     32
             scope     CTA-local
             storage   payload Int32(4,), release_fence smem_async

Streams
  work  ClusterLaunchControl(stages=2), Raster(along=m, swizzle=8)
        scheduler warp 6 broadcasts over work_ring
        roles: epilogue, load, mma

TMEM: 512 columns used, 512 allocated
  tmem acc [0, 256), [256, 512)

SMEM plan: 212992 B per CTA in 3 buffer(s), plus alignment padding
  A  BFloat16  98304 B  6 stages (mode -1)  align 1024  -> ab
  B  BFloat16  98304 B  6 stages (mode -1)  align 1024  -> ab
  C  BFloat16  16384 B                      align 128
```

Reading it: `ab` is pair-scoped, so the leader arms 64 KiB for both CTAs' TMA halves and the
MMA's release multicasts to both (`release_mask 0b11`). `acc`'s empty barrier lives on the
leader and expects 256 epilogue threads from the two CTAs. The scheduler warp sits in the padding
warpgroup and broadcasts each CLC claim through `work_ring` to every role in both CTAs.

`describe()` returns a string with roles, warp placement, register budgets, channels,
per-CTA counts and TMEM regions. Before compilation it checks declarations. After a fresh
compile it also has the recognized TMA copies and any traced diagnostics. It is not a runtime
history or a deadlock check.

For a stream named `work`, the inserted scheduler is `work.scheduler`, its broadcast ring is
`work_ring`, and CLC's response slot is `work_clc`. Print the current kernel instead of relying
on a saved dump: stage counts and placement change with the configuration.

With `SmemPlan`, the description includes buffer sizes, stage modes, overlaps and channel
bindings once their layouts have been traced. It also includes the
[alias diagnostic](storage.md#check-declared-aliases) for declared overlaps.

## Trace the protocol

```python
op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair())
op.trace_protocol = True  # before the first describe() or compile
compile_persistent_gemm(op)
print(op.describe())
```

This adds the traced acquire/commit/wait/release sites and source lines. A loop has a traced
site, not one record per runtime iteration. Use it to check which handle closes where, not
to determine when a warp stalled.

Include `trace_key()` in custom compile-cache keys so an untraced kernel cannot satisfy a
traced request. The standard CuTe cache helpers include the kernel's
`__cute_compile_key__()` hook, which covers these tracing choices.

### TMA checks

Compilation inspects recognizable copies using channel barriers. A known wrong CTA group or
a complete byte-count mismatch raises `ProtocolError`. Recognized copies report their group,
bytes per stage per CTA and source locations.

Unknown atoms, multicast, opaque pointers, nested/dynamic copies or unsupported IR can leave
this check incomplete. Compilation continues with a warning and `unverified` reasons in
`describe()`. `verified` means the recognized group and byte accounting match; it does not
certify descriptor legality, disjoint writes or all async ordering. You still supply
`Tma(tx_bytes)` yourself.

## IKET

```python
from transformer_nuggets.cute.warpdrive import StaticPersistent

op = PersistentGemm(schedule=StaticPersistent()).enable_iket(waits=False)
fn = compile_persistent_gemm(op)
```

Enable IKET before compiling. `waits=False` records role ranges without the finer wait ranges.
Start there: instrumentation around short waits can distort the timing you are trying to read.

Roles taking `WorkCursor` get one range per work item. Other roles get one range over their
body. With the default `waits=True`, blocking operations also get names such as `ab.acquire`,
`ab.wait`, `ab.tail` and `mufu.turn`.

Add a region in a role:

```python
with self.trace_region("exp"):
    ...  # exp2 work
```

Disabled tracing emits no IKET operations. Include `iket_key()` in custom compile-cache keys,
or use the standard hook as above.

The following needs NVIDIA's `run-iket` tooling and a script that launches the compiled kernel
once. `profile_kernel.py` is your script, not a file supplied by warpdrive:

```bash
run-iket -o traces/gemm --clobber profile --postprocess all -- python profile_kernel.py
```

## Join a capture to the pipeline plan

```python
from transformer_nuggets.cute.warpdrive import plan_from_warpdrive
from transformer_nuggets.cute.profiler.pipeline import (
    analyze_pipeline,
    load_iket_capture,
    write_pipeline_perfetto,
)

plan = plan_from_warpdrive(op, cta=0)  # after a fresh compile, for traced reuse counts
timeline = plan.schedule(iterations=8)
measured = load_iket_capture("capture.trace.json", timeline, cta=(0, 0, 0))
analysis = analyze_pipeline(timeline, measured)
write_pipeline_perfetto("gemm.pftrace", timeline, measured, analysis=analysis)
```

Use the JSON capture path produced by your IKET run in place of `capture.trace.json`.
Open the resulting trace in [Perfetto](https://ui.perfetto.dev/).

The plan is per CTA. `cta=1` excludes leader-only roles in a pair. Its regions are work-item
granular, so ring occupancy and reuse estimates for an inner K loop are coarse, not
measurements of individual stages. Traced reuse counts require a compile on this object;
a cache hit need not populate its trace data.

See the [pipeline profiler README](../../profiler/README.md#declarative-pipeline-analysis)
for critical-path analysis, measured timelines and capture tooling.
