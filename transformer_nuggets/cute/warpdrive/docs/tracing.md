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

For the 2-CTA persistent GEMM with CLC (its default `tile_n=224`), after the compile (print yours
for the current version):

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
             signal    Tma(30720 B) -> Umma
             full      leader 1 x6, tx 61440 B (leader-armed)
             empty     leader 1 x6, peer 1 x6
             scope     pair-scoped, release_mask 0b11, tail
             storage   buffer A, B
             tma       verified CtaGroup.TWO, 30720 B/stage per CTA @ persistent_gemm.py:194,195
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

TMEM: 448 columns used, 512 allocated
  tmem acc [0, 224), [224, 448)

SMEM plan: 200704 B per CTA in 3 buffer(s), plus alignment padding
  A  BFloat16  98304 B  6 stages (mode -1)  align 1024  -> ab
  B  BFloat16  86016 B  6 stages (mode -1)  align 1024  -> ab
  C  BFloat16  16384 B                      align 128
```

Reading it: `ab` is pair-scoped, so the leader arms 60 KiB for both CTAs' TMA halves and the
MMA's release multicasts to both (`release_mask 0b11`). `acc`'s empty barrier lives on the
leader and expects 256 epilogue threads from the two CTAs. The scheduler warp sits in the padding
warpgroup and broadcasts each CLC claim through `work_ring` to every role in both CTAs.

`describe()` returns a string with roles, warp placement, register budgets, channels,
per-CTA counts and TMEM regions. Before compilation it checks declarations. After a fresh
compile it also has the recognized TMA copies and any traced diagnostics. It is not a runtime
history or a deadlock check. There is no combined cross-role epoch-balance or wait-for-graph
verifier: balanced declarations can still form an unseeded wait cycle. Runtime synchronization
relies on a finite, live abstract execution with enough storage for overlapping lifetimes;
compilation does not establish it. A timeout needs attribution, not an assumption that either
the layer or the role body is correct.

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

A cached kernel was never traced, so the examples' `compile_*` functions compile a
`trace_protocol` kernel without the cache (`jit_cache(uncached=...)`). Other trace results
come back with a cached kernel ([compile keys](roles.md#compile-keys)).

### TMA checks

Compilation follows recognizable direct SSA uses of channel barriers and counts static copy
boxes in the handle's opening region. It recognizes tensor TMA loads (`tma_bar_ptr=`) and bulk
`cp.async.bulk` loads (`CopyBulkG2SOp`, and `CopyBulkG2SMulticastOp` by atom type, whose mask
takes the tensor multicast check); a bulk load completes on the issuing CTA's barrier, like a
`CtaGroup.ONE` tensor load, and is counted only directly inside an `elect_one()` block of the
handle's region: bulk copies do not elect a lane themselves, so one written directly in the
region (every lane issues it) leaves the stage unverified. A known wrong CTA group (a bulk load into a pair-scoped channel
included), undeclared multicast mode or complete byte-count mismatch raises `ProtocolError`.
Recognized copies report their group (`cp.async.bulk` for bulk loads), bytes per stage per CTA
and source locations. For multicast, the counted bytes are the SMEM destination box; in a
symmetric `Cluster.multicast` partition this equals the issuing CTA's delivery across its mask's
CTAs.

Unknown atoms, opaque pointers, nested/dynamic/conditional copies (other than `elect_one()`),
multicast masks not obtained from `Cluster.multicast`, or unsupported IR can leave this check
incomplete. Incomplete inspection alone does not reject; compilation continues with a warning
and `unverified` reasons in `describe()`. Positive wrong-group or undeclared-multicast evidence
still rejects even when byte accounting is incomplete. Validate raw, unverified and opted-out
uses independently. `verified` means the recognized group and byte accounting match; it does not
certify descriptor legality, the per-instruction size a bulk copy's layout lowers to (a bulk
copy moves whole contiguous vectors; give it flat views), disjoint writes or all async ordering.
You still supply `Tma(tx_bytes)` yourself.

Context atom checks inspect direct fields for MMA CTA span and unsupported group-TWO or
multicast G2S atoms. They do not recursively inspect nested objects and are not disabled by a
per-use `unchecked("tma")` opt-out.

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
`ab.wait`, `ab.tail`, `mufu.turn` and a `NamedBarrier`'s `compute_sync.sync`.

Add a region in a role:

```python
with self.trace_region("exp"):
    ...  # exp2 work
```

Disabled tracing emits no IKET operations. `iket` and `iket_waits` are part of the kernel's
cache key; `iket_key()` names them in the examples' kernel names.

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

The plan is per CTA. `cta=1` excludes leader-only roles in a pair; any rank excludes the roles
that don't run there. Its regions are work-item
granular, so ring occupancy and reuse estimates for an inner K loop are coarse, not
measurements of individual stages. Traced reuse counts come from the compile that built the
kernel; a cached kernel restores them. The exported plan is not a proof of liveness or
race freedom, and neither a plan nor an accepted kernel promises a performance threshold.

See the [pipeline profiler README](../../profiler/README.md#declarative-pipeline-analysis)
for critical-path analysis, measured timelines and capture tooling.
