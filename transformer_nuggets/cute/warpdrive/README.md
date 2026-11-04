# warpdrive

Write warp-specialized CuTeDSL kernels as separate methods, with channels for the handoffs:

```python
self.ab = Ring(3, Tma((128 + 256) * 64 * 2), Umma)
self.acc = Ring(1, Umma, Threads)
```

You choose the tiles, stages, warps, register budgets and schedule. The layer handles arrival
counts, phases, barrier init, TMEM allocation/free, warpgroup padding and CTA-pair bookkeeping.
Role bodies are ordinary CuTeDSL; raw barriers and `cutlass.pipeline` can live alongside them.

Experimental. Requires `nvidia-cutlass-dsl` 4.8 and a Blackwell SM100/SM103 GPU.
See [the guarantee contract](GUARANTEES.md) for the supported program, checked rejections,
caller obligations and known gaps; successful compilation is not a deadlock-freedom proof.

## A GEMM

BF16 `D = A @ B.T`: warp 4 loads A/B, warp 5 issues MMA, warps 0-3 store the result, and a
`WorkStream` hands out output tiles. Abridged from
[examples/persistent_gemm.py](examples/persistent_gemm.py):

```python
from dataclasses import dataclass

import cutlass
from cutlass import cute
from transformer_nuggets.cute.warpdrive import (
    Consumer, Producer, Raster, Ring, SmemPlan, StageView, StaticPersistent, Threads,
    WorkCursor, WorkStream, Tma, Tmem, TmemPlan, Umma, WarpSpecialized, role,
)

@dataclass(frozen=True)
class PersistentGemmParams:  # prologue values every role reads
    tma_a: cute.CopyAtom
    tAgA: cute.Tensor   # (TMA, m_tiles, k_tiles)
    tAsA: StageView     # TMA partition of the A stages: (TMA, stage)
    num_k_tiles: cutlass.Int32
    ...

class PersistentGemm(WarpSpecialized):  # abridged: the real one sizes stages per cluster
    def __init__(self, schedule=None, ab_stages=4, acc_stages=2):
        self.smem_plan = SmemPlan()
        a = self.smem_plan.buffer("A", cutlass.BFloat16, align=1024)
        b = self.smem_plan.buffer("B", cutlass.BFloat16, align=1024)
        self.ab = Ring(ab_stages, Tma((128 + 256) * 64 * 2), Umma, buffer={"A": a, "B": b})
        self.acc = Ring(acc_stages, Umma, Threads)                   # MMA fills, epilogue drains
        self.tmem_plan = TmemPlan().region("acc", 256, acc_stages)
        self.work = WorkStream(mapping=Raster(swizzle=8), schedule=schedule or StaticPersistent())

    @role(warps=4, regs=40)
    @cute.jit
    def load(self, params: PersistentGemmParams, *, ab: Producer, work: WorkCursor):
        ab = ab.views(A=params.tAsA, B=params.tBsB)  # s.view("A") is tAsA at s's stage
        while work.valid:
            m, n, _ = work.item
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.produce() as s:
                    cute.copy(params.tma_a, params.tAgA[None, m, k], s.view("A"),
                              tma_bar_ptr=s.barrier)
                    cute.copy(params.tma_b, params.tBgB[None, n, k], s.view("B"),
                              tma_bar_ptr=s.barrier)
            work.advance()
        return ab, work

    @role(warps=5, regs=40, cta="leader")  # leader-only in a CTA pair
    @cute.jit
    def mma(self, params: PersistentGemmParams, *, ab: Consumer, acc: Producer, work: WorkCursor, tmem: Tmem):
        tAcc = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        while work.valid:
            with acc.produce() as a:
                for k in cutlass.range(params.num_k_tiles, unroll=1):
                    with ab.consume() as s:
                        ...  # cute.gemm into tAcc[..., a.index] from s.view("A", None, None, kb), ...
            work.advance()
        return ab, acc, work

    @role(warps=range(4), regs=160)
    @cute.jit
    def epilogue(self, params: PersistentGemmParams, *, acc: Consumer, work: WorkCursor, tmem: Tmem):
        while work.valid:
            m, n, _ = work.item
            with acc.consume() as a:
                ...  # TMEM -> registers -> SMEM -> TMA store of tile (m, n)
            work.advance()
        return acc, work

    @cute.kernel
    def kernel(self, tiled_mma, tma_a, mA, tma_b, mB, tma_c, mC, a_layout, b_layout, ..., bound):
        self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout, ...})
        sA = self.buffer("A")  # the whole staged tensor
        ...  # TMA-partition the buffers, build the context
        params = PersistentGemmParams(tma_a=tma_a, tAgA=tAgA, tAsA=tAsA, num_k_tiles=..., ...)
        self.run_roles(params)
```

`params` is what you build in the kernel prologue: TMA atoms, partitioned tensors, layouts, loop
counts. The examples declare it as a frozen dataclass and annotate each role's `params` parameter
with it, so editors know every field's type and the object reaches the roles as itself. Frozen
means read-only: write into the tensors it holds, but its fields can't be rebound. A
`SimpleNamespace` still works too, with an unannotated `params`.

Swapping the schedule is the `schedule=` argument: `OneShot()`, `StaticPersistent()`,
`AtomicPersistent()`, `ClusterLaunchControl()` or your own `IndexSchedule`; the work order is
`mapping=` (a `Raster`, your own `WorkMap`, or a wrapped CUTLASS/FA4 scheduler order). The role code doesn't change. The
accumulator ring has two stages so the epilogue of one tile overlaps the mainloop of the next.

`begin_roles()` allocates and initializes barriers and payloads, allocates the `smem_plan`
buffers (see [Channel storage](#channel-storage)), and returns the `SmemAllocator` for your own
storage. `run_roles(params)` synchronizes, dispatches the roles and
frees TMEM after its users finish. Single-CTA TMEM allocation happens in `begin_roles`; CTA pairs
allocate after the cluster sync in `run_roles`.

## Roles

Put `@role` directly above `@cute.jit`. In the examples, `params` is positional and `*`
makes the injected dependencies keyword-only. The dispatcher supplies them by name. Positional
role signatures still work; annotations and names determine binding, not the position of `*`.
Declare every argument explicitly (`*args` and `**kwargs` are not supported).

- `warps=4` means warp index 4, not four warps. `warps=range(4)` means warps 0-3.
- `warps=[range(4), range(4, 8)]` creates two instances of the same method.
- `regs=160` is your `setmaxnreg` budget, not an inferred value. The default `None` leaves it
  unchanged. Budgets must agree within a warpgroup; unowned warps get the group's budget too.
  The layer applies each budget in a warpgroup-uniform branch before dispatching roles.
- `cta="leader"` runs only in the leader of a CTA pair; `"all"` is the default. Both are the
  same for a single CTA.

Arguments come from annotations. Named dependencies match attributes on the kernel:

| Parameter | Receives |
|---|---|
| `ab: Producer` / `ab: Consumer` | an end of `self.ab` |
| `s: Producer[Instance]` / `s: Consumer[Instance]` | a `self.s` end bound to this instance's slot |
| `stage: Instance` | the instance index, a Python int |
| `tmem: Tmem` | the kernel's TMEM view |
| `work: WorkCursor` | the `self.work` work cursor |
| `mufu: Turn` | a seat in `self.mufu`, a `Turnstile` |
| `bars: Barriers` | the base pointer of `self.bars` |
| `params: GemmParams`, or the one unannotated parameter | the object passed to `run_roles` |

A parameter receives `params` when it is annotated with the context's type (or a base class of it)
or is the role's only unannotated parameter. Anything else raises `ProtocolError` naming the
parameter: a missing channel name, an annotation that doesn't resolve (`ab: "Consumr"`), a layer
class used as an annotation (`work: WorkStream`, which suggests `WorkCursor`), two context
parameters, or roles annotating unrelated context types. `run_roles` checks the object it gets
against the annotated type and rejects mutable dataclasses.

### Constructor-selected roles

Keep fixed placement in the decorator, or select a `RoleConfig` from the kernel instance:

```python
from dataclasses import dataclass
from transformer_nuggets.cute.warpdrive import RoleConfig


@dataclass(frozen=True)
class AttentionRoles:
    softmax: RoleConfig
    mma: RoleConfig


class Attention(WarpSpecialized):
    def __init__(self, registers):
        self.roles = AttentionRoles(
            softmax=RoleConfig([range(4), range(4, 8)], regs=registers.softmax),
            mma=RoleConfig(12, regs=registers.other, cta="leader"),
        )
        # Declare channels and storage as usual.

    @role(config=lambda kernel: kernel.roles.softmax)
    @cute.jit
    def softmax(self, params, *, stage: Instance): ...
```

The selector is an ordinary host callable receiving the kernel and returning a `RoleConfig`.
It doesn't have to be a lambda. This is equivalent:

```python
def select_softmax(kernel: Attention) -> RoleConfig:
    return kernel.roles.softmax


# On the role method:
# @role(config=select_softmax)
```

Selectors run on first collection, after construction: `describe()`, a compile-cache lookup,
or kernel tracing can trigger it. The resolved placement is snapshotted for that instance;
construct a new kernel to change it. `RoleConfig` is frozen and copies lists of ranges into
immutable tuples. Neither the selector nor the configuration becomes device code.

`RoleConfig` has the same `warps`, `regs`, `cta` and `raw` fields as the decorator
shorthand. Don't mix `config=` with placement arguments. Omitting `regs` requests no register
adjustment, not auto-tuning. Conflicting choices in one warpgroup (including `None` versus a
numeric budget) are errors. Channel counts, scheduler placement and padding are derived from
the resolved configuration, and `describe()` shows it.

The `transformer_nuggets.cute.cache` compile helpers automatically include resolved role
placement and budgets via the kernel's `__cute_compile_key__()` hook, in addition to your
algorithm key. Equivalent configurations share an entry; different budgets or placements do
not. If you build your own cache, include `op.__cute_compile_key__()` too. Tiles, dtypes and
other algorithm choices still belong in your key.

The FA4 example exposes `Fa4Roles` and `fa4_fwd(..., roles=...)`. For another register bucket:

```python
from dataclasses import replace
from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import Fa4Roles, fa4_fwd

roles = Fa4Roles()
roles = replace(
    roles,
    softmax=replace(roles.softmax, regs=184),
    correction=replace(roles.correction, regs=96),
)
out, lse = fa4_fwd(q, k, v, roles=roles)
```

Changing the warp layout can also require changes to your tensor partitions; deriving barrier
counts does not rewrite the math's thread-to-data mapping.

### Params in, state back out

Use `self, params, *, ...state` in role signatures. `params` is the kernel-specific frozen
input dataclass; the rest includes the warpdrive objects the role uses.

**Return every stateful argument:** producer endpoints, consumer endpoints, work cursors and
turnstile seats. This is the same rule for every configuration, including IKET on/off and
single-/two-CTA kernels. No need to work out which endpoints need cleanup:

```python
@role(warps=5, regs=40)
@cute.jit
def mma(self, params: GemmParams, *, ab: Consumer, acc: Producer, work: WorkCursor, tmem: Tmem):
    while work.valid:
        ...  # use params, ab, acc, tiles and tmem
        work.advance()
    return ab, acc, work
```

`self` and `params` are input-only. The injected instance index, TMEM view and raw barrier
pointer are also inputs, not loop-carried state, so they are not returned. Stage handles must
be committed/released inside the role; they are not endpoint state to return.

Return one object or a tuple. Tuple order and local names do not matter: `return work, acc, ab`
is equivalent. Missing, duplicate, unsupported or foreign state is an error. The layer uses the
returned state for the cleanup that applies (producer tails and remaining IKET work ranges),
without recovering execution state from Python frames. Raw role results are ignored.

Return the current loop-carried values, not snapshots saved before the loop. Helpers still
return updated state to their callers (`ab = helper(ab, ...)`). The layer checks ownership and
presence, not whether an old snapshot is stale.

## Channels

`Ring(n, producer, consumer)` cycles through stages 0..n-1. `Slots(n, producer, consumer)` lets
you choose the slot; each slot has its own phase. Slots need Python int indices in `[0, n)`
and support at most 31 slots. An instance-bound end still takes the slot explicitly:
`s.wait(stage)`. Passing another instance's slot is an error.

The mechanisms say who signals:

- `Tma(tx_bytes)`: TMA completes the full barrier. Supply the bytes per stage per producing CTA;
  the layer scales the armed count for a pair.
- `Umma`: one elected thread issues `tcgen05.commit`.
- `Threads`: every thread of every bound role arrives. Multiple consumers form a join.

A `Tma` or `Umma` side must belong to exactly one single-warp role.

### Who participates in a barrier?

Participation comes from role signatures and warp placement, not analysis of which threads
reach a call. For `Ring(2, Umma, Threads)`, a single-warp MMA producer contributes one UMMA
completion to each full barrier. An epilogue consumer on warps 0-3 contributes 128 arrivals
to each empty barrier. The producer cannot reuse a stage until those arrivals complete.

`Slots` uses the same rule per slot: instance-bound ends contribute only to their own slot.
CTA-pair placement also determines which CTAs contribute and where arrivals go; inspect the
per-CTA counts with `print(op.describe())`.

With `Threads`, every thread of the bound role must participate in each managed signal.
Predicate the work, not the handoff:

```python
# All role threads wait/release; only valid lanes write output.
with acc.consume() as h:
    if output_in_bounds:
        ...  # predicated output store

# Not the Threads contract: only some role threads wait/release.
if output_in_bounds:
    with acc.consume() as h:
        ...
```

Existing kernels also use one elected arrival per warp, relay warps for async-copy completion,
and sparse work that skips stages. Those are valid patterns, but their participation need not
match the default whole-role count. Local handle checks do not prove convergence, matching
producer/consumer iteration counts, or that an elected signal follows every reader.

For nonstandard signals, use count overrides and `commit(signal=False)` or
`release(signal=False)`, or reserve raw barriers; see [Dropping down](#dropping-down). An override
only changes the expected count. It does not select participating lanes or insert the join that
makes their signal safe. An mbarrier counts arrivals, not thread identities.

```python
with ab.produce() as h:  # acquire empty stage; commit on exit
    ...  # load into stage h.index
with ab.consume() as h:  # wait for full stage; release on exit
    ...
```

Use explicit handles when lifetimes overlap without nesting. In the FA4 example K and V share
one ring. Each iteration needs V(i-1) for the PV MMA and K(i) for the next QK MMA, and V(i-1) is
released before K(i):

```python
for i in cutlass.range(n_blocks - 1, unroll=1):
    v = kv.wait()  # V(i-1)
    k = kv.wait()  # K(i)
    pv(o, p, v.index)  # O += P @ V(i-1)
    v.release()  # V(i-1) is free; K(i) is still held
    qk(s, q, k.index)  # S = Q @ K(i)^T
    k.release()
```

Nested `with` blocks can't express this: the inner handle would have to close first.

Handles must close once, in the same dynamic region (loop/branch) that opened them. Violations
raise `ProtocolError` with source lines during compilation. `h.retire()` closes a producer
handle without publishing, for a final unused acquire; it is not allowed on tailed pair channels.

Storage is separate from the channel unless you bind it: `payload=(dtype, shape)` allocates small
per-stage SMEM accessible as `h.data`, and `buffer=` binds buffers of an SMEM plan (next
section). `commit_fence` and `release_fence` accept `"tmem_store"` or `"smem_async"`; TMA
producers cannot have a commit fence.

## Channel storage

Declare SMEM once in an `SmemPlan`, bind buffers to the channels that hand them over, and let
each handle select its own stage:

```python
self.smem_plan = SmemPlan()
q = self.smem_plan.buffer("Q", cutlass.BFloat16, align=1024)       # stage_axis=-1: last mode
k = self.smem_plan.buffer("K", cutlass.BFloat16, align=1024)
v = self.smem_plan.buffer("V", cutlass.BFloat16, align=1024, overlaps=k)  # K's bytes, no new ones
self.q = Slots(2, Tma(q_bytes), Umma, buffer=q)
self.kv = Ring(3, Tma(kv_bytes), Umma, buffer={"K": k, "V": v})

# Kernel prologue: layouts (including swizzles) are trace-time values, so they bind here.
self.begin_roles(work=bound, layouts={"Q": sQ_layout, "K": sK_layout, "V": sV_layout})
sK = self.buffer("K")  # the whole staged tensor, for partitions and fragments

# A role registers the partitions it uses; handles project them at their own stage.
kv = kv.views(K=StageView(thr_qk.make_fragment_B(sK), axis=3))
with kv.consume() as h:
    tSrK = h.view("K")  # tSrK[None, None, None, h.index]
    h.data  # the canonical view of every bound buffer at h.index ({name: view} for several)
```

- A buffer has a dtype, an alignment, an explicit stage axis (the last mode by default, as in
  CUTLASS's staged layouts) and, at `begin_roles`, a plain or swizzled layout. Buffers are
  allocated after the layer's barriers, payloads and TMEM holder, in declaration order. That is
  where a prologue's hand allocation sat in the examples, so the generated code is unchanged.
- `overlaps=` reuses another buffer's bytes from its first byte (FA4's V over K, or O over Q).
  It must fit inside and not need a stricter alignment. The plan owns SMEM accounting: overlaps
  and several channels binding one buffer count once. `describe()` lists each buffer's bytes,
  stages and channels after a compile.
- The plan is per CTA: in a `CtaPair` each CTA holds its own copy at the same offset (where
  2-CTA MMAs read the peer's operands). A buffer implies no DSMEM publication and adds no fence:
  `remote_scope` and fences are what they would be without it, and pair-scoped channels may bind
  buffers (a payload may not).
- Storage projects the handle's stage snapshot (`h.index`); there is no second cursor. The
  binding checks that each buffer's stage mode, and each registered `StageView`'s, has exactly
  the channel's stage count, and that view names are buffers of the channel.
- `end.views(...)` returns an end that shares the original's state: use it and return it.
  `h.view(name, *coord)` takes the other modes' indices in the same slice
  (`s.view("A", None, None, kb)` is `tCrA[None, None, kb, s.index]`), which keeps the
  generated address arithmetic identical to hand-indexed code.
- The layer does not infer partitions, choose views by role, or certify layout transformations:
  you build every partition and the tensors, layouts and pointers stay yours.

## TMEM and turnstiles

```python
self.tmem_plan = (
    TmemPlan().region("S", 128, 2).region("O", 128, 2).alias("P", of="S", offset=64, cols=64)
)
```

Used columns are the sum of `cols * slots` over regions; aliases sit inside another region and
add nothing. Here that is 128 * 2 + 128 * 2 = 512, all of TMEM. The allocation rounds up to a
power of two (32 to 512 columns). `print(self.tmem_plan.describe())` shows the map:

```
  tmem S   [0, 128), [128, 256)
  tmem O   [256, 384), [384, 512)
  tmem P   [64, 128), [192, 256) (inside S)
```

`tmem.ptr("S", stage)` gives an fp32 TMEM pointer; `tmem.cols("S", stage)` gives its column
offset. Reinterpret packed bf16 yourself. Capacity and alias bounds are checked. You must order
accesses to aliased regions.

`self.mufu = Turnstile()` and `mufu: Turn` give role instances a round-robin critical section:
`with mufu.turn():`. FA4 uses it around exp2. `Turnstile(enabled=False)` emits nothing.

## Scheduling

```python
self.work = WorkStream(mapping=Raster(swizzle=8), schedule=ClusterLaunchControl(stages=2))


@role(warps=4, regs=40)
@cute.jit
def load(self, params: PersistentGemmParams, *, ab: Producer, work: WorkCursor):
    while work.valid:
        m, n, _ = work.item
        ...
        work.advance()
    return ab, work
```

A work stream is three independent pieces, each swapped by one argument:

- `schedule=` (an `IndexSchedule`): how a work unit (a CTA, or a cluster of a pair) gets its
  next linear index.
- `mapping=` (a `WorkMap`): the mapping from an index to the work item `work.item` returns.
- `ready=` (a `Readiness`): an optional wait before the first decode, such as `PdlWait()`.

Role code doesn't change when any of them does. A role parameter `work: WorkCursor` receives the
cursor over the stream declared as `self.work`. `WorkStream()` defaults to
`Raster(along="m", swizzle=1)` and `OneShot()`. The built-in schedules:

- `OneShot()`: one work item per CTA. With the default `Raster()` the launch grid is the tile
  space, `(m_tiles, n_tiles, l)`, and the item is read from `blockIdx`. When `n_tiles` or `l`
  exceeds the 65535 limit of `gridDim.y`/`z`, the launch falls back to a 1-D grid and the
  kernel decodes the index. The check costs about 25-50 ns on 512^3 to 2048^2 GEMMs (GB300).
  Launch with `self.launch_config(bound)` or `bound.grid()` to get the tile grid; a
  hand-built `(total, 1, 1)` grid also works, through the decode.
- `StaticPersistent()`: each CTA strides by the persistent grid size.
- `AtomicPersistent()`: claims work through a global atomic counter.
- `ClusterLaunchControl(stages=2)`: Blackwell cluster launch control.

One-shot/static decode work locally. Atomic/CLC add a scheduler warp and broadcast work through
a ring. Counts, batch and SM count are runtime values. `WorkStream(ready=PdlWait())` waits for a
programmatic dependent launch before the first decode.

Use `while work.valid`, with exactly one `work.advance()` directly in the loop body.
Every managed role must drain the stream under a persistent schedule.

The host/entry-point setup, from [persistent_gemm.py](examples/persistent_gemm.py):

```python
workspace, max_ctas = op.work.launch_args(device)  # host, outside the compiled entry point
# In __call__ (@cute.jit):
bound = self.work.bind((m_tiles, n_tiles, 1), workspace, max_ctas)
self.kernel(..., bound).launch(**self.launch_config(bound), stream=stream)
# In the kernel:
smem = self.begin_roles(work=bound)
```

`bind` returns a `BoundWork`: the mapping's bound space, the Int32 item count `total` and the
workspace. Static/atomic schedules need `max_ctas`. The atomic int32 counter starts zeroed and
resets after each launch; don't share its workspace across concurrent launches. Keep the
`launch_args` pair in the entry point's signature even for schedules that don't read it, so
swapping the schedule leaves the ABI alone.

Extents are runtime values converted to Int32, so the space is checked in the entry point's host
code, not in the kernel:

- The mapping counts its items once, on the host (`count64`, an Int64 that saturates at 2^31),
  and `bind` narrows the count to the Int32 `total` the kernel receives.
- A zero or negative extent is an empty space. Every schedule launches one work unit (one
  cluster for a pair), every role's `work.valid` is false at once, and the atomic counter is
  left zeroed. `Raster.bind` stores it as `(1, 1, 0)`, so the kernel's `bound.space` reads
  that rather than the raw extents.
- Item count, grid and indices are Int32. `OneShot`/`ClusterLaunchControl` need
  `total * cluster_size` <= 2^31-1; static/atomic need `total + launched units - 1` <= 2^31-1
  (the last unit's terminal index). `bound.grid()` makes a larger space zero-wide, so the
  launch raises a CUDA invalid-configuration error instead of covering a wrapped space. A
  hand-built grid gets no check.

### Custom schedules and mappings

`IndexSchedule`, `WorkMap` and `Readiness` are the public protocols the built-ins implement.
Subclass them; their docstrings in [scheduling.py](scheduling.py) list every method, where it runs
and what it must guarantee. A local schedule defines `next_index`, and optionally
`first_index`:

```python
@cute.jit
def _countdown(index, problem):  # a valid index, or the terminal index `total`
    return cutlass.select_(index >= 0, index, problem.total)


@dataclass(frozen=True)
class ReverseStatic(IndexSchedule):
    """Unit b runs total-1-b, total-1-b-U, ... (U = launched units)."""

    def first_index(self, problem):  # runs in every role: launch values and `problem` only
        return _countdown(problem.total - 1 - problem.first_unit(), problem)

    def next_index(self, index, problem):  # the next index after a valid `index`
        return _countdown(index - problem.units(), problem)


self.work = WorkStream(mapping=Raster(swizzle=8), schedule=ReverseStatic())
```

A broadcast schedule sets `broadcast = True` and `stages`, and defines
`claim_next(fetch, problem, index)` (plus `close_claims` and `fetch_channels` if it needs them).
It runs on the layer's scheduler warp, which receives the previous index, so a stateful sequence
needs no extra storage. The layer still owns distribution: the work ring and its counts,
publication to both CTAs of a pair, leader-only claims, the drain and the scheduler's register
budget. `close_claims` closes claim state once the unit stops claiming (`AtomicPersistent`
reports and resets its counter there). It is not a teardown or fixup hook: ring consumers
release a slot when they receive an item, so when it runs no item's work is known to be done.

Rules for every schedule: return a valid index in `[0, total)` or a terminal index `>= total`,
never a negative one; compute `first_index`/`next_index` identically in every role;
`uses_max_ctas` and `uses_workspace` declare which launch arguments it reads; `index_limit`
must bound the largest index or grid extent (the default, `total + units - 1`, bounds strided
sequences); `get_key()` names every compile-time field.

A mapping defines `words`, `bind`, `count64`, `decode` and `get_key`, and optionally `unpack`
and `skips`:

```python
@dataclass(frozen=True)
class ReverseM(WorkMap):
    words = 3
    decodes_any_index = True  # Raster arithmetic: defined for every index

    def bind(self, space):  # entry point: normalize; an empty space must stay safe
        return Raster().bind(space)

    def count64(self, space):  # entry point: Int64 item count, saturating at 2**31
        return Raster().count64(space)

    def decode(self, index, space):  # device: (keep, words) of a valid index
        keep, (m, n, batch) = Raster().decode(index, space)
        return keep, (space[0] - 1 - m, n, batch)

    def get_key(self):
        return "reverse_m"
```

- `decode` returns `(keep, words)`. The layer calls it only for valid indices, `0 <= index <
  total`, after `ready`, so it may read per-item tables. `keep = False` drops the item for
  every role; only a mapping that sets `skips = True` may return a runtime `keep` (any other
  returns the constant `True`, checked at trace time).
- `decodes_any_index = True` is the one exception: set it only when `decode` is a pure function
  of launch-time data defined for every nonnegative Int32 index, including empty spaces (no
  table reads, no division by a runtime value that may be zero). The layer may then decode
  speculatively (past the end, or before `ready` on the tile grid) and discard the words, which
  keeps a branch off the decoding warps; a skipping mapping's `keep` is still only used for
  valid indices. `Raster`, `CutlassRaster` and `Fa4Mapping` set it; a subclass that overrides
  `decode` must set it again.
- A work list built on the device binds its host-visible capacity as the count and reads the
  true length in `decode`'s `keep`; the host never reads device memory.
  [packed_work.py](examples/packed_work.py) is such a mapping (`PackedWork`, rows of an int32
  table plus a device-side length, consumed with `ready=PdlWait()`).
- Repeated outputs don't need repeated indices. Split-K maps distinct `(tile, split)` items with
  their K ranges and drops empty ranges; the partials are reduced by a later kernel, not by the
  schedule. FA4 split-KV decode adds split id and count columns the same way.

### External schedulers

[adapters.py](adapters.py) wraps an external scheduler's index-to-tile mapping as a
`WorkMap`, so it swaps in through `mapping=` and runs under any schedule. Composing it with
`StaticPersistent()` reproduces the library's own per-CTA index sequence (not its lifecycle):

```python
from flash_attn.cute.tile_scheduler import StaticPersistentTileScheduler

cutlass_order = CutlassRaster(swizzle=8)  # CUTLASS StaticPersistentTileScheduler order
self.work = WorkStream(mapping=cutlass_order, schedule=StaticPersistent())  # CUTLASS's sequence
self.work = WorkStream(mapping=cutlass_order, schedule=ClusterLaunchControl())  # claimed by CLC
self.work = WorkStream(
    mapping=Fa4Mapping(StaticPersistentTileScheduler), schedule=ClusterLaunchControl()
)
```

The kernel keeps binding `(m_tiles, n_tiles, l)`; the adapter derives the external space from it.
`Fa4Mapping(SingleTileLPTScheduler)` (FA4 runs it with `OneShot()`) sizes its L2 sections from
the K/V bytes per head, so its stream binds `(num_block, num_head, num_batch, seqlen_k)`; its
compile-time arguments (`lpt`, `lpt_global`, `head_dim`, `head_dim_v`, `element_size`) are
adapter fields and in its key. The FA4 example takes `mapping=` too. Limits:

- Only index-addressable schedulers adapt: those whose state is one linear index and whose tile
  is a function of it. FA4's `SingleTileScheduler` (3-D `blockIdx`) is rejected. QuACK's and
  the CLC/dynamic schedulers own pipelines, SMEM and barriers that would duplicate the layer's
  distribution; use their mapping with a warpdrive schedule instead.
- The external `advance_to_next_work`/`prefetch_next_work`/`producer_tail`/`get_grid_shape`
  aren't called; the schedule picks indices and the layer launches its own grid.
- Items are per work unit: tiles are work-unit tiles (FA4's params use a `(1, 1)` cluster
  shape), so a `CtaPair` gets one item per cluster.
- CUTLASS pads swizzle bands and hands out the padded tiles; the adapter drops them
  (`keep = False`). Both adapters have an exact `count64`, so spaces past Int32 are rejected at
  launch like `Raster`'s.
- An empty space has no items: CUTLASS's order binds like `Raster`; FA4's params are built
  over one tile (an empty KV as one key), so no external arithmetic divides by zero, and the
  count comes from the normalized extents. The FA4 adapter has no split-KV; a split-KV mapping
  carries split columns itself (see above).
- `CutlassRaster` reimplements CUTLASS's order natively (it is `Raster` over the
  padded space) rather than wrapping the deprecated `cutlass.utils` copy, so it imports no
  CUTLASS scheduler and emits no deprecation warning. Provenance: the BSD-3 CuTeDSL helper
  NVIDIA/cutlass `examples/python/CuTeDSL/helpers/static_persistent_tile_scheduler.py` at
  `e13f5cd`, vendored as the test oracle in `test/cutlass_static_persistent_tile_scheduler.py`;
  the tests compare per-CTA sequences against it.
- `Fa4Mapping` imports `flash_attn.cute` when constructed; warpdrive itself doesn't need it.

### Raw roles

A raw role takes no `WorkCursor`. Under `OneShot` it reads its unit's item with
`self.work.single_item()`, not `blockIdx`, which is wrong on the 1-D fallback grid, a hand-built
grid, a pair or a non-`Raster` mapping:

```python
@role(warps=range(4), regs=160, raw=True)
@cute.jit
def epilogue(self, params, *, tmem: Tmem):
    valid, (m, n, _) = self.work.single_item()  # runs ready= and drops items like a managed role
    ...
```

`valid` is false for an empty space or a dropped item. Persistent and broadcast schedules raise
`ProtocolError`: a role with several items takes `work: WorkCursor` and loops.

## CTA pairs

```python
self.cluster = CtaPair()  # cluster (2, 1, 1), CtaGroup.TWO
# Put cta="leader" on the MMA role.
```

This does not convert your atoms or layouts. Use a 2x MMA tile M, build the MMA/TMA atoms with
`self.cluster.cta_group` and `self.cluster.layout_vmnk()`, and partition by
`self.cluster.rank()`. Call `rank()` from converged warp code, not a lane-0 branch.
The persistent GEMM and FA4 examples have both versions.

The layer derives barrier locations, per-CTA counts, remote targets, commit masks, TMA byte
scaling, producer tails, TMEM allocation/free and the exit cluster barrier. Work items are per
cluster; atomic/CLC claims run on the leader and publish to both CTAs.

`remote_scope` defaults to `"cta"` without a payload and `"cluster"` with one. CTA scope is for
TMA/MMA/TMEM handoffs ordered by their own fences. If your pair channel publishes DSMEM or global
memory written by user code, set `remote_scope="cluster"`; the layer cannot detect this.
Layer-owned payloads are CTA-local, so pair-scoped channels cannot have them.

## What the layer derived

For the 2-CTA persistent GEMM with CLC:

```python
op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair())
compile_persistent_gemm(op)  # fills in the TMA check status
print(op.describe())
```

```text
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

`describe()` returns a string. Set `op.trace_protocol = True` before the first `describe()` or
compile to include the traced acquire/commit/wait/release sequence and source lines after
compilation; include `trace_key()` in custom compile-cache keys (the examples do) so a cached
untraced kernel isn't reused. This is not a runtime history.

Compilation also checks TMA copies using channel barriers against the required CTA group and
armed byte count. A confirmed mismatch raises `ProtocolError`. Unknown atoms (including
multicast), opaque pointers, nested/dynamic copies or changed CuTeDSL IR can leave the check
incomplete: it fails open with one warning per kernel and `unverified` reasons in `describe()`.
After compilation, recognized copies report their group, bytes per stage per CTA and source
lines. This check does not infer `Tma(tx_bytes)` for you.

## Dropping down

Use `Channel(override=...)` (also accepted by `Ring`/`Slots`) to replace `full_counts`,
`empty_counts`, `tx`, `full_mask` or `empty_mask`. Counts can be an int, a per-slot list or a
`{cta_rank: list}` mapping. `describe()` shows both the derived and overridden values.

```python
self.acc = Ring(2, Umma, Threads, override={"empty_counts": 4})
# Consumer: signal once per warp yourself, rather than once per thread.
h = acc.wait()
...  # your work, fences and four elected arrivals
h.release(signal=False)
```

`h.commit(signal=False)` works the same way on the producer side. These close the handle for the
checker without emitting its signal or fence; raw code must honor the effective counts/targets.

```python
self.acc_bars = Barriers(2)  # user initializes between begin_roles/run_roles
# Barriers(2, counts=[1, 128]) lets the layer initialize them instead.


@role(warps=range(4), regs=160, raw=True)
@cute.jit
def epilogue(self, params, *, acc_bars: Barriers, tmem: Tmem):
    # acc_bars is a pointer; user code drives the barriers
    ...
```

Raw roles take `Instance`, `Tmem`, `Barriers` and the context, not channel ends, work cursors or
turnstile seats; under `OneShot` they read their item with `self.work.single_item()`. They skip
protocol checks and the persistent-stream requirement but still participate in register padding
and TMEM teardown. Build your own `cutlass.pipeline` with `self.barrier_ptr("acc_bars")` and
`defer_sync=True` between `begin_roles` and `run_roles`.
Named barrier IDs 1 and 2 are reserved.

Raw state is available without a raw role:

```python
end.state  # assignable: Ring PipelineState; Slots Int32 phase mask
end.index  # Ring's next stage
end.phase()  # Ring; use phase(slot) for Slots
end.full_barrier(i), end.empty_barrier(i)
end.targets, end.scope  # target CTA ranks; effective counts, masks, tx, tail
end.channel.derived_scope  # before overrides
h.index, h.phase, h.full_barrier, h.empty_barrier  # h.barrier aliases full_barrier
tmem.base
```

Barrier pointers are local; the peer has barriers at the same offsets. For a check the code
cannot express, opt out at the relevant site:

```python
with unchecked("cross_region", reason="V(i-1) is released in the next iteration"):
    v_prev.release()
```

The checks are `"cross_region"`, `"unclosed"` (around the acquire/wait) and `"tma"` (around the
barrier access). A reason is required. `describe()` lists each opt-out, its source line and what
it suppressed; explicit TMA opt-outs are distinguished from unverified copies.

## IKET tracing

Enable before compiling; include `iket_key()` in custom compile-cache keys (the examples do):

```python
op = PersistentGemm(schedule=StaticPersistent()).enable_iket(waits=False)
fn = compile_persistent_gemm(op)
```

`enable_iket()` adds role ranges and blocking waits named `ab.acquire`, `ab.wait`, `ab.tail`,
`mufu.turn`, etc. `waits=False` omits wait ranges. Roles taking `WorkCursor` get one range per work
item; other roles get one over their body. Add a region inside a role:

```python
with self.trace_region("exp"):
    ...  # exp2 work
```

Disabled tracing emits no IKET ops. Fine-grained waits can distort timing; start with role ranges.

Run a single launch under IKET, then join its capture to the exported plan:

```bash
run-iket -o traces/gemm --clobber profile --postprocess all -- python profile_kernel.py
```

```python
from transformer_nuggets.cute.warpdrive import plan_from_warpdrive
from transformer_nuggets.cute.profiler.pipeline import (
    analyze_pipeline,
    load_iket_capture,
    write_pipeline_perfetto,
)

plan = plan_from_warpdrive(op, cta=0)  # after a compile, for traced reuse counts
timeline = plan.schedule(iterations=8)
measured = load_iket_capture("capture.trace.json", timeline, cta=(0, 0, 0))
analysis = analyze_pipeline(timeline, measured)
write_pipeline_perfetto("gemm.pftrace", timeline, measured, analysis=analysis)
```

The plan is per CTA (`cta=1` omits leader-only roles) and work-item-granular. Occupancy and depth
estimates for rings used inside a work item's inner loop are coarse, not per-stage measurements.
See the [pipeline profiler](../profiler/README.md#declarative-pipeline-analysis).

## Gotchas

- Helpers taking channel ends, turnstile seats or work cursors must return them; reassign at the
  call site (`s, stats = step(s, stats)`) so CuTeDSL carries their state through loops.
- Define kernels in source files. CuTeDSL re-reads them for AST preprocessing.

## Examples and timings

- [gemm.py](examples/gemm.py): three roles, two rings, one CTA per tile through
  `WorkStream(schedule=OneShot())`.
- [persistent_gemm.py](examples/persistent_gemm.py): all four schedules, any `mapping=`,
  double-buffered accumulators, TMA stores, `CtaPair()` and configurable `tile_n`.
- [fa4_fwd.py](examples/fa4_fwd.py): FA4-style forward, two softmax instances, correction, slots,
  fan-out/join, turnstile, S/P aliasing, split-P, persistence, `mapping=` and CTA pairs;
  channel storage with V over K and optional O over Q (`overlap_o_q=True`, one work item per
  CTA).
- [packed_work.py](examples/packed_work.py): `PackedWork`, a work list built on the device
  (capacity on the host, length read in `decode`).

GEMMs take contiguous, 16-byte-aligned bf16 `a: (M, K)`, `b: (N, K)` on one device, with K and N
multiples of 8. `out` must be bf16 `(M, N)` on the same device, contiguous, aligned and disjoint
from both inputs. Empty outputs return early; K=0 writes zeros.

FA4 takes contiguous, 16-byte-aligned bf16 `(batch, seqlen, heads, 128)` tensors, matching K/V
shapes and matching batch/head counts. Key length must be a positive multiple of 128; query
length may be ragged. It imports `flash_attn.cute` and `quack` helpers. These are teaching kernels:
no masking, varlen, GQA or other head dimensions.

Indicative CUDA-graph replay medians from the November 2026 GB300 runs, not a tuned benchmark:
bf16, CuTeDSL 4.8.0, PyTorch 2.15 nightly cu132, FA4 at `849b5daa`. The numbers below are from the
initial layer, CTA-pair and tile-N experiments in the
[original README](https://github.com/drisspg/transformer_nuggets/blob/49b9713/transformer_nuggets/cute/warp_roles/README.md),
not new measurements of this revision. FA4 shapes below are batch x heads x sequence length,
with head dim 128.

| Workload | Reference | warpdrive |
|---|---:|---:|
| GEMM 8192^3, 2-CTA persistent | `torch.mm` 694 us | 705 us |
| GEMM 4096^3, 2-CTA, `tile_n=224` | `torch.mm` 89.9 us | 89.7 us |
| FA4 4x16x4096, matched 1-CTA non-persistent schedule | FA4 375 us | 373 us |
| FA4 4x16x4096, 2-CTA CLC + split-P | FA4 default 337 us | 346 us |

## Tests

```bash
pip install -e ".[cute,dev]" pytest-xdist
pytest -n 12 test/test_cute_warpdrive*.py           # every change: ~30 s on a GB300
pytest -n 12 -m fuzz test/test_cute_warpdrive*.py   # quick fuzz, when you want it: ~2-3 min
```

Fuzz tests are skipped by default. Long campaigns (runtime, static, mutate; sanitizer and
CUTracer wrappers) run from `scripts/fuzz_warpdrive.py`; see `GUARANTEES.md`.

FA4 tests skip without `flash_attn.cute`. Coverage includes numerical correctness, determinism,
shape reuse, all scheduler/CTA-pair configurations, protocol errors, raw/managed synchronization,
TMA checks and the IKET-to-profiler path.

## Still to do

- Use CUTLASS pipeline objects where the semantics match.
- Larger clusters, TMA multicast, round-robin role instances and cp.async producers.
