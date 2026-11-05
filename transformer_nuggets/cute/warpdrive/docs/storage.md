# Storage

[Back to warpdrive](../README.md)

```python
self.smem_plan = SmemPlan()
a = self.smem_plan.buffer("A", cutlass.BFloat16, align=1024)
b = self.smem_plan.buffer("B", cutlass.BFloat16, align=1024)
self.ab = Ring(4, Tma(stage_bytes), Umma, buffer={"A": a, "B": b})
```

A channel doesn't have to own storage. You can keep allocating tensors yourself and use
`h.index`. `SmemPlan` is the option for declaring the buffers once and letting each handle
select its stage. It doesn't choose a layout or change how the channel synchronizes.

## Allocate in the prologue

```python
# Inside the kernel; a_layout and b_layout include all four stages.
smem = self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout})
sA = self.buffer("A")
sB = self.buffer("B")
...  # partition sA/sB for TMA and MMA, build params
self.run_roles(params)
```

`SmemPlan.buffer(name, dtype, *, align=16, stage_axis=-1, overlaps=None)` declares a buffer. Its layout binds
later, in `begin_roles(layouts=...)`, because it often depends on the tiled MMA built in the
entry point. Supply one layout for every plan buffer, including overlaps. Plain and swizzled
layouts are supported.

Buffers allocate in declaration order after the internal barriers, payloads and TMEM bookkeeping,
using their declared dtype, alignment and supplied layout, including its swizzle.
You can keep allocating through the returned `smem` allocator. `self.buffer(name)` returns
the whole tensor after `begin_roles()`; accesses through it or prologue tensors bypass handle
stage selection and do not acquire channel ownership.

The stage mode defaults to the last layout mode; set `stage_axis` if it is elsewhere.
Its extent must be static and match the channel's stage count. A buffer must belong to this
kernel's plan. `buffer=` accepts one buffer or a nonempty dictionary of named buffers;
`payload=` and `buffer=` are mutually exclusive.

## The handle selects the stage

With one buffer, `h.data` is a tensor. With several, it is a dictionary of tensors:

```python
with ab.produce() as h:
    stage_a = h.data["A"]
    stage_b = h.data["B"]
    ...  # the canonical A/B views for h.index
```

Canonical means the original buffer layout with the stage mode removed. For a TMA or MMA
partition, register a `StageView` on the endpoint:

```python
ab = ab.views(A=StageView(tAsA, axis=1), B=StageView(tBsB, axis=1))
with ab.produce() as h:
    cute.copy(tma_a, tAgA[None, m, k], h.view("A"), tma_bar_ptr=h.barrier)
    cute.copy(tma_b, tBgB[None, n, k], h.view("B"), tma_bar_ptr=h.barrier)
```

Rebind the returned end and return it from the role. `views()` keeps the same endpoint state
and checks the partition's stage extent. It doesn't prove that the supplied tensor actually
addresses the declared buffer. Views registered inside a loop hold that loop's values;
use `h.view()` only inside the region that registered them.

`h.view("A")` uses the registered partition when present, otherwise the canonical view.
It always selects `h.index`, the handle's snapshot, not the endpoint's next stage. Omitting
the name is allowed only when there is one buffer. `h.data` always returns the canonical
view, even after registering partitions.

Additional positional coordinates index the other modes in the same slice. For an MMA
fragment with stage axis 3:

```python
ab = ab.views(A=StageView(tCrA, axis=3))
# Inside an acquired handle:
a_fragment = h.view("A", None, None, kb)  # tCrA[None, None, kb, h.index]
```

These snippets assume the usual TMA/MMA partitions from the GEMM examples. Storage binding
is not a replacement for building those partitions.

## Overlapping buffers

```python
k = self.smem_plan.buffer("K", cutlass.BFloat16, align=1024)
v = self.smem_plan.buffer("V", cutlass.BFloat16, align=1024, overlaps=k)
self.kv = Ring(kv_stages, Tma(stage_bytes), Umma, buffer={"K": k, "V": v})
```

V reuses the root buffer's first byte, recast to V's dtype, swizzle and layout, without
allocating storage. It must fit the backing allocation and cannot require stricter alignment.
Static byte sizes are checked during allocation; dynamic sizes do not get that static bounds check. Several
channels can bind the same buffer without allocating it again.

Nothing in `overlaps=` orders K and V. Your channel epochs must prevent a reader of K from
racing a writer of V. If two staged layouts overlap across *different* stage numbers, one
ring's normal reuse ordering is not enough.

In a cluster, each CTA allocates its own copy at the same offset. A bound buffer is allowed
on a pair-scoped channel, unlike a layer-owned `payload`. Binding does not imply DSMEM
publication: `remote_scope` still defaults to `"cta"`, and no fence is added. For ordinary
memory read by the peer, use the [pair memory-scope rules](clusters.md#memory-scope).

## Check declared aliases

```python
op.alias_check = "warn"  # default; set before compiling
compile_kernel(op)  # your kernel's compile helper
print(op.describe())
```

The diagnostic checks declared SMEM overlaps and `TmemPlan.alias` regions. It only knows
about accesses you annotate. Set `"strict"` to raise `ProtocolError` on a potential hazard,
or `"off"` to skip the analysis. Strict mode does not turn an unverified case into an error.
The analysis runs while tracing the kernel. A `jit_cache` compile keys on `alias_check`
through `op.__cute_cache_key__()`, so a warn/off build cannot satisfy a strict request, and a
kernel loaded from the cache gets the traced report back, warning included
([compile keys](roles.md#compile-keys)).

For SMEM, annotate the view where the access happens:

```python
with kv.produce() as h:
    dst = h.view("K", write=True, by="tma")
    cute.copy(tma_k, src, dst, tma_bar_ptr=h.barrier)
```

If another helper already holds the tensor, `h.access("K", write=True, by="tma")` records
that access without returning a view. Use `by="threads"` for ordinary loads/stores,
`"tma"` for TMA, or `"umma"` for MMA operand reads. `read=True` and `write=True` can both
be set. Place the annotation at the operation, not just where you first obtain a pointer.

TMEM has the same two forms:

```python
# At the MMA that writes S, inside its producer handle:
s_ptr = tmem.view("S", stage, write=True, by="umma")
...  # build/use the tensor for this MMA

# At an access through a tensor built earlier:
tmem.access("P", stage, read=True, by="umma")
```

`tmem.view()` returns an fp32 pointer, not a CuTe tensor. `tmem.access()` only records the
access. TMEM accepts `by="threads"` for `tcgen05.ld/st` and `by="umma"` for MMA. The slot is
a Python int naming a [TMEM plan slot](channels.md#tmem).

For narrower TMEM footprints, supply `warps`, `lanes` and `cols`. Lanes and columns are
ranges or callables of the warp index; columns are relative to the named region. By default,
a thread access covers that warp's 32-lane TMEM subpartition, an MMA covers all 128 lanes,
and both cover the region's full width. These are declarations by the author, not inferred
addresses.

### Make ordering visible

Use the recording wrappers instead of the corresponding raw `cute.arch` calls when you want
the diagnostic to see them:

```python
from transformer_nuggets.cute.warpdrive import (
    bulk_commit_group,
    bulk_wait_group,
    fence_async_shared,
    named_barrier,
)

fence_async_shared()  # cute.arch.fence_view_async_shared()
bulk_commit_group()  # cute.arch.cp_async_bulk_commit_group()
bulk_wait_group(0, read=True)  # cute.arch.cp_async_bulk_wait_group(...)
named_barrier(barrier_id, number_of_threads=128)  # cute.arch.barrier(...)
```

These are separate examples, not a general synchronization recipe. Named barrier IDs 1 and 2
are reserved by warpdrive; prefer a declared [`NamedBarrier`](channels.md#named-barriers),
whose `sync()` records the same event, or use an available ID and the right participant count. Each wrapper
emits the underlying instruction plus a host-side trace event, not an extra device operation.
Access annotations themselves emit no device code. The analysis is host-only; switching
`alias_check` on or off does not change the generated device code.

Managed channel signals and their declared fences are recorded automatically. Raw ordering
instructions remain legal, but cannot supply an edge in the diagnostic's model. Similarly,
`signal=False` makes the managed close visible without describing the manual signal. A pair
that only such a signal would order, or that a handle opened under `unchecked("unclosed")`
and never closed might order, is reported `unverified`.

### Read the result

`describe()` reports each declared alias as one of:

- `potential hazard`: conflicting annotated accesses with a missing ordering edge. The report
  names the accesses, source lines and rule involved.
- `unverified`: insufficient modeled information, such as an unannotated access, raw ordering
  or a cluster. Phase 1 only models single-CTA kernels.
- `no hazard found`: none found among the annotated conflicting pairs, not a proof of safety.

Warn mode emits a warning for findings and leaves the report in `describe()`. Some unverified
cases only appear in the report. Inspect it even when compilation is quiet.

The model checks whole-region/stage footprints, same-channel handoffs and cross-channel paths
in a bounded trace: loops twice, and two work items for persistent schedules. It doesn't infer
all memory accesses or prove cross-role convergence. `h.data`, `tmem.ptr`, `tmem.cols` and
`tmem.base` are unannotated acquisitions, not descriptions of every later use.

A deeper ring orders epoch e before e + n, not the next epoch. Its stage handoff only orders
storage indexed by that ring's own stage, with static, disjoint stage spans. Ring-stage SMEM
shared by different channels is unverified. Consumers of one fan-out epoch, and producers of
one fan-in epoch, are not ordered with respect to each other. Round-robin channels add no
ordering edges to the model, so accesses ordered only through them can be reported as hazards.

The model assumes balanced per-item counts and matching epochs across roles. When traced
counts differ, an item's commits link only to the consumer's last wait for that item.
It assumes loops run at least twice, TMEM loads finish before the next signal, and tcgen05
thread-sync fences surround recorded barriers and channel signals. SMEM accesses in one role
are treated as one thread set; there is no cross-warp SMEM check. Annotations are trusted to
cover the role's uses of each region it annotates.

For an intentionally unmodeled access:

```python
with unchecked("alias", reason="access is ordered by the raw protocol below"):
    tmem.access("P", stage, write=True, by="threads")
```

Only annotations inside that block are excluded. It adds no fence and makes no safety claim.
The modeled rules and their PTX references live in [alias.py](../alias.py). For the broader
synchronization contract, keep [GUARANTEES.md](../GUARANTEES.md) close by.
