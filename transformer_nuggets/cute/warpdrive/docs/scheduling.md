# Scheduling

[Back to warpdrive](../README.md)

```python
self.work = WorkStream(mapping=Raster(swizzle=8), schedule=ClusterLaunchControl(stages=2))


@role(warps=4, regs=40)
@cute.jit
def load(self, params: PersistentGemmParams, *, ab: Producer, work: WorkCursor):
    while work.valid:
        m, n, _ = work.item
        ...  # load this item's operands
        work.advance()
    return ab, work
```

A work unit is one CTA, or one cluster with `CtaPair()` or `Cluster(...)`. `schedule=` chooses its linear
indices; `mapping=` turns an index into the item returned by `work.item`. `ready=` is an
optional wait before decoding starts. There can be one `WorkStream` per kernel.

`WorkStream()` defaults to `Raster()` and `OneShot()`. `Raster` maps `(m_tiles, n_tiles, l)`
to `(m, n, l)` coordinates. `along="m"` selects the raster axis, `swizzle=8` groups the other
axis into eight-tile bands, and `reverse="m"` visits m in reverse order. l is the slowest axis.

## Choose a schedule

| Schedule | How a unit gets work after its first item |
|---|---|
| `OneShot()` | No more work |
| `StaticPersistent()` | Add the number of launched units to its index |
| `AtomicPersistent(stages=2)` | Claim an index from a global atomic counter |
| `ClusterLaunchControl(stages=2)` | Cancel an unlaunched unit and run its work |

One-shot and static schedules run locally in each role. Atomic and cluster launch control
add a scheduler warp. The scheduler warp claims and decodes an item, then sends it to the
other roles through a ring. In a cluster, rank 0 sends it to every CTA.

`stages` is the work ring's depth, not the number of A/B stages. A free work-ring slot is
acquired before claiming more work. Roles release the slot when they receive the item, not
when they finish computing it. The scheduler uses the lowest free warp unless you set
`WorkStream(scheduler_warp=...)`.

Use `while work.valid:` and one `work.advance()` directly in that loop's body. Read the item
before advancing, and return the cursor. Every managed role must drain the stream with a
persistent schedule. These checks don't prove that your roles do matching work inside the loop.

## Bind and launch

From [persistent_gemm.py](../examples/persistent_gemm.py):

```python
# Python, outside the compiled entry point:
workspace, max_ctas = op.work.launch_args(device)

# In __call__ (@cute.jit):
bound = self.work.bind((m_tiles, n_tiles, 1), workspace, max_ctas)
self.kernel(..., bound).launch(**self.launch_config(bound), stream=stream)

# In the kernel:
smem = self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout, "C": c_layout})
```

Pass the layouts for this kernel's `SmemPlan`; omit `layouts` if you allocate storage yourself.
The keyword in `begin_roles(work=bound)` matches the `self.work` attribute. `bind()` returns
`BoundWork`, the runtime problem passed to the kernel. Shapes, counts and the persistent grid
cap are runtime values, not per-shape compile keys.

`StaticPersistent` and `AtomicPersistent` need `max_ctas`. It caps CTAs, not clusters; a
cluster of `cs` CTAs uses at most `max_ctas // cs` clusters, with a minimum of one.
`launch_args()` returns the CTAs of every cluster that fits at once: the SM count for one CTA
or a pair, fewer for 4-CTA clusters (36 clusters on a 152-SM GB300). Keep both launch arguments in the
entry point's signature even for schedules that don't read them, so changing the schedule
leaves the ABI alone.

`AtomicPersistent` uses two zeroed Int32 words, `[claims, finished]`. After the first item,
a claim returns `launched_units + atomicAdd(claims, 1)`. The last scheduler to report completion
zeros both words for the next launch, including ordered CUDA-graph replays. `launch_args()`
caches workspace by device and current stream. Never pass the same workspace to concurrent
launches.

### The grid and empty work

`OneShot()` with plain `Raster()` launches `(m_tiles * cluster_size, n_tiles, l)` and normally
reads the coordinates from `blockIdx`. If y or z would exceed 65535, it uses a 1-D grid and
decodes the index. A hand-built `(total * cluster_size, 1, 1)` grid is also supported for this
one-shot path. Other built-in combinations use a 1-D grid.

An extent <= 0 makes a Raster space empty. Its bound space becomes `(1, 1, 0)`, each built-in
schedule launches one work unit, and every role starts with `work.valid == False`. There is
no division by zero and the atomic workspace remains zeroed.

Counts are computed once on the host side of the JIT entry point with `mapping.count64()`.
The kernel receives the narrowed Int32 `BoundWork.total`. The launch check must admit that
count before any kernel runs:

- One-shot and cluster launch control require `total * cluster_size <= 2**31 - 1`.
- Static and atomic persistence require `total + launched_units - 1 <= 2**31 - 1`, including
  the last unit's terminal index.

A failing check makes `bound.grid()` zero-wide; CUDA rejects the launch rather than running
a wrapped space. Use `launch_config(bound)` or `bound.grid()` to get this check. A hand-built
grid is your responsibility. Raster's runtime extents are converted to Int32 before binding;
passing a wider value is not a way to request a larger index space.

## Custom mappings

Subclass `WorkMap`. The complete contract is in [scheduling.py](../scheduling.py).

`words` is the number of Int32 words in an item. `bind(space)` normalizes the runtime input
in the entry point; it can return tuples, tensors or another DSL value. `count64(bound_space)`
returns a nonnegative host-visible count, exact through `2**31` or saturating at that value,
never wrapping. Both run before the kernel; neither can read a device-built length.

On the device, `decode(index, bound_space)` returns `(keep, words)`. Set `skips = True` if
`keep` can be false. Otherwise `keep` must be the constant Python `True`, which is checked
while tracing. Dropping an index removes it from every role's sequence. `unpack(words)`
returns the public `work.item`; its default is a tuple.

Normally, decode only runs for `0 <= index < total`. A table-backed mapping can therefore
read row `index` without handling terminal indices. `decodes_any_index = True` opts out of
that guard: use it only for a pure decoder defined for every nonnegative Int32 index,
including empty spaces and indices past the end. It may also run before readiness in the
tile-grid prologue. Raster and the external adapters opt in; a decoder that reads an item
table must not. A subclass that overrides `decode` must opt in again. A skipping mapping's
`keep` is still used only for valid indices.

Mapping results must agree across roles and CTAs. Tables must be ready before decoding and
stay unchanged while the stream uses them. `get_key()` names every compile-time choice,
not runtime extents, using lowercase letters, digits and underscores.

### A device-built list

[PackedWork](../examples/packed_work.py) is a small table-backed mapping:

```python
from transformer_nuggets.cute.warpdrive import AtomicPersistent, PdlWait, WorkStream
from transformer_nuggets.cute.warpdrive.examples.packed_work import PackedWork

self.work = WorkStream(mapping=PackedWork(words=3), schedule=AtomicPersistent(), ready=PdlWait())
# In the entry point: table is int32 (capacity, 3), length is an int32 device tensor.
bound = self.work.bind((table, length), workspace, max_ctas)
```

`count64()` returns the table capacity. `decode()` reads the row and keeps it when
`index < length[0]`. The schedule still visits the unused capacity, so don't oversize it
needlessly. The row read is within capacity even for a dropped item; its words are ignored.

`PdlWait()` issues `griddepcontrol.wait` once before decoding, in each local role or in the
broadcast scheduler. It supplies the wait for a programmatic dependent launch, not the
producer or the rest of the launch setup. See the PDL tests in
[test_cute_warpdrive_scheduler_interface.py](../../../../test/test_cute_warpdrive_scheduler_interface.py)
for the producer/consumer launch. A custom hook subclasses `Readiness` and supplies `wait()`
and `get_key()`.

Split work uses distinct indices too: a split-K mapping returns `(tile, split)` and the split's
K range, dropping empty ranges. Reduce the partials in a later kernel, not in the schedule.
The decode example's `SplitKV` mapping returns `(split, kv_head, batch, splits)` and, with
per-sequence split counts from a preceding kernel, drops splits past a sequence's count after
`PdlWait()`. The FA4 adapter below doesn't provide split-KV.

## Custom index schedules

A local `IndexSchedule` implements `next_index` and can replace `first_index`:

```python
from dataclasses import dataclass
from transformer_nuggets.cute.warpdrive import IndexSchedule, Raster, WorkStream


@cute.jit
def countdown(index, problem):
    return cutlass.select_(index >= 0, index, problem.total)


@dataclass(frozen=True)
class ReverseStatic(IndexSchedule):
    def first_index(self, problem):
        return countdown(problem.total - 1 - problem.first_unit(), problem)

    def next_index(self, index, problem):
        return countdown(index - problem.units(), problem)


self.work = WorkStream(mapping=Raster(swizzle=8), schedule=ReverseStatic())
```

Every result must be a valid index or a terminal index `>= total`, never negative.
`first_index` depends only on launch values and the bound problem; `next_index` must compute
the same sequence in every role. The schedule must terminate and assign each kept index to
exactly one work unit if you want once-only coverage.

A broadcast schedule sets `broadcast = True` and a positive `stages`, then defines
`claim_next(fetch, problem, index)`. The previous index is supplied, so the claim sequence can
carry state without another buffer. `fetch_channels()` declares any extra scheduler channels;
`close_claims(fetch, problem)` runs once after its last claim. Earlier items may still be
computing: this closes claim state, not work. Do not use it for teardown, fixups or reductions.

The host methods are `launch_units(total, max_units)` and `index_limit(total, units, cs)`.
The latter bounds every computed index and grid x extent for the Int32 launch check. Its
default is `total + units - 1`, the terminal-index bound for strided schedules; override it
if that doesn't bound your sequence. Declare `uses_max_ctas` and `uses_workspace` truthfully
and include all static fields in `get_key()`.
See `IndexSchedule` in [scheduling.py](../scheduling.py) for the full method contract.

## CUTLASS and FA4 mappings

```python
from flash_attn.cute.tile_scheduler import StaticPersistentTileScheduler
from transformer_nuggets.cute.warpdrive import (
    ClusterLaunchControl,
    CutlassRaster,
    Fa4Mapping,
    StaticPersistent,
    WorkStream,
)

self.work = WorkStream(mapping=CutlassRaster(swizzle=8), schedule=StaticPersistent())
# Keep the mapping, change how indices are claimed:
self.work = WorkStream(mapping=CutlassRaster(swizzle=8), schedule=ClusterLaunchControl())
self.work = WorkStream(
    mapping=Fa4Mapping(StaticPersistentTileScheduler), schedule=ClusterLaunchControl()
)
```

These reuse an index-to-tile mapping, not the external scheduler's launch or pipeline.
`CutlassRaster` plus `StaticPersistent` reproduces CUTLASS's per-unit sequence with padded
swizzle tiles removed. To reproduce FA4's index sequence, pair
`Fa4Mapping(StaticPersistentTileScheduler)` with `StaticPersistent()`, or
`Fa4Mapping(SingleTileLPTScheduler)` with `OneShot()`. Either mapping also works with other
warpdrive schedules.

`CutlassRaster` is a native implementation of the BSD-3 CUTLASS CuTeDSL helper at `e13f5cd`,
not an import of the deprecated `cutlass.utils` scheduler. Its padded-space count includes
indices that decode with `keep=False`. The test oracle is
[cutlass_static_persistent_tile_scheduler.py](../../../../test/cutlass_static_persistent_tile_scheduler.py).

`Fa4Mapping` imports `flash_attn.cute` when constructed. The static-persistent mapping binds
`(num_block, num_head, num_batch)`. `SingleTileLPTScheduler` also needs `seqlen_k` as a fourth
entry to size L2 sections; `lpt`, `lpt_global`, `head_dim`, `head_dim_v` and `element_size`
are constructor fields. No split-KV is supported.

Only the two named FA4 schedulers are supported. `SingleTileScheduler` depends on a 3-D
`blockIdx` rather than one index and is rejected. External dynamic/CLC schedulers that own
SMEM and barriers cannot be dropped in as mappings. The adapter does not call their advance,
prefetch, tail or grid methods.

Both adapters normalize empty spaces and provide Int64 counts. In a cluster, their tiles
are work-unit tiles: one item per cluster, with FA4 params built using a `(1, 1)` cluster shape.

## One item in a raw role

```python
@role(warps=range(4), regs=160, raw=True)
@cute.jit
def epilogue(self, params, *, tmem: Tmem):
    valid, (m, n, _) = self.work.single_item()
    if valid:
        ...
```

Use this instead of reading `blockIdx` yourself. It handles the fallback grid, clusters, custom
mappings, readiness and dropped items like a managed cursor. It requires a non-persistent,
local schedule such as `OneShot()` and must run after `begin_roles()`. Persistent and broadcast
schedules reject it; their managed roles take a `WorkCursor` and loop.
