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
For positive extents this is a bijection; decode intermediates stay within
`max(index, item_count)`, so admitted indices do not overflow.

## Choose a schedule

| Schedule | How a unit gets work after its first item |
|---|---|
| `OneShot()` | Terminal `total`; dropped items are not reassigned locally |
| `StaticPersistent()` | Add the number of launched units to its index |
| `AtomicPersistent(stages=2)` | Claim an index from a global atomic counter |
| `ClusterLaunchControl(stages=2)` | Cancel an unlaunched unit and run its work |

With U launched work units, unit b starts at index b. Static persistence then visits
`b + U, b + 2*U, ...`; atomic claims are unique but have no fixed order between units.
CLC divides a canceled cluster's CTA x coordinate by the cluster size to recover the work
index. It waits for every cancellation response and issues no query after failure.

For a correctly launched and drained stream, each kept index belongs to exactly one work
unit. All its participating roles observe the same ordered sequence, in the CTAs where
those roles run: all CTAs for `all`, pair leaders for `leader`, and rank 0 for `rank0`.
This does not mean every CTA sees every work item.

One-shot and static schedules run locally in each role. Atomic and cluster launch control
add a scheduler warp. The scheduler warp claims and decodes an item, then sends it to the
other roles through a ring. In a cluster, rank 0 sends it to every CTA.

`stages` is the work ring's depth, not the number of A/B stages. A free work-ring slot is
acquired before claiming more work. Roles release the slot when they receive the item, not
when they finish computing it. Every ring consumer also receives the terminal marker, which
is not work. The scheduler uses the lowest free warp unless you set
`WorkStream(scheduler_warp=...)`.

Use `while work.valid:` and one `work.advance()` directly at the end of that loop's body.
Read the item before advancing, and return the cursor. Every managed role must drain the
stream with a persistent schedule. The checker requires the exact `work.valid` condition and
rejects item reads after advance, including lookahead reads of the next item in the same
iteration. It does not prove advance is lexically last, detect every cursor rebind or stale
state restore, or prove useful and matching work happened before it.

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
cluster of `cs` CTAs uses at most `max_ctas // cs` clusters, with a minimum of one. Both
schedules also cap the number of units by the item count, again with a minimum of one.
`launch_args()` returns the CTAs of every cluster that fits at once: the SM count for one CTA
or a pair, fewer for 4-CTA clusters (36 clusters on a 152-SM GB300). This is a capacity
estimate, not a co-residency guarantee. Keep both launch arguments in the entry point's
signature even for schedules that don't read them, so changing the schedule
leaves the ABI alone.

`AtomicPersistent` uses two zeroed Int32 words, `[claims, finished]`. After the first item,
a claim returns `launched_units + atomicAdd(claims, 1)`. The last scheduler to report completion
fences and zeros both words for the next launch, including ordered CUDA-graph replays.
`launch_args()` caches workspace by device and current stream: launches on one stream share it because they
run in order, and launches made with it on different streams get different workspaces, so
they can run concurrently. Never pass the same workspace to concurrent launches yourself.
They would claim from one counter, taking some of each other's items, and the first launch
to finish could zero it while the other is still claiming.
Schedules that read no workspace (`OneShot`, `StaticPersistent`, `ClusterLaunchControl`) without
`record` get one placeholder per device instead, never read or written by the kernel, so a
CUDA graph captured on a new stream records no fill.

### The grid and empty work

`OneShot()` with plain `Raster()` launches `(m_tiles * cluster_size, n_tiles, l)` and normally
reads the coordinates from `blockIdx`. If y or z would exceed 65535, it uses a 1-D grid and
decodes the index. A hand-built `(total * cluster_size, 1, 1)` grid is also supported for this
one-shot path. Other built-in combinations use a 1-D grid.

An extent <= 0 makes a Raster space empty. Its bound space becomes `(1, 1, 0)`, each built-in
schedule launches one work unit, and every role starts with `work.valid == False`. There is
no division by zero and the atomic workspace remains zeroed. No guarded mapping is decoded;
any-index decoders may evaluate and discard an item. Normalization happens on the host in
the JIT entry point. Direct `Raster.decode` calls must use the bound space too: raw extents
such as `(0, 1, 1)` are not safe divisors.

Counts are computed once on the host side of the JIT entry point with `mapping.count64()`.
The kernel receives the narrowed Int32 `BoundWork.total`. The launch check must admit that
count before any kernel runs:

- One-shot and cluster launch control require `total * cluster_size <= 2**31 - 1`.
- Static and atomic persistence require `total + launched_units - 1 <= 2**31 - 1`, including
  the last unit's terminal index.

A failing check makes `bound.grid()` zero-wide; CUDA rejects the launch rather than running
a wrapped space. Use `launch_config(bound)` or `bound.grid()` to get this check. A hand-built
grid is your responsibility. Raster's runtime extents are converted to Int32 before binding;
passing a wider value is not a way to request a larger index space. The failed launch leaves
no sticky CUDA error, but invalidates an active CUDA-graph capture. Custom mapping arithmetic,
`record` log offsets and other launch resources must independently fit Int32 and CUDA limits.

## Custom mappings

Subclass `WorkMap`. The complete contract is in [scheduling.py](../scheduling.py).

`words` is the number of Int32 words in an item. `bind(space)` runs once to normalize the
runtime input in the entry point; it can return tuples, tensors or another DSL value.
`count64(bound_space)` returns a nonnegative host-visible count, exact through `2**31` or saturating at that value,
never wrapping. Both run before the kernel; neither can read a device-built length.

On the device, `decode(index, bound_space, state)` returns `(keep, words, state)`. A stateless
mapping passes `state` (`None`) through; see [Decode state](#decode-state). Set `skips = True` if
`keep` can be false. Otherwise `keep` must be the constant Python `True`, which is checked
while tracing. Dropping an index permanently removes it from every role's sequence. `unpack(words)`
returns the public `work.item`; its default is a tuple.

Normally, decode only runs for `0 <= index < total`. A table-backed mapping can therefore
read row `index` without handling terminal indices. `decodes_any_index = True` opts out of
that guard: use it only for a pure decoder defined for every nonnegative Int32 index,
including empty spaces and indices past the end. It may also run before readiness in the
tile-grid prologue. Raster and the external adapters opt in; a decoder that reads an item
table must not. A subclass that overrides `decode` must opt in again. A skipping mapping's
`keep` is still used only for valid indices. A skipping mapping is decoded once per visited
index at each decoding site.

Mapping results must agree across roles and CTAs. Tables must be ready before decoding and
stay unchanged while the stream uses them. `get_key()` names every compile-time choice,
not runtime extents, using lowercase letters, digits and underscores. Every kept index must
represent a distinct work item, and every needed item must be reachable among `[0, count)`;
the layer does not check either property.

### Decode state

Some decodes are cheap only incrementally. The grouped GEMM numbers its groups' m-tiles back to
back, so finding a tile's group takes a prefix sum of the groups' tile counts over device
offsets. A unit's indices increase under static and atomic persistence (and in practice under
cluster launch control), so the prefix reached for one item is where the walk for the next one
starts. `start(space)` returns that
carried state, and `decode` receives and returns it:

```python
class Window(NamedTuple):
    group: Int32
    tile_lo: Int32  # group's m-tiles [tile_lo, tile_hi), rows [row_lo, row_hi)
    tile_hi: Int32
    row_lo: Int32
    row_hi: Int32


class GroupedTiles(WorkMap):
    def start(self, space):
        return Window(Int32(-1), Int32(0), Int32(0), Int32(0), Int32(0))  # before group 0

    def decode(self, index, space, state):
        offsets, total_m, n_tiles = space
        mt = index // n_tiles
        w = _seek(state, mt, offsets, total_m, self.tile_m)  # no load while mt stays in w
        row0 = w.row_lo + (mt - w.tile_lo) * self.tile_m
        rows = cutlass.min(w.row_hi - row0, self.tile_m)
        return row0 < w.row_hi, (w.group, row0, rows, index % n_tiles), w
```

Each decoding site owns one state: every role of a local schedule, the scheduler warp of a
broadcast one, and `single_item`. The site calls `start` once, after the readiness hook, even
when the stream is empty, and then passes each decode the state the previous one returned,
dropped items included. Ring receivers hold no state. The one exception is a
`decodes_any_index` mapping without hooks: every role decodes its first item from a fresh
`start` state.

The state is a cache, never an input to the answer. From any state reachable from `start`,
`decode` must return the `keep`, and for a kept index the words, that `start`'s own state would.
Indices can arrive in any order: strided by the unit count, claimed by other units in between,
or backwards under a custom schedule. Make the common increasing case fast and the rest correct;
the grouped GEMM walks forward from its window and restarts from group 0 when an index goes
backwards.

`start` must be safe for every bound space, because it also runs for empty streams. It may read
what readiness made visible, but item-table reads belong in the guarded `decode`. Build the
state from DSL values (`Int32(0)`, not `0`) in tuples or a `NamedTuple`; a plain dataclass's
fields are not carried. `decode` must return the same structure and types it received. These
rules are checked while tracing. The state lives in registers of every decoding warp, so keep it
to a few words and name it in `describe()`: the grouped GEMM's 5-word window is
`GroupedTiles(tile_m=128, state=Window: 5 x Int32)`.

Prefer a broadcast schedule when the decode stays expensive even with a state. The scheduler
warp then decodes once per item, ahead of the roles, instead of every role decoding between
items. Choose a local schedule when the carried state makes the decode cheap; it needs no
scheduler warp or workspace. The grouped GEMM's static and atomic schedules perform the same.

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
`index < length[0]`; its state stays `None`. The schedule still visits the unused capacity, so don't oversize it
needlessly. The row read is within capacity even for a dropped item; its words are ignored.

`PdlWait()` issues `griddepcontrol.wait` once before decoding, in each local role or in the
broadcast scheduler. It supplies the wait for a programmatic dependent launch, not the
producer or the rest of the launch setup. See the PDL tests in
[test_cute_warpdrive_scheduler_interface.py](../../../../test/test_cute_warpdrive_scheduler_interface.py)
for the producer/consumer launch. A custom hook subclasses `Readiness` and supplies `wait()`
and `get_key()`. Its wait must terminate in every warp that calls it.

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
the same sequence in every role. Both are pure functions, and these properties are caller
obligations, not trace-time proofs. The schedule must terminate and assign each kept index to
exactly one work unit if you want once-only coverage. Each local stream user calls
`first_index` at initialization, then `next_index` on each advance; skip loops also call
`next_index` for every dropped valid index. A nonpersistent schedule must return a terminal
index from `next_index`.

A broadcast schedule sets `broadcast = True` and a positive `stages`, then defines
`claim_next(fetch, problem, index)`. The previous index is supplied, so the claim sequence can
carry state without another buffer. `fetch_channels()` declares any extra scheduler channels;
`close_claims(fetch, problem)` runs once after its last claim. Earlier items may still be
computing: this closes claim state, not work. Do not use it for teardown, fixups or reductions.

Only the rank-0 scheduler warp calls `claim_next`, after acquiring a free work-ring slot and
while the previous claim is valid. The layer publishes `[index, *words]` to all stream users
and owns the ring's counts, payload, fences, tails and drain; extra `fetch_channels` use the
same channel derivation. The scheduler decodes and broadcasts the first item too, so roles
never decode, unless the mapping sets `decodes_any_index` and the stream neither skips nor
waits for readiness. Then every role decodes `first_index` locally from launch-time data and
only later items come from the ring.

The host methods are `launch_units(total, max_units)` and `index_limit(total, units, cs)`.
The former receives the cap in work units when `uses_max_ctas`, otherwise `None`.
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
