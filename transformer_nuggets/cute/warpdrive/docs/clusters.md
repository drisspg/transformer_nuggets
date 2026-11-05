# Clusters and CTA pairs

[Back to warpdrive](../README.md)

```python
self.cluster = CtaPair()  # cluster (2, 1, 1), tcgen05 CtaGroup.TWO


@role(warps=5, regs=40, cta="leader")
@cute.jit
def mma(self, params, *, ab: Consumer, acc: Producer, tmem: Tmem):
    ...
    return ab, acc
```

`CtaPair()` configures two CTAs working on one item. It doesn't convert your single-CTA
atoms or tensor layouts. Double the MMA tile's M dimension, build the atoms with
`self.cluster.cta_group` and `self.cluster.layout_vmnk()`, and partition with
`self.cluster.rank()`.

Call `rank()` from converged warp code, not a lane-0 branch. Call `begin_roles()` and
`run_roles()` from converged CTA/cluster code. Launch with the declared block and cluster
shape through `launch_config()`.

The [persistent GEMM](../examples/persistent_gemm.py) and [FA4](../examples/fa4_fwd.py) have
single- and two-CTA paths with the corresponding atoms and layouts. The persistent GEMM also
runs on [larger clusters](#larger-clusters).

## Which channels cross CTAs?

If all participants have the same placement and nothing is multicast, the channel is
CTA-local: all/all means an independent channel in each CTA; leader/leader means one in each
pair leader (rank 0 of a `CtaPair`). Mixed placements or [multicast](#multicast) copies make it
cross-CTA (pair-scoped in a `CtaPair`).

For the GEMM's A/B channel, the load role runs in both CTAs and the MMA role only on the
leader. Each CTA issues its TMA copies; the leader arms twice the declared per-CTA byte count.
Use group-TWO copies for this handoff. Ordinary leader-only TMA producing to both CTAs would
need multicast and is rejected.

For the accumulator channel, the leader's UMMA completion signals both CTAs. Their epilogue
threads release back to the leader. Counts and masks follow the declared placement; you don't
have to double the thread count manually. All UMMA signalers in a `CtaPair` must be
leader-only, even on CTA-local channels.

`print(op.describe())` shows the per-CTA counts, targets, masks and transaction bytes derived
by the [placement rules](#placement-rules).

## Placement rules

Let `C(role)` be the ranks where a role runs: every rank for `all`, the even ranks for
`leader` with MMA pairs (every rank without pairs), and `{0}` for `rank0`.
`reach(p)` is the SMEM written by producer rank p: p itself plus the ranks on
`Cluster.line(p, mode)` for each declared multicast mode. This is a union of lines, not a
Cartesian product when both modes are declared.

When every participant has the same `C` and there is no multicast, the channel is CTA-local.
It uses ordinary per-slot counts, unscaled transaction bytes and self-targets, with no tail.
Masks are 1 except that a UMMA commit to its own CTA uses `1 << rank`.

Otherwise, define a synchronization unit for each rank r:

- r alone when every participant has the same `C` (only multicast crosses CTAs);
- r's MMA pair when each participant runs somewhere in every pair;
- the whole cluster otherwise, when placement restricts a participant to rank 0.

The full targets of producer rank p are the consumer CTAs in p's unit. The empty targets of
consumer rank c are the producer CTAs p for which `reach(p)` intersects c's unit. In a
`CtaPair`, mixed placement therefore targets the union of consumer or producer CTAs.
For each target and slot, `Threads` counts `32 * role.num_warps` for each signaling
(instance, CTA); TMA/UMMA count one arrival per signaling CTA. Nontarget ranks have placeholder
count 1 and no promised waiter. Masks contain one bit per target and are a single integer
when all signaling CTAs agree, otherwise a per-rank mapping.

UMMA signalers in a cluster with pairs must run only on pair leaders and use group TWO.
The UMMA side must include the leader of every pair the other side uses: each CTA's operands
and accumulator belong to its own leader's MMAs. Without pairs, a UMMA producer must include
every consumer CTA; a UMMA consumer may release other CTAs' storage.

Ordinary mixed-placement TMA has one supported shape: producers in every CTA feed exactly
the pair leaders. Only leaders arm full barriers, each for `2 * Tma.tx_bytes`; each CTA's
copies deliver its declared per-CTA bytes with group TWO, multicast or not. These leader-only
arming targets replace the general full targets above. All other mixed-placement TMA shapes
are rejected. Same-placement TMA, multicast or not, arms each participating CTA's own barrier
with unscaled bytes. All/all requires group ONE; with pairs, leader/leader accepts either
group in the channel's group check (descriptor legality remains yours).

Multicast cannot have a `rank0` participant. Its byte count is still what lands in each CTA's
SMEM per stage; a copy delivers its box to every CTA in its mask. The declaration derives the
release targets, not the copy itself. The internal scheduler is separate: rank 0 explicitly
fills each target's payload and uses unscaled payload bytes, not ordinary multicast TMA.

## Work and exit

A work index belongs to the cluster, not to each CTA independently. Local schedules compute
the same sequence in every CTA. `AtomicPersistent` and `ClusterLaunchControl` claim on rank 0
(a pair's leader) and publish each decoded item to every CTA. CLC cancels whole clusters.

In a cluster, TMEM allocation happens in `run_roles()` after the cluster synchronization (a
pair's two CTAs allocate together). At least one TMEM-using role must run in every CTA; the
first such role allocates. All declared
TMEM users join teardown before free, including raw roles. Pair allocation/free uses the
two-CTA handshake with each CTA's own peer, `rank XOR 1`.

Producers automatically tail when a channel's UMMA consumer in another CTA can still read
their storage, including a multicast partner outside its pair. Return their final endpoint
state so the tail uses the right phases: Ring tails drain from that returned state; Slots
tails wait only on that end's acquired slots with their final phase bits. A retired acquire
cannot be tailed and is rejected on these channels. All threads reach the exit
cluster barrier after dispatch and TMEM teardown. This does not wait for arbitrary async work
that your role started.

## Memory scope

Without a payload, `remote_scope` defaults to `"cta"`. That is the normal setting for
TMA/MMA/TMEM handoffs ordered by their own mechanisms. If ordinary SMEM or global writes must
be read by the peer, set `remote_scope="cluster"` and address the peer's storage correctly.
warpdrive cannot infer that requirement from the channel declaration.

Layer-owned `payload=` buffers are CTA-local and disallowed on user pair-scoped channels.
The internal work ring is a separate case: its scheduler explicitly fills every CTA's payload.

`SmemPlan` buffers can bind to pair-scoped channels. They still allocate separately in each
CTA at matching offsets, and `buffer=` neither changes `remote_scope` nor adds a fence.
The phase-1 alias diagnostic reports cluster aliases as `unverified`; it does not analyze
cross-CTA ordering.

## Larger clusters

```python
self.cluster = Cluster(2, 2, pair=True)  # two MMA pairs side by side along N
self.ab = Ring(stages, Tma(stage_bytes, multicast="n"), Umma, buffer={"A": a, "B": b})
```

`Cluster(m, n, pair=...)` is `m x n` CTAs sharing a work item, up to 8 CTAs. With `pair=True`,
consecutive CTAs along M form MMA pairs, and `cta="leader"` runs in each pair's leader (the
even ranks). Without pairs every CTA is a leader. `cta="rank0"` runs once per cluster.
`CtaPair()` is `Cluster(2, 1, pair=True)` and `SingleCta()` is `Cluster(1, 1)`.

Rank `r` has coordinates `(v, mi, ni)` with the pair fastest: `self.cluster.coords(rank)`.
It matches `layout_vmnk()`, the `(V, M, N, K)` cluster layout CUTLASS uses. The launch
cluster is `(m * n, 1, 1)` CTAs, so a work unit is still a run of consecutive `blockIdx.x`.

### Multicast

Pairs side by side along N need the same rows of A. Instead of each pair loading them, each
CTA loads part of its A tile and multicasts it to the CTA with its coordinates in every other
pair along N. Pairs along M share B the same way. In the 2x2 cluster only A is multicast:
B's tile is already split across the pair (each CTA loads half of its rows and the 2-CTA MMA
reads both halves), and there is only one pair along M to share it with.

```python
share_a = self.cluster.multicast("n")  # coord, layout, mask, ctas
tAsA, tAgA = cpasync.tma_partition(tma_a, share_a.coord, share_a.layout, sA, gA)
...
cute.copy(tma_a, tAgA[None, k], s.view("A"), tma_bar_ptr=s.barrier, mcast_mask=share_a.mask)
```

Build the atom with `CopyBulkTensorTileG2SMulticastOp(self.cluster.cta_group)` and pass
`layout_vmnk().shape` to `make_tiled_tma_atom_A`/`_B`. Declare the modes on the producer:
`Tma(bytes, multicast="n")`. `bytes` is still what lands in each CTA's SMEM per stage, so a
multicast channel declares the same bytes as an unshared one.

The declaration changes the empty side. A stage in CTA r is read by its own pair's MMA and, via
its multicast A part, by the other pair's MMA. Each leader's commit releases all four CTAs, and
each CTA's empty barrier waits for both leaders. Releasing only the own pair is the classic
2x2 bug: the other pair's next multicast overwrites a stage that is still being read.

Get the multicast partition and mask from `Cluster.multicast` after `begin_roles()`. The TMA
check verifies the mask's mode and the copy bytes. A multicast copy into a channel without
`multicast=` is rejected; a mask from elsewhere leaves the channel `unverified`. The check
does not prove the TMA descriptor legal: a 3-way multicast partition compiles and then faults
at runtime, so the persistent GEMM rejects clusters with a non-power-of-two count of MMA tiles
along either mode.

### Work items and the scheduler

The [persistent GEMM](../examples/persistent_gemm.py) runs one MMA tile per pair. A work item is
the cluster tile (256 x 448 for the 2x2 cluster at `tile_n=224`), and each pair adds its
position: `m * pairs_m + mi`, `n * pairs_n + ni`. A ragged last cluster tile has pairs entirely
past the edge; TMA loads zeros there and clips the stores.

The scheduler warp runs on rank 0 and publishes each item to all four CTAs. Its ring's empty
barrier on rank 0 counts every stream user of every CTA. `launch_args()` caps a persistent grid
at the clusters that fit at once (`HardwareInfo().get_max_active_clusters(4)` is 36 on a 152-SM
GB300, not 38). This capacity estimate is not a guarantee that all work units are co-resident.

### What the 2x2 GEMM derives

```python
op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=Cluster(2, 2, pair=True), tile_n=224)
compile_persistent_gemm(op)
print(op.describe())
```

```
PersistentGemm: 8 warps, 256 threads
  cluster  cluster 2 x 2 (M x N), tcgen05 CtaGroup.TWO, leaders = even ranks
  params   PersistentGemmParams

Roles
  warps  role            regs  cta     raw
  0-3    epilogue         160  all
  4      load              40  all
  5      mma               40  leader
  6      work.scheduler    40  rank0
  7      (padding)         40

Channels
  ab         Ring(6)  load -> mma
             signal    Tma(30720 B) -> Umma
             full      rank 0 1 x6, rank 2 1 x6, tx 61440 B (leader-armed)
             empty     rank 0 2 x6, rank 1 2 x6, rank 2 2 x6, rank 3 2 x6
             scope     cross-CTA, multicast n, release_mask 0b1111, tail
             storage   buffer A, B
             tma       verified CtaGroup.TWO, 30720 B/stage per CTA @ persistent_gemm.py:194,195
  acc        Ring(2)  mma -> epilogue
             signal    Umma -> Threads
             full      rank 0 1 x2, rank 1 1 x2, rank 2 1 x2, rank 3 1 x2
             empty     rank 0 256 x2, rank 2 256 x2
             scope     cross-CTA, commit_mask rank 0 0b0011, rank 2 0b1100, remote_scope cta
  work_ring  Ring(2)  work.scheduler -> epilogue, load, mma, work.scheduler
             signal    Tma(16 B) -> Threads
             full      rank 0 1 x2, rank 1 1 x2, rank 2 1 x2, rank 3 1 x2, tx 16 B (leader-armed)
             empty     rank 0 736 x2
             scope     cross-CTA, remote_scope cluster
             storage   payload Int32(4,)
  ...
```

Reading it: each leader arms 60 KiB for its pair. Every CTA's `ab` stage waits for 2 releases,
one from each leader, and each leader's release mask names all four CTAs. The accumulator stays
inside a pair: rank 0 commits to ranks 0-1, rank 2 to ranks 2-3. The work ring expects
4 x (128 + 32) threads from the epilogue and load warps, 2 x 32 from the leaders' MMA warps and
32 from the scheduler.

### Limits

- Multicast channels cannot have a `rank0` participant. Mixed-placement TMA still needs
  producers in every CTA feeding the pair leaders. Same-placement multicast (all/all or
  leader/leader) arms each CTA's own barrier and releases every CTA its copies reach.
- A leader-only TMA producer can't fill other CTAs' stages for consumers there (that would
  need the leader to arm their barriers).
- Clusters have at most 8 CTAs (the portable limit).
- The alias diagnostic doesn't model clusters.
