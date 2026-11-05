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
pair leader (rank 0 of a `CtaPair`). Mixed placements, [multicast](#multicast) copies or a
producer that [fills other CTAs](#one-cta-loads-for-others) make it cross-CTA (pair-scoped in
a `CtaPair`).

For the GEMM's A/B channel, the load role runs in both CTAs and the MMA role only on the
leader. Each CTA issues its TMA copies; the leader arms twice the declared per-CTA byte count.
Use group-TWO copies for this handoff. A leader-only TMA producer can also issue both CTAs'
copies when it declares `fills="pair"`; see [One CTA loads for others](#one-cta-loads-for-others).

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
Cartesian product when both modes are declared. A `Tma(fills=...)` producer in rank p issues
the copies of `filled(p)` (p's pair for `"pair"`, every rank for `"cluster"`) and writes
`W(p)`, the union of `reach(d)` over `d` in `filled(p)`; without fills `W(p) = reach(p)`.

When every participant has the same `C` and there is no multicast or fills, the channel is
CTA-local.
It uses ordinary per-slot counts, unscaled transaction bytes and self-targets, with no tail.
Masks are 1 except that a UMMA commit to its own CTA uses `1 << rank`.

Otherwise, define a synchronization unit for each rank r:

- r alone when every participant has the same `C` (only multicast or fills copies cross CTAs);
- r's MMA pair when each participant runs somewhere in every pair;
- the whole cluster otherwise, when placement restricts a participant to rank 0.

The full targets of producer rank p are the consumer CTAs in p's unit. The empty targets of
consumer rank c are the producer CTAs in c's unit. User TMA channels follow the TMA rule below
instead. In a
`CtaPair`, mixed placement therefore targets the union of consumer or producer CTAs.
For each target and slot, `Threads` counts `32 * role.num_warps` for each signaling
(instance, CTA); TMA/UMMA count one arrival per signaling CTA. Nontarget ranks have placeholder
count 1 and no promised waiter. Masks contain one bit per target and are a single integer
when all signaling CTAs agree, otherwise a per-rank mapping.

UMMA signalers in a cluster with pairs must run only on pair leaders and use group TWO.
The UMMA side must include the leader of every pair the other side uses: each CTA's operands
and accumulator belong to its own leader's MMAs. Without pairs, a UMMA producer must include
every consumer CTA; a UMMA consumer may release other CTAs' storage.

Cross-CTA TMA: let `W` be every CTA some `W(p)` contains and `F` the consumer CTAs. If
`W ⊆ F`, the bytes landing in CTA d complete on d's own barrier (group ONE). Otherwise, with
pairs and `F` within the pair leaders, they complete on d's pair leader (group TWO), and every
such leader must be in `F`. Every other shape is rejected, as is a consumer CTA no copy
completes on. Exactly one CTA arms each barrier, for every CTA whose bytes complete there
(`k * Tma.tx_bytes`, `k = 2` with group TWO, else 1): the barrier's own CTA, which must run the
producer, or with fills the CTA that fills it. So each full barrier expects one arrival.
Consumer CTA c releases every producer p whose `W(p)` meets the CTAs completing on c's
barrier or c's unit (the unit keeps the releases of placements accepted before fills).
Producers in every CTA feeding the pair leaders are the GEMM's case: each leader arms
`2 * Tma.tx_bytes`. Same-placement TMA arms each CTA's own barrier with unscaled bytes.
All/all requires group ONE; with pairs, leader/leader without fills accepts either group in
the channel's group check (descriptor legality remains yours).

`Tma.tx_bytes` is still what lands in each CTA's SMEM per stage; a copy delivers its box to
every CTA in its mask. The declaration derives the release targets, not the copy itself. A
`rank0` consumer may join a multicast channel as long as every written CTA keeps a waiting
consumer. The internal scheduler is separate: rank 0 explicitly fills each target's payload
and uses unscaled payload bytes, not ordinary multicast TMA.

These rules rest on PTX ISA 9.5. `cp.async.bulk.tensor` (9.7.10.28.5.3) with a
`.shared::cluster` destination may write any CTA of the cluster. A `.multicast::cluster`
`ctaMask` names the destination CTAs (nothing requires the issuing CTA among them), and the
data lands at the same offset in each. With `.cta_group::1` each destination's mbarrier at
`mbar`'s offset is signalled; with `.cta_group::2` the signal goes to the destination or its
peer by the parity of the CTA holding `mbar`, so a leader's barrier collects its pair's bytes
(CUTLASS `SM100_TMA_2SM_LOAD*` clears the peer bit of the barrier address for the same
effect). `mbarrier.arrive.expect_tx.shared::cluster` (9.7.15.16.16) arms a remote barrier. The
tx-count ranges over `±(2^20 - 1)` (9.7.15.16.3), so a copy may complete before its arm; a
phase completes when both the pending arrivals and the tx-count reach zero (9.7.15.16.8). Each
phase must be observed complete by a wait before an arrive of the next phase (9.7.15.16.5.1).
A filler arms CTA d's next phase only after its empty barrier saw d's consumers release, which
follows their full wait. A stage that fills in a CTA with no waiter would break that rule,
which is why every written CTA needs a consumer.

Teardown: a filled CTA's consumers wait on every stage filled for them, so every copy and
remote arm into that CTA has landed before they reach the exit cluster barrier. Thread
releases back to the filler precede the releasing threads' arrival at that barrier, as for
every cross-CTA thread release; UMMA releases into another CTA tail as described below.

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

`remote_scope` derives from what the channel's thread arrivals publish. Without a payload it
is `"cta"`: the normal setting for TMA/MMA/TMEM handoffs ordered by their own mechanisms, and
for releases that only hand a stage back. A payload that a thread producer writes and commits
derives `"cluster"`, since the commit itself must publish those words to a remote reader. A
payload filled by a `Tma` producer (completed by bytes on the reader's full barrier) derives
`"cta"` again: its consumers' releases publish nothing. If ordinary SMEM or global writes
outside a payload must be read by the peer, set `remote_scope="cluster"` and address the peer's
storage correctly; warpdrive cannot infer that requirement from the channel declaration. The
cluster scope is a GPU-scope membar per arriving thread
([measured](examples.md#tile-boundaries) on the 2-CTA persistent GEMM's work ring).

Layer-owned `payload=` buffers are CTA-local and disallowed on user pair-scoped channels.
The internal work ring is the one pair-scoped payload channel: its scheduler fills every CTA's
payload with `st.async`.

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
             scope     cross-CTA, remote_scope cta
             storage   payload Int32(4,)
  ...
```

Reading it: each leader arms 60 KiB for its pair. Every CTA's `ab` stage waits for 2 releases,
one from each leader, and each leader's release mask names all four CTAs. The accumulator stays
inside a pair: rank 0 commits to ranks 0-1, rank 2 to ranks 2-3. The work ring expects
4 x (128 + 32) threads from the epilogue and load warps, 2 x 32 from the leaders' MMA warps and
32 from the scheduler.

## One CTA loads for others

`Tma(fills=)` is experimental: no example uses it, its only users are tests, and its one
measured use was slower (below). It stays as is until a real kernel needs it.

```python
self.ld = Ring(2, Tma(stage_bytes, fills="pair"), Threads)  # load on cta="leader"


@role(warps=0, cta="leader")
@cute.jit
def load(self, ld: Producer, ctx):
    rank = self.cluster.rank()
    own, peer = self.cluster.fill(rank), self.cluster.fill(rank + 1)
    for i in cutlass.range(ctx.n, unroll=1):
        with ld.produce() as s:
            cute.copy(
                ctx.tma,
                ctx.g_own[None, i],
                ctx.s[None, s.index],
                tma_bar_ptr=s.barrier,
                mcast_mask=own.mask,
            )
            cute.copy(
                ctx.tma,
                ctx.g_peer[None, i],
                ctx.s[None, s.index],
                tma_bar_ptr=s.barrier,
                mcast_mask=peer.mask,
            )
    return ld
```

`Tma(bytes, fills="pair")` puts the producer on each pair leader, which issues the copies of
both CTAs in its pair. `fills="cluster"` puts it on `rank0`, which issues every CTA's copies.
`bytes` is still what lands in each CTA per stage. `Cluster.fill(rank)` gives the mask of a copy
that lands only in CTA `rank`, at the same SMEM offset as the issuing CTA's stage. Build the atom
with a multicast op, since even the copy into the issuing CTA needs a mask.
`Cluster.fill(rank, "n")` multicasts that copy along N as `rank`'s own `multicast("n")` copy
would; fills and multicast compose.

Where the bytes complete depends on the consumers. If every CTA a copy lands in has a
consumer, each CTA's bytes complete on its own barrier (use `CtaGroup.ONE`). If the consumers
are the pair leaders, as for a 2-CTA MMA, both CTAs' bytes complete on the leader's barrier
(use `CtaGroup.TWO`). The filling CTA arms each of those barriers, the remote ones with
`mbarrier.arrive.expect_tx` on the target CTA, so each full barrier expects one arrival. The
consumers where the copies land release the filling CTA.

```
  ld  Ring(2)  load -> use
      signal    Tma(4096 B) -> Threads
      full      rank 0 1 x2, rank 1 1 x2, rank 2 1 x2, rank 3 1 x2, tx 4096 B, rank 0 arms 0+1, rank 2 arms 2+3
      empty     rank 0 128 x2, rank 2 128 x2
      scope     cross-CTA, fills pair, multicast n, remote_scope cta
```

This is a 2x2 cluster of pairs whose leaders fill their pair and multicast along N. Rank 1's
stage holds half a tile from each leader, so it releases both leaders (each leader's empty
barrier waits for 4 x 32 consumer threads), but only rank 0 arms it. The TMA check counts a
filling CTA's copies for every CTA it fills: 8192 bytes per stage here, reported as 4096 per
CTA. It can't see which CTA each byte lands in, so a stage whose copies take fewer distinct
`Cluster.fill` masks than the CTAs it fills (two copies into the same CTA, say) is reported
`unverified` rather than `verified`.

Fills are for CTAs that can't issue their own loads, not a speedup. A persistent GEMM whose
pair leaders load the A/B ring for both CTAs (`LeaderLoadGemm` in
[`test_cute_warpdrive_fills.py`](../../../../test/test_cute_warpdrive_fills.py)) was 6-13%
slower than loading from both CTAs for a pair on a GB300, and within noise for the 2x2 cluster
(November 2026, while adding fills; CUDA-graph replay, warm L2, default clocks, best of five
interleaved rounds at 4096^3, 8192^3, 8192 x 11008 x 4096 and 2048 x 2048 x 8192).

### A rank0 consumer on a multicast channel

A multicast channel may have a `rank0` consumer next to the consumers in the other CTAs, for
example an observer that also reads rank 0's stages. Every CTA a copy lands in still needs a
consumer that waits on its barrier: a stage that fills with no waiter would take completions on
a barrier nobody arms. With a rank0 participant the channel spans the whole cluster, so each
consumer releases every producer.

## Limits

- A copy can land only in CTAs that have a consumer (or, with pairs, whose leader has one).
  Multicast copies need a producer in each CTA they land in, unless `fills` names the CTA that
  arms there.
- `fills="pair"` needs MMA pairs and a `leader` producer; `fills="cluster"` needs a `rank0`
  producer.
- Clusters have at most 8 CTAs (the portable limit).
- The alias diagnostic doesn't model clusters.
