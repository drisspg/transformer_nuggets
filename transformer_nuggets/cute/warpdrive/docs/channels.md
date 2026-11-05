# Channels

[Back to warpdrive](../README.md)

```python
self.ab = Ring(3, Tma(stage_bytes), Umma)
self.acc = Ring(2, Umma, Threads)
```

## Stages and phases

A `Ring` cycles through stages `0..n-1`. The producer acquires an empty stage, fills it and
commits; the consumer waits for it, reads it and releases. For zero-based operation k on an
n-stage end without round-robin rotation:

- Stage: `k % n`.
- Consumer phase: `(k // n) % 2`.
- Producer phase: `1 XOR ((k // n) % 2)`.

Acquire/wait advances the endpoint, not commit/release. A handle keeps the stage and phase
it acquired even after the endpoint moves on. Matching epochs preserve publication order and
prevent reuse until the declared consumers release. For mbarrier channels, full barriers
occupy offsets `[0, n)` and empty barriers `[n, 2*n)` from the channel's base in each CTA.

```python
with ab.produce() as h:
    ...  # fill stage h.index; commit on exit

with ab.consume() as h:
    ...  # read stage h.index; release on exit
```

`Tma(tx_bytes)` completes the full barrier through TMA. Supply the bytes per stage per
producing CTA. `Tma(tx_bytes, multicast="n")` declares copies multicast along a cluster mode;
see [clusters](clusters.md#multicast). `Tma(tx_bytes, fills="pair")` has one CTA issue the
copies of others; see [one CTA loads for others](clusters.md#one-cta-loads-for-others). `Umma`
signals with an elected `tcgen05.commit`; `Threads` has every thread of every bound role
arrive, `Warps` one elected lane of each of their warps. A TMA or UMMA side must bind to exactly one single-warp role (or the one instance whose
turn it is on a round-robin side, or the owner of each epoch of
[exclusive consumers](#exclusive-consumers)). TMA acquire arms the expected bytes; commit emits no arrival
because the copies complete the full barrier.
TMA is a producer mechanism, not a consumer mechanism.

## Counts come from the roles

For `Ring(2, Umma, Threads)`, an MMA producer on one warp contributes one UMMA completion to
the full barrier. A consumer on warps 0–3 contributes 128 arrivals to the empty barrier.
The producer cannot reuse the stage until all 128 arrive. Multiple consumers form a join.
In general, CTA-local counts sum the bound contributions for each slot: `Threads` contributes
`32 * role.num_warps`, and TMA/UMMA one. Cross-CTA contributions follow the
[placement rules](clusters.md#placement-rules).

For `Threads`, predicate the work, not the handoff:

```python
with acc.consume() as h:
    if output_in_bounds:
        ...  # only valid lanes store; every role thread releases
```

Putting the whole `with` under `if output_in_bounds` would change who arrives without changing
the barrier count. Compilation doesn't prove that all the declared threads reach each signal,
or that producers and consumers run the same number of iterations.

`Warps` counts warps instead: `Ring(2, Umma, Warps)` expects 4 arrivals from warps 0–3.
Every thread of the bound roles still runs the release, exactly as for `Threads`: it
executes the side's fence, the warp synchronizes with `bar.warp.sync` (which orders the
accesses of its lanes, PTX ISA `bar.warp.sync`), and one elected lane arrives. So the same
rule holds: predicate the work, not the handoff; a lane that skips the release leaves its warp
waiting at the `bar.warp.sync`. Cross-CTA `Warps` arrivals count one per warp of each
signalling CTA. FA4's and CUTLASS's pipelines signal per warp: on the FA4 backward, 256
per-thread arrivals per release instead of 8 cost about 1.5% of the main kernel. `Warps` is an mbarrier mechanism; named-barrier
channels count threads and reject it.

Async-copy relay warps and sparse protocols can use
[count overrides or raw barriers](dropping_down.md). Changing a count doesn't select lanes
or add the synchronization those lanes need.

## Slots and instances

```python
self.s = Slots(2, Umma, Threads)


@role(warps=[range(4), range(4, 8)], regs=192)
@cute.jit
def softmax(self, params, *, s: Consumer[Instance], stage: Instance):
    with s.consume(stage) as h:
        ...
    return s
```

`Slots` lets you choose the slot; each has an independent phase bit. Indices must be Python
ints in `[0, n)`, with at most 31 slots. Producers start every slot at phase 1, consumers at
phase 0; acquire/wait flips only the named slot's bit. Slots need not be visited in numeric
order. An instance-bound end contributes only to its own slot and must still pass it
explicitly. Passing another instance's slot is an error.

An unbound end can visit any slot. This is useful for one MMA warp producing two slots that
two softmax instances consume independently. The [FA4 example](../examples/fa4_fwd.py) uses
this pattern.

## Round-robin instances

```python
self.s = Ring(4, Umma, Threads, round_robin=True)


@role(warps=[range(4, 8), range(8, 12)], regs=120)
@cute.jit
def softmax(self, params, *, stage: Instance, s: Consumer):
    for j in cutlass.range(stage, n_blocks, 2, unroll=1):
        with s.consume() as h:  # block j's S
            ...
    return s
```

With `round_robin=True`, the instances on a side take the epochs in turn. Instance j of k
handles epochs j, j + k, j + 2k, ..., in instance-binding order. Its zero-based operation m
uses stage `(j + m*k) % n` and that epoch's [phase](#stages-and-phases), so softmax[1] above
starts at the second stage and skips softmax[0]'s. Either or both sides can rotate; a side
with one instance is unchanged. Each epoch's count is one instance's arrivals: 128 here,
not the 256 of a join.
`describe()` prints the rotation and the per-epoch count. The instances must be one role with
the same width, and their loops must visit their own epochs; nothing checks that the trip counts
add up. Epochs continue across work items; entering a new work iteration does not reset an
end's rotation. Round-robin rings are CTA-local and cannot multicast. On an mbarrier ring the
stage count must be a multiple of k, and the layer adds no turn-taking. A parity wait is exact only
when the waiter saw the stage's previous epoch itself; with 3 instances on 2 stages, instance 2's
first wait would pass on a barrier nobody has filled.
The [decode example](../examples/fa4_decode_swapab.py) alternates KV blocks this way.

## Exclusive consumers

```python
self.kv = Ring(
    12, Tma(stage_bytes), Umma,
    consumer_roles=("mma_qk", "mma_pv"), consumer_mode="exclusive",
    buffer={"K": k_buf, "V": v_buf},
)

# in mma_qk, for each KV block j >= 3:
kv.skip(v_stages)  # V(j - 3), which mma_pv owns, was filled before K(j)
for _ in cutlass.range_constexpr(k_stages):
    with kv.consume() as h:  # this warp's own epochs: wait, read, release
        ...
```

With `consumer_mode="exclusive"` each epoch of the ring has exactly one owner among the
consumers. The owner waits for it and releases it as usual; every other consumer calls
`skip(n)`, which advances its end's stage, phase and count by n epochs without waiting or
arriving. The counts are one consumer's contribution (1 for `Umma`), not a join's, and the
producer still fills every epoch in order. The default, `"join"`, keeps the fan-out join
described in [counts](#counts-come-from-the-roles). Selecting neither mode changes nothing in
other channels; `skip()` emits only the index arithmetic.

The layer checks the topology: a `Ring` with mbarriers, without `round_robin` or
`sync="named"`; `consumer_roles` declared, each bound to a consumer end; at least two consumer
instances with equal contributions (single-warp roles for `Umma`); one producer instance; and
CTA-local placement. [`override=`](dropping_down.md) applies as on any mbarrier ring. Full-side
keys (`tx`, `full_counts`, `full_mask`) leave ownership untouched; `empty_counts` replaces the
per-epoch owner count, lifts the equal-contribution check, and `describe()` marks the counts
unverified: each owner's release (typically `release(signal=False)` plus a raw arrive) must
supply them. `skip()` exists only on an exclusive ring's consumer ends and is
`end.state.advance(n)` plus bookkeeping. It takes a positive Python int or a dynamic `Int32`
(n >= 0, as `PipelineState.advance` takes); a dynamic count leaves the ownership check
`unverified`. Skipping makes the end's outstanding [try tokens](#try-tokens) stale, and
skipping past an unconsumed [preview](#previews) raises.

When every acquire, wait and skip on the ring is static (no dynamic loop or branch around
them except the work-item loop) and every role uses the ring either once per work item or
once per launch, collection checks ownership: each epoch the producer publishes has exactly
one waiting consumer (a final acquire closed with `retire()` has none), no consumer runs past
the producer, and under a persistent schedule every consumer advances through every epoch of
an item. A violation raises. `describe()` prints `epochs  verified: ...` or, when any of them
(or a skip count) is dynamic, `unverified (...)` with the reason; then ownership is your
obligation.

**A skip can make the next wait inexact.** A wait of epoch e tests parity `(e // n) % 2` on
stage `e % n`, which is exact once epoch e - n was filled: before that, the stage is one
generation behind and the wait passes at once. A joining or round-robin waiter observed epoch
e - n itself. An exclusive consumer observed only the epochs it waited for, so its wait is
exact only when its previous wait was of epoch e - n or later, and its first wait of an epoch
below n. Otherwise another consumer must tell it that epoch e - n was filled. The ownership
check names the first wait that needs this; dynamic schedules leave it to you. A test toy with
a three-stage ring whose second consumer first waits for epoch 4 hangs without that
rendezvous. The layer synthesizes no handshake; the recipe below is FA4's.

The alias diagnostic treats an exclusive ring's stage handoff like any ring's: the owner's
release precedes the stage's reuse, so accesses ordered through one stage of the ring are
checked, and two consumers are not an unordered fan-out. It cannot match epochs across
channels, because owners follow runtime skips: an ordering that goes through the exclusive
ring and another channel is reported `unverified` ("exclusive ring, epoch owners follow
runtime skips").

### Rendezvous recipe: FA4's QKIssued/PVIssued

The [decode example](../examples/fa4_decode_swapab.py) (`SharedKVDecode`) shares one ring
between its QK and PV MMA warps. The loader fills K(0), K(1), then K(j + 2) and V(j) for each
block j, then the last two V blocks. With 12 stages of four K and four V stages per block,
PV's first wait, of V(0), is epoch 12: a full lap after launch. Two
[named barriers](#named-barriers) bound the skew:

```python
self.qk_issued = NamedBarrier(threads=64)  # mma_qk arrives, mma_pv syncs
self.pv_issued = NamedBarrier(threads=64)  # mma_pv arrives, mma_qk syncs

# mma_qk, per block j (after skipping V(j - 3) for j >= 3):
self.pv_issued.sync()
...  # wait for, issue and release K(j)'s stages
self.qk_issued.arrive()
# after the last block: two more pv_issued.sync(); qk_issued.arrive()

# mma_pv: pv_issued.arrive(), then for each of K(0) and K(1): skip it, qk_issued.sync(),
# pv_issued.arrive(); per block j (after skipping K(j + 2) if it exists):
self.qk_issued.sync()
...  # wait for, issue and release V(j)'s stages; arrive on pv_issued after the last MMA
```

When PV waits for V(j), QK has issued K(j + 2), the group filled just before V(j); when QK
waits for K(j) (j >= 3), PV has issued V(j - 3), the group just before K(j). Every earlier
epoch is therefore filled, and both waits are exact for any ring depth. Each side syncs on one
barrier between two arrivals on the other, so neither can arrive twice in one generation (see
[reusable rendezvous](#reusable-rendezvous)). PV's last `pv_issued` arrival is never matched,
as in FA4: that is sound because the decode is `OneShot`, one work item per CTA. A persistent
kernel would balance it. The barriers also pace the two warps, which is part of the measured
effect, not only correctness.

## Named-barrier channels

```python
self.row_sum = Ring(2, Threads, Threads, sync="named", round_robin=True)
self.row_max = Ring(1, Threads, Threads, sync="named", round_robin=True, turnstile="mufu")
self.mufu = Turnstile()

with mufu.turn():
    mh = row_max.acquire()
    ...  # atomically max this warpgroup's block max into sRowMax
    mh.commit(wait=True)  # every producer and consumer thread has arrived
    ...  # read the shared max
```

`sync="named"` builds each slot from a full and an empty CTA named barrier instead of mbarriers:
`acquire`/`wait` are unaligned `bar.sync`, `commit`/`release` unaligned `bar.arrive`. A named
barrier has no phase. It completes once the slot's producer and consumer threads have all
arrived, so the count is their sum. Both sides must be `Threads` and in one CTA, and overrides
don't apply. Each consumer instance arrives once on its slots' empty barriers when its role
starts: its bound slot, every slot for an unbound nonrotating end, or slots s with `s % k == j`
for round-robin instance j of k. That is the named version of a ring starting empty.
Consumers therefore can't write anything the producer's first acquire depends on before that
arrival. The final release is left unmatched at exit. Keep per-epoch participation balanced:
named barriers count threads, not phases, so a fast side must not arrive twice in one generation.

`commit(wait=True)` makes the producer wait on the full barrier too. When it returns, every
thread of the exchange has arrived, so the producers also see each other's writes. FA4's decode
uses it for the shared running max. It applies the same `commit_fence` as an ordinary commit;
it cannot be combined with `signal=False` or used on an mbarrier channel.

Because a named barrier can't tell instances apart, two round-robin instances that share a
stage's barriers (`n % k != 0`) could complete it between themselves. Name an enabled
turnstile over exactly those instances; their acquires (or waits) must run inside its turn,
which is checked while tracing.

IDs are allocated per channel in declaration order from 15 downward, in consecutive blocks
of n full IDs followed by n empty IDs. `channel.named_ids[0][s]` and
`channel.named_ids[1][s]` name slot s's full and empty barriers; `op.named_barrier_ids` and
`describe()` list them. ID 0 is for CTA synchronization, and 1 and 2 belong to the layer,
leaving 13 IDs for channels and [named barriers](#named-barriers). Hand-picked IDs must avoid
the allocated IDs even if they are small. Calls through `alias.named_barrier` are checked
against them; raw `cute.arch.barrier` and `barrier_arrive` calls are not. Named channels have
no mbarrier pointer accessors.

## Named barriers

```python
self.compute_sync = NamedBarrier(threads=256)  # both compute warpgroups
self.max_done = NamedBarrier(threads=160)  # the correction warpgroup and the load warp

self.compute_sync.sync()  # in the compute role
self.max_done.arrive()  # correction: publish without waiting
self.max_done.sync()  # load: arrive and wait
```

A `NamedBarrier` is one CTA named barrier: one ID, which the kernel allocates, and one thread
count. Declare it in `__init__` like a channel; managed and raw roles call it through `self`.
`sync()` arrives and waits until the generation completes; `arrive()` arrives and continues.
A generation completes when `threads` threads have arrived through either call. Both are
unaligned (`barrier.cta.sync`/`barrier.cta.arrive`), so threads may reach them from different
code sites. `aligned=True` emits `bar.sync`/`bar.arrive` instead, which requires every thread
of each participating warp to execute that same instruction convergently. The GEMM epilogues
use it: there it reproduces the hand-picked barrier it replaced, SASS included.

Named barriers take IDs from 3 upward in declaration order, named channels from 15 downward,
so declaring one moves no channel's IDs. Collection raises when they don't fit in 13 IDs.
`describe()` lists each barrier's ID and count and, after a compile, the roles that arrived or
synced. `.id` is the ID for native calls (`cute.arch.barrier(barrier_id=nb.id, ...)`, a
library function that takes one), which may be mixed freely with `sync()`/`arrive()` on the
same barrier. The layer does not see the native calls themselves, but it records which roles
read `.id` while tracing (`native id by ...` in `describe()`). With IKET waits enabled, `sync()` is a
`<name>.sync` range; [pipeline plans](tracing.md) model channels only and leave named barriers
out.

What the layer checks: the count is a multiple of 32, at most 1024 and at most the kernel's
threads. While tracing, an `alias.named_barrier` call raises if its ID is outside 0-15 or
owned by the layer, a named channel or a `NamedBarrier`. A barrier also raises if the roles
using it in some CTA have fewer threads in total than its count, since it could never
complete there. When a role running in that CTA (or kernel code outside the roles) read
`.id`, native calls may supply the missing arrivals: `describe()` reports the participation as
unverified instead, and the barrier raises only if its count exceeds every thread that runs
a role there. More threads than the count is not an error, because a role may take part
with some of its warps (FA4 backward's per-warpgroup epilogue barriers). The layer doesn't
check which threads arrive in each generation, that participation is balanced, or that the
protocol makes progress.

`sync()` records the same diagnostic event as `alias.named_barrier`: a sync whose count is
every thread of the calling role orders that role's warps for the alias diagnostic. An
`arrive()` orders nothing for the arriving thread and doesn't count as one.

### One-shot done flag

One ID, with no empty barrier and no acknowledgement:

```python
self.done = NamedBarrier(threads=128 + 32)  # publishers + waiters

# publishers (128 threads)
...  # write the result to SMEM
self.done.arrive()

# waiters (32 threads)
self.done.sync()
...  # read it
```

The barrier orders the publishers' memory accesses before their arrival with the waiters'
accesses after the sync, as for a named channel. This includes ordinary SMEM stores: the
[PTX ISA's `bar`/`barrier` contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
says that **sync, red and arrive** make the calling thread's prior accesses performed relative
to all participants when the barrier completes; a performed write is visible to them. Sync
additionally prevents that thread from requesting new accesses before completion. Thus
`st.shared; barrier.arrive id, N` in a publisher and `barrier.sync id, N; ld.shared` in a
waiter publish the stores without an extra `membar`. The ISA explicitly allows this mixed
arrive/sync producer-consumer pattern; `bar.arrive`/`bar.sync` are its aligned spellings.
Every writer must reach the arrival, every waiter the sync, with the same ID and count.
Asynchronous proxies still need their own completion waits and fences: this argument is for
ordinary shared-memory accesses, not a replacement for a TMA or TMEM handoff.

Racecheck can report this correctly ordered pattern. On the swap-AB decode (GB300, CuTeDSL
4.8.0, compute-sanitizer 2025.4.1), it flags the correction warps' lane-0 SMEM row-sum stores
against the load warp's reads after `sum_done.sync()`. All 128 correction threads arrive and
all 32 load threads sync on the same 160-thread barrier; the generated PTX and SASS retain
that order, and the row sums are not overwritten. These specific reports are false positives,
not evidence that arrive lacks publication semantics. The same pair is reported for the
older managed named-channel version and FA4's own decode. See
[Decode racecheck](examples.md#decode-racecheck) for source/access mappings, versioned results
and the isolated test invocation. Do not blanket-ignore named-barrier findings: recheck the
accesses, participants, generations and emitted instructions for each report.

It is sound when the flag completes at most once per launch, or once per work item with at
most one item per CTA (the [decode example](../examples/fa4_decode_swapab.py) is `OneShot`).
Otherwise a publisher could arrive for the next item before every waiter had synced, and its
arrival would count toward the earlier generation. Add an edge back to the publishers, or use
a named channel, whose empty barrier is that edge.

### Reusable rendezvous

```python
self.compute_sync = NamedBarrier(threads=256)

for t in cutlass.range(n_tiles, unroll=1):
    ...  # each warpgroup writes its half
    self.compute_sync.sync()
    ...  # read the other half
```

If exactly `threads` threads take part in every generation and all of them sync, reuse is
safe: nobody leaves generation g until all have arrived, so nobody arrives early for g + 1.
A reusable barrier with participants that only arrive needs more: an argument that a fast
participant can't arrive twice before generation g completes, since its second arrival would
count toward g. The barrier supplies no such argument. The protocol must, for example by making
the arriving side wait on another edge before it arrives again, as a named channel's
producer waits on its empty barrier.

## Overlapping handles

A handle does not have to use a context manager. FA4's K and V share a ring; the MMA can hold
V(i-1) and K(i) together, then release V first:

```python
for i in cutlass.range(n_blocks - 1, unroll=1):
    v = kv.wait()  # V(i-1)
    k = kv.wait()  # K(i)
    pv(o, p, v.index)
    v.release()
    qk(s, q, k.index)
    k.release()
```

Here `pv` and `qk` stand for the kernel's MMA code. Nested `with` blocks would release K
first, so they don't express this lifetime.

Close each handle once, in the same dynamic loop/branch region that opened it. A double
close, missing close or cross-region close raises `ProtocolError` with source locations.
Return the updated endpoint at role exit, not the handles.

`h.retire()` closes a producer handle without publishing it. Use it only for a final unused
acquire whose stage will never be consumed. It is rejected on automatically tailed pair channels.
Post-close payload or pointer use is still your responsibility; the checker is not an ownership system.

## Try tokens

```python
token = kv.try_wait()  # probe; no handle, no advance
...  # independent work
h = kv.wait(token=token)  # waits only if the probe failed; advances once

token = out.try_acquire()
h = out.acquire(token=token)  # a Tma producer arms the stage here, once, either way
```

`try_wait()` and `try_acquire()` probe the barrier the end's next `wait()`/`acquire()` takes
(the full or the empty barrier) with one `mbarrier.try_wait.parity`. They return a `Token`
whose `ready` is the probe's dynamic result. Passing it back as `token=` makes the wait
conditional on that result; everything else, including the handle, is unchanged. Slots take
the slot on both calls. Without `token=` the generated code is exactly the blocking wait.

The probe is a single `try_wait`, which the hardware may suspend for a system-dependent
time: it is not a latency bound and not a nonblocking poll. A token records readiness only.
It carries no stage index and can't release, commit or arm anything, and an unused token is
harmless.

A token names one end, side and epoch, not just a parity. While tracing, `wait`/`acquire`
raise `ProtocolError` when the token:

- is not a `Token` from this end's own probe (a value built by hand, another channel's end,
  or the other side of the channel);
- names another slot;
- is stale: an acquire/wait on the end (on that slot, for `Slots`) happened since the probe,
  which covers reuse of the token and an epoch n stages later with the same stage and parity,
  or raw code read or assigned `end.state` (reading hands out the live state, which may be
  advanced in place);
- was probed in another dynamic region (loop or branch). The trace sees a loop body once,
  so a token taken before a loop could be consumed by the first iteration and then reused as
  stale. Probe in the region that consumes the token, as FA4 does inside an unrolled K-step
  loop. Tokens can't be carried across loop iterations.

Named-barrier channels have no probe: waiting on a named barrier (`bar.sync`) is itself an
arrival, so `try_wait`, `try_acquire` and `token=` raise there. The checks don't establish
that probing pays off. On the decode example it measured neutral to slower
([examples](examples.md#single-token-decode)).

## Previews

```python
preview = q.peek()  # wait for the epoch the next q.wait() takes; do not advance
issue_s(preview.view("Q", read=True, by="umma"))
...
h = q.wait()  # the same epoch: this handle owns it, and its release covers the preview
...
h.release()
```

`peek()` (consumer ends) blocks until the stage that the next `wait()` takes is full and
returns a read-only `Preview`. It adds no arrival, opens no handle and doesn't advance the
end. `preview.view(name, *coords, by=...)` and `preview.access(name, by=...)` work like a
handle's, as reads: `write=True` and `commit`/`release`/`retire` raise `ProtocolError`.
`preview.index`/`phase` are the stage and parity it waited on. The FA4 backward uses one for
S_{t+1}'s read of Q_{t+1}, while Q_t's handle stays open for dK_t.

The next `wait()` on the end (same slot for `Slots`) after the peek, in trace order, owns the
previewed epoch. The alias diagnostic records the preview's accesses under that handle, so
they are checked against its release (its mechanism and fences, as for the handle's own
accesses). Trace order approximates the runtime owner: a preview taken in a loop body is
attributed to the wait after the loop, while at runtime the next iteration's wait owns all
but the last. Both are waits of the same channel and side. While tracing, it is an error to
use a preview after that handle closed, in another dynamic region than the peek, to read or
assign `end.state` between the peek and the owning wait, or to leave a preview with no later
wait in its role.
A loop that previews the next iteration's epoch therefore peels its last iteration, as the
backward does; the peeled wait owns the loop's final preview. CuTeDSL also refuses to carry
a preview through a dynamic loop or branch that calls one of its methods.

These checks are trace-local. They don't prove that the owning wait takes the previewed
epoch at runtime, that the preview's asynchronous reads complete before the owner's release,
or that the producer ever fills the epoch. Previews are not general carried handles: a handle still closes in the
region that opened it.

## Storage and fences

The channel can just carry synchronization while you allocate and index the buffers yourself.
For a small CTA-local payload:

```python
self.stats = Ring(2, Threads, Threads, payload=(cutlass.Float32, (32,)))
```

This allocates uninitialized, 16-byte-aligned SMEM with full shape `(32, 2)`; in general the
shape is `(*shape, n)`. `h.data` selects that handle's stage. User pair-scoped channels cannot have these layer-owned payloads.

For swizzled tiles, several buffers per channel or overlapping allocations, use
[`SmemPlan` and `buffer=`](storage.md). Those buffers are also selected through the handle;
binding them does not add fences or change remote memory scope.

`commit_fence` and `release_fence` accept `"tmem_store"` (`tcgen05.wait::st`), `"smem_async"`
(`fence.proxy.async`), or an ordered tuple of both for a side that hands over TMEM stores and
SMEM stores at once (FA4 backward's dS: `commit_fence=("tmem_store", "smem_async")`). A string
is the one-element tuple. The layer emits exactly the declared fences, in that order, before the
corresponding managed signal; unknown or repeated kinds and a commit fence on a TMA producer
are rejected at declaration. `commit(signal=False)` and `release(signal=False)` omit the signal
and all of its fences. The alias diagnostic credits a side's fences by membership: a TMEM
store handed over by a side without `"tmem_store"`, or an SMEM store read by the async proxy
after a side without `"smem_async"`, stays unverified.
Your other async operations still need their own completion waits and proxy fences,
including the lifecycles of TMA stores and TMEM alias accesses.

## TMEM

```python
self.tmem_plan = (
    TmemPlan().region("S", 128, 2).region("O", 128, 2).alias("P", of="S", offset=64, cols=64)
)
```

The two S slots use 256 columns; O uses another 256. P aliases the upper 64 columns of each
S slot and adds no allocation. `print(self.tmem_plan.describe())` shows the column map:

```text
  tmem S   [0, 128), [128, 256)
  tmem O   [256, 384), [384, 512)
  tmem P   [64, 128), [192, 256) (inside S)
```

`tmem.ptr("S", stage)` returns an fp32 TMEM pointer. `tmem.cols("S", stage)` returns the
column offset. `stage` here is a Python int naming a plan slot. To index a ring stage
dynamically, build a tensor covering the region's slots and index it with `h.index`, as the
GEMM examples do. Reinterpret packed BF16 yourself.

Nonalias regions occupy declaration-order intervals of `cols * slots` columns; aliases stay
inside their parent in each slot. Allocations round up to a power of two from 32 to 512
columns. Capacity, duplicate names and alias bounds are checked. Every role using TMEM must
take `tmem: Tmem`, including raw roles, so it participates in teardown. Without any annotated users, no TMEM allocation is made.
Single-CTA allocation happens in `begin_roles()` from the lowest warp other than warp 0
among the TMEM roles (warp 0 when it is the only one), so it overlaps warp 0's barrier
initialization; that warp also frees. Clusters allocate after the cluster sync in `run_roles()` as described in
[work and exit](clusters.md#work-and-exit). All declared users join an unaligned teardown
barrier before the allocator frees TMEM.

Aliasing shares columns, not lifetimes. Order accesses to S and P yourself.

The [alias diagnostic](storage.md#check-declared-aliases) can inspect annotated overlaps, but
it does not prove them safe.

## Turnstiles

```python
self.mufu = Turnstile()


@role(warps=[range(4), range(4, 8)], regs=192)
@cute.jit
def softmax(self, params, *, mufu: Turn):
    with mufu.turn():
        ...  # exp2 work
    return mufu
```

A turnstile gives instances of one method a round-robin critical section. It needs at least
two instances, with matching CTA placement. With p seats, it orders a finite prefix of
`0, 1, ..., p-1, 0, ...`; each seat's barrier counts the previous seat's threads. All threads
of each seat participate.
`Turnstile(enabled=False)` emits no synchronization and provides no exclusion; the declaration
still has to be valid. FA4 uses it to alternate the two softmax instances around exp2.
