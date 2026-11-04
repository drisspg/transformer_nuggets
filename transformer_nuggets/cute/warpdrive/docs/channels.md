# Channels

[Back to warpdrive](../README.md)

```python
self.ab = Ring(3, Tma(stage_bytes), Umma)
self.acc = Ring(2, Umma, Threads)
```

A `Ring` cycles through stages `0..n-1`. The producer acquires an empty stage, fills it and
commits; the consumer waits for it, reads it and releases. Each endpoint carries the next
stage and phase. A handle keeps the stage and phase it acquired even after the endpoint moves on.

```python
with ab.produce() as h:
    ...  # fill stage h.index; commit on exit

with ab.consume() as h:
    ...  # read stage h.index; release on exit
```

`Tma(tx_bytes)` completes the full barrier through TMA. Supply the bytes per stage per
producing CTA. `Umma` signals with an elected `tcgen05.commit`; `Threads` has every thread
of every bound role arrive. A TMA or UMMA side must bind to exactly one single-warp role.
TMA is a producer mechanism, not a consumer mechanism.

## Counts come from the roles

For `Ring(2, Umma, Threads)`, an MMA producer on one warp contributes one UMMA completion to
the full barrier. A consumer on warps 0–3 contributes 128 arrivals to the empty barrier.
The producer cannot reuse the stage until all 128 arrive. Multiple consumers form a join.

For `Threads`, predicate the work, not the handoff:

```python
with acc.consume() as h:
    if output_in_bounds:
        ...  # only valid lanes store; every role thread releases
```

Putting the whole `with` under `if output_in_bounds` would change who arrives without changing
the barrier count. Compilation doesn't prove that all the declared threads reach each signal,
or that producers and consumers run the same number of iterations.

Elected-per-warp arrivals, async-copy relay warps and sparse protocols can use
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
ints in `[0, n)`, with at most 31 slots. An instance-bound end contributes only to its own slot
and must still pass it explicitly. Passing another instance's slot is an error.

An unbound end can visit any slot. This is useful for one MMA warp producing two slots that
two softmax instances consume independently. The [FA4 example](../examples/fa4_fwd.py) uses
this pattern.

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
Post-close pointer use is still your responsibility; the checker is not an ownership system.

## Storage and fences

The channel can just carry synchronization while you allocate and index the buffers yourself.
For a small CTA-local payload:

```python
self.stats = Ring(2, Threads, Threads, payload=(cutlass.Float32, (32,)))
```

This allocates uninitialized, 16-byte-aligned SMEM with full shape `(32, 2)`. `h.data` selects
that handle's stage. User pair-scoped channels cannot have these layer-owned payloads.

For swizzled tiles, several buffers per channel or overlapping allocations, use
[`SmemPlan` and `buffer=`](storage.md). Those buffers are also selected through the handle;
binding them does not add fences or change remote memory scope.

`commit_fence` and `release_fence` accept `"tmem_store"` or `"smem_async"`. They run before
the corresponding managed signal. TMA producers cannot have a commit fence.
`commit(signal=False)` and `release(signal=False)` omit both the signal and its fence.
Your other async operations still need their own completion waits.

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

Allocations round up to a power of two from 32 to 512 columns. Capacity, duplicate names and
alias bounds are checked. Every role using TMEM must take `tmem: Tmem`, including raw roles,
so it participates in teardown. Without any annotated users, no TMEM allocation is made.
Single-CTA allocation happens in `begin_roles()`; pairs allocate after the cluster sync in
`run_roles()`.

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
two instances, with matching CTA placement. All threads of each seat participate.
`Turnstile(enabled=False)` emits no synchronization and provides no exclusion; the declaration
still has to be valid. FA4 uses it to alternate the two softmax instances around exp2.
