# Dropping down

[Back to warpdrive](../README.md)

Role bodies are CuTeDSL, and every managed feature hands out the barrier, ID, phase or state
it drives. Raw code can wait or signal on those directly, beside the managed calls. The layer
then checks less: each row of the table says what it no longer verifies.
[raw_interop.py](../examples/raw_interop.py) runs a managed producer, a raw role that drives
the same ring's external consumer side by pointer, and a raw role that waits natively on a
`NamedBarrier`'s ID ([below](#example-managed-and-raw-on-one-ring)).

## Escape hatches by feature

| Feature | Raw handle | Mixing raw code with it | What becomes unverified |
|---|---|---|---|
| Channel ends (`Ring`, `Slots`) | `end.full_barrier(i)`, `end.empty_barrier(i)`, `end.state` (Ring `PipelineState`, Slots Int32 phase mask), `end.index`, `end.phase(slot)`; `self.barrier_ptr(name)` in the prologue (full at `[0, n)`, empty at `[n, 2n)`) | Wait with `cute.arch.mbarrier_wait(end.full_barrier(st.index), st.phase)` and signal with `cute.arch.mbarrier_arrive(end.empty_barrier(st.index))`; advance the state you read, and assign it back (`end.state = st`) before the next managed call | Handle lifetimes and the alias diagnostic for the raw waits; reading or assigning `end.state` makes outstanding tokens and previews stale |
| Handles | `h.index`, `h.phase`, `h.full_barrier`/`h.barrier`, `h.empty_barrier`, `h.data`, `h.view(...)` | `h.commit(signal=False)`/`h.release(signal=False)` close the handle without its signal or fence; signal the barrier yourself | The signal's count and fence; the alias diagnostic reports orderings only a raw signal provides as `unverified` |
| `override=` | `end.scope` (counts, masks, tx after overrides), `end.channel.derived_scope` (before), `end.targets` | Override the count your raw signals supply: `override={"empty_counts": 1}` for one elected release. Keys: `full_counts`, `empty_counts`, `tx`, `full_mask`, `empty_mask`; an int, a per-slot list or `{cta rank: list}` | The values themselves: `tx=0` and a well-typed mask naming a nonexistent rank are accepted. `describe()` prints each override next to the derived value |
| Half-managed channels (`external=`) | `self.barrier_ptr(name)` (full at `[0, n)`, empty at `[n, 2n)`) | `Ring(n, external="consumer", override={"empty_counts": ...})` (or `"producer"` with `full_counts`): no role binds that side, so the override gives its counts, and a raw role or kernel code waits and signals by pointer, as `consumer` does in the example. Binding a managed end to that side raises, as do its role names, fences, TMA multicast or fills (raw code drives it in each CTA). Alternatively bind the side to a managed role with a raw body (`end.state`, above); use `Barriers` when neither side is managed | Everything the external side does; `describe()` marks its counts the override's |
| `Threads` / `Warps` sides | `end.empty_barrier(i)` / `end.full_barrier(i)` | `Warps` is one elected arrive per warp after `__syncwarp`: `cute.arch.sync_warp()` then `with cute.arch.elect_one(): cute.arch.mbarrier_arrive(bar)`. A count other than threads or warps needs `override` | The override's count |
| Exclusive rings (`consumer_sharing=Sharing.EXCLUSIVE`) | `skip(n)` is `end.state.advance(n)` plus trace bookkeeping | `skip` takes a Python int or a dynamic `Int32`; raw code may advance `end.state` itself. `override=` applies; `empty_counts` lifts the equal-contribution rule | A dynamic skip or raw state makes the ownership check `unverified`; an `empty_counts` override marks the counts the author's |
| Round-robin sides (`Sharing.ROUND_ROBIN`) | `end.state` | Instance j of k starts at epoch j and each acquire/wait advances k (`end.channel.turns(side)`); raw code advancing the state by hand must step k too | Raw state makes the ownership check (the trip counts) `unverified`; the alias diagnostic treats orderings through the ring as `unverified` |
| Try tokens | `token.ready` (a `Boolean`) | Loops may carry a `Token` (checked). For a protocol the checks can't follow, pass a raw `Boolean`, e.g. `cute.arch.mbarrier_try_wait(bar, phase)`, inside `with unchecked("token", reason=...)` | Inside the block, the token's end, epoch and region |
| Previews (`peek()`) | `preview.index`, `preview.phase` | Index raw tensors with them; the next `wait()` on the end owns the epoch | Raw accesses through the indices are outside the alias diagnostic |
| Named channels (`sync="named"`) | `end.channel.full_barriers[s]`, `end.channel.empty_barriers[s]` (a `NamedBarrier` per slot, ring stages with consecutive IDs), `op.named_barrier_ids` | Their `.sync()`/`.arrive()`, or native `cute.arch.barrier(barrier_id=nb.id, number_of_threads=nb.threads)` / `barrier_arrive`. Named channels take no override and have no mbarriers | Participation per generation (never checked) |
| `NamedBarrier` | `nb.id`, `nb.threads` | Native `cute.arch.barrier(barrier_id=nb.id, number_of_threads=nb.threads)` joins the same barrier as `sync()`/`arrive()`. Reading `.id` while tracing records the role (`native id by ...`) | Incomplete observed participation in a CTA where a role (or kernel code) read `.id` is reported `unverified`, not rejected; a count above every thread that runs a role there, and ID collisions, still raise |
| `alias.named_barrier(id, n)` | The ID you pick | For barriers no declaration owns | Collisions with layer, channel and `NamedBarrier` IDs still raise |
| `Barriers(n, counts=...)` | A pointer parameter (`name: Barriers`) or `self.barrier_ptr(name)` | Allocated and fenced with the layer's barriers; `counts=None` leaves initialization to you between `begin_roles()` and `run_roles()`, e.g. `cutlass.pipeline.PipelineUmmaAsync.create(barrier_storage=self.barrier_ptr(name), defer_sync=True)` | Everything: no protocol checks apply |
| Raw roles (`@role(raw=True)`) | The context, `Instance`, `Tmem`, `Barriers`, `self.<NamedBarrier>` | They can't take managed ends, work cursors or turnstile seats, and their return values are ignored. A non-persistent local schedule's item is `self.work.single_item()`. In a persistent kernel, use a managed role with a raw body | Handle checks and the persistent-stream drain requirement |
| `unchecked(check, reason=...)` | The opt-out record in `describe()` | Checks: `unclosed`, `cross_region`, `tma`, `alias`, `token` | Only the named check, only inside the block |

## Example: managed and raw on one ring

```python
self.ring = Ring(2, external="consumer", override={"empty_counts": 1})
self.read = NamedBarrier(threads=64)  # the consumer's threads
self.done = NamedBarrier(threads=96)  # the consumer arrives, the watcher syncs


@role(warps=range(1, 3), raw=True)
@cute.jit
def consumer(self, ctx):
    full = self.barrier_ptr("ring")  # full barriers at [0, n), empty at [n, 2n)
    empty = full + self.stages
    state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
    for e in cutlass.range(ctx.n, unroll=1):
        cute.arch.mbarrier_wait(full + state.index, state.phase)
        ...  # read stage state.index
        self.read.sync()  # every thread has read the stage before the one release
        if t == 0:
            cute.arch.mbarrier_arrive(empty + state.index)
        state.advance()
    self.done.arrive()


@role(warps=3, raw=True)
@cute.jit
def watcher(self, ctx):
    cute.arch.barrier(barrier_id=self.done.id, number_of_threads=self.done.threads)
    ...  # read what the consumer published before its arrive
```

The producer uses `ring.produce()` unchanged. No role binds the ring's consumer side, so the
override gives its count: one arrival per stage; the named barrier orders the consumer
threads' reads before that arrival, which the count alone does not. `describe()` shows
`ring  Ring(2)  producer -> raw code`, `external  consumer side driven by raw code; empty
counts are the override's (unverified)` and, for `done`, `arrive by consumer; native id by
watcher; unverified: roles calling it have 64 of 96 threads; ...`.
`test/test_cute_warpdrive_raw_interop.py` runs it over several stage counts and epoch counts.

## Rules that still apply

- An mbarrier counts arrivals, not thread identities: an overridden count does not elect a
  lane or order the other threads' work before the arrival. That is your code's job.
- Barrier pointers are local. A peer CTA has its barriers at the same offsets, but the
  pointer alone doesn't perform a remote arrival. Raw signals must use the effective counts,
  target CTAs (`end.targets`) and masks.
- Raw accessors don't check bounds or ownership: `Slots.phase(slot)` checks for a Python int,
  not the slot's range or instance binding, and restoring an old `end.state` can break the
  protocol without a diagnostic.
- An opt-out changes the diagnostic, not the hardware synchronization. Keep the raw
  protocol's counts, completion waits and memory ordering documented at the call site.
  Opt-out bookkeeping is tracing-global, not a thread-safe compilation API; use isolated
  processes rather than concurrent compilation threads.
- `unchecked("alias")` excludes the access annotations inside the block from the
  [alias diagnostic](storage.md#check-declared-aliases); it certifies nothing about the
  unmodeled access. A TMA opt-out is distinct from a copy the checker could not recognize.
