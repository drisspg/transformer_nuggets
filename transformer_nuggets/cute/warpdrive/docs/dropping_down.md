# Dropping down

[Back to warpdrive](../README.md)

Role bodies are CuTeDSL. Raw barriers and `cutlass.pipeline` can live beside managed channels.
Use the part you need; the checks can't validate a protocol you implement outside them.

## Override the count, signal yourself

```python
self.acc = Ring(2, Umma, Threads, override={"empty_counts": 4})

# Consumer on four warps:
h = acc.wait()
...  # finish the work, fence/join as needed, issue four elected arrivals yourself
h.release(signal=False)
```

An override replaces `full_counts`, `empty_counts`, `tx`, `full_mask` or `empty_mask`.
`Channel`, `Ring` and `Slots` accept it. Counts can be an int, a per-slot list, or a
`{cta_rank: list}` mapping. `describe()` shows the derived and effective values.

The example changes the expected count from 128 thread arrivals to four, but it does not
elect a lane or order the other readers before it. That is your code's job. An mbarrier
counts arrivals, not thread identities.

`h.release(signal=False)` closes the handle without its signal or release fence.
`h.commit(signal=False)` does the same on the producer side. Your manual signals must use
the effective counts, target CTAs and masks. The type checks on an override don't prove that
its values describe a valid protocol.

## Raw barriers and roles

```python
self.acc_bars = Barriers(2)
# Or Barriers(2, counts=[1, 128]) to initialize them automatically.


@role(warps=range(4), regs=160, raw=True)
@cute.jit
def epilogue(
    self, params, *, acc_bars: Barriers, tmem: Tmem
): ...  # acc_bars is a pointer; drive the barriers yourself
```

`Barriers(2)` reserves storage without initializing it. Initialize between `begin_roles()`
and `run_roles()`. `self.barrier_ptr("acc_bars")` gives the pointer in the prologue, so you
can build your own `cutlass.pipeline` there with `defer_sync=True`. Named barrier IDs 1 and 2
are reserved, and [named-barrier channels](channels.md#named-barrier-channels) take IDs from 15
down. Use `alias.named_barrier` for hand-picked IDs so a collision raises while tracing.

Raw roles can take the context, `Instance`, `Tmem` and `Barriers`. They cannot take managed
channel ends, work cursors or turnstile seats, and their return values are ignored. They
still get warp placement and register budgets and join TMEM teardown when annotated as
users. They don't get managed handle checks or the persistent-stream drain requirement.

A raw role under a non-persistent local schedule reads its item with
`self.work.single_item()`, not `blockIdx`. See [raw work items](scheduling.md#one-item-in-a-raw-role).
Persistent and broadcast schedules reject this helper.

## Raw state in a managed role

```python
end.state  # assignable Ring PipelineState or Slots Int32 phase mask
end.index  # Ring's next stage
end.phase()  # Ring; phase(slot) for Slots
end.full_barrier(i), end.empty_barrier(i)
end.targets, end.scope  # target ranks and effective counts/masks/tx/tail
end.channel.derived_scope  # before overrides
h.index, h.phase  # the handle's snapshot
h.full_barrier, h.empty_barrier  # h.barrier is h.full_barrier
tmem.base
```

Barrier pointers are local. A peer CTA has its barriers at the same offsets, but the pointer
alone doesn't perform a remote arrival. Raw phase/barrier accessors do not provide full
bounds or ownership checks. Restoring an old `end.state` can break the protocol without
triggering a diagnostic.

## Suppress one check

```python
with unchecked("cross_region", reason="V(i-1) is released in the next iteration"):
    v_prev.release()
```

`"cross_region"` applies at close; `"unclosed"` applies around acquire/wait; `"tma"` applies
around the tracked barrier access. The reason must be nonblank. `describe()` lists the site,
reason and suppressed check. An explicit TMA opt-out is distinct from a copy the checker
could not recognize.

`"alias"` excludes access annotations inside the block from the
[alias diagnostic](storage.md#check-declared-aliases). It doesn't disable other protocol
checks or certify the unmodeled access.

An opt-out changes the diagnostic, not the hardware synchronization. Keep the raw protocol's
counts, completion waits and memory ordering documented at the call site.
