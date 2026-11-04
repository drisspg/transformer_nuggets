# CTA pairs

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
single- and two-CTA paths with the corresponding atoms and layouts.

## Which channels cross CTAs?

If all participants have the same placement, the channel is CTA-local: all/all means an
independent channel in each CTA; leader/leader means one in rank 0. Mixed placements make
it pair-scoped.

For the GEMM's A/B channel, the load role runs in both CTAs and the MMA role only on the
leader. Each CTA issues its TMA copies; the leader arms twice the declared per-CTA byte count.
Use group-TWO copies for this handoff. Ordinary leader-only TMA producing to both CTAs would
need multicast and is rejected.

For the accumulator channel, the leader's UMMA completion signals both CTAs. Their epilogue
threads release back to the leader. Counts and masks follow the declared placement; you don't
have to double the thread count manually. All UMMA signalers in a `CtaPair` must be
leader-only, even on CTA-local channels.

`print(op.describe())` shows the per-CTA counts, targets, masks and transaction bytes. The
exact placement rules are in [GUARANTEES.md](../GUARANTEES.md#pair-placement-rules-r11).

## Work and exit

A work index belongs to the cluster, not to each CTA independently. Local schedules compute
the same sequence in both CTAs. `AtomicPersistent` and `ClusterLaunchControl` claim on the
leader and publish each decoded item to both CTAs. CLC cancels whole clusters.

Pair TMEM allocation happens in `run_roles()` after the cluster synchronization. At least
one TMEM-using role must run in both CTAs; its first eligible role allocates. All declared
TMEM users join teardown before free, including raw roles.

Producers automatically tail when a pair channel's UMMA consumer can still read peer storage.
Return their final endpoint state so the tail uses the right phases. A retired acquire
cannot be tailed and is rejected on these channels. All threads reach the exit cluster
barrier after dispatch and TMEM teardown. This does not wait for arbitrary async work that
your role started.

## Memory scope

Without a payload, `remote_scope` defaults to `"cta"`. That is the normal setting for
TMA/MMA/TMEM handoffs ordered by their own mechanisms. If ordinary SMEM or global writes must
be read by the peer, set `remote_scope="cluster"` and address the peer's storage correctly.
warpdrive cannot infer that requirement from the channel declaration.

Layer-owned `payload=` buffers are CTA-local and disallowed on user pair-scoped channels.
The internal work ring is a separate case: its scheduler explicitly fills both CTAs' payloads.

`SmemPlan` buffers can bind to pair-scoped channels. They still allocate separately in each
CTA at matching offsets, and `buffer=` neither changes `remote_scope` nor adds a fence.
The phase-1 alias diagnostic reports CTA-pair aliases as `unverified`; it does not analyze
cross-CTA ordering.
