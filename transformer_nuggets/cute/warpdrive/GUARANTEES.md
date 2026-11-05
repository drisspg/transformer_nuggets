# Safety contract

[Write a kernel](docs/writing-kernels.md) · [API guides](README.md#more-examples-and-details)

warpdrive derives synchronization from your roles and channels. You supply the algorithm,
layouts and matching work on both sides of each handoff. **A kernel that compiles can still
race or deadlock.**

This contract covers CuTeDSL 4.8 on Blackwell SM100/SM103, with a single CTA, a CTA pair or a
cluster of at most eight CTAs. Example wrappers have their own tensor and numerical contracts.

## What the layer handles

These guarantees assume matching channel epochs and participants, enough storage for live
values, a dependency graph that can make progress, and terminating user work.

| When you use… | warpdrive handles… | You still supply… |
|---|---|---|
| Managed roles and the declared launch shape | Warp placement, register-budget redistribution, barrier initialization and dispatch | Converged entry/exit and correct per-role math |
| A managed `Ring` or `Slots` handoff | Stage/phase bookkeeping, declared arrival counts, publication and reuse after the required releases | Matching epochs, participating threads and storage lifetimes |
| `Tma` / `Umma` mechanisms | Transaction arming / elected MMA completion signals and supported cross-CTA targets | Correct atoms, descriptors, copies and declared byte counts |
| `SmemPlan` and handle-bound views | Allocation and projection onto the handle's stage | Layouts, partitions and views of the named buffer |
| `TmemPlan` and annotated users | Column allocation and coordinated teardown | Every actual TMEM user declared, and correct alias ordering |
| A drained `WorkStream` | Once-only assignment of kept indices to work units, with the same sequence for their participating roles | A valid mapping/schedule and correct work for each item |

A work unit is one CTA or one cluster, not each CTA of a cluster independently. Custom
schedules and mappings must meet the [scheduling protocol](docs/scheduling.md#custom-mappings).

## The authoring rules that matter

**Keep participants and epochs balanced.** A `Threads` side counts every thread of its bound
roles. Predicate the data access, not the handoff:

```python
with acc.consume() as h:
    if output_in_bounds:
        ...  # store valid output; every participating thread still releases
```

Putting the whole `with` inside a lane-dependent condition changes the arrivals without
changing the count. Likewise, producing n+1 stages while consuming n can compile. Counts
alone do not establish liveness: two roles can each wait for a value the other hasn't produced.

**Carry current state.** Return every managed endpoint, work cursor and turnstile seat from a
role. Helpers return updated state and callers reassign it. Close each handle once, in its
opening dynamic region, and do not access its storage after closing. The checker detects
missing returns and local lifetime errors, not stale snapshots or all post-close accesses.

**Drain the work loop.** Use `while work.valid:`, read `work.item` before advancing, and call
`work.advance()` once directly at the end of the body. Return the cursor. The layer checks
the traced loop structure, not matching runtime trip counts inside different roles.

**Own the memory ordering.** A channel's declared fences run before its managed signal.
Buffer binding does not add fences. TMA stores, raw async copies, TMEM aliases and storage
shared across channels may need additional waits and proxy fences. Complete that work before
reusing its storage or exiting. For ordinary writes read by another CTA, use the correct peer
address and `remote_scope="cluster"`; a channel cannot infer either for you.

**Launch what you declared.** Call `begin_roles` and `run_roles` in converged CTA/cluster code.
Use `launch_config(bound)` for the declared block/cluster shape and checked work grid. Keep
raw named barriers clear of IDs 1 and 2 and `op.named_barrier_ids`. Initialize any
`Barriers(counts=None)` yourself before dispatch. Cluster exit synchronization and automatic
producer tails cover the managed protocol, not arbitrary user async operations.

**Keep configuration and workspaces valid.** Construct a new op when static declarations
change; do not mutate a collected one. Include all static algorithm choices in its compile
key, using the [cache integration](docs/roles.md#compile-keys). Do not share a scheduler
workspace between concurrent launches; use `launch_args()` for the current device and stream.
The launch checks do not validate arbitrary indices, pointer arithmetic or resource use in
user code.

## What a successful check means

- **`describe()`:** declaration checks and derived roles, counts, masks, storage and budgets.
  Before a body has been traced, it cannot report that body's checks.
  A build loaded through `jit_cache` restores its saved TMA/alias reports and warnings;
  protocol-traced builds compile uncached.
- **Compilation:** also checks local handle lifetimes, required returned state, recognizable
  work-loop structure and attributable TMA operations. It does not compare arbitrary role
  bodies to prove balanced epochs or a live dependency graph.
- **TMA `verified`:** recognized group and byte accounting. Not proof of descriptor legality,
  disjoint writes or async safety. Opaque uses may remain `unverified`; inspect the reasons.
- **Alias `no hazard found`:** no conflicting annotated pair found under the diagnostic's
  rules. Not proof of safe aliasing. Clusters, raw accesses and round-robin channels have
  [limited or no ordering coverage](docs/storage.md#read-the-result).
- **Passing tests:** evidence for the executed shapes, schedules and toolchain, not all
  accepted programs. Neither profiling plans nor protocol logs prove liveness or performance.

Invalid configurations generally raise `ValueError`; invalid bindings and traced protocols
raise `ProtocolError`. Read the message and any nested compiler cause. **There is no public
numbered error-code API.** The E/R/U/G labels in the maintainer reference identify test cases
and contract clauses, not exception attributes or message prefixes.

## Dropping down

Raw roles still get placement, initialization and annotated TMEM teardown, but not managed
endpoint/cursor checks. `signal=False` omits both the signal and its fence. Overrides and
`unchecked(...)` leave the corresponding reasoning to you; suppress only the specific check
and document the reason. See [Dropping down](docs/dropping_down.md) for the APIs and boundaries.

## Detailed reference

The [maintainer reference](docs/contract-reference.md) retains the precise placement rules,
numbered checks, limitations and validation evidence. It is useful when extending the layer
or investigating a rejection; it is not required reading before writing your first kernel.

<details>
<summary>Links to sections moved from this page</summary>

<a id="the-supported-program"></a>
- [The supported program](docs/contract-reference.md#the-supported-program)
<a id="runtime-guarantees"></a>
- [Runtime guarantees](docs/contract-reference.md#runtime-guarantees)
<a id="round-robin-and-named-channels-r22-r23"></a>
- [Round-robin and named channels (R22, R23)](docs/contract-reference.md#round-robin-and-named-channels-r22-r23)
<a id="constructor-selected-role-configurations"></a>
- [Constructor-selected role configurations](docs/contract-reference.md#constructor-selected-role-configurations)
<a id="alias-diagnostic-a-bug-finder"></a>
- [Alias diagnostic: a bug finder](docs/contract-reference.md#alias-diagnostic-a-bug-finder)
<a id="cluster-placement-rules-r11"></a>
- [Cluster placement rules (R11)](docs/contract-reference.md#cluster-placement-rules-r11)
<a id="work-assignment-r12"></a>
- [Work assignment (R12)](docs/contract-reference.md#work-assignment-r12)
<a id="custom-schedules-r19"></a>
- [Custom schedules (R19)](docs/contract-reference.md#custom-schedules-r19)
<a id="rejection-catalogue"></a>
- [Rejection catalogue](docs/contract-reference.md#rejection-catalogue)
<a id="roles-and-injection--kernelpy"></a>
- [Roles and injection — `kernel.py`](docs/contract-reference.md#roles-and-injection--kernelpy)
<a id="channels-raw-barriers-pair-scope-and-tmem--channelspy-clusterpy-tmempy"></a>
- [Channels, raw barriers, pair scope and TMEM — `channels.py`, `cluster.py`, `tmem.py`](docs/contract-reference.md#channels-raw-barriers-pair-scope-and-tmem--channelspy-clusterpy-tmempy)
<a id="bodies-and-calls--channelspy-protocolpy-kernelpy-schedulingpy"></a>
- [Bodies and calls — `channels.py`, `protocol.py`, `kernel.py`, `scheduling.py`](docs/contract-reference.md#bodies-and-calls--channelspy-protocolpy-kernelpy-schedulingpy)
<a id="tma-and-scheduling--clusterpy-tmapy-schedulingpy-kernelpy"></a>
- [TMA and scheduling — `cluster.py`, `tma.py`, `scheduling.py`, `kernel.py`](docs/contract-reference.md#tma-and-scheduling--clusterpy-tmapy-schedulingpy-kernelpy)
<a id="domain-hardening-added-by-fuzzing"></a>
- [Domain hardening added by fuzzing](docs/contract-reference.md#domain-hardening-added-by-fuzzing)
<a id="what-remains-your-responsibility"></a>
- [What remains your responsibility](docs/contract-reference.md#what-remains-your-responsibility)
<a id="known-gaps"></a>
- [Known gaps](docs/contract-reference.md#known-gaps)
<a id="fuzzing-and-evidence"></a>
- [Fuzzing and evidence](docs/contract-reference.md#fuzzing-and-evidence)

</details>
