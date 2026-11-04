# What warpdrive guarantees

warpdrive derives synchronization for your declared roles. It does not prove that your
algorithm is live. A successful `describe()` checks declarations; a successful compile also
checks local handle lifetimes, tile-loop structure and recognizable TMA copies. Neither means
that arbitrary accepted role bodies cannot deadlock.

This contract is for CuTeDSL 4.8 on Blackwell SM100/SM103, with `SingleCta()` or `CtaPair()`.
It describes the layer, not the GEMM/FA4 example wrappers' tensor and numerical contracts.
R/E/U/G identifiers are stable references for tests and fuzz findings.

## The supported program

For the runtime guarantees below, your program must have a finite, live abstract execution:
matching channel epochs and contributors, enough storage for overlapping lifetimes, no
unseeded wait cycle, and terminating user work. Specifically:

- Launch the declared block/cluster shape. Call `begin_roles` and `run_roles` in converged
  CTA/cluster code; every participating thread reaches initialization and exit. Named barrier
  IDs **1 and 2** are reserved. Initialize `Barriers(counts=None)` before `run_roles`.
- Carry ends, cursors and turnstile state through loops/helpers: return and reassign them.
  Keep tailed producer ends and IKET tile cursors under their original parameter names.
- `Threads` means every thread of every bound role; `Tma`/`Umma` mean one single-warp role
  per side. Use matching dynamic counts/predicates across roles, slots and relevant CTAs.
- Close each handle once in its opening dynamic region. Use storage only during its owned
  lifetime. `retire()` is only for a genuinely terminal, unused acquire.
- Drain each managed stream with `while tiles.valid:` and one direct-body `advance()` at the
  end. Read the current item before advancing; do not restore stale cursor state. Custom
  mappings/readiness hooks are coherent and terminating; `skip` permanently hides an item.
- Supply valid storage, layouts, descriptors and atoms, and the required async/proxy fences
  and completion waits. Order alias accesses. Declare every TMEM user.
- Raster extents may be any Int32 values (wider values are converted to Int32 first): empty
  and negative spaces are in the domain (R17),
  and spaces past the Int32 limit are rejected at launch (E102) when you launch with
  `launch_config(work)`/`work.grid()`. A hand-built grid must itself respect that limit and
  CUDA's grid limits. Custom mappings, `record` log offsets and resource limits must fit Int32
  and CUDA limits on their own (U10).
- Use distinct workspaces for concurrent launches. A dynamic workspace starts with two zero
  Int32 words. Use `launch_config(work)`/`work.grid()`, or the supported OneShot 1-D grid.

## Runtime guarantees

These are conditional guarantees relative to that program, not a theorem about accepted source.

| ID | The layer's part |
|---|---|
| R1 | Each role instance runs on exactly its declared contiguous warps. `leader` means rank 0 in a pair and is a no-op for one CTA. Warp holes run no user role. Full/empty barrier offsets are `[0,n)`/`[n,2n)` in each CTA. Elected initialization and the common fence/sync precede dispatch. |
| R2 | A Ring end's operation k uses stage `k % n`; consumer phase is `(k // n) % 2`, producer phase is `1 XOR ((k // n) % 2)`. Acquire/wait advances the end; handles retain their snapshots. Matching epochs preserve publication order and block reuse until the declared consumers release. |
| R3 | Slots keep one independent phase bit per slot (producer starts 1, consumer 0). Only the named slot flips; access order need not be numeric. Instance-bound acquire/wait only accepts its own slot. |
| R4 | CTA-local counts sum bound contributions per slot: Threads contributes `32 * role.num_warps`, Tma/Umma one. Fan-out forms a join: the last contributing consumer controls reuse. Pair contributions are specified below. |
| R5 | Ordinary producer/consumer close signals full/empty. TMA acquire arms bytes and TMA commit emits no signal. UMMA uses an elected `tcgen05.commit`. Declared `tmem_store`/`smem_async` fences precede that side's signal. `signal=False` emits neither signal nor its fence. |
| R6 | Layer payloads are uninitialized, 16-byte-aligned, per-stage CTA-local SMEM, shaped `(*shape,n)`. `h.data` selects h.index. User pair-scoped payloads are rejected; scheduler payloads are explicitly filled in both CTAs. |
| R7 | Pair-scoped producers automatically tail when the consumer is Umma and empty targets include the peer. Ring tails use the live final state; Slots tails drain only that end's acquired slots with their final phases. `retire()` is disallowed there. All threads perform cluster exit arrive/wait after dispatch/TMEM teardown. This is not a wait for arbitrary user async work. |
| R8 | Nonalias TMEM regions occupy declaration-order intervals of `cols * slots`; aliases add no columns and stay inside their parent in each slot. Allocation is the next power of two covering `max(32,used)`, at most 512. No annotated users means no allocation. The first TMEM role allocates in one CTA; a pair needs the first all-CTA TMEM role. All declared users, raw included, join the unaligned teardown barrier before free. |
| R9 | Numeric regs are integer multiples of 8 in [24,256]. Any numeric budget pads launch warps to full warpgroups. Owned warps in each group agree, including None; holes take that budget and entirely idle groups take 24. Adjustment branches are warpgroup-uniform, including peer padding. Accounted registers including default budgets and holes cannot exceed 65536. |
| R10 | An enabled Turnstile orders a legal finite prefix of seats `0,1,...,p-1,0,...`; each seat's previous party supplies its arrival count. All seat threads participate. Disabled turnstiles emit no synchronization and promise no exclusion. |
| R11 | Pair scope, barrier targets/counts, UMMA masks, TMA byte scaling and producer tails follow the placement rules below. Pair TMEM allocation/free uses the two-CTA handshake. |
| R12 | For a correctly launched, drained stream, each kept linear index belongs to exactly one CTA/cluster work unit. Every participating role in that unit observes the same ordered sequence. All-CTA pair roles see it in each CTA; leader roles only in rank 0. This is not "every CTA sees every item." |
| R13 | Raster is a bijection over positive `(m,n,l)` extents, l slowest, with selected raster, bands and reversals; decode intermediates stay within max(index, item count), so it is exact for every index R18 admits. Plain OneShot Raster uses `(m*cs,n,l)` when n/l <=65535, otherwise a 1-D grid. Device decoding follows the actual grid, so the supported hand-built `(total*cs,1,1)` grid preserves item identity too. |
| R14 | Skip runs only for valid indices. Broadcast schedulers acquire a free work slot before claiming, bounding lookahead by stages. CLC waits every response and issues none after failure. Dynamic schedulers report once at exit; the last report fences and zeros `[claims,finished]` for sequential launches/graph replay. |
| R15 | Generated synchronization must implement the finite, live abstract protocol without extra waits, premature reuse, lost publications or added deadlock. Compilation alone does not establish that precondition; a timeout is a candidate failure requiring attribution, not proof of a layer bug. |
| R16 | `describe()` reports derived/effective structure, not runtime correctness. Protocol logs are static traced sites, not execution histories. IKET off emits no IKET ops; cache keys must include IKET/protocol trace choices. Exported plans are per-CTA, work-item-granular approximations, not deadlock proofs. |
| R17 | A space with any Raster extent <= 0 is empty: `bind` stores `(1,1,0)` (what the kernel's `work.space` reads), `work.grid()` launches one work unit under every policy, single CTA or pair, every role's first `valid` is false, no decode divides by zero (host or device), and the dynamic workspace is left zeroed for the next launch. The check is host code in the jit entry point; kernels pay nothing. |
| R18 | `work.grid()` admits a space only when `policy.reach` of Raster's saturating Int64 count fits Int32: `total*cs` for OneShot/Clc (the 1-D grid; terminal index `total`), `total + units - 1` for Static/Dynamic (the last unit's terminal index). Within it, no grid extent, index, claim or CLC result wraps. |

### Constructor-selected role configurations

`@role(config=selector)` evaluates the host callable with the kernel instance on first
collection. It must return a frozen `RoleConfig`; the same placement, participation and budget
checks apply as for decorator shorthand. The configuration is snapshotted for that instance,
not polled on subsequent calls. Construct a new instance for another configuration (U12).
Selectors are never GPU operations. They must not depend on runtime tensor contents.

The standard `cute.cache` compile helpers add `op.__cute_compile_key__()` to the author's
algorithm key. It includes resolved roles (including an inserted scheduler), CTA topology and
tracing flags. Equivalent configurations reuse a cache entry; changed role budgets/placement
cannot collide just because the author's key omitted them. Custom caches must include the hook
explicitly, and other static algorithm choices remain the author's responsibility.

- A non-callable `config` raises `TypeError` at decoration.
- Mixing `config` with explicit non-default placement arguments raises `ValueError`.
- A selector returning anything other than `RoleConfig` raises `ProtocolError` at collection.
- `RoleConfig.raw` must be a bool; other values raise `ValueError`.

Tests: `test/test_cute_warpdrive_role_config.py` covers named functions, lambdas and callable
objects, per-instance snapshots, graph validation, raw roles, equivalent-key reuse and differing
configurations executing correct handoffs through the same algorithm cache key.

### Pair placement rules (R11)

Write `C(r)={0}` for a leader role, `{0,1}` for an all-CTA role.

- Same placement on every participant is **CTA-local**, including all/all (one independent
  channel in each CTA) and leader/leader. Counts are ordinary slot counts, tx unscaled,
  targets empty, masks 1, no tail.
- Otherwise full targets are the union of consumer CTAs, empty targets the union of producer
  CTAs. For each target and slot, Threads counts sum
  `32 * role.num_warps * len(C(role))` across matching signalers. TMA/UMMA contributes one
  arrival per target. Nontarget ranks have placeholder count 1 and no promised waiter.
- Masks have one bit per target. Pair UMMA signalers must be leader-only and use group TWO.
- Ordinary pair-scoped TMA producers in both CTAs feed only leader-only consumers. The leader
  arms `2 * Tma.tx_bytes`; each CTA copies the declared per-CTA bytes using group TWO.
  Leader-only producer → both-CTA consumer needs unsupported multicast and is rejected.
  All/all local TMA instead needs group ONE; leader/leader local accepts either group as far
  as the layer's channel-group check is concerned.
- The internal leader scheduler explicitly publishes each target's payload and uses unscaled
  payload bytes. It is not ordinary multicast TMA.
- Without a payload, remote thread-arrival memory scope defaults to `cta`. If your ordinary
  SMEM/global writes are read by the peer, select `remote_scope="cluster"`. The layer cannot
  infer that requirement or make a pointer address peer SMEM for you.

### Work assignment (R12)

With U launched work units, initial item is unit b. Subsequent indices are:

| Policy | Subsequent work |
|---|---|
| OneShot | Terminal total; skipped items are not reassigned locally. |
| StaticPersistent | `b+U, b+2U, ...`, filtered by skip. |
| DynamicPersistent | `U + atomicAdd(claims,1)`, unique claims, no fixed inter-unit order. |
| Clc | Successfully canceled launch unit, or terminal on failure; pair canceled CTA x is divided by 2. |

Static/dynamic launch at least one unit, capped by total and the supplied capacity; a pair
capacity below two CTAs still launches one cluster. OneShot/CLC launch at least one unit or
the OneShot tile grid. An empty space launches exactly one unit, which receives no work
(R17). Dynamic/CLC add the lowest free scheduler warp unless explicitly set.
Work-ring consumers release on **receipt**, not after computing the item; everyone receives
the terminal marker, which is not work.

## Rejection catalogue

ValueError means invalid configuration, ProtocolError means an invalid declaration binding or
traced protocol, TypeError covers annotation subscripts/decorator placement. Compiler wrappers
can nest the cause; test its type/cause, not just an unrelated compiler error. Diagnostics name
the role/channel/parameter and relevant source sites. A declaration may still have no traced
body at `describe()` time.

### Roles and injection — `kernel.py`

| ID | Rejected input | Error |
|---|---|---|
| E01 | cta other than all/leader | ValueError |
| E02 | Empty outer warp list; empty/non-unit-step/non-range normalized ranges | ValueError |
| E03 | Overlapping role warp ownership | ValueError |
| E04 | Mixed regs (including None) within an owned warpgroup | ValueError |
| E05 | Accounted registers including padding >65536 | ValueError |
| E06 | More than one TileStream | ProtocolError |
| E07 | Unresolvable string annotation | ProtocolError |
| E08 | Layer declaration/implementation class instead of a marker annotation | ProtocolError |
| E09 | Annotation neither a marker, bound marker, Python type, nor absent | ProtocolError |
| E10 | Raw role takes a managed end, Tiles or Turn | ProtocolError |
| E11 | Barriers parameter has no matching declaration name | ProtocolError |
| E12 | More than one parameter would receive context | ProtocolError |
| E13 | Annotated context type is not a frozen dataclass | ProtocolError |
| E14 | Context annotations lack a most-derived type compatible with the others | ProtocolError |
| E15 | Actual context is not an instance of the annotated type | ProtocolError |
| E16 | Actual context has neither attributes nor dataclass status | ProtocolError |
| E17 | Actual context is a mutable dataclass, even if unannotated | ProtocolError |
| E18 | Producer/Consumer parameter has no matching channel name | ProtocolError |
| E19 | Instance-bound end names a Ring | ProtocolError |
| E20 | Instance index >=Slots.n | ProtocolError |
| E21 | Producer method disagrees with declared producer_role | ProtocolError |
| E22 | Consumer method absent from declared consumer_roles | ProtocolError |
| E23 | Turn parameter has no matching turnstile | ProtocolError |
| E24 | Channel slot has no bound producer or consumer contributions | ProtocolError |
| E25 | Tma/Umma side belongs to several role instances | ProtocolError |
| E26 | Tma/Umma side belongs to a multi-warp role | ProtocolError |
| E27 | Turnstile has <2 parties or parties are not instances of one method (disabled too) | ProtocolError |
| E28 | A role takes Tmem but no tmem_plan exists | ProtocolError |

### Channels, raw barriers, pair scope and TMEM — `channels.py`, `cluster.py`, `tmem.py`

| ID | Rejected input | Error |
|---|---|---|
| E29 | Channel stage count not a positive Python int (bool excluded) | ValueError |
| E30 | Fence not None/tmem_store/smem_async | ValueError |
| E31 | TMA producer has a commit fence | ValueError |
| E32 | remote_scope not None/cta/cluster | ValueError |
| E33 | Unknown scope-override key | ValueError |
| E34 | Slots.n >31 | ValueError |
| E35 | Barriers.n not a positive Python int | ValueError |
| E36 | Raw counts not a positive Python int or n positive integer entries | ValueError |
| E37 | Pair UMMA signaler is not leader-only | ProtocolError |
| E38 | Pair turnstile seats have different placements | ProtocolError |
| E39 | Pair has TMEM users but none runs in both CTAs | ProtocolError |
| E40 | User pair-scoped channel has a layer-owned payload | ProtocolError |
| E41 | Ordinary leader-only TMA producer feeds both CTAs | ProtocolError |
| E42 | Nonlocal both-CTA TMA producer has any nonleader consumer | ProtocolError |
| E43 | Tma used as a consumer, including on a local channel | ValueError (constructor) |
| E44 | tx/mask override not a nonnegative Python int | ProtocolError |
| E45 | tx override on a channel without derived TMA bytes | ProtocolError |
| E46 | Count-override rank mapping does not name exactly the integer cluster ranks | ProtocolError |
| E47 | Override counts have wrong length/type or nonpositive entries | ProtocolError |
| E48 | TMEM cols/slots not positive Python ints | ValueError |
| E49 | Alias offset negative or offset+cols exceeds parent width | ValueError |
| E50 | TMEM col lookup names an unknown region | ProtocolError |
| E51 | TMEM slot not a Python int in the region's slot range | ProtocolError |
| E52 | Rounded TMEM allocation exceeds 512 columns | ValueError |

### Bodies and calls — `channels.py`, `protocol.py`, `kernel.py`, `scheduling.py`

| ID | Rejected input | Error |
|---|---|---|
| E53 | h.data on a channel without payload | ProtocolError |
| E54 | retire on an automatically tailed pair channel | ProtocolError |
| E55 | Wrong-side handle close (producer release, consumer commit/retire) | ProtocolError |
| E56 | A handle closes twice, including manual close inside a with scope | ProtocolError |
| E57 | Close outside its opening MLIR block without cross_region opt-out | ProtocolError |
| E58 | Unclosed handle without unclosed opt-out at open | ProtocolError |
| E59 | Slot argument on a Ring acquire/wait | ProtocolError |
| E60 | Non-Python-int Slots acquire/wait slot | ProtocolError |
| E61 | Slots acquire/wait slot outside [0,n) | ProtocolError |
| E62 | Instance-bound acquire/wait uses another slot | ProtocolError |
| E63 | .index on a Slots end | ProtocolError |
| E64 | Non-Python-int Slots phase argument (range/binding are not checked here) | ProtocolError |
| E65 | Unknown fence reaches device-side fence selection | ValueError |
| E66 | Unknown barrier_ptr name after begin_roles | ProtocolError |
| E67 | begin_roles problem names differ from declared streams | ProtocolError |
| E68 | Tiles annotation names no declared stream, including when none exists | ProtocolError |
| E69 | Persistent policy has a nonraw role without Tiles | ProtocolError |
| E70 | Explicit broadcast scheduler warp collides with a role | ValueError |
| E71 | Cursor has no traced advance | ProtocolError |
| E72 | Cursor has more than one traced advance site | ProtocolError |
| E73 | Advance is not directly in the body of the while whose condition reads valid | ProtocolError |
| E74 | IKET cursor parameter no longer holds the live cursor at role return | ProtocolError |
| E75 | Tailed producer parameter no longer holds the same live end record | ProtocolError |
| E76 | Unknown unchecked check (allowed: unclosed/cross_region/tma) | ValueError |
| E77 | unchecked reason not a nonblank string | ValueError |

These checks compare traced structure, not runtime trip counts. Opening/closing within the
same dynamic branch is locally accepted; another participant's matching branch is your job.

### TMA and scheduling — `cluster.py`, `tma.py`, `scheduling.py`, `kernel.py`

| ID | Rejected input | Error |
|---|---|---|
| E78 | Direct context TiledMma has known CTA span inconsistent with cluster.size | ProtocolError |
| E79 | Direct context group-TWO G2S CopyAtom in a single-CTA kernel | ProtocolError |
| E80 | Recognized TMA copy through a tracked barrier has a known wrong required group | ProtocolError |
| E81 | Complete attributable stage's known copy-byte sum differs from effective expected per-CTA bytes | ProtocolError |
| E82 | Raster.along not m/n | ValueError |
| E83 | Raster.swizzle not a positive Python int (bool excluded) | ValueError |
| E84 | Raster.reverse contains an axis outside m/n/l | ValueError |
| E85 | Raster space does not have three entries | ValueError |
| E86 | WorkMapping.words not a positive Python int | ValueError |
| E87 | Dynamic or recorded problem has no workspace | ValueError |
| E88 | Recorded stream calls launch_args before roles are bound | ValueError |
| E89 | Static/dynamic grid requested without max_ctas | ValueError |
| E90 | Producer/Consumer subscript other than Instance | TypeError |
| E91 | Role-marked function lacks the JIT wrapper | TypeError |

TMA checking is **positive evidence only**. It follows recognizable direct SSA uses of
handed-out barriers and counts static copy boxes in the opening region. A complete mismatch
rejects; incomplete inspection alone must not. Opaque/raw pointers, nested/dynamic copies,
multicast/new atoms, unknown IR or opted-out uses can leave a channel `unverified`, with a
warning and reasons in `describe()`. Known wrong-group evidence still rejects. Context atom
checks visit direct fields and are not disabled by a per-use TMA opt-out. `verified` means
recognized group/byte accounting, not descriptor legality, disjoint writes or async safety.

### Domain hardening added by fuzzing

| ID | Rejected input | Error / owner |
|---|---|---|
| E92 | Warp input cannot normalize, or any warp lies outside [0,32); kernel has no roles | ValueError / kernel.py |
| E93 | regs not None or an integer multiple of 8 in [24,256] | ValueError / kernel.py |
| E94 | Mechanism other than the Threads/Umma class or a Tma instance | ValueError / channels.py |
| E95 | Tma.tx_bytes not a positive Python int | ValueError / channels.py |
| E96 | Duplicate TMEM region/alias name (including a self-alias); alias cols not positive int; offset not int | ValueError / tmem.py |
| E97 | Alias names an unknown parent | ProtocolError / tmem.py |
| E98 | Two attributes reference the same channel/turnstile/barrier/stream object, or a generated scheduler name overwrites an attribute | ProtocolError / kernel.py |
| E99 | scheduler_warp not None or int in [0,32); record not a nonnegative int; broadcast stages not positive int | ValueError / scheduling.py |
| E100 | Tile-loop yielded condition is not exactly the cursor's valid SSA value | ProtocolError / scheduling.py |
| E101 | tiles.work read after advance in the same iteration's block or descendant block | ProtocolError / scheduling.py |
| E102 | Space whose `policy.reach` exceeds Int32 (R18), launched through `launch_config(work)`/`work.grid()` | CUDA launch error (`CUDA_LAUNCH_INVALID_CONFIG`; zero-wide grid, raised at the call) / scheduling.py |

E102 is a runtime rejection: extents are only known inside the compiled entry point, which
cannot raise a Python exception, so it launches a zero-wide grid that CUDA rejects. It leaves no
sticky CUDA error; later launches run normally. Inside CUDA-graph stream capture the rejected
launch invalidates the capture, like any failed launch.

Boolean values are not accepted as integer declaration quantities. E60/E64 retain the older
Python-int slot accessor behavior; this is not a blanket bounds/ownership check on raw accessors.

## What remains your responsibility

| ID | Not established by declaration/compile acceptance |
|---|---|
| U1 | Cross-role/slot/CTA epoch balance and uniform arrival participation. n+1 producer versus n consumer, or lane-gating a complete handle pair, can compile. |
| U2 | Algorithm liveness/capacity: balanced counts can still form an unseeded dependency cycle. |
| U3 | User math, indices, layouts, initialization, storage identity, descriptors and alias-safe access order. |
| U4 | Async completion and proxy fences beyond the declared handoff fence; TMA stores and TMEM alias users need their own lifecycle. |
| U5 | Peer ordinary-memory publication needs cluster scope and correctly addressed peer storage. |
| U6 | Unverified/raw-pointer TMA and opted-out uses require independent validation. |
| U7 | Overrides, signal=False, raw state/pointers/barriers. Validly typed masks may name nonexistent targets; tx=0 is still permitted as an override. |
| U8 | unchecked suppresses only its selected property/site: unclosed at open, cross_region at close, tma at barrier access. Tracing-global opt-out lists are not a thread-safe compilation API. |
| U9 | Raw roles get placement/regs/init/annotated TMEM teardown, not managed end/cursor/turnstile checks or the persistent-drain requirement. |
| U10 | State return/reassignment, coherent skip/maps, workspace ABI/zeros, and avoiding concurrent workspace reuse. A custom mapping's `bind` must make `decode` safe on empty spaces and past the end, and it must override `size64` when its count can exceed Int32 (the default widens an already-wrapped `size`). `record` log offsets and launch resources other than the grid extent must fit Int32 and CUDA limits. The strict tile-loop checks do not recognize stale state restoration or prove all per-item work is correct. |
| U11 | Frozen contexts are shallow: referenced tensors/lists/objects remain mutable. Namespaces are not made immutable. |
| U12 | No declaration mutation/recollection promise after first collection. Fresh compile keys must include all static choices, IKET and protocol tracing. |
| U13 | No post-close payload/pointer ownership check; raw barrier accessors and Slots phase range/binding are not bounds-checked. |
| U14 | No performance threshold, inter-unit deterministic order, race-freedom proof or profiler-plan proof of liveness. |

## Known gaps

- **G1 (partly closed):** exact condition and post-advance work reads now reject (E100/E101).
  The checker does not prove advance is lexically last, recognize every cursor rebind or
  stale-state restore, or prove useful work happened before it. Correct state carry remains U10.
- **G2/G3/G4 (declaration gaps closed):** invalid warp/regs/mechanism/byte/count domains and
  corrupt TMEM declarations now reject. This does not certify CUDA resource feasibility of
  arbitrary user code or caller-mutated declarations after collection.
- **G5:** no combined cross-role balance or wait-for-graph verifier. U1/U2 are prerequisites,
  not missing mandatory diagnostics.
- **G6 (closed for Raster):** empty and negative spaces run no work under every policy (R17),
  and spaces past Int32 reject at launch (E102, R18). `decode` is defined on bound spaces:
  `Raster().decode(0,(0,1,1))` on raw extents still divides by zero, while
  `decode(i, bind((0,1,1)))` does not. Still the caller's: hand-built grids, custom mappings
  (U10), `record` log offsets, and launch resources other than the grid extent.
- **G7 (partly closed):** missing streams, duplicate declaration objects and scheduler name
  collisions now reject. Post-collection mutation/recollection is still unsupported (U12).
- **G8:** post-close handle access and raw phase/barrier accessor limits remain U13.
- **G9:** TMA recognition depends on the supported IR and direct context fields. Wider
  late-copy/atom-reuse/mixed-path coverage is still desirable; `verified` remains local evidence.
- **G10:** FA4 wrapper validation is narrower than its advertised rank/head-dimension/device
  contract. Wrapper hardening is separate from this layer contract.

## Fuzzing and evidence

The harness separates **required rejection**, **valid abstract runtime program**, and
**accepted documented obligation/unverified**. Never launch a negative compile mutation.
Timeouts and mismatches preserve their seed/source and require minimization and attribution.
Run compilation in isolated processes, not concurrent threads sharing tracing state.

Quick host/compile checks (no GPU needed with an explicit target):

```bash
CUTE_DSL_ARCH=sm_100a .venv/bin/pytest -n 6 -q test/test_cute_warpdrive_fuzz_static.py -m fuzz
```

Long-run CLI, from the checkout root:

```bash
.venv/bin/python scripts/fuzz_warpdrive.py static --seed 0 --budget 10000
CUTE_DSL_ARCH=sm_100a .venv/bin/python scripts/fuzz_warpdrive.py mutate --seed 0 --budget 1000
```

Static probes cover declaration-domain boundaries, bindings, contexts, pair restrictions,
TMEM aliases, overrides and valid random depths/widths/budgets. Mutation probes vary generated
source, ring depth, Slots/instance binding, both close sides, loop/branch placement, all four
tile policies, TMA group, copy multiplicity, byte scaling and opaque barriers. Separate
negative controls compile missing helper returns, unequal epochs and lane-predicated pairs
as documented obligations; they are never launched.

Declaration/compile campaign: seeds **0–49**, **9,200 host probes + 2,850 compile mutations**,
no unexpected outcomes after hardening. Of the compile mutations, 2,150 rejected as required,
400 valid templates accepted and 300 documented-obligation/unverified controls accepted.
The full warpdrive suite including these regressions/quick checks passed **494 tests**.
The three examples' normalized SASS was unchanged by rejection-only hardening.

**Runtime fuzz results** (`test/warpdrive_fuzz/{gen,oracle,runtime}.py`, GB300, November 2026):
generated valid programs (2-6 roles, holes, instances, Rings/Slots up to 31, Threads/Umma/Tma
including 2-CTA TMA, fan-in and joins, overlapping and nested handles, turnstiles, TMEM users,
all four tile policies with skips, caps, the 65535 fallback, runtime-shape relaunches, CUDA
graphs, and CtaPair) are checked for progress (process timeout), exact per-handle epochs and
counts, placement, per-unit item sequences and coverage, and bitwise determinism.

- About 12,800 kernels before the G6 fix and 2,545 after it (plus 500 zero-work spaces) passed.
- One bug found and fixed: F-A1, the inserted scheduler warp took the smallest declared budget
  instead of its warpgroup's `regs=None` (`test_scheduler_takes_its_warpgroups_default_budget`).
- compute-sanitizer racecheck/synccheck/memcheck and CUTracer random delays (5k-100k ns) on 56
  sampled seeds: no hazards or mismatches.
- Layer mutants caught by the oracle: short joins (single and pair), wrong ring parity, shared
  slot phases, static over-stride, dynamic counter never reset, turnstile without waits.
  Blind spots: a work-ring consumer releasing before it reads (window too narrow) and a dropped
  pair tail (no value changes).
- Not covered: real MMA/TMEM data (the example tests cover those), raw roles, `Barriers`,
  overrides, `unchecked`, IKET, concurrent launches sharing a workspace, real PDL.

Run `pytest -m fuzz test/test_cute_warpdrive_fuzz_*.py` for the fixed-seed quick mode and
`python scripts/fuzz_warpdrive.py {runtime,static,mutate} --help` for long campaigns.
