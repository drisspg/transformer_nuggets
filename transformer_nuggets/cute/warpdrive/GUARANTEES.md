# What warpdrive guarantees

warpdrive derives synchronization for your declared roles. It does not prove that your
algorithm is live. A successful `describe()` checks declarations; a successful compile also
checks local handle lifetimes, work-loop structure and recognizable TMA copies. Neither means
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
  Managed roles return every stateful argument: producer/consumer endpoints, work cursors and
  turnstile seats (one object or a tuple). Self, params, instance indices, TMEM views and raw
  barrier pointers are input-only. Local names and tuple order do not matter; a stale snapshot
  is not detected and violates the state-carry obligation.
- `Threads` means every thread of every bound role; `Tma`/`Umma` mean one single-warp role
  per side. Use matching dynamic counts/predicates across roles, slots and relevant CTAs.
- Close each handle once in its opening dynamic region. Use storage only during its owned
  lifetime. `retire()` is only for a genuinely terminal, unused acquire.
- Drain each managed stream with `while work.valid:` and one direct-body `advance()` at the
  end. Read the current item before advancing; do not restore stale cursor state. Custom
  mappings/readiness hooks are coherent and terminating; `keep = False` from `decode`
  permanently hides an item. Custom schedules, mappings and adapters meet the protocol
  obligations in
  [Custom schedules](#custom-schedules-r19) (U15). Raw roles read a OneShot item through
  `self.work.single_item()`, not `blockIdx`.
- Supply valid storage, layouts, descriptors and atoms, and the required async/proxy fences
  and completion waits. Order alias accesses (the alias diagnostic is a bug finder, not a
  check that you did). Declare every TMEM user. Give `begin_roles(layouts=...)` a layout for
  every `smem_plan` buffer and build `StageView` partitions from the buffers they name (U16).
- Raster extents may be any Int32 values (wider values are converted to Int32 first): empty
  and negative spaces are in the domain (R17),
  and spaces past the Int32 limit are rejected at launch (E102) when you launch with
  `launch_config(bound)`/`bound.grid()`. A hand-built grid must itself respect that limit and
  CUDA's grid limits. Custom mappings, `record` log offsets and resource limits must fit Int32
  and CUDA limits on their own (U10).
- Use distinct workspaces for concurrent launches. An atomic workspace starts with two zero
  Int32 words. Use `launch_config(bound)`/`bound.grid()`, or the supported OneShot 1-D grid.

## Runtime guarantees

These are conditional guarantees relative to that program, not a theorem about accepted source.

| ID | The layer's part |
|---|---|
| R1 | Each role instance runs on exactly its declared contiguous warps. `leader` means rank 0 in a pair and is a no-op for one CTA. Warp holes run no user role. Full/empty barrier offsets are `[0,n)`/`[n,2n)` in each CTA. Elected initialization and the common fence/sync precede dispatch. |
| R2 | A Ring end's operation k uses stage `k % n`; consumer phase is `(k // n) % 2`, producer phase is `1 XOR ((k // n) % 2)`. Acquire/wait advances the end; handles retain their snapshots. Matching epochs preserve publication order and block reuse until the declared consumers release. |
| R3 | Slots keep one independent phase bit per slot (producer starts 1, consumer 0). Only the named slot flips; access order need not be numeric. Instance-bound acquire/wait only accepts its own slot. |
| R4 | CTA-local counts sum bound contributions per slot: Threads contributes `32 * role.num_warps`, Tma/Umma one. Fan-out forms a join: the last contributing consumer controls reuse. Pair contributions are specified below. |
| R5 | Ordinary producer/consumer close signals full/empty. TMA acquire arms bytes and TMA commit emits no signal. UMMA uses an elected `tcgen05.commit`. Declared `tmem_store`/`smem_async` fences precede that side's signal. `signal=False` emits neither signal nor its fence. |
| R6 | Layer payloads are uninitialized, 16-byte-aligned, per-stage CTA-local SMEM, shaped `(*shape,n)`. `h.data` selects h.index. User pair-scoped payloads are rejected; scheduler payloads are explicitly filled in both CTAs. Channel buffers are covered by R21. |
| R7 | Pair-scoped producers automatically tail when the consumer is Umma and empty targets include the peer. Ring tails use the explicitly returned final state; Slots tails drain only that end's acquired slots with their final phases. `retire()` is disallowed there. All threads perform cluster exit arrive/wait after dispatch/TMEM teardown. This is not a wait for arbitrary user async work. |
| R8 | Nonalias TMEM regions occupy declaration-order intervals of `cols * slots`; aliases add no columns and stay inside their parent in each slot. Allocation is the next power of two covering `max(32,used)`, at most 512. No annotated users means no allocation. The first TMEM role allocates in one CTA; a pair needs the first all-CTA TMEM role. All declared users, raw included, join the unaligned teardown barrier before free. |
| R9 | Numeric regs are integer multiples of 8 in [24,256]. Any numeric budget pads launch warps to full warpgroups. Owned warps in each group agree, including None; holes take that budget and entirely idle groups take 24. Adjustment branches are warpgroup-uniform, including peer padding. Accounted registers including default budgets and holes cannot exceed 65536. |
| R10 | An enabled Turnstile orders a legal finite prefix of seats `0,1,...,p-1,0,...`; each seat's previous party supplies its arrival count. All seat threads participate. Disabled turnstiles emit no synchronization and promise no exclusion. |
| R11 | Pair scope, barrier targets/counts, UMMA masks, TMA byte scaling and producer tails follow the placement rules below. Pair TMEM allocation/free uses the two-CTA handshake. |
| R12 | For a correctly launched, drained stream, each kept linear index belongs to exactly one CTA/cluster work unit. Every participating role in that unit observes the same ordered sequence. All-CTA pair roles see it in each CTA; leader roles only in rank 0. This is not "every CTA sees every item." |
| R13 | Raster is a bijection over positive `(m,n,l)` extents, l slowest, with selected raster, bands and reversals; decode intermediates stay within max(index, item count), so it is exact for every index R18 admits. Plain OneShot Raster uses `(m*cs,n,l)` when n/l <=65535, otherwise a 1-D grid. Device decoding follows the actual grid, so the supported hand-built `(total*cs,1,1)` grid preserves item identity too. |
| R14 | `decode` runs only for valid indices `[0,total)`, after the readiness hook, for every mapping without `decodes_any_index` (local cursors, the scheduler warp, skip loops and `single_item()` alike); a mapping with `decodes_any_index` may also be evaluated speculatively (at terminal indices, or before readiness in the tile-grid prologue) and the words are discarded; a skipping mapping's `keep` is only used for valid indices. A subclass that overrides `decode` does not inherit `decodes_any_index`. Only a mapping with `skips` may return a runtime `keep` (trace-time check), and a skipping mapping is decoded once per visited index. Broadcast schedulers acquire a free work slot before claiming, bounding lookahead by stages. CLC waits every response and issues none after failure. `AtomicPersistent` reports once in `close_claims`; the last report fences and zeros `[claims,finished]` for sequential launches/graph replay. |
| R15 | Generated synchronization must implement the finite, live abstract protocol without extra waits, premature reuse, lost publications or added deadlock. Compilation alone does not establish that precondition; a timeout is a candidate failure requiring attribution, not proof of a layer bug. |
| R16 | `describe()` reports derived/effective structure (including the SMEM plan and the alias diagnostic's findings), not runtime correctness. Protocol logs are static traced sites, not execution histories. IKET off emits no IKET ops; cache keys must include IKET/protocol trace choices. Exported plans are per-CTA, work-item-granular approximations, not deadlock proofs. |
| R17 | A space with any Raster extent <= 0 is empty: `bind` stores `(1,1,0)` (what the kernel's `bound.space` reads), `count64` is 0, `bound.grid()` launches one work unit under every schedule, single CTA or pair, every role's first `valid` is false, no mapping without `decodes_any_index` is decoded (Raster and the adapters may evaluate and discard index 0), no decode divides by zero (host or device), and the atomic workspace is left zeroed for the next launch. The check is host code in the jit entry point; kernels pay nothing. |
| R18 | The mapping's `count64` (Int64, saturating at 2^31; Raster's never wraps) is evaluated once on the host and narrowed to the Int32 `total` the kernel receives. `bound.grid()` admits a space only when `schedule.index_limit` of that count fits Int32: `total*cs` for OneShot/ClusterLaunchControl (the 1-D grid; terminal index `total`), `total + units - 1` for Static/Atomic (the last unit's terminal index). Within it, no grid extent, index, claim or CLC result wraps, and the narrowing never truncates a launched count. |
| R19 | For any `IndexSchedule`/`WorkMap`/`Readiness` subclass meeting U15, the layer calls the protocol exactly as documented (below) and keeps owning distribution. Built-in schedules go through the same protocol. Normalized SASS of seven example configurations (gemm OneShot; persistent GEMM static single/pair and 2-CTA CLC; FA4 single, 2-CTA CLC, 2-CTA CLC split-P; CuTeDSL 4.8.0, GB300) was unchanged by making the protocol public and by the `(keep, words)` decode; host-side counting changed it only by reading `total` from a kernel parameter instead of recomputing the extent product, with neutral CUDA-graph timings. |
| R20 | `self.work.single_item()` returns, in a raw role under a non-persistent local schedule, the item and validity a managed role's cursor would start with: the same `first_index`, readiness hook, dropped items and tile-grid/1-D decoding, so it is correct on the 65535 fallback, hand-built 1-D grids, pairs and custom mappings. |
| R21 | `smem_plan` buffers are allocated by `begin_roles` after barriers, payloads and the TMEM holder, in declaration order, with their declared dtype, layout (swizzle included) and alignment; an `overlaps=` buffer is the root buffer's first byte recast to its own dtype/swizzle/layout and allocates nothing. Each CTA of a cluster holds every buffer at the same offset. A binding's stage mode and every registered `StageView` have exactly the channel's stage count. `h.data`, `h.view` and `h.access` address the handle's own stage (`h.index`), never a separately advanced cursor. Binding a buffer changes no count, scope, `remote_scope` or fence. Porting the examples' hand-indexed stages to buffers left the normalized SASS of the seven R19 configurations unchanged (CuTeDSL 4.8.0, sm_100a target compiled on a GB300 host, November 2026; both revisions compiled in one process, `nvdisasm` listings compared with addresses stripped). |

Role parameters may be positional or keyword-only. The dispatcher follows each signature,
while dependency binding remains by parameter name and annotation. The examples use
`self, params, *, ...state`; keyword-only state is returned in the same way as positional state.

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

## Alias diagnostic: a bug finder

The alias diagnostic ([alias.py](alias.py), [usage](docs/storage.md#check-declared-aliases)) is
**not a guarantee**.
It reports `potential hazard` when it finds a conflicting annotated access pair on declared
overlaps with no ordering edge under its rules, and `unverified` when an access or an edge is
not visible to it. `no hazard found` means only that it found nothing among the annotated
accesses; it never reports an alias as proven or safe. It does not establish U3/U4 for you.

What it does promise: it is host-only (annotations, `alias_check` and the recording wrappers
`fence_async_shared`/`bulk_commit_group`/`bulk_wait_group`/`named_barrier` emit exactly the
`cute.arch` instructions the code would emit without it; normalized SASS of the seven R19
configurations is identical with `alias_check` "warn" and "off", sm_100a target compiled on a
GB300 host, November 2026); it reads only the traced
annotations and managed channel operations; `"strict"` raises only on a potential hazard; and
`unchecked("alias", reason=...)` removes exactly the accesses annotated inside the block.
`__cute_compile_key__()` includes `alias_check`, so the standard cache helpers never serve a
strict build from a warn/off one (custom caches must include the hook, U12).

Known blind spots (phase 1): CtaPair kernels (reported unverified), unannotated or raw accesses,
raw ordering instructions, raw roles, raw signals (`signal=False`), ring-stage SMEM shared by two
channels, loops that run fewer than twice, epochs whose traced positions do not correspond
across roles (U1), partial footprints narrower than the annotation states, and the tcgen05
waits/thread-sync fences around barriers and signals (assumed, not checked). Rules and PTX
citations: `alias.RULES`.

Tests: `test/test_cute_warpdrive_alias.py` (unmodified FA4 single/split-P/O-over-Q report no
hazard; QK-before-PV, a dropped epilogue->load edge with sO over sQ and a raw TMEM read are
reported; strict, opt-out, off and cached-mode cases), `test/test_cute_warpdrive_alias_models.py`
(an FA4-backward-style cross-warp P-over-S model, a two-stage ring handing over unstaged TMEM, a
loop-carried same-role MMA pair, a TMA load read through another TMA channel, IR identical on
and off) and `test/test_cute_warpdrive_storage.py`.

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

| IndexSchedule | Subsequent work |
|---|---|
| OneShot | Terminal total; dropped items are not reassigned locally. |
| StaticPersistent | `b+U, b+2U, ...`, filtered by `keep`. |
| AtomicPersistent | `U + atomicAdd(claims,1)`, unique claims, no fixed inter-unit order. |
| ClusterLaunchControl | Successfully canceled launch unit, or terminal on failure; pair canceled CTA x is divided by 2. |

Static/atomic launch at least one unit, capped by total and the supplied capacity; a pair
capacity below two CTAs still launches one cluster. OneShot/CLC launch at least one unit or
the OneShot tile grid. An empty space launches exactly one unit, which receives no work
(R17). Atomic/CLC add the lowest free scheduler warp unless explicitly set.
Work-ring consumers release on **receipt**, not after computing the item; everyone receives
the terminal marker, which is not work.

### Custom schedules (R19)

What the layer does for any schedule, mapping and readiness hook:

- Host, in the jit entry point: `mapping.bind(space)` once in `WorkStream.bind()`, then
  `mapping.count64` once, narrowed to the Int32 `total` the kernel receives (R18);
  `schedule.launch_units(total, cap)` (`cap` is `max_ctas` in work units when `uses_max_ctas`,
  else `None`) and `schedule.index_limit` in `bound.grid()`. A space whose limit passes Int32
  gets the zero-wide grid (E102). The workspace is required and passed to the kernel when
  `uses_workspace` (or `record`). The host never reads device memory: a device-built work list
  binds its capacity and reads its length in `decode` (`examples/packed_work.py`).
- Device, local schedules: every stream user computes `first_index(problem)`, then
  `next_index(index, problem)` once per `advance()` while its index is valid, with readiness and
  dropped items applied as for the built-ins (R14).
- Device, broadcast schedules: one scheduler warp, leader CTA only in a pair, calls
  `claim_next(fetch, problem, previous index)` only after acquiring a free work slot and only
  while the previous claim was valid, decodes the kept item, publishes `[index, *words]` to
  every user (both CTAs of a pair), and calls `close_claims` once after its last claim. When a
  readiness hook or `skips` may move the first item, the scheduler also decodes and publishes
  the first item. Otherwise every user starts at `first_index` and decodes that first item
  locally (no hook runs then, so a local decode reads only launch-time data); every later item
  comes from the ring. The layer derives the ring counts, payload, fences, tails, drain and the
  scheduler's warp and budget; `fetch_channels` channels get the same derivation.
- `close_claims` closes claim state only. It runs when the unit stops claiming, while roles may
  still be computing earlier items (consumers release ring slots on receipt), so it is not a
  teardown, fixup or reduction point. Split-K/stream-K partials need a separate reduction or an
  explicit synchronization of their own.
- `decode` returns `(keep, words)` and runs only for valid indices, after readiness (R14),
  except that a mapping declaring `decodes_any_index` may be evaluated speculatively and the
  result discarded. `unpack(words)` is
  what `work.item` returns.

What it does not check (U15): that indices are in range, that `first_index`/`next_index` agree
across roles, that a mapping covers what the kernel needs, that `decodes_any_index` is truthful,
or that `index_limit` really bounds the schedule.

Tests: `test/test_cute_warpdrive_scheduler_interface.py` runs a local and a broadcast custom
schedule (exact per-unit sequences, drops, pairs, empty and overflowing spaces), checks the
CUTLASS adapter against the vendored CuTeDSL helper and the FA4 adapter against FA4's
schedulers (per-CTA sequences), covers `single_item()` on every grid shape, runs the persistent
GEMM under every built-in, custom and adapted schedule and the FA4 example under swapped
mappings, consumes a PDL-produced `PackedWork` list under every schedule, and checks split-K
partials (host reduction) and split-KV descriptors (host order check).
`test/test_cute_warpdrive_scheduling.py` traps if a mapping without `decodes_any_index` is
decoded outside `[0, total)` under any schedule, with drops, pairs and empty spaces.

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
| E06 | More than one WorkStream | ProtocolError |
| E07 | Unresolvable string annotation | ProtocolError |
| E08 | Layer declaration/implementation class instead of a marker annotation | ProtocolError |
| E09 | Annotation neither a marker, bound marker, Python type, nor absent | ProtocolError |
| E10 | Raw role takes a managed end, WorkCursor or Turn | ProtocolError |
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
| E53 | h.data on a channel without payload or buffer | ProtocolError |
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
| E68 | WorkCursor annotation names no declared stream, including when none exists | ProtocolError |
| E69 | Persistent schedule has a nonraw role without WorkCursor | ProtocolError |
| E70 | Explicit broadcast scheduler warp collides with a role | ValueError |
| E71 | Cursor has no traced advance | ProtocolError |
| E72 | Cursor has more than one traced advance site | ProtocolError |
| E73 | Advance is not directly in the body of the while whose condition reads valid | ProtocolError |
| E74 | Managed role does not return its final work cursor (IKET on or off) | ProtocolError |
| E75 | Managed role does not return a producer/consumer endpoint or turnstile seat | ProtocolError |
| E103 | Managed role returns an unsupported value, state from another role, or duplicate endpoint/cursor state | ProtocolError |
| E76 | Unknown unchecked check (allowed: unclosed/cross_region/tma/alias) | ValueError |
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
| E86 | WorkMap.words not a positive Python int | ValueError |
| E87 | Atomic or recorded stream bound without a workspace | ValueError |
| E88 | Recorded stream calls launch_args before roles are bound | ValueError |
| E89 | Static/atomic grid requested without max_ctas | ValueError |
| E90 | Producer/Consumer subscript other than Instance | TypeError |
| E91 | Role-marked function lacks the JIT wrapper | TypeError |
| E104 | Role signature has variadic positional or keyword parameters | ProtocolError |
| E105 | WorkStream `mapping`/`schedule`/`ready` is not a `WorkMap`/`IndexSchedule`/`Readiness` (or None) subclass instance | TypeError |
| E106 | Local schedule without `next_index`, or broadcast schedule without `claim_next` | TypeError |
| E107 | `single_item()` under a persistent or broadcast schedule, or outside a role (before `begin_roles`) | ProtocolError |
| E108 | Unsupported adapter configuration: `Fa4Mapping` of a scheduler that is not index-addressable, non-positive-int head dims/element size, non-bool `lpt` flags, or a space of the wrong rank; `CutlassRaster` along not m/n or swizzle not a positive int | ValueError / adapters.py |
| E109 | `decode` returns a non-constant `keep` from a mapping without `skips`, or a word count other than `words` | ProtocolError (trace time) |
| E110 | SmemPlan buffer with an empty/duplicate name, a non-power-of-two alignment, a non-int stage axis, or an overlap that needs a stricter alignment than its root | ValueError / storage.py |
| E111 | SmemPlan overlap naming an unknown buffer or a buffer of another plan; a channel buffer that is not in the kernel's `smem_plan` | ProtocolError / storage.py, kernel.py |
| E112 | Channel with both `payload=` and `buffer=`, or a `buffer=` that is not a `SmemBuffer` or a non-empty `{str: SmemBuffer}` dict | ValueError / channels.py |
| E113 | `begin_roles(layouts=...)` not naming exactly the plan's buffers; a stage axis outside the layout's rank; an overlap larger than its root (static layouts); a bound buffer or `StageView` whose static stage mode differs from the channel's stage count, or is dynamic | ProtocolError / kernel.py, storage.py |
| E114 | `end.views(...)` naming no buffer of the channel or passing a non-StageView; `h.view`/`h.access` naming an unknown buffer, or omitting the name on a multi-buffer channel; `h.data` on a channel without payload or buffer (E53); `self.buffer(name)` before allocation | ProtocolError / channels.py, kernel.py |
| E115 | Annotation without read/write, an unknown agent, or a TMEM agent other than threads/umma; TMEM `lanes`/`cols` that are not a range or callable, or a range leaving the region; `warps` that are not ints (at annotation); warps outside the role or a callable footprint leaving the region (when the diagnostic runs) | ValueError / channels.py, tmem.py |
| E116 | `alias_check` other than warn/strict/off; `"strict"` with a potential hazard | ValueError; ProtocolError / kernel.py, alias.py |

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
| E100 | Work-loop yielded condition is not exactly the cursor's valid SSA value | ProtocolError / scheduling.py |
| E101 | work.item read after advance in the same iteration's block or descendant block (this also rejects lookahead reads of the next item; read it on the next iteration instead) | ProtocolError / scheduling.py |
| E102 | Space whose `schedule.index_limit` exceeds Int32 (R18), launched through `launch_config(bound)`/`bound.grid()` | CUDA launch error (`CUDA_LAUNCH_INVALID_CONFIG`; zero-wide grid, raised at the call) / scheduling.py |

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
| U8 | unchecked suppresses only its selected property/site: unclosed at open, cross_region at close, tma at barrier access, alias for accesses annotated inside the block. Tracing-global opt-out lists are not a thread-safe compilation API. |
| U9 | Raw roles get placement/regs/init/annotated TMEM teardown, not managed end/cursor/turnstile checks or the persistent-drain requirement. |
| U10 | State return/reassignment, coherent `keep`/maps, workspace ABI/zeros, and avoiding concurrent workspace reuse. A custom mapping's `count64` must be exact up to 2^31 and never wrap (saturate instead), and its `decode` must be safe for every valid index of a bound space; with `decodes_any_index` it must also be safe past the end and on empty spaces. A device-built list must not change after readiness. `record` log offsets and launch resources other than the grid extent must fit Int32 and CUDA limits. The strict work-loop checks do not recognize stale state restoration or prove all per-item work is correct. |
| U11 | Frozen contexts are shallow: referenced tensors/lists/objects remain mutable. Namespaces are not made immutable. |
| U12 | No declaration mutation/recollection promise after first collection. Fresh compile keys must include all static choices, IKET and protocol tracing. |
| U13 | No post-close payload/pointer ownership check; raw barrier accessors and Slots phase range/binding are not bounds-checked. |
| U14 | No performance threshold, inter-unit deterministic order, race-freedom proof or profiler-plan proof of liveness. |
| U15 | Custom schedules: every returned index is in `[0,total)` or terminal `>= total` (never negative); `first_index` depends only on launch values and `problem`, and `first_index`/`next_index` are identical pure functions in every role; claims across units cover each kept index once (R12 then holds); the schedule terminates; `uses_max_ctas`/`uses_workspace`/`index_limit`/`get_key` are truthful (`index_limit` bounds every index and grid extent; the key names every compile-time field, in lowercase identifier characters); `close_claims` only closes claim state. Custom mappings: U10; each kept index is a distinct work item (repeated outputs such as split-K partials are distinct `(tile, split)` items reduced elsewhere), and every needed item must be reachable as a kept index in `[0,count)`. |
| U16 | Storage: layouts and partitions are yours (the layer infers none and certifies no layout transformation); a `StageView`'s tensor must view the buffer its name binds; accesses through `self.buffer(...)` or prologue tensors bypass handle stage projection; buffer reuse across channels, overlaps and peer (DSMEM) reads need your ordering and `remote_scope="cluster"` where U5 applies. |

## Known gaps

- **G1 (partly closed):** exact condition and post-advance work reads now reject (E100/E101).
  The checker does not prove advance is lexically last, recognize every cursor rebind or
  stale-state restore, or prove useful work happened before it. Correct state carry remains U10.
- **G2/G3/G4 (declaration gaps closed):** invalid warp/regs/mechanism/byte/count domains and
  corrupt TMEM declarations now reject. This does not certify CUDA resource feasibility of
  arbitrary user code or caller-mutated declarations after collection.
- **G5:** no combined cross-role balance or wait-for-graph verifier. U1/U2 are prerequisites,
  not missing mandatory diagnostics.
- **G6 (closed for Raster):** empty and negative spaces run no work under every schedule (R17),
  and spaces past Int32 reject at launch (E102, R18). Raster's `decode` (`decodes_any_index`)
  is defined on bound spaces: `Raster().decode(0,(0,1,1))` on raw extents still divides by
  zero, while `decode(i, bind((0,1,1)))` does not. The adapters normalize extents like Raster
  and have exact `count64` counts; FA4's params are built over one tile (an empty KV as one
  key), and its count is 0. Still the caller's: hand-built grids, custom mappings and schedules
  (U10, U15), `record` log offsets, and launch resources other than the grid extent.
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
built-in schedules, TMA group, copy multiplicity, byte scaling and opaque barriers. Separate
negative controls compile missing helper returns, unequal epochs and lane-predicated pairs
as documented obligations; they are never launched.

Declaration/compile campaign: seeds **0–49**, **9,200 host probes + 2,850 compile mutations**,
no unexpected outcomes after hardening. Of the compile mutations, 2,150 rejected as required,
400 valid templates accepted and 300 documented-obligation/unverified controls accepted.
The full warpdrive suite including these regressions/quick checks passed (494 tests then; 642 with the runtime fuzzer and the G6 tests).
The three examples' normalized SASS was unchanged by rejection-only hardening.

**Runtime fuzz results** (`test/warpdrive_fuzz/{gen,oracle,runtime}.py`, GB300, November 2026):
generated valid programs (2-6 roles, holes, instances, Rings/Slots up to 31, Threads/Umma/Tma
including 2-CTA TMA, fan-in and joins, overlapping and nested handles, turnstiles, TMEM users,
all four built-in schedules with dropped items, caps, the 65535 fallback, runtime-shape relaunches, CUDA
graphs, and CtaPair) are checked for progress (process timeout), exact per-handle epochs and
counts, placement, per-unit item sequences and coverage, and bitwise determinism.

- About 12,800 kernels before the G6 fix and 2,545 after it (plus zero-work spaces from the `--empty` corpus) passed.
- One bug found and fixed: F-A1, the inserted scheduler warp took the smallest declared budget
  instead of its warpgroup's `regs=None` (`test_scheduler_takes_its_warpgroups_default_budget`).
- compute-sanitizer racecheck/synccheck/memcheck and CUTracer random delays (5k-100k ns) on 56
  sampled seeds: no hazards or mismatches.
- Layer mutants caught by the oracle: short joins (single and pair), wrong ring parity, shared
  slot phases, static over-stride, atomic counter never reset, turnstile without waits.
  Blind spots: a work-ring consumer releasing before it reads (window too narrow) and a dropped
  pair tail (no value changes).
- Not covered: real MMA/TMEM data (the example tests cover those), raw roles, `Barriers`,
  overrides, `unchecked`, IKET, concurrent launches sharing a workspace, real PDL.

Run `pytest -m fuzz test/test_cute_warpdrive_fuzz_*.py` for the fixed-seed quick mode and
`python scripts/fuzz_warpdrive.py {runtime,static,mutate} --help` for long campaigns.
