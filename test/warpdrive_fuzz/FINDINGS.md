# warpdrive fuzz findings

## Declaration and compile-time fuzzing

### Declaration campaign (seed 0)

Each row is a minimized host repro; these are declaration validation bugs/hardening gaps, not evidence of a valid-program GPU hang. Regressions live in `test/test_cute_warpdrive_fuzz_static.py`.

| Seed | Contract | Minimized repro | Before | Class/status | Commit |
|---|---|---|---|---|---|
| 0 | G3 / E29,E35,E36,E43 | `Tma(-1)`; `Ring(1,Threads,Tma(16))`; `Ring(1,Threads())`; `Barriers(1,counts=[1.5])` | Accepted (bad scalar counts can instead TypeError) | Validation bug; fixed; 21 regression cases failed before / passed after | e1872d6 |
| 0 | G4 / R8 | `TmemPlan().region('a',32).alias('a',of='a',offset=0,cols=16)`; alias width 0/-1; duplicate region; unknown parent | Self-cycle accepted (lookup recurses), widths accepted, unknown parent KeyError | Validation bug; fixed; 6 regressions before / after (nonpositive region cases already rejected, alias cases did not) | bd6d5f6 |
| 0 | G2 / R9 | `role(warps=-1)`; `role(warps=32)`; `role(warps=[])`; `role(warps=0,regs=25)` | Accepted; role-free kernel reports unrelated max() error | Validation bug; fixed; 16 regressions failed before / passed after | 2c1f892 |
| 0 | G2,G6 / E83 | `TileStream(scheduler_warp=-1)`; `TileStream(record=-1)`; `Raster(swizzle=True)` | Accepted | Validation bug; fixed; 8 regressions failed before / passed after | 8962cdf |
| 0 | E44–E47 | `Ring(1,override={'full_counts':1.5})` collected with Producer/Consumer; boolean counts/mask/rank | TypeError or accepted bool-as-int | Diagnostic/domain bug; fixed; 4 regressions failed before / passed after | db8baa0 |
| 0 | G7 / E68 | One role `tiles: Tiles`, no stream; declare `tiles_work` beside a dynamic/CLC stream; same Channel under two attributes | Accepted; later lookup failure/overwritten channel/aliased bindings | Validation bug; fixed; 4 regressions failed before / passed after | db37604 |

### Deferred runtime-domain issue (G6)

Exact host repro: `Raster().decode(0,(0,1,1))` raises `ZeroDivisionError`. `Raster(swizzle=2).decode(0,(1,0,1))` also divides by zero. Host bind/grid paths do not reject negative runtime spaces or signed-Int32 product/terminal-index overflow. Fixing dynamic empty spaces requires safe nonzero decode denominators or a device empty-space branch; runtime negative/overflow rejection requires device predicates/traps or a wider arithmetic domain. Those change generated code, outside B ownership. This is a contract-domain issue / runtime candidate, not a GPU-reproduced layer hang. A's empty-space runtime campaign or parent must establish the runtime failure and choose semantics.

### Cursor mutation campaign (seed 0)

| Seed | Contract | Minimized source | Before | Class/status | Commit |
|---|---|---|---|---|---|
| 0 | G1 / E73 | `while tiles.valid & (count < ctx.n): count += 1; tiles.advance()` | Compile accepted; compound condition can stop before terminal | Validation hardening; fixed with exact-condition check; regression failed before / passed after across all four policies | c08178b |
| 0 | G1 / R12 | `while tiles.valid: tiles.advance(); m,n,l = tiles.work; ctx.out[0] = m` | Compile accepted; reads next item as current | Validation hardening; fixed with same-iteration/descendant-block work-read check; regression failed before / passed after across all four policies | c08178b |
| 0 | U1 | Producer n+1 vs consumer n; complete acquire/commit pair gated by lane 0 | Compile accepted | Accepted (documented obligation); never launched | n/a |
| 0 | U10 | `self.step(ch)` where step acquires/commits but does not return/reassign end | Compile accepted | Accepted (documented obligation); never launched | n/a |
| 0 | U6 | TMA copy barrier rebuilt through an opaque pointer, in single/pair-local/pair-scoped channels | Compile accepted with unverified warning/status | Accepted (documented obligation); never launched | n/a |

Mutation sources are written to `$WARPDRIVE_FUZZ_MUTATIONS_DIR` (default: a temp dir) as `mutation_<sha256-prefix>.py`. Reproduce deterministically from the normal harness: `.venv/bin/python -c "from test.warpdrive_fuzz.mutate import compile_mutation,mutations; compile_mutation(next(m for m in mutations() if m.name=='compound_condition'),0)"` (now raises ProtocolError). `advance_before_work` uses the same command with that mutation name.

Validation: required `gpu-run --timeout 120 auto -- timeout --signal=TERM --kill-after=5s 900s .venv/bin/pytest -n 6 -q test/test_cute_warpdrive*.py`: **494 passed in 54.01s**. No timeouts. Before/after normalized SASS identical for `gemm`, pair-CLC `persistent_gemm`, pair-CLC/split-P `fa4_fwd`; artifacts in `sass_before_b/` and `sass_after_b/`. Compile-only seed-0 full 57-case corpus: 43 expected rejects, 8 valid accepts, 6 accepted obligations, no unexpected findings.

Expanded campaign: seeds **0–49**, each full 184-case declaration corpus + 57 compile mutations, in six isolated CPU processes. **12,050 outcomes**, **7,750 required rejects**, **4,000 declaration/valid compile accepts**, **300 accepted documented obligations**, **0 unexpected findings**. Final contract + README link: aa10c87. Ruff lint/format and 101-unique-ID/link checks passed.

## Runtime fuzzing

Runtime fuzzer (`test/warpdrive_fuzz/{gen,oracle,runtime,sched}.py`). Each row: seed(s), the
minimized standalone regression, contract classification, status.

### F-A1: the inserted scheduler warp breaks a valid `regs=None` warpgroup (R9)

- **Seeds:** first seen as a generator-model mismatch; with the generator following R9 it is
  ~3% of `generate()` seeds, e.g. 54, 70, 73, 95, 100, 116, 124, 240, 267 (dynamic and CLC,
  single CTA and pair). The campaign reports them as `known:F-A1`.
- **Minimized repro** (host only, `describe()`): role `a` on warp 0 with `regs=None`, role `b`
  on warps 4-7 with `regs=160`, `TileStream(policy=DynamicPersistent())`. The scheduler takes
  the lowest unclaimed warp (1, a hole in warpgroup 0) and `_bind_stream` gives it
  `min(regs of all roles) = 160` because its group has no *numeric* budget, so `_validate`
  raises `ValueError: warpgroup 0 has mixed register budgets {'a': None, 'tiles.scheduler': 160}`.
  Regression: `test/test_cute_warpdrive_fuzz_runtime.py::test_scheduler_takes_its_warpgroups_default_budget`
  (strict xfail until fixed).
- **Classification:** bug (declaration-time rejection of a valid program). The user's
  declaration satisfies R9 (each warpgroup uniform, `None` a distinct choice); the conflicting
  budget is layer-derived, and the error blames the user's warpgroup. Workarounds: give the
  group a numeric budget or pass `scheduler_warp=` outside it. Suggested fix: in
  `kernel.py:_bind_stream`, when the scheduler's warpgroup has owned warps, take their budget
  even if it is `None`; use `min(regs)` only for an otherwise idle warpgroup.
- **Status:** fixed by parent in 727c3b9 (`_bind_stream`: the scheduler inherits its warpgroup's
  budget, `None` included; `min(regs)` only for an idle warpgroup). The regression now passes
  as a plain test; the earlier strict xfail XPASSed on 727c3b9.

### Coverage-gap campaign (November 2026, GB300)

New generator knobs (draw from their own RNG stream, so earlier seeds keep their base kernel):
raw roles over a `Barriers` ring (layer- and kernel-initialized; `single_item()` under OneShot;
raw TMEM users), `override=` counts closed with `signal=False` and one arrival per warp (int,
per-slot and `{rank: list}` forms; `Channel(...)` declarations), `unchecked` unclosed /
cross_region / tma / alias sites, SmemPlan `buffer=` storage with TMA `StageView`s and
overlapped buffers under `alias_check` warn/strict/off, IKET builds (both modes, compiled and
run without a capture), concurrent launches on two streams with `launch_args()` workspaces, a
TMA -> tcgen05 MMA -> TMEM epilogue with an exact-integer accumulator check (single CTA, BF16
128 x {32,64} x 64, 1-3 K stages per tile), and the scheduler API: a custom local and a custom
broadcast `IndexSchedule`, a table `WorkMap` without `decodes_any_index` that counts decodes
outside `[0, total)`, and `PackedWork` filled by a PDL producer kernel (table and length start
wrong, so a consumer that read before `griddepcontrol.wait` fails).

- Campaign 1 (seeds 1000+, 19 min, `--jobs 10`, stopped to restart on fixed code): 1,262
  kernels, 1,160 passed. 21 hangs, and 70 declaration rejections once the fix landed mid-run,
  were F-C2; 7 failures and 1 strict-mode rejection were F-C1; 3 ptxas errors were F-C3.
- Campaign 2 (seeds 100000+, 35 min, `--jobs 10`): **2,709 kernels, 2,703 passed**: 3 were F-C1
  (the unclosed variant, fixed during the run), 3 harness artifacts (the checkout was rebased while
  those three children imported it). No hangs, value mismatches or device faults. Rerunning every
  saved non-passing spec of both campaigns on the final code: the F-C1 cases and harness artifacts
  pass, the F-C2 hangs are rejected at declaration, and the F-C3 specs (from the earlier generator)
  still fail in ptxas. Passing kernels per knob: MMA 295, raw roles 623 (95 via `single_item()`),
  `signal=False` overrides 1,251, unclosed/cross_region/tma/alias opt-outs 457/737/35/309,
  overlapped buffers 866, strict/off alias modes 680/439, IKET 426, concurrent launches 333,
  table/packed mappings 235/241 (149 with a PDL producer), reverse local/broadcast schedules
  127/114, CtaPair 801.
- Campaign 3 (seeds 200000+, 10 min, after review fixes): 867 kernels, 866 passed; one more
  F-C1 variant (200811), fixed. Campaigns 1 and 2 checked the reverse schedules' coverage,
  agreement and determinism but not their per-unit order (an oracle gap found in review);
  campaign 3 and 24 targeted reverse seeds check the exact order and pass.
- Clusters (`--cluster`, the knobs apply there too): 2x2 with MMA pairs, 516 kernels then 492
  after a generator fix, and flat 2x2, 319 kernels, all passing after the fix. The two
  failures before it (seeds 300150, 300175: stale TMA words) were generator bugs: manual
  per-warp releases on a multicast TMA channel arrive only in their own CTA, while a multicast
  stage must be released in every CTA it was copied into (U7). Overrides, manual signals and
  unclosed handles now skip multicast channels.
- New mutants, all caught: ignored overrides (hang), `signal=False` still signalling (device
  fault: extra arrivals), any-index decoding of a table mapping (counted decodes), `PdlWait`
  without the wait (wrong items), doubled `Barriers` counts (hang), `tcgen05.commit` replaced
  by a plain arrive (accumulator mismatch) and stage views addressing the endpoint's next stage
  (stale words). Blind spot: replacing only the Umma consumer's release by a plain arrive; the
  MMA finishes reading long before the next TMA write lands.

### F-C1: orderings only a raw signal provides were reported as potential hazards

- **Seeds:** 1045, 1157, 1459, 1598, 1667, 1683, 1689 and strict builds 1086, 2062 (campaign 1;
  `signal=False` closes), 100071, 100074, 100094 (campaign 2; `unchecked("unclosed")` handles),
  200811 (campaign 3; an unclosed handle beside an Umma consumer's release).
- **Minimized repro:** `Slots(2, Threads, Threads, buffer={"P": p, "Q": q})` with `q`
  declared `overlaps=p`; the producer writes Q, the consumer reads P and closes with
  `release(signal=False)` after one elected `mbarrier_arrive` per warp (override
  `empty_counts=1`). The diagnostic reported `potential hazard ... no channel path orders them`.
  Variants: a managed TMA commit beside the raw producer commit (reported its complete_tx
  failure instead), and a consumer that opens under `unchecked("unclosed")` and arrives itself.
- **Classification:** diagnostic bug, not a synchronization bug. The program is ordered (the
  per-warp arrival follows `__syncwarp`, the producer's acquire waits on it), and the contract
  documents raw signals as `unverified`. `_Graph._ops` dropped `signal=False` closes, so no
  path existed and no failure was recorded.
- **Fix:** `alias.py` keeps those closes as separate raw edges; a pair reachable through one (even
  beside a failed managed edge) is `unverified`, and so is a missing path on a channel stage with a
  never-closed handle (again even beside a failed managed edge). Retired stages still publish
  nothing (the close event now records a retire), and a potential hazard on any instance pair wins
  over an unverified pair found earlier in the scan, so the new `unverified` answers cannot hide
  one. Regressions:
  `test/test_cute_warpdrive_alias_models.py::test_raw_release_is_unverified_not_a_hazard`,
  `::test_raw_commit_beside_a_tma_commit_is_unverified`,
  `::test_unclosed_handle_is_unverified_not_a_hazard` (each failed before, passes after), and
  `::test_an_unverified_pair_does_not_hide_a_later_hazard`.

### F-C2: register budgets the layer accepted could hang in `setmaxnreg.inc` (R9)

- **Seeds:** 21 hangs in campaign 1 (1008, 1012, 1023, 1028, 1053, 1106, 1156, 1179, 1187, ...),
  deterministic. All had 20, 24 or 28 padded warps and budgets totalling more than
  `default_regs x threads`.
- **Minimized repro:** 12 warps, budgets 232/232/48 per warpgroup: 65536 registers, which the
  R9 check (`<= 65536`) accepted. Each warp starts with `default_regs = 65536 / 384` rounded
  down to 168, so the launch holds 64512, and `setmaxnreg.inc` only takes registers other
  warps released: the increase waits forever. Seed 1028 with one budget lowered to fit the pool
  passes; unchanged it hangs.
- **Classification:** layer bug (valid-looking declaration accepted, then a device hang).
- **Fix:** `kernel.py` rejects a total above `default_regs x padded threads` at declaration
  (E05), with the pool in the message. Host check only; the seven standard configurations use
  6, 8 or 16 warps, whose pools are exactly 65536, and their normalized SASS is unchanged.
  Regression: `test/test_cute_warpdrive.py::test_register_budgets_fit_the_registers_the_launch_starts_with`
  (accepted before, raises after; 232/232/40 runs).

### F-C3: ptxas register allocation fails beside MMA code at 24/32-register budgets

- **Seeds:** 1512, 1613, 1976 (campaign 1), compile errors, not runtime failures.
- **Repro:** the generated MMA subgraph (budgets 64-224) next to generic roles whose
  warpgroups have 24 or 32 registers: `ptxas fatal: (C7600) Register allocation failed with
  register count of '224'`. 40 for those warpgroups compiles; without the MMA roles 24 compiles.
- **Classification:** toolchain resource limit, outside the contract (G2/G3: the layer does not
  certify CUDA resource feasibility). The generator now keeps budgets >= 40 in kernels with
  the MMA subgraph.
