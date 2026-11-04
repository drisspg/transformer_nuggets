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
