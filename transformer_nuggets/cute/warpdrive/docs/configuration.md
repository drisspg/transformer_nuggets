# Configuration: what is tunable and what is fixed

[Back to warpdrive](../README.md)

A compiled kernel is keyed by its constructor call ([compile keys](roles.md#compile-keys)),
so every choice a kernel author might tune has to be a constructor argument: a module
constant would silently share one compiled kernel between two configurations. This page is
the audit of the layer and the four examples. Each hard-coded choice is either fixed by
hardware or the algorithm, fixed by the layer's abstraction (with the change that would lift
it), or a constructor argument.

## The layer

The layer takes no tuning decisions of its own. Everything that changes generated code is
declared by the kernel: channels and their stage counts, signal mechanisms and fences, the
`SmemPlan` and `TmemPlan`, the `WorkStream`, the cluster and the roles.

| Choice | Where | Status |
|---|---|---|
| Role warps and register budgets | `@role(warps=, regs=)`, overridden per instance by `self.roles` ([roles](roles.md#choose-placement-in-the-constructor)) | Constructor-level. Budgets are multiples of 8 in [24, 256] (`setmaxnreg`), one per warpgroup (`setmaxnreg` is `.aligned` over a warpgroup), and the total must fit the launch's registers: hardware. |
| Warps no role owns | Padded to whole warpgroups when any role sets a budget; padding takes its warpgroup's budget, an all-idle warpgroup 24 | Fixed by hardware (`setmaxnreg` scope). 24 is the floor, so an idle warpgroup releases as much as it can; nothing to tune. |
| Dispatch | One warpgroup-uniform branch per register budget, roles split inside it (as CUTLASS does) | Fixed by the abstraction: `setmaxnreg` is `.sync.aligned` over a warpgroup, and `ptxas` allocates each role's code under its budget when the adjustment dominates the role's branch. |
| Named barrier IDs | 1 and 2 for the TMEM base exchange and teardown; named channels take IDs from 15 downward | Fixed by hardware (16 IDs, 0 is `bar.sync 0`). Which IDs are used has no performance effect; raw `cute.arch.barrier` calls in role bodies pick an unused ID (`named_barrier_ids`), and the trace rejects a collision. Lifting it would mean a declared `NamedBarrier` the layer allocates like a channel. |
| Prologue order | Warp 0 initializes mbarriers; one CTA allocates TMEM before the CTA sync from the lowest other warp of a TMEM role (warp 0 if none), a cluster after the cluster sync from the first TMEM role's first warp; `SmemPlan` buffers last; `run_roles` fences, syncs, adjusts registers, dispatches | Fixed by the abstraction. Allocating off warp 0 measured -0.4 to -0.7% on short decodes and 190 ns on the empty-roles kernel ([fixed costs](examples.md#warpdrive-fixed-costs)); starting a TMA before the sync measured +5.9% ([examples](examples.md#single-token-decode)). |
| Teardown order | TMEM users arrive on barrier 2 after their role; the allocator warp waits and frees; a cluster then arrives and waits | Fixed by hardware (`tcgen05.dealloc` needs every reader done; remote arrivals must land before a CTA exits). |
| Wait style | `mbarrier.try_wait.parity` with CuTeDSL's 10 ms suspend hint, in a retry loop; named channels use unaligned `bar.sync`/`bar.arrive` | Fixed by the abstraction. Try-first and hintless retries changed all seven standard configurations' SASS and were 0.7-2.3% slower on FA4 2-CTA CLC split-P; aligned named barriers were within noise ([examples](examples.md#single-token-decode)). A `wait=` policy per channel would lift it. |
| TMEM allocation | The plan's columns rounded up to a power of two, at least 32 | Hardware (`tcgen05.alloc` granularity). |
| Scheduler warp | The first warp no role owns, with its warpgroup's budget, or `WorkStream(scheduler_warp=)` | Constructor-level. |
| Round-robin depth | A round-robin ring needs a multiple of its instance count in stages | Correctness: an instance's parity wait is only exact when it saw the stage's previous epoch itself. |
| TMA, MMA and fences | `Tma(bytes, multicast=, fills=)`, `Umma`, `Threads`, `commit_fence=`, `release_fence=` | Declared per channel by the kernel. |

## The examples

Defaults reproduce the measured configurations: the seven standard builds in
`scripts/warpdrive_sass.py` keep identical normalized SASS with every knob at its default,
and the decode port still matches FA4's partials bit for bit.

### gemm.py

| Choice | Status |
|---|---|
| `tile_n` (256) | Constructor. Multiples of the 32-column epilogue subtile up to the MMA's 256. |
| `ab_stages` (3), `acc_stages` (1) | Constructor. |
| Roles 0-3 / 4 / 5 at 160 / 40 / 40 | `@role` defaults; `roles=` overrides. |
| `TILE_M = 128` | Fixed by the epilogue: one `tcgen05.ld 32x32b` lane per output row. |
| `TILE_K = 64` | Fixed by the layout: 128-byte K-major bf16 rows, the SW128 atom. Smaller steps are possible (the decode example exposes `tile_k`); the teaching kernel keeps the one step so its layouts stay readable. |
| `EPI_COLS = 32` | Fixed by the instruction: the x32 repetition of the epilogue's TMEM load. |

### persistent_gemm.py

| Choice | Status |
|---|---|
| `tile_n` (224 for a pair, 256 otherwise) | Constructor, measured defaults ([examples](examples.md#gemm)). |
| `ab_stages` (fills ~192 KB), `acc_stages` (2) | Constructor. |
| `c_stages` (2), `epi_tile_n` (32) | Constructor. 1 to 8 staging buffers; 16, 32 or 64 columns: 32- to 128-byte bf16 rows of the TMA store box, which must divide `tile_n`. |
| `schedule`, `mapping`, `cluster` | Constructor. |
| Roles 0-3 / 4 / 5 at 160 / 40 / 40 | `@role` defaults; `roles=` overrides. |
| `CTA_TILE_M = 128`, `TILE_K = 64` | Fixed as in gemm.py (CUTLASS's epilogue partition helpers assume 128 rows per CTA). |
| `EPI_BARRIER_ID = 3` | Fixed: a hand-picked ID for the epilogue warps' intra-role sync, above the layer's 1 and 2 and below the named channels' top-down range. No performance effect. |
| MMA K 16 | Hardware: bf16 `tcgen05.mma`. |

### fa4_fwd.py

| Choice | Status |
|---|---|
| `tile_n` (128) | Constructor. 64 or 128 keys per K/V tile; TMEM holds `Q_STAGE x (tile_n + head_dim)` columns, so 256 does not fit. |
| `head_dim` (128, from the tensors) | Constructor. 64 or 128. |
| `kv_stages` (3 per CTA), `split_p`, `turnstile`, `overlap_o_q`, `use_ldred_rowmax`, `schedule`, `mapping`, `cluster` | Constructor. |
| `rescale_threshold` (8.0) | Constructor. FA4's bf16 lazy-rescale threshold in log2 units; 0 rescales on every growth of the row max. |
| Roles softmax 0-3, 4-7 / correction 8-11 / mma 12 / epilogue 13 / load 14 at 192 / 80 / 48 | `@role` defaults; `roles=` overrides. FA4's 176/80/80 and other splits measured no consistent gain ([examples](examples.md#attention-forward)). |
| `M_BLOCK = 128` | Fixed by the algorithm: one softmax thread per query row. |
| `Q_STAGE = 2` | Fixed by the algorithm: FA4's two softmax warpgroups each own a Q tile, with one S and one O region each. Another count is a different role graph, not a knob. |
| `SPLIT_P = 3` (of 4) | Fixed: FA4's fraction of the P stores before the early release. Both sides derive their split from the same store count, so it holds at `tile_n=64` (1 of 2). Lifting it is a `split_p_stores` argument. |
| 16-column TMEM loads in rescale and O store | Fixed by FA4's register footprint for those loops. |

### fa4_decode_swapab.py

| Choice | Status |
|---|---|
| `tile_hq` (from the tensors) | Constructor. A power of two up to 32 query heads per KV head. |
| `tile_n` (256) | Constructor. A power-of-two multiple of 128 whose `tile_hq x tile_n / 128` S load stays within one 128-column `tcgen05.ld` repetition. |
| `tile_k` (64) | Constructor. 16, 32 or 64: 32- to 128-byte K-major rows; the K and V^T tiles keep equal bytes. |
| `s_stages`, `p_stages` (4, 4) | Constructor. Even, one half per softmax warpgroup. |
| `head_dim`, `head_dim_v` (128, from the tensors) | Constructor. `head_dim` is a multiple of `tile_k` and only sizes Q and the QK K steps; `head_dim_v` is a multiple of 128 and adds O tiles to TMEM (`S + L + O <= 512` columns: 256 fits at every `tile_hq` with the default S ring). |
| `kv_stages` (FA4's total split 7:5) | Constructor. |
| Roles at 64 / 120 / 208 for 32-head tiles, unadjusted otherwise | Defaults computed from `tile_hq`; `roles=` overrides. |
| `O_STAGES = 2` | Fixed by the algorithm: one O slot per softmax warpgroup (even and odd blocks), merged by the correction tail. |
| `TILE_M = 128` | Fixed: one lane per KV token row of an S subtile and per head dimension of an O tile. |
| `MMA_N_MIN = 8` | Hardware: the smallest `tcgen05.mma` N. |
| `lookahead = 2` (K issued two blocks ahead of V) | Fixed: FA4's issue order. The K ring already bounds the lead, and a larger value with a shallow P ring deadlocks (the load waits for a K stage the QK MMA cannot free until a P stage the PV MMA cannot read without the V the load has not issued), which the layer cannot check. |

A combination that does not fit raises at construction (`ValueError` from the TMEM plan or
the default K/V stage budget) or at first collection (`ProtocolError`/`ValueError` from the
role checks). SMEM is not budgeted at construction: an explicit `kv_stages`, `ab_stages` or
`c_stages` beyond the capacity fails when the kernel compiles.
