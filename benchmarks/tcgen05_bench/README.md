# tcgen05 instruction benchmark

`benchmarks/tcgen05_throughput.py` measures the throughput and latency of every `tcgen05`
instruction form that CuTeDSL can emit on sm_100, sm_103, and sm_107 GPUs. Each
variant compiles to a small kernel whose timed region contains only the measured instruction, so
rows compare instruction forms rather than GEMM pipelines.

| Module | Role |
| --- | --- |
| `variants.py` | Pure-Python variant table, legality prefilter, selectors, public peaks |
| `probes.py` | CuTeDSL kernels: `MmaProbe` (tcgen05.mma) and `TmemProbe` (ld/st/cp/shift/...) |
| `measure.py` | CUDA-graph replay, slope fits, NVML sampling |
| `sass.py` | Hot-loop opcode check on the dumped cubin |

## Requirements

An sm_10x GPU, `pip install -e ".[cute]"` with CuTeDSL 4.8 or newer, and `nvdisasm` on `PATH`
for the SASS check. `list` needs only `typer`; it does not import torch or CuTeDSL.

## Running

Run from the repository root. Without `--arch`, `run` and `sass` use the current GPU's
architecture; `run --arch` must match the GPU.

```shell
# Inspect the table (no GPU needed). --json prints the static description of each variant.
python benchmarks/tcgen05_throughput.py list --arch sm_103 --family mma

# Full sweep on the current GPU (sm_100: ~350 variants, sm_103: ~380, sm_107: ~450).
python benchmarks/tcgen05_throughput.py run --output agent_space/tc_$(hostname).jsonl

# A focused sweep with hot-loop SASS checks.
python benchmarks/tcgen05_throughput.py run --family mma --select 'e4m3_f32' \
  --sass-dir agent_space/sass --output agent_space/fp8.jsonl

# Cross-compile and check SASS for another architecture without running it.
CUTE_DSL_ARCH=sm_107a python benchmarks/tcgen05_throughput.py sass --select 'nvf4_k128'

# Print rows from one or more result files.
python benchmarks/tcgen05_throughput.py summarize agent_space/*.jsonl
```

Selectors combine: `--arch`, `--family mma|tmem`, `--mode throughput|latency` (both
repeatable), `--select REGEX`, and `--exclude REGEX` on variant names.

Rows stream to the JSONL file as each variant finishes. Rerunning the same command resumes: any
variant that already has a row (of any status) is skipped, and the variant that was in flight when
a previous process died is recorded as `crashed`. `--no-resume` overwrites the file. A watchdog
exits with status 124 when one variant runs longer than `--variant-timeout` seconds, so a driver
loop can simply rerun. Use a separate `--output` per trial; `--seed` only shuffles the order.

Close other GPU work first: per-clock rates are robust to DVFS, but absolute rates are not.

## Variant names

`mma.<kind>.cg<1|2>.<ss|ts>[.sp].m<M>n<N>k<K>[.areuse|.breuse][.latency]`

- `kind` is a row of `MMA_KINDS` (for example `bf16_f32`, `e4m3_f32_k64`, `mxf8_e4m3`,
  `nvf4_k96`). Suffixes `_k64`, `_k96`, and `_k128` mark the sm_107 and sm_103 wide-K forms.
- `ss` reads A and B from SMEM; `ts` reads A from TMEM. `.sp` is 2:4 structured-sparse A, whose
  instruction K is twice the dense K.
- `breuse` and `areuse` are sm_107 operand-collector pairs: each K step issues a collector-fill
  instruction and a lastuse instruction that keep B (or A) and write two accumulators. Both
  instructions read the same SMEM addresses.

`tcgen05.<op>[.<shape>][.x<num>.k<inflight>][.cols<n>].w<warps>[.latency]` names the non-MMA
forms; `k<inflight>` is the number of `tcgen05.ld/st` issued before one wait.

## What each sweep measures

- **MMA throughput.** Each CTA repeats a fixed envelope of 128 accumulator rows x 256 TMEM columns
  x 256 SMEM bytes of K (384 B for K=96), tiled by one physical instruction shape. `full`-tier
  kinds sweep M 64/128 (cta_group 1) or 128/256 (cta_group 2) by every legal N; other kinds
  measure three representative shapes. Equal work per repeat lets
  `cycles_per_repeat` compare instruction granularity directly. Collector-reuse variants instead
  cover two accumulator panels of the instruction shape.
- **MMA latency.** One CTA issues one instruction, commits, and waits on the mbarrier per repeat.
- **TMEM throughput.** `tcgen05.ld/st` per shape and repetition count at 4 warps, at the most warps
  that fit the register budget, and with up to 4 loads in flight per wait; `tcgen05.cp` per shape;
  `tcgen05.shift`; and `tcgen05.alloc`/`dealloc` round trips.
- **TMEM latency.** One CTA issues one instruction plus its completion wait per repeat
  (`tcgen05.ld/st` from a single warp); also `commit` and both `tcgen05.fence` forms.

Operands are zero-filled once before the timed region, so absolute rates are an optimistic
low-switching roofline. Variants that CuTeDSL or ptxas rejects are recorded as `unsupported`
with the error text; the table's legality filter is only a coarse prefilter.

## How a measurement works

For each variant, `measure.py` captures one CUDA graph per (grid multiplier, repeat count) pair,
replays them in interleaved, rotated rounds, and fits two lines against the repeat count:

- CUDA-event time per repeat, which gives absolute throughput (`work_per_s`);
- median in-kernel `clock64` cycles per repeat across CTAs, which gives per-clock rates
  (`work_per_clk_sm`, `cycles_per_repeat`).

Slopes cancel launch, prologue, and completion overhead. Throughput variants run at 1, 2, and 4
CTAs (or clusters) per SM; the one-CTA-per-SM grid provides the per-clock rates and the effective
SM clock (`effective_mhz = cycles slope / time slope`), and `best_grid_multiplier` reports which
grid had the highest absolute rate. An `nvidia-smi` child process samples clocks, power, and
temperature over the variant's window.

`--sass-dir` keeps cubins and checks the innermost loop that contains the target opcode family:
the loop must contain exactly the expected number of target instructions (for example
`UTC*MMA`, `LDTM`) and no foreign memory or barrier opcodes. A clean result proves the probe
measures only the intended instruction family; it does not decode kind or modifier bits.

## Result rows

Each line of the output file is one JSON object. Undefined values are `null`.

| Field | Meaning |
| --- | --- |
| `name`, `family`, `mode` | Variant identity |
| `status` | `ok`, `unsupported` (compile rejected), `failed` (runtime error), or `crashed` (killed the process) |
| `error` | Error text for non-`ok` rows |
| `context` | `date`, `commit`, `gpu`, `arch`, `num_sms`, `driver`, `cuda`, `torch`, `cutedsl`, `python` |
| `seed` | Shuffle seed of the run |
| `work_unit` | `flop`, `op` (integer MMA), `byte` (ld/st/cp), or `instr` |
| `work_per_cta_repeat` | Work in `work_unit` that one CTA performs per repeat (cta_group 2 splits each MMA across both CTAs) |
| MMA static fields | `kind_key`, `kind`, `a`, `b`, `acc`, `sf`, `sf_vec`, `scale_vec`, `op`, `m`, `n`, `k`, `sparse`, `collector`, `k_env`, `cta_group`, `a_src`, `instructions_per_repeat`, `flops_per_instruction` |
| TMEM static fields | `op`, `shape`, `num`, `warps`, `registers`, `unroll`, `bytes_per_instruction` |
| `compile_s`, `smem_bytes`, `tmem_columns` | Compile time and probe resources |
| `work_per_s` | Absolute rate at the best grid, in `work_unit` per second |
| `best_grid_multiplier` | CTAs per SM of the grid with the highest `work_per_s` |
| `one_wave_ctas`, `one_wave_distinct_sms` | CTAs launched on the one-CTA-per-SM grid and the distinct SMs they ran on |
| `work_per_clk_sm` | Per-clock rate per CTA on the one-CTA-per-SM grid; valid as per-SM when `one_wave_distinct_sms == one_wave_ctas` |
| `cycles_per_repeat`, `us_per_repeat` | One-wave fitted slopes |
| `latency_cycles`, `latency_ns` | Latency rows only: slope per issued instruction |
| `peak_per_clk_sm`, `efficiency_vs_peak` | Throughput MMA rows with a public sm_100/sm_103 peak (sparse rows use the 2x dense-equivalent peak) |
| `effective_mhz` | SM clock during the one-wave grid, `null` if CTAs shared an SM |
| `min_r_squared` | Worst fit quality across grids; values well below 1 mean a noisy row |
| `sass` | `--sass-dir` only: `found`, `clean`, `target_opcodes`, `expected_targets`, `foreign_opcodes`, `loop_opcodes`, `cubin` |
| `measurement` | Raw per-grid fits, medians per repeat count, and the NVML window |

## Reading results

Compare instruction forms with per-clock rates (`work_per_clk_sm` or `cycles_per_repeat`); they
are independent of the clock the GPU happened to run at. Treat `work_per_s` as an upper bound on
what a real kernel can reach. Check `min_r_squared` and `sass.clean` before trusting a surprising
row, and rerun with a different `--seed` into a new file to estimate run-to-run spread.

### SMEM operand bandwidth model

Most cta_group 1 SS rows that fall short of the datapath peak are limited by SMEM operand reads.
A simple model predicts their per-clock rate:

```
smem_bytes = A bytes (0 for TS, halved for sparse A) + B bytes      per instruction per CTA
           = (M / cta_group) * K * a_bits / 8 + (N / cta_group) * K * b_bits / 8
cycles     = max(flops_per_instruction / cta_group / datapath_peak, smem_bytes / smem_bw)
rate       = flops_per_instruction / cta_group / cycles
```

`a_bits`/`b_bits` are the SMEM bits per element (8 for FP4/FP6 in `kind::f8f6f4` containers), and
`datapath_peak` is the best per-clock rate any form of the instruction reaches. Collector-reuse
pairs read the kept operand once per pair. With `smem_bw` of about 128 B/clk/SM, as measured with
this harness on B200 (sm_100) and B300 (sm_103), the model explains why small N (more A bytes per
FLOP), SS versus TS, and cta_group 2 (each CTA reads half of B) change the rate at a fixed
instruction K. M=64 rows use half the datapath and fall outside the model.

For example, `mma.bf16_f32.cg1.ss.m128n64k16` reads 4096 B of A plus 2048 B of B per
instruction: 48 clk of SMEM reads against 32 clk of datapath at 8192 FLOP/clk, so the model
predicts 5461 FLOP/clk/SM, which is what B200 and B300 measure. The TS form drops the A bytes and
reaches the 8192 peak.
