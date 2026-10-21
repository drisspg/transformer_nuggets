# FlexGEMM benchmark harness

Answers "where does a fused FlexGEMM/QUACK epilogue win or lose against the stock `torch.compile`
path, and against the roofline of the bytes it saves?" Every run records the torch commit, GPU,
measured copy bandwidth and pinned `GemmConfig`s so results are comparable across runs and
checkouts. Every ratio this tool prints is a **time ratio: lower is better**.

```
bench_flex_gemm.py   CLI: families x shapes x modes -> results.csv, summary.md, heatmaps, codes/
config_sweep.py      one (shape, family): time every SM100 priority GemmConfig pinned via kernel_options
report.py            summarize a results.csv (summary.md + heatmaps) or --diff two of them
epilogues.py         epilogue catalog (fused call, stock spelling, bare GEMM, fp64 reference)
harness.py           compile, route check, correctness, interleaved CUDA-graph timing, clock sampling
shapes.py            named shape sets and --shape / --moe parsing
```

## Running (from the repo root)

GPU work goes through `gpu-run`; results default to `benchmarks/data/flex_gemm/<timestamp>[_<label>]_<commit>/`
(gitignored) or `--out DIR`.

```zsh
# smoke: 2 shapes x 3 families x {default, autotune}, 3 rounds x 30 replays (~3 min on GB200)
gpu-run auto -- python benchmarks/flex_gemm/bench_flex_gemm.py --fast

# dense pointwise sweep on the frozen grid, one process per shape
gpu-run auto -- python benchmarks/flex_gemm/bench_flex_gemm.py --families relu,silu_fast --shapes grid --isolate

# MoE SwiGLU vs stock and the two-GEMM spelling
gpu-run auto -- python benchmarks/flex_gemm/bench_flex_gemm.py --families grouped_swiglu_interleaved --shapes moe_16b --baselines cublas,stock,two_gemm

# single cells; --moe is E,K,N,M[,balanced|skewed|hot]
gpu-run auto -- python benchmarks/flex_gemm/bench_flex_gemm.py --shape 1024,4096,1024 --moe 8,1024,2048,4096,skewed --families identity,amax_g32 --modes default

# which pinned GemmConfig is best for one cell?
gpu-run auto -- python benchmarks/flex_gemm/config_sweep.py --shape 16384,8192,4096 --family relu

# A/B two checkouts: run each with its own --label/--out, then diff (CPU only)
gpu-run auto -- python benchmarks/flex_gemm/bench_flex_gemm.py --shapes grid --families relu --label before --out benchmarks/data/flex_gemm/before
gpu-run auto -- python benchmarks/flex_gemm/bench_flex_gemm.py --shapes grid --families relu --label after --out benchmarks/data/flex_gemm/after
python benchmarks/flex_gemm/report.py --diff benchmarks/data/flex_gemm/before/results.csv benchmarks/data/flex_gemm/after/results.csv

# regenerate summary.md and heatmaps from an existing CSV
python benchmarks/flex_gemm/report.py summarize benchmarks/data/flex_gemm/after/results.csv
```

Modes (`--modes`, default `default,autotune`): `default` = `kernel_options={"backend": "QUACK", "autotune": False}`
(Inductor's heuristic `GemmConfig`), `autotune` = `autotune=True` (Inductor benchmarks the measured SM100
priority subset), `exhaustive` (`--exhaustive`) = `autotune=True` under
`inductor_config.max_autotune_gemm_search_space="EXHAUSTIVE"`. Families marked `fast_math` (`silu_fast`,
`grouped_situ_glu_interleaved_fast`) pass `fast_math=True` in every mode. Only the QUACK backend is
benchmarked; TRITON/NVGEMM are not covered.

## What is measured

| column | what | how |
|---|---|---|
| `t_cublas` | bare GEMM floor | eager `torch.mm` / `F.grouped_mm` / `F.scaled_mm` |
| `t_stock` | stock path | `torch.compile(fullgraph=True)` of the same math without `flex_gemm` |
| `t_flex` | fused path | `torch.compile` of `flex_gemm(gemm_op, (a, b), epilogue, kernel_options=...)` |
| `t_flex_identity` | bare QuACK GEMM | `flex_gemm` with `lambda acc: acc` at the same physical width |
| `t_two_gemm` | SwiGLU only | `silu(a @ w1) * (a @ w3)` (`--baselines ...,two_gemm`) |

Roofline: `T_stock ~= T_gemm + T_epilogue_kernels`, and fusion can at best remove the intermediate write + read:

```
saved_bytes        = 2 * M * N_phys * sizeof(dtype)      (0 for identity and residual: stock lowers to addmm)
saved_time_at_bw   = saved_bytes / hbm_bw_gbs             (measured D2D copy bandwidth, read + write)
flex_over_stock    = t_flex / t_stock                     practically achievable bound
flex_over_roofline = t_flex / (t_stock - saved_time_at_bw) bandwidth-optimal-stock bound (hard to beat)
gemm_gap           = t_flex_identity / t_cublas           GEMM-engine gap independent of the epilogue
```

`N` is the family's *logical* output width; SwiGLU families run a `[M, 2N]` GEMM (`n_phys` column).

## Measurement contract

- **Timing**: `transformer_nuggets.utils.benchmark.benchmark_cuda_function_in_microseconds(..., USE_CUDA_GRAPHS=True)`:
  one CUDA graph per variant, fixed pointers, warm L2, per-replay CUDA-event latency. All variants of a shape are
  timed **interleaved**: `--settle` seconds of replay first (GB200 clocks drop under load), then `--rounds` rounds in
  a deterministic shuffled order reversed every other round, each round = `--replays` replays; reported value is the
  median of round medians, `*_spread = (max - min) / median`. `nvidia-smi` samples SM clock / power / temperature
  during each round (`clocks` column).
- **Correctness before timing**: every compiled output is compared with an fp64 reference. A fused variant is timed
  only if it is finite and its normalized RMS error is `<= max(2.5 x stock nrmse, 1e-2)`; otherwise
  `status=correctness_failed`.
- **Route check**: the generated code must contain `flex_gemm_epilogue(` and no extern GEMM, otherwise
  `status=fallback`. The pinned config is parsed from the generated code (`flex_config`, label
  `tile_m x tile_n _c cluster_m x cluster_n _clc|static[_pp][_swap]`); generated code is saved under `<out>/codes/`.
  Families the current checkout cannot lower are recorded as `status=compile_error` with the message, families
  that cannot run on the device/shape as `status=unavailable`; the run continues.
- **Data**: `a[M, K]` uniform in `[-1, 1]`, weights uniform scaled by `sqrt(3 / K)`, seeded per shape. Dense
  weights are `[N_phys, K]` passed as `w.mT`; grouped weights `[E, N_phys, K]`. SwiGLU `interleaved` interleaves
  gate/up rows, `chunked` stacks them.
- **Isolation**: `--isolate` runs one subprocess per shape; otherwise `torch._dynamo.reset()` per shape with
  raised recompile limits.

## Families (`epilogues.py`)

| family | epilogue | notes |
|---|---|---|
| `identity` | `acc` | row ratio == GEMM gap; works on MoE shapes too |
| `relu`, `silu`, `silu_fast`, `bias_gelu`, `residual` | pointwise in fp32 | `silu_fast` sets `fast_math=True` |
| `amax_g32`, `amax_g128`, `sum_g32` | `acc.relu()` + grouped reduction aux output | `N % g == 0` |
| `lse_g256`, `lse_g256_gather` | grouped `logsumexp` (+ `gather(1, targets)`) finalized outside | `N % 256 == 0`; both currently `compile_error`: grouped logsumexp needs group <= 32, and indexed gather is unsupported |
| `mxfp8_producer` | E4M3 data + `to_blocked` E8M0 scales | compared after dequantization |
| `swiglu_interleaved`, `swiglu_chunked`, `situ_glu_interleaved` | grouped-main contraction `[M, 2N] -> [M, N]` | `n_phys = 2N`; `two_gemm` baseline |
| `grouped_*` | same epilogues over `F.grouped_mm(..., offs=offs)` | MoE shapes only |
| `scaled_mm_mxfp8` | `acc.relu()` over `F.scaled_mm` with `BlockWise1x32` scales | SM100+, `N, K % 128 == 0`; no `gemm_gap` |

## Shape sets (`shapes.py`)

`grid` (`M in {256, 1024, 4096, 16384} x N in {1024, 4096, 8192} x K in {1024, 4096, 8192}`), `skinny`,
`dsv3_16b` / `dsv3_671b` (DeepSeek-V3 dense layers at 2048 and 8192 tokens), `moe_16b` / `moe_671b` (expert
GEMMs, balanced and skewed `offs`). `--shape M,N,K` and `--moe E,K,N,M[,kind]` add single cells (repeatable).

Outputs: `results.csv` (one row per shape x family x mode; `status` in
`ok | fallback | correctness_failed | compile_error | unavailable`), `summary.md`, `heat_<metric>_<family>_<mode>.png`
for dense cells, `codes/`. Before quoting numbers, state the GPU and `clocks`, the `commit`, and that values are
per-replay CUDA-graph latencies of warm fixed buffers.
