# FlexGEMM performance

Use [examples/flex_gemm/](../../../examples/flex_gemm/README.md) for runnable patterns and [benchmarks/flex_gemm/](../../../benchmarks/flex_gemm/README.md) for sweeps. Paths below beginning with `torch/` are relative to the active PyTorch checkout.

## Tune, but keep an untuned control

```python
kernel_options={"backend": "QUACK", "autotune": True}
```

`autotune=False` is the implementation default. Dense calls use the measured shape table when legal; variable-length-M calls use the grouped priority order; otherwise selection uses QuACK's preferred legal config. Block-scaled calls retain QuACK's shape-aware default. `config={...}` filters the legal set before either selection mode and can pin a complete config. The old `tuned` option is rejected.

With `autotune=True`, Inductor benchmarks legal choices through `autotune_select_algorithm` and caches the winner; QuACK's own tuner never runs. The current policy in `torch/_inductor/heuristics/template/flex_gemm.py` is not a universal 12-choice cap:

- Dense SM100 has 12 measured priority entries, plus QuACK's preferred config if not already included.
- Skinny dense `mm` (M <= 128) can add swapped small-N and M64 configs.
- Variable-length-M calls have a separate nine-entry priority list and do not append the dense default.
- Legal sets with no ranked entry retain QuACK's order capped at 12.
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE` keeps every legal config; the count depends on device, dtype/layout, epilogue, and constraints, not a fixed 200.

The earlier GB200 examples sweep saw roughly 12 choices per tuned call. That shortlist can miss a good tile, and timing noise can change the winner. Large preactivation and dense SwiGLU examples favored the untuned pick. Compare True/False on the same workload rather than treating autotuning as a guaranteed improvement.

Compilation/tuning is not steady-state execution. Kernel compiles use Inductor compile workers when available; `TORCHINDUCTOR_COMPILE_THREADS=1` uses in-process compilation. For cold-compile measurements, use a fresh, owned `TORCHINDUCTOR_CACHE_DIR`, including its `quack/` subtree; do not delete shared caches. Separate compile time from benchmarking time before enlarging the search space.

## Size the opportunity

Fusion removes intermediates, not GEMM arithmetic. For a bf16 `[M,K] @ [K,N]` with a shape-preserving pointwise epilogue, eliminating the intermediate write+read saves **4MN bytes** against **2MNK FLOPs**, or **2/K saved bytes per FLOP**. Bias, scale, residual reads, and final output writes remain.

- If the GEMM result must also be returned (preactivation or bf16 plus fp8 copy), its store remains: only **2MN bytes** of reread are removable, or **1/K bytes/FLOP**.
- SwiGLU's physical `[M,2N]` accumulator saves **8MN bytes** against **4MNK FLOPs**, again **2/K bytes/FLOP**; N here is the logical output width.
- If Inductor already uses `addmm` for a residual/bias, there may be no removable intermediate at all.

These are algorithmic traffic estimates, not measured DRAM counters. At fixed GEMM FLOPs, reducing K and increasing M*N can enlarge the opportunity. Shrinking M alone can instead make weight traffic and tile efficiency dominate. A 4096-cubed GEMM is compute-heavy: removing a few microseconds of epilogue cannot yield a large whole-call speedup. Expensive quantization or backward math can also make the fused epilogue slower than a separate kernel.

Measured GB200 examples illustrate the regimes (speedup = faster compiled baseline time / FlexGEMM time; >1 is faster):

| Pattern | Shape (M,N,K) | Measured lesson |
| --- | --- | --- |
| Bias + GELU | (8192,4096,2048) | About 1.10x with original GELU, 1.29x with matched tanh math and autotune. |
| Residual add | (8192,4096,2048) | About 1.00x: the baseline already used addmm. |
| Group RMS norm, G=32 | (16384,4096,1024) | About 1.33x versus the faster baseline; 4096-cubed was only about 1.03x. |
| Output + group amax | (16384,4096,1024) | About 1.12x: the required main output store remains. |
| MXFP8 producer | (4096,4096,1024) | About 0.82x: quantization overhead outweighed saved traffic. At (16384,8192,2048), about 1.17x. |

LM-head cross-entropy statistics at K=4096 remained GEMM-heavy (roughly 1.05-1.14x in the sweep); avoiding materialized fp32 logits can matter more than latency. Custom SwiGLU MLP forward+backward at representative D=2048-4096 likewise showed only about 1.03-1.10x, not the speedup of an isolated fused stage. These are shape/hardware/math/timing-contract observations, not selector rules or guarantees.

These GB200 runs used PyTorch `98156a61b4ff`. The 01-03 report used matched approximate baselines for its fast-math recommendations; the 04-06 fast-math backward timings were diagnostic comparisons and are not a matched-approximation speedup claim.

## Fast math and fair baselines

`fast_math=True` affects supported CuTeDSL arithmetic and QUACK body tracing. Sigmoid becomes `torch.tanh(x * 0.5) * 0.5 + 0.5`; SiLU becomes `half = x * 0.5; half * torch.tanh(half) + half`; exact GELU selects the standard tanh approximation. It is not merely a compiler flag. Eager FlexGEMM does not perform those tracing substitutions.

- Compare against baseline GELU with `approximate="tanh"`, or the exact same tanh-form SiLU/sigmoid expressions. Keep an original-math, `fast_math=False` control.
- Fusion can change intermediate bf16 rounding even without fast math. Check every output against low-precision eager and use FP64 as the error measuring stick; do not claim bitwise identity from toleranced checks.
- Validate manually authored backward formulas separately. A forward decomposition change does not rewrite the backward.
- Compare default `torch.compile` and `mode="max-autotune-no-cudagraphs"`; report both or use the faster as denominator. Max-autotune is not always the faster baseline.
- Keep a same-config ablation: the exact dequant+SiLU example changed from 136.9 us to 108.0 us with fast math at the same untuned tile, showing epilogue math cost, not just removed memory traffic. The backward sweep likewise found a large nonlinear-epilogue cost; it does not establish that every exp/log reduction benefits from fast math.

## Timing contract: clocks, cache, and ordering

Use an exclusive GPU reservation and a unique cache directory. Compile and validate all variants before timing. Preserve the benchmark contract across candidates: tensors, dtype/layout, math, compiler mode, cache behavior, and full callable boundary (including metadata/finalize kernels).

The examples' default benchmark helper uses a profiler-backed sum of workload GPU-event durations and clears a scratch cache surface before measured invocations. `NUM_ITERS` controls a target duration, not an exact invocation count. This excludes host/compile time and is **not** warm CUDA-graph replay or a verified rotating-buffer cold-DRAM contract. If the application uses graph replay, measure that separately and label it; do not reuse these speedups across contracts.

On the measured GB200, sustained GEMM dropped SM clocks from about 2062 to 1200 MHz within roughly 0.5 s under the software power cap. Timing the first variant at boost and the next after throttling produced swings up to 2x. Therefore:

1. Record hardware, power/clock policy, observed clocks, shape, seed/input distribution, dtype/layout, source revision, and timing scope.
2. Warm **all** variants until clocks/power settle; the reduction/backward investigation used about three seconds of mixed burn-in, not just a fixed small call count.
3. Rotate/reverse variant order over at least three rounds and retain samples/dispersion; use the median. Cool between shapes and restabilize before measuring.
4. Treat low-single-digit changes as ties unless measured dispersion supports them. The settled investigation still saw about 2% spread around 100 us and 5-10% on millisecond CE/MLP workloads.

The 01-03 report used three interleaved profiler rounds but factory clocks without a fixed-frequency policy; its numbers are representative, not frequency-normalized rooflines. Never attribute an autotuner's slow winner solely to clock throttling without matched evidence: shortlist restrictions and differing cache/timing contracts also matter.

## Diagnose a fused slowdown

- Freeze the generated wrapper/config before profiling. Separate same-config causal ablations from public-selector confirmation, and retain both results.
- Count intermediates in executed stock kernels, not Python source. Compare DRAM and L2 requests: narrower tiles can increase operand rereads despite removing an output store.
- Cross epilogue math and tile shape. If wider tiles reduce traffic/instructions but slow down, inspect static code size and instruction-supply stalls as well as spills/residency; nonlinear unrolling can starve tensor-core progress without changing occupancy.
- For a captured gate, compare identity and gated epilogues at the same config. Inspect packed selects and vector-load waits rather than attributing cost to a capture barrier from source alone.
- Check producer changes across the HOP boundary. A context addmm may retain bias that stock sinks into a later pointwise kernel. If moving bias into the capture, preserve addition order and validate the whole region.

## Selection-path contract (when changing lowering, template, or heuristics)

- QuACK's `legal_mod_configs` checks the exact EpiMod/`ModProblem`; lowering also filters TF32 operand-layout restrictions. Inductor owns ranking, defaults, constraints, and candidate policy. Verify every supported capability and constrained/block-scaled path rather than extrapolating the dense SM100 table.
- Shape-dependent winner guards live in `FlexGemmEpilogueCaller.output_node()`: `swap_ab` requires N divisible by 8; grouped-main contraction requires `tile_n <= N`. Check both direct and multi-template winners. Do not guard a rejected candidate's restrictions or revalidate the pinned config on every runtime launch. Boundary test: change a dynamic dimension within the winner's legal region without recompilation, then cross its restriction and require recompilation.
- The generated body gets a stable digest-derived `__module__`, so tuning and final-wrapper modules share QuACK epilogue identity. `test_tuned_winner_is_not_recompiled_by_final_wrapper` checks this identity. It does not prove zero cold compiles: the full key also includes dtype, tile, and layout. Count cold misses in QuACK's `cache/jit.py`, not calls that can hit cache or only in-process `cute.compile` calls that miss worker activity.
- In-process precompile takes the shared `torch._inductor.codegen.cutedsl.compile_lock.CUTEDSL_COMPILE_LOCK`; NVGEMM uses it too. CuTeDSL compilation is not thread-safe. Use process isolation or the shared lock, not a backend-local lock. Compare scheduling with the existing compile pool and `MultiTemplateBuffer` before creating another path.
