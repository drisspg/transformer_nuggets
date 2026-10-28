---
name: analyzing-pytorch-rooflines
description: Runs and extends transformer_nuggets eager/AOT logical roofline analysis. Use when annotating PyTorch profiler traces, attributing GPU time to FX nodes or kernels, finding fusion opportunities, ranking performance work, or adding parent/inner-kernel FLOP and byte formulas.
---

# Analyzing PyTorch Rooflines

The implementation is split by ownership:

- `transformer_nuggets/roofline.py`: formula registry, raw Kineto decoration, CLI.
- `transformer_nuggets/fx_analysis.py`: eager FX/AOT capture, fusion regions, ranking, replay bundles.

All formulas describe **logical useful work and minimal traffic**. Physical DRAM/L2 traffic and executed instructions require NCU/CUPTI.

## Preflight

1. Resolve the Python environment with the `env` skill.
2. Verify the current checkout is imported:

```text
PYTHONPATH=$PWD <python> -c 'import transformer_nuggets; print(transformer_nuggets.__file__)'
```

3. Eager/AOT workflows require static shapes and one CUDA device.
4. Raw traces require Kineto JSON/JSON.GZ with `record_shapes=True` and `External id` fields.
5. Put generated traces and replay artifacts under `artifacts/` or `agent_space/`.

## Workflows

### Eager FX

```python
from transformer_nuggets.fx_analysis import profile_fx_fusion

result = profile_fx_fusion(fn, (x, scale), "artifacts/run.pftrace")
```

Ceilings default to `roofline_spec="measured"`: `measure_roofline_spec()` calibrates
the profiled GPU once (about 1.5 s) and caches the result in
`~/.cache/transformer_nuggets/roofline_specs.json`. It times random-data bf16 GEMMs
and `copy_` at about 50% duty cycle, matching how eager steps leave the GPU partly
idle, and also reports the power-capped sustained GEMM rate in the spec name.
Datasheet peaks overstate attainable throughput when the GPU cannot hold its
maximum clock: on GB200, back-to-back random GEMMs fall from about 1.7 to 1.3
PFLOP/s under the power cap. Pass an explicit `RooflineSpec` to override, or
`None` to skip roofline metrics. Re-measure with `use_cache=False` after changing
clocks or power limits.

Use `cuda_graph=True` only when CUDA Graph replay is the intended workload contract.

### Forward/backward training

```python
from transformer_nuggets.fx_analysis import profile_aot_training

result = profile_aot_training(
    model,
    (input_ids,),
    "artifacts/train.pftrace",
    kwargs={"labels": labels},
    loss_selector=lambda output: output.loss,
)
```

AOTAutograd is used for graph capture only; execution remains eager. The module must produce one scalar loss. Optimizer execution is not included.

### Existing raw trace

```text
annotate-roofline trace.json.gz -o trace.roofline.pftrace \
  --formula-module my_project.roofline_formulas \
  --peak-compute-tflops ... \
  --peak-memory-gbps ... \
  --launch-latency-us ...
```

Custom formula modules must be imported before decoration. Unregistered ops retain duration and known input-read metadata with `confidence=unknown`.

## Reading Perfetto

- Original CUDA stream: untouched kernels.
- `Roofline <stream>`: one color-coded box per attributable kernel.
  - 🔴 high-confidence, material, below threshold.
  - 🟡 medium finding.
  - 🟢 healthy modeled kernel.
  - 🔵 low-confidence model; needs NCU/model work.
  - ⚪ unknown work; not an optimization finding.
- `GPU annotations <stream>`: FX node and fusion-region annotations.
  - Region and node findings carry `why` and `suggested_fix` arguments.
  - Split regions are labeled `part i/n` and chained with flow arrows.

Severity uses both an emoji in the name and a fill color. Find offenders with SQL,
for example `select * from slice where name like '🔴%'`. Perfetto colors a slice by
hashing its correlation ID, so findings set `slice_color_key` to a string that
hashes to red, amber, or green (`SEVERITY_COLOR_KEYS` in `roofline.py`).

Click a box to inspect formula confidence, parent op, work scope, FLOPs, read/write bytes, TFLOP/s, TB/s, floors, and bound classification.

## Interpretation Order

1. Rank by absolute time and step impact.
2. Check `model_confidence` and `trace_roofline_work_scope`.
3. Prefer high/medium-confidence red/yellow findings.
4. Treat blue entries as NCU/formula targets, not optimization proof.
5. Ignore low-duration kernels even when their percentage is poor.
6. Validate logical conclusions with physical counters before claiming speedup.

## Formula Registration

Parent and inner-kernel formulas share:

```python
def formula(inputs, outputs, kwargs) -> RooflineWork: ...
```

See [formulas.md](formulas.md) for registration, replacement, trace compatibility, and coverage rules.

## Follow-ups

FX/AOT profiles write:

```text
<trace>.findings.json
<trace>.followups/NN_<region>/diagram.md
<trace>.followups/NN_<region>/manifest.replay.json
<trace>.followups/NN_<region>/region.pt
<trace>.followups/NN_<region>/replay.py
<trace>.followups/NN_<region>/ncu.sh
```

Run `replay.py` successfully before using the NCU template.

## Validation Gates

After formula or trace-decoration changes:

```text
PYTHONPATH=$PWD <pytest> -q test/test_roofline.py test/test_fx_analysis.py test/test_perfetto.py
```

GPU changes additionally run through the exclusive lock:

```text
gpu-run auto -- env PYTHONPATH=$PWD <pytest> -q \
  test/test_roofline.py test/test_fx_analysis.py test/test_perfetto.py test/test_profiler.py
```

New formulas require a synthetic `cpu_op` + kernel trace test. Multi-kernel formulas must report coverage or deliberately remain unknown.

## Hard Stops

Stop and report rather than inventing metrics when:

- Shapes are dynamic/data-dependent or cannot be converted to static integers.
- Multiple CUDA devices participate or another thread is concurrently profiling/capturing RNG state.
- `External id` attribution is absent or ambiguous.
- Output bytes cannot be inferred from trace metadata; use `None`, never zero.
- A formula requires measured counters or implementation-specific cache behavior.
- An inner KDA/custom-kernel allocation is not analytically derived; label it low confidence/unknown and profile it with NCU.
- A trace was captured without shapes; recapture rather than fabricating tensor sizes.
