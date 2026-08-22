# Trace and Findings Schema

## Native Perfetto tracks

- `Roofline <stream>`: one `gpu_roofline_annotation` box per attributable kernel.
- `GPU annotations <stream>`: eager FX nodes and fusion regions.
- Original CUDA streams are unchanged.

Roofline boxes carry `severity`, `is_optimization_finding`, `needs_measurement`, and the original kernel name. Native TrackEvent output relies on colored Unicode icons and severity metadata; exact forced fill colors require a Perfetto plugin.

## Kernel work scope

- `kernel_specific`: an inner-kernel formula owns the FLOPs/bytes.
- `single_kernel_parent_op`: one-kernel parent formula is equivalent to kernel work.
- `parent_op_shared`: multi-kernel parent totals are context only. Kernel-specific FLOPs/bytes remain absent.

## Operation fields

CPU operation annotations include:

```text
logical_flops
logical_read_bytes
logical_write_bytes
logical_bytes
model_kind
model_confidence
observed_op_wall_us
observed_op_gpu_busy_us
observed_op_gpu_time_us
observed_op_gap_us
inner_kernel_*_coverage_ratio
formula_error
```

## Stage fields

`user_annotation` ranges receive inclusive stage summaries:

```text
stage_attribution_scope
stage_aggregation
stage_operation_count
stage_kernel_count
stage_gpu_launch_count
stage_gpu_wall_us
stage_gpu_busy_us        # interval union
stage_gpu_time_us        # sum of kernel durations
stage_known_logical_flops/bytes
stage_low_confidence_operation_count
stage_unknown_*_operation_count
stage_achieved_known_tflops/tbps
```

Backward ranges use cross-thread fallback only when same-thread containment finds no operations.

## Findings JSON

FX/AOT workflows create `<trace>.findings.json`:

- `regions`: ranked by `priority_recoverable_us`, then observed wall time.
- `priority_basis=supplied_roofline_fused_floor`: explicit ceilings were supplied.
- `priority_basis=traffic_reduction_proxy`: heuristic only.
- `nodes`: ranked by observed node wall time.

## Replay bundle

Each ranked region gets a directory with:

```text
diagram.md
manifest.replay.json
region.pt
replay.py
ncu.sh
```

Replay manifests preserve shape, dtype, stride, storage offset, and alias/storage groups for tensor inputs. Random replay data is diagnostic; it does not reproduce application values or cache state.
