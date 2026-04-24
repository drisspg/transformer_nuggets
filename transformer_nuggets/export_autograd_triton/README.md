# export_autograd_triton

`transformer_nuggets.export_autograd_triton` exports a Python function into a small importable wrapper plus Inductor/Triton artifact files. It is intended for iterating on generated Triton kernels while keeping eager-style autograd behavior at the Python API boundary.

## Quick usage

```python
from pathlib import Path

import torch

from transformer_nuggets.export_autograd_triton import (
    Specialization,
    export_autograd_triton,
    load_exported_module,
)


def rms_norm(x, weight, *, eps=1e-5):
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps)).to(x.dtype) * weight


x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)

export_autograd_triton(
    rms_norm,
    specializations=[
        Specialization(
            args=(x, weight),
            kwargs={"eps": 1e-5},
            dynamic_shapes={"x": {0: torch.export.Dim("tokens", min=1, max=4096)}},
        )
    ],
    out=Path("agent_space/generated_rms_norm.py"),
    source_backend="clean_triton",
)

generated = load_exported_module("agent_space/generated_rms_norm.py")
y = generated.rms_norm_compiled(x, weight, eps=1e-5)
```

`Specialization.name` is optional. Unnamed specializations are named `spec_0`, `spec_1`, etc. Artifact filenames for unnamed specs are compact (`spec_0_forward.py`, `spec_0_backward.py`).

## Generated layout

For `out=agent_space/generated_rms_norm.py`, export writes:

```text
agent_space/generated_rms_norm.py
agent_space/generated_rms_norm_artifacts/
  spec_0_forward.py
  spec_0_backward.py
```

The top-level generated wrapper is intentionally small and labeled, but keeps the autograd node
visible. It contains:

- a generated artifact index with direct relative paths to each forward/backward artifact
- one `_SPEC_<n>` block per specialization, with metadata and tensor guards
- `_SPECS`: the ordered specialization list used for dispatch
- `_RUNTIME = ExportedAutogradRuntime(...)`
- one explicit `torch.autograd.Function` subclass whose `forward` and `backward` call the
  generated artifact runners through the runtime helper
- one documented public function such as `rms_norm_compiled(...)` that binds the public
  signature and delegates dispatch/execution to the shared runtime

Artifact files start with a short generated-artifact docstring that names the specialization,
forward/backward direction, runtime tensor order, static arguments, tensor guards, output
metadata, and saved-residual order. The artifact directory is recreated on each export so stale
kernels from earlier runs do not hang around.

Shared runtime behavior lives in `runtime.py`, not in every generated file:

- artifact module loading
- specialization dispatch
- static/tensor guard checking
- autograd forward/backward helper plumbing
- saved residual reordering for backward graphs
- non-differentiable output marking
- no-match diagnostics

The generated source template is `templates/autograd_module.py.j2`.

## Source backends

### `source_backend="inductor"`

Writes raw Inductor source. This keeps Inductor runtime launch wrappers such as `triton_heuristics`, which preserve symbolic dynamic launch logic.

Use this when:

- you want the safest dynamic-shape behavior
- a dynamic clean-Triton kernel family is unsupported
- you need to inspect raw Inductor launch expressions

### `source_backend="clean_triton"`

Runs `torch.utils._get_clean_triton.get_clean_triton(...)` and then applies the dynamic-clean validation/patching layer in `clean_triton.py`.

The goal is readable inline `@triton.jit` artifacts without `async_compile.triton(...)` while
preserving dynamic launch arguments. The cleaner also removes unused imports/top-level aliases,
standalone benchmark scaffolding (`get_args`, `benchmark_compiled_module`,
`if __name__ == "__main__"`), and formats patched dynamic launches as multiline Triton calls.

Supported dynamic clean families today:

- pointwise / fused pointwise (`triton_poi_*`)
- simple reductions (`triton_red_*`) that follow the observed raw `.run(...)` argument pattern
- narrow matmul/template launches (`triton_tem_*`) where raw `.run(...)` carries `M/N/K` and the three grid expressions

Unsupported dynamic clean launches fail loudly instead of emitting a shape-baked artifact.

Error messages include the artifact path, kernel name, likely family, launch line, and recommendation to use `source_backend="inductor"` or add a cleaner.

## Dynamic-shape support

Current supported dynamic shape scope:

- tensor dimensions are guarded by `Specialization.dynamic_shapes`
- dim `0` dynamic is supported
- `torch.export.Dim(...)` and string symbols are accepted
- `dynamic_shapes` keys must name Tensor arguments exactly
- bounded and unbounded `torch.export.Dim` are represented in guards
- symbol equality across tensor guards is checked at runtime

Current limitations:

- dynamic dims other than dim `0` are rejected
- dynamic forward-only exports are rejected
- exact strides are still guarded; this works for the current dim-0 contiguous batch/token use case but is not full symbolic-stride support
- generated artifacts hardcode the CUDA device index from capture

## Autograd behavior

AOTAutograd emits forward outputs followed by saved residuals. Backward graph inputs may expect those residuals in a different order. Capture records:

- `forward_residual_names`
- `backward_saved_input_names`
- `differentiable_output_mask`

The runtime uses those to:

- save tensors via `ctx.save_for_backward`
- keep non-tensor residuals on `ctx`
- reorder saved residuals before calling the backward artifact
- filter `grad_outputs` for non-differentiable outputs
- call `ctx.mark_non_differentiable(...)` for non-differentiable tensor outputs

Tests cover dynamic view residual ordering and mixed differentiable/non-differentiable outputs.

## Tuning knobs

`export_autograd_triton(...)` supports:

```python
export_autograd_triton(..., max_autotune=True)
```

which threads `max_autotune=True` to Inductor config patches.

For experiments, pass explicit Inductor config patches:

```python
export_autograd_triton(
    ...,
    max_autotune=True,
    inductor_config_patches={
        "coordinate_descent_tuning": True,
        "coordinate_descent_check_all_directions": True,
    },
)
```

Coordinate descent tuning may or may not affect a given graph. For the current RMSNorm example, the generated forward kernel is a reduction-style kernel; max-autotune and coordinate descent produce similar timings on the tested setup.

## Examples

Examples live in `examples/export_autograd_triton/`:

- `basic.py`: minimal static affine/ReLU export
- `minimal.py`: static MLP-style export with artifact summary
- `dynamic.py`: dynamic token/batch export and gradient validation
- `rms_norm.py`: LLM-sized bf16 RMSNorm with dynamic tokens, max-autotune flow, coordinate descent flow, and logical memory-bandwidth reporting

Run the RMSNorm example:

```bash
python examples/export_autograd_triton/rms_norm.py
```

The RMSNorm bandwidth printed there is a logical forward-bandwidth estimate, not exact HBM traffic. It assumes:

- read `x` for variance
- read `x` again for normalization/output
- read `weight`
- write output

So the displayed bytes are currently:

```python
4 * x.numel() * x.element_size()
```

Compiler fusion, caching, and saved-forward intermediates mean real hardware traffic can differ. Treat the number as a comparable benchmark metric for this generated implementation, not an Nsight Compute measurement.

## Development commands

Targeted tests:

```bash
pytest test/test_export_autograd_triton.py -q
```

Formatting/lint/precommit:

```bash
prek run --all-files
```

Useful one-off checks:

```bash
python examples/export_autograd_triton/rms_norm.py
python examples/export_autograd_triton/dynamic.py
```

## Current implementation map

- `api.py`: public `export_autograd_triton(...)`, specialization capture orchestration
- `capture.py`: AOTAutograd/Inductor capture, differentiable-output and residual metadata extraction
- `codegen.py`: wrapper source rendering and artifact writing
- `templates/autograd_module.py.j2`: small generated wrapper template
- `runtime.py`: imported helper runtime for generated wrappers
- `clean_triton.py`: clean-Triton conversion, dynamic launch patching, baked-launch validation
- `guards.py`: static and tensor guard metadata
- `loading.py`: `load_exported_module(path)` helper
- `specs.py`: public and internal dataclasses

## Known next iterations

- Symbolic stride guards for broader dynamic layouts
- Dynamic CUDA device ordinal support
- More dynamic clean-Triton families and variants as Inductor output changes
- Semantic names for common generated symbols such as `arg0_1`, `buf*`, and `s*`
- Optional Nsight Compute-based bandwidth measurement for exact HBM traffic
