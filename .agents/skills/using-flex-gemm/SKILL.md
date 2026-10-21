---
name: using-flex-gemm
description: Guides FlexGEMM fusion and reductions in QUACK-enabled PyTorch checkouts. Use when implementing, debugging, or tuning FlexGEMM workloads and their backward formulas.
---

# Using FlexGEMM

Use FlexGEMM to fuse a GEMM with a functional FX epilogue. This transformer_nuggets repo copy is the maintained skill. The API is experimental: check the active PyTorch checkout before relying on a backend or composition.

Runnable references, relative to this repo root:

- [examples/flex_gemm/](../../../examples/flex_gemm/README.md): teaching scripts 01-08, including pointwise fusion, multiple outputs, SwiGLU, reductions, quantization, backward, stochastic rounding, and function transforms.
- [benchmarks/flex_gemm/](../../../benchmarks/flex_gemm/README.md): benchmark harness and config sweeps. Keep benchmark exploration out of the teaching scripts.

## Preflight

Check the interpreter's `torch.__file__` against the intended checkout, then check imports:

```python
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import to_blocked
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
```

Authoritative paths relative to the PyTorch checkout:

- `torch/_higher_order_ops/flex_gemm.py`: public normalization, body decompositions, alias/mutation and autograd contracts.
- `torch/_inductor/kernel/flex_gemm/constraints.py`: layout/reduction descriptors and composition constraints.
- `torch/_inductor/kernel/flex_gemm/fx_cutedsl_codegen.py`: TensorSSA epilogue and callback emission.
- `torch/_inductor/kernel/flex_gemm/lowering.py`: operand validation, output planning, legal configs, backend dispatch.
- `torch/_inductor/kernel/flex_gemm/template.py`, `compile_pool.py`: generated wrapper, winner guards, compilation.
- `torch/_inductor/kernel/flex_gemm/runtime.py`: generated-code callee and QuACK EpiMod construction.
- `torch/_inductor/kernel/flex_gemm/output_layout.py`, `output_layout_cutedsl.py`, `quack_reductions.py`: output storage and reduction support.
- `torch/_inductor/kernel/flex_gemm/quack_ops/`: PyTorch-owned EpiOps (`col_load`, `grouped_reduce`, `main_store`, `interleaved`, `packed_select`).
- `torch/_inductor/heuristics/template/flex_gemm.py`: dense/grouped priorities, untuned shape table, search-space policy.
- `torch/_inductor/kernel/flex_gemm/debug.py` and `test/inductor/test_flex_gemm.py`.

QUACK requires CUDA tensors and importable `nvidia-cutlass-dsl`. The runnable examples target Blackwell (GB200/SM100 verified); hardware and epilogue legality are config-dependent, not a blanket promise for every CUDA GPU. QuACK is vendored at `torch/_vendor/quack`; do not pip-install `quack` for this path. Concrete FlexGEMM EpiOps belong in `quack_ops/`, not the vendoring patches. Eager execution is ordinary PyTorch math, not evidence of fusion.

The public shape is:

```python
flex_gemm(
    gemm_op,
    gemm_args,
    epilogue_fn,
    *,
    gemm_kwargs=None,
    kernel_options=None,
)
```

Entry points include `torch.mm`, `torch.addmm`, `torch.bmm`, `torch.baddbmm`, `torch.nn.functional.grouped_mm` (also `torch._grouped_mm`), and `torch.nn.functional.scaled_mm`. Grouped GEMM takes `(a, b)` and `gemm_kwargs={"offs": offs}`; only variable-length M/K forms are normalized. Scaled-MM requires QUACK, matching supported MXFP8/NVFP4 recipes, and `(a, b, scale_a, scale_b)`; see [constraints.md](constraints.md). Direct `aten._scaled_mm_v2` and scaled grouped GEMM calls are rejected.

Omitting `kernel_options["backend"]` selects `TRITON`: this lowers the body through ordinary Inductor and does not guarantee a fused Triton GEMM. `QUACK` uses only the QuACK template path; unsupported compositions fail rather than silently becoming an external GEMM. `NVGEMM` uses ordinary Inductor lowering with NVGEMM choices enabled; its bf16x9 precision case explicitly falls back to ATen/cuBLAS. Inspect the executed route instead of inferring fusion from the option name.

## Mental model

The GEMM creates a **physical accumulator**. The epilogue interprets it and returns logical outputs: a main result, full-shape auxiliaries, recognized grouped reductions, or transformed storage. A `view`, `select`, or `chunk` describes how physical lanes map to grouped values or a contracted logical output. Not every ordinary PyTorch composition is representable.

Express semantics with ordinary PyTorch FX first. Use pure `inline_asm_elementwise` only for register-level operations with no suitable PyTorch expression; it does not bypass storage, lifetime, or synchronization contracts.

## Default invocation

```python
import torch
from torch._higher_order_ops import flex_gemm


def epilogue(acc):
    return torch.nn.functional.silu(acc.float()).to(acc.dtype)


def fn(a, b):
    return flex_gemm(
        torch.mm,
        (a, b),
        epilogue,
        kernel_options={"backend": "QUACK", "autotune": True},
    )


compiled = torch.compile(fn, backend="inductor", fullgraph=True)
```

QUACK accepts `backend`, `autotune`, `fast_math`, and `config`; unknown keys are rejected. Use booleans for `autotune` and `fast_math` (both default to False). `config` is a dict of exact, type-sensitive `GemmConfig` field constraints, not a separate launch API. Partial constraints filter legal configs before default selection or tuning; a complete legal config pins one choice. Unknown fields and empty legal intersections fail. `tuned` is not an accepted option. NVGEMM accepts only `backend` and `autotune`.

With `autotune=False`, dense calls use Inductor's measured shape table when a table entry is legal, variable-length-M calls use the grouped priority order, and otherwise the selector takes QuACK's preferred legal config. Block-scaled calls keep QuACK's shape-aware block-scaled default. With `autotune=True`, Inductor benchmarks a bounded legal shortlist and caches its winner; QuACK's own tuner does not run. Compare both settings: tuning is not guaranteed to beat the untuned pick. See [performance.md](performance.md) for the current search policy and measurement contract.

`fast_math=True` changes supported CuTeDSL math emission **and body decompositions**: sigmoid/SiLU use tanh identities and exact GELU becomes tanh GELU. It is not merely a compiler flag, and eager execution does not apply those QUACK tracing substitutions. Validate the changed numerical contract against an explicitly matched reference. A manually implemented backward does not update itself when forward math changes.

Epilogues must remain functional and FX-traceable. FlexGEMM has no built-in autograd formula. Wrap the complete block in `torch.autograd.Function` and own an explicit backward; see [autograd.md](autograd.md).

## Choose a pattern

Start with shape-preserving pointwise FX. Add captures, grouped reductions, grouped-main contraction, or quantized blocked outputs only when their constraints are satisfied.

- [patterns.md](patterns.md): copyable epilogue forms.
- [constraints.md](constraints.md): device, layout, reduction, composition, and config limits.
- [inline-asm.md](inline-asm.md): pure inline PTX, `pack`, register constraints, packed storage, and eager/compiled semantics.
- [autograd.md](autograd.md): forward/backward stage reversal.
- [performance.md](performance.md): tuning and diagnosis; read its selection-path contract when changing lowering, templates, or heuristics.

## Validate

1. Write an ordinary PyTorch reference using the same GEMM and epilogue, including explicit fast-math approximations when enabled.
2. Compile with `torch.compile(..., fullgraph=True)`.
3. Compare every output; use a higher-precision GEMM reference as the measuring stick for low-precision error.
4. Inspect generated code or diagnostics to confirm QUACK rather than an external GEMM.
5. Exercise the risky boundary: dynamic dimensions, non-aligned sizes, captures, tuple outputs, or grouped layouts.
6. For custom autograd, compare every requested input gradient against the reference.

Current sharp edges (checked against the local PyTorch checkout based on `98156a61b4ff`, with small GB200 probes):

- Matrix operands and outputs require supported layouts and 16-byte aligned outer strides; invalid examples such as row-major bf16 B with odd N fail during lowering. Making an already contiguous tensor contiguous again does not fix an odd leading stride; pad or choose an aligned layout.
- Returned values must not alias each other or inputs, and the body must not mutate inputs. Returning `x` and `x.t()` is rejected even when the apparent cast creating `x` is a no-op.
- Float64 captures are rejected during lowering. Strided fp32 vector captures such as `bias[::2]` are supported; do not impose a blanket stride-1 capture rule.
- Chunked grouped-main contraction requires column-major B; row-major B is rejected during lowering.
- Explicit configs remain dtype/layout-sensitive. In particular, illegal two-CTA TF32 MN-major tiles are filtered and rejected rather than launched; partial constraints select among the remaining legal tiles. Always validate a pinned config on the actual dtype/layout.

Do not weaken a guard until the physical layout, storage, synchronization, and configuration contracts are understood.

## Diagnose

```zsh
TORCH_LOGS=flex_gemm python repro.py
TORCH_LOGS=+flex_gemm python repro.py
TORCH_LOGS='flex_gemm,kernel_code' python repro.py
TORCH_LOGS='flex_gemm,output_code' python repro.py
TORCH_LOGS='+inductor,flex_gemm' python repro.py
```

- `flex_gemm`: problem, analysis, lowering-plan, and selection artifacts; `+flex_gemm` adds detailed decisions. These phases also emit structured trace artifacts named `flex_gemm_<phase>`.
- `kernel_code`: generated kernel source; `output_code`: final wrapper and emitted consumers.
- `+inductor`: compiler/candidate diagnostics. `TORCH_LOGS_FORMAT` controls formatting globally.

The QUACK template emits `@cute.jit def flex_gemm_epilogue_<hash>(acc, ...)` and imports `gemm_epilogue as flex_gemm_epilogue` from `torch._inductor.kernel.flex_gemm.runtime`. Its `cutedsl_<hash>_main(..., stream)` calls that runtime with the body, a pinned `config`, capture/output arguments, and any `FlexGemmRuntimeLocalReducePlan`. The body returns named ports such as `D`, `output0`, and reduction sinks; reductions can also generate finalize/prepass callbacks. This is generated code to inspect, not a stable API to copy.

For an isolated FlexGEMM region, check for `async_compile.cutedsl(` and the runtime call, and check that no `extern_kernels.{mm,bmm,addmm,baddbmm,_grouped_mm,_scaled_mm}` replaced the intended GEMM. In a larger region, unrelated external GEMMs are legitimate: attribute calls to the correct stage. Grouped metadata preparation or reduction consumers may add kernels, so QUACK selection does not imply the entire callable launches only one kernel. CSE forwarding aliases such as `tmpN = tmpM` inside the generated body are not themselves a bug.
