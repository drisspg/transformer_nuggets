# FlexGEMM examples

`flex_gemm` fuses a GEMM with an epilogue written as an ordinary PyTorch function:

```python
from torch._higher_order_ops import flex_gemm

@torch.compile(fullgraph=True)
def linear_gelu(a, b, bias):
    def epilogue(acc):                  # acc is the [M, N] GEMM accumulator
        return F.gelu(acc + bias)       # tensors from the enclosing scope are captured

    return flex_gemm(torch.mm, (a, b), epilogue, kernel_options={"backend": "QUACK"})
```

One kernel computes `epilogue(torch.mm(a, b))`: the epilogue runs on the accumulator in
registers before anything is stored (`acc` has the output dtype; `acc.float()` keeps fp32). Each script writes its epilogue once, runs it inside `flex_gemm` and
on `a @ b` under plain `torch.compile`, checks they agree, and prints both GPU times.

| Script | Pattern | GB200 vs `torch.compile` |
| --- | --- | --- |
| `01_pointwise.py` | bias, activations, per-row/per-column scales; `addmm` with alpha | 1.12-1.32x |
| `02_multiple_outputs.py` | return a tuple: keep the pre-activation, write a bf16 + fp8 copy | 1.10-1.18x |
| `03_swiglu.py` | `[M, 2N]` GEMM stores only `[M, N]`: interleaved, chunked, MoE `F.grouped_mm` | 1.17-1.28x |
| `04_reductions.py` | grouped RMS norm, amax for scales, partial logsumexp for a whole row | 1.04-1.34x |
| `05_mx_quant.py` | GEMM writes MXFP8 data + blocked E8M0 scales for the next `F.scaled_mm` | 1.18x |
| `06_backward.py` | `autograd.Function` whose backward GEMM runs `torch.func.vjp` in its epilogue | 1.19x |
| `07_stochastic_rounding.py` | inline PTX: Philox RNG + stochastic bf16 rounding in the epilogue | 1.27x |

The full worked example is the MoE in
[`transformer_nuggets/moe/flex_gemm_experts.py`](../../transformer_nuggets/moe/flex_gemm_experts.py):
six grouped GEMMs forward and backward, fused step by step.

```bash
python examples/flex_gemm/01_pointwise.py
```

Requires a PyTorch nightly with `torch._higher_order_ops.flex_gemm`, the `nvidia-cutlass-dsl`
package, and a Blackwell GPU (SM100). Hopper runs plain and grouped GEMMs with pointwise
epilogues; the SwiGLU, reduction, MX and stochastic-rounding patterns are Blackwell-only.

## What to expect

Fusion removes the second kernel that re-reads the GEMM result: about `4MN` bytes against the
GEMM's `2MNK` FLOPs. The win therefore grows as K shrinks and as M*N grows, and the shapes here
(K = 1024-2048) are chosen to show it. At 4096^3 the same scripts print 1.0-1.1x: the GEMM is
~110 us and the elementwise kernel ~10 us. It is not a universal win:

- `torch.compile` already fuses some epilogues into cuBLAS: `a @ b + residual` becomes one
  `addmm`, and FlexGEMM ties.
- A heavy epilogue costs time inside the GEMM. The MXFP8 quantize loses to a separate kernel
  at K <= 1024, and so do exact-math SiLU epilogues at small M.
- Whole-step speedups shrink when the GEMMs dominate: the MLP in `06` is ~1.05x at Llama-8B
  sizes.

## Inline PTX

`inline_asm_elementwise` is the escape hatch: a PTX snippet becomes an elementwise PyTorch op
that works inside the epilogue, so any per-element instruction the GPU has is available.

```python
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise

scale = inline_asm_elementwise(
    amax / 448.0,
    asm_str="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",  # fp32 -> E8M0, rounding up
    constraints="=h,r",  # $0: 16-bit output, $1: 32-bit input
    dtype=torch.float8_e8m0fnu,
)
```

Inputs broadcast like any other capture, an op can return several outputs (`"=r,=r,r"` with a
tuple `dtype`), and ordinary PyTorch ops compose around it. `05` uses one instruction for MX
scales; `07` builds a full Philox RNG out of int32 PyTorch ops plus a PTX wide multiply, then
stochastically rounds with `cvt.rs.satfinite.bf16x2.f32`, all in the epilogue.

## Kernel options

- `"backend": "QUACK"` selects the fused CuTeDSL kernel and raises `NotImplementedError` naming
  the limit if the epilogue is unsupported. Without `backend`, Inductor lowers the epilogue as
  ordinary PyTorch (a cuBLAS GEMM plus a separate kernel): no fusion.
- `"autotune": True` benchmarks a measured shortlist of legal tile configs (about a dozen;
  `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE` tries all). Heavy epilogues need it
  (`06`); for light ones the default tile is often as good.
- `"fast_math": True` rewrites sigmoid/SiLU through tanh, uses tanh GELU, and allows approximate
  hardware math. Faster for nonlinear epilogues; check accuracy. The examples spell the SiLU
  rewrite out (`_common.silu`) so both sides compute the same thing.

## Gotchas

- Do the math in fp32 (`acc.float()`) and cast the outputs.
- There is no autograd formula; write the backward yourself (`06`).
- Split interleaved lanes with `view(M, N, 2)`; for a captured tensor, split before converting
  (`h.view(M, N, 2).unbind(-1)`, then `.float()`).
- Grouped reductions feeding the main output (and logsumexp) need group size <= 32.
- Python int constants above 2**31 in int32 math crash compilation; write them as signed int32
  (`07`).

## Debugging

```bash
TORCH_LOGS=flex_gemm python examples/flex_gemm/01_pointwise.py               # analysis and plan
TORCH_LOGS='flex_gemm,output_code' python examples/flex_gemm/01_pointwise.py # generated code
```

The generated wrapper calls `async_compile.cutedsl(...)` for the fused GEMM.
