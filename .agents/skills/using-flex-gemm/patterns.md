# FlexGEMM epilogue patterns

These examples use the FX epilogue path and QUACK backend. Compile the enclosing function with `torch.compile(..., fullgraph=True)`. Check [constraints.md](constraints.md) before combining grouped, reduced, blocked, or quantized outputs. Runnable, benchmarked versions of most patterns live in `examples/flex_gemm/` (see its README); file references below are repo-relative.

## Common imports

```python
import torch
import torch.nn.functional as F
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import to_blocked
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise

KERNEL_OPTIONS = {"backend": "QUACK", "autotune": True}
```

## Shape-preserving pointwise epilogue

```python
def epilogue(acc):
    return F.silu(acc.float()).to(acc.dtype)


def fn(a, b):
    return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=KERNEL_OPTIONS)
```

Keep numerically sensitive math in `float()` and cast only the returned value when appropriate. More pointwise forms (bias, per-row/per-column scales, `addmm` with `alpha`): `examples/flex_gemm/01_pointwise.py`.

## Captured broadcast tensors

```python
def fn(a, b, per_row_scale, per_column_bias):
    # per_row_scale: [M, 1]; per_column_bias: [1, N]
    def epilogue(acc):
        return ((acc.float() + per_column_bias) * per_row_scale).to(acc.dtype)

    return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=KERNEL_OPTIONS)
```

Use ordinary PyTorch broadcasting. If the capture does not match a supported row, column, scalar, or tile profile, the fused route should reject rather than guess a physical layout.

When a captured `[M, 2N]` tensor holds interleaved lanes, split the lanes on the captured dtype and convert afterwards. `h.view(m, n, 2).unbind(-1)` then `.float()` lowers; `h.float().view(m, n, 2).unbind(-1)` is rejected with `unsupported FlexGEMM epilogue op: aten.unbind.int`. `examples/flex_gemm/06_backward.py` does this for the saved pre-activation.

## N-axis group sum feeding main

```python
def fn(a, b, group=32):
    m, n = a.shape[0], b.shape[1]

    def epilogue(acc):
        x = acc.float().view(m, -1, group)
        scale = x.sum(-1, keepdim=True)
        return (x * scale.reciprocal()).view(m, n).to(acc.dtype)

    return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=KERNEL_OPTIONS)
```

This form requires `N % group == 0`. The `view` identifies the physical output dimension and group consumed by the supported reduction.

## Return a compressed auxiliary reduction

```python
def fn(a, b, group=16):
    m = a.shape[0]

    def epilogue(acc):
        x = acc.float().view(m, -1, group)
        aux = x.abs().amax(-1)
        return acc.relu(), aux

    return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=KERNEL_OPTIONS)
```

The function returns `(main [M, N], aux [M, N // group])`. The same aux form works over M groups: `acc.float().view(-1, group, n).abs().amax(1)` returns `[M // group, N]`. One epilogue reduces through exactly one grouped view; two reductions are accepted only when they share the same N-axis geometry and the first (keepdim) feeds the second, which is how `logsumexp` lowers. Check [constraints.md](constraints.md) before composing this with grouped-main contraction or blocked storage. Grouped RMS norm, amax and partial logsumexp with timings: `examples/flex_gemm/04_reductions.py`.

## Return grouped logsumexp state

Write stable grouped logsumexp as ordinary PyTorch. It lowers as a same-geometry max-then-sum reduction chain (group <= 32); there is no logsumexp-specific user API.

```python
def fn(a, b, group=32):
    m = a.shape[0]

    def epilogue(acc):
        logits = acc.float()
        lse = logits.view(m, -1, group).logsumexp(-1)
        return logits.to(acc.dtype), lse

    return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=KERNEL_OPTIONS)
```

The group must be `<= 32` (`group=128` is rejected as two mismatched grouped reductions). A whole-row logsumexp is `lse.logsumexp(-1)` over the returned partials outside `flex_gemm`. It does not permit arbitrary independent reductions from one pass. Gathering one indexed value per row (`logits.gather(1, indices[:, None])`) is not supported: aux outputs must match the GEMM shape or be the partials of one grouped reduction.

## Interleaved SwiGLU grouped-main contraction

The physical GEMM computes `[M, 2N]`; the epilogue returns logical `[M, N]`.

```python
def fn(a, b):
    # b is contiguous [K, 2N], with columns ordered
    # gate0, up0, gate1, up1, ...
    m = a.shape[0]
    n = b.shape[1] // 2

    def epilogue(acc):
        lanes = acc.float().view(m, n, 2)
        gate, up = lanes[..., 0], lanes[..., 1]
        return (F.silu(gate) * up).to(acc.dtype)

    return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=KERNEL_OPTIONS)
```

Selected lanes must stay within one recognized lane domain (interleaved pairs or chunked halves); this is a contracted main output, not arbitrary slicing of the accumulator. Interleaved, chunked and MoE variants with timings: `examples/flex_gemm/03_swiglu.py`.

### Preserve packed lanes for backward

An interleaved, non-chunked, non-reducing epilogue may return the unchanged
accumulator as an auxiliary. This keeps the interleaved gate/up values for an
explicit backward without retaining or recomputing the original GEMM.

```python
def fn(a, b):
    m = a.shape[0]
    n = b.shape[1] // 2

    def epilogue(acc):
        lanes = acc.view(m, n, 2)
        gate = lanes.select(-1, 0)
        up = lanes.select(-1, 1)
        return (F.silu(gate) * up).to(acc.dtype), acc

    return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=KERNEL_OPTIONS)
```

Auxiliaries must have the full physical GEMM shape: the unchanged `acc` or a derived
full-shape value (`acc.relu()`), plus optional N-group partials; slices and
individual lanes are rejected. Grouped-main remains unsupported for batched GEMMs; reshape a
singleton-batch `bmm`'s operands to 2-D only when that reshape preserves its
exact contraction and interleaved output layout. `examples/flex_gemm/06_backward.py`
saves `acc` this way and consumes it in the backward epilogue.

Do not assume an explicit low-precision cast roundtrip preserves eager activation
semantics: the emitter can elide it through compute-type promotion. Verify
generated code and a roundtrip-to-FP32 numerical probe before claiming the live
activation uses the saved auxiliary's rounded values; see the precision contract
in [constraints.md](constraints.md).

## Chunked SwiGLU grouped-main contraction

```python
def fn(a, packed_weight):
    def epilogue(acc):
        gate, up = acc.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(acc.dtype)

    # Store the linear weight as [2N, K] and pass its transpose. The GEMM's
    # output dimension must be noncontiguous for the chunked grouped layout.
    return flex_gemm(
        torch.mm,
        (a, packed_weight.t()),
        epilogue,
        kernel_options=KERNEL_OPTIONS,
    )
```

The transposed `[2N, K]` storage is the chunked layout contract; a contiguous `[K, 2N]` tensor represents the interleaved form.

## Grouped SwiGLU (MoE experts)

`F.grouped_mm` accepts the same interleaved and chunked SwiGLU epilogues as dense `torch.mm`; `offs`
travels in `gemm_kwargs`. `w13_t` is `[E, K, 2 * hidden]` with w1/w3 rows interleaved per expert
(or `[E, 2 * hidden, K].transpose(-2, -1)` for the chunked form).

```python
def swiglu(acc):
    pairs = acc.float().view(acc.shape[0], -1, 2)
    gate, up = pairs.select(-1, 0), pairs.select(-1, 1)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


h = flex_gemm(
    torch.nn.functional.grouped_mm,
    (x, w13_t),
    swiglu,
    gemm_kwargs={"offs": offs},
    kernel_options={"backend": "QUACK", "autotune": True},
)
```

Grouped-main stores under varlen are legal only for `cluster_n == 1` configs; the untuned default is
`128x128 c1x1`. Measured 1.1-1.6x faster than compiled `grouped_mm + silu * mul` at DeepSeek-V3 expert shapes
(GB200). Varlen-M also supports stored N-group reductions and full-tile captures; unsupported
QUACK compositions raise rather than fall back. The MoE version with timings is the last case in `examples/flex_gemm/03_swiglu.py`.

## MXFP8 producer with blocked scales

Keep the recipe local: ordinary PyTorch expresses the reduction and normalization, while inline PTX supplies only the E8M0 conversion that PyTorch cannot otherwise spell.

```python
def mx_scale(amax):
    return inline_asm_elementwise(
        amax.float() / 448.0,
        asm_str="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
        constraints="=h,r",
        dtype=torch.float8_e8m0fnu,
    )


def mx_quant_epilogue(acc):
    group = 32
    x = acc.float().view(acc.shape[0], -1, group)
    scale = mx_scale(x.abs().amax(-1, keepdim=True))
    quantized = (x * scale.float().reciprocal()).view_as(acc)
    return (
        quantized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn),
        to_blocked(scale.squeeze(-1)),
    )
```

The blocked scale feeds `F.scaled_mm` with `F.ScalingType.BlockWise1x32` and `F.SwizzleType.SWIZZLE_32_4_4`. The E8M0 helper needs a separate eager reference if it is called outside `torch.compile`; see [inline-asm.md](inline-asm.md). The result is bit-exact against `torch.compile(mx_quant_epilogue)(a.float() @ b.float())`; compare against the fp32 accumulator, not a bf16-rounded GEMM, or fp8 codes differ by one ulp near rounding boundaries. Full producer plus the consuming `F.scaled_mm`: `examples/flex_gemm/05_mx_quant.py`.

Quantizing the same accumulator along N and along M in one epilogue (dual-axis scales) is rejected (`physical local-reduce feed-main source expressions require two-phase local-reduce source lowering`); run a second pass for the transposed operand.

## NVFP4 packed producer with blocked scales

The shape-changing pair contract belongs in ordinary FX. Inline assembly performs the register conversion but does not decide which values form a semantic pair.

```python
def pack_e2m1x2(pairs):
    low = pairs[..., 0]
    high = pairs[..., 1]
    packed = inline_asm_elementwise(
        high,
        low,
        asm_str=(
            "{ .reg .b8 packed; "
            "cvt.rn.satfinite.e2m1x2.f32 packed, $1, $2; "
            "cvt.u32.u8 $0, packed; }"
        ),
        constraints="=r,r,r",
        dtype=torch.uint8,
    )
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def nvfp4_quant_epilogue(acc):
    group = 16
    x = acc.float().view(acc.shape[0], -1, group)
    scale = torch.clamp(
        x.abs().amax(-1, keepdim=True) / 6.0,
        min=torch.finfo(torch.float8_e4m3fn).tiny,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)
    normalized = x * scale.float().reciprocal()
    packed = pack_e2m1x2(normalized.view(acc.shape[0], -1, 2))
    return packed, to_blocked(scale.squeeze(-1))
```

Use `F.ScalingType.BlockWise1x16` with `F.SwizzleType.SWIZZLE_32_4_4` at the `F.scaled_mm` consumer. Keep the same-width dtype view terminal. This local helper replaces the need for a dedicated core packing operation; see [inline-asm.md](inline-asm.md) for the exact semantics. Validate against `torch.compile(nvfp4_quant_epilogue)(a.float() @ b.float())`: the eager Jiterator cannot compile this asm.

## Stochastic rounding with an in-kernel RNG

`examples/flex_gemm/07_stochastic_rounding.py` builds Philox4x32-10 from int32 PyTorch ops plus one PTX wide multiply, then rounds with `cvt.rs.satfinite.bf16x2.f32` (SM100a/SM103a). Two rules it depends on: write int32 constants above `2**31` as their signed int32 value (`0x9E3779B9 - 2**32`); the unsigned Python int can crash CuTeDSL codegen with `'arith.bitcast' op operand type 'i64' and result type 'i32'`. Keep `mul.lo.u32`/`mul.hi.u32` in one asm block with a tuple `dtype` so ptxas emits a single wide multiply; see [inline-asm.md](inline-asm.md).

## torch.func inside the epilogue

The epilogue is traced like any PyTorch function, so `torch.func` transforms compose inside it and the derivative math runs in the GEMM:

```python
def derivative(f):
    return lambda x: torch.func.vjp(f, x)[1](torch.ones_like(x))[0]


def silu_and_derivatives(acc):
    h = acc.float()
    fns = (F.silu, derivative(F.silu), derivative(derivative(F.silu)))
    return tuple(f(h).to(acc.dtype) for f in fns)
```

Nested `vjp` with an explicit `torch.ones_like` cotangent and `torch.func.jvp(F.silu, (h,), (tangent,))` with a captured tangent both lower. `aten.expand` is not lowered, so `torch.func.grad` of a `.sum()` and `jvp` through a Python-scalar multiply (`0.5 * x`) fail with `unsupported FlexGEMM epilogue op: aten.expand.default`. `examples/flex_gemm/08_func_transforms.py` times both forms; `06_backward.py` uses `vjp` to turn a forward SwiGLU into its backward epilogue.

## AddMM and batched GEMMs

The epilogue syntax is unchanged when using `torch.addmm`, `torch.bmm`, or `torch.baddbmm`; pass their normal positional operands as `gemm_args` and static `alpha`/`beta` through `gemm_kwargs` when needed:

```python
return flex_gemm(
    torch.addmm,
    (bias, a, b),
    epilogue,
    gemm_kwargs={"alpha": 1.0, "beta": 1.0},
    kernel_options=KERNEL_OPTIONS,
)
```

Feature support can be narrower for batched or C-input GEMMs. Check the current constraints and tests before assuming a grouped or reduced output composes.
