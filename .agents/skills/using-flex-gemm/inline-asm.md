# Inline PTX in FlexGEMM epilogues

Use `inline_asm_elementwise` when ordinary PyTorch FX cannot express a pure register-level operation. It is a computation extension point, not a way to bypass FlexGEMM's output-layout, lifetime, synchronization, or configuration rules.

```python
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
```

Prefer ordinary PyTorch whenever it has the needed semantics. Inline PTX makes the numerical and bit-level contract the user's responsibility.

## Public tensor contract

```python
inline_asm_elementwise(
    *inputs,
    asm_str=...,
    constraints=...,
    dtype=...,
    is_pure=True,
    pack=1,
)
```

- At least one tensor input is required.
- Inputs broadcast using ordinary pointwise broadcasting.
- The result has the broadcast shape, pointwise-propagated strides, and requested logical `dtype`.
- `asm_str` uses `$N` operands. Outputs come first, followed by inputs in constraint order.
- A tuple `dtype` returns several outputs from one block (see below).
- `is_pure` must be true. The compiler may reorder or common-subexpression-eliminate the operation.
- Autograd is not implemented. Own an explicit backward if the enclosing operation requires gradients.
- Eager execution uses CUDA Jiterator, requires CUDA tensors with one common input dtype, and supports only `pack=1`. It also rejects some output dtypes (`float8_e8m0fnu`) and fails to compile `=r` outputs declared as `uint8`. For a reference, run the same epilogue under plain `torch.compile` on `a.float() @ b.float()` instead of eagerly.
- `pack > 1` requires `torch.compile`.

Under the QUACK FlexGEMM path, Inductor lowers the operation into the generated CuTeDSL EpiMod so it can compose with surrounding pointwise and grouped-reduction expressions.

## Register constraints

The current CuTeDSL lowering accepts these NVPTX register classes:

| Constraint | Register width/type |
| --- | --- |
| `h` | 16-bit integer register |
| `r` | 32-bit integer register |
| `l` | 64-bit integer register |
| `f` | Float32 register |
| `d` | Float64 register |

A constraint selects the physical register representation. It does not request a numerical cast. Cast in PyTorch before the call or write an explicit PTX conversion when numerical conversion is intended.

For example, one Float32 output from two Float32 inputs uses:

```text
constraints="=f,f,f"
operands: $0=output, $1=input0, $2=input1
```

The requested output `dtype` controls the logical result and eventual storage. Integer register results are narrowed to that dtype's width before reconstruction.

## Multiple outputs from one block

Pass a tuple `dtype` with one entry per output constraint. Both halves of a 32x32 -> 64-bit multiply in one block let ptxas emit a single wide multiply (two separate calls cost about 5% in the Philox epilogue of `examples/flex_gemm/07_stochastic_rounding.py`):

```python
def mul_wide_u32(x, multiplier: int):
    return inline_asm_elementwise(
        x,
        asm_str=f"mul.lo.u32 $0, $2, {multiplier:#x}; mul.hi.u32 $1, $2, {multiplier:#x};",
        constraints="=r,=r,r",
        dtype=(torch.int32, torch.int32),
    )
```

Int32 PyTorch arithmetic around such blocks wraps modulo `2**32`, but write Python constants above `2**31` as their signed int32 value (`0x9E3779B9 - 2**32`): the unsigned literal can crash CuTeDSL codegen with `'arith.bitcast' op operand type 'i64' and result type 'i32'`.

## `pack` preserves logical shape

`pack=P` invokes one assembly block for `P` logical output elements. It does **not** divide a tensor dimension, change the result shape, or define which values form a semantic storage pair.

The constraint list must contain:

- exactly `P` output constraints;
- exactly `P` input constraints for each input source.

Inputs are source-major, then lane-major. For one source and `pack=2`:

```python
result = inline_asm_elementwise(
    x,
    asm_str="mov.b32 $0, $2; mov.b32 $1, $3;",
    constraints="=r,=r,r,r",
    dtype=torch.float32,
    pack=2,
)
```

The operands are:

```text
$0 output lane 0
$1 output lane 1
$2 x lane 0
$3 x lane 1
```

For two sources, the inputs after the outputs are `x lane 0`, `x lane 1`, `y lane 0`, `y lane 1`.

The public API intentionally does not promise which global tensor indices share an invocation. Use `pack=2` only when each returned lane has the same elementwise meaning regardless of grouping. Do not use `pack` to define low/high nibble order or another user-visible layout.

The current CuTeDSL implementation zero-pads an incomplete tail and discards outputs for padded lanes. Treat that as lowering behavior, not a reason to make a valid output depend on a padded lane. Float lanes may also be declared `=f,=f,f,f`; the `mov.b32` example above only moves bits, so either register class works.

## Shape-changing packed storage

When two logical values must become one physical storage element, express the pair axis explicitly in FX and pass the two lanes as separate inputs with `pack=1`:

```python
def pack_e2m1x2(pairs):
    if pairs.dtype is not torch.float32 or pairs.ndim == 0 or pairs.shape[-1] != 2:
        raise ValueError("expected Float32 pairs with final dimension 2")

    low = pairs[..., 0]
    high = pairs[..., 1]
    byte = inline_asm_elementwise(
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
    return byte.contiguous().view(torch.float4_e2m1fn_x2)
```

The explicit `[..., 2]` dimension determines the low/high pair and causes the selected inputs to have half as many logical elements. `inline_asm_elementwise` itself remains shape-preserving over those selected inputs.

For the QUACK FlexGEMM route, a terminal dtype view is supported when shape, stride, and element size are unchanged. Keep the view terminal. A different-width bitcast or further arithmetic through a packed storage dtype requires another supported value-domain representation.

## E8M0 results

E8M0 is useful as storage but fused consumers generally need a numeric scale. Request the logical E8M0 dtype directly:

```python
def mx_scale(amax):
    return inline_asm_elementwise(
        amax.float() / 448.0,
        asm_str="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
        constraints="=h,r",
        dtype=torch.float8_e8m0fnu,
    )
```

The CuTeDSL lowering preserves E8M0 as the logical stored dtype while decoding the register result to an exact Float32 value for downstream fused arithmetic:

- codes 1 through 254 map directly to the corresponding Float32 exponent;
- code 0 maps to `2^-127`;
- code 255 maps to NaN.

This works because the finite E8M0 exponent encoding maps exactly into Float32 exponent bits. It is not an approximate floating conversion.

CUDA Jiterator does not provide the same special E8M0 path (`Unsupported output dtype for inline asm: torch.float8_e8m0fnu`). For eager references use the ordinary-PyTorch form `torch.exp2(torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)).to(torch.float8_e8m0fnu)`, which matches the fused codes bit-exactly when fed the fp32 accumulator; `examples/flex_gemm/05_mx_quant.py` keeps both helpers side by side.

## Validation

For every inline-assembly helper:

1. Compare against an independent ordinary-PyTorch or bit-level reference, including saturation, ties, zero, infinities, and NaNs where relevant.
2. Test broadcasting and the exact packed/view dtype contract.
3. For `pack > 1`, include a fragment or tensor extent not divisible by `pack` when the surrounding lowering permits it.
4. Inspect generated code for `inline_asm_elementwise_intrinsic` and confirm the FlexGEMM route did not fall back to an external GEMM.
5. Benchmark against the compiled expression users would otherwise run. A paired instruction is not automatically faster when surrounding conversion, unpacking, or storage dominates.
