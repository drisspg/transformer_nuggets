"""Inline PTX: a GEMM that generates random bits and stochastically rounds its output.

``inline_asm_elementwise`` drops a PTX snippet into the epilogue as an ordinary PyTorch op:
inputs are tensors (broadcast like any capture), outputs are tensors, and ``constraints`` map
them to PTX registers (``=r`` 32-bit output, ``f`` fp32 input, ``h`` 16-bit). Anything the
GPU can do per element becomes usable from a PyTorch epilogue.

Two instructions do the work here:

    cvt.rs.satfinite.bf16x2.f32   fp32 -> bf16, rounding up with probability equal to the
                                  dropped fraction (stochastic rounding, SM100a/SM103a)
    mul.lo/mul.hi.u32             the 32x32 -> 64-bit multiply that Philox needs

The Philox4x32-10 counter RNG is plain PyTorch int32 arithmetic around that multiply. Its
counter is (row, col, invocation) and its key is a seed, so the bits depend only on explicit
inputs: no hidden RNG state, the same seed and invocation reproduce the output, and bumping
``invocation`` draws new bits without recompiling. Unlike a precomputed random [M, N] tensor,
nothing random is read from memory.

Run: python examples/flex_gemm/07_stochastic_rounding.py   (SM100a/SM103a only)
"""

import torch
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise

from _common import QUACK_AUTOTUNE, report

M, N, K = 4096, 4096, 4096


def stochastic_bf16(x, random_bits):
    """Round fp32 ``x`` to bf16 using the low 16 bits of ``random_bits`` (one lane of x2)."""
    return inline_asm_elementwise(
        x,
        random_bits,
        asm_str=(
            "{ .reg .b32 packed; "
            "cvt.rs.satfinite.bf16x2.f32 packed, 0f00000000, $1, $2; "
            "cvt.u16.u32 $0, packed; }"
        ),
        constraints="=h,f,r",
        dtype=torch.bfloat16,
    )


def mul_wide_u32(x, multiplier: int):
    """Low and high 32 bits of an unsigned 32x32 product, in one asm block.

    Keeping both halves together lets ptxas emit a single wide multiply; two separate ops
    cost ~5% here.
    """
    return inline_asm_elementwise(
        x,
        asm_str=f"mul.lo.u32 $0, $2, {multiplier:#x}; mul.hi.u32 $1, $2, {multiplier:#x};",
        constraints="=r,=r,r",
        dtype=(torch.int32, torch.int32),
    )


def as_int32(x: int) -> int:
    """The int32 with the same bits as unsigned ``x``, so int32 tensor math stays int32."""
    return x - 2**32 if x >= 2**31 else x


def philox(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 word 0. int32 adds and xors wrap modulo 2**32 like the reference."""
    for _ in range(10):
        lo0, hi0 = mul_wide_u32(c0, 0xD2511F53)
        lo1, hi1 = mul_wide_u32(c2, 0xCD9E8D57)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0, k1 = k0 + as_int32(0x9E3779B9), k1 + as_int32(0xBB67AE85)
    return c0


def split_u64(x):
    """int64 [1, 1] -> two int32 [1, 1] views (low, high); no copy, no kernel."""
    return x.view(torch.int32).split(1, dim=-1)


def gemm_stochastic_bf16(a, b, rows, cols, seed, invocation, return_debug=False):
    """bf16 ``a @ b`` with stochastic rounding; rows [M, 1] and cols [1, N] are indices."""
    k0, k1 = split_u64(seed)
    c2, c3 = split_u64(invocation)

    def epilogue(acc):
        bits = philox(rows, cols, c2, c3, k0, k1)
        out = stochastic_bf16(acc.float(), bits)
        # The debug outputs expose the fp32 accumulator and bits for the bitwise check.
        return (out, acc.float(), bits) if return_debug else out

    return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=QUACK_AUTOTUNE)


def stochastic_bf16_reference(x, random_bits):
    """Integer oracle for finite in-range values: add 16 random bits below bf16's LSB."""
    rounded = (x.view(torch.int32) + (random_bits & 0xFFFF)) >> 16
    return rounded.to(torch.int16).view(torch.bfloat16)


def main() -> None:
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    a, b = torch.randn(M, K, **kw), torch.randn(K, N, **kw) / K**0.5
    rows = torch.arange(M, device="cuda", dtype=torch.int32)[:, None]
    cols = torch.arange(N, device="cuda", dtype=torch.int32)[None, :]
    seed = torch.tensor([[1234]], device="cuda")
    invocation = torch.zeros(1, 1, dtype=torch.int64, device="cuda")

    fused = torch.compile(gemm_stochastic_bf16, fullgraph=True)

    # 1. Bitwise check against the integer oracle on the kernel's own accumulator and bits.
    out, acc, bits = fused(a, b, rows, cols, seed, invocation, return_debug=True)
    assert torch.equal(out, stochastic_bf16_reference(acc, bits)), "rounding mismatch"

    # 2. Same inputs reproduce; bumping the counter (outside the kernel) draws new bits.
    first = fused(a, b, rows, cols, seed, invocation)
    assert torch.equal(first, fused(a, b, rows, cols, seed, invocation))
    invocation.add_(1)
    changed = (fused(a, b, rows, cols, seed, invocation) != first).float().mean()
    print(f"bitwise oracle ok; next draw changes {changed:.1%} of outputs")

    # 3. Unbiased: averaging draws converges to the fp32 result; round-to-nearest stays put.
    draws = []
    for _ in range(64):
        invocation.add_(1)
        draws.append(fused(a, b, rows, cols, seed, invocation).float())
    exact = a.float() @ b.float()
    sr_err = (torch.stack(draws).mean(0) - exact).abs().mean()
    rn_err = ((a @ b).float() - exact).abs().mean()
    print(f"mean |error| vs fp32: 64-draw SR average {sr_err:.2e}, round-to-nearest {rn_err:.2e}")

    # Unfused: store the fp32 GEMM result, then a second kernel generates bits and rounds.
    @torch.compile(fullgraph=True)
    def unfused(a, b, rows, cols, seed, invocation):
        k0, k1 = split_u64(seed)
        c2, c3 = split_u64(invocation)
        acc = torch.mm(a, b, out_dtype=torch.float32)
        return stochastic_bf16(acc, philox(rows, cols, c2, c3, k0, k1))

    args = (a, b, rows, cols, seed, invocation)
    report("gemm + philox + SR", fused, unfused, *args, check=False)


if __name__ == "__main__":
    main()
