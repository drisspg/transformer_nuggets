"""MXFP8 producer: a GEMM that quantizes its own output for the next scaled GEMM.

One kernel computes ``a @ b``, takes the amax of every 32-column block, picks a power-of-two
E8M0 scale, and stores fp8 data plus the scales in the 128x4 blocked layout ``F.scaled_mm``
reads. Unfused, that is a GEMM, a bf16 round trip through memory, and a quantize kernel.

Two pieces make this work:

* ``to_blocked`` is an ordinary function (it runs eagerly too). In an epilogue, FlexGEMM
  recognizes it and writes the reduced scales straight into blocked storage.
* ``inline_asm_elementwise`` computes the scale with one PTX instruction,
  ``cvt.rp.satfinite.ue8m0x2.f32`` (fp32 -> E8M0, rounding up). See 07 for more inline PTX.

The quantize epilogue is real work: at small K (<= 1024) its cost is no longer hidden behind
the GEMM mainloop, and a separate quantize kernel wins.

Run: python examples/flex_gemm/05_mx_quant.py   (Blackwell only)
"""

import torch
import torch.nn.functional as F
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import to_blocked
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise

from _common import QUACK, assert_rel_error, report

M, N, K = 16384, 8192, 2048
BLOCK = 32
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def e8m0_scale_asm(amax):
    """Smallest power of two >= amax / 448, as E8M0, in one PTX instruction."""
    return inline_asm_elementwise(
        amax.float() / FP8_MAX,
        asm_str="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
        constraints="=h,r",
        dtype=torch.float8_e8m0fnu,
    )


def e8m0_scale_eager(amax):
    """The same rounding in plain PyTorch, for the unfused baseline and the weights."""
    exponent = torch.ceil(torch.log2(amax.float() / FP8_MAX)).clamp(-127, 127)
    return torch.exp2(exponent).to(torch.float8_e8m0fnu)


def mx_quantize(out, e8m0_scale):
    x = out.float().view(out.shape[0], -1, BLOCK)
    scale = e8m0_scale(x.abs().amax(-1, keepdim=True))
    data = (x / scale.float()).clamp(-FP8_MAX, FP8_MAX).view(out.shape).to(torch.float8_e4m3fn)
    return data, to_blocked(scale.squeeze(-1))


def mx_mm(a_data, a_scale, b_data, b_scale):
    return F.scaled_mm(
        a_data,
        b_data.t(),
        a_scale,
        F.ScalingType.BlockWise1x32,
        b_scale,
        F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )


def main() -> None:
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    a, b = torch.randn(M, K, **kw), torch.randn(K, N, **kw) / K**0.5

    @torch.compile(fullgraph=True)
    def fused_gemm_mx_quant(a, b):
        def epilogue(acc):
            return mx_quantize(acc, e8m0_scale_asm)

        return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=QUACK)

    def gemm_mx_quant(a, b):
        return mx_quantize(a @ b, e8m0_scale_eager)

    # Check the quantized output by using it: (a @ b) @ w2.T as an MXFP8 GEMM.
    w2 = torch.randn(N, N, **kw) / N**0.5
    out = mx_mm(*fused_gemm_mx_quant(a, b), *mx_quantize(w2, e8m0_scale_eager))
    expected = (a.float() @ b.float()) @ w2.float().t()
    assert_rel_error("chained MXFP8 GEMM", out, expected, tol=0.05)

    # Bitwise data can differ from the bf16-rounded baseline near rounding boundaries.
    report(
        "gemm + mxfp8 quant", fused_gemm_mx_quant, torch.compile(gemm_mx_quant), a, b, check=False
    )


if __name__ == "__main__":
    main()
