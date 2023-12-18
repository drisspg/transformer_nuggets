import torch
import triton
import triton.language as tl


@triton.jit
def scaled_cast(
    inpt_ptr: torch.Tensor,
    output_ptr: torch.Tensor,
    scale_ptr: torch.Tensor,
    abs_max_ptr: torch.Tensor,
    numel: int,
    XBLOCK: tl.constexpr,
    float8_dtype: tl.constexpr,
    max_val: tl.constexpr,
):
    """Quantize tensor to fp8 using a delayed scaled and calculate abs_max"""
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    index = tl.max_contiguous(tl.multiple_of(index, XBLOCK), XBLOCK)
    mask = index < numel
    inpt = tl.load(inpt_ptr + (index), mask=mask)
    block_max = tl.max(tl.abs(inpt))
    tl.atomic_max(abs_max_ptr, block_max)
    scale = tl.load(scale_ptr)
    scaled_inpt = inpt * scale
    if max_val != 0.0:
        # Wanted to pass in a `saturated : bool` but it doesn't work
        # and can't branch off of tl.dtype
        tl.where(scaled_inpt > max_val, max_val, scaled_inpt)
        tl.where(scaled_inpt < -1 * max_val, -1 * max_val, scaled_inpt)
    tl.store(output_ptr + (index), scaled_inpt.to(float8_dtype), mask=mask)


def scaled_quant(
    inpt_tensor: torch.Tensor,
    scale: torch.Tensor,
    abs_max: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    saturated: bool = False,
):
    """Quantize tensor to fp8 using a delayed scaled and calculate abs_max
    for use in the next iteration of quantization.

    Args:
        inpt_tensor: Input tensor to quantize
        scale: Scale to apply to input tensor, calculated from previous abs_max
        abs_max: Absolute maximum value of input tensor, will be updated
        fp8_dtype: FP8 datatype to quantize to
        saturated: Whether to saturate the output tensor to the maximum value
            of the fp8 datatype
    """
    assert scale.dtype == torch.float32
    assert abs_max.dtype == torch.float32
    assert scale.numel() == 1
    assert abs_max.numel() == 1
    assert inpt_tensor.is_contiguous(), "Input tensor must be contiguous"

    out_tensor = torch.empty_like(inpt_tensor, dtype=fp8_dtype, device="cuda")
    numel = inpt_tensor.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["XBLOCK"]),)
    tl_dtype = {torch.float8_e4m3fn: tl.float8e4nv, torch.float8_e5m2: tl.float8e5}[fp8_dtype]
    max_val = torch.finfo(fp8_dtype).max if saturated else 0.0
    scaled_cast[grid](
        inpt_tensor, out_tensor, scale, abs_max, numel, 4096, tl_dtype, max_val, num_warps=8
    )
    return out_tensor


def eager_scaled_quant(
    a: torch.Tensor,
    scale: torch.Tensor,
    abs_max: torch.Tensor,
    fp8_dtype: torch.dtype,
    saturated: torch.dtype = False,
):
    """Quantize tensor to fp8 using a delayed scaled and calculate abs_max

    Args:
        a: Input tensor to quantize
        scale: Scale to apply to input tensor, calculated from previous abs_max
        abs_max: Absolute maximum value of input tensor, will be updated
        fp8_dtype: FP8 datatype to quantize to
        saturated: Whether to saturate the output tensor to the maximum value
            of the fp8 datatype
    """
    out = a * scale
    if saturated:
        out = torch.where(out > torch.finfo(fp8_dtype).max, torch.finfo(fp8_dtype).max, out)
        out = torch.where(
            out < -1 * torch.finfo(fp8_dtype).max, -1 * torch.finfo(fp8_dtype).max, out
        )
    abs_max = torch.max(torch.abs(out))
    return out.to(fp8_dtype)
