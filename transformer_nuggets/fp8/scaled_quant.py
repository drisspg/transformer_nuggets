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
):
    """Quantize tensor to fp8 using a delayed scaled and calculate abs_max"""
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < numel
    inpt = tl.load(inpt_ptr + (index), mask=mask)
    block_max = tl.max(tl.abs(inpt))
    tl.atomic_max(abs_max_ptr, block_max)
    scale = tl.load(scale_ptr)
    scaled_inpt = inpt * scale
    tl.store(output_ptr + (index), scaled_inpt.to(float8_dtype), mask=mask)


def scaled_quant(
    inpt_tensor: torch.Tensor,
    scale: torch.Tensor,
    abs_max: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
):
    """Quantize tensor to fp8 using a delayed scaled and calculate abs_max"""
    assert scale.dtype == torch.float32
    assert abs_max.dtype == torch.float32
    assert scale.numel() == 1
    assert abs_max.numel() == 1

    out_tensor = torch.empty_like(inpt_tensor, dtype=fp8_dtype, device="cuda")
    numel = inpt_tensor.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["XBLOCK"]),)
    tl_dtype = {torch.float8_e4m3fn: tl.float8e4nv, torch.float8_e5m2: tl.float8e5}[fp8_dtype]
    scaled_cast[grid](inpt_tensor, out_tensor, scale, abs_max, numel, 4096, tl_dtype, num_warps=8)
    return out_tensor
