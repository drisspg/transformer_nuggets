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
        inpt_tensor,
        out_tensor,
        scale,
        abs_max,
        numel,
        4096,
        tl_dtype,
        max_val,
        num_warps=8,
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
    abs_max.fill_(torch.max(torch.abs(a)))
    return out.to(fp8_dtype)


# ----------- Dynamic Scaled Quantization ------------


@triton.jit
def dynamic_scaled_cast(
    inpt_ptr: torch.Tensor,
    output_ptr: torch.Tensor,
    abs_max_ptr: torch.Tensor,
    spin_lock: torch.Tensor,
    numel: int,
    XBLOCK: tl.constexpr,
    float8_dtype: tl.constexpr,
    max_val: tl.constexpr,
):
    """Quantize tensor to fp8 using current global absmax"""
    n_blocks = tl.num_programs(0)
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    index = tl.max_contiguous(tl.multiple_of(index, XBLOCK), XBLOCK)
    mask = index < numel
    inpt = tl.load(inpt_ptr + (index), mask=mask)
    block_max = tl.max(tl.abs(inpt))
    tl.atomic_max(abs_max_ptr, block_max)
    # Spinlock global barrier
    tl.atomic_add(spin_lock, 1, sem="release")
    while tl.load(spin_lock, volatile=True) < n_blocks:
        pass
    scale = max_val / (tl.clamp(tl.load(abs_max_ptr), -1e12, float("inf")))
    scaled_inpt = inpt * scale
    # Saturated casting
    scaled_inpt = tl.clamp(scaled_inpt, -1 * max_val, max_val)
    tl.store(output_ptr + (index), scaled_inpt.to(float8_dtype), mask=mask)


def dynamic_scaled_quant(
    inpt_tensor: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e4m3fn
) -> torch.Tensor:
    """Quantize tensor to fp8 using dynamic scale calculated from abs_max
    It will do saturated casting

    Args:
        inpt_tensor: Input tensor to quantize
        fp8_dtype: FP8 datatype to quantize to
    """
    assert inpt_tensor.is_contiguous(), "Input tensor must be contiguous"

    out_tensor = torch.empty_like(inpt_tensor, dtype=fp8_dtype, device="cuda")
    numel = inpt_tensor.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["XBLOCK"]),)
    assert inpt_tensor.is_contiguous(), "Input tensor must be contiguous"
    tl_dtype = {torch.float8_e4m3fn: tl.float8e4nv, torch.float8_e5m2: tl.float8e5}[fp8_dtype]
    max_val = torch.finfo(fp8_dtype).max
    abs_max_scratch = torch.empty((), dtype=inpt_tensor.dtype, device="cuda")
    spin_lock = torch.zeros((), dtype=torch.int32, device="cuda")
    dynamic_scaled_cast[grid](
        inpt_tensor,
        out_tensor,
        abs_max_scratch,
        spin_lock,
        numel,
        16384,
        tl_dtype,
        max_val,
    )
    return out_tensor


def eager_dynamic_scaled_quant(
    a: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> torch.Tensor:
    """Quantize tensor to fp8 using the current amax value to generate scale
    Args:
        a: Input tensor to quantize
        fp8_dtype: FP8 datatype to quantize to
    """
    from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated

    scale = tensor_to_scale(a, fp8_dtype)
    tensor_scaled = a.to(torch.float32) * scale
    return to_fp8_saturated(tensor_scaled, fp8_dtype)
