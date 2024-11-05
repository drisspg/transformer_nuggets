import torch
import triton
import triton.language as tl

from transformer_nuggets.quant.nf4_tensor import NF4Tensor


@triton.jit
def dequantize(inputs, nf4_lut):
    """Dequantizes the nf4 data to bfloat16"""
    return tl.load(nf4_lut + inputs)


@triton.jit
def dequantize_scalers(
    quantized_scalers_ptr,
    quantization_factor_ptr,
    scaler_mean_ptr,
    block_size,
    scaler_block_size,
):
    """Dequantizes the quantized scalers to bfloat16
    Args:
        quantized_scalers_ptr: Pointer to the quantized scalers
        quantization_factor_ptr: Pointer to the quantization factor
        scaler_mean_ptr: Pointer to the scaler mean
        block_size: Size of the block
        scaler_block_size: Size of the scaler block
    """
    block_idx = tl.program_id(0)
    quantization_factor_idx = block_idx // scaler_block_size

    # # Load the quantization factor for the given block
    scaler_quantization_factor = tl.load(quantization_factor_ptr + quantization_factor_idx)

    # # Load the quantized block scaler
    block_scaler = tl.load(quantized_scalers_ptr + block_idx)

    # # Load the scaler mean
    scaler_mean = tl.load(scaler_mean_ptr)

    dequantized_block_scaler = (block_scaler / scaler_quantization_factor).to(tl.bfloat16)
    dequantized_block_scaler = dequantized_block_scaler + scaler_mean

    return dequantized_block_scaler


@triton.jit
def dequant_nf4_tensor_kernel(
    inpt_ptr,
    output_ptr,
    quantized_scalers_ptr,
    quantization_factor_ptr,
    scaler_mean_ptr,
    nf4_lut_ptr,
    scaler_block_size: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    """Dequantizes a tensor from nf4 to bfloat16"""
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]

    index = tl.max_contiguous(tl.multiple_of(index, XBLOCK), XBLOCK)
    # Load packed, quantized data, no need to mask
    inpt = tl.load(inpt_ptr + index)
    first_elements = inpt >> 4
    second_elements = inpt & 0xF

    # Dequantize the nf4 data
    dequantized_first = dequantize(first_elements, nf4_lut_ptr)
    dequantized_second = dequantize(second_elements, nf4_lut_ptr)

    # Dequantize the double quantized scalers
    block_scaler = dequantize_scalers(
        quantized_scalers_ptr,
        quantization_factor_ptr,
        scaler_mean_ptr,
        XBLOCK,
        scaler_block_size,
    )

    scaled_first = dequantized_first * block_scaler
    scaled_second = dequantized_second * block_scaler

    #  Lets hope this function stays 🤞
    store_indices = offset * 2 + tl.arange(0, XBLOCK * 2)[:]
    interleaved = tl.interleave(scaled_first, scaled_second)
    tl.store(output_ptr + store_indices, interleaved)


def dequant_nf4_tensor(weight: NF4Tensor):
    """Takes a quantized tensor and dequantizes it to bfloat16"""
    assert isinstance(weight, NF4Tensor), "Input tensor must be of type NF4Tensor"
    assert weight.shape.numel() % weight.block_size == 0, (
        "Input tensor must be a multiple of block size"
    )
    out_tensor = torch.empty(weight.shape, dtype=weight.dtype, device="cuda")
    numel = weight.shape.numel()
    grid = (triton.cdiv(numel, (weight.block_size)),)

    dequant_nf4_tensor_kernel[grid](
        weight.quantized_data,
        out_tensor,
        weight.quantized_scalers,  # in int8
        weight.quantization_factor,
        weight.scaler_mean,
        weight.nf4,
        weight.scaler_block_size,
        weight.block_size // 2,  # Each block is responsible for 2 output elements
    )

    return out_tensor
