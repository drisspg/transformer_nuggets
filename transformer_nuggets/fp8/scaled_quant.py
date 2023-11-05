import torch
import triton


def scaled_quant(
    inpt_tensor: torch.Tensor,
    scale: torch.Tensor,
    abs_max: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
):
    """Quantize tensor to fp8 using a delayed scaled and calculate abs_max"""

    out_tensor = torch.empty_like(inpt_tensor, dtype=torch.float8_e4m3fn, device="cuda")
