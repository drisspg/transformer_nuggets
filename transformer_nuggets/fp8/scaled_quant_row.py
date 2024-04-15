import torch

from float8_experimental.float8_utils import to_fp8_saturated


def get_row_scales(input_tensor, float8_dtype: torch.dtype) -> torch.Tensor:
    row_maxes = torch.max(input_tensor, dim=1).values
    row_scales = (torch.finfo(float8_dtype).max) / torch.clamp(row_maxes, min=1e-12)
    return row_scales


def row_wise_quant(inpt_tensor: torch.Tensor) -> torch.Tensor:
    """
    Row-wise scaling of a 2D tensor
    Args:
        inpt_tensor: 2D tensor of shape (N, D)
    """
    assert inpt_tensor.ndim == 2, "Input tensor must be 2D"

    row_scales = get_row_scales(inpt_tensor, torch.float8_e4m3fn)
    row_scales = row_scales.unsqueeze(-1)
    scaled_tensor = inpt_tensor * row_scales.unsqueeze(-1)
    return to_fp8_saturated(scaled_tensor, torch.float8_e4m3fn)
