import torch
from dataclasses import dataclass

from typing import Tuple

@dataclass(frozen=True)
class SubclassTensorArgs:
    original_shape: torch.Size
    original_strides: Tuple
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "SubclassTensorArgs":
        return SubclassTensorArgs(
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
        )
