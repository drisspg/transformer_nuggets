from typing import Tuple

import torch


def input_requires_grad(*tensors: Tuple[torch.Tensor]):
    return any(t.requires_grad for t in tensors)
