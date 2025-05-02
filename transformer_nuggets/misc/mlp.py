import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import torch

Tensor = torch.Tensor


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.dim = dim

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

    def get_input(
        self, num_tokens: int, device: str = "cuda", dtype: torch.dtype = torch.bfloat16
    ) -> Tensor:
        return torch.randn((num_tokens, self.dim), device=device, dtype=dtype)

    @classmethod
    def llama3_mlp(cls, flavor: Literal["8B", "70B", "405B"] = "8B"):
        arg_map = {
            "8B": dict(
                dim=4096,
                hidden_dim=14336,
                multiple_of=1024,
                ffn_dim_multiplier=1.3,
            ),
            "70B": dict(
                dim=8192,
                hidden_dim=28672,
                multiple_of=4096,
                ffn_dim_multiplier=1.3,
            ),
            "405B": dict(
                dim=16384,
                hidden_dim=57344,
                multiple_of=4096,
                ffn_dim_multiplier=1.2,
            ),
        }
        return cls(**arg_map[flavor])
