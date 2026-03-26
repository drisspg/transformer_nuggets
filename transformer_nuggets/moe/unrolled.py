# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from transformer_nuggets.utils.benchmark import profiler, save_memory_snapshot
from pathlib import Path
from transformer_nuggets import init_logging
import torch.nn as nn


class LoopMoE(nn.Module):
    """
    Reference token-choice Mixture of Experts module.

    This implementation materializes routed token-expert assignments on each
    forward pass, so it is not CUDA-graph compatible.

    Args:
        hidden_size: Size of the input hidden states
        intermediate_size: Size of the intermediate representation
        num_experts: Number of experts in the model
        topk: Number of top experts to select for each token
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        topk: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk

        # Initialize expert weights
        # W1 for up-projection with gating: [num_experts, intermediate_size * 2, hidden_size]
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size * 2, hidden_size))
        # W2 for down-projection: [num_experts, hidden_size, intermediate_size]
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

        # Router (gate) for selecting experts
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with standard distribution."""
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        renormalize: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the MoE layer.

        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            renormalize: Whether to renormalize weights after top-k selection

        Returns:
            Tensor of shape [..., hidden_size]
        """
        orig_shape = hidden_states.shape
        hidden_size = hidden_states.shape[-1]

        num_tokens = (hidden_states.shape[:-1]).numel()
        dtype = hidden_states.dtype

        # Splat batch of tokens
        hidden_states = hidden_states.view(num_tokens, hidden_size)

        # Compute gating outputs (router logits)
        gating_output = self.router(hidden_states)

        # Get top-k experts and their weights
        topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
        topk_weights, selected_experts = topk_weights.topk(self.topk, dim=-1)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        topk_weights = topk_weights.to(dtype)

        flat_token_indices = torch.arange(
            num_tokens, device=hidden_states.device
        ).repeat_interleave(self.topk)
        flat_experts = selected_experts.reshape(-1)
        flat_topk_weights = topk_weights.reshape(-1)

        routed_hidden_states = hidden_states.index_select(0, flat_token_indices)
        routed_w1 = self.w1.index_select(0, flat_experts)
        x = torch.bmm(
            routed_hidden_states.unsqueeze(1),
            routed_w1.transpose(-1, -2),
        ).squeeze(1)
        gate = F.silu(x[:, : self.intermediate_size])
        x = x[:, self.intermediate_size :] * gate

        routed_w2 = self.w2.index_select(0, flat_experts)
        x = torch.bmm(
            x.unsqueeze(1),
            routed_w2.transpose(-1, -2),
        ).squeeze(1)

        final_hidden_states = torch.zeros(
            num_tokens,
            hidden_size,
            dtype=x.dtype,
            device=x.device,
        )
        final_hidden_states.index_add_(
            0,
            flat_token_indices,
            x * flat_topk_weights.unsqueeze(-1),
        )
        return final_hidden_states.view(orig_shape)


@torch.no_grad()
def main():
    init_logging()
    hidden_size = 7168
    intermediate_size = 2048
    topk = 8
    num_experts = 128

    batch_size, num_tokens = 2, 2

    def make_tensor(*shape):
        return torch.randn(shape, dtype=torch.float16, device="cuda")

    # Create inputs
    hidden_states = make_tensor(batch_size, num_tokens, hidden_size)

    # Create MoE module
    moe = LoopMoE(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
    ).to(dtype=torch.float16, device="cuda")

    # Warmup
    for i in range(10):
        output = moe(hidden_states)

    torch.cuda.synchronize()

    with profiler(Path("/tmp/loop_moe")):
        output = moe(hidden_states)
    torch.cuda.synchronize()

    print(output.shape)

    # Memory View of eager
    # Go smaller
    hidden_size = 7168
    intermediate_size = 2048
    topk = 2
    num_experts = 8
    with save_memory_snapshot(Path("/tmp/loop_moe_eager")):
        small_moe = LoopMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            topk=topk,
        ).to(dtype=torch.float16, device="cuda")
        torch.cuda.synchronize()
        _ = small_moe(hidden_states)


if __name__ == "__main__":
    main()
