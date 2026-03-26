# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from transformer_nuggets.utils.benchmark import profiler, save_memory_snapshot
from pathlib import Path
from transformer_nuggets import init_logging
import torch.nn as nn

Tensor = torch.Tensor


class DeviceMOE(nn.Module):
    """Token Choice

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
        topk: int = 8,
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
        hidden_states: Tensor,
        renormalize: bool = False,
    ) -> Tensor:
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

        # Compute gating outputs (router logits) -> [num_tokens, num_experts]
        gating_output = self.router(hidden_states)

        # Get top-k experts and their weights
        topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
        topk_weights, selected_experts = topk_weights.topk(self.topk, dim=-1)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        topk_weights = topk_weights.to(dtype)

        permuted_tokens, expert_offsets, permuted_token_indices, dispatch_order = (
            self.local_dispatch(hidden_states, selected_experts)
        )
        # (TOPK * NUM_TOKENS) * HIDDEN_SIZE @ G * HIDDEN_SIZE * 2 * EXPERT_SIZE
        x = F.grouped_mm(permuted_tokens, self.w1.transpose(-1, -2), offs=expert_offsets)
        gate = F.silu(x[:, : self.intermediate_size])
        x = x[:, self.intermediate_size :] * gate
        x = F.grouped_mm(x, self.w2.transpose(-1, -2), offs=expert_offsets)
        # basically unpermutes outputs and puts tokens back in order, but naming for EP
        x = self.local_combine(x, dispatch_order, permuted_token_indices, topk_weights, num_tokens)

        return x.view(orig_shape)

    def local_dispatch(
        self, x: Tensor, selected_experts: Tensor
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Permute tokens to experts.

        Args:
            x: Input tensor of shape [num_tokens, hidden_size]
            selected_experts: Tensor of shape [num_tokens, topk] containing selected experts

        Returns:
            permuted_x: Permutated input tensor of shape [top_k * num_tokens, hidden_size
            expert_counts: Tensor of shape [num_experts] containing number of tokens per expert
            permuted_token_indices: Tensor of shape [num_tokens * topk] containing permuted token indices
            dispatch_order: Tensor of shape [num_tokens * topk] containing dispatch order


        """
        num_tokens, _ = x.shape
        flat_token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.topk)

        # Figure out how many tokens are assigned to each expert and create offsets
        flat_experts = selected_experts.reshape(-1)
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.int32)
        expert_counts.scatter_add_(
            0, flat_experts, torch.ones_like(flat_experts, dtype=torch.int32)
        )
        expert_offsets = torch.cumsum(expert_counts, dim=0, dtype=torch.int32)
        # Sort the tokens by their expert index lower to high expert number
        _, dispatch_order = torch.sort(flat_experts, stable=True)
        # build a map from token id to its new position
        permuted_token_indices = flat_token_indices[dispatch_order]
        # slice original with peremuted indices [0, num_tokens), this essetnially scatters and duplicate the tokens
        permuted_x = x[permuted_token_indices]

        return permuted_x, expert_offsets, permuted_token_indices, dispatch_order

    def local_combine(
        self,
        x: Tensor,
        dispatch_order: Tensor,
        permuted_token_indices: Tensor,
        topk_weights: Tensor,
        num_tokens: int,
    ) -> Tensor:
        """Combine tokens from experts, basically permute weight scores, and puts tokens back in order.

        Args:
            x: Input tensor of shape [topk * num_tokens, hidden_size]
            expert_offsets: Tensor of shape [num_experts] containing number of tokens per expert
            permuted_token_indices: Tensor of shape [num_tokens * topk] containing permuted token indices
            topk_weights: Tensor of shape [num_tokens, topk] containing topk weights

        Returns:
            Tensor of shape [num_tokens, hidden_size]
        """
        flat_weights = topk_weights.reshape(-1)

        # lets swizzle up the weights to match the dispatch order
        permuted_weights = flat_weights[dispatch_order]

        # now we need to scatter the weights back to their original positions multiplying by the router weights
        final = torch.zeros(num_tokens, self.hidden_size, dtype=x.dtype, device=x.device)
        final.index_add_(
            0,
            permuted_token_indices,
            x * permuted_weights.unsqueeze(-1),
        )
        return final


@torch.no_grad()
def main():
    init_logging()
    hidden_size = 7168
    intermediate_size = 2048
    topk = 8
    num_experts = 128

    batch_size, num_tokens = 2, 2

    def make_tensor(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    # Create inputs
    hidden_states = make_tensor(batch_size, num_tokens, hidden_size)

    # Create MoE module
    moe = DeviceMOE(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
    ).to(dtype=torch.bfloat16, device="cuda")

    # Warmup
    for i in range(10):
        output = moe(hidden_states)

    torch.cuda.synchronize()

    with profiler(Path("/tmp/DeviceMOE")):
        output = moe(hidden_states)
    torch.cuda.synchronize()

    # Create CUDA graph
    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        _ = moe(hidden_states)

    with profiler(Path("/tmp/device_moe_cuda_graph")):
        g.replay()
        torch.cuda.synchronize()

    print(output.shape)

    # Memory View of eager
    # Go smaller
    hidden_size = 7168
    intermediate_size = 2048
    topk = 2
    num_experts = 8
    with save_memory_snapshot(Path("/tmp/device_moe_eager")):
        small_moe = DeviceMOE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            topk=topk,
        ).to(dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        _ = small_moe(hidden_states)


if __name__ == "__main__":
    main()
