import torch
from typing import Union, Callable
import matplotlib.pyplot as plt
from torch.nn.attention._flex_attention import (
    _score_mod_signature,
    _mask_fn_signature,
    _vmap_for_bhqkv,
)
from torch._higher_order_ops.flex_attention import TransformGetItemToIndex
from contextlib import nullcontext

Tensor = torch.Tensor


def create_score_mod(
    query: torch.Tensor,
    key: torch.Tensor,
    mod_fn: Union[_score_mod_signature, _mask_fn_signature],
    device: str = "cuda",
    _compile: bool = False,
) -> torch.Tensor:
    (
        B,
        H,
    ) = (
        1,
        1,
    )
    M = query.shape[0]
    N = key.shape[0]

    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)

    if _compile:
        ctx = nullcontext()
    else:
        ctx = TransformGetItemToIndex()

    with ctx:
        score_mod = _vmap_for_bhqkv(mod_fn, prefix=(0,))
        scores = query @ key.transpose(-2, -1)
        scores = scores.view(1, 1, M, N)
        out = score_mod(scores, b, h, m, n)

    return out


def visualize_attention_scores(
    query: Tensor,
    key: Tensor,
    mod_fn: Callable,
    device: str = "cuda",
    filename: str = "attention_scores.png",
    title: str = "Attention Scores Visualization",
    batch_idx: int = 0,
    head_idx: int = 0,
):
    """
    Generate and save a visualization of attention scores.

    Args:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        mod_fn (Callable): The score modification function.
        device (str): Device to run computations on (default: "cuda").
        filename (str): Name of the file to save the visualization (default: 'attention_scores.png').
        title (str): Title for the visualization (default: "Attention Scores Visualization").
        batch_idx (int): Index of the batch to visualize (default: 0).
        head_idx (int): Index of the head to visualize (default: 0).

    Returns:
        None
    """
    query = query[batch_idx, head_idx, :, :]
    key = key[batch_idx, head_idx, :, :]
    scores_viz = create_score_mod(query, key, mod_fn, device=device)

    plt.figure(figsize=(10, 8))
    plt.matshow(scores_viz.cpu().detach()[0, 0, :, :])
    plt.colorbar()
    plt.title(f"{title}\nBatch {batch_idx}, Head {head_idx}")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free up memory

    print(f"Visualization saved as {filename}")
