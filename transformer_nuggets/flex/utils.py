import torch
from typing import Union, Callable, Optional
import matplotlib.pyplot as plt
from pathlib import Path
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


def _name_to_title(name: str) -> str:
    title = name.replace("_", " ")
    title = " ".join(word.capitalize() for word in title.split())
    return title


def visualize_attention_scores(
    query: Tensor,
    key: Tensor,
    mod_fn: Callable,
    device: str = "cuda",
    name: str = "attention_scores",
    path: Optional[Path] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
):
    """
    Generate and save a visualization of attention scores.

    Args:
        query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        mod_fn (Callable): The score modification function.
        device (str): Device to run computations on (default: "cuda").
        name (str): Base name for the file and title (default: 'attention_scores').
        path (Path): Path to save the visualization. If None, will be saved to the current working directory.
        batch_idx (int): Index of the batch to visualize (default: 0).
        head_idx (int): Index of the head to visualize (default: 0).

    Returns:
        None
    """
    query = query[batch_idx, head_idx, :, :]
    key = key[batch_idx, head_idx, :, :]
    scores_viz = create_score_mod(query, key, mod_fn, device=device)

    suffix_title = (
        "" if batch_idx == 0 and head_idx == 0 else f"Batch {batch_idx}, Head {head_idx}"
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap="viridis")
    fig.colorbar(im)

    title = _name_to_title(name)
    file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
    ax.set_title(f"{title}\n{suffix_title}")

    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")

    # Move y-axis ticks and labels to the top
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Add tick labels if the number of tokens is manageable
    num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
    if num_query_tokens <= 32 and num_kv_tokens <= 32:
        ax.set_xticks(range(num_kv_tokens))
        ax.set_xticklabels([f"KV{i}" for i in range(num_kv_tokens)])
        ax.set_yticks(range(num_query_tokens))
        ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)])

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory

    print(f"Visualization saved as {file_path}")
