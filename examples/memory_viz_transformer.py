"""
Example: 8-layer transformer memory visualization with D3.

Creates an AttentionStack (8 RopeAttention layers), runs a forward + backward
pass on GPU, and captures a memory snapshot rendered as an interactive D3 HTML.

Usage:
    python examples/memory_viz_transformer.py
    python examples/memory_viz_transformer.py --batch_size 2 --seq_len 1024
"""

import torch
from pathlib import Path
from jsonargparse import CLI

from transformer_nuggets.misc.attention import AttentionStack
from transformer_nuggets.utils.memory_viz import generate_memory_html


def main(
    batch_size: int = 2,
    seq_len: int = 512,
    dim: int = 1024,
    num_heads: int = 8,
    num_layers: int = 8,
    output: str = "data/transformer_memory_viz.html",
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="all")

    model = AttentionStack(
        num_layers=num_layers,
        dim=dim,
        num_heads=num_heads,
        backend="sdpa",
        causal=True,
    ).to("cuda", torch.bfloat16)

    x, block_mask = model.get_input(batch_size=batch_size, seq_len=seq_len)

    model(x, block_mask=block_mask)
    torch.cuda.synchronize()

    out = model(x, block_mask=block_mask)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()

    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    html = generate_memory_html(snapshot, title=output_path.stem)
    output_path.write_text(html)
    print(f"D3 memory visualization saved to: {output_path}")


if __name__ == "__main__":
    CLI(main, as_positional=False)
