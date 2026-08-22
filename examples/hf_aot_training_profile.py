"""Profile eager Hugging Face GPT-2 forward and backward FX graphs.

Install the optional dependency with ``pip install transformer-nuggets[huggingface]``.
"""

from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_nuggets.fx_analysis import profile_aot_training


def main() -> None:
    """Create a small GPT-2 training workload and emit an annotated trace."""
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA")

    model = (
        GPT2LMHeadModel(
            GPT2Config(
                vocab_size=256,
                n_embd=64,
                n_layer=2,
                n_head=4,
                n_positions=64,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
        .cuda()
        .train()
    )
    input_ids = torch.randint(0, 256, (2, 32), device="cuda")
    labels = torch.randint(0, 256, (2, 32), device="cuda")

    result = profile_aot_training(
        model,
        (input_ids,),
        Path("artifacts/hf_gpt2_training.pftrace"),
        kwargs={"labels": labels},
        loss_selector=lambda output: output.loss,
    )

    print(
        "Forward regions:",
        sum(len(analysis.regions) for analysis in result.analysis.forward),
    )
    print(
        "Backward regions:",
        sum(len(analysis.regions) for analysis in result.analysis.backward),
    )
    print(f"Wrote {result.trace_path.resolve()}")


if __name__ == "__main__":
    main()
