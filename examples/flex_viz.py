"""
Shows how you can use visualize_attention_scores to explore score_mod functions
to be used with flex_attention.
"""

import torch
from transformer_nuggets.flex import visualize_attention_scores

# Constants
B, H, SEQ_LEN, HEAD_DIM = 2, 2, 6, 64
SLIDING_WINDOW = 2


# Helper function
def make_tensor():
    return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device="cuda")


# Score modification functions
def relative_positional(score, b, h, q_idx, kv_idx):
    scale = 1 / HEAD_DIM
    bias = (q_idx - kv_idx) * scale
    return score + bias


def checkerboard(score, batch, head, q_idx, kv_idx):
    score = torch.where(torch.abs(kv_idx - q_idx) % 1 == 0, score * 0.5, score)
    score = torch.where(torch.abs(kv_idx - q_idx) % 2 == 0, score * 2.0, score)
    return score


# Mask modification functions
def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW
    return causal_mask & window_mask


def create_doc_mask_func(document_id):
    def doc_mask_wrapper(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    return doc_mask_wrapper


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


# Main execution
def main():
    query, key = make_tensor(), make_tensor()

    visualize_attention_scores(
        query,
        key,
        relative_positional,
        name="relative_positional",
    )

    visualize_attention_scores(query, key, score_mod=checkerboard, name="checkerboard")

    visualize_attention_scores(
        query, key, mask_mod=sliding_window_causal, name="sliding_window_causal"
    )

    # Example usage:
    document_id = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2], dtype=torch.int32, device="cuda")

    doc_mask = create_doc_mask_func(document_id)

    visualize_attention_scores(
        torch.ones(B, H, document_id.numel(), HEAD_DIM, device="cuda"),
        torch.ones(B, H, document_id.numel(), HEAD_DIM, device="cuda"),
        mask_mod=doc_mask,
        name="document_mask",
    )

    visualize_attention_scores(query, key, mask_mod=causal_mask, name="causal_mask")


if __name__ == "__main__":
    main()
