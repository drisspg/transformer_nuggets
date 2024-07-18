"""Shows how you can use visualize_attention_scores to explore score_mod functions to be use with flex_attention"""

import torch
from functools import partial
from transformer_nuggets.flex import visualize_attention_scores


def relative_positional(score, b, h, q_idx, kv_idx):
    scale = 1 / HEAD_DIM
    bias = (q_idx - kv_idx) * scale
    return score + bias


def checkerboard(score, batch, head, q_idx, kv_idx):
    score = torch.where(torch.abs(kv_idx - q_idx) % 1 == 0, score * 0.5, score)
    score = torch.where(torch.abs(kv_idx - q_idx) % 2 == 0, score * 2.0, score)
    return score


SLIDING_WINDOW = 2


def sliding_window_causal(score, b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW
    return torch.where(causal_mask & window_mask, score, -float("inf"))


if __name__ == "__main__":
    B, H, SEQ_LEN, HEAD_DIM = 2, 2, 6, 64
    make_tensor = partial(torch.ones, B, H, SEQ_LEN, HEAD_DIM, device="cuda")
    query, key, value = make_tensor(), make_tensor(), make_tensor()

    visualize_attention_scores(
        query,
        key,
        relative_positional,
        name="relative_positional",
    )
    visualize_attention_scores(query, key, checkerboard, name="checkerboard")
    visualize_attention_scores(query, key, sliding_window_causal, name="sliding_window_causal")