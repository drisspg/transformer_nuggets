import enum

import torch

import triton
import triton.language as tl


class BiasMode(enum.Enum):
    none = 0
    rel_pos = 1
    alibi = 2
    inverse_causal = 3
    causal = 4


def build_causal_mask(seq_len_q, seq_len_kv):
    temp_mask = torch.ones((seq_len_q, seq_len_kv)).tril_().bool()
    mask = torch.zeros_like(temp_mask, dtype=torch.float32)
    mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    return mask


def build_rel_mask(
    n_queries: int,
    n_keys: int,
    n_heads: int,
    mode: BiasMode,
    causal: bool,
):
    """Builds torch equivalent mask
    Args:
        n_queries: Number of queries.
        n_keys: Number of keys.
        n_heads: Number of attention heads.
        mode: Bias mode for the attention mask.
        causal: Whether to include causal mask. Defaults to True.

    Returns:
        torch.Tensor: The alibi attention mask.
    """
    if mode == BiasMode.alibi:
        assert n_heads % 8 == 0
    m_0 = 2.0 ** (-8.0 / n_heads)
    slopes = torch.pow(m_0, torch.arange(1, 1 + n_heads))[:, None, None]
    base = -1 * (torch.arange(n_queries)[:, None] - torch.arange(n_keys)[None, :])
    mask = base
    mask = mask * slopes if mode == BiasMode.alibi else mask
    mask = mask.expand(n_heads, n_queries, n_keys)
    if causal:
        causal_mask = build_causal_mask(n_queries, n_keys)
        causal_mask = causal_mask.expand(n_heads, n_queries, n_keys)
        full_mask = mask + causal_mask
    else:
        full_mask = mask
    return full_mask


@triton.jit
def rel_attention_triton(score, batch, head, seq_len_q, seq_len_kv):
    bias = seq_len_kv - seq_len_q
    score = score + bias
    return score


@triton.jit
def alibi_attention_triton(score, batch, head, seq_len_q, seq_len_kv, num_heads):
    # 0 Indexing
    alibi_scale = tl.math.exp2(-((head + 1) * 8.0 / num_heads))
    bias = seq_len_kv - seq_len_q
    score = score + (alibi_scale * bias)
    return score


@triton.jit
def causal_mask_triton(score, batch, head, seq_len_q, seq_len_kv):
    score = tl.where(seq_len_q >= seq_len_kv, score, float("-inf"))
    return score


@triton.jit
def inverse_causal_mask_triton(score, batch, head, seq_len_q, seq_len_kv):
    score = tl.where(seq_len_q > seq_len_kv, float("-inf"), score)
    return score


@triton.jit
def score_modification(
    score,
    offs_m,
    start_n,
    offs_n,
    off_hz,
    num_heads,
    q,
    k,
    mask_block_ptr,
    BIAS_CHOICE: tl.constexpr,
    DEBUG_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MATMUL_PRECISION: tl.constexpr = tl.float16,
):
    batch = off_hz // num_heads
    head = off_hz % num_heads
    seq_len_q = offs_m[:, None]
    seq_len_kv = start_n + offs_n[None, :]
    if BIAS_CHOICE == BiasMode.rel_pos.value:
        score = rel_attention_triton(score, batch, head, seq_len_q, seq_len_kv)
    elif BIAS_CHOICE == BiasMode.alibi.value:
        score = alibi_attention_triton(score, batch, head, seq_len_q, seq_len_kv, num_heads)
    elif BIAS_CHOICE == BiasMode.inverse_causal.value:
        score = inverse_causal_mask_triton(score, batch, head, seq_len_q, seq_len_kv)
    elif BIAS_CHOICE == BiasMode.causal.value:
        # CAUSAL MASK
        score = causal_mask_triton(score, batch, head, seq_len_q, seq_len_kv)
    if DEBUG_MASK and BIAS_CHOICE != BiasMode.none:
        mask = score - tl.dot(q.to(MATMUL_PRECISION), k.to(MATMUL_PRECISION))
        # if IS_CAUSAL:
        #     mask = tl.where(seq_len_q >= seq_len_kv, mask, float("-inf"))
        tl.store(mask_block_ptr, mask)

    return score
