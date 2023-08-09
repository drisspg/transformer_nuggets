import torch
import triton
import triton.language as tl


def build_causal_mask(seq_len_q, seq_len_kv):
    temp_mask = torch.ones((seq_len_q, seq_len_kv)).tril_().bool()
    mask = torch.zeros_like(temp_mask, dtype=torch.float32)
    mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    return mask


def build_alibi_mask(n_queries, n_keys, n_heads, scale=None, causal=True):
    if scale is None:
        assert n_heads % 8 == 0
    m_0 = 2.0 ** (-8.0 / n_heads)
    slopes = torch.pow(m_0, torch.arange(1, 1 + n_heads))[:, None, None]
    base = -1 * (torch.arange(n_queries)[:, None] - torch.arange(n_keys)[None, :])
    if scale is not None:
        alibi_base = base * scale
    else:
        alibi_base = base * slopes
    alibi_base = alibi_base.expand(n_heads, n_queries, n_keys)
    if causal:
        causal_mask = build_causal_mask(n_queries, n_keys)
        causal_mask = causal_mask.expand(n_heads, n_queries, n_keys)
        full_mask = alibi_base + causal_mask
    else:
        full_mask = alibi_base
    return full_mask


@triton.jit
def rel_attention_triton(cur, m, n, head_num, num_heads):
    bias = n - m
    cur = cur + bias
    return cur


@triton.jit
def alibi_attention_triton(cur, m, n, head_num, num_heads):
    # 0 Indexing
    alibi_scale = tl.math.exp2(-((head_num + 1) * 8.0 / num_heads))
    bias = n - m
    cur = cur + (alibi_scale * bias)
    return cur


@triton.jit
def causal_mask_triton(cur, m, n, head_num, num_heads):
    cur = tl.where(m >= n, cur, float("-inf"))
    return cur


@triton.jit
def inverse_causal_mask_triton(cur, m, n, head_num, num_heads):
    cur = tl.where(m > n, float("-inf"), cur)
    return cur
