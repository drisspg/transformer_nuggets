from typing import Optional

import torch
from torch.nn.functional import scaled_dot_product_attention
from transformer_nuggets.sdpa.attn_mask import (
    AttnMask,
    CausalMask,
    CausalVariant,
    LambdaMask,
    TensorMask,
)


def input_requires_grad(*tensors):
    return any([t.requires_grad for t in tensors])


def dispatch_causal_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[AttnMask],
    causal: bool,
    scale: Optional[float],
    dropout_p: float,
):
    if attn_mask.seq_len_q == attn_mask.seq_len_kv:
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
            scale=scale,
        )
    if attn_mask.variant == CausalVariant.UPPER_LEFT:
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
            scale=scale,
        )
    elif attn_mask.variant == CausalVariant.LOWER_RIGHT:
        # Figure out how to use the sdp_utils to verify this is okay to run?
        compute_log_sumexp = False
        if input_requires_grad(query, key, value):
            compute_log_sumexp = True
        print("running efficient attention")
        return torch._efficient_attention_forward(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
            scale=scale,
            compute_log_sumexp=compute_log_sumexp,
            custom_mask_type=attn_mask.variant,
        )[0]
    else:
        raise ValueError("Invalid causal variant")


def sdpa_prototype(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[AttnMask] = None,
    causal: bool = False,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
):
    assert attn_mask is None or isinstance(attn_mask, AttnMask)

    if not attn_mask:
        return scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=causal, scale=scale
        )

    if attn_mask.needs_materialization():
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask.materialize(query.device),
            dropout_p=dropout_p,
            is_causal=causal,
            scale=scale,
        )
    # I think we should be able to define dispatch logic on mask types...
    if isinstance(attn_mask, CausalMask):
        return dispatch_causal_mask(query, key, value, attn_mask, causal, scale, dropout_p)

    raise ValueError("slow down chief")
