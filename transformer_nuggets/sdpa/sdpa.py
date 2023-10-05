from typing import Optional
from warnings import warn

import torch
from torch.nn.functional import scaled_dot_product_attention
from transformer_nuggets.sdpa.attn_mask import AttnMask


def sdpa_prototype(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[AttnMask] = None,
    causal: bool = False,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
):
    assert attn_mask is None or isinstance(attn_mask, (AttnMask, torch.Tensor))

    if isinstance(attn_mask, torch.Tensor):
        warn("Passing a tensor as an attn_mask is deprecated. Please use TensorMask instead.")
        return scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, causal, scale)

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
    # After this point all AttnMask are required to have defined their own dispatching logic
    return attn_mask.dispatch(query, key, value, attn_mask, causal, scale, dropout_p)
