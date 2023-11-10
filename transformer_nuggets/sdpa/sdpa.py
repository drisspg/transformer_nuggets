from typing import Optional
from warnings import warn

import torch
from torch.nn.functional import scaled_dot_product_attention
from transformer_nuggets.sdpa.attn_bias import AttnBias


def sdpa_prototype(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[AttnBias] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    assert attn_mask is None or isinstance(attn_mask, (AttnBias, torch.Tensor))

    if isinstance(attn_mask, torch.Tensor):
        warn("Passing a tensor as an attn_mask is deprecated. Please use TensorBias instead.")
        return scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)

    if attn_mask is None or attn_mask.needs_materialization():
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask.materialize(query.device) if attn_mask else None,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    # After this point all AttnBias are required to have defined their own dispatching logic
    return attn_mask.dispatch(query, key, value, attn_mask, dropout_p, is_causal, scale)
