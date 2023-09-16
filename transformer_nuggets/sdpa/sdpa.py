from typing import Optional

import torch
from torch.nn.functional import scaled_dot_product_attention
from transformer_nuggets.sdpa.attn_mask import (AttnMask, CausalMask,
                                                CausalVariant, LambdaMask,
                                                TensorMask)


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
        return scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=causal, scale=scale)
    else:
        if attn_mask.needs_materialization():
            return scaled_dot_product_attention(query, key, value, attn_mask=attn_mask.materialize(), dropout_p=dropout_p, is_causal=causal, scale=scale)
        else:
            raise("slow down chief")