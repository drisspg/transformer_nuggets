""" Define Base class as well as some crowd favorites """
from typing import Optional

import torch
from torch.utils import _pytree as pytree
from transformer_nuggets.sdpa.attn_bias import AttnBias, CausalBias, TensorBias


def materialize_if_needed(bias: "AttnBias", device: Optional[torch.device] = None) -> torch.Tensor:
    if bias.needs_materialization():
        return bias.materialize(device)
    return bias


class TensorBiasSubclass(TensorBias):
    """A bias that is a tensor"""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func != torch.nn.functional.scaled_dot_product_attention:
            return NotImplemented
        args = pytree.tree_map_only(TensorBias, lambda x: materialize_if_needed(x), args)
        kwargs = pytree.tree_map_only(TensorBias, lambda x: materialize_if_needed(x), kwargs)
        return func(*args, **kwargs)


class CausalBiasSubclass(CausalBias):
    """A bias representing causal attention patterns"""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func != torch.nn.functional.scaled_dot_product_attention:
            return NotImplemented
        return cls.dispatch(*args, **kwargs)
