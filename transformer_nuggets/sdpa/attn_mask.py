""" Lets define some things """
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.backends.cuda import 
from transformer_nuggets.sdpa.utils import input_requires_grad



class AttnMask(ABC):
    """Abstract base class for attention masks"""

    @abstractmethod
    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        raise NotImplementedError("This is an abstract base class")

    @abstractmethod
    def needs_materialization(self) -> bool:
        raise NotImplementedError("This is an abstract base class")

    @staticmethod
    @abstractmethod
    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "AttnMask" = None,
        causal: bool = False,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ):
        raise NotImplementedError("This is an abstract base class")


class TensorMask(AttnMask):
    """A mask that is a tensor"""

    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.mask

    def needs_materialization(self) -> bool:
        return True

    @staticmethod
    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "TensorMask" = None,
        causal: bool = False,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ):
        raise NotImplementedError(
            "TensorMask requires materialization, so this should never be called!"
        )


class LambdaMask(AttnMask):
    """A mask that is a function"""

    def __init__(self, mask_fn):
        self.mask_fn = mask_fn

    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.mask_fn()

    def needs_materialization(self) -> bool:
        return False

    @staticmethod
    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "LambdaMask" = None,
        causal: bool = False,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ):
        raise NotImplementedError("TODO FIGURE OUT!")


class CausalVariant(Enum):
    """Enum for causal causal varients"""

    UPPER_LEFT = 1
    LOWER_RIGHT = 2


class CausalMask(TensorMask):
    """A mask representing causal attention patterns"""

    def __init__(self, variant: CausalVariant, seq_len_q: int, seq_len_kv: int):
        assert isinstance(variant, CausalVariant)
        self.variant = variant
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv

    def _upper_left(self, device: torch.device) -> torch.Tensor:
        """Upper left causal mask"""
        return torch.tril(
            torch.ones(self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool)
        )

    def _lower_right(self, device: torch.device) -> torch.Tensor:
        """Lower right causal mask"""
        diagonal_offset = self.seq_len_kv - self.seq_len_q
        return torch.tril(
            torch.ones(self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")
        if self.variant == CausalVariant.UPPER_LEFT:
            return self._upper_left(device)
        elif self.variant == CausalVariant.LOWER_RIGHT:
            return self._lower_right(device)

    def needs_materialization(self) -> bool:
        return False

    @staticmethod
    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "CausalMask" = None,
        causal: bool = False,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
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
