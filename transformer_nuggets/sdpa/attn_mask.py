""" Define Base class as well as some crowd favorites """
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional
from warnings import warn

import torch
from torch.backends.cuda import SDPAParams, can_use_efficient_attention
from torch.nn.functional import scaled_dot_product_attention
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
        attn_mask: "AttnMask",
        causal: bool,
        scale: Optional[float],
        dropout_p: float,
    ) -> torch.Tensor:
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
        attn_mask: "TensorMask",
        causal: bool,
        scale: Optional[float],
        dropout_p: float,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "TensorMask requires materialization, so this should never be called!"
        )

    def __repr__(self) -> str:
        return f"TensorMask(mask={self.mask})"


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
        attn_mask: "LambdaMask",
        causal: bool,
        scale: Optional[float],
        dropout_p: float,
    ) -> torch.Tensor:
        raise NotImplementedError("TODO FIGURE OUT!")


class CausalVariant(IntEnum):
    """Enum for causal variants"""

    UPPER_LEFT = 1
    LOWER_RIGHT = 2


class CausalMask(TensorMask):
    """A mask representing causal attention patterns"""

    def __init__(self, variant: CausalVariant, seq_len_q: int, seq_len_kv: int):
        assert isinstance(variant, CausalVariant)
        self.variant = variant
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        if seq_len_q > seq_len_kv and variant == CausalVariant.LOWER_RIGHT:
            warn(
                "Lower right causal mask will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )

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
        attn_mask: "CausalMask",
        causal: bool,
        scale: Optional[float],
        dropout_p: float,
    ) -> torch.Tensor:
        if causal:
            raise ValueError("CausalMask should not be used with causal=True")

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
            sdpa_params = SDPAParams(query, key, value, None, dropout_p, causal)
            if can_use_efficient_attention(sdpa_params):
                compute_log_sumexp = False
                if input_requires_grad(query, key, value):
                    compute_log_sumexp = True
                return torch.ops.aten._efficient_attention_forward(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    bias=None,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    max_seqlen_q=None,
                    dropout_p=dropout_p,
                    custom_mask_type=int(attn_mask.variant),
                    compute_log_sumexp=compute_log_sumexp,
                    scale=scale,
                    causal_diagonal=None,
                    seqlen_k=None,
                )[0].transpose(1, 2)
            else:
                # TODO This will warn with the reason why we cant use efficient attention
                # Should this default to on?
                can_use_efficient_attention(sdpa_params, True)
                # We cant use efficient attention the only support for lower right is via materialization
                return scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask.materialize(query.device),
                    dropout_p=dropout_p,
                    is_causal=False,
                    scale=scale,
                )
        else:
            raise ValueError("Invalid causal variant")

    def __repr__(self) -> str:
        return f"CausalMask(variant={self.variant.name}, seq_len_q={self.seq_len_q}, seq_len_kv={self.seq_len_kv})"
