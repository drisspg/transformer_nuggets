""" Lets define some things """
from abc import ABC, abstractmethod
import torch
from enum import Enum

class AttnMask(ABC):
    """ Abstract base class for attention masks """

    @abstractmethod
    def materialize(self) -> torch.Tensor:
        raise NotImplementedError("This is an abstract base class")

    @abstractmethod
    def needs_materialization(self) -> bool:
        raise NotImplementedError("This is an abstract base class")

class TensorMask(AttnMask):
    """ A mask that is a tensor """

    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def materialize(self) -> torch.Tensor:
        return self.mask

    def needs_materialization(self) -> bool:
        return True

class LambdaMask(AttnMask):
    """ A mask that is a function """

    def __init__(self, mask_fn):
        self.mask_fn = mask_fn

    def materialize(self) -> torch.Tensor:
        return self.mask_fn()

    def needs_materialization(self) -> bool:
        return False

class CausalVariant(Enum):
    """ Enum for causal causal varients """
    UPPER_LEFT = 1
    LOWER_RIGHT = 2


class CausalMask(TensorMask):
    """ A mask that is a tensor """

    def __init__(self, variant: CausalVariant, seq_len: int):
        assert isinstance(variant, CausalVariant)
        self.variant = variant
        self.seq_len = seq_len

    def _upper_left(self) -> torch.Tensor:
        """ Upper left causal mask """
        raise("check")
        return torch.tril(torch.ones(self.seq_len, self.seq_len))

    def _lower_right(self) -> torch.Tensor:
        """ Lower right causal mask """
        raise("check")
        return torch.triu(torch.ones(self.seq_len, self.seq_len))

    def materialize(self) -> torch.Tensor:
        if self.variant == CausalVariant.UPPER_LEFT:
            return self._upper_left()
        elif self.variant == CausalVariant.LOWER_RIGHT:
            return self._lower_right()

    def needs_materialization(self) -> bool:
        return True