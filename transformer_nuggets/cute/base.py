from abc import ABC, abstractmethod
from typing import Generic, ParamSpec, TypeVar

import cutlass.cute as cute


_P = ParamSpec("_P")
_R = TypeVar("_R")


class CuteOp(Generic[_P, _R], ABC):
    """Abstract base class for CUTE operations.

    This class provides a consistent interface for implementing CUTE kernels with:
    - Cache key generation via get_key()
    - Kernel definition via kernel() method decorated with @cute.kernel
    - JIT function as __call__
    - Public Python-facing API via interface()
    """

    def __init__(self):
        pass

    @abstractmethod
    def kernel(self, *args, **kwargs):
        """The kernel function that must be decorated with @cute.kernel.

        Subclasses must implement this method with the @cute.kernel decorator
        to define their kernel logic.
        """
        pass

    def get_key(self, *args, **kwargs) -> str:
        """Generate cache key for this operation.

        Default implementation returns operation class name.
        Override to add operation-specific cache key components.

        Args:
            *args: Arguments that affect kernel compilation
            **kwargs: Keyword arguments that affect kernel compilation

        Returns:
            String cache key for this specific configuration
        """
        return self.__class__.__name__

    def _generate_tensor_key(self, tensor: cute.Tensor) -> str:
        """Generate a cache key component for a CUTE tensor."""
        tensor_str = str(tensor)
        if " o " in tensor_str and ")>" in tensor_str:
            inner_part = tensor_str.split(" o ")[1].rstrip(">")
            return f"tensor_{inner_part}_dtype={tensor.element_type}"
        else:
            return f"tensor_shape={tensor.shape}_dtype={tensor.element_type}"

    @abstractmethod
    def interface(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Public entrypoint for running the operation from Python code."""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """The @cute.jit decorated function that launches the kernel.

        This method should contain the kernel launch logic.
        """
        pass
