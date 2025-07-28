from abc import ABC, abstractmethod

import cutlass.cute as cute


class CuteOp(ABC):
    """Abstract base class for CUTE operations.

    This class provides a consistent interface for implementing CUTE kernels with:
    - Cache key generation via get_key()
    - Kernel definition via get_kernel()
    - JIT function as __call__
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_kernel(self):
        """Return the @cute.kernel decorated function."""
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
    def __call__(self, *args, **kwargs):
        """The @cute.jit decorated function that launches the kernel.

        This method should contain the kernel launch logic.
        """
        pass
