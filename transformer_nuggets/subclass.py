from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Type, Dict, Any
import torch
import functools

# Type alias for PyTorch operations
Op = Type[torch._ops.OpOverloadPacket]


@dataclass(frozen=True)
class SubclassTensorArgs:
    """Contains the essential arguments needed to reconstruct a tensor subclass."""

    original_shape: torch.Size
    original_strides: Tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "SubclassTensorArgs":
        """Creates SubclassTensorArgs from an existing tensor."""
        return SubclassTensorArgs(
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
        )


def _implements(cls, aten_ops_or_torch_fns: Iterable[Op] | Op) -> callable:
    """Decorator to implement functions for aten ops in __torch_dispatch__ or torch functions in __torch_function__.

    Args:
        cls: The tensor subclass
        aten_ops_or_torch_fns: Single operation or iterable of operations to implement

    Returns:
        Decorator function that registers the implementation

    Example:
        class MyTensor(PT2Subclass):
            implements = classmethod(_implements)

            @implements(torch.nn.functional.linear)
            def _(func, types, args, kwargs):
                # Implementation here
                pass
    """
    if not hasattr(cls, "_ATEN_OP_OR_TORCH_FN_TABLE"):
        cls._ATEN_OP_OR_TORCH_FN_TABLE = {}

    if not isinstance(aten_ops_or_torch_fns, (list, tuple)):
        aten_ops_or_torch_fns = [aten_ops_or_torch_fns]

    def decorator(func):
        for op in aten_ops_or_torch_fns:

            @functools.wraps(op)
            def wrapper(f, types, args, kwargs):
                return func(f, types, args, kwargs)

            cls._ATEN_OP_OR_TORCH_FN_TABLE[op] = wrapper
        return func

    return decorator


class PT2Subclass(ABC, torch.Tensor):
    """Abstract base class for PyTorch 2.0 compliant tensor subclasses.

    This class enforces implementation of required methods for proper tensor subclassing
    while maintaining inheritance from torch.Tensor.

    Required Methods:
        __new__: Constructor for creating new instances
        __tensor_flatten__: Method for flattening the tensor into constituent parts
        __tensor_unflatten__: Method for reconstructing the tensor from flattened parts
        __torch_dispatch__: Handler for tensor operations
    """

    implements = classmethod(_implements)

    @staticmethod
    @abstractmethod
    def __new__(cls, *args, **kwargs) -> "PT2Subclass":
        """Create a new instance of the tensor subclass.
        I like structuring this as SubclassArgs then everything else that
        goes on the instance
        Example:
            subclass = torch.Tensor._make_wrapper_subclass(
                cls,
                tensor_meta.original_shape,
                tensor_meta.original_strides,
                tensor_meta.storage_offset,
                dtype=tensor_meta.dtype,
                device=tensor_meta.device,
                requires_grad=tensor_meta.requires_grad,
            )
            return subclass

        """
        pass

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the tensor subclass instance."""
        pass

    @abstractmethod
    def __tensor_flatten__(self) -> Tuple[List[str], Dict[str, Any]]:
        """Flatten the tensor into its constituent parts.

        Returns:
            Tuple containing:
                - List of the attributes on the subclass that are tensors
                - Dictionary of metadata needed for reconstruction
        """
        pass

    @staticmethod
    @abstractmethod
    def __tensor_unflatten__(
        inner_tensors: Dict[str, torch.Tensor], meta: Dict[str, Any], outer_size: torch.Size, outer_stride: torch.Size
    ) -> "PT2Subclass":
        """Reconstruct the tensor from flattened parts.

        Args:
            inner_tensors: Dictionary mapping names to constituent tensors
            meta: Metadata dictionary from __tensor_flatten__
            *args, **kwargs: Additional arguments for reconstruction

        Returns:
            Reconstructed tensor subclass instance
        """
        pass

    @classmethod
    @abstractmethod
    def __torch_dispatch__(
        cls, func: Op, types: Tuple[Type, ...], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Handle tensor operations.

        Args:
            func: The operation to perform
            types: Tuple of argument types
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Result of the operation
        """
        pass
