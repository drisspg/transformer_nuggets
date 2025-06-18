from dataclasses import dataclass
from collections.abc import Iterable
import torch
import functools

# Type alias for PyTorch operations
Op = type[torch._ops.OpOverloadPacket]


@dataclass(frozen=True)
class SubclassTensorArgs:
    """Contains the essential arguments needed to reconstruct a tensor subclass."""

    original_shape: torch.Size
    original_strides: tuple[int, ...]
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
