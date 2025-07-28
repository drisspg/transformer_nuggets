import torch
import hashlib
from typing import Any
import cutlass.cute as cute


def get_tensor_alignment(tensor: torch.Tensor, dim: int) -> int:
    """Calculate the maximum alignment for a tensor assuming a specific dimension is contiguous.

    Args:
        tensor: The tensor to check
        dim: The dimension assumed to be contiguous (negative indexing supported)

    Returns:
        Maximum alignment in bytes that divides both the pointer and the contiguous region size
    """
    # Handle negative indexing
    if dim < 0:
        dim = tensor.ndim + dim

    # Get the size of the assumed contiguous dimension
    contiguous_elements = tensor.shape[dim]

    # Convert to bytes
    element_size = tensor.element_size()
    contiguous_bytes = contiguous_elements * element_size

    # Get pointer
    ptr = tensor.data_ptr()

    # Find the best alignment that divides both pointer and size
    max_align = 128

    while max_align > 1:
        if ptr % max_align == 0 and contiguous_bytes % max_align == 0:
            break
        max_align //= 2

    return max_align


def generate_tensor_cache_key(tensor: cute.Tensor) -> str:
    """Generate a cache key component for a CUTE tensor.

    Args:
        tensor: CUTE tensor to generate key for

    Returns:
        String representation suitable for cache key
    """
    tensor_str = str(tensor)
    if " o " in tensor_str and ")>" in tensor_str:
        # Extract everything after ' o ' and before '>'
        inner_part = tensor_str.split(" o ")[1].rstrip(">")
        return f"tensor_{inner_part}_dtype={tensor._dtype}"
    else:
        # Fallback if format is different
        return f"tensor_shape={tensor.shape}_dtype={tensor._dtype}"


def hash_cache_key(key_parts: list | tuple, use_sha256: bool = True) -> str:
    """Hash cache key components into a fixed-length string.

    Args:
        key_parts: List or tuple of cache key components
        use_sha256: If True, use SHA256 hash; otherwise join with underscores

    Returns:
        Hashed or joined cache key
    """
    key_str = "_".join(str(part) for part in key_parts)

    if use_sha256:
        return hashlib.sha256(key_str.encode()).hexdigest()
    else:
        return key_str


def extract_tensor_properties(tensor: torch.Tensor) -> dict[str, Any]:
    """Extract relevant properties from a PyTorch tensor for caching.

    Args:
        tensor: PyTorch tensor

    Returns:
        Dictionary of tensor properties
    """
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": tuple(tensor.stride()),
        "is_contiguous": tensor.is_contiguous(),
        "data_ptr": tensor.data_ptr(),
    }
