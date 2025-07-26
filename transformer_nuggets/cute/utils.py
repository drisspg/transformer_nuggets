import torch


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
