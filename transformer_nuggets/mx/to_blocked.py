import torch

Tensor = torch.Tensor


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix: Tensor) -> Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.

    See:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)

    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype
        )
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def _to_blocked_single_manual(scales: Tensor) -> Tensor:
    """Slow for testing"""
    scales = scales.view(-1, 32, 4)
    output = torch.zeros(512, dtype=scales.dtype, device=scales.device).view(32, 16)
    for i in range(4):
        start = i * 4
        end = start + 4
        output[:, start:end] = scales[i, :, :]  # copying 32x4 blocks
    return output


def _to_blocked_single(scales: Tensor) -> Tensor:
    """Assume that we have a 128x4 block of scales in K Major order

    To see more information on the individual tile layout:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    assert scales.shape == (128, 4)
    scales_tiled = scales.view(4, 32, 4)  # view as 4 - (32, 4) tiles
    return scales_tiled.transpose(0, 1).reshape(32, 16)  # Interleave tiles


def to_blocked_manual(input_matrix) -> Tensor:
    """Slow for testing purposes"""
    device = input_matrix.device
    dtype = input_matrix.dtype

    rows, cols = input_matrix.shape

    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Create output tensor
    output = torch.zeros(512 * n_row_blocks * n_col_blocks, dtype=dtype, device=device)
    # output = torch.zeros((32 * n_row_blocks, 16 * n_col_blocks), dtype=dtype, device=device)

    # Process each block
    for row_block in range(n_row_blocks):
        for col_block in range(n_col_blocks):
            lineared_index = row_block * n_col_blocks + col_block
            # Calculate input block boundaries
            row_start = row_block * 128
            row_end = min(row_start + 128, rows)  # Avoid going out of bounds
            col_start = col_block * 4
            col_end = min(col_start + 4, cols)  # Avoid going out of bounds

            block = input_matrix[row_start:row_end, col_start:col_end]

            row_size = row_end - row_start
            col_size = col_end - col_start
            if row_size < 128 or col_size < 4:
                # pad out local block with zeros
                block = torch.nn.functional.pad(block, (0, 4 - col_size, 0, 128 - row_size))

            rearranged_block = _to_blocked_single(block)

            start = lineared_index * 512
            end = start + 512
            output[start:end] = rearranged_block.view(-1)

    return output
