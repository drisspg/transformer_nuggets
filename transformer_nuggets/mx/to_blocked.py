import torch

Tensor = torch.Tensor


def ceil_div(a, b):
    return (a + b - 1) // b


def _to_blocked_single(scales: Tensor) -> Tensor:
    """Assume that we have a 128x4 block of scales in K Major order

    To see more information on the individual tile layout:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    scales = scales.view(-1, 32, 4)
    output = torch.zeros(512, dtype=scales.dtype, device=scales.device).view(32, 16)
    for i in range(4):
        start = i * 4
        end = start + 4
        output[:, start:end] = scales[i, :, :]  # copying 32x4 blocks
    return output


def _to_blocked_single_vmap(scales: Tensor) -> Tensor:
    """Assume that we have a 128x4 block of scales in K Major order
    To see more information on the individual tile layout:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """

    def to_offset(row, col):
        return (row % 32) * 16 + (row // 32) * 4 + col

    rows = torch.arange(128)
    cols = torch.arange(4)

    vmap_func = torch.vmap(to_offset, in_dims=(None, 0))
    vmap_func = torch.vmap(vmap_func, in_dims=(0, None))

    indices = vmap_func(rows, cols)

    final = torch.zeros(512, dtype=scales.dtype, device=scales.device)
    final.scatter_(0, indices.view(-1), scales.flatten())
    return final.view(32, 16)


def to_blocked(input_matrix) -> Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.

    Args:
        input_matrix: Input tensor of shape (H, W)

    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """
    device = input_matrix.device
    dtype = input_matrix.dtype

    rows, cols = input_matrix.shape

    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Create output tensor
    output = torch.zeros((32 * n_row_blocks, 16 * n_col_blocks), dtype=dtype, device=device)

    # Process each block
    for row_block in range(n_row_blocks):
        for col_block in range(n_col_blocks):
            # Calculate input block boundaries
            row_start = row_block * 128
            row_end = min(row_start + 128, rows)  # Avoid going out of bounds
            col_start = col_block * 4
            col_end = min(col_start + 4, cols)  # Avoid going out of bounds

            # Calculate output block boundaries
            out_row_start = row_block * 32
            out_row_end = out_row_start + 32
            out_col_start = col_block * 16
            out_col_end = out_col_start + 16

            block = input_matrix[row_start:row_end, col_start:col_end]

            row_size = row_end - row_start
            col_size = col_end - col_start
            if row_size < 128 or col_size < 4:
                # pad out local block with zeros
                block = torch.nn.functional.pad(block, (0, 4 - col_size, 0, 128 - row_size))

            rearranged_block = _to_blocked_single(block)
            output[out_row_start:out_row_end, out_col_start:out_col_end] = rearranged_block

    return output


# outer = ((offset % 16) / 4) * 32 + (offset / 16)
# inner = (offset % 4)

# @triton.jit
# def _to_blocked(
#     scale_ptr,
#     output_ptr,
#     M: tl.constexpr,
#     K: tl.constexpr,
#     GROUP_SIZE: tl.constexpr,
#     BLOCK_SIZE_H: tl.constexpr,
#     BLOCK_SIZE_W: tl.constexpr,
# ):
#     """Convert 1D tensor to block scaling layout."""
#     pid_h, pid_w = tl.program_id(0), tl.program_id(1)

#     # Offset calculations
#     h_offset = pid_h * BLOCK_SIZE_H
#     w_offset = pid_w * BLOCK_SIZE_W

#     scale_block = tl.make_block_ptr(
#         scale_ptr,
#         shape=(M, K // GROUP_SIZE),
#         strides=(K // GROUP_SIZE, 1),
#         offsets=(h_offset, w_offset),
#         block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_W),
#         order=(1, 0),
#     )

#     # Load scales
#     scales = tl.load(scale_block, boundary_check=(0, 1), other=0.0)


# def scale_to_blocked(scale: Tensor, M, K, GROUP_SIZE=32) -> Tensor:
#     """Convert 1D tensor to block scaling layout.

#     # For simplicity just doing 1 scale group per cta for now

#     """
#     assert scale.dtype == torch.uint8
#     BLOCK_HEIGHT = 128
#     assert GROUP_SIZE in (32, 16)
#     assert K % GROUP_SIZE == 0, f"K {K} must be divisible by GROUP_SIZE {GROUP_SIZE}"
#     NUM_K_SCALES = K // GROUP_SIZE
#     BLOCK_WIDTH = 4

#     num_blocks_h = ceil_div(M, BLOCK_HEIGHT)
#     num_blocks_w = ceil_div(NUM_K_SCALES, BLOCK_WIDTH)

#     # Always create full output tensor per docs
#     output = torch.zeros(num_blocks_h * num_blocks_w * 512, dtype=scale.dtype, device=scale.device)
#     _to_blocked[(num_blocks_h, num_blocks_w)](
#         scale,
#         output,
#         M,
#         K,
#         GROUP_SIZE,
#         BLOCK_HEIGHT,
#         BLOCK_WIDTH,
#     )
#     return output
