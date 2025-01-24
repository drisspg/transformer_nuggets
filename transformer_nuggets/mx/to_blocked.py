import torch
import triton
import triton.language as tl
from triton import cdiv


Tensor = torch.Tensor


@triton.jit
def _to_blocked(
    scale_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """Convert 1D tensor to block scaling layout."""
    pid_h, pid_w = tl.program_id(0), tl.program_id(1)

    # Offset calculations
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W

    scale_block = tl.make_block_ptr(
        scale_ptr,
        shape=(M, K // GROUP_SIZE),
        strides=(K // GROUP_SIZE, 1),
        offsets=(h_offset, w_offset),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_W),
        order=(1, 0),
    )

    # Load scales
    scales = tl.load(scale_block, boundary_check=(0, 1), other=0.0)

    # Calculate output block offset
    block_offset = (pid_h * (K // GROUP_SIZE // BLOCK_SIZE_W) + pid_w) * 512

    # Store in output layout
    for i in range(BLOCK_SIZE_H):
        offset = block_offset + (i % 32) * 16 + (i // 32) * 4
        if h_offset + i < M:
            for j in range(BLOCK_SIZE_W // GROUP_SIZE):
                tl.store(output_ptr + offset + j, scales[i, j])


def scale_to_blocked(scale: Tensor, M, K, GROUP_SIZE=32) -> Tensor:
    """Convert 1D tensor to block scaling layout.

    # For simplicity just doing 1 scale group per cta for now

    """
    assert scale.dtype == torch.uint8
    BLOCK_HEIGHT = 128
    assert GROUP_SIZE in (32, 16)
    assert K % GROUP_SIZE == 0, f"K {K} must be divisible by GROUP_SIZE {GROUP_SIZE}"
    NUM_K_SCALES = K // GROUP_SIZE
    BLOCK_WIDTH = 4

    num_blocks_h = cdiv(M, BLOCK_HEIGHT)
    num_blocks_w = cdiv(NUM_K_SCALES, BLOCK_WIDTH)

    # Always create full output tensor per docs
    output = torch.zeros(num_blocks_h * num_blocks_w * 512, dtype=scale.dtype, device=scale.device)
    _to_blocked[(num_blocks_h, num_blocks_w)](
        scale,
        output,
        M,
        K,
        GROUP_SIZE,
        BLOCK_HEIGHT,
        BLOCK_WIDTH,
    )
    return output
