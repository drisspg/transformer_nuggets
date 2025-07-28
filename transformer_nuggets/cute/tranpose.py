import torch


import cutlass.cute as cute

from transformer_nuggets.cute.cache import (
    cute_compile_and_cache,
)
from cutlass.cute.runtime import from_dlpack
from transformer_nuggets.cute.utils import get_tensor_alignment
from transformer_nuggets import init_logging
import logging

init_logging(logging.INFO)


# @cute.kernel
# def transpose_kernel(
#     tma_load_atom: cute.CopyAtom,
#     tma_load_tensor: cute.Tensor,
#     tma_store_atom: cute.CopyAtom,
#     tma_store_tensor: cute.Tensor,
# ):
#     tidx, _, _ = cute.arch.thread_idx()
#     bidx, _, _ = cute.arch.block_idx()
#     bdim, _, _ = cute.arch.block_dim()

#     cute.copy()

#     thread_idx = bidx * bdim + tidx

#     m, n = gA.shape
#     ni = thread_idx % n
#     mi = thread_idx // n

#     a_val = gA[mi, ni]
#     b_val = gB[mi, ni]

#     gC[mi, ni] = a_val + b_val


@cute.jit
def transpose_launcher(gInpt: cute.Tensor, gOutput: cute.Tensor):
    smem_layout = cute.make_ordered_layout((128, 128), order=(1, 0))
    tile_shape = cute.shape((128, 128))

    # Print layouts and shapes
    cute.printf("smem_layout: {}", smem_layout)
    cute.printf("tile_shape: {}", tile_shape)
    cute.printf("gInpt layout: {}", gInpt.layout)
    cute.printf("gOutput layout: {}", gOutput.layout)

    tma_load_atom, tma_load_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        gInpt,  # Global memory tensor
        smem_layout,  # Shared memory layout
        tile_shape,
    )

    tma_store_atom, tma_store_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        gOutput,  # Global memory tensor
        smem_layout,  # Shared memory layout
        tile_shape,
    )


def transpose(self: torch.Tensor, dim1: int, dim2: int) -> torch.Tensor:
    """
    Transpose the input tensor.
    """
    # Normalize dimensions to handle negative indices
    ndim = self.dim()
    dim1 = dim1 if dim1 >= 0 else ndim + dim1
    dim2 = dim2 if dim2 >= 0 else ndim + dim2

    # Validate dimensions
    if dim1 < 0 or dim2 < 0 or dim1 >= ndim or dim2 >= ndim:
        raise IndexError("Transpose dimensions out of range")

    assert self.is_contiguous(), "Input tensor must be contiguous"
    assert abs(dim1 - dim2) == 1, "Transpose dimensions must be adjacent"

    alignment = get_tensor_alignment(self, -1)

    leading_dims = self.shape[:dim1]
    trailing_dims = self.shape[max(dim2 + 1, self.ndim - 1) :]
    new_sizes = list(leading_dims) + [self.size(dim2), self.size(dim1)] + list(trailing_dims)

    out = torch.empty(*new_sizes, device=self.device, dtype=self.dtype).transpose(dim1, dim2)

    inpt_cute = from_dlpack(self, alignment).mark_layout_dynamic(1)
    output_cute = from_dlpack(out, alignment).mark_layout_dynamic(0)

    cute.nvgpu.cpasync.make_tiled_tma_atom

    compiled_kernel = cute_compile_and_cache(
        transpose_launcher,
        inpt_cute,
        output_cute,
        cache_extra=(alignment,),
    )

    compiled_kernel(inpt_cute, output_cute)
    return out


if __name__ == "__main__":
    # Example usage of the elementwise_op_dynamic function
    M, N = 1024, 1024
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)

    transposed_tensor = transpose(a, 0, 1)
    print("Transposed tensor shape:", transposed_tensor.shape)

    # # Perform element-wise addition using the dynamic kernel
    # c = elementwise_op_dynamic(cute.add, a, b)

    # # Verify the result
    # torch.testing.assert_close(c, a + b)

    # # Benchmark the operation
    # time_taken = benchmark_cuda_function_in_microseconds(lambda: elementwise_op_dynamic(cute.add, a, b))
    # print(f"Time taken for element-wise addition: {time_taken} microseconds")
