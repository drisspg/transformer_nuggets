"""
Copy Between Layouts Example using CuTe Copy Atoms

This example demonstrates how to use CuTe's copy atoms to efficiently copy data
between different tensor layouts on GPU. Copy atoms are fundamental building blocks
that enable optimized memory transfers with different access patterns.

Key concepts demonstrated:
1. Using copy atoms for layout transformations (e.g., row-major to column-major)
2. Thread-value (TV) layouts for efficient memory access patterns
3. Tiled copy operations for handling large tensors
4. Predication for non-tile-aligned shapes

Copy atoms in CuTe provide:
- Vectorized loads/stores for maximum memory bandwidth utilization
- Automatic handling of alignment requirements
- Support for different memory spaces (global, shared, register)
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds
from transformer_nuggets.cute.cache import compile_and_cache, get_cache_stats, print_cache
from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.utils import get_tensor_alignment


class CopyLayoutOp(CuteOp):
    """Copy operation that transforms data between different layouts using copy atoms."""

    def __init__(self, src_layout_type: str = "row_major", dst_layout_type: str = "col_major"):
        super().__init__()
        self.src_layout_type = src_layout_type
        self.dst_layout_type = dst_layout_type

    @cute.kernel
    def kernel(
        self,
        gSrc: cute.Tensor,
        gDst: cute.Tensor,
        cCoord: cute.Tensor,  # coordinate tensor for predication
        shape: cute.Shape,
        src_thr_layout: cute.Layout,
        src_val_layout: cute.Layout,
        dst_thr_layout: cute.Layout,
        dst_val_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Slice tensors for current thread block
        blk_coord = ((None, None), bidx)
        blkSrc = gSrc[blk_coord]
        blkDst = gDst[blk_coord]
        blkCoord = cCoord[blk_coord]

        # Create copy atoms for source and destination
        # Using CopyUniversalOp for general-purpose copying
        copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gSrc.element_type)
        copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gDst.element_type)

        # Create tiled copy operations with different layouts for src and dst
        tiled_copy_src = cute.make_tiled_copy_tv(copy_atom_load, src_thr_layout, src_val_layout)
        tiled_copy_dst = cute.make_tiled_copy_tv(copy_atom_store, dst_thr_layout, dst_val_layout)

        # Get thread-specific slices
        thr_copy_src = tiled_copy_src.get_slice(tidx)
        thr_copy_dst = tiled_copy_dst.get_slice(tidx)

        # Partition source for loading and destination for storing
        thrSrc = thr_copy_src.partition_S(blkSrc)
        thrDst = thr_copy_dst.partition_D(blkDst)

        # Allocate register fragments
        frgData = cute.make_fragment_like(thrSrc)

        # Create predicate mask for boundary handling
        thrCoord = thr_copy_src.partition_S(blkCoord)
        frgPred = cute.make_fragment(thrCoord.shape, cutlass.Boolean)

        # Set predicate based on tensor bounds
        for i in range(0, cute.size(frgPred), 1):
            frgPred[i] = cute.elem_less(thrCoord[i], shape)

        # Copy from source to registers with predication
        cute.copy(copy_atom_load, thrSrc, frgData, pred=frgPred)

        # Copy from registers to destination with predication
        cute.copy(copy_atom_store, frgData, thrDst, pred=frgPred)

    def get_key(self, *args, **kwargs) -> str:
        """Generate cache key including layout transformation type."""
        key_parts = [
            self.__class__.__name__,
            f"src={self.src_layout_type}",
            f"dst={self.dst_layout_type}",
        ]

        # Add tensor properties
        for arg in args:
            if isinstance(arg, cute.Tensor):
                key_parts.append(self._generate_tensor_key(arg))

        # Add layout parameters if provided
        if "params" in kwargs:
            src_thr_m, src_thr_n, src_val_m, src_val_n = kwargs["params"]["src"]
            dst_thr_m, dst_thr_n, dst_val_m, dst_val_n = kwargs["params"]["dst"]
            key_parts.extend(
                [
                    f"src_thr={src_thr_m}x{src_thr_n}",
                    f"src_val={src_val_m}x{src_val_n}",
                    f"dst_thr={dst_thr_m}x{dst_thr_n}",
                    f"dst_val={dst_val_m}x{dst_val_n}",
                ]
            )

        return "_".join(key_parts)

    @cute.jit
    def __call__(
        self,
        mSrc: cute.Tensor,
        mDst: cute.Tensor,
        src_thr_m: cutlass.Constexpr,
        src_thr_n: cutlass.Constexpr,
        src_val_m: cutlass.Constexpr,
        src_val_n: cutlass.Constexpr,
        dst_thr_m: cutlass.Constexpr,
        dst_thr_n: cutlass.Constexpr,
        dst_val_m: cutlass.Constexpr,
        dst_val_n: cutlass.Constexpr,
    ):
        """JIT function that launches the copy kernel with specified layouts."""
        # Create source layout (row-major optimized)
        src_thr_layout = cute.make_layout((src_thr_m, src_thr_n), stride=(src_thr_n, 1))
        src_val_layout = cute.make_layout((src_val_m, src_val_n), stride=(src_val_n, 1))
        src_tiler_mn, src_tv_layout = cute.make_layout_tv(src_thr_layout, src_val_layout)

        # Create destination layout (can be column-major or other patterns)
        if self.dst_layout_type == "col_major":
            # Column-major: swap strides
            dst_thr_layout = cute.make_layout((dst_thr_m, dst_thr_n), stride=(1, dst_thr_m))
            dst_val_layout = cute.make_layout((dst_val_m, dst_val_n), stride=(1, dst_val_m))
        else:
            # Row-major (same as source)
            dst_thr_layout = cute.make_layout((dst_thr_m, dst_thr_n), stride=(dst_thr_n, 1))
            dst_val_layout = cute.make_layout((dst_val_m, dst_val_n), stride=(dst_val_n, 1))

        dst_tiler_mn, dst_tv_layout = cute.make_layout_tv(dst_thr_layout, dst_val_layout)

        # Use source tiler for both to ensure consistent tiling
        gSrc = cute.zipped_divide(mSrc, src_tiler_mn)
        gDst = cute.zipped_divide(mDst, src_tiler_mn)

        # Create coordinate tensor for predication
        idCoord = cute.make_identity_tensor(mSrc.shape)
        cCoord = cute.zipped_divide(idCoord, tiler=src_tiler_mn)

        self.kernel(
            gSrc,
            gDst,
            cCoord,
            mSrc.shape,
            src_thr_layout,
            src_val_layout,
            dst_thr_layout,
            dst_val_layout,
        ).launch(
            grid=[cute.size(gSrc, mode=[1]), 1, 1],
            block=[cute.size(src_tv_layout, mode=[0]), 1, 1],
        )


def copy_between_layouts(
    src: torch.Tensor,
    layout_transform: str = "row_to_col",
) -> torch.Tensor:
    """
    Copy tensor data between different layouts using CuTe copy atoms.

    Args:
        src: Source tensor (PyTorch tensor)
        layout_transform: Type of layout transformation
            - "row_to_col": Row-major to column-major
            - "transpose": Transpose the tensor
            - "identity": Copy with same layout (for baseline)

    Returns:
        Destination tensor with transformed layout
    """
    M, N = src.shape

    # Create destination tensor
    if layout_transform == "transpose":
        dst = torch.empty(N, M, device="cuda", dtype=src.dtype)
    else:
        dst = torch.empty(M, N, device="cuda", dtype=src.dtype)

    # Determine layout types
    if layout_transform == "row_to_col":
        src_layout = "row_major"
        dst_layout = "col_major"
    elif layout_transform == "transpose":
        src_layout = "row_major"
        dst_layout = "row_major"  # But with transposed dimensions
    else:  # identity
        src_layout = "row_major"
        dst_layout = "row_major"

    # Create operation instance
    copy_op = CopyLayoutOp(src_layout, dst_layout)

    # Choose thread-value parameters based on tensor size
    # These control how work is distributed among threads
    if M * N < 1024 * 1024:
        # Small tensors: use smaller thread blocks
        src_params = (4, 32, 2, 4)  # (thr_m, thr_n, val_m, val_n)
        dst_params = (4, 32, 2, 4)
    else:
        # Large tensors: use larger thread blocks for better occupancy
        src_params = (8, 64, 4, 4)
        dst_params = (8, 64, 4, 4)

    src_thr_m, src_thr_n, src_val_m, src_val_n = src_params
    dst_thr_m, dst_thr_n, dst_val_m, dst_val_n = dst_params

    # Calculate alignment
    align_src = get_tensor_alignment(src, dim=-1)
    align_dst = get_tensor_alignment(dst, dim=-1)

    # Convert to CUTE tensors
    mSrc = from_dlpack(src, assumed_align=align_src).mark_layout_dynamic(1)
    mDst = from_dlpack(dst, assumed_align=align_dst).mark_layout_dynamic(1)

    # Handle transpose by swapping dimensions
    if layout_transform == "transpose":
        # Create a transposed view for destination
        mDst = mDst.view((N, M))

    # Generate cache key
    cache_key = copy_op.get_key(
        mSrc,
        mDst,
        params={"src": src_params, "dst": dst_params},
        alignments=(align_src, align_dst),
    )

    # Compile and run
    compiled_kernel = compile_and_cache(
        copy_op,
        cache_key,
        mSrc,
        mDst,
        src_thr_m,
        src_thr_n,
        src_val_m,
        src_val_n,
        dst_thr_m,
        dst_thr_n,
        dst_val_m,
        dst_val_n,
    )

    compiled_kernel(mSrc, mDst)
    return dst


def benchmark_layout_copy(M: int, N: int, layout_transform: str):
    """Benchmark different layout transformations."""
    print(f"\nBenchmarking {layout_transform} for {M}x{N} tensor:")

    # Create source tensor
    src = torch.randn(M, N, device="cuda", dtype=torch.float16)

    # Warm up
    _ = copy_between_layouts(src, layout_transform)

    # Benchmark CuTe implementation
    time_cute = benchmark_cuda_function_in_microseconds(
        lambda: copy_between_layouts(src, layout_transform)
    )

    # Benchmark PyTorch baseline
    if layout_transform == "row_to_col":
        # PyTorch doesn't have explicit row/col major, so we measure copy time
        time_torch = benchmark_cuda_function_in_microseconds(lambda: src.clone())
        baseline_name = "PyTorch clone"
    elif layout_transform == "transpose":
        time_torch = benchmark_cuda_function_in_microseconds(lambda: src.T.contiguous())
        baseline_name = "PyTorch transpose"
    else:  # identity
        time_torch = benchmark_cuda_function_in_microseconds(lambda: src.clone())
        baseline_name = "PyTorch clone"

    # Calculate throughput
    total_bytes = 2 * M * N * src.element_size()  # Read + Write
    cute_throughput = total_bytes / time_cute * 1e-3  # GB/s
    torch_throughput = total_bytes / time_torch * 1e-3  # GB/s

    print(f"  CuTe time: {time_cute:.2f} μs ({cute_throughput:.2f} GB/s)")
    print(f"  {baseline_name} time: {time_torch:.2f} μs ({torch_throughput:.2f} GB/s)")
    print(f"  Speedup: {time_torch / time_cute:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("CuTe Copy Atoms: Layout Transformation Examples")
    print("=" * 60)

    # Test different tensor sizes and transformations
    test_configs = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]

    transformations = ["row_to_col", "transpose", "identity"]

    for M, N in test_configs:
        for transform in transformations:
            # Create test tensor
            src = torch.randn(M, N, device="cuda", dtype=torch.float16)

            # Perform transformation
            dst = copy_between_layouts(src, transform)

            # Verify correctness
            if transform == "transpose":
                expected = src.T.contiguous()
                torch.testing.assert_close(dst, expected)
                print(f"✓ Transpose {M}x{N} verified")
            elif transform == "identity":
                torch.testing.assert_close(dst, src)
                print(f"✓ Identity copy {M}x{N} verified")
            else:  # row_to_col
                # For row/col major, data should be the same, just layout different
                torch.testing.assert_close(dst, src)
                print(f"✓ Row-to-column {M}x{N} verified")

    print("\n" + "=" * 60)
    print("Performance Benchmarks")
    print("=" * 60)

    # Benchmark larger sizes
    benchmark_configs = [
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ]

    for M, N in benchmark_configs:
        for transform in transformations:
            benchmark_layout_copy(M, N, transform)

    # Print cache statistics
    print("\n" + "=" * 60)
    print("Cache Statistics")
    print("=" * 60)
    stats = get_cache_stats()
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print_cache()
