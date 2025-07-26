import torch
from operator import mul

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds
from transformer_nuggets.cute.cache import cute_compile_and_cache, get_cache_stats
from transformer_nuggets.cute.element_wise import elementwise_apply_kernel
from rich import print
from transformer_nuggets import init_logging
import logging

init_logging(logging.DEBUG)


@cute.jit
def elem_kernel_parameterized(
    op: cutlass.Constexpr,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    thr_m: cutlass.Constexpr,
    thr_n: cutlass.Constexpr,
    val_m: cutlass.Constexpr,
    val_n: cutlass.Constexpr,
):
    """Parameterized kernel that accepts layout dimensions as constexpr"""
    thr_layout = cute.make_layout((thr_m, thr_n), stride=(thr_n, 1))
    val_layout = cute.make_layout((val_m, val_n), stride=(val_n, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    elementwise_apply_kernel(op, gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def elementwise_op_dynamic(
    op: cutlass.Constexpr,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    M, N = a.shape
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    # Choose parameters based on size
    total_elements = M * N
    if total_elements < 1024 * 1024:
        thr_m, thr_n, val_m, val_n = 8, 32, 2, 8
    elif total_elements < 16 * 1024 * 1024:
        thr_m, thr_n, val_m, val_n = 4, 64, 4, 8
    else:
        thr_m, thr_n, val_m, val_n = 2, 128, 8, 8

    # Create tensors with optimization hints
    mA = from_dlpack(a, assumed_align=16).mark_layout_dynamic(1)
    mB = from_dlpack(b, assumed_align=16).mark_layout_dynamic(1)
    mC = from_dlpack(c, assumed_align=16).mark_layout_dynamic(1)

    # Convert to compile-time constants - this creates separate kernels for each configuration
    compiled_kernel = cute_compile_and_cache(
        elem_kernel_parameterized, op, mA, mB, mC, thr_m, thr_n, val_m, val_n
    )
    compiled_kernel(mA, mB, mC)
    return c


if __name__ == "__main__":
    shapes = [(2**i, 2**i) for i in range(8, 14)]
    # shapes.extend([(1000, 1000), (1234, 5678), (3333, 7777), (999, 1001)])
    for M, N in shapes:
        a = torch.randn(M, N, device="cuda", dtype=torch.float16)
        b = torch.randn(M, N, device="cuda", dtype=torch.float16)

        # Test the new API that takes regular PyTorch tensors
        out = elementwise_op_dynamic(mul, a, b)
        torch.testing.assert_close(out, mul(a, b))

        time_torch = benchmark_cuda_function_in_microseconds(lambda: mul(a, b))
        time_cute = benchmark_cuda_function_in_microseconds(
            lambda: elementwise_op_dynamic(mul, a, b)
        )
        print(f"M = {M}, N = {N}")
        print(f"torch GB/s = {M * N * 3 * out.element_size() / time_torch * 1e-3}")
        print(f"cute GB/s = {M * N * 3 * out.element_size() / time_cute * 1e-3}")

    stats = get_cache_stats()
    print(f"Cache stats: {stats}")
