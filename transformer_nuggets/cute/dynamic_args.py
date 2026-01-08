import torch
from operator import add

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds
from transformer_nuggets.cute.cache import (
    compile_and_cache,
    get_cache_stats,
    print_cache,
    set_cache_hashing,
)
from transformer_nuggets.cute.utils import get_tensor_alignment
from transformer_nuggets.cute.base import CuteOp
from rich import print
from transformer_nuggets import init_logging
import logging

init_logging(logging.INFO)


class DynamicElementwiseOp(CuteOp[[torch.Tensor, torch.Tensor], torch.Tensor]):
    """Elementwise operation with dynamic parameter selection based on tensor size."""

    def __init__(self, op: cutlass.Constexpr):
        super().__init__()
        self.op = op

    @cute.kernel
    def kernel(
        self,
        op: cutlass.Constexpr,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        blk_coord = ((None, None), bidx)

        blkA = gA[blk_coord]
        blkB = gB[blk_coord]
        blkC = gC[blk_coord]

        tidfrgA = cute.composition(blkA, tv_layout)

        tidfrgB = cute.composition(blkB, tv_layout)

        tidfrgC = cute.composition(blkC, tv_layout)

        thr_coord = (tidx, None)

        thrA = tidfrgA[thr_coord]
        thrB = tidfrgB[thr_coord]
        thrC = tidfrgC[thr_coord]

        thrC[None] = op(thrA.load(), thrB.load())

    def get_key(self, *args, **kwargs) -> str:
        """Generate cache key including operation type, tensor properties, and dynamic parameters."""
        op_name = getattr(self.op, "__name__", str(self.op))
        key_parts = [self.__class__.__name__, f"op={op_name}"]

        # Extract parameters if provided in kwargs
        if "params" in kwargs:
            thr_m, thr_n, val_m, val_n, order = kwargs["params"]
            key_parts.extend(
                [
                    f"thr_m={thr_m}",
                    f"thr_n={thr_n}",
                    f"val_m={val_m}",
                    f"val_n={val_n}, order={order}",
                ]
            )

        # Add tensor properties
        for arg in args:
            if isinstance(arg, cute.Tensor):
                key_parts.append(self._generate_tensor_key(arg))

        # Add alignment info if provided
        if "alignments" in kwargs:
            align_a, align_b, align_c = kwargs["alignments"]
            key_parts.extend([f"align_a={align_a}", f"align_b={align_b}", f"align_c={align_c}"])

        return "_".join(key_parts)

    def _select_parameters(self, M: int, N: int) -> tuple[int, int, int, int]:
        """Choose parameters based on tensor size."""
        total_elements = M * N
        if total_elements < 1024 * 1024:
            return 8, 32, 2, 8
        else:
            return 4, 32, 8, 8

    def interface(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        M, N = a.shape
        c = torch.empty(M, N, device="cuda", dtype=torch.float16)

        thr_m, thr_n, val_m, val_n = self._select_parameters(M, N)

        align_a = get_tensor_alignment(a, dim=-1)
        align_b = get_tensor_alignment(b, dim=-1)
        align_c = get_tensor_alignment(c, dim=-1)

        mA = from_dlpack(a, assumed_align=align_a).mark_layout_dynamic()
        mB = from_dlpack(b, assumed_align=align_b).mark_layout_dynamic()
        mC = from_dlpack(c, assumed_align=align_c).mark_layout_dynamic()

        dim_orders = tuple(reversed(a.dim_order()))
        compile_and_cache(
            self,
            self.get_key(
                mA,
                mB,
                mC,
                params=(thr_m, thr_n, val_m, val_n, dim_orders),
                alignments=(align_a, align_b, align_c),
            ),
            self.op,
            mA,
            mB,
            mC,
            thr_m,
            thr_n,
            val_m,
            val_n,
            dim_orders,
        )(mA, mB, mC)
        return c

    @cute.jit
    def __call__(
        self,
        op: cutlass.Constexpr,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        thr_m: cutlass.Constexpr,
        thr_n: cutlass.Constexpr,
        val_m: cutlass.Constexpr,
        val_n: cutlass.Constexpr,
        order: cutlass.Constexpr,
    ):
        """Parameterized kernel that accepts layout dimensions as constexpr"""
        thr_layout = cute.make_ordered_layout((thr_m, thr_n), order)
        val_layout = cute.make_ordered_layout((val_m, val_n), order)
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gA = cute.zipped_divide(mA, tiler_mn)
        gB = cute.zipped_divide(mB, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        self.kernel(op, gA, gB, gC, tv_layout).launch(
            grid=[cute.size(gC, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )


def elementwise_op_dynamic(
    op: cutlass.Constexpr,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Apply elementwise operation with dynamic parameter selection.

    This function automatically selects kernel parameters based on tensor size
    for optimal performance.
    """
    return DynamicElementwiseOp(op).interface(a, b)


def cute_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return elementwise_op_dynamic(add, a, b)


if __name__ == "__main__":
    set_cache_hashing(True)
    shapes = [(2**i, 2**i) for i in range(8, 14)]
    shapes.extend([(1000, 1000), (1234, 5678), (3333, 7777), (999, 1001)])
    for M, N in shapes:
        a = torch.randn(M, N, device="cuda", dtype=torch.float16)
        b = torch.randn(M, N, device="cuda", dtype=torch.float16)

        # Test the new API that takes regular PyTorch tensors

        out = elementwise_op_dynamic(add, a, b)
        torch.testing.assert_close(out, a + b)

        time_torch = benchmark_cuda_function_in_microseconds(lambda: a + b)
        time_cute = benchmark_cuda_function_in_microseconds(
            lambda: elementwise_op_dynamic(add, a, b)
        )
        print(f"M = {M}, N = {N}")
        print(f"torch GB/s = {M * N * 3 * out.element_size() / time_torch * 1e-3}")
        print(f"cute GB/s = {M * N * 3 * out.element_size() / time_cute * 1e-3}")

    stats = get_cache_stats()
    print(f"Cache stats: {stats}")
    print_cache()
