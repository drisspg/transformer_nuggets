import torch
from dataclasses import dataclass
from operator import add

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds
from transformer_nuggets.cute.cache import (
    compile_tvm_ffi_and_cache,
    get_cache_stats,
    print_cache,
    set_cache_hashing,
)
from transformer_nuggets.cute.utils import (
    fake_stream,
    get_tensor_alignment,
    make_fake_compact_tensor,
    torch_dtype_to_cute_dtype,
)
from transformer_nuggets.cute.base import CuteOp
from rich import print
from transformer_nuggets import init_logging
import logging

init_logging(logging.INFO)


@dataclass(frozen=True)
class DynamicElementwiseConfig:
    thr_m: int
    thr_n: int
    val_m: int
    val_n: int
    order: tuple[int, int]


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
        cC: cute.Tensor,
        tv_layout: cute.Layout,
        shape: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        blk_coord = ((None, None), bidx)

        blkA = gA[blk_coord]
        blkB = gB[blk_coord]
        blkC = gC[blk_coord]
        blkCoord = cC[blk_coord]

        tidfrgA = cute.composition(blkA, tv_layout)

        tidfrgB = cute.composition(blkB, tv_layout)

        tidfrgC = cute.composition(blkC, tv_layout)
        tidfrgCoord = cute.composition(blkCoord, tv_layout)

        thr_coord = (tidx, None)

        thrA = tidfrgA[thr_coord]
        thrB = tidfrgB[thr_coord]
        thrC = tidfrgC[thr_coord]
        thrCoord = tidfrgCoord[thr_coord]

        for i in cutlass.range_constexpr(cute.size(thrC)):
            if cute.elem_less(thrCoord[i], shape):
                thrC[i] = op(thrA[i], thrB[i])

    def get_name(self, config: DynamicElementwiseConfig) -> str:
        op_name = getattr(self.op, "__name__", str(self.op)).lower()
        return (
            f"dynamic_elementwise_{op_name}"
            f"_t{config.thr_m}x{config.thr_n}"
            f"_v{config.val_m}x{config.val_n}"
        )

    def get_key(self, dtype: torch.dtype, config: DynamicElementwiseConfig) -> str:
        return f"{self.get_name(config)}_dtype={dtype}_order={config.order}"

    def _select_parameters(self, M: int, N: int) -> tuple[int, int, int, int]:
        """Choose parameters based on tensor size."""
        total_elements = M * N
        if total_elements < 1024 * 1024:
            return 8, 32, 2, 8
        else:
            return 4, 32, 8, 8

    def interface(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        M, N = a.shape
        c = torch.empty(M, N, device=a.device, dtype=a.dtype)

        thr_m, thr_n, val_m, val_n = self._select_parameters(M, N)
        config = DynamicElementwiseConfig(
            thr_m=thr_m,
            thr_n=thr_n,
            val_m=val_m,
            val_n=val_n,
            order=tuple(reversed(a.dim_order())),
        )
        dtype = torch_dtype_to_cute_dtype(c.dtype)
        m, n = cute.sym_int(), cute.sym_int()
        assumed_align = min(
            get_tensor_alignment(a, dim=-1),
            get_tensor_alignment(b, dim=-1),
            get_tensor_alignment(c, dim=-1),
        )
        fake_a = make_fake_compact_tensor(dtype, (m, n), assumed_align=assumed_align)
        fake_b = make_fake_compact_tensor(dtype, (m, n), assumed_align=assumed_align)
        fake_c = make_fake_compact_tensor(dtype, (m, n), assumed_align=assumed_align)
        compiled = compile_tvm_ffi_and_cache(
            self,
            self.get_key(c.dtype, config),
            self.op,
            fake_a,
            fake_b,
            fake_c,
            config.thr_m,
            config.thr_n,
            config.val_m,
            config.val_n,
            config.order,
            fake_stream(),
        )
        compiled(a, b, c)
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
        stream: cuda.CUstream,
    ):
        """Parameterized kernel that accepts layout dimensions as constexpr"""
        thr_layout = cute.make_ordered_layout((thr_m, thr_n), order)
        val_layout = cute.make_ordered_layout((val_m, val_n), order)
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gA = cute.zipped_divide(mA, tiler_mn)
        gB = cute.zipped_divide(mB, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)
        cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler_mn)

        op_name = getattr(self.op, "__name__", str(self.op)).lower()
        kernel_name = f"dynamic_elementwise_{op_name}_t{thr_m}x{thr_n}_v{val_m}x{val_n}"
        self.kernel(op, gA, gB, gC, cC, tv_layout, mC.shape, _name_prefix=kernel_name).launch(
            grid=[cute.size(gC, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
            stream=stream,
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
