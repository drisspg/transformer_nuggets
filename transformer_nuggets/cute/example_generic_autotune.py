"""CuTeDSL kernel autotuned with Helion's generic autotune API.

Compare with example_helion_autotune.py which uses the old adapter layer
(TunableKernel subclass, HelionAutotuner, KernelAdapter, etc.).

This version uses the upstream helion.autotuner.generic.autotune() function
directly -- just pass tunables, compile_fn, baseline_fn, and args.
"""

from operator import add

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from helion.autotuner import PowerOfTwoFragment
from helion.autotuner.generic import autotune
from helion.runtime.config import Config

from transformer_nuggets.cute.cache import compile_and_cache
from transformer_nuggets.cute.utils import (
    get_tensor_alignment,
)
from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds


class ElementwiseAddOp(CuteOp[[torch.Tensor, torch.Tensor], torch.Tensor]):
    tunables = {
        "thr_m": PowerOfTwoFragment(4, 32, 8),
        "thr_n": PowerOfTwoFragment(8, 128, 32),
        "val_m": PowerOfTwoFragment(1, 16, 4),
        "val_n": PowerOfTwoFragment(1, 32, 8),
    }

    def __init__(self):
        super().__init__()
        self.op = add

    def compile(self, config: Config):
        thr_m, thr_n = config["thr_m"], config["thr_n"]
        val_m, val_n = config["val_m"], config["val_n"]

        def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            M, N = a.shape
            tile_m = thr_m * val_m
            tile_n = thr_n * val_n
            if M % tile_m != 0 or N % tile_n != 0:
                raise ValueError(
                    f"Invalid tile for shape {M}x{N}: tile_m={tile_m}, tile_n={tile_n}"
                )
            if thr_m * thr_n > 1024:
                raise ValueError(f"Invalid thread block size: thr_m*thr_n={thr_m * thr_n}")
            c = torch.empty(M, N, device="cuda", dtype=a.dtype)
            mA = from_dlpack(a, assumed_align=get_tensor_alignment(a, dim=-1)).mark_layout_dynamic(
                leading_dim=1
            )
            mB = from_dlpack(b, assumed_align=get_tensor_alignment(b, dim=-1)).mark_layout_dynamic(
                leading_dim=1
            )
            mC = from_dlpack(c, assumed_align=get_tensor_alignment(c, dim=-1)).mark_layout_dynamic(
                leading_dim=1
            )

            dim_orders = tuple(reversed(a.dim_order()))
            cache_key = f"add_{thr_m}_{thr_n}_{val_m}_{val_n}_{dim_orders}"

            compile_and_cache(
                self,
                cache_key,
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

        return run

    def baseline(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def interface(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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

        thrC.store(op(thrA.load(), thrB.load()))

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


_op = ElementwiseAddOp()
_config_cache: dict[str, Config] = {}


def autotuned_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    cache_key = f"add_{M}x{N}_{a.dtype}"

    if cache_key not in _config_cache:
        _config_cache[cache_key] = autotune(
            tunables=_op.tunables,
            compile_fn=_op.compile,
            baseline_fn=_op.baseline,
            args=(a, b),
            algorithm="LFBOPatternSearch",
            autotune_accuracy_check=True,
            autotune_ignore_errors=True,
        )
        print(f"Best config for {M}x{N}: {dict(_config_cache[cache_key])}")

    return _op.compile(_config_cache[cache_key])(a, b)


if __name__ == "__main__":
    from rich import print as rprint

    shapes = [(1024, 1024), (2048, 2048), (4096, 4096)]

    for M, N in shapes:
        print(f"\n--- {M} x {N} ---")
        a = torch.randn(M, N, device="cuda", dtype=torch.float16)
        b = torch.randn(M, N, device="cuda", dtype=torch.float16)

        out = autotuned_add(a, b)
        torch.testing.assert_close(out, a + b)

        time_torch = benchmark_cuda_function_in_microseconds(lambda: a + b)
        time_cute = benchmark_cuda_function_in_microseconds(lambda: autotuned_add(a, b))

        rprint(f"  PyTorch: {time_torch:.1f} us ({M * N * 3 * 2 / time_torch * 1e-3:.1f} GB/s)")
        rprint(f"  CuTeDSL: {time_cute:.1f} us ({M * N * 3 * 2 / time_cute * 1e-3:.1f} GB/s)")
