"""Measure kernel runtime with and without compact profiling instrumentation.

Run with:
    python examples/measure_profiler_overhead.py

The same kernel is compiled twice: ``enabled=False`` specializes the profiling region away, so
the unprofiled build emits no timer reads or stores.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.profiler import (
    allocate_profile_buffer,
    compact_anchor_init,
    compact_profile_region,
)
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds


TAG_KERNEL = 0
NUM_BLOCKS = 256
THREADS_PER_BLOCK = 128
ELEMENTS = NUM_BLOCKS * THREADS_PER_BLOCK
ITERS = 1000
MAX_EVENTS_PER_UNIT = 1


class ConstantKernel(CuteOp):
    """Write a constant; with profiling, warp 0 of each block records the store as one region."""

    def __init__(self, enable_profiling: bool):
        super().__init__()
        self.enable_profiling = enable_profiling

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
        enable_profiling: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        idx = bidx * bdim + tidx

        if cutlass.const_expr(enable_profiling):
            compact_anchor_init(prof_buf, bidx, max_events_per_unit)
        with compact_profile_region(
            prof_buf,
            max_events_per_unit,
            Int32(TAG_KERNEL),
            bidx,
            Int32(0),
            enabled=enable_profiling,
        ):
            if idx < cute.size(output):
                output[idx] = 1.0

    @cute.jit()
    def __call__(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
        enable_profiling: cutlass.Constexpr,
    ):
        self.kernel(output, prof_buf, max_events_per_unit, enable_profiling).launch(
            grid=(NUM_BLOCKS, 1, 1), block=(THREADS_PER_BLOCK, 1, 1)
        )

    def interface(self, output: torch.Tensor, prof_buf: torch.Tensor, max_events: int):
        self.__call__(
            from_dlpack(output), from_dlpack(prof_buf), Int32(max_events), self.enable_profiling
        )


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        return

    device = torch.device("cuda")
    output = torch.empty(ELEMENTS, dtype=torch.float32, device=device)
    prof_buf = allocate_profile_buffer(
        max_events_per_unit=MAX_EVENTS_PER_UNIT,
        num_units=(NUM_BLOCKS, "Block"),
        device=device,
        compact=True,
    )

    times_us = {}
    for enabled in (False, True):
        kernel = ConstantKernel(enable_profiling=enabled)
        times_us[enabled] = benchmark_cuda_function_in_microseconds(
            lambda kernel=kernel: kernel.interface(
                output, prof_buf.tensor, prof_buf.max_events_per_unit
            ),
            NUM_ITERS=ITERS,
        )

    overhead_avg_pct = 100.0 * (times_us[True] - times_us[False]) / times_us[False]
    print(f"Unprofiled avg: {times_us[False]:.3f} us")
    print(f"Profiled avg:   {times_us[True]:.3f} us")
    print(f"Overhead avg:   {overhead_avg_pct:.2f}%")


if __name__ == "__main__":
    main()
