"""Measure kernel runtime with and without static profiling instrumentation.

Run with:
    python examples/measure_profiler_overhead.py

Toggle profiling codegen by setting TNUGGETS_ENABLE_PROFILING before launch:
    TNUGGETS_ENABLE_PROFILING=0 python examples/measure_profiler_overhead.py
    TNUGGETS_ENABLE_PROFILING=1 python examples/measure_profiler_overhead.py

Note: the flag is read at import / kernel staging time, so you must rerun the
script after changing it.
"""

import torch
import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.profiler.host import allocate_profile_buffer
from transformer_nuggets.cute.profiler.ops import profile_region
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds


TAG_KERNEL = 0
NUM_BLOCKS = 256
THREADS_PER_BLOCK = 128
ELEMENTS = NUM_BLOCKS * THREADS_PER_BLOCK
ITERS = 1000
MAX_EVENTS_PER_UNIT = 1


class BaselineKernel(CuteOp):
    """Simple kernel that writes a constant without profiling."""

    @cute.kernel
    def kernel(self, output: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        idx = bidx * bdim + tidx
        if idx < cute.size(output):
            output[idx] = 1.0

    @cute.jit()
    def __call__(self, output: cute.Tensor):
        self.kernel(output).launch(grid=(NUM_BLOCKS, 1, 1), block=(THREADS_PER_BLOCK, 1, 1))

    def interface(self, output: torch.Tensor):
        self.__call__(from_dlpack(output))


class ProfiledKernel(CuteOp):
    """Same kernel but with static profiling on warp 0, lane 0."""

    @cute.kernel
    def kernel(
        self, output: cute.Tensor, prof_buf: cute.Tensor, max_events_per_unit: cutlass.Int32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        prof_tid = bidx
        idx = bidx * bdim + tidx

        with profile_region(
            prof_buf,
            max_events_per_unit,
            TAG_KERNEL,
            prof_tid,
            target_warp=Int32(0),
            event_idx=Int32(0),
        ):
            if idx < cute.size(output):
                output[idx] = 1.0

    @cute.jit()
    def __call__(
        self, output: cute.Tensor, prof_buf: cute.Tensor, max_events_per_unit: cutlass.Int32
    ):
        self.kernel(output, prof_buf, max_events_per_unit).launch(
            grid=(NUM_BLOCKS, 1, 1), block=(THREADS_PER_BLOCK, 1, 1)
        )

    def interface(self, output: torch.Tensor, prof_buf: torch.Tensor, max_events: int):
        self.__call__(from_dlpack(output), from_dlpack(prof_buf), Int32(max_events))


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        return

    profiling_enabled = os.getenv("TNUGGETS_ENABLE_PROFILING", "1")
    print(f"TNUGGETS_ENABLE_PROFILING={profiling_enabled} (set before launch to toggle)")

    device = torch.device("cuda")
    baseline_out = torch.empty(ELEMENTS, dtype=torch.float32, device=device)
    profiled_out = torch.empty_like(baseline_out)

    baseline = BaselineKernel()
    profiled = ProfiledKernel()

    baseline_us = benchmark_cuda_function_in_microseconds(
        lambda: baseline.interface(baseline_out), NUM_ITERS=ITERS
    )

    prof_buf = allocate_profile_buffer(
        max_events_per_unit=MAX_EVENTS_PER_UNIT,
        num_units=(NUM_BLOCKS, "Block"),
        device=device,
    )

    profiled_us = benchmark_cuda_function_in_microseconds(
        lambda: profiled.interface(profiled_out, prof_buf.tensor, prof_buf.max_events_per_unit),
        NUM_ITERS=ITERS,
    )

    overhead_avg_pct = 100.0 * (profiled_us - baseline_us) / baseline_us

    print(f"Baseline avg:   {baseline_us:.3f} ms")
    print(f"Profiled avg:   {profiled_us:.3f} ms")
    print(f"Overhead avg:   {overhead_avg_pct:.2f}%")


if __name__ == "__main__":
    main()
