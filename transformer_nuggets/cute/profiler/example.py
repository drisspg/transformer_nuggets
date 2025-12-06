"""Example demonstrating NVIDIA intra-kernel profiling with CUTE DSL.

This example shows BOTH profiling modes:
1. Atomic mode: No event_idx needed, indices allocated via atomics (simple)
2. Static mode: Explicit event_idx via runtime expressions (no atomics)

Run with:
    python -m transformer_nuggets.cute.profiler.example

View the generated trace.json at https://ui.perfetto.dev/
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.profiler.host import profile_session
from transformer_nuggets.cute.profiler.ops import profile_region


TAG_ITERATION = 0
TAG_COMPUTE = 1
TAG_STORE = 2
NUM_BLOCKS = 4
THREADS_PER_BLOCK = 64


class ProfiledKernelAtomic(CuteOp):
    """Kernel using ATOMIC mode profiling.

    No event_idx needed - indices allocated automatically via atomics.
    Simple to use, slight atomic overhead per region.
    """

    def __init__(self, num_iterations: int = 4):
        super().__init__()
        self.num_iterations = num_iterations

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        prof_tid = bidx
        global_idx = bidx * bdim + tidx

        with profile_region(prof_buf, max_events_per_unit, TAG_ITERATION, prof_tid):
            for i in cutlass.range(self.num_iterations):
                with profile_region(prof_buf, max_events_per_unit, TAG_COMPUTE, prof_tid):
                    with profile_region(prof_buf, max_events_per_unit, TAG_STORE, prof_tid):
                        if global_idx < cute.size(output):
                            val = output[global_idx]
                            output[global_idx] = val + 1

    @cute.jit()
    def __call__(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
    ):
        self.kernel(output, prof_buf, max_events_per_unit).launch(
            grid=(NUM_BLOCKS, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
        )

    def interface(self, output: torch.Tensor, prof_buf: torch.Tensor, max_events: int):
        self.__call__(from_dlpack(output), from_dlpack(prof_buf), Int32(max_events))


class ProfiledKernelStatic(CuteOp):
    """Kernel using STATIC mode profiling.

    Explicit event_idx via runtime expressions - no atomics!
    Maximum performance, requires manual index calculation.
    """

    def __init__(self, num_iterations: int = 4):
        super().__init__()
        self.num_iterations = num_iterations

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        prof_tid = bidx
        global_idx = bidx * bdim + tidx

        with profile_region(
            prof_buf,
            max_events_per_unit,
            TAG_ITERATION,
            prof_tid,
            event_idx=Int32(0),
        ):
            for i in cutlass.range(self.num_iterations):
                with profile_region(
                    prof_buf,
                    max_events_per_unit,
                    TAG_COMPUTE,
                    prof_tid,
                    event_idx=Int32(1) + i * Int32(3),
                ):
                    with profile_region(
                        prof_buf,
                        max_events_per_unit,
                        TAG_STORE,
                        prof_tid,
                        event_idx=Int32(2) + i * Int32(3),
                    ):
                        if global_idx < cute.size(output):
                            val = output[global_idx]
                            output[global_idx] = val + 1

    @cute.jit()
    def __call__(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
    ):
        self.kernel(output, prof_buf, max_events_per_unit).launch(
            grid=(NUM_BLOCKS, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
        )

    def interface(self, output: torch.Tensor, prof_buf: torch.Tensor, max_events: int):
        self.__call__(from_dlpack(output), from_dlpack(prof_buf), Int32(max_events))


def run_atomic_mode():
    """Run the atomic mode example."""
    print("\n" + "=" * 60)
    print("ATOMIC MODE: No event_idx, indices via atomics")
    print("=" * 60)

    device = torch.device("cuda")
    output = torch.zeros(256, dtype=torch.float32, device=device)

    import transformer_nuggets

    trace_path = transformer_nuggets.DATA_DIR / "profiler_atomic_trace.json"

    with profile_session(
        max_events_per_unit=64,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=["iteration", "compute", "store"],
        trace_path=str(trace_path),
        device=device,
    ) as (prof, tag_table):
        print(f"Tags: {tag_table.names}")
        kernel = ProfiledKernelAtomic(num_iterations=4)
        kernel.interface(output, prof.tensor, prof.max_events_per_unit)

    print(f"Trace: {trace_path}")

    expected = torch.full_like(output, 4.0)
    if torch.allclose(output, expected):
        print("✓ Output verification passed!")
    else:
        print("✗ Output verification failed!")


def run_static_mode():
    """Run the static mode example."""
    print("\n" + "=" * 60)
    print("STATIC MODE: Explicit event_idx, no atomics")
    print("=" * 60)

    device = torch.device("cuda")
    output = torch.zeros(256, dtype=torch.float32, device=device)

    import transformer_nuggets

    trace_path = transformer_nuggets.DATA_DIR / "profiler_static_trace.json"

    with profile_session(
        max_events_per_unit=64,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=["iteration", "compute", "store"],
        trace_path=str(trace_path),
        device=device,
    ) as (prof, tag_table):
        print(f"Tags: {tag_table.names}")
        kernel = ProfiledKernelStatic(num_iterations=4)
        kernel.interface(output, prof.tensor, prof.max_events_per_unit)

    print(f"Trace: {trace_path}")

    expected = torch.full_like(output, 4.0)
    if torch.allclose(output, expected):
        print("✓ Output verification passed!")
    else:
        print("✗ Output verification failed!")


def main():
    print("=" * 60)
    print("NVIDIA Intra-Kernel Profiling Example")
    print("Demonstrating BOTH atomic and static modes")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        exit(1)

    run_atomic_mode()
    run_static_mode()

    print("\n" + "=" * 60)
    print("Done! View traces at https://ui.perfetto.dev/")
    print("=" * 60)


if __name__ == "__main__":
    main()
