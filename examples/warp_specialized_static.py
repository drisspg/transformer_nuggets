"""Warp-specialized profiling example using static indices and lane-0 tracing.

Run with:
    python examples/warp_specialized_static.py
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import from_dlpack

import transformer_nuggets
from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.profiler import profile_session, group_by_unit, rename_processes
from transformer_nuggets.cute.profiler.ops import profile_region


TAG_PRODUCER = 0
TAG_CONSUMER = 1
NUM_BLOCKS = 4
THREADS_PER_BLOCK = 64
MAX_EVENTS_PER_UNIT = 4
PRODUCER_WORK_ITERS = 200_000
CONSUMER_WORK_ITERS = 120_000


class WarpSpecializedStaticProfile(CuteOp):
    """Profiles producer/consumer warps with static event indices."""

    @cute.kernel
    # pyrefly: ignore [bad-override]
    def kernel(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        block_base = bidx * bdim

        if warp_idx == 0:
            with profile_region(
                prof_buf,
                max_events_per_unit,
                TAG_PRODUCER,
                bidx,
                target_warp=Int32(0),
                event_idx=Int32(0),
            ):
                idx = block_base + lane_idx
                if idx < cute.size(output):
                    # Do extra writes so producer spans a measurable interval.
                    # pyrefly: ignore [not-iterable]
                    for i in cutlass.range(bidx * 2, unroll=-1):
                        # per bidx timing changes
                        output[idx + i] = Float32(i)
                    for i in cutlass.range(Int32(PRODUCER_WORK_ITERS)):
                        output[idx + i] = Float32(i)

        if warp_idx == 1:
            with profile_region(
                prof_buf,
                max_events_per_unit,
                TAG_CONSUMER,
                bidx,
                target_warp=Int32(1),
                event_idx=Int32(1),
            ):
                idx = block_base + Int32(32) + lane_idx
                if idx < cute.size(output):
                    # Make consumer slightly shorter to show overlap.
                    # pyrefly: ignore [not-iterable]
                    for i in cutlass.range(bidx * 2, unroll=-1):
                        # per bidx timing changes
                        output[idx + i] = Float32(i)
                    for i in cutlass.range(Int32(CONSUMER_WORK_ITERS)):
                        output[CONSUMER_WORK_ITERS + idx + i] = Float32(i)

    @cute.jit()
    def __call__(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
    ):
        # pyrefly: ignore [bad-argument-count, missing-attribute]
        self.kernel(output, prof_buf, max_events_per_unit).launch(
            grid=(NUM_BLOCKS, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
        )

    def interface(self, output: torch.Tensor, prof_buf: torch.Tensor, max_events: int):
        self.__call__(from_dlpack(output), from_dlpack(prof_buf), Int32(max_events))


def run():
    """Run the warp-specialized static profiling demo."""
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        return

    device = torch.device("cuda")
    output = torch.zeros(
        NUM_BLOCKS * THREADS_PER_BLOCK * (PRODUCER_WORK_ITERS + CONSUMER_WORK_ITERS),
        dtype=torch.float32,
        device=device,
    )

    trace_path = transformer_nuggets.DATA_DIR / "profiler_warp_static_trace.json"

    process_names = {i: f"CTA {i}" for i in range(NUM_BLOCKS)}

    with profile_session(
        max_events_per_unit=MAX_EVENTS_PER_UNIT,
        num_units=(NUM_BLOCKS, "Warp"),
        tag_names=["producer", "consumer"],
        trace_path=str(trace_path),
        device=device,
        post_process_events=group_by_unit,
        post_process_trace=rename_processes(process_names),
    ) as (prof, tag_table):
        print(f"Tags: {tag_table.names}")
        kernel = WarpSpecializedStaticProfile()
        kernel.interface(output, prof.tensor, prof.max_events_per_unit)

    print(f"Trace: {trace_path}")


if __name__ == "__main__":
    run()
