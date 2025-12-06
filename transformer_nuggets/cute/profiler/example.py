"""Example demonstrating NVIDIA intra-kernel profiling with CUTE DSL.

This example shows how to:
1. Set up a tag table for named events
2. Allocate a profile buffer
3. Instrument a CUTE kernel with profiling helpers
4. Decode events and export to Perfetto trace

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
from transformer_nuggets.cute.profiler.host import (
    TagTable,
    allocate_profile_buffer,
    decode_events,
    events_to_perfetto,
    profile_session,
)
from transformer_nuggets.cute.profiler.ops import (
    lane0_warp0_start,
    lane0_warp0_stop,
    profile_region,  # DSL context manager - use with `with` statement!
    # Also available:
    # ProfileRegion              - helper class for start()/stop() pattern
    # elected_start, elected_stop  - profile from one lane per warp (all warps)
    # warp_start, warp_stop        - profile from a specific warp only
)


# Define tag IDs as module-level constants (these match the order in TagTable)
TAG_ITERATION = 0
TAG_COMPUTE = 1  # Includes load, compute, and store
TAG_BARRIER = 2  # Final sync barrier


class ProfiledKernelExample(CuteOp):
    """Example kernel demonstrating intra-kernel profiling.

    Follows the CuteOp pattern:
    - kernel(): @cute.kernel decorated GPU code
    - __call__(): @cute.jit decorated launcher
    - interface(): Public Python API (handles torch↔cute conversion)
    """

    def __init__(self, num_iterations: int = 4):
        super().__init__()
        self.num_iterations = num_iterations

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events: cutlass.Int32,
    ):
        """A simple kernel that profiles each iteration of a loop.

        This kernel demonstrates:
        - Using lane0_warp0_start/stop for low-contention profiling
        - Using ProfileRegion for cleaner syntax (see store section)
        - Profiling multiple regions with different tags
        - Computing tid from blockIdx
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        # Use block index directly as tid for cleaner Perfetto visualization
        # Each block gets its own row in the trace (Block 0, Block 1, Block 2, Block 3)
        prof_tid = bidx

        # Global thread index for computation
        global_idx = bidx * bdim + tidx

        # =====================================================================
        # Method 1: Explicit start/stop (verbose but clear)
        # =====================================================================

        # Profile the entire iteration loop
        eid_iter = lane0_warp0_start(prof_buf, max_events)

        # Simple computation loop
        for i in cutlass.range(self.num_iterations):
            # Profile compute phase using explicit start/stop
            eid_compute = lane0_warp0_start(prof_buf, max_events)
            if global_idx < cute.size(output):
                val = output[global_idx]
                output[global_idx] = val + cutlass.Float32(1.0)
            lane0_warp0_stop(prof_buf, eid_compute, Int32(TAG_COMPUTE), prof_tid, max_events)

        # End iteration profiling
        lane0_warp0_stop(prof_buf, eid_iter, Int32(TAG_ITERATION), prof_tid, max_events)

        # =====================================================================
        # Method 2: `with` statement context manager (cleanest!)
        # =====================================================================

        # Profile a final sync barrier using `with` - demonstrates the cleaner syntax
        with profile_region(prof_buf, max_events, Int32(TAG_BARRIER), prof_tid):
            cute.arch.sync_threads()

    @cute.jit()
    def __call__(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events: cutlass.Int32,
    ):
        self.kernel(output, prof_buf, max_events).launch(
            grid=(4, 1, 1),  # 4 blocks
            block=(64, 1, 1),  # 64 threads per block (2 warps)
        )

    def interface(
        self,
        output: torch.Tensor,
        prof_buf: torch.Tensor,
        max_events: int,
    ) -> None:
        """Public entrypoint - runs the kernel with profiling.

        Args:
            output: Output tensor to modify (torch.Tensor).
            prof_buf: Profile buffer tensor (torch.int64).
            max_events: Maximum number of profiling events.
        """
        output_cute = from_dlpack(output)
        prof_cute = from_dlpack(prof_buf)
        self.__call__(output_cute, prof_cute, Int32(max_events))


def run_profiled_example():
    """Run the profiled kernel example and generate a Perfetto trace."""
    print("=" * 60)
    print("NVIDIA Intra-Kernel Profiling Example")
    print("=" * 60)

    # Setup
    device = torch.device("cuda")
    output = torch.zeros(256, dtype=torch.float32, device=device)

    # Define tags for this session
    tag_table = TagTable(["iteration", "compute", "barrier"])
    print(f"\nTag table: {tag_table.names}")
    print(f"  TAG_ITERATION = {tag_table.id('iteration')}")
    print(f"  TAG_COMPUTE = {tag_table.id('compute')}")
    print(f"  TAG_BARRIER = {tag_table.id('barrier')}")

    # Allocate profile buffer
    # We have 4 blocks, each with 2 warps, but only warp 0 profiles
    # Expected events per block: 1 (iteration) + 4 (compute) + 1 (barrier) = 6
    # Total: 4 blocks * 6 = 24 events (plus some margin)
    max_events = 64
    prof = allocate_profile_buffer(max_events, device=device)
    print(f"\nProfile buffer: {prof.tensor.shape[0]} int64s for {max_events} events")

    # Create and run the profiled kernel
    kernel = ProfiledKernelExample(num_iterations=4)
    print("\nRunning profiled kernel...")
    kernel.interface(output, prof.tensor, max_events)
    torch.cuda.synchronize()

    # Decode events
    events, overflow = decode_events(prof, tag_table, pid=0)
    print(f"\nDecoded {len(events)} events (overflow: {overflow})")

    # Print events summary
    print("\nEvents by tag:")
    tag_counts = {}
    for event in events:
        tag_counts[event.tag_name] = tag_counts.get(event.tag_name, 0) + 1
    for tag_name, count in sorted(tag_counts.items()):
        print(f"  {tag_name}: {count} events")

    # Print first few events
    print("\nFirst 10 events:")
    for i, event in enumerate(events[:10]):
        print(
            f"  [{i}] {event.tag_name:12s} tid={event.tid:2d} "
            f"start={event.start_ns:15d} dur={event.dur_ns:10d} ns"
        )

    # Export to Perfetto trace
    import transformer_nuggets

    trace_path = transformer_nuggets.DATA_DIR / "profiler_example_trace.json"
    events_to_perfetto(events, str(trace_path), pid=0)
    print(f"\nTrace written to: {trace_path}")
    print("View at: https://ui.perfetto.dev/")

    # Verify output
    expected = torch.full_like(output, 4.0)  # 4 iterations, each adds 1.0
    if torch.allclose(output, expected):
        print("\n✓ Output verification passed!")
    else:
        print("\n✗ Output verification failed!")
        print(f"  Expected: {expected[:8]}")
        print(f"  Got: {output[:8]}")

    return events, overflow


def run_with_context_manager():
    """Demonstrate using the profile_session context manager."""
    print("\n" + "=" * 60)
    print("Using profile_session context manager")
    print("=" * 60)

    device = torch.device("cuda")
    output = torch.zeros(256, dtype=torch.float32, device=device)

    import transformer_nuggets

    trace_path = transformer_nuggets.DATA_DIR / "profiler_example_ctx_trace.json"

    with profile_session(
        max_events=64,
        tag_names=["iteration", "compute", "barrier"],
        trace_path=str(trace_path),
        device=device,
        pid=0,
    ) as (prof, tag_table):
        print(f"\nAllocated buffer with {prof.max_events} max events")
        print(f"Tags: {tag_table.names}")

        kernel = ProfiledKernelExample(num_iterations=4)
        kernel.interface(output, prof.tensor, prof.max_events)

    print("\nContext manager automatically decoded and wrote trace")
    print(f"Trace written to: {trace_path}")


if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        exit(1)

    # Run examples
    run_profiled_example()
    run_with_context_manager()

    print("\n" + "=" * 60)
    print("Done! Open the trace files in Perfetto to visualize.")
    print("=" * 60)
