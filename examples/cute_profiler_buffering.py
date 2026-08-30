"""Use intra-kernel traces to diagnose and fix a serialized warp pipeline.

Warp 0 loads one tile from global memory into shared memory. Warp 1 waits for that
tile, applies a dependent FMA chain, and stores it. With one shared-memory stage,
the roles run in lockstep. With two stages, the producer can load tile ``i + 1``
while the consumer processes tile ``i``.

Run with:
    python examples/cute_profiler_buffering.py

The script checks correctness, reports unprofiled latency, summarizes the sampled
CTA's compact shared-memory records, and writes one Perfetto trace per variant.
"""

from collections.abc import Callable
from enum import IntEnum
from pathlib import Path
import statistics

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, pipeline
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets import DATA_DIR
from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.cache import compile_and_cache
from transformer_nuggets.cute.profiler import (
    Event,
    PostProcessContext,
    compact_flush_smem_to_gmem,
    compact_prepare_smem,
    compact_profile_region,
    dependency_gaps,
    overlap_between,
    profile_session,
    summarize_by_tag,
)
from transformer_nuggets.cute.profiler.postprocessors import (
    compose,
    rename_processes,
    rename_threads,
)
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds


NUM_BLOCKS = 32
TILES_PER_CTA = 256
VALUES_PER_LANE = 4
COMPUTE_ITERS = 64
THREADS = 64
WARP_SIZE = 32
PRODUCER_WARP = 0
CONSUMER_WARP = 1
PROFILED_CTA = 0
PROFILE_UNIT = 0
EVENTS_PER_TILE = 4
TILE_ELEMENTS = WARP_SIZE * VALUES_PER_LANE
MAX_PROFILE_EVENTS = EVENTS_PER_TILE * TILES_PER_CTA


class ProfileTag(IntEnum):
    PRODUCER_ACQUIRE = 0
    PRODUCER_LOAD = 1
    CONSUMER_WAIT = 2
    CONSUMER_COMPUTE_STORE = 3


PROFILE_TAGS = tuple(tag.name.lower() for tag in ProfileTag)


class WarpSpecializedTransform(CuteOp):
    """Transform tiles with distinct producer and consumer warps."""

    def __init__(self, num_stages: int, enable_profiling: bool = False):
        super().__init__()
        if num_stages not in (1, 2):
            raise ValueError("num_stages must be 1 or 2")
        self.num_stages = num_stages
        self.enable_profiling = enable_profiling

    def shared_storage(self):
        """Return shared storage without profiling state in the production specialization."""
        if self.enable_profiling:

            @cute.struct
            class SharedStorage:
                barriers: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages], 8
                ]
                tiles: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_stages * TILE_ELEMENTS],
                    128,
                ]
                profile: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Int64, 1 + MAX_PROFILE_EVENTS], 8
                ]

        else:

            @cute.struct
            class SharedStorage:
                barriers: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages], 8
                ]
                tiles: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_stages * TILE_ELEMENTS],
                    128,
                ]

        return SharedStorage

    def profile_scope(
        self,
        profile_records,
        tag: ProfileTag,
        event_idx,
        target_warp: int,
        record,
        enable_profiling: cutlass.Constexpr,
    ):
        """Create one compact shared-memory profiling region."""
        return compact_profile_region(
            profile_records,
            Int32(MAX_PROFILE_EVENTS),
            Int32(tag),
            Int32(PROFILE_UNIT),
            event_idx,
            target_warp=Int32(target_warp),
            enabled=enable_profiling,
            record=record,
            smem=True,
        )

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        profile_gmem: cute.Tensor | None,
        shared_storage: cutlass.Constexpr,
        enable_profiling: cutlass.Constexpr,
    ):
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage)
        staged = storage.tiles.get_tensor(
            cute.make_ordered_layout(
                (self.num_stages, VALUES_PER_LANE, WARP_SIZE), order=(2, 1, 0)
            )
        )
        barriers = storage.barriers.data_ptr()

        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        record = block_idx == PROFILED_CTA

        if cutlass.const_expr(enable_profiling):
            assert profile_gmem is not None
            profile_records = storage.profile.get_tensor(
                cute.make_layout((1 + MAX_PROFILE_EVENTS,))
            )
            compact_prepare_smem(profile_records, Int32(MAX_PROFILE_EVENTS), record)
        else:
            profile_records = None

        producer, consumer = pipeline.PipelineAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, WARP_SIZE),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, WARP_SIZE),
            barrier_storage=barriers,
        ).make_participants()

        block_base = block_idx * TILES_PER_CTA * TILE_ELEMENTS

        if warp_idx == PRODUCER_WARP:
            for tile_idx in cutlass.range(TILES_PER_CTA, unroll=1):
                event_base = tile_idx * Int32(EVENTS_PER_TILE)
                with self.profile_scope(
                    profile_records,
                    ProfileTag.PRODUCER_ACQUIRE,
                    event_base,
                    PRODUCER_WARP,
                    record,
                    enable_profiling,
                ):
                    stage = producer.acquire_and_advance()
                with self.profile_scope(
                    profile_records,
                    ProfileTag.PRODUCER_LOAD,
                    event_base + Int32(1),
                    PRODUCER_WARP,
                    record,
                    enable_profiling,
                ):
                    tile_base = block_base + tile_idx * TILE_ELEMENTS
                    for value_idx in cutlass.range_constexpr(VALUES_PER_LANE):
                        staged[(stage.index, value_idx, lane_idx)] = src[
                            tile_base + value_idx * WARP_SIZE + lane_idx
                        ]
                    stage.commit()
            producer.tail()

        if warp_idx == CONSUMER_WARP:
            for tile_idx in cutlass.range(TILES_PER_CTA, unroll=1):
                event_base = tile_idx * Int32(EVENTS_PER_TILE)
                with self.profile_scope(
                    profile_records,
                    ProfileTag.CONSUMER_WAIT,
                    event_base + Int32(2),
                    CONSUMER_WARP,
                    record,
                    enable_profiling,
                ):
                    stage = consumer.wait_and_advance()
                with self.profile_scope(
                    profile_records,
                    ProfileTag.CONSUMER_COMPUTE_STORE,
                    event_base + Int32(3),
                    CONSUMER_WARP,
                    record,
                    enable_profiling,
                ):
                    tile_base = block_base + tile_idx * TILE_ELEMENTS
                    for value_idx in cutlass.range_constexpr(VALUES_PER_LANE):
                        value = staged[(stage.index, value_idx, lane_idx)]
                        for _ in cutlass.range_constexpr(COMPUTE_ITERS):
                            value = value * Float32(1.0001) + Float32(0.0001)
                        dst[tile_base + value_idx * WARP_SIZE + lane_idx] = value
                    stage.release()

        if cutlass.const_expr(enable_profiling):
            assert profile_records is not None and profile_gmem is not None
            if record:
                compact_flush_smem_to_gmem(
                    profile_records,
                    profile_gmem,
                    Int32(PROFILE_UNIT),
                    Int32(MAX_PROFILE_EVENTS),
                )

    @cute.jit
    def __call__(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        profile_gmem: cute.Tensor | None,
    ):
        self.kernel(
            src,
            dst,
            profile_gmem,
            self.shared_storage(),
            self.enable_profiling,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1))

    def get_key(self, src: cute.Tensor, dst: cute.Tensor, profile_gmem) -> str:
        """Return the specialization cache key."""
        return (
            f"warp_transform_stages={self.num_stages}_profile={self.enable_profiling}_"
            f"src={src.shape}_dst={dst.shape}_profile_arg={profile_gmem is not None}"
        )

    def compile(self, src: torch.Tensor, dst: torch.Tensor, profile_gmem: torch.Tensor | None):
        """Compile and return a no-allocation launch callable."""
        src_cute = from_dlpack(src)
        dst_cute = from_dlpack(dst)
        profile_cute = from_dlpack(profile_gmem) if profile_gmem is not None else None
        compiled = compile_and_cache(
            self,
            self.get_key(src_cute, dst_cute, profile_cute),
            src_cute,
            dst_cute,
            profile_cute,
        )
        return lambda: compiled(src_cute, dst_cute, profile_cute)

    def interface(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        profile_gmem: torch.Tensor | None = None,
    ) -> None:
        """Compile and launch the selected specialization once."""
        self.compile(src, dst, profile_gmem)()


def group_profile_roles(events: list[Event], _ctx: PostProcessContext) -> list[Event]:
    """Place producer and consumer events on stable Perfetto lanes."""
    for event in events:
        event.pid = event.unit_id
        event.tid = PRODUCER_WARP if event.tag_id <= ProfileTag.PRODUCER_LOAD else CONSUMER_WARP
    return events


def benchmark_unprofiled(
    launches: dict[int, Callable[[], object]],
) -> dict[int, list[float]]:
    """Collect interleaved timing rounds to reduce order and clock bias."""
    samples = {stages: [] for stages in launches}
    for round_idx in range(5):
        order = (1, 2) if round_idx % 2 == 0 else (2, 1)
        for stages in order:
            samples[stages].append(
                benchmark_cuda_function_in_microseconds(launches[stages], NUM_ITERS=100)
            )
    return samples


def capture_profile(
    num_stages: int, src: torch.Tensor, dst: torch.Tensor, trace_path: Path
) -> dict[str, float]:
    """Capture one warmed profiled launch and return decoded summary statistics."""
    op = WarpSpecializedTransform(num_stages=num_stages, enable_profiling=True)
    with profile_session(
        max_events_per_unit=MAX_PROFILE_EVENTS,
        num_units=(1, "CTA"),
        tag_names=list(PROFILE_TAGS),
        trace_path=str(trace_path),
        compact=True,
        expected_events_per_unit=MAX_PROFILE_EVENTS,
        post_process_events=group_profile_roles,
        post_process_trace=compose(
            rename_processes({PROFILE_UNIT: f"{num_stages}-stage CTA {PROFILED_CTA}"}),
            rename_threads({PRODUCER_WARP: "Producer warp", CONSUMER_WARP: "Consumer warp"}),
        ),
    ) as session:
        launch = op.compile(src, dst, session.prof.tensor)
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        session.prof.reset()
        launch()

    summaries = summarize_by_tag(session.events)
    overlap = overlap_between(
        session.events,
        ProfileTag.PRODUCER_LOAD,
        ProfileTag.CONSUMER_COMPUTE_STORE,
    )
    wait_to_compute = dependency_gaps(
        session.events,
        ProfileTag.CONSUMER_WAIT,
        ProfileTag.CONSUMER_COMPUTE_STORE,
    )
    return {
        **{tag.name.lower(): summaries[tag.name.lower()].p50_ns for tag in ProfileTag},
        "producer_load_overlap_pct": 100.0 * overlap.left_fraction,
        "wait_to_compute_gap_ns": wait_to_compute.p50_ns or 0.0,
    }


def main() -> None:
    """Compare lockstep and double-buffered kernels, then write both traces."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    numel = NUM_BLOCKS * TILES_PER_CTA * TILE_ELEMENTS
    src = torch.linspace(-1.0, 1.0, numel, device="cuda", dtype=torch.float32)
    outputs = {stages: torch.empty_like(src) for stages in (1, 2)}
    launches = {}

    for stages in (1, 2):
        op = WarpSpecializedTransform(num_stages=stages)
        launches[stages] = op.compile(src, outputs[stages], None)
        launches[stages]()
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[1], outputs[2], rtol=0, atol=0)
    expected = src
    for _ in range(COMPUTE_ITERS):
        expected = expected * 1.0001 + 0.0001
    rounding_tolerance = COMPUTE_ITERS * torch.finfo(torch.float32).eps
    torch.testing.assert_close(
        outputs[1], expected, rtol=rounding_tolerance, atol=rounding_tolerance
    )

    timing_samples = benchmark_unprofiled(launches)
    timings = {stages: statistics.median(samples) for stages, samples in timing_samples.items()}

    trace_dir = DATA_DIR
    summaries = {
        stages: capture_profile(
            stages,
            src,
            outputs[stages],
            trace_dir / f"buffering_{stages}_stage.pftrace",
        )
        for stages in (1, 2)
    }

    print("production timing (fixed-pointer, warm-buffer, five interleaved rounds)")
    for stages in (1, 2):
        samples = timing_samples[stages]
        print(
            f"  stages={stages}: median={timings[stages]:.3f} us "
            f"range=[{min(samples):.3f}, {max(samples):.3f}]"
        )
    print(f"  speedup: {timings[1] / timings[2]:.3f}x")
    print("profile summary for sampled CTA 0 (median ns across tiles)")
    for stages in (1, 2):
        values = summaries[stages]
        print(
            f"  stages={stages}: acquire={values['producer_acquire']:.0f}, "
            f"load={values['producer_load']:.0f}, wait={values['consumer_wait']:.0f}, "
            f"compute_store={values['consumer_compute_store']:.0f}, "
            f"load_overlap={values['producer_load_overlap_pct']:.1f}%, "
            f"wait_to_compute_gap={values['wait_to_compute_gap_ns']:.0f}"
        )
        print(f"    trace: {trace_dir / f'buffering_{stages}_stage.pftrace'}")


if __name__ == "__main__":
    main()
