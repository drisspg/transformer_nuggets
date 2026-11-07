"""NVIDIA Intra-Kernel Profiling for CUTE DSL Kernels.

Record code regions *inside* GPU kernels into a profile buffer, decode them on the host, and
export native Perfetto TrackEvent traces (https://ui.perfetto.dev/). See README.md.

Quick Start:
    from transformer_nuggets.cute.profiler import (
        compact_anchor_init, compact_profile_region, profile_session
    )

    @cute.kernel
    def my_kernel(output, prof_buf, max_events):
        bidx, _, _ = cute.arch.block_idx()
        compact_anchor_init(prof_buf, bidx, max_events)
        with compact_profile_region(prof_buf, max_events, TAG_COMPUTE, bidx, Int32(0)):
            compute_something()

    with profile_session(
        max_events_per_unit=1,
        num_units=(num_blocks, "Block"),
        tag_names=["compute"],
        trace_path="trace.pftrace",
        compact=True,
    ) as session:
        my_kernel(output, session.prof.tensor, session.prof.max_events_per_unit)
    session.ranges  # decoded after the block
"""

from transformer_nuggets.cute.profiler.host import (
    ProfileBuf,
    ProfileSession,
    TagTable,
    allocate_profile_buffer,
    decode_ranges,
    profile_session,
)
from transformer_nuggets.cute.profiler.iket import parse_iket
from transformer_nuggets.cute.profiler.ops import (
    compact_anchor_init,
    compact_event_stop,
    compact_event_stop_smem,
    compact_flush_smem_to_gmem,
    compact_prepare_smem,
    compact_profile_region,
    event_stop,
    read_globaltimer,
    read_globaltimer_lo32,
)
from transformer_nuggets.cute.profiler.perfetto import write_ranges
from transformer_nuggets.cute.profiler.ranges import (
    OverlapSummary,
    Range,
    RangeSummary,
    overlap_between,
    summarize_by_tag,
)

__all__ = [
    # Device
    "compact_anchor_init",
    "compact_event_stop",
    "compact_event_stop_smem",
    "compact_flush_smem_to_gmem",
    "compact_prepare_smem",
    "compact_profile_region",
    "event_stop",
    "read_globaltimer",
    "read_globaltimer_lo32",
    # Host capture
    "ProfileBuf",
    "ProfileSession",
    "TagTable",
    "allocate_profile_buffer",
    "decode_ranges",
    "profile_session",
    # Ranges, statistics and traces
    "OverlapSummary",
    "Range",
    "RangeSummary",
    "overlap_between",
    "parse_iket",
    "summarize_by_tag",
    "write_ranges",
]
