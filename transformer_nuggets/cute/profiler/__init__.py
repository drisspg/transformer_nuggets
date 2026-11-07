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
    session.events  # decoded after the block
"""

from transformer_nuggets.cute.profiler.host import (
    ProfileBuf,
    TagTable,
    Event,
    PostProcessContext,
    ProfileSession,
    allocate_profile_buffer,
    decode_events,
    events_to_perfetto,
    profile_session,
    validate_event_counts,
)

from transformer_nuggets.cute.profiler.analysis import (
    EventSummary,
    OverlapSummary,
    DependencyGapSummary,
    summarize_by_tag,
    overlap_between,
    dependency_gaps,
)

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

from transformer_nuggets.cute.profiler.postprocessors import (
    group_by_unit,
    group_by_tag,
    strip_tid_suffix,
    prefix_tag_with_unit,
    filter_by_tag,
    compose,
    rename_processes,
    rename_threads,
    link_dependency_flow,
    link_payload_flow,
)

__all__ = [
    # Host-side
    "ProfileBuf",
    "TagTable",
    "Event",
    "PostProcessContext",
    "ProfileSession",
    # Host-side analysis
    "EventSummary",
    "OverlapSummary",
    "DependencyGapSummary",
    "summarize_by_tag",
    "overlap_between",
    "dependency_gaps",
    "allocate_profile_buffer",
    "decode_events",
    "events_to_perfetto",
    "profile_session",
    "validate_event_counts",
    # Device-side
    "read_globaltimer",
    "read_globaltimer_lo32",
    "event_stop",
    "compact_event_stop",
    "compact_anchor_init",
    "compact_event_stop_smem",
    "compact_prepare_smem",
    "compact_flush_smem_to_gmem",
    "compact_profile_region",
    # Post-processors
    "group_by_unit",
    "group_by_tag",
    "strip_tid_suffix",
    "prefix_tag_with_unit",
    "filter_by_tag",
    "compose",
    "rename_processes",
    "rename_threads",
    "link_dependency_flow",
    "link_payload_flow",
]
