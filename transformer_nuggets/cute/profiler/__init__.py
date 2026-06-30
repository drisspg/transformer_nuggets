"""NVIDIA Intra-Kernel Profiling for CUTE DSL Kernels.

This package provides utilities for profiling code regions *inside* GPU kernels,
generating native Perfetto TrackEvent traces viewable in Perfetto (https://ui.perfetto.dev/).

See README.md for the full list of recording modes (atomic / static / token /
compact gmem / compact smem) and a guide on which to pick.

Quick Start:
    from transformer_nuggets.cute.profiler import profile_session, profile_region

    @cute.kernel
    def my_kernel(output, prof_buf, max_events):
        bidx, _, _ = cute.arch.block_idx()
        # Atomic mode: just omit event_idx
        with profile_region(prof_buf, max_events, TAG_COMPUTE, bidx):
            compute_something()

    with profile_session(
        max_events_per_unit=64,
        num_units=(num_blocks, "Block"),
        tag_names=["compute"],
        trace_path="trace.pftrace",
    ) as (prof, tag_table):
        my_kernel(output, prof.tensor, prof.max_events_per_unit)

See the README.md in this directory for full documentation.
"""

from transformer_nuggets.cute.profiler.host import (
    ProfileBuf,
    TagTable,
    Event,
    PostProcessContext,
    allocate_profile_buffer,
    decode_events,
    events_to_perfetto,
    profile_session,
)

from transformer_nuggets.cute.profiler.ops import (
    read_globaltimer,
    read_globaltimer_lo32,
    static_start,
    static_stop,
    warp_atomic_alloc,
    warp_start,
    warp_stop,
    profile_region,
    region_start,
    region_end,
    RegionToken,
    raw_event_stop,
    compact_event_stop,
    compact_anchor_init,
    compact_event_stop_smem,
    compact_anchor_init_smem,
    compact_flush_smem_to_gmem,
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
)

__all__ = [
    # Host-side
    "ProfileBuf",
    "TagTable",
    "Event",
    "PostProcessContext",
    "allocate_profile_buffer",
    "decode_events",
    "events_to_perfetto",
    "profile_session",
    # Device-side
    "read_globaltimer",
    "static_start",
    "static_stop",
    "warp_atomic_alloc",
    "warp_start",
    "warp_stop",
    "profile_region",
    "read_globaltimer_lo32",
    "region_start",
    "region_end",
    "RegionToken",
    "raw_event_stop",
    "compact_event_stop",
    "compact_anchor_init",
    "compact_event_stop_smem",
    "compact_anchor_init_smem",
    "compact_flush_smem_to_gmem",
    # Post-processors
    "group_by_unit",
    "group_by_tag",
    "strip_tid_suffix",
    "prefix_tag_with_unit",
    "filter_by_tag",
    "compose",
    "rename_processes",
    "rename_threads",
]
