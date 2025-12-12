"""NVIDIA Intra-Kernel Profiling for CUTE DSL Kernels.

This package provides utilities for profiling code regions *inside* GPU kernels,
generating Chrome trace format files viewable in Perfetto (https://ui.perfetto.dev/).

Two modes are supported:
- Atomic mode: No event_idx needed, indices allocated via atomics (simple)
- Static mode: Explicit event_idx, no atomics (maximum performance)

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
        trace_path="trace.json",
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
    static_start,
    static_stop,
    warp_atomic_alloc,
    warp_start,
    warp_stop,
    profile_region,
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
