"""NVIDIA Intra-Kernel Profiling for CUTE DSL Kernels.

This package provides utilities for profiling code regions *inside* GPU kernels,
generating Chrome trace format files viewable in Perfetto (https://ui.perfetto.dev/).

Modules:
    ops: Device-side profiling operations (inline PTX, start/stop helpers)
    host: Host-side utilities (buffer allocation, decoding, Perfetto export)
    example: Complete working example

Quick Start:
    from transformer_nuggets.cute.profiler import (
        # Host-side
        TagTable,
        allocate_profile_buffer,
        decode_events,
        events_to_perfetto,
        profile_session,
        # Device-side
        profile_region,
        lane0_warp0_start,
        lane0_warp0_stop,
    )

See the README.md in this directory for full documentation.
"""

# Host-side exports
from transformer_nuggets.cute.profiler.host import (
    ProfileBuf,
    TagTable,
    Event,
    allocate_profile_buffer,
    decode_events,
    events_to_perfetto,
    profile_session,
)

# Device-side exports
from transformer_nuggets.cute.profiler.ops import (
    read_globaltimer,
    profile_start,
    profile_stop,
    lane0_warp0_start,
    lane0_warp0_stop,
    elected_start,
    elected_stop,
    warp_start,
    warp_stop,
    ProfileRegion,
    profile_region,
)

__all__ = [
    # Host-side
    "ProfileBuf",
    "TagTable",
    "Event",
    "allocate_profile_buffer",
    "decode_events",
    "events_to_perfetto",
    "profile_session",
    # Device-side
    "read_globaltimer",
    "profile_start",
    "profile_stop",
    "lane0_warp0_start",
    "lane0_warp0_stop",
    "elected_start",
    "elected_stop",
    "warp_start",
    "warp_stop",
    "ProfileRegion",
    "profile_region",
]
