"""Emit measured CuTeDSL pipeline regions and direct dependencies as Perfetto TrackEvents."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any

from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor, TrackEvent

from transformer_nuggets.cute.profiler.pipeline.analysis import (
    PipelineAnalysis,
    ring_occupancy_samples,
)
from transformer_nuggets.cute.profiler.pipeline.iket import MeasuredCapture, MeasuredRegion
from transformer_nuggets.cute.profiler.pipeline.plan import Timeline
from transformer_nuggets.utils.track_event import default_track_event_path


def stable_id(*parts: object) -> int:
    """Return a deterministic nonzero uint64 identifier."""
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") or 1


def add_annotation(event: Any, name: str, value: object) -> None:
    """Attach one scalar or string debug annotation to a TrackEvent."""
    annotation = event.debug_annotations.add()
    annotation.name = name
    if isinstance(value, bool):
        annotation.bool_value = value
    elif isinstance(value, int):
        annotation.int_value = value
    elif isinstance(value, float):
        annotation.double_value = value
    else:
        annotation.string_value = str(value)


def assign_role_lanes(
    measured: MeasuredCapture,
) -> tuple[dict[tuple[str, int], int], dict[str, int]]:
    """Assign crossing role intervals to non-overlapping backing tracks."""
    regions_by_role: dict[str, list[MeasuredRegion]] = defaultdict(list)
    for region in measured.regions:
        regions_by_role[region.role].append(region)

    lane_by_region: dict[tuple[str, int], int] = {}
    lane_count_by_role: dict[str, int] = {}
    for role, regions in regions_by_role.items():
        lane_ends: list[int] = []
        for region in sorted(regions, key=lambda item: (item.start_ns, item.end_ns)):
            lane = next(
                (index for index, end_ns in enumerate(lane_ends) if end_ns <= region.start_ns),
                len(lane_ends),
            )
            if lane == len(lane_ends):
                lane_ends.append(region.end_ns)
            else:
                lane_ends[lane] = region.end_ns
            lane_by_region[(region.name, region.iteration)] = lane
        lane_count_by_role[role] = len(lane_ends)
    return lane_by_region, lane_count_by_role


def write_pipeline_perfetto(
    path: str | Path,
    timeline: Timeline,
    measured: MeasuredCapture,
    *,
    analysis: PipelineAnalysis | None = None,
) -> Path:
    """Write one measured-role trace enriched with plan, flow, and analysis metadata."""
    output = default_track_event_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_build_pipeline_perfetto(timeline, measured, analysis))
    return output


def _build_pipeline_perfetto(
    timeline: Timeline,
    measured: MeasuredCapture,
    analysis: PipelineAnalysis | None,
) -> bytes:
    """Build a native Perfetto trace for one measured CTA."""
    stage_analysis = analysis.perfetto_annotations() if analysis is not None else {}
    summary = analysis.perfetto_summary() if analysis is not None else {}
    ring_counters = ring_occupancy_samples(timeline, measured) if analysis is not None else {}
    builder = TraceProtoBuilder()
    process_uuid = stable_id("pipeline", timeline.plan_name, measured.cta)
    descriptor = builder.add_packet().track_descriptor
    descriptor.uuid = process_uuid
    mode = (
        "logical plan (synthetic ticks)"
        if measured.source.startswith("logical")
        else "measured pipeline"
    )
    descriptor.process.pid = stable_id("pipeline-pid", timeline.plan_name, measured.cta) % (2**31)
    descriptor.process.process_name = f"{timeline.plan_name} · CTA {measured.cta} · {mode}"
    descriptor.child_ordering = TrackDescriptor.EXPLICIT

    role_by_name = {role.name: role for role in timeline.roles}
    lane_by_region, lane_count_by_role = assign_role_lanes(measured)
    track_by_role_lane: dict[tuple[str, int], int] = {}
    for rank, role in enumerate(timeline.roles):
        lane_count = lane_count_by_role.get(role.name, 1)
        for lane in range(lane_count):
            track_uuid = stable_id(
                "pipeline-role", timeline.plan_name, measured.cta, role.name, lane
            )
            track_by_role_lane[(role.name, lane)] = track_uuid
            descriptor = builder.add_packet().track_descriptor
            descriptor.uuid = track_uuid
            descriptor.parent_uuid = process_uuid
            suffix = "" if lane == 0 else f" overlap {lane}"
            descriptor.name = f"{role.label} · warps {role.warp_start}:{role.warp_end}{suffix}"
            descriptor.sibling_order_rank = rank * 100 + lane
            if lane_count > 1:
                descriptor.sibling_merge_behavior = (
                    TrackDescriptor.SIBLING_MERGE_BEHAVIOR_BY_SIBLING_MERGE_KEY
                )
                descriptor.sibling_merge_key = f"{measured.cta}:{role.name}"

    analysis_track = stable_id("pipeline-analysis", timeline.plan_name, measured.cta)
    if summary:
        descriptor = builder.add_packet().track_descriptor
        descriptor.uuid = analysis_track
        descriptor.parent_uuid = process_uuid
        descriptor.name = "Scheduling analysis"
        descriptor.sibling_order_rank = len(timeline.roles)
        packet = builder.add_packet()
        packet.timestamp = 0
        packet.trusted_packet_sequence_id = 2001
        event = packet.track_event
        event.type = TrackEvent.TYPE_INSTANT
        event.track_uuid = analysis_track
        event.name = "Critical path / cycle / depth summary"
        for name, value in summary.items():
            add_annotation(event, name, value)

    resource_by_name = {resource.name: resource for resource in timeline.resources}
    counter_tracks: dict[str, int] = {}
    for rank, resource_name in enumerate(ring_counters, start=len(timeline.roles) + 1):
        track_uuid = stable_id(
            "pipeline-ring-counter", timeline.plan_name, measured.cta, resource_name
        )
        counter_tracks[resource_name] = track_uuid
        descriptor = builder.add_packet().track_descriptor
        descriptor.uuid = track_uuid
        descriptor.parent_uuid = process_uuid
        descriptor.name = (
            f"ring {resource_name} occupancy / depth {resource_by_name[resource_name].depth}"
        )
        descriptor.sibling_order_rank = rank
        descriptor.counter.SetInParent()

    planned = {(item.region.name, item.iteration): item for item in timeline.regions}
    measured_regions = {(item.name, item.iteration): item for item in measured.regions}
    outgoing: dict[tuple[str, int], list[int]] = defaultdict(list)
    incoming: dict[tuple[str, int], list[int]] = defaultdict(list)
    edge_metadata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for edge_index, edge in enumerate(timeline.dependencies):
        source = edge.dependency.source, edge.source_iteration
        target = edge.dependency.target, edge.target_iteration
        if source not in measured_regions or target not in measured_regions:
            continue
        flow_id = stable_id("pipeline-flow", timeline.plan_name, edge_index, source, target)
        outgoing[source].append(flow_id)
        incoming[target].append(flow_id)
        edge_description = (
            f"{edge.dependency.kind}:{edge.dependency.resource}:"
            f"{source[0]}[{source[1]}]->{target[0]}[{target[1]}]"
        )
        edge_metadata[source].append(f"unblocks {edge_description}")
        edge_metadata[target].append(f"depends_on {edge_description}")

    markers: list[tuple[int, bool, MeasuredRegion]] = []
    for region in measured.regions:
        markers.append((region.start_ns - measured.origin_ns, True, region))
        markers.append((region.end_ns - measured.origin_ns, False, region))
    markers.sort(key=lambda marker: (marker[0], marker[1], marker[2].role, marker[2].name))

    for timestamp_ns, is_begin, region in markers:
        packet = builder.add_packet()
        packet.timestamp = timestamp_ns
        packet.trusted_packet_sequence_id = 2001
        event = packet.track_event
        lane = lane_by_region[(region.name, region.iteration)]
        event.track_uuid = track_by_role_lane[(region.role, lane)]
        if not is_begin:
            event.type = TrackEvent.TYPE_SLICE_END
            continue

        key = region.name, region.iteration
        plan_region = planned[key]
        role = role_by_name[region.role]
        event.type = TrackEvent.TYPE_SLICE_BEGIN
        event.name = f"{plan_region.region.label} · chunk {region.iteration}"
        event.flow_ids.extend(outgoing.get(key, ()))
        event.terminating_flow_ids.extend(incoming.get(key, ()))
        add_annotation(event, "region_id", region.name)
        add_annotation(event, "iteration", region.iteration)
        add_annotation(event, "role", region.role)
        add_annotation(event, "warp_start", role.warp_start)
        add_annotation(event, "warp_end", role.warp_end)
        add_annotation(event, "start_ns", region.start_ns - measured.origin_ns)
        add_annotation(event, "duration_ns", region.end_ns - region.start_ns)
        add_annotation(event, "median_duration_ns", region.median_duration_ns)
        add_annotation(event, "warp_samples", region.sample_count)
        add_annotation(event, "logical_start", plan_region.start)
        add_annotation(event, "logical_end", plan_region.end)
        add_annotation(event, "logical_weight", plan_region.region.weight)
        if plan_region.region.source is not None:
            add_annotation(
                event,
                "source",
                f"{plan_region.region.source.path}:{plan_region.region.source.line}",
            )
        for index, description in enumerate(edge_metadata.get(key, ())):
            add_annotation(event, f"dependency[{index}]", description)
        for name, value in stage_analysis.get(key, {}).items():
            add_annotation(event, name, value)

    for resource_name, samples in ring_counters.items():
        for timestamp_ns, occupancy in samples:
            packet = builder.add_packet()
            packet.timestamp = timestamp_ns
            packet.trusted_packet_sequence_id = 2001
            event = packet.track_event
            event.type = TrackEvent.TYPE_COUNTER
            event.track_uuid = counter_tracks[resource_name]
            event.counter_value = occupancy

    return builder.serialize()
