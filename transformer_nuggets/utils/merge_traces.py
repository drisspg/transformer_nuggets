"""Merge per-rank Chrome/Perfetto traces into a single multi-process trace."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Annotated

import typer

from transformer_nuggets.utils.track_event import write_track_event_trace

app = typer.Typer(help="Merge per-rank Chrome/Perfetto traces into one file.")


def _open_trace(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _get_min_ts(events: list[dict]) -> float:
    return min(
        (ev["ts"] for ev in events if "ts" in ev and ev.get("ph") != "M"),
        default=0.0,
    )


def _is_native_trace(path: str | Path) -> bool:
    return Path(path).suffix in {".pftrace", ".perfetto-trace"}


# TrackDescriptor.child_ordering value meaning "sort children by sibling_order_rank".
_EXPLICIT_CHILD_ORDERING = 3
# Kineto exports one synthetic "python" process per CUDA device with this rank offset.
_KINETO_DEVICE_PROCESS_RANK = 5_000_000
_THREAD_RANK_BASE = 100
_GLOBAL_TRACK_RANK = 10_000_000
_TID_STRIDE = 1_000_000


def _merge_native_traces(
    input_paths: list[str],
    output_path: str,
    labels: list[str] | None,
) -> None:
    """Merge native traces into one process group per input.

    Every input's sequence, track, and flow ids are remapped into disjoint
    ranges. Each input's processes are folded into a single process named by its
    label: Kineto emits the python process plus one empty "python" process per
    CUDA device, all with small pids that collide across ranks, which otherwise
    pools streams from different ranks into one Perfetto group. Stream,
    annotation, and global tracks are re-parented under the folded process,
    groups are ordered by input index, and python threads sort before GPU
    streams inside a group.
    """
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace

    merged = Trace()
    next_sequence_id = 1
    next_track_uuid = 1
    next_flow_id = 1

    for index, path in enumerate(input_paths):
        source = Trace()
        source.ParseFromString(Path(path).read_bytes())
        sequence_ids: dict[int, int] = {}
        track_uuids: dict[int, int] = {}
        flow_ids: dict[int, int] = {}
        label = labels[index] if labels else f"Rank {index}"
        pid = index + 1

        def remap(mapping: dict[int, int], value: int, next_id: int) -> tuple[int, int]:
            if value == 0:
                return 0, next_id
            mapped = mapping.get(value)
            if mapped is None:
                mapped = next_id
                mapping[value] = mapped
                next_id += 1
            return mapped, next_id

        # Pick the input's primary process (the real one, not a Kineto device
        # process) and alias every other process uuid onto it so their children
        # re-parent through the ordinary uuid remap below.
        process_descriptors = [
            packet.track_descriptor
            for packet in source.packet
            if packet.HasField("track_descriptor") and packet.track_descriptor.HasField("process")
        ]
        real_processes = [
            descriptor
            for descriptor in process_descriptors
            if descriptor.sibling_order_rank < _KINETO_DEVICE_PROCESS_RANK
        ] or process_descriptors
        primary_uuid = real_processes[0].uuid if real_processes else 0
        if primary_uuid:
            _, next_track_uuid = remap(track_uuids, primary_uuid, next_track_uuid)
            for descriptor in process_descriptors:
                track_uuids[descriptor.uuid] = track_uuids[primary_uuid]
        emitted_process = False
        thread_rank = _THREAD_RANK_BASE

        for source_packet in source.packet:
            if source_packet.HasField("track_descriptor"):
                descriptor = source_packet.track_descriptor
                if descriptor.HasField("process") and descriptor.uuid != primary_uuid:
                    continue  # folded into the primary process

            packet = merged.packet.add()
            packet.CopyFrom(source_packet)

            sequence_id, next_sequence_id = remap(
                sequence_ids,
                packet.trusted_packet_sequence_id,
                next_sequence_id,
            )
            packet.trusted_packet_sequence_id = sequence_id

            if packet.HasField("track_descriptor"):
                descriptor = packet.track_descriptor
                descriptor.uuid, next_track_uuid = remap(
                    track_uuids,
                    descriptor.uuid,
                    next_track_uuid,
                )
                descriptor.parent_uuid, next_track_uuid = remap(
                    track_uuids,
                    descriptor.parent_uuid,
                    next_track_uuid,
                )
                if descriptor.HasField("process"):
                    descriptor.process.pid = pid
                    descriptor.process.process_name = label
                    descriptor.sibling_order_rank = index
                    descriptor.child_ordering = _EXPLICIT_CHILD_ORDERING
                    emitted_process = True
                else:
                    if descriptor.HasField("thread"):
                        descriptor.thread.pid = pid
                        descriptor.thread.tid += index * _TID_STRIDE
                    if descriptor.parent_uuid == 0 and primary_uuid:
                        # Global tracks (e.g. Kineto events) join the rank's group.
                        descriptor.parent_uuid = track_uuids[primary_uuid]
                        descriptor.sibling_order_rank = _GLOBAL_TRACK_RANK
                    elif descriptor.name.startswith("thread ") or descriptor.HasField("thread"):
                        descriptor.sibling_order_rank = thread_rank
                        thread_rank += 1
                if descriptor.sibling_merge_key:
                    descriptor.sibling_merge_key = f"{index}:{descriptor.sibling_merge_key}"

            if packet.HasField("track_event"):
                event = packet.track_event
                event.track_uuid, next_track_uuid = remap(
                    track_uuids,
                    event.track_uuid,
                    next_track_uuid,
                )
                for field_name in (
                    "extra_counter_track_uuids",
                    "extra_double_counter_track_uuids",
                ):
                    track_ids = getattr(event, field_name)
                    for position, track_uuid in enumerate(track_ids):
                        track_ids[position], next_track_uuid = remap(
                            track_uuids,
                            track_uuid,
                            next_track_uuid,
                        )
                for field_name in (
                    "flow_ids_old",
                    "flow_ids",
                    "terminating_flow_ids_old",
                    "terminating_flow_ids",
                ):
                    ids = getattr(event, field_name)
                    for position, flow_id in enumerate(ids):
                        ids[position], next_flow_id = remap(
                            flow_ids,
                            flow_id,
                            next_flow_id,
                        )

        assert emitted_process or not primary_uuid, f"{path}: primary process descriptor dropped"

    Path(output_path).write_bytes(merged.SerializeToString())


def merge_traces(
    input_paths: list[str],
    output_path: str,
    labels: list[str] | None = None,
    align_timestamps: bool = False,
) -> None:
    """Merge homogeneous Chrome JSON or native Perfetto traces.

    Output format follows the ``output_path`` suffix: ``.pftrace`` writes a native
    Perfetto TrackEvent protobuf, anything else writes Chrome JSON (gzipped for
    ``.gz``). Native inputs require native output because conversion back to Chrome
    JSON is intentionally unsupported.
    """
    native_inputs = [_is_native_trace(path) for path in input_paths]
    if any(native_inputs):
        if not all(native_inputs):
            raise ValueError("cannot merge a mixture of native Perfetto and Chrome JSON traces")
        if not _is_native_trace(output_path):
            raise ValueError("native Perfetto inputs require a .pftrace or .perfetto-trace output")
        if align_timestamps:
            raise ValueError("timestamp alignment is not supported for native Perfetto inputs")
        _merge_native_traces(input_paths, output_path, labels)
        return

    merged_events: list[dict] = []

    for idx, path in enumerate(input_paths):
        with _open_trace(path, "r") as f:
            data = json.load(f)

        events = data.get("traceEvents", data) if isinstance(data, dict) else data

        ts_offset = _get_min_ts(events) if align_timestamps else 0.0
        label = labels[idx] if labels else f"Rank {idx}"

        merged_events.extend(
            [
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": idx,
                    "tid": 0,
                    "args": {"name": label},
                },
                {
                    "ph": "M",
                    "name": "process_sort_index",
                    "pid": idx,
                    "tid": 0,
                    "args": {"sort_index": idx},
                },
            ]
        )

        for ev in events:
            if ev.get("ph") == "M" and ev.get("name") in {
                "process_name",
                "process_sort_index",
            }:
                continue
            ev["pid"] = idx
            if align_timestamps and "ts" in ev:
                ev["ts"] = ev["ts"] - ts_offset
            if "id" in ev and ev.get("ph") in ("s", "t", "f"):
                ev["id"] = ev["id"] + idx * (1 << 32)
            merged_events.append(ev)

    if output_path.endswith(".pftrace"):
        write_track_event_trace(output_path, {"traceEvents": merged_events})
    else:
        with _open_trace(output_path, "w") as f:
            json.dump({"traceEvents": merged_events}, f, indent=0)


@app.command()
def main(
    traces: Annotated[
        list[Path], typer.Argument(help="Input trace files, one per rank, in rank order.")
    ],
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Output path (.pftrace for native Perfetto, .json/.json.gz for Chrome JSON).",
        ),
    ] = Path("merged_trace.json.gz"),
    label: Annotated[
        list[str] | None,
        typer.Option("-l", "--label", help="Label for each trace (repeat for each file)."),
    ] = None,
    align: Annotated[
        bool, typer.Option("--align", help="Align timestamps so all traces start at t=0.")
    ] = False,
):
    """Merge per-rank Chrome/Perfetto traces into a single multi-process Perfetto trace."""
    for p in traces:
        if not p.exists():
            typer.echo(f"Error: {p} not found", err=True)
            raise typer.Exit(1)

    merge_traces([str(p) for p in traces], str(output), labels=label, align_timestamps=align)
    typer.echo(f"Merged {len(traces)} traces -> {output}")


if __name__ == "__main__":
    app()
