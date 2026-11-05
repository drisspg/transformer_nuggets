"""Replay and provenance mechanics shared by the examples' CUDA-graph benchmarks.

``bench_fa4_decode.py`` keeps its own copies: ``docs/examples.md`` runs it against an archived
package that has no helper module."""

import hashlib
import importlib.metadata
import os
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path

import torch


def settle(graphs: Iterable[torch.cuda.CUDAGraph], seconds: float) -> None:
    """Replay the graphs in turn for ``seconds`` so DVFS settles before the first timed round."""
    graphs = list(graphs)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()


def replay_timed(
    graph: torch.cuda.CUDAGraph, reps: int, warmup: int
) -> tuple[list[float], dict[str, list[float]]]:
    """After ``warmup`` untimed replays, the CUDA-event time of each of ``reps`` whole replays
    and CUPTI's device time of every kernel occurrence, grouped by kernel name; microseconds."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for _ in range(warmup):
        graph.replay()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for s, e in zip(starts, ends):
            s.record()
            graph.replay()
            e.record()
        torch.cuda.synchronize()
    kernels = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.device_time > 0:
            kernels.setdefault(ev.name, []).append(ev.device_time)
    return [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)], kernels


def source_metadata(source: Path) -> dict:
    """The source path, its checkout's commit and dirty flag, and the SHA-256 of the file as it
    is (uncommitted edits included); commit and dirty are None without Git metadata."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(source.parent), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "source": str(source),
        "commit": commit,
        "dirty": dirty,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def provenance(sources: dict[str, Path], arguments: dict) -> dict:
    """The run's metadata: package versions, the GPU, the reservation, ``source_metadata`` of
    each named source and the benchmark's arguments."""
    return {
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "triton", "nvidia-cutlass-dsl")
        },
        "gpu": torch.cuda.get_device_name(),
        "reserved_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **{name: source_metadata(path) for name, path in sources.items()},
        "arguments": arguments,
    }
