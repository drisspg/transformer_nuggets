import gc
import gzip
import json
from pathlib import Path

import pytest

from transformer_nuggets.utils import benchmark
from transformer_nuggets.utils.benchmark import (
    DEFAULT_CUPTI_MONITOR_PM_METRICS,
    CuptiMonitorConfig,
    _CudaGraphAnnotationSnapshot,
    _write_cupti_monitor_trace,
)


def test_cupti_monitor_defaults_enable_rich_trace_data():
    config = CuptiMonitorConfig()

    assert config.environment_counters
    assert config.pm_metrics == DEFAULT_CUPTI_MONITOR_PM_METRICS
    assert config.graph_dependencies
    assert config.event_node_ids


def test_cupti_monitor_config_builds_pytorch_options():
    config = CuptiMonitorConfig(
        environment_counters=True,
        pm_metrics=("sm__cycles_active.avg.pct_of_peak_sustained_elapsed",),
        graph_dependencies=True,
        event_node_ids=True,
        cuda_sync_events=True,
        pftrace_compression_level=4,
    )

    assert config.custom_profiler_config() == {
        "backend": "cupti_monitor",
        "enable_environment_counters": True,
        "enable_pm_sampling": True,
        "pm_metrics": ["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"],
        "enable_graph_dependencies": True,
        "enable_event_node_ids": True,
        "enable_cuda_sync_events": True,
        "pftrace_compression_level": 4,
    }


@pytest.mark.parametrize("level", [-1, 10])
def test_cupti_monitor_config_rejects_invalid_compression(level):
    with pytest.raises(ValueError, match="between 0 and 9"):
        CuptiMonitorConfig(pftrace_compression_level=level)


class GzipExportProfiler:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.export_paths = []

    def export_chrome_trace(self, path: str) -> None:
        self.export_paths.append(path)
        with gzip.open(f"{path}.gz", "wb") as output:
            output.write(self.payload)


def test_cupti_monitor_pftrace_export_hides_gzip_suffix(tmp_path):
    path = tmp_path / "trace.pftrace"
    path.write_bytes(b"stale trace")
    profiler = GzipExportProfiler(b"compressed native trace")

    _write_cupti_monitor_trace(profiler, path, trace_format="track_event")

    assert path.exists()
    assert not Path(f"{path}.monitor-export.gz").exists()
    with gzip.open(path, "rb") as trace:
        assert trace.read() == b"compressed native trace"


def test_cupti_monitor_perfetto_trace_uses_native_export_suffix(tmp_path):
    path = tmp_path / "trace.perfetto-trace"
    profiler = GzipExportProfiler(b"native trace")

    _write_cupti_monitor_trace(profiler, path, trace_format="track_event")

    assert profiler.export_paths[0].endswith(".monitor-export.pftrace")
    with gzip.open(path, "rb") as trace:
        assert trace.read() == b"native trace"


def test_cupti_monitor_pftrace_keeps_native_export_with_graph_annotations(monkeypatch, tmp_path):
    graph_id = 2
    monkeypatch.setattr(
        benchmark,
        "_cuda_graph_annotations",
        lambda: {(graph_id << 32): [{"name": "attention"}]},
    )
    path = tmp_path / "trace.pftrace"
    profiler = GzipExportProfiler(b"native gpu_render_stage_event trace")

    _write_cupti_monitor_trace(profiler, path, trace_format="track_event")

    with gzip.open(path, "rb") as trace:
        assert trace.read() == b"native gpu_render_stage_event trace"


def test_cupti_monitor_json_export_hides_gzip_suffix(tmp_path):
    path = tmp_path / "trace.json"
    profiler = GzipExportProfiler(b'{"traceEvents": []}')

    _write_cupti_monitor_trace(profiler, path, trace_format="chrome_json")

    assert path.read_bytes() == b'{"traceEvents": []}'
    assert not Path(f"{path}.monitor-export.gz").exists()


def test_cupti_monitor_preserves_requested_json_gzip(tmp_path):
    path = tmp_path / "trace.json.gz"
    profiler = GzipExportProfiler(b'{"traceEvents": []}')

    _write_cupti_monitor_trace(profiler, path, trace_format="chrome_json")

    with gzip.open(path, "rb") as trace:
        assert json.load(trace) == {"traceEvents": []}


def test_cupti_monitor_json_synthesizes_embedded_graph_annotation_boxes(tmp_path):
    path = tmp_path / "trace.json"
    profiler = GzipExportProfiler(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "kernel",
                        "pid": 0,
                        "tid": 7,
                        "ts": 10,
                        "dur": 3,
                        "args": {
                            "graph id": 2,
                            "graph node id": 0,
                            "correlation": 4,
                            "annotation": '[{"name": "attention"}]',
                        },
                    }
                ]
            }
        ).encode()
    )

    _write_cupti_monitor_trace(profiler, path, trace_format="chrome_json")

    trace = json.loads(path.read_text())
    assert trace["traceEvents"][-1]["cat"] == "gpu_user_annotation"
    assert trace["traceEvents"][-1]["name"] == "attention"


def test_cuda_graph_annotation_snapshot_keeps_prior_graph_instantiations(monkeypatch):
    import torch.cuda.graphs as cuda_graphs

    first_tools_id = (2 << 32) | 1
    second_tools_id = (3 << 32) | 1
    live_annotations = {first_tools_id: [{"name": "first"}]}
    callbacks = []

    class Handle:
        removed = False

        def remove(self):
            self.removed = True

    handle = Handle()
    monkeypatch.setattr(benchmark, "_cuda_graph_annotations", lambda: live_annotations)
    monkeypatch.setattr(
        cuda_graphs,
        "register_graph_instantiate_hook",
        lambda callback: callbacks.append(callback) or handle,
    )

    snapshot = _CudaGraphAnnotationSnapshot()
    live_annotations.clear()
    live_annotations[second_tools_id] = [{"name": "second"}]
    callbacks[0](object())
    live_annotations.clear()
    snapshot.close()

    assert snapshot.annotations == {
        first_tools_id: [{"name": "first"}],
        second_tools_id: [{"name": "second"}],
    }
    assert handle.removed


@pytest.mark.slow
@pytest.mark.skipif(
    not benchmark.torch.cuda.is_available(),
    reason="CUDA Graph annotation lifecycle test requires CUDA",
)
def test_cuda_graph_annotation_snapshot_retains_reinstantiated_graphs():
    from torch.cuda._graph_annotations import clear_kernel_annotations, remove_kernel_annotations
    from torch.cuda.graph_annotations import is_available, mark_kernels
    from torch.cuda.graphs import register_graph_destroy_hook

    if not is_available():
        pytest.skip("CUDA Graph annotation support is unavailable")

    clear_kernel_annotations()
    destroy_hook = register_graph_destroy_hook(remove_kernel_annotations)
    snapshot = _CudaGraphAnnotationSnapshot()
    graph = benchmark.torch.cuda.CUDAGraph(keep_graph=True)
    try:
        x = benchmark.torch.ones(8, device="cuda")
        with benchmark.torch.cuda.graph(graph, enable_annotations=True), mark_kernels("attention"):
            x.add_(1)
        graph.instantiate()
        first_instantiation = dict(snapshot.annotations)

        graph.instantiate()
        retained = dict(snapshot.annotations)
        graph.reset()

        assert first_instantiation
        assert len(retained) > len(first_instantiation)
        assert snapshot.annotations == retained
    finally:
        snapshot.close()
        destroy_hook.remove()
        clear_kernel_annotations()


@pytest.mark.slow
@pytest.mark.skipif(
    not benchmark.torch.cuda.is_available(),
    reason="Native CUPTI annotation test requires CUDA",
)
def test_cupti_monitor_native_trace_includes_annotation_track(tmp_path):
    pytest.importorskip("cupti")
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent
    from torch.cuda.graph_annotations import clear_kernel_annotations, is_available, mark_kernels

    if not is_available():
        pytest.skip("CUDA Graph annotation support is unavailable")

    clear_kernel_annotations()
    path = tmp_path / "annotated.perfetto-trace"
    config = CuptiMonitorConfig(
        environment_counters=False,
        pm_metrics=(),
        graph_dependencies=True,
        event_node_ids=True,
    )
    monitor = benchmark.profiler(
        path,
        backend="cupti_monitor",
        trace_format="track_event",
        cupti_monitor_config=config,
    )
    x = benchmark.torch.ones(32, device="cuda")
    graph = benchmark.torch.cuda.CUDAGraph()
    try:
        with benchmark.torch.cuda.graph(graph, enable_annotations=True), mark_kernels("attention"):
            output = x + 1
        with monitor:
            graph.replay()
            benchmark.torch.cuda.synchronize()

        payload = path.read_bytes()
        if payload.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(payload)
        trace = Trace()
        trace.ParseFromString(payload)

        assert any(packet.HasField("gpu_render_stage_event") for packet in trace.packet)
        annotation_begins = [
            packet.track_event
            for packet in trace.packet
            if packet.HasField("track_event")
            and packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN
            and packet.track_event.name == "attention"
        ]
        assert annotation_begins
        assert output is not None
    finally:
        clear_kernel_annotations()


def test_profiler_closes_annotation_snapshot_when_monitor_config_fails(monkeypatch, tmp_path):
    class Snapshot:
        def __init__(self):
            self.annotations = {}
            self.closed = False

        def close(self):
            self.closed = True

    snapshot = Snapshot()
    monkeypatch.setattr(benchmark, "_CudaGraphAnnotationSnapshot", lambda: snapshot)
    monkeypatch.setattr(
        benchmark,
        "_cupti_monitor_experimental_config",
        lambda config: (_ for _ in ()).throw(RuntimeError("config failed")),
    )

    with pytest.raises(RuntimeError, match="config failed"):
        benchmark.profiler(tmp_path / "trace", backend="cupti_monitor")

    assert snapshot.closed


def test_unentered_profiler_context_releases_annotation_snapshot(monkeypatch, tmp_path):
    class Snapshot:
        def __init__(self):
            self.annotations = {}
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProfiler:
        def start(self):
            pass

        def stop(self):
            pass

    snapshot = Snapshot()
    monkeypatch.setattr(benchmark, "_CudaGraphAnnotationSnapshot", lambda: snapshot)
    monkeypatch.setattr(benchmark.torch.profiler, "profile", lambda **kwargs: FakeProfiler())

    context = benchmark.profiler(tmp_path / "trace", backend="kineto")
    del context
    gc.collect()

    assert snapshot.closed


def test_cupti_monitor_profiler_is_constructed_before_context_entry(monkeypatch, tmp_path):
    experimental_config = object()
    constructed = []

    class FakeProfiler:
        def start(self):
            pass

        def stop(self):
            pass

    def fake_profile(**kwargs):
        constructed.append(kwargs)
        return FakeProfiler()

    monkeypatch.setattr(
        benchmark,
        "_cupti_monitor_experimental_config",
        lambda config: experimental_config,
    )
    monkeypatch.setattr(benchmark.torch.profiler, "profile", fake_profile)

    context = benchmark.profiler(
        tmp_path / "trace",
        backend="cupti_monitor",
    )

    assert constructed[0]["experimental_config"] is experimental_config
    with context:
        pass


def test_cupti_monitor_config_selects_monitor_backend(monkeypatch, tmp_path):
    constructed = []

    class FakeProfiler:
        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(
        benchmark,
        "_cupti_monitor_experimental_config",
        lambda config: constructed.append(config) or object(),
    )
    monkeypatch.setattr(benchmark.torch.profiler, "profile", lambda **kwargs: FakeProfiler())

    benchmark.profiler(
        tmp_path / "trace",
        cupti_monitor_config=CuptiMonitorConfig(environment_counters=False),
    )

    assert not constructed[0].environment_counters
