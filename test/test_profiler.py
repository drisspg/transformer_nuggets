import gzip
from pathlib import Path

import pytest

from transformer_nuggets.utils import benchmark
from transformer_nuggets.utils.benchmark import (
    DEFAULT_CUPTI_MONITOR_PM_METRICS,
    CuptiMonitorConfig,
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

    def export_chrome_trace(self, path: str) -> None:
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
        assert trace.read() == b'{"traceEvents": []}'


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
