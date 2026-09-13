import os
from contextlib import contextmanager

import pytest
import torch

from transformer_nuggets.utils.benchmark import (
    CudaBenchmarkStats,
    benchmark_cuda_function_in_microseconds,
    benchmark_cuda_function_stats,
    benchmark_cuda_graph_stats,
)


def test_benchmark_cuda_function_stats_uses_all_samples(monkeypatch):
    class FakeBenchmarker:
        def benchmark_gpu(self, fn, **kwargs):
            fn()
            assert kwargs["benchmark_iters"] == 7
            assert kwargs["memory_warmup_iters"] == 11
            assert kwargs["return_mode"] == "all"
            return [0.010, 0.012, 0.011]

    monkeypatch.setattr("torch._inductor.runtime.benchmarking.benchmarker", FakeBenchmarker())

    called = {"count": 0}

    def fn():
        called["count"] += 1

    stats = benchmark_cuda_function_stats(
        fn,
        NUM_ITERS=7,
        MEMORY_WARMUP_ITERS=11,
        CONFIDENCE=0.90,
        N_RESAMPLES=200,
        SEED=0,
    )

    assert called["count"] == 1
    assert isinstance(stats, CudaBenchmarkStats)
    assert stats.samples_us == pytest.approx((10.0, 12.0, 11.0))
    assert stats.quantiles_us == pytest.approx((10.1, 11.0, 11.9))
    assert stats.p05_us == pytest.approx(10.1)
    assert stats.p50_us == pytest.approx(11.0)
    assert stats.p95_us == pytest.approx(11.9)
    assert stats.median_us == pytest.approx(11.0)
    assert stats.median_ci_us[0] <= stats.median_us <= stats.median_ci_us[1]
    assert stats.confidence == pytest.approx(0.90)


def test_benchmark_cuda_function_stats_singleton(monkeypatch):
    class FakeBenchmarker:
        def benchmark_gpu(self, fn, **kwargs):
            fn()
            return [0.010]

    monkeypatch.setattr("torch._inductor.runtime.benchmarking.benchmarker", FakeBenchmarker())

    stats = benchmark_cuda_function_stats(lambda: None)

    assert stats.samples_us == pytest.approx((10.0,))
    assert stats.quantiles_us == pytest.approx((10.0, 10.0, 10.0))
    assert stats.median_us == pytest.approx(10.0)
    assert stats.median_ci_us == pytest.approx((10.0, 10.0))


def test_benchmark_cuda_function_sets_kineto_log_level_around_profiler_call(monkeypatch):
    monkeypatch.delenv("KINETO_LOG_LEVEL", raising=False)

    def fake_do_bench_using_profiling(fn, *, rep, is_vetted_benchmarking):
        assert rep == 3
        assert is_vetted_benchmarking is False
        assert os.environ["KINETO_LOG_LEVEL"] == "6"
        fn()
        return 0.123

    monkeypatch.setattr(
        "transformer_nuggets.utils.benchmark.do_bench_using_profiling",
        fake_do_bench_using_profiling,
    )

    latency_us = benchmark_cuda_function_in_microseconds(lambda: None, NUM_ITERS=3)

    assert latency_us == pytest.approx(123.0)
    assert "KINETO_LOG_LEVEL" not in os.environ


def test_benchmark_cuda_function_restores_existing_kineto_log_level(monkeypatch):
    monkeypatch.setenv("KINETO_LOG_LEVEL", "2")

    def fake_do_bench_using_profiling(fn, *, rep, is_vetted_benchmarking):
        assert os.environ["KINETO_LOG_LEVEL"] == "6"
        fn()
        return 0.123

    monkeypatch.setattr(
        "transformer_nuggets.utils.benchmark.do_bench_using_profiling",
        fake_do_bench_using_profiling,
    )

    benchmark_cuda_function_in_microseconds(lambda: None)

    assert os.environ["KINETO_LOG_LEVEL"] == "2"


def test_benchmark_cuda_function_stats_sets_kineto_log_level_around_profiler_call(monkeypatch):
    class FakeBenchmarker:
        def benchmark_gpu(self, fn, **kwargs):
            assert os.environ["KINETO_LOG_LEVEL"] == "6"
            fn()
            return [0.010, 0.012, 0.011]

    monkeypatch.delenv("KINETO_LOG_LEVEL", raising=False)
    monkeypatch.setattr("torch._inductor.runtime.benchmarking.benchmarker", FakeBenchmarker())

    stats = benchmark_cuda_function_stats(lambda: None)

    assert stats.samples_us == pytest.approx((10.0, 12.0, 11.0))
    assert "KINETO_LOG_LEVEL" not in os.environ


@pytest.mark.parametrize("warmup_iters", [0, 2])
@pytest.mark.parametrize("lock_clocks", [False, True])
def test_benchmark_cuda_graph_stats_does_not_capture(monkeypatch, warmup_iters, lock_clocks):
    calls = []
    timings_ms = iter([0.010, 0.012, 0.011])

    class FakeGraph:
        def replay(self):
            calls.append("replay")

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            calls.append("record")

        def elapsed_time(self, end):
            assert end is not self
            calls.append("elapsed")
            return next(timings_ms)

    def unexpected_capture(*args, **kwargs):
        pytest.fail("An existing graph must not be captured again")

    @contextmanager
    def fake_locked_clocks():
        calls.append("lock")
        yield
        calls.append("unlock")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", unexpected_capture)
    monkeypatch.setattr(torch.cuda, "graph", unexpected_capture)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr("transformer_nuggets.utils.benchmark.locked_clocks", fake_locked_clocks)

    stats = benchmark_cuda_graph_stats(
        FakeGraph(),
        num_iters=3,
        warmup_iters=warmup_iters,
        confidence=0.90,
        n_resamples=50,
        seed=7,
        lock_clocks=lock_clocks,
    )

    expected_calls = ["replay"] * warmup_iters + ["sync"]
    expected_calls += ["record", "replay", "record", "sync", "elapsed"] * 3
    if lock_clocks:
        expected_calls = ["lock", *expected_calls, "unlock"]
    assert calls == expected_calls
    assert stats == CudaBenchmarkStats.from_samples(
        [10.0, 12.0, 11.0], confidence=0.90, n_resamples=50, seed=7
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_iters": 0}, "num_iters must be positive"),
        ({"num_iters": -1}, "num_iters must be positive"),
        ({"warmup_iters": -1}, "warmup_iters must be non-negative"),
    ],
)
def test_benchmark_cuda_graph_stats_rejects_invalid_counts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        benchmark_cuda_graph_stats(object(), **kwargs)


@pytest.mark.parametrize(
    "benchmark_fn", [benchmark_cuda_function_stats, benchmark_cuda_function_in_microseconds]
)
def test_capture_benchmarks_share_replay_timing(monkeypatch, benchmark_fn):
    graph = object()
    calls = []

    @contextmanager
    def capture(captured_graph):
        assert captured_graph is graph
        calls.append("capture")
        yield

    def sample_replays(captured_graph, *, num_iters, warmup_iters):
        assert captured_graph is graph
        assert (num_iters, warmup_iters) == (3, 2)
        calls.append("time")
        return [10.0, 12.0, 11.0]

    def fn(value, *, scale):
        assert (value, scale) == (3, 4)
        calls.append("body")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        "transformer_nuggets.utils.benchmark._time_cuda_graph_replay_samples_us", sample_replays
    )
    result = benchmark_fn(
        fn, 3, scale=4, USE_CUDA_GRAPHS=True, NUM_ITERS=3, CUDAGRAPH_WARMUP_ITERS=2
    )
    assert calls == ["body", "body", "capture", "body", "time"]
    if isinstance(result, CudaBenchmarkStats):
        assert result.samples_us == (10.0, 12.0, 11.0)
    else:
        assert result == 11.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("side_stream", [False, True])
def test_benchmark_existing_cuda_graph_with_updated_inputs(monkeypatch, side_stream):
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        torch.add(x, 3, out=out)
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        torch.add(x, 3, out=out)

    def unexpected_capture(*args, **kwargs):
        pytest.fail("Timing must reuse the existing graph")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", unexpected_capture)
    monkeypatch.setattr(torch.cuda, "graph", unexpected_capture)
    stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for offset in (0, 5):
            x.add_(offset)
            stats = benchmark_cuda_graph_stats(graph, num_iters=5, warmup_iters=2)
            torch.testing.assert_close(out, x + 3)
            assert len(stats.samples_us) == 5
            assert all(sample > 0 for sample in stats.samples_us)
            assert stats.p05_us <= stats.median_us <= stats.p95_us
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "benchmark_fn", [benchmark_cuda_function_stats, benchmark_cuda_function_in_microseconds]
)
def test_capture_cuda_benchmark(benchmark_fn):
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    result = benchmark_fn(
        torch.add, x, 3, out=out, USE_CUDA_GRAPHS=True, NUM_ITERS=5, CUDAGRAPH_WARMUP_ITERS=2
    )
    torch.testing.assert_close(out, x + 3)
    if isinstance(result, CudaBenchmarkStats):
        assert len(result.samples_us) == 5
        assert all(sample > 0 for sample in result.samples_us)
    else:
        assert result > 0
