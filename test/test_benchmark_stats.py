import os

import pytest

from transformer_nuggets.utils.benchmark import (
    CudaBenchmarkStats,
    benchmark_cuda_function_in_microseconds,
    benchmark_cuda_function_stats,
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
