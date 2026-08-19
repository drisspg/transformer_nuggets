import ctypes
import ctypes.util
import functools
import gzip
import inspect
import json
import logging
import os
import random
import statistics
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch._inductor.utils import do_bench_using_profiling
from torch.cuda._memory_viz import profile_plot  # type: ignore
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils import benchmark

from transformer_nuggets.utils.perfetto import (
    perfetto_trace_path,
    read_trace,
    write_perfetto_trace,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_KINETO_LOG_LEVEL_ENV = "KINETO_LOG_LEVEL"
_KINETO_SUPPRESS_ALL_LOGS_LEVEL = "6"
_KEEP_KINETO_LOG_LEVEL_ENV = "TRANSFORMER_NUGGETS_KEEP_KINETO_LOG_LEVEL"


@contextmanager
def _suppress_sync_activity_profiler_logs():
    """Suppress noisy Kineto profiler start/stop logs during benchmark timing.

    Recent PyTorch nightlies can print lines like::

        USDT:... SyncActivityProfilerHandler.cpp:52] profiler_start
        USDT:... SyncActivityProfilerHandler.cpp:59] profiler_stop

    These are emitted by Kineto native logging, not Python ``warnings``.
    ``KINETO_LOG_LEVEL=6`` disables these logs and is read dynamically by
    Kineto, so setting it only around the profiler-backed benchmark call avoids
    subprocesses and fd-level stderr redirection.
    """
    if os.environ.get(_KEEP_KINETO_LOG_LEVEL_ENV):
        yield
        return

    previous_level = os.environ.get(_KINETO_LOG_LEVEL_ENV)
    os.environ[_KINETO_LOG_LEVEL_ENV] = _KINETO_SUPPRESS_ALL_LOGS_LEVEL
    try:
        yield
    finally:
        if previous_level is None:
            os.environ.pop(_KINETO_LOG_LEVEL_ENV, None)
        else:
            os.environ[_KINETO_LOG_LEVEL_ENV] = previous_level


def _nvml():
    library_path = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
    try:
        return ctypes.CDLL(library_path)
    except OSError as exc:
        raise RuntimeError(f"Unable to load NVML from {library_path}") from exc


def _check_nvml_status(status: int, operation: str):
    if status != 0:
        raise RuntimeError(f"NVML call failed for {operation} with status {status}")


def _get_nvml_handle(nvml, device: int = 0):
    handle = ctypes.c_void_p()
    get_handle = getattr(nvml, "nvmlDeviceGetHandleByIndex_v2", None)
    if get_handle is None:
        get_handle = nvml.nvmlDeviceGetHandleByIndex
    status = get_handle(ctypes.c_uint(device), ctypes.byref(handle))
    _check_nvml_status(status, f"device handle for device {device}")
    return handle


def _get_max_sm_clock(nvml, handle) -> int:
    nvml_clock_type_sm = ctypes.c_uint(1)
    clock_mhz = ctypes.c_uint()
    status = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml_clock_type_sm, ctypes.byref(clock_mhz))
    _check_nvml_status(status, "max SM clock")
    return int(clock_mhz.value)


@contextmanager
def locked_clocks(device: int = 0, clock_mhz: int | None = None):
    """Lock GPU SM clocks for stable benchmarking.

    Requires root and uses ``nvidia-smi -lgc`` to lock and ``nvidia-smi -rgc``
    to reset.

    Args:
        device: CUDA device index.
        clock_mhz: SM clock frequency in MHz. If None, locks to the GPU's max SM clock.
    """
    import subprocess

    if os.geteuid() != 0:
        raise RuntimeError("Requires root to lock GPU clocks")

    if clock_mhz is None:
        nvml = _nvml()
        status = nvml.nvmlInit()
        _check_nvml_status(status, "nvmlInit")
        try:
            clock_mhz = _get_max_sm_clock(nvml, _get_nvml_handle(nvml, device))
        finally:
            shutdown = nvml.nvmlShutdown()
            _check_nvml_status(shutdown, "nvmlShutdown")

    subprocess.check_call(["nvidia-smi", "-i", str(device), "-lgc", f"{clock_mhz},{clock_mhz}"])
    logger.info(f"Locked GPU {device} SM clocks to {clock_mhz} MHz")
    try:
        yield clock_mhz
    finally:
        subprocess.call(["nvidia-smi", "-i", str(device), "-rgc"])
        logger.info(f"Reset GPU {device} SM clocks")


def lazy_import_error(error_msg: str):
    """Decorator that allows functions with imports to be defined without the dependency"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError:
                raise ImportError(error_msg)

        return wrapper

    return decorator


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class ProfileConfig:
    file_path: str | None = None
    name: str | None = None
    cuda: bool = True
    iters: int = 0
    warmup_iters: int = 0
    sync: bool = False
    extra_kwargs: dict = field(default_factory=dict)
    memory_profile_path: str | None = None
    row_limit: int = 10
    gzip_trace: bool = True
    fix_overlapping_events: bool = True
    overlap_track_pattern: str | None = "stream.*"
    trace_format: Literal["chrome_json", "track_event"] = "track_event"


DEFAULT_CUPTI_MONITOR_PM_METRICS = (
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
)


@dataclass(frozen=True)
class CuptiMonitorConfig:
    """Options for PyTorch's experimental CUPTI monitor profiler backend.

    Construct the profiler before CUDA Graph capture when ``graph_dependencies`` or
    ``event_node_ids`` is enabled so the monitor can observe graph instantiation. To enable
    CUPTI hardware-event sampling, call ``torch.profiler._cupti.monitor.enable_hes_early()``
    before importing transformer-nuggets or creating any CUDA context.
    """

    environment_counters: bool = True
    pm_metrics: tuple[str, ...] = DEFAULT_CUPTI_MONITOR_PM_METRICS
    graph_dependencies: bool = True
    event_node_ids: bool = True
    cuda_sync_events: bool = False
    pftrace_compression_level: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.pftrace_compression_level <= 9:
            raise ValueError("pftrace_compression_level must be between 0 and 9")

    def custom_profiler_config(self) -> dict[str, object]:
        """Build the JSON-compatible PyTorch custom profiler configuration."""
        return {
            "backend": "cupti_monitor",
            "enable_environment_counters": self.environment_counters,
            "enable_pm_sampling": bool(self.pm_metrics),
            "pm_metrics": list(self.pm_metrics),
            "enable_graph_dependencies": self.graph_dependencies,
            "enable_event_node_ids": self.event_node_ids,
            "enable_cuda_sync_events": self.cuda_sync_events,
            "pftrace_compression_level": self.pftrace_compression_level,
        }


def supported_cupti_monitor_metrics(*, with_sub_metrics: bool = False) -> frozenset[str]:
    """Return PM-sampling metrics supported by the current CUDA device."""
    try:
        from torch.profiler._cupti.pm_sampling import supported_metrics
    except ImportError as exc:
        raise ImportError(
            "CUPTI metric discovery requires the transformer-nuggets[cupti-monitor] extra"
        ) from exc
    return supported_metrics(with_sub_metrics=with_sub_metrics)


@dataclass(frozen=True)
class CudaBenchmarkStats:
    samples_us: tuple[float, ...]
    median_us: float
    median_ci_us: tuple[float, float]
    quantiles_us: tuple[float, float, float]
    confidence: float

    @staticmethod
    def quantile(samples: Sequence[float], q: float) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")
        if len(samples) == 0:
            raise ValueError("samples must be non-empty")

        ordered = sorted(float(sample) for sample in samples)
        if len(ordered) == 1:
            return ordered[0]

        position = (len(ordered) - 1) * q
        lower_idx = int(position)
        upper_idx = min(lower_idx + 1, len(ordered) - 1)
        if lower_idx == upper_idx:
            return ordered[lower_idx]

        weight = position - lower_idx
        lower = ordered[lower_idx]
        upper = ordered[upper_idx]
        return lower + (upper - lower) * weight

    @classmethod
    def bootstrap_median_confidence_interval(
        cls,
        samples: Sequence[float],
        confidence: float = 0.95,
        n_resamples: int = 1000,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Estimate a percentile bootstrap confidence interval for the sample median.

        This uses the standard nonparametric bootstrap: resample the observed
        timings with replacement, compute the median of each resample, then take
        lower and upper quantiles of that bootstrap distribution. The intuition
        is that the empirical sample distribution stands in for the unknown
        underlying timing distribution, so repeated draws from the observed
        samples approximate repeated draws from the process that produced them.

        This does not assume a specific parametric input distribution such as a
        Gaussian. It does assume the timings are a reasonable sample from one
        stable benchmark regime and are approximately exchangeable, which in
        practice means: same workload, after warmup, without strong time-order
        effects such as thermal drift, autotuning phase changes, or one-time
        allocator/startup behavior dominating the run.

        Like any bootstrap interval, this can be unstable with very small sample
        counts. There is no universal minimum, but single-digit samples are weak
        and even low tens should be treated cautiously. For benchmark summaries,
        this is most credible once you have enough steady-state samples that the
        median is no longer moving much when a few points are added or removed.
        """
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if len(samples) == 0:
            raise ValueError("samples must be non-empty")
        if len(samples) == 1:
            value = float(samples[0])
            return value, value

        rng = random.Random(seed)
        estimates = [
            statistics.median(rng.choices(samples, k=len(samples)))
            for _ in range(max(1, n_resamples))
        ]
        alpha = 1.0 - confidence
        return cls.quantile(estimates, alpha / 2), cls.quantile(estimates, 1.0 - alpha / 2)

    @classmethod
    def from_samples(
        cls,
        samples_us: Sequence[float],
        confidence: float = 0.95,
        n_resamples: int = 1000,
        seed: int = 0,
    ) -> "CudaBenchmarkStats":
        samples = tuple(float(sample) for sample in samples_us)
        quantiles_us = (
            cls.quantile(samples, 0.05),
            cls.quantile(samples, 0.50),
            cls.quantile(samples, 0.95),
        )
        return cls(
            samples_us=samples,
            median_us=quantiles_us[1],
            median_ci_us=cls.bootstrap_median_confidence_interval(
                samples,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed,
            ),
            quantiles_us=quantiles_us,
            confidence=confidence,
        )

    @property
    def p05_us(self) -> float:
        return self.quantiles_us[0]

    @property
    def p50_us(self) -> float:
        return self.quantiles_us[1]

    @property
    def p95_us(self) -> float:
        return self.quantiles_us[2]


def _call_do_bench_using_profiling(
    fn: Callable[[], object],
    *,
    rep: int,
    is_vetted_benchmarking: bool,
) -> float:
    """Call Inductor's profiler benchmark across torch versions.

    Torch versions differ in whether ``do_bench_using_profiling`` accepts the
    ``is_vetted_benchmarking`` kwarg. Detect that at runtime so the benchmark
    helper works on both older and newer torch builds.
    """
    params = inspect.signature(do_bench_using_profiling).parameters
    call_kwargs = {"rep": rep}
    if "is_vetted_benchmarking" in params:
        call_kwargs["is_vetted_benchmarking"] = is_vetted_benchmarking
    with _suppress_sync_activity_profiler_logs():
        return do_bench_using_profiling(fn, **call_kwargs)


def _benchmark_cuda_graph_replay_samples_us(
    func: Callable,
    *args,
    **kwargs,
) -> list[float]:
    """Capture one CUDA graph and return per-replay CUDA-event timings in us.

    This measures steady-state replay latency. It intentionally excludes host
    launch gaps that dominate tiny eager kernels. Any GPU-side work inside the
    captured callable is included. Host-to-device copy-in from CPU tensors is
    not represented unless the callable stages data into static GPU buffers as
    part of the captured region.
    """
    num_iters = kwargs.pop("NUM_ITERS", 100)
    warmup_iters = kwargs.pop("CUDAGRAPH_WARMUP_ITERS", max(10, min(25, num_iters)))
    lock = kwargs.pop("LOCK_CLOCKS", False)
    ctx = locked_clocks() if lock else nullcontext()
    with ctx:
        no_args = lambda: func(*args, **kwargs)
        for _ in range(warmup_iters):
            no_args()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            no_args()
        torch.cuda.synchronize()

        for _ in range(warmup_iters):
            graph.replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples_us = []
        for _ in range(num_iters):
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples_us.append(start.elapsed_time(end) * 1e3)
        return samples_us


def benchmark_cuda_function_stats(func: Callable, *args, **kwargs) -> CudaBenchmarkStats:
    """Benchmark a CUDA callable and return median-centered summary stats.

    By default this collects per-iteration timings from Inductor's GPU
    benchmarker. With ``USE_CUDA_GRAPHS=True`` it instead captures one static
    CUDA graph and returns per-replay timings, which is often closer to NCU for
    tiny static kernels.

    Args:
        func: Callable to benchmark.
        *args: Positional arguments forwarded to ``func``.
        **kwargs: Benchmark configuration and keyword arguments forwarded to
            ``func``. The following benchmark-control keys are consumed by this
            helper before calling ``func``: ``NUM_ITERS``,
            ``MEMORY_WARMUP_ITERS``, ``CONFIDENCE``, ``N_RESAMPLES``, ``SEED``,
            ``IS_VETTED_BENCHMARKING``, ``LOCK_CLOCKS``, ``USE_CUDA_GRAPHS``,
            and ``CUDAGRAPH_WARMUP_ITERS``.

    Returns:
        CudaBenchmarkStats with raw samples, the sample median, a bootstrap
        median confidence interval, and `(p05, p50, p95)` sample quantiles.

    Notes:
        The bootstrap interval assumes the collected timings are representative
        samples from a single steady-state benchmark regime. It is most useful
        after warmup, when samples are not dominated by obvious drift or phase
        changes such as autotuning, thermal throttling, or one-time allocator
        effects.
    """
    num_iters = kwargs.pop("NUM_ITERS", 100)
    memory_warmup_iters = kwargs.pop("MEMORY_WARMUP_ITERS", 100)
    confidence = kwargs.pop("CONFIDENCE", 0.95)
    n_resamples = kwargs.pop("N_RESAMPLES", 1000)
    seed = kwargs.pop("SEED", 0)
    is_vetted_benchmarking = kwargs.pop("IS_VETTED_BENCHMARKING", False)
    use_cuda_graphs = kwargs.pop("USE_CUDA_GRAPHS", False)

    if use_cuda_graphs:
        samples_us = _benchmark_cuda_graph_replay_samples_us(
            func,
            *args,
            NUM_ITERS=num_iters,
            CUDAGRAPH_WARMUP_ITERS=kwargs.pop("CUDAGRAPH_WARMUP_ITERS", memory_warmup_iters),
            LOCK_CLOCKS=kwargs.pop("LOCK_CLOCKS", False),
            **kwargs,
        )
        return CudaBenchmarkStats.from_samples(
            samples_us,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        )

    lock = kwargs.pop("LOCK_CLOCKS", False)
    ctx = locked_clocks() if lock else nullcontext()
    with ctx:
        no_args = lambda: func(*args, **kwargs)
        from torch._inductor.runtime.benchmarking import benchmarker

        with _suppress_sync_activity_profiler_logs():
            samples_ms = benchmarker.benchmark_gpu(
                no_args,
                benchmark_iters=num_iters,
                memory_warmup_iters=memory_warmup_iters,
                return_mode="all",
                is_vetted_benchmarking=is_vetted_benchmarking,
            )
    return CudaBenchmarkStats.from_samples(
        (float(sample) * 1e3 for sample in samples_ms),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    lock = kwargs.pop("LOCK_CLOCKS", False)
    ctx = locked_clocks() if lock else nullcontext()
    with ctx:
        for _ in range(5):
            func(*args, **kwargs)
        t0 = benchmark.Timer(
            stmt="func(*args, **kwargs)",
            globals={"args": args, "kwargs": kwargs, "func": func},
        )
        return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Benchmark a CUDA callable and return median latency in microseconds.

    By default this uses Inductor's profiler-based benchmark helper. With
    ``USE_CUDA_GRAPHS=True`` it instead captures one static CUDA graph and times
    replay latency with CUDA events.

    Consumed benchmark kwargs:
      - ``NUM_ITERS``
      - ``IS_VETTED_BENCHMARKING``
      - ``LOCK_CLOCKS``
      - ``USE_CUDA_GRAPHS``
      - ``CUDAGRAPH_WARMUP_ITERS``
    """
    num_iters = kwargs.pop("NUM_ITERS", 100)
    is_vetted_benchmarking = kwargs.pop("IS_VETTED_BENCHMARKING", False)
    use_cuda_graphs = kwargs.pop("USE_CUDA_GRAPHS", False)

    if use_cuda_graphs:
        samples_us = _benchmark_cuda_graph_replay_samples_us(
            func,
            *args,
            NUM_ITERS=num_iters,
            LOCK_CLOCKS=kwargs.pop("LOCK_CLOCKS", False),
            CUDAGRAPH_WARMUP_ITERS=kwargs.pop(
                "CUDAGRAPH_WARMUP_ITERS", max(10, min(25, num_iters))
            ),
            **kwargs,
        )
        return statistics.median(samples_us)

    lock = kwargs.pop("LOCK_CLOCKS", False)
    ctx = locked_clocks() if lock else nullcontext()
    with ctx:
        no_args = lambda: func(*args, **kwargs)
        return (
            _call_do_bench_using_profiling(
                no_args,
                rep=num_iters,
                is_vetted_benchmarking=is_vetted_benchmarking,
            )
            * 1e3
        )


@lazy_import_error("This function requires Triton. Please install it with: pip install triton")
def benchmark_cuda_function_in_microseconds_triton(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench"""
    from triton.testing import do_bench

    lock = kwargs.pop("LOCK_CLOCKS", False)
    ctx = locked_clocks() if lock else nullcontext()
    with ctx:
        no_args = lambda: func(*args, **kwargs)
        return do_bench(no_args) * 1e3


def _write_profiler_trace(
    prof: torch.profiler.profile,
    trace_path: Path,
    *,
    trace_format: Literal["chrome_json", "track_event"],
    split_overlaps: bool,
    track_pattern: str | None,
    gzip_trace: bool = False,
) -> None:
    """Export a torch profiler trace through the canonical Perfetto writer."""
    can_export_direct_json = (
        trace_format == "chrome_json" and not split_overlaps and trace_path.suffix != ".gz"
    )
    if can_export_direct_json:
        prof.export_chrome_trace(str(trace_path))
        return

    export_path = trace_path.with_name(f"{trace_path.name}.tmp.json")
    try:
        prof.export_chrome_trace(str(export_path))
        write_perfetto_trace(
            trace_path,
            read_trace(export_path),
            trace_format=trace_format,
            split_overlaps=split_overlaps,
            track_pattern=track_pattern,
            gzip_trace=gzip_trace,
        )
    finally:
        export_path.unlink(missing_ok=True)


def _cupti_monitor_experimental_config(
    config: CuptiMonitorConfig,
) -> torch._C._profiler._ExperimentalConfig:
    """Prepare PyTorch's experimental CUPTI monitor before profiler construction."""
    try:
        import cupti  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The CUPTI monitor backend requires cupti-python. Install the "
            "transformer-nuggets[cupti-monitor] extra or cupti-python directly."
        ) from exc

    return torch._C._profiler._ExperimentalConfig(
        custom_profiler_config=json.dumps(config.custom_profiler_config())
    )


def _write_cupti_monitor_trace(
    prof: torch.profiler.profile,
    trace_path: Path,
    *,
    trace_format: Literal["chrome_json", "track_event"],
) -> None:
    """Export a monitor trace while hiding its unconditional ``.gz`` suffix."""
    export_path = trace_path.with_name(f"{trace_path.name}.monitor-export")
    monitor_path = Path(f"{export_path}.gz")
    export_path.unlink(missing_ok=True)
    monitor_path.unlink(missing_ok=True)
    prof.export_chrome_trace(str(export_path))

    generated_path = export_path if export_path.exists() else monitor_path
    if not generated_path.exists():
        raise FileNotFoundError(
            f"CUPTI monitor export produced neither {export_path} nor {monitor_path}"
        )

    trace_path.unlink(missing_ok=True)
    if trace_format == "track_event" or trace_path.suffix == ".gz":
        generated_path.replace(trace_path)
        return

    try:
        with gzip.open(generated_path, "rb") as source, trace_path.open("wb") as destination:
            destination.write(source.read())
    finally:
        generated_path.unlink(missing_ok=True)


def profile_function(
    config: ProfileConfig, func: Callable, *args, **kwargs
) -> torch.profiler.profile:
    """Profile a torch function and save the result to a file"""
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    activities = [ProfilerActivity.CPU]
    if config.cuda:
        activities.append(ProfilerActivity.CUDA)

    if config.warmup_iters >= 0:
        for _ in range(config.warmup_iters):
            func(*args, **kwargs)
    if config.sync:
        torch.cuda.synchronize()
    name_context = nullcontext() if config.name is None else record_function(config.name)
    profile_memory = config.memory_profile_path is not None
    with profile(
        activities=activities,
        profile_memory=profile_memory,
        record_shapes=profile_memory,
        with_stack=profile_memory,
        **config.extra_kwargs,
    ) as prof:
        for _ in range(config.iters):
            with name_context:
                func(*args, **kwargs)
                if config.sync:
                    torch.cuda.synchronize()

    if config.file_path is not None:
        trace_path = perfetto_trace_path(
            config.file_path,
            trace_format=config.trace_format,
            gzip_trace=config.gzip_trace,
        )
        _write_profiler_trace(
            prof,
            trace_path,
            trace_format=config.trace_format,
            split_overlaps=config.fix_overlapping_events,
            track_pattern=config.overlap_track_pattern,
            gzip_trace=config.gzip_trace,
        )
        logger.info(f"💾 Trace file 📄 saved to: {bcolors.OKGREEN}{trace_path}{bcolors.ENDC}")

    if profile_memory and config.memory_profile_path is not None:
        with open(config.memory_profile_path, "w") as f:
            f.write(profile_plot(prof))

    if config.file_path is None:
        sort_by = "cpu_time_total" if not config.cuda else "cuda_time_total"
        print(prof.key_averages().table(sort_by=sort_by, row_limit=config.row_limit))

    return prof


class max_memory_usage:
    """Tracks maximum CUDA memory usage within a context manager region

    Args:
        log (bool): Whether to print the memory usage to the console
        precision (int): The number of decimal places to print

    Usage:
    ```
        with max_memory_usage() as mem:
            # code to profile
        print(mem.max_memory)
    ```
    """

    def __init__(self, log=False, precision=2):
        self.log = log
        self.precision = precision
        self.max_memory = 0

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.max_memory = torch.cuda.max_memory_allocated()
        if self.log:
            max_memory_gib = self.max_memory / (1024**3)
            print(f"Max CUDA Memory Allocated: {max_memory_gib:.{self.precision}f} GiB")


class cuda_memory_usage:
    """Prints the difference CUDA memory usage at the end of a context manager

    Args:
        log (bool): Whether to print the memory usage to the console
        precision (int): The number of decimal places to print

    Usage:
    ```
        with cuda_memory_usage() as mem:
            # code to profile
        print(mem.memory_usage)
    ```

    """

    def __init__(self, log=False, precision=2):
        self.log = log
        self.precision = precision
        self.memory_usage = 0

    def __enter__(self):
        self.initial_memory = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.memory_usage = torch.cuda.memory_allocated() - self.initial_memory
        if self.log:
            memory_usage_gib = self.memory_usage / (1024**3)
            print(f"CUDA memory usage: {memory_usage_gib:.{self.precision}f} GiB")


@contextmanager
def save_memory_snapshot(file_path: Path | str, viz: Literal["torch", "d3", "pickle"] = "torch"):
    """Save a memory snapshot information to a folder

    Args:
        file_path: The path to the folder to save the snapshot to
                    will create the folder if it doesn't exist
        viz: Visualization backend - "torch" for PyTorch's built-in viz,
             "d3" for custom D3.js interactive viz

    Usage:
    ```
        with save_memory_snapshot(file_path):
            # code to profile

        with save_memory_snapshot(file_path, viz="d3"):
            # code to profile with custom D3 visualization
    ```
    """
    from transformer_nuggets import init_logging

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    init_logging()
    dist_avail = False
    try:
        import torch.distributed as dist

        dist_avail = True
    except ImportError:
        pass

    dist_avail = dist_avail and dist.is_initialized()
    if dist_avail:
        if not file_path.is_dir():
            raise ValueError(
                f"{file_path} is not a directory, but is required for distributed profiling"
            )
    else:
        if file_path.is_dir():
            raise ValueError(f"{file_path} is a directory")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(stacks="all")
    try:
        yield
    finally:
        s = torch.cuda.memory._snapshot()
        snapshot_device = torch.cuda.current_device()
        if viz == "pickle":
            import pickle

            suffix = ".pickle"
        else:
            suffix = ".html"

        if dist_avail:
            local_rank = dist.get_rank()
            output_path = file_path / f"_rank_{local_rank}{suffix}"
        else:
            output_path = file_path.with_suffix(suffix)

        match viz:
            case "pickle":
                import pickle

                with open(output_path, "wb") as fb:
                    pickle.dump(s, fb)
            case "torch":
                html = torch.cuda._memory_viz.trace_plot(s)  # type: ignore
                with open(output_path, "w") as f:
                    f.write(html)
            case "d3":
                from transformer_nuggets.utils.memory_viz import generate_memory_html

                html = generate_memory_html(s, device=snapshot_device, title=file_path.stem)
                with open(output_path, "w") as f:
                    f.write(html)
            case _:
                raise ValueError(
                    f"Unknown viz backend: {viz!r}, expected 'torch', 'd3', or 'pickle'"
                )

        logger.info(f"💾 Trace file 📄 saved to: {bcolors.OKGREEN}{output_path}{bcolors.ENDC}")


def _is_distributed():
    try:
        import torch.distributed as dist

        return dist.is_initialized()
    except ImportError:
        pass
    return False


def attach_oom_observer(
    save_path: Path | None = None,
    max_entries: int = 1000000,
    viz: Literal["torch", "d3"] = "torch",
):
    """
    Attach an out-of-memory (OOM) observer to the CUDA device.
    The observer will save a memory snapshot when an OOM error occurs.

    Args:
        save_path (Path): Directory where memory snapshots will be saved.
                         The cwd will be used.
        max_entries (int): Maximum number of memory history entries to record.
                           Default is 1000000.
        viz: Visualization backend - "torch" or "d3"

    Usage:
    ```
        attach_oom_observer(Path("memory_snapshots"))
        # All cuda cuda events from this point to OOM program termination will be recorded and saved
        <Code that OOMS>
    ```
    """
    import torch.cuda.memory

    if save_path is None:
        save_path = Path.cwd() / "memory_snapshots"
    trace_dir = save_path
    trace_dir.mkdir(parents=True, exist_ok=True)
    assert trace_dir.is_dir(), "save_path must be a directory."

    def oom_observer(device, alloc, device_alloc, device_free):
        try:
            rank = "0"
            if _is_distributed():
                import torch.distributed as dist

                rank = dist.get_rank()

            curr_trace_name = f"memory_snapshots_rank_{rank}_snapshot.html"
            current_trace_name = trace_dir / Path(curr_trace_name)

            logger.info("Saving allocated state during OOM")
            snapshot = torch.cuda.memory._snapshot()

            match viz:
                case "torch":
                    html = torch.cuda._memory_viz.trace_plot(snapshot)  # type: ignore
                case "d3":
                    from transformer_nuggets.utils.memory_viz import generate_memory_html

                    html = generate_memory_html(snapshot, device=device, title=f"OOM rank {rank}")
                case _:
                    html = torch.cuda._memory_viz.trace_plot(snapshot)  # type: ignore

            with open(current_trace_name, "w") as f:
                f.write(html)
            logger.info("Wrote memory snapshot to %s", current_trace_name)
        except Exception:
            logger.exception("Failed to save memory snapshot")

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)  # type: ignore
    torch.cuda.memory._record_memory_history(max_entries=max_entries, stacks="all")


def get_process_rank():
    """Get process rank even if distributed is not initialized"""
    import os

    # Check for LOCAL_RANK which torchrun sets
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return None


def _ranked_trace_path(path: Path, rank: int) -> Path:
    """Insert a distributed rank before a trace format suffix."""
    if path.suffixes[-2:] == [".json", ".gz"]:
        stem = path.name[: -len(".json.gz")]
        return path.with_name(f"{stem}_rank_{rank}.json.gz")
    return path.with_name(f"{path.stem}_rank_{rank}{path.suffix}")


def profiler(
    path: Path | str,
    record_shapes: bool = True,
    profile_memory: bool = False,
    with_stack: bool = False,
    warmup: int = 0,
    fix_overlapping_events: bool = True,
    overlap_track_pattern: str | None = "stream.*",
    gzip_trace: bool = False,
    trace_format: Literal["chrome_json", "track_event"] = "track_event",
    backend: Literal["kineto", "cupti_monitor"] = "kineto",
    cupti_monitor_config: CuptiMonitorConfig | None = None,
):
    """Create a torch profiler context and save its trace when the context exits.

    Args:
        path: The path to save the trace file to.
        record_shapes: Record shapes of tensors.
        profile_memory: Profile memory usage.
        with_stack: Record stack traces. This can substantially increase memory use.
        warmup: Number of scheduled warmup steps before recording.
        fix_overlapping_events: Postprocess Kineto duration slices into sibling lanes so
            Perfetto does not hide overlaps. The CUPTI monitor's native exporter owns its
            lane layout and does not use this postprocessor.
        overlap_track_pattern: Regex for Kineto tracks to postprocess. Defaults to CUDA
            stream tracks; pass ``None`` to process every track.
        gzip_trace: Write ``.json.gz`` instead of ``.json`` for Chrome JSON traces.
        trace_format: ``"track_event"`` writes native Perfetto ``.pftrace`` output;
            ``"chrome_json"`` writes Chrome JSON/JSON.GZ output.
        backend: Profiler backend. ``"kineto"`` preserves the standard PyTorch profiler;
            ``"cupti_monitor"`` enables counters, CUDA Graph dependencies, kernel
            annotations, and native Perfetto export.
        cupti_monitor_config: Advanced CUPTI monitor overrides. Passing this also selects
            the monitor backend. Construct the profiler before CUDA Graph capture when
            recording graph dependencies, then enter it around replay.

    Usage:
    ```
        with profiler(Path("trace.pftrace")):
            run_workload()

        with profiler(Path("monitor_trace.pftrace"), backend="cupti_monitor"):
            run_workload()

        monitor = profiler(Path("graph_trace.pftrace"), backend="cupti_monitor")
        graph = capture_graph()
        with monitor:
            graph.replay()
    ```
    """
    from transformer_nuggets import init_logging

    init_logging()
    if backend not in ("kineto", "cupti_monitor"):
        raise ValueError(f"Unsupported profiler backend: {backend!r}")
    if cupti_monitor_config is not None:
        backend = "cupti_monitor"
    monitor_config = CuptiMonitorConfig() if cupti_monitor_config is None else cupti_monitor_config
    use_cupti_monitor = backend == "cupti_monitor"
    if use_cupti_monitor and warmup:
        raise ValueError("warmup schedules are not supported by the CUPTI monitor backend")

    path = Path(path)
    path = perfetto_trace_path(path, trace_format=trace_format, gzip_trace=gzip_trace)
    rank = get_process_rank()
    if rank is not None:
        path = _ranked_trace_path(path, rank)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Trace file 📄 saved to: {bcolors.OKGREEN}{path}{bcolors.ENDC}")

    def trace_handler(prof) -> None:
        if use_cupti_monitor:
            _write_cupti_monitor_trace(prof, path, trace_format=trace_format)
            return
        _write_profiler_trace(
            prof,
            path,
            trace_format=trace_format,
            split_overlaps=fix_overlapping_events,
            track_pattern=overlap_track_pattern,
            gzip_trace=gzip_trace,
        )

    prof_sched = schedule(wait=0, warmup=warmup, active=1_000_000) if warmup > 0 else None
    experimental_config = (
        _cupti_monitor_experimental_config(monitor_config) if use_cupti_monitor else None
    )
    torch_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=trace_handler,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        schedule=prof_sched,
        experimental_config=experimental_config,
    )

    @contextmanager
    def profile_context():
        try:
            torch_profiler.start()
            yield torch_profiler
        finally:
            torch_profiler.stop()

    return profile_context()
