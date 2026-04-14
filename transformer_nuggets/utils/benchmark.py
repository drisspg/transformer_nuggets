import ctypes
import ctypes.util
import logging
import os
import random
import statistics
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from collections.abc import Callable, Sequence

import torch
import torch.utils.benchmark as benchmark
from torch._inductor.utils import do_bench_using_profiling
import functools

from torch.cuda._memory_viz import profile_plot  # type: ignore
from torch.profiler import profile, ProfilerActivity, record_function, schedule

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


def benchmark_cuda_function_stats(func: Callable, *args, **kwargs) -> CudaBenchmarkStats:
    """Benchmark a CUDA callable and return median-centered summary stats.

    This collects per-iteration timings from Inductor's GPU benchmarker and
    returns the raw samples, the sample median, a bootstrap confidence interval
    for that median, and `(p05, p50, p95)` sample quantiles.

    Args:
        func: Callable to benchmark.
        *args: Positional arguments forwarded to ``func``.
        **kwargs: Benchmark configuration and keyword arguments forwarded to
            ``func``. The following benchmark-control keys are consumed by this
            helper before calling ``func``: ``NUM_ITERS``,
            ``MEMORY_WARMUP_ITERS``, ``CONFIDENCE``, ``N_RESAMPLES``, ``SEED``,
            and ``IS_VETTED_BENCHMARKING``.

    Returns:
        CudaBenchmarkStats with raw samples, the sample median, a bootstrap
        median confidence interval, and `(p05, p50, p95)` sample quantiles.

    Notes:
        The bootstrap interval assumes the collected timings are representative
        samples from a single steady-state benchmark regime. It is most useful
        after warmup, when samples are not dominated by obvious drift or phase
        changes such as autotuning, thermal throttling, or one-time allocator
        effects.

    Examples:
        Basic usage::

            stats = benchmark_cuda_function_stats(lambda: kernel(x, y), NUM_ITERS=200)
            print(stats.median_us)
            print(stats.median_ci_us)
            print(stats.quantiles_us)

        With locked clocks::

            with locked_clocks():
                stats = benchmark_cuda_function_stats(lambda: kernel(x, y), NUM_ITERS=200)
    """
    num_iters = kwargs.pop("NUM_ITERS", 100)
    memory_warmup_iters = kwargs.pop("MEMORY_WARMUP_ITERS", 100)
    confidence = kwargs.pop("CONFIDENCE", 0.95)
    n_resamples = kwargs.pop("N_RESAMPLES", 1000)
    seed = kwargs.pop("SEED", 0)
    is_vetted_benchmarking = kwargs.pop("IS_VETTED_BENCHMARKING", False)
    no_args = lambda: func(*args, **kwargs)
    from torch._inductor.runtime.benchmarking import benchmarker

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
    """Thin wrapper around do_bench_using_profiling.

    Accepts NUM_ITERS, IS_VETTED_BENCHMARKING, and lock_clocks as kwargs but
    removes them before calling func so they never leak into the benchmarked callable.
    """
    num_iters = kwargs.pop("NUM_ITERS", 100)
    is_vetted_benchmarking = kwargs.pop("IS_VETTED_BENCHMARKING", False)
    lock = kwargs.pop("LOCK_CLOCKS", False)
    ctx = locked_clocks() if lock else nullcontext()
    with ctx:
        no_args = lambda: func(*args, **kwargs)
        return (
            do_bench_using_profiling(
                no_args, rep=num_iters, is_vetted_benchmarking=is_vetted_benchmarking
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
        trace_path = Path(config.file_path).with_suffix(".json")
        prof.export_chrome_trace(str(trace_path))
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

            logging.info("Saving allocated state during OOM")
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
            logging.info(f"Wrote memory snapshot to {current_trace_name}")
        except Exception as e:
            logging.error(f"Failed to save memory snapshot: {e}")

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)  # type: ignore
    torch.cuda.memory._record_memory_history(max_entries=max_entries, stacks="all")


def get_process_rank():
    """Get process rank even if distributed is not initialized"""
    import os

    # Check for LOCAL_RANK which torchrun sets
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return None


@contextmanager
def profiler(
    path: Path | str,
    record_shapes: bool = True,
    profile_memory: bool = False,
    with_stack: bool = False,
    warmup: int = 0,
):
    """Thin wrapper around torch.profiler

    Args:
        path: The path to save the trace file to
        record_shapes: Record shapes of tensors
        profile_memory: Profile memory usage
        with_stack: Record stack traces - Blows up memory
        warmup: If greater than 0 then it will warmup record before recording

    Usage:
    ```
        with profiler(Path("trace.json")):
            # code to profile

        # With steps (e.g. in a training loop) this will record 7 iterations
        with profiler(Path("trace.json"), warmup=3) as p:
            for i in range(10):
                # Your code for this step (e.g. forward, backward, optimize)

                # Call step() after each iteration
                p.step()
    ```
    """
    from transformer_nuggets import init_logging

    init_logging()

    if not isinstance(path, Path):
        path = Path(path)

    rank = get_process_rank()

    # Create path with suffix
    path = path.with_suffix(".json")

    # Add rank to filename if distributed
    if rank is not None:
        path = path.parent / f"{path.stem}_rank_{rank}{path.suffix}"

    # make parent dir if it doesn't exist
    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Trace file 📄 saved to: {bcolors.OKGREEN}{path}{bcolors.ENDC}")

    def trace_handler(prof) -> None:
        prof.export_chrome_trace(path.as_posix())

    prof_sched = schedule(wait=0, warmup=warmup, active=int(1_000_000)) if warmup > 0 else None
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=trace_handler,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        schedule=prof_sched,
    )

    try:
        profiler.start()
        yield profiler
    finally:
        profiler.stop()
