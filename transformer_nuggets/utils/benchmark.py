import logging
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.utils.benchmark as benchmark
from torch._inductor.utils import do_bench_using_profiling

from torch.cuda._memory_viz import profile_plot
from torch.profiler import profile, ProfilerActivity, record_function

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    file_path: Optional[str] = None
    name: Optional[str] = None
    cuda: bool = True
    iters: int = 0
    warmup_iters: int = 0
    sync: bool = False
    extra_kwargs: dict = field(default_factory=dict)
    memory_profile_path: Optional[str] = None
    row_limit: int = 10


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


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

    if profile_memory:
        with open(config.memory_profile_path, "w") as f:
            f.write(profile_plot(prof))

    if config.file_path is None:
        sort_by = "cpu_time_total" if not config.cuda else "cuda_time_total"
        print(prof.key_averages().table(sort_by=sort_by, row_limit=config.row_limit))

    return prof


@contextmanager
def print_max_memory_usage(precision: int = 2):
    """Prints the maximum CUDA memory usage at the end of a context manager

    Args:
        precision (int): The number of decimal places to print

    Usage:
    ```
        with print_max_memory_usage():
            # code to profile
    ```
    """
    try:
        yield
    finally:
        print(f"Max Cuda Memory Used: {torch.cuda.max_memory_allocated() / (1024**3):.4f} GiB")


@contextmanager
def print_cuda_memory_usage(precision: int = 2):
    """Prints the difference CUDA memory usage at the end of a context manager

    Args:
        precision (int): The number of decimal places to print

    Usage:
    ```
        with print_cuda_memory_usage():
            # code to profile
    ```

    """
    initial_memory = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        memory_usage = torch.cuda.memory_allocated() - initial_memory
        memory_usage_gb = memory_usage / (1024**3)
        print(f"CUDA memory usage: {memory_usage_gb:.{precision}f} GB")


@contextmanager
def save_memory_snapshot(file_path: Path):
    """Save a memory snapshot information to a folder

    Args:
        file_path: The path to the folder to save the snapshot to
                    will create the folder if it doesn't exist

    Usage:
    ```
        with save_memory_snapshot(file_path):
            # code to profile
    ```
    """
    from transformer_nuggets import init_logging

    init_logging()
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

    # make parent dir
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history()
    try:
        yield
    finally:
        s = torch.cuda.memory._snapshot()
        if dist_avail:
            local_rank = dist.get_rank()
            output_path = file_path / f"_rank_{local_rank}.html"
        else:
            output_path = file_path.with_suffix(".html")
        with open(output_path, "w") as f:
            f.write(torch.cuda._memory_viz.trace_plot(s))
            logger.info(f"💾 Trace file 📄 saved to: {bcolors.OKGREEN}{output_path}{bcolors.ENDC}")


def _is_distributed():
    try:
        import torch.distributed as dist

        return dist.is_initialized()
    except ImportError:
        pass
    return False


def attach_oom_observer(save_path: Optional[Path] = None, max_entries: int = 1000000):
    """
    Attach an out-of-memory (OOM) observer to the CUDA device.
    The observer will save a memory snapshot when an OOM error occurs.

    Args:
        save_path (Path): Directory where memory snapshots will be saved.
                         The cwd will be used.
        max_entries (int): Maximum number of memory history entries to record.
                           Default is 1000000.

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
            with open(current_trace_name, "w") as f:
                f.write(torch.cuda._memory_viz.trace_plot(snapshot))
            logging.info(f"Wrote memory snapshot to {current_trace_name}")
        except Exception as e:
            logging.error(f"Failed to save memory snapshot: {e}")

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)
    torch.cuda.memory._record_memory_history(max_entries=max_entries)


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
):
    """Thin wrapper around torch.profiler

    Args:
        path: The path to save the trace file to
        record_shapes: Record shapes of tensors
        profile_memory: Profile memory usage
        with_stack: Record stack traces - Blows up memory

    Usage:
    ```
        with profiler(Path("trace.json")):
            # code to profile
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

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=trace_handler,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )

    try:
        profiler.start()
        yield profiler
    finally:
        profiler.stop()
