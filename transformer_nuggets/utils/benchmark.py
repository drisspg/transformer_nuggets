import logging
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.utils.benchmark as benchmark

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


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "func": func}
    )
    return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


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
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return prof


@contextmanager
def print_max_memory_usage():
    try:
        yield
    finally:
        print(f"Max Cuda Memory Used: {torch.cuda.max_memory_allocated() / (1024**3):.4f} GiB")


@contextmanager
def print_cuda_memory_usage():
    initial_memory = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        memory_usage = torch.cuda.memory_allocated() - initial_memory
        memory_usage_gb = memory_usage / (1024**3)
        print(f"CUDA memory usage: {memory_usage_gb:.2f} GB")


@contextmanager
def save_memory_snapshot(file_path: Path):
    """Save a memory snapshot information to a folder
    Usage:
        with save_memory_snapshot(file_path):
            # code to profile

    Args:
        file_path: The path to the folder to save the snapshot to
                    will create the folder if it doesn't exist
    """
    try:
        import torch.distributed as dist

        dist_avail = True
    except ImportError:
        pass

    if dist_avail and dist.is_initialized():
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
        dist_avail = False
        if dist_avail and dist.is_initialized():
            local_rank = dist.get_rank()
            output_path = file_path / f"_rank_{local_rank}.html"
        else:
            output_path = file_path.with_suffix(".html")
        with open(output_path, "w") as f:
            f.write(torch.cuda._memory_viz.trace_plot(s))
            logger.info(f"💾 Trace file 📄 saved to: {bcolors.OKGREEN}{output_path}{bcolors.ENDC}")
