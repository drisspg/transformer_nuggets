import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional
from contextlib import contextmanager
from pickle import dump

import torch
import torch.utils.benchmark as benchmark
from torch.profiler import ProfilerActivity, profile, record_function

# Patched version until https://github.com/pytorch/pytorch/pull/103384 lands.
from transformer_nuggets.utils.memory_viz import profile_plot


@dataclass
class ProfileConfig:
    file_path: Optional[str] = None
    name: Optional[str] = None
    cuda: bool = True
    iters: int = 0
    warmup_iters: int = 0
    sync: bool = False
    profile_memory: bool = False
    extra_kwargs: dict = field(default_factory=dict)


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "func": func}
    )
    return t0.blocked_autorange().mean * 1e6


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

    name_context = nullcontext() if config.name is None else record_function(config.name)
    with profile(
        activities=activities,
        profile_memory=config.profile_memory,
        record_shapes=config.profile_memory,
        with_stack=config.profile_memory,
        **config.extra_kwargs,
    ) as prof:
        for _ in range(config.iters):
            with name_context:
                func(*args, **kwargs)
                if config.sync:
                    torch.cuda.synchronize()

    if config.file_path is not None:
        prof.export_chrome_trace(config.file_path)

    if config.profile_memory:
        with open("memory_output.html", "w") as f:
            f.write(profile_plot(prof))

    return prof


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
def save_memory_snapshot(file_path):
    torch.cuda.memory._record_memory_history()
    try:
        yield
    finally:
        snapshot = torch.cuda.memory._snapshot()
        with open(file_path, "wb") as f:
            dump(snapshot, f)
