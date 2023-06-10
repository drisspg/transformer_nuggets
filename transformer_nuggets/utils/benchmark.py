import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.utils.benchmark as benchmark
from torch.profiler import ProfilerActivity, profile, record_function


@dataclass
class ProfileConfig:
    file_path: Optional[str] = None
    name: Optional[str] = None
    cuda: bool = True
    iters: int = 0
    warmup_iters: int = 0
    extra_kwargs: dict = field(default_factory=dict)


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "func": func}
    )
    return t0.blocked_autorange().mean * 1e6


def profile_function(config: ProfileConfig, func: Callable, *args, **kwargs) -> None:
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
    with profile(activities=activities, record_shapes=False, **config.extra_kwargs) as prof:
        for _ in range(config.iters):
            with name_context:
                func(*args, **kwargs)

    if config.file_path is not None:
        prof.export_chrome_trace(config.file_path)
