"""Compile, route-check, validate and time FlexGEMM variants under one measurement contract.

Timing: every variant is captured into a CUDA graph and replayed by
``transformer_nuggets.utils.benchmark.benchmark_cuda_function_in_microseconds``
(``USE_CUDA_GRAPHS=True``): fixed pointers, warm caches, no Python dispatch. All variants
of one shape are timed interleaved: a DVFS settle phase, then ``rounds`` rounds in a
deterministic shuffled order (reversed every other round), each round = one graph capture
plus ``replays`` event-timed replays; the reported value is the median of the round medians.
"""

from __future__ import annotations

import ast
import math
import random
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from epilogues import Variant
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

import torch
from torch._inductor.utils import run_and_get_code


EXTERN_GEMM = re.compile(
    r"extern_kernels\.\w*mm\w*|aten\._grouped_mm\.default\(|aten\._scaled_mm_v2\.default\(|aten\.mm\.default\("
)


@dataclass
class Compiled:
    variant: Variant
    output: Any
    code: str
    compile_s: float


@dataclass
class Route:
    flex_calls: int
    extern: list[str]
    triton_kernels: int
    configs: list[dict[str, Any]]
    fast_math: bool

    @property
    def fused(self) -> bool:
        return self.flex_calls >= 1 and not self.extern

    @property
    def config_label(self) -> str:
        return ";".join(config_label(c) for c in self.configs)


@dataclass
class ErrorStats:
    finite: bool
    shape_ok: bool
    nrmse: list[float]
    max_abs: list[float]


@dataclass
class Timing:
    median_us: float
    spread: float
    samples_us: list[float] = field(default_factory=list)


def sm_clock() -> str:
    """``clocks.sm, power.draw, temperature.gpu`` of the visible device via nvidia-smi."""
    uuid = torch.cuda.get_device_properties(0).uuid
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id=GPU-{uuid}",
            "--query-gpu=clocks.sm,power.draw,temperature.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() or "n/a"


def copy_bandwidth_gbs(numel: int = 1 << 29, samples: int = 20) -> float:
    """Algorithmic (read + write) D2D ``copy_`` bandwidth on a 1 GiB bf16 tensor, median of ``samples``."""
    x = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    for _ in range(5):
        y.copy_(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        y.copy_(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e-3)
    del x, y
    return 2 * numel * 2 / statistics.median(times) / 1e9


def compile_and_run(variant: Variant) -> Compiled:
    compiled = torch.compile(variant.fn, backend="inductor", fullgraph=True, dynamic=False)
    start = time.perf_counter()
    output, codes = run_and_get_code(compiled, *variant.args)
    torch.cuda.synchronize()
    return Compiled(
        Variant(compiled, variant.args),
        output,
        "\n".join(codes),
        time.perf_counter() - start,
    )


def route_info(code: str) -> Route:
    configs = [
        dict(ast.literal_eval(c))
        for c in re.findall(r"config=(\(\(.*?\)\))", code, flags=re.DOTALL)
    ]
    source = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("#"))
    return Route(
        flex_calls=code.count("flex_gemm_epilogue("),
        extern=sorted(set(EXTERN_GEMM.findall(source))),
        triton_kernels=len(re.findall(r"^def triton_[a-zA-Z0-9_]+\(", code, flags=re.MULTILINE)),
        configs=configs,
        fast_math="fastmath=True" in code,
    )


def config_label(cfg: dict[str, Any]) -> str:
    label = f"{cfg['tile_m']}x{cfg['tile_n']}_c{cfg['cluster_m']}x{cfg['cluster_n']}"
    label += "_clc" if cfg.get("is_dynamic_persistent") else "_static"
    if cfg.get("pingpong"):
        label += "_pp"
    if cfg.get("swap_ab"):
        label += "_swap"
    return label


def launch_geometry(cfg: dict[str, Any] | None, m: int, n: int) -> dict[str, Any]:
    """Launched CTAs and waves (vs. the device SM count) for a pinned QuACK config.
    Static-persistent launches are capped at one cluster per SM group; CLC launches one CTA per tile."""
    if cfg is None:
        return {}
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    gm, gn = (n, m) if cfg.get("swap_ab") else (m, n)
    tile_m = int(cfg["tile_m"])
    if int(cfg["cluster_m"]) % 2 == 0 and tile_m in (128, 256):
        tile_m //= 2
    tiles = math.ceil(gm / tile_m) * math.ceil(gn / int(cfg["tile_n"]))
    cluster = int(cfg["cluster_m"]) * int(cfg["cluster_n"])
    tile_ctas = math.ceil(tiles / cluster) * cluster
    ctas = (
        tile_ctas if cfg.get("is_dynamic_persistent") else min(tile_ctas, sms // cluster * cluster)
    )
    return {"ctas": ctas, "waves": round(tile_ctas / sms, 2), "sms": sms}


def compare(actual: list[torch.Tensor], reference: list[torch.Tensor]) -> ErrorStats:
    """fp64-rooted normalized RMS and max-abs error per output."""
    shape_ok = len(actual) == len(reference) and all(
        a.shape == r.shape for a, r in zip(actual, reference)
    )
    if not shape_ok:
        return ErrorStats(False, False, [], [])
    nrmse, max_abs = [], []
    finite = True
    for a, r in zip(actual, reference):
        a64, r64 = a.double(), r.double()
        finite &= bool(torch.isfinite(a64).all())
        err = a64 - r64
        denom = r64.square().mean().sqrt().clamp_min(1e-12)
        nrmse.append((err.square().mean().sqrt() / denom).item())
        max_abs.append(err.abs().max().item())
    return ErrorStats(finite, True, nrmse, max_abs)


def accept(
    flex: ErrorStats,
    stock: ErrorStats | None,
    *,
    floor: float = 1e-2,
    ratio: float = 2.5,
) -> bool:
    """Fused error must be finite and no worse than ``ratio`` x the stock path (or ``floor`` without one)."""
    if not (flex.finite and flex.shape_ok):
        return False
    bounds = (
        [max(ratio * s, floor) for s in stock.nrmse]
        if stock is not None and stock.shape_ok
        else []
    )
    if len(bounds) != len(flex.nrmse):
        bounds = [floor] * len(flex.nrmse)
    return all(e <= b for e, b in zip(flex.nrmse, bounds))


def time_variants(
    variants: dict[str, Variant],
    *,
    rounds: int,
    replays: int,
    settle_s: float,
    seed: int = 0,
) -> tuple[dict[str, Timing], list[str]]:
    """Interleaved CUDA-graph timing of all variants; returns per-variant timing and per-round SM clocks."""
    names = list(variants)
    deadline = time.perf_counter() + settle_s
    while names and time.perf_counter() < deadline:
        for name in names:
            v = variants[name]
            benchmark_cuda_function_in_microseconds(
                v.fn, *v.args, USE_CUDA_GRAPHS=True, NUM_ITERS=max(10, replays // 5)
            )
    samples: dict[str, list[float]] = {name: [] for name in names}
    clocks: list[str] = []
    rng = random.Random(seed)
    for r in range(rounds):
        order = names[:]
        rng.shuffle(order)
        if r % 2:
            order.reverse()
        # Sample the SM clock while the round's replays are in flight, not after the final sync.
        sampler = threading.Thread(target=lambda: (time.sleep(0.05), clocks.append(sm_clock())))
        sampler.start()
        for name in order:
            v = variants[name]
            samples[name].append(
                benchmark_cuda_function_in_microseconds(
                    v.fn, *v.args, USE_CUDA_GRAPHS=True, NUM_ITERS=replays
                )
            )
        sampler.join()
    timings = {}
    for name, values in samples.items():
        med = statistics.median(values)
        timings[name] = Timing(med, (max(values) - min(values)) / med if med else 0.0, values)
    return timings, clocks
