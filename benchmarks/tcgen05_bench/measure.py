"""Shared measurement core for tcgen05 probes.

For one compiled probe, every (grid multiplier, repeat count) pair is captured as a CUDA graph and
replayed in interleaved, rotated rounds. Two slopes are fitted against the repeat count:

* CUDA-event time per repeat, which gives absolute throughput under normal DVFS;
* in-kernel ``clock64`` cycles per repeat (median over timed CTAs), which gives per-clock rates
  and, at a one-CTA-per-SM grid, the effective SM clock ``cycles / time``.

Fitted slopes exclude launch, prologue, and completion overhead. NVML clocks, power, and
temperature are sampled in a background thread and attributed to each variant's window.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial

import torch

MMA_REPEATS = (16, 64, 256, 512)
TMEM_REPEATS = (128, 512, 2048, 4096)
LATENCY_REPEATS = (64, 256, 1024, 2048)
GRID_MULTIPLIERS = (1, 2, 4)
SLEEP_CYCLES = 200_000


@dataclass(frozen=True)
class Fit:
    """Linear fit ``y = intercept + slope * repeats``."""

    slope: float
    intercept: float
    r_squared: float


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> Fit:
    """Return the least-squares line through ``(xs, ys)``."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("linear_fit needs at least two matched points")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys, strict=True))
    total = sum((y - y_mean) ** 2 for y in ys)
    return Fit(slope, intercept, 1.0 if total == 0 else 1.0 - residual / total)


@dataclass
class GridResult:
    """Fits and derived rates for one launch grid."""

    grid_multiplier: int
    clusters: int
    ctas: int
    distinct_sms: int
    time_us: Fit
    cycles: Fit
    median_us: list[float]
    median_cycles: list[float]
    work_per_s: float
    work_per_clk_cta: float
    effective_mhz: float


@dataclass
class NvmlWindow:
    """NVML samples attributed to one measurement window."""

    samples: int = 0
    sm_mhz_mean: float | None = None
    sm_mhz_min: float | None = None
    sm_mhz_max: float | None = None
    power_w_mean: float | None = None
    temp_c_max: float | None = None


class NvmlSampler:
    """Sample SM clock, power, and temperature with an ``nvidia-smi`` child process.

    A separate process keeps NVML queries off the benchmarking process's driver lock and GIL,
    which otherwise inflate CUDA-event intervals with host-side gaps.
    """

    QUERY = "timestamp,clocks.sm,power.draw,temperature.gpu"

    def __init__(self, device: str, period_ms: int = 50) -> None:
        self.samples: list[tuple[float, float, float, float]] = []
        self.command = [
            "nvidia-smi",
            f"--query-gpu={self.QUERY}",
            "--format=csv,noheader,nounits",
            f"-lms={period_ms}",
            f"--id={device}",
        ]
        self._process = None
        self._thread = None

    def start(self) -> None:
        """Start the sampler and a reader thread; silently disabled without nvidia-smi."""
        try:
            self._process = subprocess.Popen(
                self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except OSError:
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _read(self) -> None:
        for line in self._process.stdout:
            fields = [field.strip() for field in line.split(",")]
            try:
                stamp = datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                self.samples.append((stamp, float(fields[1]), float(fields[2]), float(fields[3])))
            except (ValueError, IndexError):
                continue

    def window(self, start: float, end: float) -> NvmlWindow:
        """Summarize samples whose wall-clock timestamp falls in ``[start, end]``."""
        rows = [row for row in list(self.samples) if start <= row[0] <= end]
        if not rows:
            return NvmlWindow()
        clocks = [row[1] for row in rows]
        return NvmlWindow(
            samples=len(rows),
            sm_mhz_mean=statistics.fmean(clocks),
            sm_mhz_min=min(clocks),
            sm_mhz_max=max(clocks),
            power_w_mean=statistics.fmean(row[2] for row in rows),
            temp_c_max=max(row[3] for row in rows),
        )


def capture_graph(function: Callable[[], None], warmup: int) -> torch.cuda.CUDAGraph:
    """Prewarm and capture one fixed-pointer callable."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def timed_cycles(output: torch.Tensor) -> list[int]:
    """Return positive per-CTA cycle samples from one probe output."""
    return [int(value) for value in output.view(-1, 2)[:, 0].tolist() if value > 0]


@dataclass
class Measurement:
    """All grids for one variant plus its NVML window."""

    grids: list[GridResult] = field(default_factory=list)
    nvml: NvmlWindow = field(default_factory=NvmlWindow)

    def best(self) -> GridResult:
        """Return the grid with the highest absolute work rate."""
        return max(self.grids, key=lambda grid: grid.work_per_s)

    def to_json(self) -> dict:
        """Return a JSON-serializable dict."""
        return {"grids": [asdict(grid) for grid in self.grids], "nvml": asdict(self.nvml)}


def measure_probe(
    callable_,
    *,
    cluster_size: int,
    work_per_cta_repeat: float,
    num_sms: int,
    latency: bool,
    repeats: Sequence[int],
    rounds: int,
    warmup: int,
    sampler: NvmlSampler | None = None,
    grid_multipliers: Sequence[int] = GRID_MULTIPLIERS,
) -> Measurement:
    """Capture, replay, and fit one compiled probe ``callable_(output, repeats, clusters)``."""
    repeats_list = tuple(repeats)
    grids = (1,) if latency else tuple(grid_multipliers)
    graphs = {}
    outputs = {}
    for grid_multiplier in grids:
        clusters = 1 if latency else max(1, grid_multiplier * num_sms // cluster_size)
        output = torch.full((2 * clusters * cluster_size,), -1, device="cuda", dtype=torch.int64)
        outputs[grid_multiplier] = (clusters, output)
        for count in repeats_list:
            graphs[(grid_multiplier, count)] = capture_graph(
                partial(callable_, output, count, clusters), warmup
            )
    for _, output in outputs.values():
        sms = output.view(-1, 2)[:, 1]
        if bool((sms < 0).any()):
            raise RuntimeError("a probe CTA did not write its completion sample")

    keys = tuple(graphs)
    times = {key: [] for key in keys}
    cycles = {key: [] for key in keys}
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    window_start = time.time()
    for round_index in range(rounds):
        order = keys if round_index % 2 == 0 else tuple(reversed(keys))
        offset = round_index % len(order)
        for key in order[offset:] + order[:offset]:
            # Keep the stream busy while the host enqueues the events and replay, so the event
            # interval contains no host submission gap.
            torch.cuda._sleep(SLEEP_CYCLES)
            start_event.record()
            graphs[key].replay()
            end_event.record()
            torch.cuda.synchronize()
            times[key].append(start_event.elapsed_time(end_event) * 1e3)
            samples = timed_cycles(outputs[key[0]][1])
            cycles[key].append(statistics.median(samples) if samples else 0.0)
    window_end = time.time()

    measurement = Measurement()
    if sampler is not None:
        measurement.nvml = sampler.window(window_start, window_end)
    for grid_multiplier in grids:
        clusters, output = outputs[grid_multiplier]
        ctas = clusters * cluster_size
        median_us = [statistics.median(times[(grid_multiplier, r)]) for r in repeats_list]
        median_cycles = [statistics.median(cycles[(grid_multiplier, r)]) for r in repeats_list]
        time_fit = linear_fit(repeats_list, median_us)
        cycle_fit = linear_fit(repeats_list, median_cycles)
        timed_ctas = len(timed_cycles(output))
        distinct_sms = len(set(output.view(-1, 2)[:, 1].tolist()))
        work_per_s = _ratio(timed_ctas * work_per_cta_repeat, time_fit.slope * 1e-6)
        measurement.grids.append(
            GridResult(
                grid_multiplier=grid_multiplier,
                clusters=clusters,
                ctas=ctas,
                distinct_sms=distinct_sms,
                time_us=time_fit,
                cycles=cycle_fit,
                median_us=median_us,
                median_cycles=median_cycles,
                work_per_s=work_per_s,
                work_per_clk_cta=_ratio(work_per_cta_repeat, cycle_fit.slope),
                # Only meaningful when every CTA ran concurrently on its own SM.
                effective_mhz=_ratio(cycle_fit.slope, time_fit.slope)
                if distinct_sms == ctas
                else float("nan"),
            )
        )
    return measurement
