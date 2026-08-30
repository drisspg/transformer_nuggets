"""Measure SM100 CTA-group-1 TCGEN instruction-size throughput.

The repeated region contains only dense BF16 SS ``tcgen05.mma`` operations into TMEM. Each shape
covers the same physical envelope, so fitted time-per-repeat compares instruction granularity rather
than useful-work count. The script runs two sweeps:

* 128x128x128 using M64/M128 by N16/32/64/128;
* 128x256x128 comparing two M128xN128 panels with the legal M128xN256 maximum.

Each CTA zero-fills its operands once before the repeated region. Absolute TFLOP/s is therefore an
optimistic low-switching roofline under the GPU's normal DVFS policy; relative shape ratios are the
primary result. Generated SASS should order the repeated UTCHMMA body before UTCBAR and the
completion wait before the final sentinel store.

Run on SM100 with CuTeDSL 4.7 or newer:

    python benchmarks/tcgen05_throughput.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
import typer
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Float32, Int32, Int64

from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.utils.benchmark import CudaBenchmarkStats

CTA_THREADS = 128
MMA_WARP = 0
COMPLETE_WARP = 1
K = 128
TMEM_COLUMNS = 256
REPEATS = (16, 64, 256, 512)
GRID_MULTIPLIERS = (1, 2, 4)
DEFAULT_ROUNDS = 31
DEFAULT_WARMUP = 5
DEFAULT_OUTPUT = Path("agent_space/tcgen05_throughput.json")


@dataclass(frozen=True)
class Shape:
    """A physical TCGEN tile and the fixed-work envelope used to measure it."""

    m: int
    n: int
    envelope_m: int
    envelope_n: int

    @property
    def name(self) -> str:
        """Return the compact physical-shape label."""
        return f"m{self.m}n{self.n}k{K}"

    @property
    def panels(self) -> int:
        """Return source-level GEMM panels per envelope."""
        return (self.envelope_m // self.m) * (self.envelope_n // self.n)

    @property
    def flops(self) -> int:
        """Return physical multiply-add FLOPs per CTA and repeat."""
        return 2 * self.envelope_m * self.envelope_n * K


@dataclass(frozen=True)
class Sweep:
    """A matched fixed-work collection of physical instruction shapes."""

    name: str
    shapes: tuple[Shape, ...]

    @property
    def envelope(self) -> tuple[int, int, int]:
        """Return the common physical work envelope."""
        shape = self.shapes[0]
        return shape.envelope_m, shape.envelope_n, K


SWEEPS = (
    Sweep(
        "cta1_shape_sweep",
        tuple(Shape(m, n, 128, 128) for m in (64, 128) for n in (16, 32, 64, 128)),
    ),
    Sweep(
        "cta1_max_n",
        (Shape(128, 128, 128, 256), Shape(128, 256, 128, 256)),
    ),
)


class TcgenThroughputProbe:
    """Issue repeated SS TCGEN work and store one completion sentinel per CTA."""

    def __init__(self, shape: Shape) -> None:
        self.shape = shape
        self.tile = (shape.m, shape.n, K)
        self.io_type = cutlass.BFloat16
        self.acc_type = cutlass.Float32

    def get_name(self) -> str:
        """Return a stable artifact and kernel prefix."""
        return (
            f"tcgen_throughput_{self.shape.name}_env{self.shape.envelope_m}x"
            f"{self.shape.envelope_n}_tm{TMEM_COLUMNS}"
        )

    @cute.jit
    def __call__(self, output: cute.Tensor, repeats: Int32, blocks: Int32, stream):
        mma = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_type,
            tcgen05.CtaGroup.ONE,
            self.tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tile, self.io_type, 1)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tile, self.io_type, 1)
        accumulator_layout = mma.make_fragment_C(mma.partition_shape_C(self.tile[:2])).layout
        self.k_blocks = cute.size(a_layout.outer, mode=[2])

        @cute.struct
        class Operands:
            a: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(a_layout)],
                1024,
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(b_layout)],
                1024,
            ]

        @cute.struct
        class SharedStorage:
            completion_barrier: cute.struct.MemRange[Int64, 2]
            tmem_holder: Int32
            operands: Operands

        self.shared_type = SharedStorage
        self.operand_bytes = Operands.__sizeof__()
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            mma,
            a_layout,
            b_layout,
            accumulator_layout,
            output,
            repeats,
        ).launch(
            grid=(blocks, 1, 1),
            block=(CTA_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def run_mma(
        self,
        mma,
        a_fragment,
        b_fragment,
        accumulator_layout,
        tmem_pointer,
        repeats,
        producer,
    ):
        """Issue every panel in the fixed envelope and commit one completion transaction."""
        phase = producer.acquire_and_advance()
        for repeat in cutlass.range(repeats, unroll=1):
            for m_panel in cutlass.range_constexpr(self.shape.envelope_m // self.shape.m):
                for n_panel in cutlass.range_constexpr(self.shape.envelope_n // self.shape.n):
                    accumulator = cute.make_tensor(
                        tmem_pointer + m_panel * self.shape.envelope_n + n_panel * self.shape.n,
                        accumulator_layout,
                    )
                    for k_block in cutlass.range(
                        cute.size(a_fragment, mode=[2]),
                        unroll_full=True,
                    ):
                        mma.set(
                            tcgen05.Field.ACCUMULATE,
                            cutlass.Boolean(repeat != 0 or k_block != 0),
                        )
                        cute.gemm(
                            mma,
                            accumulator,
                            a_fragment[None, None, k_block, 0],
                            b_fragment[None, None, k_block, 0],
                            accumulator,
                        )
        phase.commit()

    @cute.kernel
    def kernel(
        self,
        mma: cute.TiledMma,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        accumulator_layout: cute.Layout,
        output: cute.Tensor,
        repeats: Int32,
    ):
        """Zero operands, execute one MMA warp, wait for completion, and store a sentinel."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        thread, _, _ = cute.arch.thread_idx()
        shared = utils.SmemAllocator().allocate(self.shared_type)

        words = cute.recast_ptr(shared.operands.a.data_ptr(), dtype=cutlass.Uint32)
        word = thread
        while word < self.operand_bytes // 4:
            words[word] = cutlass.Uint32(0)
            word += CTA_THREADS
        cute.arch.fence_view_async_shared()
        pipeline.NamedBarrier(barrier_id=2, num_threads=CTA_THREADS).arrive_and_wait()

        producer, consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, cute.arch.WARP_SIZE),
            barrier_storage=shared.completion_barrier.data_ptr(),
        ).make_participants()
        tmem = utils.TmemAllocator(
            shared.tmem_holder,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=CTA_THREADS),
            allocator_warp_id=MMA_WARP,
        )
        tmem.allocate(TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_pointer = tmem.retrieve_ptr(self.acc_type)

        a = shared.operands.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        b = shared.operands.b.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        if warp == MMA_WARP:
            self.run_mma(
                mma,
                mma.make_fragment_A(a),
                mma.make_fragment_B(b),
                accumulator_layout,
                tmem_pointer,
                repeats,
                producer,
            )
        elif warp == COMPLETE_WARP:
            phase = consumer.wait_and_advance()
            if thread == cute.arch.WARP_SIZE:
                output[cute.arch.block_idx()[0]] = Float32(1.0)
            phase.release()

        tmem.relinquish_alloc_permit()
        pipeline.NamedBarrier(barrier_id=3, num_threads=CTA_THREADS).arrive_and_wait()
        tmem.free(tmem_pointer)


def compile_probe(shape: Shape):
    """Compile one TVM-FFI specialization for a fixed probe shape."""
    op = TcgenThroughputProbe(shape)
    output = make_fake_compact_tensor(
        Float32,
        (cute.sym_int(),),
        stride_order=(0,),
        assumed_align=16,
    )
    return op, compile_tvm_ffi(op, output, 1, 1)


@dataclass(frozen=True)
class TimingKey:
    """Identify one graph-replayed shape, grid, and repeat count."""

    sweep: str
    shape: Shape
    grid_multiplier: int
    repeats: int


@dataclass(frozen=True)
class Fit:
    """Steady-state slope and derived physical throughput for one launch configuration."""

    shape: str
    grid_multiplier: int
    blocks: int
    slope_us: float
    intercept_us: float
    r_squared: float
    physical_tflops: float
    panels: int
    tcgen_instructions: int
    operand_smem_bytes: int


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


def benchmark_graphs(
    graphs: dict[TimingKey, torch.cuda.CUDAGraph],
    rounds: int,
) -> dict[TimingKey, CudaBenchmarkStats]:
    """Collect interleaved event timings while rotating candidate order."""
    keys = tuple(graphs)
    samples = {key: [] for key in keys}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for round_index in range(rounds):
        order = keys if round_index % 2 == 0 else tuple(reversed(keys))
        offset = round_index % len(order)
        for key in order[offset:] + order[:offset]:
            start.record()
            graphs[key].replay()
            end.record()
            torch.cuda.synchronize()
            samples[key].append(start.elapsed_time(end) * 1e3)
    return {key: CudaBenchmarkStats.from_samples(values) for key, values in samples.items()}


def linear_fit(xs: tuple[int, ...], ys: list[float]) -> tuple[float, float, float]:
    """Fit time = intercept + slope * repeats."""
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys, strict=True))
    total = sum((y - y_mean) ** 2 for y in ys)
    return slope, intercept, 1.0 if total == 0 else 1.0 - residual / total


def run(rounds: int, warmup: int, output_path: Path) -> dict[str, object]:
    """Compile both sweeps, benchmark them, print summaries, and write JSON."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("tcgen05 throughput requires an SM100 or newer CUDA GPU")

    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    compiled = {}
    metadata = {}
    for shape in {shape for sweep in SWEEPS for shape in sweep.shapes}:
        op, callable_ = compile_probe(shape)
        compiled[shape] = callable_
        metadata[shape] = (
            shape.panels * int(op.k_blocks),
            int(op.operand_bytes),
        )

    graphs = {}
    outputs = []
    for sweep in SWEEPS:
        for shape in sweep.shapes:
            for grid_multiplier in GRID_MULTIPLIERS:
                blocks = grid_multiplier * num_sms
                output = torch.empty(blocks, device="cuda", dtype=torch.float32)
                outputs.append(output)
                for repeats in REPEATS:
                    key = TimingKey(sweep.name, shape, grid_multiplier, repeats)
                    graphs[key] = capture_graph(
                        partial(compiled[shape], output, repeats, blocks),
                        warmup,
                    )

    for output in outputs:
        if not torch.all(output == 1.0).item():
            raise RuntimeError("a probe did not complete every launched CTA")
    stats = benchmark_graphs(graphs, rounds)

    sweep_reports = []
    for sweep in SWEEPS:
        fits = []
        for shape in sweep.shapes:
            instructions, operand_bytes = metadata[shape]
            for grid_multiplier in GRID_MULTIPLIERS:
                medians = [
                    stats[TimingKey(sweep.name, shape, grid_multiplier, repeats)].median_us
                    for repeats in REPEATS
                ]
                slope, intercept, r_squared = linear_fit(REPEATS, medians)
                blocks = grid_multiplier * num_sms
                fits.append(
                    Fit(
                        shape=shape.name,
                        grid_multiplier=grid_multiplier,
                        blocks=blocks,
                        slope_us=slope,
                        intercept_us=intercept,
                        r_squared=r_squared,
                        physical_tflops=blocks * shape.flops / slope / 1e6,
                        panels=shape.panels,
                        tcgen_instructions=instructions,
                        operand_smem_bytes=operand_bytes,
                    )
                )
        best = [
            max(
                (fit for fit in fits if fit.shape == shape.name),
                key=lambda fit: fit.physical_tflops,
            )
            for shape in sweep.shapes
        ]
        sweep_reports.append(
            {
                "name": sweep.name,
                "envelope": sweep.envelope,
                "best": [asdict(fit) for fit in best],
                "all_fits": [asdict(fit) for fit in fits],
            }
        )
        print(f"\n{sweep.name}: physical envelope {sweep.envelope}")
        for fit in best:
            print(
                f"  {fit.shape}: {fit.physical_tflops:8.1f} TFLOP/s, "
                f"grid={fit.grid_multiplier}x, instructions={fit.tcgen_instructions}, "
                f"R²={fit.r_squared:.5f}"
            )

    report = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutedsl": cutlass.__version__,
        "gpu": torch.cuda.get_device_name(),
        "num_sms": num_sms,
        "contract": (
            "zero-input fixed-pointer warm-cache CUDA Graph replay; normal DVFS; interleaved "
            "rotated order; slope excludes fitted launch/setup intercept"
        ),
        "repeats": REPEATS,
        "grid_multipliers": GRID_MULTIPLIERS,
        "rounds": rounds,
        "warmup": warmup,
        "sweeps": sweep_reports,
        "raw_timings": [
            {
                "sweep": key.sweep,
                "shape": key.shape.name,
                "grid_multiplier": key.grid_multiplier,
                "repeats": key.repeats,
                **asdict(value),
            }
            for key, value in stats.items()
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {output_path}")
    return report


def main(
    rounds: Annotated[int, typer.Option(min=5)] = DEFAULT_ROUNDS,
    warmup: Annotated[int, typer.Option(min=1)] = DEFAULT_WARMUP,
    output: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT,
) -> None:
    """Run the fixed CTA-group-1 throughput sweeps."""
    run(rounds, warmup, output)


if __name__ == "__main__":
    typer.run(main)
