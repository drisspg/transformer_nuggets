"""Capture the sampled producer/consumer pipeline of the CuTeDSL SGEMM.

    SGEMM_PROFILE=1 python examples/cute_sgemm_profile.py
    SGEMM_PROFILE=1 python examples/cute_sgemm_profile.py --k 260 --config small --stages 2

Requires Blackwell or newer. SGEMM_PROFILE=1 selects the public API's compile-time
profiling specialization; this example never changes the environment variable.
Only CTA (0, 0), producer warp config.num_threads // 32, and consumer warp 0 are
sampled. Four compact records per tile cover the first min(ceil(K/tile_k), 32)
tiles, not the whole kernel or all consumer warps.

The producer_issue region measures TMA enqueue, not copy completion. The
consumer_compute region includes stage release before the final register FMAs;
its end is not the release timestamp. No copy-completion or release flows are
inferred. Instrumented widths and gaps are not production idle-time estimates.
"""

import math
import os
import statistics
from collections.abc import Callable
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated

import torch
import typer

from transformer_nuggets.cute.profiler import (
    Event,
    PostProcessContext,
    profile_session,
    summarize_by_tag,
)
from transformer_nuggets.cute.profiler.postprocessors import (
    compose,
    rename_processes,
    rename_threads,
)
from transformer_nuggets.cute.sgemm import (
    SGEMM_PROFILE_EVENTS,
    SGEMM_PROFILE_TAGS,
    SGEMM_PROFILE_TILES,
    Sgemm,
    SgemmConfig,
    SgemmProfileTag,
    operand_major,
    sgemm,
)
from transformer_nuggets.utils.benchmark import benchmark_cuda_graph_stats


class Layout(str, Enum):
    row = "row"
    col = "col"


class Config(str, Enum):
    default = "default"
    small = "small"


def make_operand(rows: int, cols: int, layout: Layout) -> torch.Tensor:
    tensor = torch.empty(rows, cols, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    return tensor if layout == Layout.row else tensor.t().contiguous().t()


def check_output(output: torch.Tensor, reference: torch.Tensor, error_bound: torch.Tensor) -> None:
    """Check pointwise error against the sequential FP32 FMA rounding bound."""
    normalized_error = (output.double() - reference).abs() / error_bound
    torch.testing.assert_close(normalized_error, torch.zeros_like(reference), atol=1.0, rtol=0.0)


def capture_checked(
    launch: Callable[[], torch.Tensor],
    output: torch.Tensor,
    reference: torch.Tensor,
    error_bound: torch.Tensor,
) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        launch()
    check_output(output, reference, error_bound)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    # A correct eager result must not mask an empty graph or a stale output.
    output.fill_(math.nan)
    graph.replay()
    check_output(output, reference, error_bound)
    torch.cuda.synchronize()
    return graph


def main(
    m: Annotated[int, typer.Option(min=4)] = 1024,
    n: Annotated[int, typer.Option(min=4)] = 1024,
    k: Annotated[
        int, typer.Option(min=4, max=2**24 - 4, help="FP32 FMA bound requires K < 2**24.")
    ] = 1024,
    config: Annotated[
        Config,
        typer.Option(help="default: 128x128/256 consumer threads; small: 64x128/128 threads."),
    ] = Config.default,
    tile_k: Annotated[int, typer.Option(help="K tile size: 8, 16, or 32.")] = 16,
    stages: Annotated[int, typer.Option(min=2)] = 4,
    a_layout: Annotated[Layout, typer.Option(help="Storage layout of A (M, K).")] = Layout.row,
    b_layout: Annotated[Layout, typer.Option(help="Storage layout of B (K, N).")] = Layout.row,
    trace_path: Path = Path("agent_space/sgemm_profile.pftrace"),
) -> None:
    """Validate SGEMM, compare profiling overhead, and export one warmed launch."""
    if any(dim % 4 for dim in (m, n, k)):
        raise typer.BadParameter("M, N, and K must be positive multiples of 4")
    if tile_k not in (8, 16, 32):
        raise typer.BadParameter("--tile-k must be 8, 16, or 32")
    if os.environ.get("SGEMM_PROFILE") != "1":
        raise typer.BadParameter(
            "Set SGEMM_PROFILE=1 to enable tracing, e.g. "
            "SGEMM_PROFILE=1 python examples/cute_sgemm_profile.py"
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("This TMA/FFMA2 SGEMM requires a Blackwell or newer CUDA GPU")

    small = config == Config.small
    cfg = SgemmConfig(
        tile_m=64 if small else 128,
        tile_k=tile_k,
        num_stages=stages,
        num_threads=128 if small else 256,
        threads_n=8 if small else 16,
    )
    torch.manual_seed(0)
    a = make_operand(m, k, a_layout)
    b = make_operand(k, n, b_layout)
    output = torch.empty(m, n, device="cuda", dtype=torch.float32)
    reference = a.double() @ b.double()
    # Sequential fused multiply-add error is <= gamma_K * sum(abs(a_i * b_i)).
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    gamma_k = k * unit_roundoff / (1 - k * unit_roundoff)
    error_bound = (gamma_k * (a.double().abs() @ b.double().abs())).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    producer_warp = cfg.num_threads // 32
    sampled_tiles = min(math.ceil(k / cfg.tile_k), SGEMM_PROFILE_TILES)

    def group_roles(events: list[Event], _ctx: PostProcessContext) -> list[Event]:
        for event in events:
            event.pid = event.unit_id
            event.tid = producer_warp if event.tag_id <= SgemmProfileTag.PRODUCER_ISSUE else 0
            assert event.event_idx is not None
            event.extra_args = {"k_tile": event.event_idx // len(SGEMM_PROFILE_TAGS)}
            if event.tag_id == SgemmProfileTag.PRODUCER_ISSUE:
                event.extra_args["meaning"] = "TMA enqueue, not copy completion"
            if event.tag_id == SgemmProfileTag.CONSUMER_COMPUTE:
                event.extra_args["meaning"] = "Stage release occurs before the final register FMAs"
        return events

    print(f"GPU: {torch.cuda.get_device_name()}; seed=0; float32 uniform[-1,1]")
    print(f"M,N,K={m},{n},{k}; A={a_layout.value}; B={b_layout.value}; config={cfg.name}")
    print("SGEMM_PROFILE=1; explicit A/B overrides: enable_profiling=False / True")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_session(
        max_events_per_unit=SGEMM_PROFILE_EVENTS,
        num_units=(1, "CTA"),
        tag_names=list(SGEMM_PROFILE_TAGS),
        trace_path=str(trace_path),
        compact=True,
        expected_events_per_unit=len(SGEMM_PROFILE_TAGS) * sampled_tiles,
        post_process_events=group_roles,
        post_process_trace=compose(
            rename_processes({0: "SGEMM CTA (0, 0)"}),
            rename_threads(
                {producer_warp: f"Producer warp {producer_warp}", 0: "Consumer warp 0"}
            ),
        ),
    ) as session:
        graphs = {}
        for enabled in (False, True):
            op = Sgemm(
                cfg,
                a_major=operand_major(a),
                b_major=operand_major(b.t()),
                enable_profiling=enabled,
            )
            assert op.enable_profiling == enabled
            print(f"  enable_profiling={enabled}: {op.get_name()}")
            launch = partial(
                op.interface,
                a,
                b,
                output,
                profile_buffer=session.prof.tensor if enabled else None,
            )
            graphs[enabled] = capture_checked(launch, output, reference, error_bound)

        # Both graphs reuse identical pointers and contain exactly one interface launch.
        samples = {False: [], True: []}
        for round_idx in range(5):
            order = (False, True) if round_idx % 2 == 0 else (True, False)
            for enabled in order:
                stats = benchmark_cuda_graph_stats(graphs[enabled], num_iters=100, warmup_iters=20)
                samples[enabled].append(stats.median_us)

        # Exercise the environment-selected public API too; retain just one final launch.
        for _ in range(3):
            sgemm(a, b, output, cfg, profile_buffer=session.prof.tensor)
        torch.cuda.synchronize()
        session.prof.reset()
        sgemm(a, b, output, cfg, profile_buffer=session.prof.tensor)

    check_output(output, reference, error_bound)
    print("FP64 checks passed for both specializations, poisoned graph replay, and trace launch")
    print(
        "Warm fixed-pointer CUDA graph microbenchmark: one GEMM/replay, five alternating "
        "rounds, 100 measured + 20 warmup replays/round; CUDA-event time includes replay "
        "overhead, excludes allocation/compilation/capture. Clocks/power externally managed; "
        "no stabilization or cold-DRAM claim."
    )
    medians = {enabled: statistics.median(values) for enabled, values in samples.items()}
    for enabled, values in samples.items():
        print(
            f"  enable_profiling={enabled}: median={medians[enabled]:.3f} us; "
            f"round medians (us)={[round(value, 3) for value in values]}"
        )
    ratio = medians[True] / medians[False]
    print(f"  profiled/unprofiled latency: {ratio:.3f}x ({100 * (ratio - 1):+.1f}% overhead)")
    print(f"Sampled CTA (0, 0): {sampled_tiles} K tiles, {len(session.events)} compact events")
    for name, summary in summarize_by_tag(session.events).items():
        print(f"  {name}: count={summary.count}, median={summary.p50_ns:.0f} ns")
    print("producer_issue = enqueue, NOT copy completion; compute end is NOT stage release.")
    print(
        "Use this perturbed trace for ordering/wait identity, not production idle-time estimates."
    )
    print(f"Trace: {trace_path.resolve()}")


if __name__ == "__main__":
    typer.run(main)
