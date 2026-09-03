"""Plain-text summary of one measured pipeline capture.

Perfetto is the tool for exploring a trace; this report answers the question that
follows every capture: how long is one steady-state iteration, which regions on
which roles fill it, and what does the scheduler say limits it.
"""

from __future__ import annotations

from collections import defaultdict
import statistics

from transformer_nuggets.cute.profiler.pipeline.analysis import PipelineAnalysis
from transformer_nuggets.cute.profiler.pipeline.iket import MeasuredCapture
from transformer_nuggets.cute.profiler.pipeline.plan import Timeline

# Approximate cost of one IKET range push/pop pair on the issuing warp (GB200,
# CuTeDSL 4.7). Regions shorter than twice this are dominated by instrumentation.
RANGE_OVERHEAD_NS = 50


def steady_iterations(iterations: int) -> range:
    """Return the interior iterations, excluding pipeline fill and drain."""
    if iterations < 3:
        return range(iterations)
    return range(1, iterations - 1)


def report(
    timeline: Timeline,
    measured: MeasuredCapture,
    analysis: PipelineAnalysis | None = None,
    *,
    unprofiled_iteration_ns: float | None = None,
) -> str:
    """Render per-region medians, one steady-state iteration, and scheduler findings."""
    iterations = timeline.iterations
    steady = steady_iterations(iterations)
    lines = [
        f"kernel {measured.kernel_name[:72]}",
        f"cta {measured.cta}  duration {measured.duration_ns} ns  "
        f"iterations {iterations}  per iteration ~{measured.duration_ns / iterations:.0f} ns"
        + (
            f"  (unprofiled ~{unprofiled_iteration_ns:.0f} ns, instrumentation "
            f"+{measured.duration_ns / iterations - unprofiled_iteration_ns:.0f} ns)"
            if unprofiled_iteration_ns is not None
            else ""
        ),
        "",
    ]

    durations: dict[tuple[str, str], list[float]] = defaultdict(list)
    for region in measured.regions:
        if region.iteration in steady:
            durations[(region.role, region.name)].append(region.median_duration_ns)
    role_order = {role.name: position for position, role in enumerate(timeline.roles)}
    lines.append(
        f"steady-state iterations {steady.start}..{steady.stop - 1}, ns per region "
        f"(median / min / max; * = under 2x the ~{RANGE_OVERHEAD_NS} ns range overhead)"
    )
    lines.append(f"  {'role':8} {'region':32} {'median':>8} {'min':>7} {'max':>7}")
    for (role, name), values in sorted(
        durations.items(), key=lambda item: (role_order[item[0][0]], item[0][1])
    ):
        median = statistics.median(values)
        flag = "*" if median < 2 * RANGE_OVERHEAD_NS else " "
        lines.append(
            f"  {role:8} {name:32} {median:8.0f} {min(values):7.0f} {max(values):7.0f} {flag}"
        )

    middle = iterations // 2
    lines.append("")
    lines.append(f"iteration {middle} timeline, ns from capture origin (start -> end, duration)")
    spans = sorted(
        (
            (region.start_ns - measured.origin_ns, region.end_ns - measured.origin_ns, region)
            for region in measured.regions
            if region.iteration == middle
        ),
        key=lambda item: item[0],
    )
    for start, end, region in spans:
        lines.append(
            f"  {region.role:8} {region.name:32} {start:8d} -> {end:8d}  ({end - start:5d})"
        )

    if analysis is not None:
        summary = analysis.perfetto_summary()
        lines.append("")
        lines.append("scheduler")
        lines.append(f"  RecMII {summary['recurrence_mii']}  ResMII {summary['resource_mii']}")
        lines.append(f"  critical cycle: {summary['critical_cycle']}")
        for ring in analysis.rings:
            if ring.reuse_wait_ns or ring.peak_occupancy == ring.depth:
                lines.append(
                    f"  ring {ring.resource}: peak {ring.peak_occupancy}/{ring.depth}, "
                    f"reuse wait {ring.reuse_wait_ns} ns"
                )
        saved = [item for item in analysis.depth_counterfactuals if item.saved > 0]
        for item in saved:
            lines.append(f"  depth+1 {item.resource}: saves {item.saved:.0f} ns")
    return "\n".join(lines)
