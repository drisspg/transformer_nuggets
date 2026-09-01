"""Decode IKET JSON ranges and join them to annotated logical pipeline regions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics

from transformer_nuggets.cute.profiler.pipeline.plan import Timeline


@dataclass(frozen=True)
class MeasuredRegion:
    """Aggregated IKET range envelope for one role region and iteration."""

    name: str
    role: str
    iteration: int
    start_ns: int
    end_ns: int
    median_duration_ns: float
    sample_count: int


@dataclass(frozen=True)
class MeasuredCapture:
    """One CTA's measured region timeline joined to a logical plan."""

    source: str
    kernel_name: str
    cta: tuple[int, int, int]
    origin_ns: int
    duration_ns: int
    regions: tuple[MeasuredRegion, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a browser-friendly normalized capture."""
        return {
            "source": self.source,
            "kernel_name": self.kernel_name,
            "cta": list(self.cta),
            "origin_ns": self.origin_ns,
            "duration_ns": self.duration_ns,
            "regions": [
                {
                    **asdict(region),
                    "start_ns": region.start_ns - self.origin_ns,
                    "end_ns": region.end_ns - self.origin_ns,
                }
                for region in self.regions
            ],
        }


def load_iket_capture(
    path: str | Path,
    timeline: Timeline,
    *,
    cta: tuple[int, int, int] = (0, 0, 0),
) -> MeasuredCapture:
    """Aggregate IKET ranges for ``cta`` into the timeline's role/iteration identities."""
    source = Path(path)
    payload = json.loads(source.read_text())
    launches = payload.get("launches", [])
    if len(launches) != 1:
        raise ValueError(f"Expected one IKET launch, found {len(launches)}")
    launch = launches[0]
    names = payload["stringTable"]
    locations = payload["locationTable"]
    role_by_region = {
        scheduled.region.name: scheduled.region.role for scheduled in timeline.regions
    }
    role_specs = {role.name: role for role in timeline.roles}

    ranges_by_warp: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for item in launch.get("ranges", []):
        location = locations[item["warpLocIdxs"][0]]
        if tuple(location["ctaId"]) != cta:
            continue
        name = names[item["rangeNameIdx"]]
        role_name = role_by_region.get(name)
        if role_name is None:
            continue
        role = role_specs[role_name]
        warp_id = location["warpId"]
        if not role.warp_start <= warp_id < role.warp_end:
            raise ValueError(
                f"IKET range {name!r} appeared on warp {warp_id}, outside role {role_name!r}"
            )
        ranges_by_warp[warp_id].append((item["startTs"], item["endTs"], name))

    grouped: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for warp_ranges in ranges_by_warp.values():
        occurrences: dict[str, int] = defaultdict(int)
        for start_ns, end_ns, name in sorted(warp_ranges):
            iteration = occurrences[name]
            occurrences[name] += 1
            if iteration < timeline.iterations:
                grouped[(name, iteration)].append((start_ns, end_ns))

    expected = {scheduled.key for scheduled in timeline.regions}
    missing = sorted(expected - grouped.keys())
    if missing:
        raise ValueError(f"IKET capture is missing annotated regions: {missing}")

    measured: list[MeasuredRegion] = []
    for (name, iteration), samples in grouped.items():
        role = role_by_region[name]
        measured.append(
            MeasuredRegion(
                name=name,
                role=role,
                iteration=iteration,
                start_ns=min(start for start, _ in samples),
                end_ns=max(end for _, end in samples),
                median_duration_ns=statistics.median(end - start for start, end in samples),
                sample_count=len(samples),
            )
        )
    measured.sort(key=lambda item: (item.start_ns, item.role, item.iteration, item.name))
    origin_ns = min(item.start_ns for item in measured)
    end_ns = max(item.end_ns for item in measured)
    return MeasuredCapture(
        source=str(source),
        kernel_name=launch["kernelName"],
        cta=cta,
        origin_ns=origin_ns,
        duration_ns=end_ns - origin_ns,
        regions=tuple(measured),
    )
