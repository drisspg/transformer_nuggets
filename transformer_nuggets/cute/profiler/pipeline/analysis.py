"""Critical-path, blocker, ring, and depth analysis for unrolled pipeline schedules."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
import statistics
from collections.abc import Mapping

from transformer_nuggets.cute.profiler.pipeline.iket import MeasuredCapture
from transformer_nuggets.cute.profiler.pipeline.plan import Dependency, Timeline

NodeKey = tuple[str, int]


@dataclass(frozen=True)
class ConcreteEdge:
    """One unrolled ordering or semantic dependency edge."""

    source: NodeKey
    target: NodeKey
    kind: str
    resource: str


@dataclass(frozen=True)
class CriticalPath:
    """Finite-DAG critical-path method result."""

    makespan: float
    earliest_start: dict[NodeKey, float]
    latest_start: dict[NodeKey, float]
    slack: dict[NodeKey, float]
    path: tuple[NodeKey, ...]


@dataclass(frozen=True)
class StageBlocker:
    """Measured wait/busy decomposition and latest-arriving predecessor."""

    name: str
    iteration: int
    start_ns: int
    end_ns: int
    duration_ns: int
    ready_ns: int
    wait_inside_ns: int
    ready_delay_ns: int
    busy_ns: int
    blocker: NodeKey | None
    blocker_kind: str | None
    blocker_resource: str | None
    blocker_robustness_ns: int | None
    slack_ns: float
    critical: bool


@dataclass(frozen=True)
class RingAnalysis:
    """Measured occupancy, run-ahead, and reuse backpressure for one ring."""

    resource: str
    depth: int
    peak_occupancy: int
    final_occupancy: int
    mean_occupancy: float
    full_fraction: float
    max_run_ahead: int
    reuse_wait_ns: int


@dataclass(frozen=True)
class DepthCounterfactual:
    """First-order makespan estimate after increasing one ring depth."""

    resource: str
    baseline_depth: int
    candidate_depth: int
    baseline_makespan: float
    candidate_makespan: float
    saved: float


@dataclass(frozen=True)
class CriticalCycle:
    """Maximum cycle-ratio result for the compact modulo schedule."""

    initiation_interval: float
    resource_initiation_interval: float
    nodes: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    total_weight: float
    total_distance: int
    cycles_checked: int


@dataclass(frozen=True)
class PipelineAnalysis:
    """Combined finite and steady-state scheduling analysis."""

    logical_critical_path: CriticalPath
    measured_critical_path: CriticalPath
    blockers: tuple[StageBlocker, ...]
    rings: tuple[RingAnalysis, ...]
    depth_counterfactuals: tuple[DepthCounterfactual, ...]
    logical_critical_cycle: CriticalCycle

    def perfetto_annotations(self) -> dict[NodeKey, dict[str, object]]:
        """Return click-detail metadata keyed by measured region identity."""
        cycle_nodes = set(self.logical_critical_cycle.nodes)
        annotations = {}
        for stage in self.blockers:
            key = stage.name, stage.iteration
            annotations[key] = {
                "ready_ns": stage.ready_ns,
                "wait_inside_ns": stage.wait_inside_ns,
                "ready_delay_ns": stage.ready_delay_ns,
                "busy_ns": stage.busy_ns,
                "blocker": format_node(stage.blocker),
                "blocker_kind": stage.blocker_kind or "none",
                "blocker_resource": stage.blocker_resource or "none",
                "blocker_robustness_ns": stage.blocker_robustness_ns or 0,
                "critical_path": stage.critical,
                "slack_ns": stage.slack_ns,
                "critical_cycle_member": stage.name in cycle_nodes,
            }
        return annotations

    def perfetto_summary(self) -> dict[str, object]:
        """Return compact trace-level scheduling findings."""
        summary: dict[str, object] = {
            "logical_makespan": self.logical_critical_path.makespan,
            "measured_modeled_makespan_ns": self.measured_critical_path.makespan,
            "recurrence_mii": self.logical_critical_cycle.initiation_interval,
            "resource_mii": self.logical_critical_cycle.resource_initiation_interval,
            "critical_cycle": " -> ".join(self.logical_critical_cycle.nodes),
            "critical_path": " -> ".join(
                format_node(node) for node in self.measured_critical_path.path
            ),
        }
        for ring in self.rings:
            summary[f"ring.{ring.resource}.peak"] = ring.peak_occupancy
            summary[f"ring.{ring.resource}.depth"] = ring.depth
            summary[f"ring.{ring.resource}.reuse_wait_ns"] = ring.reuse_wait_ns
        for candidate in self.depth_counterfactuals:
            summary[f"depth_plus_one.{candidate.resource}.saved_ns"] = candidate.saved
        return summary

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""
        return {
            "logical_critical_path": critical_path_dict(self.logical_critical_path),
            "measured_critical_path": critical_path_dict(self.measured_critical_path),
            "blockers": [asdict(item) for item in self.blockers],
            "rings": [asdict(item) for item in self.rings],
            "depth_counterfactuals": [asdict(item) for item in self.depth_counterfactuals],
            "logical_critical_cycle": asdict(self.logical_critical_cycle),
        }


def analyze_pipeline(
    timeline: Timeline,
    measured: MeasuredCapture,
    *,
    timer_epsilon_ns: int = 32,
) -> PipelineAnalysis:
    """Analyze the unrolled logical plan and one wait-inclusive measured capture."""
    edges = concrete_edges(timeline)
    logical_weights = {item.key: float(item.region.weight) for item in timeline.regions}
    logical_critical_path = critical_path(tuple(logical_weights), edges, logical_weights)
    blockers, measured_weights = measured_blockers(
        timeline,
        measured,
        edges,
        timer_epsilon_ns=timer_epsilon_ns,
    )
    measured_critical_path = critical_path(tuple(measured_weights), edges, measured_weights)
    blocker_by_key = {(item.name, item.iteration): item for item in blockers}
    blockers = tuple(
        replace(
            item,
            slack_ns=measured_critical_path.slack[(item.name, item.iteration)],
            critical=measured_critical_path.slack[(item.name, item.iteration)] == 0,
        )
        for item in blockers
    )
    return PipelineAnalysis(
        logical_critical_path=logical_critical_path,
        measured_critical_path=measured_critical_path,
        blockers=blockers,
        rings=ring_analysis(timeline, measured, blocker_by_key),
        depth_counterfactuals=depth_counterfactuals(
            timeline,
            measured_weights,
        ),
        logical_critical_cycle=critical_cycle(timeline, logical_weights),
    )


def concrete_edges(
    timeline: Timeline,
    *,
    depth_overrides: Mapping[str, int] | None = None,
) -> tuple[ConcreteEdge, ...]:
    """Reconstruct role serialization and semantic edges for an unrolled window."""
    depth_overrides = depth_overrides or {}
    edges: list[ConcreteEdge] = []
    regions_by_role: dict[str, list[object]] = defaultdict(list)
    for item in timeline.regions:
        if item.iteration == 0:
            regions_by_role[item.region.role].append(item.region)
    for role, regions in regions_by_role.items():
        regions.sort(key=lambda region: region.order)
        for iteration in range(timeline.iterations):
            for left, right in zip(regions, regions[1:], strict=False):
                edges.append(
                    ConcreteEdge(
                        (left.name, iteration),
                        (right.name, iteration),
                        "role",
                        role,
                    )
                )
            if iteration + 1 < timeline.iterations:
                edges.append(
                    ConcreteEdge(
                        (regions[-1].name, iteration),
                        (regions[0].name, iteration + 1),
                        "role",
                        role,
                    )
                )

    for dependency in unique_dependencies(timeline):
        distance = (
            depth_overrides.get(dependency.resource, dependency.distance)
            if dependency.kind == "reuse"
            else dependency.distance
        )
        for source_iteration in range(timeline.iterations):
            target_iteration = source_iteration + distance
            if target_iteration >= timeline.iterations:
                continue
            edges.append(
                ConcreteEdge(
                    (dependency.source, source_iteration),
                    (dependency.target, target_iteration),
                    dependency.kind,
                    dependency.resource,
                )
            )
    return tuple(unique_edges(edges))


def critical_path(
    nodes: tuple[NodeKey, ...],
    edges: tuple[ConcreteEdge, ...],
    weights: Mapping[NodeKey, float],
) -> CriticalPath:
    """Compute earliest/latest times, slack, and one deterministic critical path."""
    predecessors: dict[NodeKey, list[ConcreteEdge]] = defaultdict(list)
    successors: dict[NodeKey, list[ConcreteEdge]] = defaultdict(list)
    indegree: dict[NodeKey, int] = dict.fromkeys(nodes, 0)
    for edge in edges:
        predecessors[edge.target].append(edge)
        successors[edge.source].append(edge)
        indegree[edge.target] += 1
    queue = deque(node for node in nodes if indegree[node] == 0)
    order: list[NodeKey] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for edge in successors[node]:
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                queue.append(edge.target)
    if len(order) != len(nodes):
        raise ValueError("Critical-path graph is cyclic after finite unrolling")

    earliest: dict[NodeKey, float] = {}
    binding: dict[NodeKey, ConcreteEdge] = {}
    for node in order:
        candidates = [
            (earliest[edge.source] + weights[edge.source], edge) for edge in predecessors[node]
        ]
        if candidates:
            end, edge = max(
                candidates,
                key=lambda item: (item[0], item[1].kind == "role", item[1].source),
            )
            earliest[node] = end
            binding[node] = edge
        else:
            earliest[node] = 0.0
    makespan = max(earliest[node] + weights[node] for node in nodes)

    latest_finish: dict[NodeKey, float] = {}
    latest_start: dict[NodeKey, float] = {}
    for node in reversed(order):
        if successors[node]:
            latest_finish[node] = min(latest_start[edge.target] for edge in successors[node])
        else:
            latest_finish[node] = makespan
        latest_start[node] = latest_finish[node] - weights[node]
    slack = {node: latest_start[node] - earliest[node] for node in nodes}

    sink = max(nodes, key=lambda node: (earliest[node] + weights[node], node))
    path = [sink]
    while path[-1] in binding:
        path.append(binding[path[-1]].source)
    path.reverse()
    return CriticalPath(makespan, earliest, latest_start, slack, tuple(path))


def measured_blockers(
    timeline: Timeline,
    measured: MeasuredCapture,
    edges: tuple[ConcreteEdge, ...],
    *,
    timer_epsilon_ns: int,
) -> tuple[tuple[StageBlocker, ...], dict[NodeKey, float]]:
    """Attribute wait-inclusive overlap to each stage's latest-arriving predecessor."""
    measured_by_key = {(item.name, item.iteration): item for item in measured.regions}
    predecessors: dict[NodeKey, list[ConcreteEdge]] = defaultdict(list)
    for edge in edges:
        predecessors[edge.target].append(edge)
    blockers: list[StageBlocker] = []
    weights: dict[NodeKey, float] = {}
    for key, item in measured_by_key.items():
        start_ns = item.start_ns - measured.origin_ns
        end_ns = item.end_ns - measured.origin_ns
        arrivals = sorted(
            [
                (
                    measured_by_key[edge.source].end_ns - measured.origin_ns,
                    edge,
                )
                for edge in predecessors[key]
            ],
            key=lambda item: (item[0], item[1].kind, item[1].source),
        )
        ready_ns = arrivals[-1][0] if arrivals else 0
        binding = arrivals[-1][1] if arrivals else None
        robustness = ready_ns - arrivals[-2][0] if len(arrivals) > 1 else None
        wait_inside_ns = min(max(ready_ns - start_ns, 0), end_ns - start_ns)
        if wait_inside_ns <= timer_epsilon_ns:
            wait_inside_ns = 0
        ready_delay_ns = max(start_ns - ready_ns, 0)
        busy_ns = max(end_ns - max(start_ns, ready_ns), 0)
        weights[key] = float(busy_ns + ready_delay_ns)
        blockers.append(
            StageBlocker(
                name=key[0],
                iteration=key[1],
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=end_ns - start_ns,
                ready_ns=ready_ns,
                wait_inside_ns=wait_inside_ns,
                ready_delay_ns=ready_delay_ns,
                busy_ns=busy_ns,
                blocker=binding.source if binding and wait_inside_ns else None,
                blocker_kind=binding.kind if binding and wait_inside_ns else None,
                blocker_resource=binding.resource if binding and wait_inside_ns else None,
                blocker_robustness_ns=robustness,
                slack_ns=0.0,
                critical=False,
            )
        )
    blockers.sort(key=lambda item: (item.start_ns, item.name, item.iteration))
    return tuple(blockers), weights


def ring_occupancy_samples(
    timeline: Timeline,
    measured: MeasuredCapture,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return normalized occupancy counter samples for every released resource."""
    measured_by_key = {(item.name, item.iteration): item for item in measured.regions}
    samples = {}
    for dependency in unique_dependencies(timeline):
        if dependency.kind != "reuse":
            continue
        deltas: dict[int, int] = defaultdict(int)
        for iteration in range(timeline.iterations):
            producer = measured_by_key.get((dependency.target, iteration))
            release = measured_by_key.get((dependency.source, iteration))
            if producer is not None:
                deltas[producer.end_ns - measured.origin_ns] += 1
            if release is not None:
                deltas[release.end_ns - measured.origin_ns] -= 1
        occupancy = 0
        resource_samples = [(0, 0)]
        for timestamp, delta in sorted(deltas.items()):
            occupancy += delta
            resource_samples.append((timestamp, occupancy))
        samples[dependency.resource] = tuple(resource_samples)
    return samples


def ring_analysis(
    timeline: Timeline,
    measured: MeasuredCapture,
    blockers: Mapping[NodeKey, StageBlocker],
) -> tuple[RingAnalysis, ...]:
    """Measure ring occupancy, run-ahead, and reuse-attributed wait."""
    measured_by_key = {(item.name, item.iteration): item for item in measured.regions}
    resources = {resource.name: resource for resource in timeline.resources}
    results = []
    for dependency in unique_dependencies(timeline):
        if dependency.kind != "reuse":
            continue
        resource = resources[dependency.resource]
        events: list[tuple[int, int]] = []
        fills: list[int] = []
        releases: list[int] = []
        for iteration in range(timeline.iterations):
            producer_key = dependency.target, iteration
            release_key = dependency.source, iteration
            if producer_key in measured_by_key:
                timestamp = measured_by_key[producer_key].end_ns - measured.origin_ns
                fills.append(timestamp)
                events.append((timestamp, 1))
            if release_key in measured_by_key:
                timestamp = measured_by_key[release_key].end_ns - measured.origin_ns
                releases.append(timestamp)
                events.append((timestamp, -1))
        events.sort(key=lambda event: (event[0], event[1]))
        occupancy = 0
        peak = 0
        area = 0
        full_time = 0
        previous_time = events[0][0]
        for timestamp, delta in events:
            duration = timestamp - previous_time
            area += occupancy * duration
            if occupancy == resource.depth:
                full_time += duration
            occupancy += delta
            peak = max(peak, occupancy)
            previous_time = timestamp
        span = max(events[-1][0] - events[0][0], 1)
        max_run_ahead = 0
        for fill_index, timestamp in enumerate(sorted(fills), start=1):
            released = sum(release <= timestamp for release in releases)
            max_run_ahead = max(max_run_ahead, fill_index - released)
        reuse_wait_ns = sum(
            stage.wait_inside_ns
            for key, stage in blockers.items()
            if stage.blocker_kind == "reuse" and stage.blocker_resource == dependency.resource
        )
        results.append(
            RingAnalysis(
                resource=dependency.resource,
                depth=resource.depth,
                peak_occupancy=peak,
                final_occupancy=occupancy,
                mean_occupancy=area / span,
                full_fraction=full_time / span,
                max_run_ahead=max_run_ahead,
                reuse_wait_ns=reuse_wait_ns,
            )
        )
    return tuple(results)


def depth_counterfactuals(
    timeline: Timeline,
    weights: Mapping[NodeKey, float],
) -> tuple[DepthCounterfactual, ...]:
    """Estimate depth+1 benefit by relaxing one reuse distance at a time."""
    nodes = tuple(weights)
    baseline = critical_path(nodes, concrete_edges(timeline), weights).makespan
    resources = {resource.name: resource for resource in timeline.resources}
    results = []
    for dependency in unique_dependencies(timeline):
        if dependency.kind != "reuse":
            continue
        resource = resources[dependency.resource]
        candidate = critical_path(
            nodes,
            concrete_edges(
                timeline,
                depth_overrides={dependency.resource: resource.depth + 1},
            ),
            weights,
        ).makespan
        results.append(
            DepthCounterfactual(
                dependency.resource,
                resource.depth,
                resource.depth + 1,
                baseline,
                candidate,
                baseline - candidate,
            )
        )
    return tuple(results)


def critical_cycle(
    timeline: Timeline,
    logical_weights: Mapping[NodeKey, float],
    *,
    cycle_cap: int = 10_000,
) -> CriticalCycle:
    """Enumerate compact simple cycles and return the maximum weight/distance ratio."""
    region_weights = {
        name: statistics.median(
            weight for (region_name, _), weight in logical_weights.items() if region_name == name
        )
        for name, _ in logical_weights
    }
    symbolic: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    roles: dict[str, list[object]] = defaultdict(list)
    for item in timeline.regions:
        if item.iteration == 0:
            roles[item.region.role].append(item.region)
    for role, regions in roles.items():
        regions.sort(key=lambda region: region.order)
        for left, right in zip(regions, regions[1:], strict=False):
            symbolic[left.name].append((right.name, "role", 0))
        symbolic[regions[-1].name].append((regions[0].name, f"role:{role}", 1))
    for dependency in unique_dependencies(timeline):
        symbolic[dependency.source].append(
            (dependency.target, f"{dependency.kind}:{dependency.resource}", dependency.distance)
        )

    cycles: list[tuple[tuple[str, ...], tuple[str, ...], float, int]] = []
    for start in sorted(region_weights):
        stack: list[tuple[str, list[str], list[str], int]] = [(start, [start], [], 0)]
        while stack:
            node, path, kinds, distance = stack.pop()
            for target, kind, edge_distance in symbolic[node]:
                if target == start:
                    total_distance = distance + edge_distance
                    if total_distance:
                        total_weight = sum(region_weights[item] for item in path)
                        cycles.append(
                            (tuple(path), tuple(kinds + [kind]), total_weight, total_distance)
                        )
                        if len(cycles) > cycle_cap:
                            raise ValueError(f"Cycle enumeration exceeded cap {cycle_cap}")
                elif target not in path and target >= start:
                    stack.append(
                        (target, path + [target], kinds + [kind], distance + edge_distance)
                    )
    if not cycles:
        raise ValueError("Modulo schedule contains no positive-distance cycle")
    nodes, kinds, total_weight, total_distance = max(
        cycles,
        key=lambda cycle: (cycle[2] / cycle[3], cycle[0]),
    )
    role_busy = max(
        sum(region_weights[region.name] for region in regions) for regions in roles.values()
    )
    return CriticalCycle(
        initiation_interval=total_weight / total_distance,
        resource_initiation_interval=role_busy,
        nodes=nodes,
        edge_kinds=kinds,
        total_weight=total_weight,
        total_distance=total_distance,
        cycles_checked=len(cycles),
    )


def unique_dependencies(timeline: Timeline) -> tuple[Dependency, ...]:
    """Return one copy of each symbolic dependency declaration."""
    unique = {}
    for item in timeline.dependencies:
        dependency = item.dependency
        key = (
            dependency.source,
            dependency.target,
            dependency.resource,
            dependency.distance,
            dependency.kind,
        )
        unique.setdefault(key, dependency)
    return tuple(unique.values())


def unique_edges(edges: list[ConcreteEdge]) -> list[ConcreteEdge]:
    """Deduplicate exact concrete edges while preserving order."""
    return list(dict.fromkeys(edges))


def critical_path_dict(result: CriticalPath) -> dict[str, object]:
    """Serialize a critical-path result with string node keys."""
    return {
        "makespan": result.makespan,
        "path": [format_node(node) for node in result.path],
        "earliest_start": {
            format_node(node): value for node, value in result.earliest_start.items()
        },
        "latest_start": {format_node(node): value for node, value in result.latest_start.items()},
        "slack": {format_node(node): value for node, value in result.slack.items()},
    }


def format_node(node: NodeKey | None) -> str:
    """Format one concrete node identity for reports and Perfetto details."""
    return "none" if node is None else f"{node[0]}[{node[1]}]"
