"""Adapt CUTLASS experimental Task Scheduling metadata to ``PipelinePlan``.

The adapter is intentionally duck-typed so importing this host-only module does
not require a particular CuTeDSL release. It consumes the stable concepts present
in CUTLASS 4.7: tasks, warp ranges, resources, pipeline configs, and normalized
loop schedule entries.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from transformer_nuggets.cute.profiler.pipeline.plan import (
    Dependency,
    PipelinePlan,
    Region,
    Resource,
    Role,
)

ROLE_COLORS = (
    "#4f9cf9",
    "#5bc8a5",
    "#ef6a8a",
    "#c88bf2",
    "#f0b35b",
    "#63c7da",
)


def plan_from_task_manager(
    manager,
    *,
    name: str = "task_scheduling_kernel",
    iteration_name: str = "domain",
) -> PipelinePlan:
    """Convert one TaskManager-like object into the neutral timeline plan."""
    plan = PipelinePlan(name, iteration_name=iteration_name)
    tasks = sorted(manager.tasks, key=lambda task: (int(task.warp_start), task.name))
    task_names = unique_task_names(tasks)

    resources = unique_by_identity(
        resource
        for task in tasks
        for resource in tuple(task.src_resources) + tuple(task.dst_resources)
        if resource_name(resource)
    )
    for resource in resources:
        config = resource.pipeline_config
        pipeline_type = enum_text(config.pipeline_type) if config else None
        plan.add_resource(
            Resource(
                name=resource_name(resource),
                label=resource_name(resource),
                depth=int(config.num_stages) if config else 1,
                storage=resource_storage(resource),
                description=(
                    f"CUTLASS Task Scheduling resource; pipeline={pipeline_type}."
                    if pipeline_type
                    else "CUTLASS Task Scheduling resource without a staged pipeline."
                ),
            )
        )

    regions_by_task: dict[int, list[str]] = defaultdict(list)
    producer_regions: dict[tuple[int, int], str] = {}
    consumer_regions: dict[tuple[int, int], str] = {}
    for task_index, task in enumerate(tasks):
        task_name = task_names[id(task)]
        plan.add_role(
            Role(
                name=task_name,
                label=task.name or task_name,
                warp_start=int(task.warp_start),
                warp_end=int(task.warp_end),
                color=ROLE_COLORS[task_index % len(ROLE_COLORS)],
            )
        )
        work_entries = loop_work_entries(task)
        if not work_entries:
            region_name = f"{task_name}.work"
            plan.add_region(
                Region(
                    region_name,
                    task_name,
                    task.name or task_name,
                    0,
                    1,
                    "Task Scheduling task body.",
                )
            )
            regions_by_task[id(task)].append(region_name)
            continue

        for order, (resource, stage_name, call_id, label) in enumerate(work_entries):
            resource_label = resource_name(resource)
            work_label = label or stage_name
            region_name = unique_region_name(
                plan,
                f"{task_name}.{resource_label}.{work_label}.{call_id}",
            )
            plan.add_region(
                Region(
                    region_name,
                    task_name,
                    f"{resource_label}: {work_label}",
                    order,
                    1,
                    f"Task Scheduling {stage_name} on {resource_label}.",
                )
            )
            regions_by_task[id(task)].append(region_name)
            key = id(task), id(resource)
            if "ProducerWork" in stage_name:
                producer_regions[key] = region_name
            if "ConsumerWork" in stage_name:
                consumer_regions[key] = region_name

    for resource in resources:
        producer_tasks = [
            task for task in tasks if contains_identity(task.dst_resources, resource)
        ]
        consumer_tasks = [
            task for task in tasks if contains_identity(task.src_resources, resource)
        ]
        if not producer_tasks or not consumer_tasks:
            continue
        for producer_task in producer_tasks:
            source = producer_regions.get((id(producer_task), id(resource)))
            if source is None:
                source = regions_by_task[id(producer_task)][-1]
            for consumer_task in consumer_tasks:
                if id(producer_task) == id(consumer_task):
                    continue
                target = consumer_regions.get((id(consumer_task), id(resource)))
                if target is None:
                    target = regions_by_task[id(consumer_task)][0]
                plan.add_dependency(
                    Dependency(
                        source,
                        target,
                        resource_name(resource),
                        kind="data",
                        label=resource_name(resource),
                    )
                )
                config = resource.pipeline_config
                if config is not None:
                    plan.add_dependency(
                        Dependency(
                            target,
                            source,
                            resource_name(resource),
                            distance=int(config.num_stages),
                            kind="reuse",
                            label=f"reuse {resource_name(resource)}",
                        )
                    )
    return plan


def loop_work_entries(task) -> list[tuple[object, str, int, str | None]]:
    """Return normalized loop work entries from a CUTLASS 4.7 Task."""
    entries = []
    for resource, stage, call_id, _guard, label in task.loop_schedule_list:
        stage_name = enum_text(stage)
        if "Work" in stage_name:
            entries.append((resource, stage_name, int(call_id), label))
    return entries


def unique_task_names(tasks: Iterable[object]) -> dict[int, str]:
    """Return stable unique role names for possibly duplicate task labels."""
    names: dict[int, str] = {}
    counts: dict[str, int] = defaultdict(int)
    for index, task in enumerate(tasks):
        base = sanitize_name(task.name or f"task_{index}")
        occurrence = counts[base]
        counts[base] += 1
        names[id(task)] = base if occurrence == 0 else f"{base}_{occurrence}"
    return names


def unique_region_name(plan: PipelinePlan, base: str) -> str:
    """Add a numeric suffix when a schedule repeats the same work label."""
    candidate = sanitize_name(base)
    suffix = 1
    while candidate in plan.regions:
        candidate = f"{sanitize_name(base)}_{suffix}"
        suffix += 1
    return candidate


def resource_name(resource) -> str:
    """Return the Task Scheduling resource's required human-readable name."""
    return str(resource.name)


def resource_storage(resource) -> str:
    """Infer a display storage class from the resource type name."""
    class_name = type(resource).__name__.lower()
    if "smem" in class_name:
        return "SMEM"
    if "tmem" in class_name:
        return "TMEM"
    if "gmem" in class_name:
        return "GMEM"
    if "workqueue" in class_name:
        return "control"
    return "resource"


def enum_text(value: object) -> str:
    """Return a stable string for enum-like CUTLASS values."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def contains_identity(values: Iterable[object], target: object) -> bool:
    """Return whether ``values`` contains ``target`` by object identity."""
    return any(id(value) == id(target) for value in values)


def unique_by_identity(values: Iterable[object]) -> list[object]:
    """Deduplicate an iterable while preserving identity-based order."""
    result = []
    seen: set[int] = set()
    for value in values:
        if id(value) in seen:
            continue
        seen.add(id(value))
        result.append(value)
    return result


def sanitize_name(name: str) -> str:
    """Convert a display label into a stable dotted identifier."""
    cleaned = "".join(character.lower() if character.isalnum() else "." for character in name)
    return ".".join(part for part in cleaned.split(".") if part) or "unnamed"
