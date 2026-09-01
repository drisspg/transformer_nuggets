"""Inline annotation markers and a static extractor for CuTeDSL role kernels.

``PipelineAnnotations`` emits no device IR unless IKET is enabled. The static
extractor reads the same declarations from Python source and lowers them to the
neutral ``PipelinePlan`` model.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from collections.abc import Callable, Iterable

from transformer_nuggets.cute.profiler.pipeline.plan import (
    Dependency,
    PipelinePlan,
    Region,
    Resource,
    Role,
    SourceLocation,
)


def load_iket():
    """Load the optional IKET API with an actionable version error."""
    try:
        from cutlass.cute.experimental import iket
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "IKET pipeline ranges require a CuTeDSL release that provides "
            "cutlass.cute.experimental.iket"
        ) from exc
    return iket


class PipelineAnnotations:
    """Source markers with optional IKET instrumentation during CuTeDSL tracing."""

    def __init__(self, name: str, *, iteration_name: str) -> None:
        self.name = name
        self.iteration_name = iteration_name
        self.iket_enabled = False
        self.active_role: str | None = None
        self.region_open = False

    def enable_iket(self) -> None:
        """Enable IKET ranges for subsequent kernel compilation in this process."""
        self.iket_enabled = True

    def role(
        self,
        name: str,
        *,
        label: str,
        warp_start: int,
        warp_end: int,
        color: str,
    ) -> Callable[[Callable], Callable]:
        """Delimit optional IKET region switching for one annotated role."""

        def decorate(function: Callable) -> Callable:
            @wraps(function)
            def wrapped(*args, **kwargs):
                if self.active_role is not None:
                    raise RuntimeError(
                        f"Pipeline role {name!r} entered while {self.active_role!r} is active"
                    )
                self.active_role = name
                try:
                    result = function(*args, **kwargs)
                    if self.iket_enabled and self.region_open:
                        raise RuntimeError(
                            f"Pipeline role {name!r} exited with an open region; "
                            "add PIPELINE.iteration_end() after the final phase"
                        )
                    return result
                finally:
                    self.active_role = None
                    self.region_open = False

            return wrapped

        return decorate

    def resource(
        self,
        name: str,
        *,
        label: str,
        depth: int,
        storage: str,
        description: str,
        kind: str = "buffer",
    ) -> None:
        """Declare a logical resource; statically consumed and runtime-inert."""

    def region(
        self,
        name: str,
        *,
        label: str,
        weight: int,
        description: str,
        consumes: Iterable[str] = (),
        produces: Iterable[str] = (),
        releases: Iterable[str] = (),
    ) -> None:
        """Start a named IKET range when profiling is enabled."""
        if not self.iket_enabled:
            return
        if self.active_role is None:
            raise RuntimeError(f"Pipeline region {name!r} executed outside an annotated role")
        iket = load_iket()
        if self.region_open:
            iket.range_pop()
        iket.range_push(name)
        self.region_open = True

    def iteration_end(self) -> None:
        """Close the final region in one traced loop iteration."""
        if not self.iket_enabled:
            return
        if not self.region_open:
            raise RuntimeError("PIPELINE.iteration_end() has no active region")
        load_iket().range_pop()
        self.region_open = False


@dataclass(frozen=True)
class ParsedRegion:
    """One inline region marker plus its token declarations."""

    region: Region
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    releases: tuple[str, ...]


def extract_plan(path: str | Path) -> PipelinePlan:
    """Extract a validated pipeline plan from inline source annotations."""
    return _AnnotatedPlanExtractor(path).extract()


class _AnnotatedPlanExtractor:
    """Lower literal ``PipelineAnnotations`` calls from Python AST to a plan."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.text = self.path.read_text()
        self.tree = ast.parse(self.text, filename=str(self.path))
        self.pipeline_name, self.iteration_name, self.annotation_name = self.find_pipeline()

    def extract(self) -> PipelinePlan:
        """Build a validated plan and derive token and ring-reuse dependencies."""
        plan = PipelinePlan(self.pipeline_name, iteration_name=self.iteration_name)
        for resource in self.find_resources():
            plan.add_resource(resource)

        parsed_roles = self.find_roles()
        for _, role in parsed_roles:
            plan.add_role(role)

        parsed_regions: list[ParsedRegion] = []
        for function, role in parsed_roles:
            role_regions = self.find_regions(function, role)
            parsed_regions.extend(role_regions)
            for parsed_region in role_regions:
                plan.add_region(parsed_region.region)

        producers: dict[str, tuple[str, int]] = {}
        releases: dict[str, str] = {}
        for parsed in parsed_regions:
            for token in parsed.produces:
                resource, offset = parse_token(token)
                if resource in producers:
                    raise ValueError(
                        f"Resource {resource!r} has multiple producers: "
                        f"{producers[resource][0]!r} and {parsed.region.name!r}"
                    )
                producers[resource] = parsed.region.name, offset
            for resource in parsed.releases:
                if resource in releases:
                    raise ValueError(
                        f"Resource {resource!r} has multiple release regions: "
                        f"{releases[resource]!r} and {parsed.region.name!r}"
                    )
                releases[resource] = parsed.region.name

        for parsed in parsed_regions:
            for token in parsed.consumes:
                resource_name, consumer_offset = parse_token(token)
                if resource_name not in producers:
                    raise ValueError(
                        f"Region {parsed.region.name!r} consumes {resource_name!r} "
                        "without an annotated producer"
                    )
                producer_name, producer_offset = producers[resource_name]
                distance = producer_offset - consumer_offset
                if distance < 0:
                    raise ValueError(
                        f"Dependency {producer_name!r} -> {parsed.region.name!r} "
                        f"requires negative distance {distance}"
                    )
                resource = plan.resources[resource_name]
                plan.add_dependency(
                    Dependency(
                        producer_name,
                        parsed.region.name,
                        resource_name,
                        distance=distance,
                        kind="state" if resource.kind == "state" else "data",
                        label=(
                            f"{resource.label}[c+{producer_offset}]"
                            if distance
                            else resource.label
                        ),
                    )
                )

        for resource_name, release_region in releases.items():
            if resource_name not in producers:
                raise ValueError(f"Released resource {resource_name!r} has no producer")
            resource = plan.resources[resource_name]
            if resource.kind == "state":
                raise ValueError(f"State resource {resource_name!r} cannot use ring release")
            producer_region, _ = producers[resource_name]
            plan.add_dependency(
                Dependency(
                    release_region,
                    producer_region,
                    resource_name,
                    distance=resource.depth,
                    kind="reuse",
                    label=f"reuse {resource.label}",
                )
            )
        return plan

    def find_pipeline(self) -> tuple[str, str, str]:
        """Find the unique module-level ``PipelineAnnotations`` assignment."""
        matches: list[tuple[str, ast.Call]] = []
        for node in self.tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name) or not isinstance(node.value, ast.Call):
                continue
            if call_name(node.value.func) == "PipelineAnnotations":
                matches.append((node.targets[0].id, node.value))
        if len(matches) != 1:
            raise ValueError(f"Expected one PipelineAnnotations assignment, found {len(matches)}")
        annotation_name, call = matches[0]
        name = positional_literal(call, 0, str)
        iteration_name = keyword_literal(call, "iteration_name", str)
        return name, iteration_name, annotation_name

    def find_resources(self) -> list[Resource]:
        """Read module-level resource declarations in source order."""
        resources: list[Resource] = []
        for node in self.tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not is_annotation_call(call, self.annotation_name, "resource"):
                continue
            resources.append(
                Resource(
                    name=positional_literal(call, 0, str),
                    label=keyword_literal(call, "label", str),
                    depth=keyword_literal(call, "depth", int),
                    storage=keyword_literal(call, "storage", str),
                    description=keyword_literal(call, "description", str),
                    kind=keyword_literal(call, "kind", str, default="buffer"),
                )
            )
        if not resources:
            raise ValueError("Annotated pipeline declares no resources")
        return resources

    def find_roles(
        self,
    ) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Role]]:
        """Read role decorators from nested functions in source order."""
        roles: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Role]] = []
        functions = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for function in sorted(functions, key=lambda node: node.lineno):
            decorators = [
                decorator
                for decorator in function.decorator_list
                if isinstance(decorator, ast.Call)
                and is_annotation_call(decorator, self.annotation_name, "role")
            ]
            if not decorators:
                continue
            if len(decorators) != 1:
                raise ValueError(f"Function {function.name!r} has multiple role annotations")
            call = decorators[0]
            roles.append(
                (
                    function,
                    Role(
                        name=positional_literal(call, 0, str),
                        label=keyword_literal(call, "label", str),
                        warp_start=keyword_literal(call, "warp_start", int),
                        warp_end=keyword_literal(call, "warp_end", int),
                        color=keyword_literal(call, "color", str),
                        source=SourceLocation(
                            str(self.path), function.lineno, function.end_lineno
                        ),
                    ),
                )
            )
        if not roles:
            raise ValueError("Annotated pipeline declares no roles")
        return roles

    def find_regions(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        role: Role,
    ) -> list[ParsedRegion]:
        """Read region marker calls within one role function."""
        calls = [
            node
            for node in descendants_without_nested_functions(function)
            if isinstance(node, ast.Call)
            and is_annotation_call(node, self.annotation_name, "region")
        ]
        calls.sort(key=lambda node: node.lineno)
        regions: list[ParsedRegion] = []
        for order, call in enumerate(calls):
            end_line = (
                calls[order + 1].lineno - 1 if order + 1 < len(calls) else function.end_lineno
            )
            regions.append(
                ParsedRegion(
                    region=Region(
                        name=positional_literal(call, 0, str),
                        role=role.name,
                        label=keyword_literal(call, "label", str),
                        order=order,
                        weight=keyword_literal(call, "weight", int),
                        description=keyword_literal(call, "description", str),
                        source=SourceLocation(str(self.path), call.lineno, end_line),
                    ),
                    consumes=keyword_string_tuple(call, "consumes"),
                    produces=keyword_string_tuple(call, "produces"),
                    releases=keyword_string_tuple(call, "releases"),
                )
            )
        if not regions:
            raise ValueError(f"Role function {function.name!r} declares no regions")
        return regions


def descendants_without_nested_functions(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    """Return descendants of ``function`` without entering nested callables."""
    descendants: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            ):
                continue
            descendants.append(child)
            visit(child)

    visit(function)
    return descendants


def parse_token(token: str) -> tuple[str, int]:
    """Parse ``resource`` or ``resource@offset`` token syntax."""
    resource, separator, raw_offset = token.partition("@")
    if not resource:
        raise ValueError(f"Invalid empty resource token {token!r}")
    offset = int(raw_offset) if separator else 0
    return resource, offset


def is_annotation_call(call: ast.Call, object_name: str, method: str) -> bool:
    """Return whether ``call`` targets ``object_name.method``."""
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == object_name
        and call.func.attr == method
    )


def call_name(node: ast.expr) -> str | None:
    """Return the terminal name of a call target."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def positional_literal(call: ast.Call, index: int, expected_type: type):
    """Read and type-check one literal positional argument."""
    if index >= len(call.args):
        raise ValueError(f"Missing positional argument {index} at line {call.lineno}")
    value = ast.literal_eval(call.args[index])
    if type(value) is not expected_type:
        raise ValueError(
            f"Expected {expected_type.__name__} at line {call.lineno}, got {type(value).__name__}"
        )
    return value


def keyword_literal(call: ast.Call, name: str, expected_type: type, *, default=None):
    """Read and type-check one literal keyword argument."""
    keyword = next((item for item in call.keywords if item.arg == name), None)
    if keyword is None:
        if default is not None:
            return default
        raise ValueError(f"Missing keyword {name!r} at line {call.lineno}")
    value = ast.literal_eval(keyword.value)
    if type(value) is not expected_type:
        raise ValueError(
            f"Expected {expected_type.__name__} for {name!r} at line {call.lineno}, "
            f"got {type(value).__name__}"
        )
    return value


def keyword_string_tuple(call: ast.Call, name: str) -> tuple[str, ...]:
    """Read an optional tuple/list of string tokens."""
    keyword = next((item for item in call.keywords if item.arg == name), None)
    if keyword is None:
        return ()
    value = ast.literal_eval(keyword.value)
    if not isinstance(value, (tuple, list)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name!r} at line {call.lineno} must be a tuple/list of strings")
    return tuple(value)
