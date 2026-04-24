from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CleanTritonResult:
    source: str
    patched_launches: list[str]
    unsupported_launches: list[str]


@dataclass
class _Launch:
    kernel_name: str
    call: ast.Call
    line: str
    family: str


@dataclass
class _Patch:
    start: int
    end: int
    replacement: str
    kernel_name: str


class CleanTritonUnsupportedError(RuntimeError):
    pass


def clean_triton_module(path: Path, *, dynamic: bool) -> CleanTritonResult:
    from torch.utils._get_clean_triton import get_clean_triton

    original_source = path.read_text()
    get_clean_triton(path.resolve(), path.resolve(), auto_generate_params=True)
    result = clean_triton_source(
        path.read_text(),
        artifact_path=path,
        dynamic=dynamic,
        original_source=original_source,
    )
    path.write_text(result.source)
    launch_params_path = Path(f"{path}.launch_params")
    if launch_params_path.exists():
        launch_params_path.unlink()
    return result


def clean_triton_source(
    source: str,
    *,
    artifact_path: Path | None = None,
    dynamic: bool,
    original_source: str | None = None,
) -> CleanTritonResult:
    if not dynamic:
        return CleanTritonResult(source=source, patched_launches=[], unsupported_launches=[])

    raw_run_args = _raw_run_args_by_kernel(original_source or "")
    patched_source, patched_launches = _patch_dynamic_launches(source, raw_run_args)
    unsupported_launches = _find_unsupported_dynamic_launches(patched_source)
    if unsupported_launches:
        raise CleanTritonUnsupportedError(
            _format_unsupported_launch_error(artifact_path, unsupported_launches)
        )
    return CleanTritonResult(
        source=patched_source,
        patched_launches=patched_launches,
        unsupported_launches=[],
    )


def _patch_dynamic_launches(
    source: str,
    raw_run_args: dict[str, list[list[str]]],
) -> tuple[str, list[str]]:
    tree = ast.parse(source)
    line_offsets = _line_offsets(source)
    function_args = _function_args_by_name(tree)
    xnumel_names = _assigned_names_with_suffix(source, "_xnumel")
    patches: list[_Patch] = []
    launch_counts: dict[str, int] = {}
    launches = sorted(
        _iter_triton_launches(tree, source),
        key=lambda launch: _node_start(line_offsets, launch.call),
    )
    for launch in launches:
        launch_index = launch_counts.get(launch.kernel_name, 0)
        launch_counts[launch.kernel_name] = launch_index + 1
        raw_args = _raw_args_for_launch(raw_run_args, launch.kernel_name, launch_index)
        if launch.family == "pointwise":
            patch = _pointwise_launch_patch(
                source,
                line_offsets,
                function_args,
                raw_args,
                xnumel_names,
                launch,
            )
        elif launch.family == "reduction":
            patch = _reduction_launch_patch(
                source,
                line_offsets,
                function_args,
                raw_args,
                launch,
            )
        elif launch.family == "matmul/template":
            patch = _template_launch_patch(source, line_offsets, raw_args, launch)
        else:
            patch = None
        if patch is not None:
            patches.append(patch)

    if not patches:
        return source, []

    patched_source = source
    patched_launches = []
    for patch in sorted(patches, key=lambda item: item.start, reverse=True):
        patched_source = (
            patched_source[: patch.start] + patch.replacement + patched_source[patch.end :]
        )
        patched_launches.append(patch.kernel_name)
    patched_launches.reverse()
    return patched_source, patched_launches


def _pointwise_launch_patch(
    source: str,
    line_offsets: list[int],
    function_args: dict[str, list[str]],
    raw_args: list[str] | None,
    xnumel_names: set[str],
    launch: _Launch,
) -> _Patch | None:
    xnumel_name = f"{launch.kernel_name}_xnumel"
    if xnumel_name not in xnumel_names and raw_args is None:
        return None
    xnumel_arg_index = _kernel_arg_index(function_args, launch.kernel_name, "xnumel")
    if xnumel_arg_index is None:
        xnumel_arg_index = _last_integer_positional_arg_index(launch.call)
    if xnumel_arg_index is None:
        return None
    return _numel_launch_patch(
        source,
        line_offsets,
        raw_args,
        launch,
        xnumel_arg_index,
        "XBLOCK",
        fallback_numel_expression=xnumel_name,
    )


def _reduction_launch_patch(
    source: str,
    line_offsets: list[int],
    function_args: dict[str, list[str]],
    raw_args: list[str] | None,
    launch: _Launch,
) -> _Patch | None:
    xnumel_arg_index = _kernel_arg_index(function_args, launch.kernel_name, "xnumel")
    if xnumel_arg_index is None or raw_args is None:
        return None
    return _numel_launch_patch(
        source,
        line_offsets,
        raw_args,
        launch,
        xnumel_arg_index,
        "XBLOCK",
    )


def _template_launch_patch(
    source: str,
    line_offsets: list[int],
    raw_args: list[str] | None,
    launch: _Launch,
) -> _Patch | None:
    if not _has_concrete_grid(launch.call):
        return None
    if raw_args is None or len(raw_args) < len(launch.call.args) + 3:
        return None
    positional_args = _patched_positional_args(source, launch.call, raw_args)
    if any(arg is None for arg in positional_args):
        return None
    keyword_args = [_keyword_source(source, keyword) for keyword in launch.call.keywords]
    if any(keyword is None for keyword in keyword_args):
        return None

    grid_args = raw_args[len(launch.call.args) : len(launch.call.args) + 3]
    args = [*positional_args, *keyword_args]
    return _Patch(
        start=_node_start(line_offsets, launch.call),
        end=_node_end(line_offsets, launch.call),
        replacement=f"{launch.kernel_name}[({', '.join(grid_args)})]("
        f"{', '.join(arg for arg in args if arg is not None)})",
        kernel_name=launch.kernel_name,
    )


def _numel_launch_patch(
    source: str,
    line_offsets: list[int],
    raw_args: list[str] | None,
    launch: _Launch,
    xnumel_arg_index: int,
    xblock_keyword_name: str,
    fallback_numel_expression: str | None = None,
) -> _Patch | None:
    if not _has_concrete_grid(launch.call):
        return None
    xblock_keyword = _keyword_by_name(launch.call, xblock_keyword_name)
    if xblock_keyword is None:
        return None
    xblock = ast.get_source_segment(source, xblock_keyword.value)
    if xblock is None:
        return None

    if raw_args is None:
        raw_args = []
    numel_expression = _raw_arg_at(raw_args, xnumel_arg_index) or fallback_numel_expression
    if numel_expression is None:
        return None

    positional_args = _patched_positional_args(source, launch.call, raw_args)
    if xnumel_arg_index < len(positional_args):
        positional_args[xnumel_arg_index] = numel_expression
    if any(arg is None for arg in positional_args):
        return None
    keyword_args = [_keyword_source(source, keyword) for keyword in launch.call.keywords]
    if any(keyword is None for keyword in keyword_args):
        return None

    args = [*positional_args, *keyword_args]
    replacement = (
        f"{launch.kernel_name}[(triton.cdiv({numel_expression}, {xblock}), 1, 1)]("
        f"{', '.join(arg for arg in args if arg is not None)})"
    )
    return _Patch(
        start=_node_start(line_offsets, launch.call),
        end=_node_end(line_offsets, launch.call),
        replacement=replacement,
        kernel_name=launch.kernel_name,
    )


def _patched_positional_args(
    source: str,
    call: ast.Call,
    raw_args: list[str],
) -> list[str | None]:
    positional_args = []
    for index, arg in enumerate(call.args):
        clean_arg = ast.get_source_segment(source, arg)
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            positional_args.append(_raw_arg_at(raw_args, index) or clean_arg)
        else:
            positional_args.append(clean_arg)
    return positional_args


def _find_unsupported_dynamic_launches(source: str) -> list[_Launch]:
    if not _has_dynamic_shape_evidence(source):
        return []
    tree = ast.parse(source)
    return [
        launch for launch in _iter_triton_launches(tree, source) if _has_concrete_grid(launch.call)
    ]


def _iter_triton_launches(tree: ast.AST, source: str) -> list[_Launch]:
    launches = []
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kernel_name = _launch_kernel_name(node)
        if kernel_name is None:
            continue
        launches.append(
            _Launch(
                kernel_name=kernel_name,
                call=node,
                line=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                family=_classify_kernel_family(kernel_name),
            )
        )
    return launches


def _launch_kernel_name(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Subscript):
        return None
    if not isinstance(call.func.value, ast.Name):
        return None
    return call.func.value.id


def _raw_run_args_by_kernel(source: str) -> dict[str, list[list[str]]]:
    if not source:
        return {}
    tree = ast.parse(source)
    line_offsets = _line_offsets(source)
    raw_args: dict[str, list[list[str]]] = {}
    run_calls = sorted(
        _iter_raw_run_calls(tree),
        key=lambda call: _node_start(line_offsets, call),
    )
    for node in run_calls:
        if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
            continue
        args = [ast.get_source_segment(source, arg) for arg in node.args]
        if any(arg is None for arg in args):
            continue
        raw_args.setdefault(node.func.value.id, []).append(
            [arg for arg in args if arg is not None]
        )
    return raw_args


def _iter_raw_run_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
    ]


def _raw_args_for_launch(
    raw_run_args: dict[str, list[list[str]]],
    kernel_name: str,
    launch_index: int,
) -> list[str] | None:
    kernel_args = raw_run_args.get(kernel_name)
    if kernel_args is None or launch_index >= len(kernel_args):
        return None
    return kernel_args[launch_index]


def _function_args_by_name(tree: ast.AST) -> dict[str, list[str]]:
    return {
        node.name: [arg.arg for arg in node.args.args]
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def _kernel_arg_index(
    function_args: dict[str, list[str]],
    kernel_name: str,
    arg_name: str,
) -> int | None:
    args = function_args.get(kernel_name)
    if args is None or arg_name not in args:
        return None
    return args.index(arg_name)


def _raw_arg_at(raw_args: list[str], index: int) -> str | None:
    if index >= len(raw_args):
        return None
    return raw_args[index]


def _has_concrete_grid(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Subscript):
        return False
    grid = call.func.slice
    if isinstance(grid, ast.Tuple) and len(grid.elts) == 1 and isinstance(grid.elts[0], ast.Tuple):
        grid = grid.elts[0]
    if not isinstance(grid, ast.Tuple) or not grid.elts:
        return False
    return all(
        isinstance(element, ast.Constant) and isinstance(element.value, int)
        for element in grid.elts
    )


def _has_dynamic_shape_evidence(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            for name in _target_names(target):
                if name.startswith("s") and name[1:].isdigit():
                    return True
                if name.endswith(("_xnumel", "_rnumel")):
                    return True
    return False


def _assigned_names_with_suffix(source: str, suffix: str) -> set[str]:
    tree = ast.parse(source)
    return {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        for name in _target_names(target)
        if name.endswith(suffix)
    }


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _target_names(item)]
    return []


def _keyword_by_name(call: ast.Call, name: str) -> ast.keyword | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword
    return None


def _keyword_source(source: str, keyword: ast.keyword) -> str | None:
    if keyword.arg is None:
        return ast.get_source_segment(source, keyword.value)
    value = ast.get_source_segment(source, keyword.value)
    if value is None:
        return None
    return f"{keyword.arg}={value}"


def _last_integer_positional_arg_index(call: ast.Call) -> int | None:
    for index in range(len(call.args) - 1, -1, -1):
        arg = call.args[index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            return index
    return None


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_start(line_offsets: list[int], node: ast.AST) -> int:
    return line_offsets[node.lineno - 1] + node.col_offset


def _node_end(line_offsets: list[int], node: ast.AST) -> int:
    return line_offsets[node.end_lineno - 1] + node.end_col_offset


def _classify_kernel_family(kernel_name: str) -> str:
    if kernel_name.startswith("triton_poi_"):
        return "pointwise"
    if kernel_name.startswith("triton_red_"):
        return "reduction"
    if kernel_name.startswith("triton_per_") or "persistent" in kernel_name:
        return "persistent reduction"
    if kernel_name.startswith("triton_tem_") or "mm" in kernel_name or "matmul" in kernel_name:
        return "matmul/template"
    return "unknown"


def _format_unsupported_launch_error(
    artifact_path: Path | None,
    unsupported_launches: list[_Launch],
) -> str:
    path_text = str(artifact_path) if artifact_path is not None else "<in-memory source>"
    lines = [
        f"Dynamic clean_triton artifact contains unsupported baked Triton launches: {path_text}",
    ]
    for launch in unsupported_launches:
        lines.append(
            f"  - {launch.kernel_name} ({launch.family}) at line {launch.call.lineno}: "
            f"{launch.line}"
        )
    lines.append(
        'Use source_backend="inductor" for this dynamic export or add a clean_triton '
        "launch cleaner for the listed kernel family."
    )
    return "\n".join(lines)
