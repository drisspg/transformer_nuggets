from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import Path
import pprint
import re
import shutil
from textwrap import indent

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from transformer_nuggets.export_autograd_triton.clean_triton import (
    clean_triton_module,
    clean_triton_source,
)
from transformer_nuggets.export_autograd_triton.specs import CapturedSpecialization


_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).with_name("templates")),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


def generate_autograd_source(
    fn: Callable[..., object],
    exported_name: str,
    specializations: list[CapturedSpecialization],
    *,
    artifact_dir_name: str | None = None,
) -> str:
    signature = inspect.signature(fn)
    return _TEMPLATE_ENV.get_template("autograd_module.py.j2").render(
        spec_literals=[
            pprint.pformat(_spec_to_metadata(index, spec), width=100, sort_dicts=False)
            for index, spec in enumerate(specializations)
        ],
        artifact_index=_artifact_index(specializations, artifact_dir_name),
        autograd_class=_autograd_class_name(exported_name),
        wrapper=_generate_public_wrapper(exported_name, signature).rstrip(),
    )


def write_autograd_source(
    path: Path,
    source: str,
    specializations: list[CapturedSpecialization],
    source_backend: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    artifact_dir = path.with_name(f"{path.stem}_artifacts")
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(specializations):
        forward_path = artifact_dir / _module_filename(index, spec, "forward")
        _write_compiled_module(
            forward_path,
            spec.forward_source,
            source_backend,
            spec=spec,
            direction="forward",
            dynamic=spec.dynamic,
        )
        if spec.backward_source is not None:
            _write_compiled_module(
                artifact_dir / _module_filename(index, spec, "backward"),
                spec.backward_source,
                source_backend,
                spec=spec,
                direction="backward",
                dynamic=spec.dynamic,
            )


def _write_compiled_module(
    path: Path,
    source: str,
    source_backend: str,
    *,
    spec: CapturedSpecialization,
    direction: str,
    dynamic: bool,
) -> None:
    path.write_text(source)
    if source_backend == "clean_triton":
        if "async_compile.triton(" in source:
            clean_triton_module(path, dynamic=dynamic)
        else:
            path.write_text(clean_triton_source(path.read_text(), dynamic=False).source)
    path.write_text(_artifact_header(path.name, spec, direction) + path.read_text())


def _artifact_index(
    specializations: list[CapturedSpecialization],
    artifact_dir_name: str | None,
) -> list[dict[str, str | None]]:
    artifact_dir = artifact_dir_name or "<this_module>_artifacts"
    return [
        {
            "name": spec.name,
            "forward_path": f"{artifact_dir}/{_module_filename(index, spec, 'forward')}",
            "backward_path": (
                f"{artifact_dir}/{_module_filename(index, spec, 'backward')}"
                if spec.backward_source is not None
                else None
            ),
        }
        for index, spec in enumerate(specializations)
    ]


def _generate_public_wrapper(exported_name: str, signature: inspect.Signature) -> str:
    class_name = _autograd_class_name(exported_name)
    body = (
        '"""Run the exported autograd/Triton specialization matching these inputs."""\n'
        f"return _RUNTIME.run_with_bound_args(locals(), {class_name})\n"
    )
    return f"def {exported_name}{signature}:\n" + indent(body, "    ")


def _autograd_class_name(exported_name: str) -> str:
    words = [word for word in re.split(r"[^0-9a-zA-Z]+", exported_name) if word]
    return "_" + "".join(word[:1].upper() + word[1:] for word in words) + "AutogradFunction"


def _artifact_header(filename: str, spec: CapturedSpecialization, direction: str) -> str:
    lines = [
        '"""Generated Triton artifact.',
        "",
        f"File: {_docstring_text(filename)}",
        f"Specialization: {_docstring_text(spec.name)}",
        f"Direction: {_docstring_text(direction)}",
        "Runtime tensor order:",
        _format_items(spec.runtime_tensor_names),
        "Static arguments:",
        _format_static_args(spec),
        "Tensor guards:",
        _format_tensor_guards(spec),
        f"User outputs: {spec.num_user_outputs} ({_docstring_text(spec.output_kind)})",
        f"Needs autograd: {spec.needs_autograd}",
        "Differentiable output mask:",
        _format_items(spec.differentiable_output_mask),
        "Forward residual order:",
        _format_items(spec.forward_residual_names),
        "Backward saved-input order:",
        _format_items(spec.backward_saved_input_names),
        "",
        "The runtime imports this file and calls call(args).",
        '"""',
        "",
    ]
    return "\n".join(lines) + "\n"


def _format_items(values: tuple[object, ...]) -> str:
    if not values:
        return "  <none>"
    return "\n".join(f"  - {_docstring_text(value)}" for value in values)


def _format_static_args(spec: CapturedSpecialization) -> str:
    if not spec.static_args:
        return "  <none>"
    return "\n".join(
        f"  - {_docstring_text(arg.name)}={_docstring_text(repr(arg.value))}"
        for arg in spec.static_args
    )


def _format_tensor_guards(spec: CapturedSpecialization) -> str:
    if not spec.tensor_guards:
        return "  <none>"
    return "\n".join(
        f"  - {_docstring_text(guard.name)}: shape={_docstring_text(guard.shape)}, "
        f"stride={_docstring_text(guard.stride)}, dtype={_docstring_text(guard.dtype)}, "
        f"device={_docstring_text(guard.device_type)}:{guard.device_index}"
        for guard in spec.tensor_guards
    )


def _docstring_text(value: object) -> str:
    return str(value).replace("\r", "\\r").replace("\n", "\\n").replace('"""', r"\"\"\"")


def _spec_to_metadata(index: int, spec: CapturedSpecialization) -> dict[str, object]:
    return {
        "name": spec.name,
        "runtime_tensor_names": spec.runtime_tensor_names,
        "static_args": {static.name: static.value for static in spec.static_args},
        "tensor_guards": [guard.__dict__ for guard in spec.tensor_guards],
        "num_user_outputs": spec.num_user_outputs,
        "output_kind": spec.output_kind,
        "needs_autograd": spec.needs_autograd,
        "differentiable_output_mask": spec.differentiable_output_mask,
        "forward_residual_names": spec.forward_residual_names,
        "backward_saved_input_names": spec.backward_saved_input_names,
        "forward_module": _module_filename(index, spec, "forward"),
        "backward_module": (
            _module_filename(index, spec, "backward") if spec.backward_source is not None else None
        ),
    }


def _module_filename(index: int, spec: CapturedSpecialization, direction: str) -> str:
    default_name = f"spec_{index}"
    if spec.name == default_name:
        return f"{default_name}_{direction}.py"
    return f"{default_name}_{_safe_name(spec.name)}_{direction}.py"


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    return safe or "unnamed"
