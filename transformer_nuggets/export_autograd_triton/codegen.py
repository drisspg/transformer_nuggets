from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import Path
import pprint
import re
from textwrap import indent

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from transformer_nuggets.export_autograd_triton.clean_triton import clean_triton_module
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
) -> str:
    signature = inspect.signature(fn)
    return _TEMPLATE_ENV.get_template("autograd_module.py.j2").render(
        specs_literal=pprint.pformat(
            [_spec_to_metadata(index, spec) for index, spec in enumerate(specializations)],
            width=100,
            sort_dicts=False,
        ),
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
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(specializations):
        forward_path = artifact_dir / _module_filename(index, spec, "forward")
        _write_compiled_module(
            forward_path,
            spec.forward_source,
            source_backend,
            dynamic=spec.dynamic,
        )
        if spec.backward_source is not None:
            _write_compiled_module(
                artifact_dir / _module_filename(index, spec, "backward"),
                spec.backward_source,
                source_backend,
                dynamic=spec.dynamic,
            )


def _write_compiled_module(
    path: Path,
    source: str,
    source_backend: str,
    *,
    dynamic: bool,
) -> None:
    path.write_text(source)
    if source_backend == "clean_triton" and "async_compile.triton(" in source:
        clean_triton_module(path, dynamic=dynamic)


def _generate_public_wrapper(exported_name: str, signature: inspect.Signature) -> str:
    return f"def {exported_name}{signature}:\n" + indent(
        "return _RUNTIME.run_with_bound_args(locals())\n", "    "
    )


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
