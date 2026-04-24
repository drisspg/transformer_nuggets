from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import Path
import pprint
import re
from textwrap import indent

from transformer_nuggets.export_autograd_triton.specs import CapturedSpecialization


def generate_autograd_source(
    fn: Callable[..., object],
    exported_name: str,
    specializations: list[CapturedSpecialization],
) -> str:
    signature = inspect.signature(fn)
    wrapper = _generate_public_wrapper(exported_name, signature)
    specs_literal = pprint.pformat(
        [_spec_to_metadata(index, spec) for index, spec in enumerate(specializations)],
        width=100,
        sort_dicts=False,
    )
    return f"""from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_ARTIFACTS_DIR = Path(__file__).with_name(f"{{Path(__file__).stem}}_artifacts")

_SPECS = {specs_literal}


def _load_runner(module_filename):
    if module_filename is None:
        return None
    module_path = _ARTIFACTS_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load compiled source module {{module_path}}")
    spec.loader.exec_module(module)
    return module.call


_FORWARD_RUNNERS = [_load_runner(spec["forward_module"]) for spec in _SPECS]
_BACKWARD_RUNNERS = [_load_runner(spec["backward_module"]) for spec in _SPECS]


class _CompiledAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spec_id, *runtime_tensors):
        spec = _SPECS[spec_id]
        outputs = _FORWARD_RUNNERS[spec_id](list(runtime_tensors))
        if not isinstance(outputs, tuple):
            outputs = tuple(outputs)
        user_outputs = outputs[: spec["num_user_outputs"]]
        ctx.spec_id = spec_id
        ctx.runtime_tensor_count = len(runtime_tensors)
        ctx.save_for_backward(*outputs[spec["num_user_outputs"] :])
        if spec["output_kind"] == "single":
            return user_outputs[0]
        return tuple(user_outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_outputs = tuple(
            grad_output.contiguous() if isinstance(grad_output, torch.Tensor) else grad_output
            for grad_output in grad_outputs
        )
        backward_runner = _BACKWARD_RUNNERS[ctx.spec_id]
        if backward_runner is None:
            raise RuntimeError("Selected export_autograd_triton specialization has no backward graph")
        grads = backward_runner(list(ctx.saved_tensors + grad_outputs))
        if not isinstance(grads, tuple):
            grads = tuple(grads)
        if len(grads) < ctx.runtime_tensor_count:
            grads = grads + (None,) * (ctx.runtime_tensor_count - len(grads))
        if len(grads) > ctx.runtime_tensor_count:
            grads = grads[: ctx.runtime_tensor_count]
        return (None, *grads)


def _run_with_bound_args(bound_args):
    spec_id = _select_spec(bound_args)
    spec = _SPECS[spec_id]
    runtime_tensors = tuple(bound_args[name] for name in spec["runtime_tensor_names"])
    if spec["needs_autograd"]:
        result = _CompiledAutogradFunction.apply(spec_id, *runtime_tensors)
        if spec["output_kind"] == "list":
            return list(result)
        return result
    return _run_forward_only(spec_id, runtime_tensors)


def _run_forward_only(spec_id, runtime_tensors):
    spec = _SPECS[spec_id]
    outputs = _FORWARD_RUNNERS[spec_id](list(runtime_tensors))
    if not isinstance(outputs, tuple):
        outputs = tuple(outputs)
    user_outputs = outputs[: spec["num_user_outputs"]]
    if spec["output_kind"] == "single":
        return user_outputs[0]
    if spec["output_kind"] == "list":
        return list(user_outputs)
    return tuple(user_outputs)


def _select_spec(bound_args):
    matches = []
    failures = []
    for spec_id, spec in enumerate(_SPECS):
        reasons = _mismatch_reasons(spec, bound_args)
        if reasons:
            failures.append((spec, reasons))
        else:
            matches.append(spec_id)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(_SPECS[spec_id]["name"] for spec_id in matches)
        raise RuntimeError(f"Ambiguous export_autograd_triton specializations matched: {{names}}")
    raise RuntimeError(_format_no_match(bound_args, failures))


def _mismatch_reasons(spec, bound_args):
    reasons = []
    symbols = {{}}
    for name, expected in spec["static_args"].items():
        actual = bound_args.get(name)
        if actual != expected:
            reasons.append(f"static {{name}}={{actual!r}} != {{expected!r}}")
    for guard in spec["tensor_guards"]:
        tensor = bound_args.get(guard["name"])
        reason = _tensor_mismatch_reason(tensor, guard, symbols)
        if reason is not None:
            reasons.append(reason)
    return reasons


def _tensor_mismatch_reason(tensor, guard, symbols):
    name = guard["name"]
    if not isinstance(tensor, torch.Tensor):
        return f"{{name}} is {{type(tensor).__name__}}, expected Tensor"
    expected_dtype = getattr(torch, guard["dtype"])
    if tensor.dtype is not expected_dtype:
        return f"{{name}} dtype {{tensor.dtype}} != torch.{{guard['dtype']}}"
    if tensor.device.type != guard["device_type"]:
        return f"{{name}} device type {{tensor.device.type}} != {{guard['device_type']}}"
    if tensor.device.index != guard["device_index"]:
        return f"{{name}} device index {{tensor.device.index}} != {{guard['device_index']}}"
    if tensor.dim() != guard["rank"]:
        return f"{{name}} rank {{tensor.dim()}} != {{guard['rank']}}"
    shape_reason = _shape_mismatch_reason(name, tensor, guard, symbols)
    if shape_reason is not None:
        return shape_reason
    if tuple(tensor.stride()) != tuple(guard["stride"]):
        return f"{{name}} stride {{tuple(tensor.stride())}} != {{tuple(guard['stride'])}}"
    return None


def _shape_mismatch_reason(name, tensor, guard, symbols):
    for dim, expected in enumerate(guard["shape"]):
        actual = int(tensor.shape[dim])
        if isinstance(expected, dict):
            symbol = expected["symbol"]
            if expected["min"] is not None and actual < expected["min"]:
                return f"{{name}} shape dim {{dim}}={{actual}} is less than {{symbol}} min {{expected['min']}}"
            if expected["max"] is not None and actual > expected["max"]:
                return f"{{name}} shape dim {{dim}}={{actual}} is greater than {{symbol}} max {{expected['max']}}"
            if symbol in symbols and symbols[symbol] != actual:
                return f"{{name}} shape dim {{dim}}={{actual}} does not equal {{symbol}}={{symbols[symbol]}}"
            symbols[symbol] = actual
        elif actual != expected:
            return f"{{name}} shape dim {{dim}}={{actual}} != {{expected}}"
    return None


def _format_no_match(bound_args, failures):
    lines = ["No export_autograd_triton specialization matched the runtime inputs.", "Received:"]
    for name, value in bound_args.items():
        if isinstance(value, torch.Tensor):
            lines.append(
                f"  {{name}}: dtype={{value.dtype}}, device={{value.device}}, "
                f"shape={{tuple(value.shape)}}, stride={{tuple(value.stride())}}"
            )
        else:
            lines.append(f"  {{name}}: {{value!r}}")
    lines.append("Expected specializations:")
    for spec, reasons in failures:
        lines.append(f"  - {{spec['name']}}:")
        for reason in reasons:
            lines.append(f"      {{reason}}")
    return "\\n".join(lines)


{wrapper}
"""


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
        _write_compiled_module(forward_path, spec.forward_source, source_backend)
        if spec.backward_source is not None:
            _write_compiled_module(
                artifact_dir / _module_filename(index, spec, "backward"),
                spec.backward_source,
                source_backend,
            )


def _write_compiled_module(path: Path, source: str, source_backend: str) -> None:
    path.write_text(source)
    if source_backend == "clean_triton" and "async_compile.triton(" in source:
        _rewrite_module_as_clean_triton(path)


def _rewrite_module_as_clean_triton(path: Path) -> None:
    from torch.utils._get_clean_triton import get_clean_triton

    get_clean_triton(path.resolve(), path.resolve(), auto_generate_params=True)
    _patch_dynamic_pointwise_grids(path)
    launch_params_path = Path(f"{path}.launch_params")
    if launch_params_path.exists():
        launch_params_path.unlink()


def _patch_dynamic_pointwise_grids(path: Path) -> None:
    source = path.read_text()
    for xnumel_name in sorted(set(re.findall(r"(\w+_xnumel)\s*=", source))):
        kernel_name = xnumel_name.removesuffix("_xnumel")
        source = re.sub(
            rf"{kernel_name}\[\(\d+, \d+, \d+\)\]\((?P<args>[^\n]*?), \d+, XBLOCK=(?P<xblock>\d+),",
            lambda match: _dynamic_pointwise_launch_replacement(
                kernel_name,
                xnumel_name,
                match,
            ),
            source,
        )
    path.write_text(source)


def _dynamic_pointwise_launch_replacement(
    kernel_name: str,
    xnumel_name: str,
    match: re.Match[str],
) -> str:
    xblock = match.group("xblock")
    return (
        f"{kernel_name}[(triton.cdiv({xnumel_name}, {xblock}), 1, 1)]("
        f"{match.group('args')}, {xnumel_name}, XBLOCK={xblock},"
    )


def _generate_public_wrapper(exported_name: str, signature: inspect.Signature) -> str:
    return f"def {exported_name}{signature}:\n" + indent(
        "return _run_with_bound_args(locals())\n", "    "
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
        "forward_module": _module_filename(index, spec, "forward"),
        "backward_module": (
            _module_filename(index, spec, "backward") if spec.backward_source is not None else None
        ),
    }


def _module_filename(index: int, spec: CapturedSpecialization, direction: str) -> str:
    return f"spec_{index}_{_safe_name(spec.name)}_{direction}.py"


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    return safe or "unnamed"
