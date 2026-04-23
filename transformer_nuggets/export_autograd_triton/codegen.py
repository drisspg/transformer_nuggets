from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import Path
from textwrap import indent

from transformer_nuggets.export_autograd_triton.specs import CapturedSpecialization


def generate_autograd_source(
    fn: Callable[..., object],
    exported_name: str,
    specializations: list[CapturedSpecialization],
) -> str:
    signature = inspect.signature(fn)
    wrapper = _generate_public_wrapper(exported_name, signature)
    specs_literal = repr([_spec_to_metadata(spec) for spec in specializations])
    forward_sources = repr([spec.forward_source for spec in specializations])
    backward_sources = repr([spec.backward_source for spec in specializations])
    return f"""from __future__ import annotations

import torch

_SPECS = {specs_literal}
_FORWARD_SOURCES = {forward_sources}
_BACKWARD_SOURCES = {backward_sources}


def _make_runner(source):
    namespace = {{"__file__": __file__}}
    exec(source, namespace)
    return namespace["call"]


_FORWARD_RUNNERS = [_make_runner(source) for source in _FORWARD_SOURCES]
_BACKWARD_RUNNERS = [
    _make_runner(source) if source is not None else None for source in _BACKWARD_SOURCES
]


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
    for name, expected in spec["static_args"].items():
        actual = bound_args.get(name)
        if actual != expected:
            reasons.append(f"static {{name}}={{actual!r}} != {{expected!r}}")
    for guard in spec["tensor_guards"]:
        tensor = bound_args.get(guard["name"])
        reason = _tensor_mismatch_reason(tensor, guard)
        if reason is not None:
            reasons.append(reason)
    return reasons


def _tensor_mismatch_reason(tensor, guard):
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
    if tuple(tensor.shape) != tuple(guard["shape"]):
        return f"{{name}} shape {{tuple(tensor.shape)}} != {{tuple(guard['shape'])}}"
    if tuple(tensor.stride()) != tuple(guard["stride"]):
        return f"{{name}} stride {{tuple(tensor.stride())}} != {{tuple(guard['stride'])}}"
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


def write_autograd_source(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def _generate_public_wrapper(exported_name: str, signature: inspect.Signature) -> str:
    return f"def {exported_name}{signature}:\n" + indent(
        "return _run_with_bound_args(locals())\n", "    "
    )


def _spec_to_metadata(spec: CapturedSpecialization) -> dict[str, object]:
    return {
        "name": spec.name,
        "runtime_tensor_names": spec.runtime_tensor_names,
        "static_args": {static.name: static.value for static in spec.static_args},
        "tensor_guards": [guard.__dict__ for guard in spec.tensor_guards],
        "num_user_outputs": spec.num_user_outputs,
        "output_kind": spec.output_kind,
        "needs_autograd": spec.needs_autograd,
    }
