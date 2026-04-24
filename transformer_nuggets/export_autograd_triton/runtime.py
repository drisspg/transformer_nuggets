from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch


class ExportedAutogradRuntime:
    def __init__(self, artifacts_dir: Path, specs: list[dict[str, Any]]) -> None:
        self.artifacts_dir = artifacts_dir
        self.specs = specs
        self.forward_runners = [self._load_runner(spec["forward_module"]) for spec in specs]
        self.backward_runners = [self._load_runner(spec["backward_module"]) for spec in specs]

    def run_with_bound_args(self, bound_args: dict[str, Any]) -> Any:
        spec_id = self.select_spec(bound_args)
        runtime_tensors = self.runtime_tensors(spec_id, bound_args)
        if self.needs_autograd(spec_id):
            return self.restore_output_container(
                spec_id,
                _CompiledAutogradFunction.apply(self, spec_id, *runtime_tensors),
            )
        return self.run_forward_only(spec_id, runtime_tensors)

    def select_spec(self, bound_args: dict[str, Any]) -> int:
        return self._select_spec(bound_args)

    def runtime_tensors(
        self, spec_id: int, bound_args: dict[str, Any]
    ) -> tuple[torch.Tensor, ...]:
        return tuple(bound_args[name] for name in self.specs[spec_id]["runtime_tensor_names"])

    def needs_autograd(self, spec_id: int) -> bool:
        return bool(self.specs[spec_id]["needs_autograd"])

    def restore_output_container(self, spec_id: int, result: Any) -> Any:
        if self.specs[spec_id]["output_kind"] == "list":
            return list(result)
        return result

    def _load_runner(self, module_filename: str | None) -> Any:
        if module_filename is None:
            return None
        module_path = self.artifacts_dir / module_filename
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError(f"Could not load compiled source module {module_path}")
        spec.loader.exec_module(module)
        return module.call

    def run_forward_only(self, spec_id: int, runtime_tensors: tuple[torch.Tensor, ...]) -> Any:
        spec = self.specs[spec_id]
        outputs = self.forward_runners[spec_id](list(runtime_tensors))
        if not isinstance(outputs, tuple):
            outputs = tuple(outputs)
        user_outputs = outputs[: spec["num_user_outputs"]]
        if spec["output_kind"] == "single":
            return user_outputs[0]
        if spec["output_kind"] == "list":
            return list(user_outputs)
        return tuple(user_outputs)

    def autograd_forward(
        self,
        ctx: Any,
        spec_id: int,
        runtime_tensors: tuple[torch.Tensor, ...],
    ) -> Any:
        spec = self.specs[spec_id]
        outputs = self.forward_runners[spec_id](list(runtime_tensors))
        if not isinstance(outputs, tuple):
            outputs = tuple(outputs)
        user_outputs = outputs[: spec["num_user_outputs"]]
        residuals = outputs[spec["num_user_outputs"] :]
        named_residuals = tuple(zip(spec["forward_residual_names"], residuals, strict=True))
        ctx.runtime = self
        ctx.spec_id = spec_id
        ctx.runtime_tensor_count = len(runtime_tensors)
        ctx.saved_tensor_residual_names = tuple(
            name for name, residual in named_residuals if isinstance(residual, torch.Tensor)
        )
        ctx.saved_non_tensor_residuals = {
            name: residual
            for name, residual in named_residuals
            if not isinstance(residual, torch.Tensor)
        }
        ctx.save_for_backward(
            *(residual for _, residual in named_residuals if isinstance(residual, torch.Tensor))
        )
        non_differentiable_outputs = tuple(
            output
            for output, is_differentiable in zip(
                user_outputs,
                spec["differentiable_output_mask"],
                strict=True,
            )
            if isinstance(output, torch.Tensor) and not is_differentiable
        )
        if non_differentiable_outputs:
            ctx.mark_non_differentiable(*non_differentiable_outputs)
        if spec["output_kind"] == "single":
            return user_outputs[0]
        return tuple(user_outputs)

    def autograd_backward(self, ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        spec = self.specs[ctx.spec_id]
        grad_outputs = tuple(
            grad_output.contiguous() if isinstance(grad_output, torch.Tensor) else grad_output
            for index, grad_output in enumerate(grad_outputs)
            if spec["differentiable_output_mask"][index]
        )
        backward_runner = self.backward_runners[ctx.spec_id]
        if backward_runner is None:
            raise RuntimeError(
                "Selected export_autograd_triton specialization has no backward graph"
            )
        saved_residuals = dict(ctx.saved_non_tensor_residuals)
        saved_residuals.update(
            zip(ctx.saved_tensor_residual_names, ctx.saved_tensors, strict=True)
        )
        backward_saved_inputs = tuple(
            saved_residuals[name] for name in spec["backward_saved_input_names"]
        )
        grads = backward_runner(list(backward_saved_inputs + grad_outputs))
        if not isinstance(grads, tuple):
            grads = tuple(grads)
        if len(grads) < ctx.runtime_tensor_count:
            grads = grads + (None,) * (ctx.runtime_tensor_count - len(grads))
        if len(grads) > ctx.runtime_tensor_count:
            grads = grads[: ctx.runtime_tensor_count]
        return (None, *grads)

    def _select_spec(self, bound_args: dict[str, Any]) -> int:
        matches = []
        failures = []
        for spec_id, spec in enumerate(self.specs):
            reasons = _mismatch_reasons(spec, bound_args)
            if reasons:
                failures.append((spec, reasons))
            else:
                matches.append(spec_id)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(self.specs[spec_id]["name"] for spec_id in matches)
            raise RuntimeError(
                f"Ambiguous export_autograd_triton specializations matched: {names}"
            )
        raise RuntimeError(_format_no_match(bound_args, failures))


class _CompiledAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime: ExportedAutogradRuntime, spec_id: int, *runtime_tensors):
        spec = runtime.specs[spec_id]
        outputs = runtime.forward_runners[spec_id](list(runtime_tensors))
        if not isinstance(outputs, tuple):
            outputs = tuple(outputs)
        user_outputs = outputs[: spec["num_user_outputs"]]
        residuals = outputs[spec["num_user_outputs"] :]
        named_residuals = tuple(zip(spec["forward_residual_names"], residuals, strict=True))
        ctx.runtime = runtime
        ctx.spec_id = spec_id
        ctx.runtime_tensor_count = len(runtime_tensors)
        ctx.saved_tensor_residual_names = tuple(
            name for name, residual in named_residuals if isinstance(residual, torch.Tensor)
        )
        ctx.saved_non_tensor_residuals = {
            name: residual
            for name, residual in named_residuals
            if not isinstance(residual, torch.Tensor)
        }
        ctx.save_for_backward(
            *(residual for _, residual in named_residuals if isinstance(residual, torch.Tensor))
        )
        non_differentiable_outputs = tuple(
            output
            for output, is_differentiable in zip(
                user_outputs,
                spec["differentiable_output_mask"],
                strict=True,
            )
            if isinstance(output, torch.Tensor) and not is_differentiable
        )
        if non_differentiable_outputs:
            ctx.mark_non_differentiable(*non_differentiable_outputs)
        if spec["output_kind"] == "single":
            return user_outputs[0]
        return tuple(user_outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        runtime = ctx.runtime
        spec = runtime.specs[ctx.spec_id]
        grad_outputs = tuple(
            grad_output.contiguous() if isinstance(grad_output, torch.Tensor) else grad_output
            for index, grad_output in enumerate(grad_outputs)
            if spec["differentiable_output_mask"][index]
        )
        backward_runner = runtime.backward_runners[ctx.spec_id]
        if backward_runner is None:
            raise RuntimeError(
                "Selected export_autograd_triton specialization has no backward graph"
            )
        saved_residuals = dict(ctx.saved_non_tensor_residuals)
        saved_residuals.update(
            zip(ctx.saved_tensor_residual_names, ctx.saved_tensors, strict=True)
        )
        backward_saved_inputs = tuple(
            saved_residuals[name] for name in spec["backward_saved_input_names"]
        )
        grads = backward_runner(list(backward_saved_inputs + grad_outputs))
        if not isinstance(grads, tuple):
            grads = tuple(grads)
        if len(grads) < ctx.runtime_tensor_count:
            grads = grads + (None,) * (ctx.runtime_tensor_count - len(grads))
        if len(grads) > ctx.runtime_tensor_count:
            grads = grads[: ctx.runtime_tensor_count]
        return (None, None, *grads)


def _mismatch_reasons(spec: dict[str, Any], bound_args: dict[str, Any]) -> list[str]:
    reasons = []
    symbols = {}
    for name, expected in spec["static_args"].items():
        actual = bound_args.get(name)
        if actual != expected:
            reasons.append(f"static {name}={actual!r} != {expected!r}")
    for guard in spec["tensor_guards"]:
        tensor = bound_args.get(guard["name"])
        reason = _tensor_mismatch_reason(tensor, guard, symbols)
        if reason is not None:
            reasons.append(reason)
    return reasons


def _tensor_mismatch_reason(
    tensor: Any,
    guard: dict[str, Any],
    symbols: dict[str, int],
) -> str | None:
    name = guard["name"]
    if not isinstance(tensor, torch.Tensor):
        return f"{name} is {type(tensor).__name__}, expected Tensor"
    expected_dtype = getattr(torch, guard["dtype"])
    if tensor.dtype is not expected_dtype:
        return f"{name} dtype {tensor.dtype} != torch.{guard['dtype']}"
    if tensor.device.type != guard["device_type"]:
        return f"{name} device type {tensor.device.type} != {guard['device_type']}"
    if tensor.device.index != guard["device_index"]:
        return f"{name} device index {tensor.device.index} != {guard['device_index']}"
    if tensor.dim() != guard["rank"]:
        return f"{name} rank {tensor.dim()} != {guard['rank']}"
    shape_reason = _shape_mismatch_reason(name, tensor, guard, symbols)
    if shape_reason is not None:
        return shape_reason
    if tuple(tensor.stride()) != tuple(guard["stride"]):
        return f"{name} stride {tuple(tensor.stride())} != {tuple(guard['stride'])}"
    return None


def _shape_mismatch_reason(
    name: str,
    tensor: torch.Tensor,
    guard: dict[str, Any],
    symbols: dict[str, int],
) -> str | None:
    for dim, expected in enumerate(guard["shape"]):
        actual = int(tensor.shape[dim])
        if isinstance(expected, dict):
            symbol = expected["symbol"]
            if expected["min"] is not None and actual < expected["min"]:
                return (
                    f"{name} shape dim {dim}={actual} is less than {symbol} min {expected['min']}"
                )
            if expected["max"] is not None and actual > expected["max"]:
                return f"{name} shape dim {dim}={actual} is greater than {symbol} max {expected['max']}"
            if symbol in symbols and symbols[symbol] != actual:
                return f"{name} shape dim {dim}={actual} does not equal {symbol}={symbols[symbol]}"
            symbols[symbol] = actual
        elif actual != expected:
            return f"{name} shape dim {dim}={actual} != {expected}"
    return None


def _format_no_match(
    bound_args: dict[str, Any], failures: list[tuple[dict[str, Any], list[str]]]
) -> str:
    lines = ["No export_autograd_triton specialization matched the runtime inputs.", "Received:"]
    for name, value in bound_args.items():
        if isinstance(value, torch.Tensor):
            lines.append(
                f"  {name}: dtype={value.dtype}, device={value.device}, "
                f"shape={tuple(value.shape)}, stride={tuple(value.stride())}"
            )
        else:
            lines.append(f"  {name}: {value!r}")
    lines.append("Expected specializations:")
    for spec, reasons in failures:
        lines.append(f"  - {spec['name']}:")
        for reason in reasons:
            lines.append(f"      {reason}")
    return "\n".join(lines)
