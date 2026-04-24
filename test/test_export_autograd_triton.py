from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path

import pytest
import torch

from transformer_nuggets.export_autograd_triton import Specialization, export_autograd_triton

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")


def affine_activation(x, w, *, activation="relu"):
    y = x @ w
    if activation == "relu":
        return torch.relu(y)
    if activation == "gelu":
        return torch.nn.functional.gelu(y)
    raise RuntimeError(f"unsupported activation: {activation}")


def two_matmuls_layernorm(x, w1, w2, *, residual=True):
    hidden = torch.relu(x @ w1)
    out = hidden @ w2
    if residual:
        out = out + x
    return torch.nn.functional.layer_norm(out, (out.shape[-1],))


def tuple_outputs(x, w):
    y = x @ w
    return torch.relu(y), torch.sigmoid(y)


def integer_tensor_output(x):
    return x.argmax(dim=-1)


def trig_pointwise(x):
    return torch.sin(x) + torch.cos(x)


def list_outputs(x, w):
    y = x @ w
    return [torch.relu(y), torch.sigmoid(y)]


def trig_reduction_mix(x, y, *, scale=0.25):
    a = torch.sin(x) + torch.cos(y)
    b = torch.softmax(a * scale, dim=-1)
    return b * b.sum(dim=-1, keepdim=True)


def conv_silu_mean(x, weight, bias, *, stride=1, padding=1):
    y = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding)
    return torch.nn.functional.silu(y).mean(dim=(-1, -2))


def einsum_chain(x, y, z):
    return torch.tanh(torch.einsum("bik,bkj->bij", x, y)) @ z


def scalar_output(x):
    return torch.logsumexp(x, dim=-1).sum()


def view_output(x):
    return torch.relu(x).view(-1)


def gather_float_then_square(x, index):
    return torch.index_select(x, 0, index).square()


def noncontiguous_input(x, w):
    return torch.relu(x @ w)


def mixed_tensor_and_static_tuple(x, *, factors=(2, 3), bias=0.5):
    return torch.relu(x * factors[0] + factors[1] + bias)


def optional_none_static(x, *, bias=None):
    if bias is None:
        return x.sin()
    return x + bias


def _requires_export_runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for export_autograd_triton tests")
    pytest.importorskip("triton")
    pytest.importorskip("torch._functorch.aot_autograd")
    pytest.importorskip("torch._inductor.compile_fx")


def _import_generated(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _clone_tensor(tensor):
    clone = torch.empty_strided(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    clone.copy_(tensor.detach())
    if tensor.requires_grad:
        clone.requires_grad_()
    return clone


def _clone_inputs(x, w):
    return _clone_tensor(x), _clone_tensor(w)


def _source_kernel_marker_count(generated_path):
    artifact_dir = generated_path.with_name(f"{generated_path.stem}_artifacts")
    return sum(
        source.count("@triton.jit")
        + source.count("async_compile.triton(")
        + source.count("async_compile.cpp")
        + source.count("extern_kernels.")
        for source in (path.read_text() for path in artifact_dir.glob("*.py"))
    )


def _clone_args(args):
    return tuple(_clone_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)


def _flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def _differentiable_loss(value):
    tensors = [
        tensor
        for tensor in _flatten_tensors(value)
        if tensor.requires_grad and (tensor.is_floating_point() or tensor.is_complex())
    ]
    if not tensors:
        return None
    return sum(tensor.real.sum() if tensor.is_complex() else tensor.sum() for tensor in tensors)


def _compare_eager_and_compiled(fn, compiled_fn, args, kwargs=None, rtol=1e-4, atol=1e-4):
    kwargs = kwargs or {}
    eager_args = _clone_args(args)
    compiled_args = _clone_args(args)
    eager = fn(*eager_args, **kwargs)
    compiled = compiled_fn(*compiled_args, **kwargs)
    eager_tensors = _flatten_tensors(eager)
    compiled_tensors = _flatten_tensors(compiled)
    assert len(compiled_tensors) == len(eager_tensors)
    for compiled_tensor, eager_tensor in zip(compiled_tensors, eager_tensors, strict=True):
        torch.testing.assert_close(compiled_tensor, eager_tensor, rtol=rtol, atol=atol)

    eager_loss = _differentiable_loss(eager)
    compiled_loss = _differentiable_loss(compiled)
    if eager_loss is None or compiled_loss is None:
        assert eager_loss is compiled_loss
        return
    eager_loss.backward()
    compiled_loss.backward()
    for compiled_arg, eager_arg in zip(compiled_args, eager_args, strict=True):
        if not isinstance(eager_arg, torch.Tensor) or not eager_arg.requires_grad:
            continue
        if eager_arg.grad is None or compiled_arg.grad is None:
            assert eager_arg.grad is compiled_arg.grad
        else:
            torch.testing.assert_close(compiled_arg.grad, eager_arg.grad, rtol=rtol, atol=atol)


def test_export_static_specialization_forward_backward_signature_and_errors(tmp_path):
    _requires_export_runtime()
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    w = torch.randn(8, 3, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_affine_relu.py"

    exported = export_autograd_triton(
        affine_activation,
        [Specialization(args=(x, w), kwargs={"activation": "relu"}, name="relu")],
        generated_path,
    )
    module = _import_generated(generated_path)

    assert exported.exported_name == "affine_activation_compiled"
    assert exported.specializations == ["relu"]
    assert "triton" in exported.source
    assert str(inspect.signature(module.affine_activation_compiled)) == str(
        inspect.signature(affine_activation)
    )

    eager_x, eager_w = _clone_inputs(x, w)
    compiled_x, compiled_w = _clone_inputs(x, w)
    eager = affine_activation(eager_x, eager_w, activation="relu")
    compiled = module.affine_activation_compiled(compiled_x, compiled_w, activation="relu")

    torch.testing.assert_close(compiled, eager)
    eager.sum().backward()
    compiled.sum().backward()
    torch.testing.assert_close(compiled_x.grad, eager_x.grad)
    torch.testing.assert_close(compiled_w.grad, eager_w.grad)

    with pytest.raises(RuntimeError, match="No export_autograd_triton specialization matched"):
        module.affine_activation_compiled(
            torch.randn(5, 8, device="cuda", requires_grad=True),
            compiled_w,
            activation="relu",
        )

    with pytest.raises(RuntimeError, match="static activation='gelu' != 'relu'"):
        module.affine_activation_compiled(compiled_x, compiled_w, activation="gelu")


def test_export_multiple_static_specializations_dispatch(tmp_path):
    _requires_export_runtime()
    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    w = torch.randn(4, 5, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_affine_multi.py"

    export_autograd_triton(
        affine_activation,
        [
            Specialization(args=(x, w), kwargs={"activation": "relu"}, name="relu"),
            Specialization(args=(x, w), kwargs={"activation": "gelu"}, name="gelu"),
        ],
        generated_path,
    )
    module = _import_generated(generated_path)

    for activation in ("relu", "gelu"):
        eager_x, eager_w = _clone_inputs(x, w)
        compiled_x, compiled_w = _clone_inputs(x, w)
        eager = affine_activation(eager_x, eager_w, activation=activation)
        compiled = module.affine_activation_compiled(
            compiled_x,
            compiled_w,
            activation=activation,
        )
        torch.testing.assert_close(compiled, eager)
        eager.sum().backward()
        compiled.sum().backward()
        torch.testing.assert_close(compiled_x.grad, eager_x.grad)
        torch.testing.assert_close(compiled_w.grad, eager_w.grad)


def test_export_multi_kernel_graph(tmp_path):
    _requires_export_runtime()
    x = torch.randn(3, 8, device="cuda", requires_grad=True)
    w1 = torch.randn(8, 16, device="cuda", requires_grad=True)
    w2 = torch.randn(16, 8, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_multi_kernel.py"

    exported = export_autograd_triton(
        two_matmuls_layernorm,
        [Specialization(args=(x, w1, w2), kwargs={"residual": True}, name="residual")],
        generated_path,
    )
    module = _import_generated(generated_path)

    assert _source_kernel_marker_count(exported.output_path) > 1
    eager_x, eager_w1, eager_w2 = (_clone_tensor(x), _clone_tensor(w1), _clone_tensor(w2))
    compiled_x, compiled_w1, compiled_w2 = (_clone_tensor(x), _clone_tensor(w1), _clone_tensor(w2))
    eager = two_matmuls_layernorm(eager_x, eager_w1, eager_w2, residual=True)
    compiled = module.two_matmuls_layernorm_compiled(
        compiled_x,
        compiled_w1,
        compiled_w2,
        residual=True,
    )

    torch.testing.assert_close(compiled, eager, rtol=2e-4, atol=2e-4)
    eager.sum().backward()
    compiled.sum().backward()
    torch.testing.assert_close(compiled_x.grad, eager_x.grad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(compiled_w1.grad, eager_w1.grad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(compiled_w2.grad, eager_w2.grad, rtol=2e-4, atol=2e-4)


def test_export_tuple_outputs(tmp_path):
    _requires_export_runtime()
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    w = torch.randn(8, 3, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_tuple_outputs.py"

    export_autograd_triton(
        tuple_outputs,
        [Specialization(args=(x, w), name="tuple")],
        generated_path,
    )
    module = _import_generated(generated_path)

    eager_x, eager_w = _clone_inputs(x, w)
    compiled_x, compiled_w = _clone_inputs(x, w)
    eager = tuple_outputs(eager_x, eager_w)
    compiled = module.tuple_outputs_compiled(compiled_x, compiled_w)

    assert isinstance(compiled, tuple)
    for compiled_output, eager_output in zip(compiled, eager, strict=True):
        torch.testing.assert_close(compiled_output, eager_output)
    sum(output.sum() for output in eager).backward()
    sum(output.sum() for output in compiled).backward()
    torch.testing.assert_close(compiled_x.grad, eager_x.grad)
    torch.testing.assert_close(compiled_w.grad, eager_w.grad)


def test_export_forward_only_integer_tensor_output(tmp_path):
    _requires_export_runtime()
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_integer_output.py"

    exported = export_autograd_triton(
        integer_tensor_output,
        [Specialization(args=(x,), name="argmax")],
        generated_path,
    )
    module = _import_generated(generated_path)

    assert "None" in exported.source
    compiled = module.integer_tensor_output_compiled(x)
    torch.testing.assert_close(compiled, integer_tensor_output(x))
    assert compiled.dtype == torch.int64
    assert not compiled.requires_grad


def _exotic_case(case):
    match case:
        case "list_outputs":
            return (
                list_outputs,
                (
                    torch.randn(4, 8, device="cuda", requires_grad=True),
                    torch.randn(8, 3, device="cuda", requires_grad=True),
                ),
                {},
                1e-4,
                1e-4,
            )
        case "trig_reduction_mix":
            return (
                trig_reduction_mix,
                (
                    torch.randn(8, 64, device="cuda", requires_grad=True),
                    torch.randn(8, 64, device="cuda", requires_grad=True),
                ),
                {"scale": 0.25},
                2e-4,
                2e-4,
            )
        case "conv_silu_mean":
            return (
                conv_silu_mean,
                (
                    torch.randn(2, 3, 8, 8, device="cuda", requires_grad=True),
                    torch.randn(4, 3, 3, 3, device="cuda", requires_grad=True),
                    torch.randn(4, device="cuda", requires_grad=True),
                ),
                {"stride": 1, "padding": 1},
                2e-4,
                2e-4,
            )
        case "einsum_chain":
            return (
                einsum_chain,
                (
                    torch.randn(2, 3, 4, device="cuda", requires_grad=True),
                    torch.randn(2, 4, 5, device="cuda", requires_grad=True),
                    torch.randn(5, 6, device="cuda", requires_grad=True),
                ),
                {},
                2e-4,
                2e-4,
            )
        case "scalar_output":
            return (
                scalar_output,
                (torch.randn(4, 8, device="cuda", requires_grad=True),),
                {},
                1e-4,
                1e-4,
            )
        case "view_output":
            return (
                view_output,
                (torch.randn(4, 8, device="cuda", requires_grad=True),),
                {},
                1e-4,
                1e-4,
            )
        case "gather_float_then_square":
            return (
                gather_float_then_square,
                (
                    torch.randn(6, 5, device="cuda", requires_grad=True),
                    torch.tensor([0, 2, 4, 2], device="cuda", dtype=torch.int64),
                ),
                {},
                1e-4,
                1e-4,
            )
        case "noncontiguous_input":
            return (
                noncontiguous_input,
                (
                    torch.randn(8, 4, device="cuda", requires_grad=True).t(),
                    torch.randn(8, 4, device="cuda", requires_grad=True),
                ),
                {},
                1e-4,
                1e-4,
            )
        case "mixed_tensor_and_static_tuple":
            return (
                mixed_tensor_and_static_tuple,
                (torch.randn(4, 8, device="cuda", requires_grad=True),),
                {"factors": (2, 3), "bias": 0.5},
                1e-4,
                1e-4,
            )
        case "optional_none_static":
            return (
                optional_none_static,
                (torch.randn(4, 8, device="cuda", requires_grad=True),),
                {"bias": None},
                1e-4,
                1e-4,
            )
        case _:
            raise AssertionError(f"unknown case {case}")


@pytest.mark.parametrize(
    "case",
    [
        "list_outputs",
        "trig_reduction_mix",
        "conv_silu_mean",
        "einsum_chain",
        "scalar_output",
        "view_output",
        "gather_float_then_square",
        "noncontiguous_input",
        "mixed_tensor_and_static_tuple",
        "optional_none_static",
    ],
)
def test_export_exotic_cases_compare_eager(tmp_path, case):
    _requires_export_runtime()
    fn, args, kwargs, rtol, atol = _exotic_case(case)
    generated_path = tmp_path / f"generated_{case}.py"

    exported = export_autograd_triton(
        fn,
        [Specialization(args=args, kwargs=kwargs, name=case)],
        generated_path,
    )
    module = _import_generated(generated_path)

    _compare_eager_and_compiled(
        fn,
        getattr(module, exported.exported_name),
        args,
        kwargs,
        rtol=rtol,
        atol=atol,
    )


def test_max_autotune_config_is_threaded_to_inductor(tmp_path):
    _requires_export_runtime()
    x = torch.randn(16, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_max_autotune.py"

    export_autograd_triton(
        trig_pointwise,
        [Specialization(args=(x,), name="max_autotune")],
        generated_path,
        source_backend="inductor",
        max_autotune=True,
    )
    artifact_source = "\n".join(
        path.read_text()
        for path in generated_path.with_name("generated_max_autotune_artifacts").glob("*.py")
    )

    assert "'max_autotune': True" in artifact_source


def test_dynamic_batch_specialization_dispatches_across_batch_sizes(tmp_path):
    _requires_export_runtime()
    batch = torch.export.Dim("batch", min=1, max=16)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    w = torch.randn(8, 3, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_dynamic.py"

    export_autograd_triton(
        affine_activation,
        [
            Specialization(
                args=(x, w),
                kwargs={"activation": "relu"},
                dynamic_shapes={"x": {0: batch}},
                name="dynamic_batch_relu",
            )
        ],
        generated_path,
        source_backend="inductor",
    )
    module = _import_generated(generated_path)
    assert "s" in "\n".join(
        path.read_text()
        for path in generated_path.with_name("generated_dynamic_artifacts").glob("*.py")
    )

    for batch_size in (1, 7, 16):
        dynamic_x = torch.randn(batch_size, 8, device="cuda", requires_grad=True)
        _compare_eager_and_compiled(
            affine_activation,
            module.affine_activation_compiled,
            (dynamic_x, w),
            {"activation": "relu"},
            rtol=2e-4,
            atol=2e-4,
        )

    with pytest.raises(RuntimeError, match="greater than batch max 16"):
        module.affine_activation_compiled(
            torch.randn(17, 8, device="cuda", requires_grad=True),
            w,
            activation="relu",
        )

    with pytest.raises(RuntimeError, match="shape dim 1=9 != 8"):
        module.affine_activation_compiled(
            torch.randn(4, 9, device="cuda", requires_grad=True),
            w,
            activation="relu",
        )


def test_dynamic_shape_limitations_are_explicitly_guarded(tmp_path):
    _requires_export_runtime()
    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    w = torch.randn(4, 5, device="cuda", requires_grad=True)

    with pytest.raises(ValueError, match="source_backend='inductor'"):
        export_autograd_triton(
            affine_activation,
            [Specialization(args=(x, w), dynamic_shapes={"x": {0: "batch"}})],
            tmp_path / "generated_dynamic_clean.py",
        )

    with pytest.raises(NotImplementedError, match="dynamic dim 0"):
        export_autograd_triton(
            affine_activation,
            [Specialization(args=(x, w), dynamic_shapes={"x": {1: "hidden"}})],
            tmp_path / "generated_dynamic_inner.py",
            source_backend="inductor",
        )

    with pytest.raises(NotImplementedError, match="additional_inputs"):
        export_autograd_triton(
            affine_activation,
            [Specialization(args=(x, w), additional_inputs=[((x, w), {})])],
            tmp_path / "generated_additional.py",
        )
