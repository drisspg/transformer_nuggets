from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch

from transformer_nuggets.export_autograd_triton import (
    Specialization,
    export_autograd_triton,
    load_exported_module,
)
from transformer_nuggets.export_autograd_triton.clean_triton import (
    CleanTritonUnsupportedError,
    clean_triton_source,
)
from transformer_nuggets.export_autograd_triton.codegen import (
    generate_autograd_source,
    write_autograd_source,
)
from transformer_nuggets.export_autograd_triton.guards import tensor_guard_for
from transformer_nuggets.export_autograd_triton.specs import (
    CapturedSpecialization,
    TensorGuardSpec,
)

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


def mixed_differentiable_and_integer_outputs(x):
    return torch.sin(x), x.argmax(dim=-1)


def mixed_differentiable_and_detached_float_outputs(x):
    return torch.sin(x), x.detach() * 2


def integer_tensor_output(x):
    return x.argmax(dim=-1)


def trig_pointwise(x):
    return torch.sin(x) + torch.cos(x)


def shared_batch_pointwise(x, y):
    return torch.sin(x) + torch.cos(y)


def dict_output(x):
    return {"y": torch.sin(x)}


def sum_last_dim(x):
    return x.sum(dim=-1)


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
    return load_exported_module(path)


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


def _fake_captured_specialization(
    forward_source="def call(args):\n    return ()\n",
    name="spec_0",
):
    return CapturedSpecialization(
        name=name,
        runtime_tensor_names=("x",),
        static_args=(),
        tensor_guards=(
            TensorGuardSpec(
                name="x",
                rank=2,
                shape=(4, 8),
                stride=(8, 1),
                dtype="torch.float32",
                device_type="cuda",
                device_index=0,
            ),
        ),
        forward_source=forward_source,
        backward_source=None,
        num_user_outputs=1,
        output_kind="single",
        needs_autograd=False,
        dynamic=False,
        differentiable_output_mask=(True,),
        forward_residual_names=(),
        backward_saved_input_names=(),
    )


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
    shared_args = _clone_args(args)
    eager = fn(*shared_args, **kwargs)
    compiled = compiled_fn(*shared_args, **kwargs)
    eager_tensors = _flatten_tensors(eager)
    compiled_tensors = _flatten_tensors(compiled)
    assert len(compiled_tensors) == len(eager_tensors)
    for compiled_tensor, eager_tensor in zip(compiled_tensors, eager_tensors, strict=True):
        torch.testing.assert_close(compiled_tensor, eager_tensor, rtol=rtol, atol=atol)
        assert compiled_tensor.requires_grad == eager_tensor.requires_grad

    eager_loss = _differentiable_loss(eager)
    compiled_loss = _differentiable_loss(compiled)
    if eager_loss is None or compiled_loss is None:
        assert eager_loss is compiled_loss
        return

    differentiable_args = tuple(
        arg
        for arg in shared_args
        if isinstance(arg, torch.Tensor)
        and arg.requires_grad
        and (arg.is_floating_point() or arg.is_complex())
    )
    eager_grads = torch.autograd.grad(
        eager_loss,
        differentiable_args,
        retain_graph=True,
        allow_unused=True,
    )
    compiled_grads = torch.autograd.grad(
        compiled_loss,
        differentiable_args,
        allow_unused=True,
    )
    for compiled_grad, eager_grad in zip(compiled_grads, eager_grads, strict=True):
        if eager_grad is None or compiled_grad is None:
            assert eager_grad is compiled_grad
        else:
            torch.testing.assert_close(compiled_grad, eager_grad, rtol=rtol, atol=atol)


def test_unbounded_dynamic_dim_guard_uses_none_for_symbolic_max():
    guard = tensor_guard_for(
        "x",
        torch.empty(4, 8),
        {0: torch.export.Dim("batch")},
    )

    assert guard.shape[0] == {"symbol": "batch", "min": 0, "max": None}


def test_generated_wrapper_source_is_labeled_and_splits_specs():
    source = generate_autograd_source(
        trig_pointwise,
        "trig_pointwise_compiled",
        [_fake_captured_specialization()],
    )

    assert source.startswith('"""Generated by transformer_nuggets.export_autograd_triton.')
    assert "# Generated Triton artifact index. Paths are relative to this wrapper file." in source
    assert "#   spec_0 forward: <this_module>_artifacts/spec_0_forward.py" in source
    assert "#   spec_0 backward: <this_module>_artifacts/spec_0_backward.py" not in source
    assert (
        "# Calls the selected *_forward.py artifact listed in the artifact index above." in source
    )
    assert (
        "# Calls the selected *_backward.py artifact listed in the artifact index above." in source
    )
    assert "\n\n\n_SPEC_0" not in source
    assert "_SPEC_0 = {" in source
    assert "_SPECS = [\n    _SPEC_0," in source
    assert "class _TrigPointwiseCompiledAutogradFunction(torch.autograd.Function):" in source
    assert "return _RUNTIME.autograd_forward(ctx, spec_id, runtime_tensors)" in source
    assert "return _RUNTIME.autograd_backward(ctx, *grad_outputs)" in source
    assert (
        "return _RUNTIME.run_with_bound_args(locals(), _TrigPointwiseCompiledAutogradFunction)"
        in source
    )
    assert "Run the exported autograd/Triton specialization" in source


def test_write_autograd_source_labels_artifacts_and_removes_stale_files(tmp_path):
    generated_path = tmp_path / "generated_fake.py"
    artifact_dir = tmp_path / "generated_fake_artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "stale.py").write_text("stale")

    write_autograd_source(
        generated_path,
        "wrapper source",
        [_fake_captured_specialization()],
        source_backend="inductor",
    )

    artifact_path = artifact_dir / "spec_0_forward.py"
    artifact_source = artifact_path.read_text()
    assert not (artifact_dir / "stale.py").exists()
    assert artifact_source.startswith('"""Generated Triton artifact.')
    assert "Runtime tensor order:\n  - x" in artifact_source
    assert "Tensor guards:\n  - x: shape=(4, 8), stride=(8, 1)" in artifact_source
    assert "The runtime imports this file and calls call(args)." in artifact_source


def test_generated_wrapper_uses_output_path_in_artifact_index(tmp_path):
    generated_path = tmp_path / "generated_fake.py"

    source = generate_autograd_source(
        trig_pointwise,
        "trig_pointwise_compiled",
        [_fake_captured_specialization()],
        artifact_dir_name=f"{generated_path.stem}_artifacts",
    )

    assert "#   spec_0 forward: generated_fake_artifacts/spec_0_forward.py" in source


def test_write_clean_triton_artifact_strips_benchmark_tail_without_async_compile(tmp_path):
    generated_path = tmp_path / "generated_fake.py"
    write_autograd_source(
        generated_path,
        "wrapper source",
        [
            _fake_captured_specialization(
                forward_source="""def call(args):
    return args


def get_args():
    return []


def benchmark_compiled_module(args, times=10, repeat=10):
    return None


if __name__ == "__main__":
    print(get_args())
"""
            )
        ],
        source_backend="clean_triton",
    )

    artifact_source = (tmp_path / "generated_fake_artifacts" / "spec_0_forward.py").read_text()
    assert "def call(args):" in artifact_source
    assert "def get_args():" not in artifact_source
    assert "benchmark_compiled_module" not in artifact_source
    assert "__main__" not in artifact_source


def test_artifact_header_escapes_docstring_metadata(tmp_path):
    generated_path = tmp_path / "generated_fake.py"
    write_autograd_source(
        generated_path,
        "wrapper source",
        [_fake_captured_specialization(name='bad"""name\nline')],
        source_backend="inductor",
    )

    artifact_path = tmp_path / "generated_fake_artifacts" / "spec_0_bad_name_line_forward.py"
    artifact_source = artifact_path.read_text()
    compile(artifact_source, str(artifact_path), "exec")
    assert 'Specialization: bad\\"\\"\\"name\\nline' in artifact_source


def test_rejects_unknown_dynamic_shape_argument_name(tmp_path):
    x = torch.randn(2, 4, requires_grad=True)
    w = torch.randn(4, 5, requires_grad=True)

    with pytest.raises(ValueError, match="unknown Tensor argument names: missing"):
        export_autograd_triton(
            affine_activation,
            [Specialization(args=(x, w), dynamic_shapes={"missing": {0: "batch"}})],
            tmp_path / "generated_unknown_dynamic.py",
        )


def test_rejects_unsupported_static_container(tmp_path):
    x = torch.randn(2, 4, requires_grad=True)

    with pytest.raises(TypeError, match="Static argument 'factors' has unsupported value"):
        export_autograd_triton(
            mixed_tensor_and_static_tuple,
            [Specialization(args=(x,), kwargs={"factors": [2, 3]})],
            tmp_path / "generated_bad_static.py",
        )


def test_rejects_unsupported_output_container(tmp_path):
    x = torch.randn(2, 4, requires_grad=True)

    with pytest.raises(TypeError, match="only supports a Tensor, tuple"):
        export_autograd_triton(
            dict_output,
            [Specialization(args=(x,))],
            tmp_path / "generated_dict_output.py",
        )


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

    _compare_eager_and_compiled(
        affine_activation,
        module.affine_activation_compiled,
        (x, w),
        {"activation": "relu"},
    )

    with pytest.raises(RuntimeError, match="No export_autograd_triton specialization matched"):
        module.affine_activation_compiled(
            torch.randn(5, 8, device="cuda", requires_grad=True),
            w,
            activation="relu",
        )

    with pytest.raises(RuntimeError, match="static activation='gelu' != 'relu'"):
        module.affine_activation_compiled(x, w, activation="gelu")


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
        _compare_eager_and_compiled(
            affine_activation,
            module.affine_activation_compiled,
            (x, w),
            {"activation": activation},
        )


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
    _compare_eager_and_compiled(
        two_matmuls_layernorm,
        module.two_matmuls_layernorm_compiled,
        (x, w1, w2),
        {"residual": True},
        rtol=2e-4,
        atol=2e-4,
    )


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

    assert isinstance(module.tuple_outputs_compiled(x, w), tuple)
    _compare_eager_and_compiled(tuple_outputs, module.tuple_outputs_compiled, (x, w))


def test_export_mixed_differentiable_and_integer_outputs(tmp_path):
    _requires_export_runtime()
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_mixed_outputs.py"

    export_autograd_triton(
        mixed_differentiable_and_integer_outputs,
        [Specialization(args=(x,), name="mixed")],
        generated_path,
    )
    module = _import_generated(generated_path)

    _compare_eager_and_compiled(
        mixed_differentiable_and_integer_outputs,
        module.mixed_differentiable_and_integer_outputs_compiled,
        (x,),
    )


def test_export_mixed_differentiable_and_detached_float_outputs(tmp_path):
    _requires_export_runtime()
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_detached_float_outputs.py"

    export_autograd_triton(
        mixed_differentiable_and_detached_float_outputs,
        [Specialization(args=(x,), name="detached_float")],
        generated_path,
    )
    module = _import_generated(generated_path)

    _compare_eager_and_compiled(
        mixed_differentiable_and_detached_float_outputs,
        module.mixed_differentiable_and_detached_float_outputs_compiled,
        (x,),
    )


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
        source_backend="clean_triton",
    )
    module = _import_generated(generated_path)
    artifact_source = "\n".join(
        path.read_text()
        for path in generated_path.with_name("generated_dynamic_artifacts").glob("*.py")
    )
    assert "@triton.jit" in artifact_source
    assert "async_compile.triton" not in artifact_source
    assert "triton.cdiv" in artifact_source

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


def test_dynamic_shared_symbol_across_tensor_guards(tmp_path):
    _requires_export_runtime()
    batch = torch.export.Dim("batch", min=1, max=16)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    y = torch.randn(4, 8, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_dynamic_shared_symbol.py"

    export_autograd_triton(
        shared_batch_pointwise,
        [
            Specialization(
                args=(x, y),
                dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
                name="shared_batch",
            )
        ],
        generated_path,
        source_backend="inductor",
    )
    module = _import_generated(generated_path)

    for batch_size in (1, 7, 16):
        _compare_eager_and_compiled(
            shared_batch_pointwise,
            module.shared_batch_pointwise_compiled,
            (
                torch.randn(batch_size, 8, device="cuda", requires_grad=True),
                torch.randn(batch_size, 8, device="cuda", requires_grad=True),
            ),
        )

    with pytest.raises(RuntimeError, match="does not equal batch=4"):
        module.shared_batch_pointwise_compiled(
            torch.randn(4, 8, device="cuda", requires_grad=True),
            torch.randn(5, 8, device="cuda", requires_grad=True),
        )


def test_dynamic_stride_and_device_mismatch_diagnostics(tmp_path):
    _requires_export_runtime()
    batch = torch.export.Dim("batch", min=1, max=16)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_dynamic_guard_diagnostics.py"

    export_autograd_triton(
        trig_pointwise,
        [Specialization(args=(x,), dynamic_shapes={"x": {0: batch}})],
        generated_path,
        source_backend="inductor",
    )
    module = _import_generated(generated_path)

    with pytest.raises(RuntimeError, match="stride"):
        module.trig_pointwise_compiled(
            torch.randn(4, 16, device="cuda", requires_grad=True)[:, ::2]
        )

    with pytest.raises(RuntimeError, match="device type cpu != cuda"):
        module.trig_pointwise_compiled(torch.randn(4, 8, requires_grad=True))


def test_dynamic_clean_triton_view_residual_ordering(tmp_path):
    _requires_export_runtime()
    batch = torch.export.Dim("batch", min=1, max=16)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_dynamic_view.py"

    export_autograd_triton(
        view_output,
        [
            Specialization(
                args=(x,),
                dynamic_shapes={"x": {0: batch}},
                name="dynamic_batch_view",
            )
        ],
        generated_path,
        source_backend="clean_triton",
    )
    module = _import_generated(generated_path)

    for batch_size in (1, 7, 16):
        dynamic_x = torch.randn(batch_size, 8, device="cuda", requires_grad=True)
        _compare_eager_and_compiled(
            view_output,
            module.view_output_compiled,
            (dynamic_x,),
        )


def test_dynamic_clean_triton_source_patches_multiple_pointwise_launches():
    source = """
import triton

def call(args):
    s0 = args[0].size(0)
    triton_poi_fused_sin_0_xnumel = s0 * 8
    triton_poi_fused_cos_1_xnumel = s0 * 16
    triton_poi_fused_sin_0[(1, 1, 1)](buf0, 32, XBLOCK=32, num_warps=1)
    triton_poi_fused_cos_1[(1, 1, 1)](
        buf1,
        64,
        XBLOCK=64,
        num_warps=2,
    )
"""

    result = clean_triton_source(source, dynamic=True)

    assert result.patched_launches == ["triton_poi_fused_sin_0", "triton_poi_fused_cos_1"]
    assert "triton.cdiv(triton_poi_fused_sin_0_xnumel, 32)" in result.source
    assert "triton.cdiv(triton_poi_fused_cos_1_xnumel, 64)" in result.source
    assert "buf0,\n        triton_poi_fused_sin_0_xnumel,\n        XBLOCK=32" in result.source
    assert "buf1,\n        triton_poi_fused_cos_1_xnumel,\n        XBLOCK=64" in result.source
    assert "[(1, 1, 1)]" not in result.source


def test_clean_triton_source_removes_unused_imports_assignments_and_blank_runs():
    source = """
import math
import os
import torch
import torch
from pathlib import Path
from torch._C import unused_stream, _cuda_getCurrentRawStream as get_raw_stream

unused_alias = torch.ops.aten
used_alias = torch.ops.aten




def call(args):
    stream = get_raw_stream(0)
    return torch.empty(1), used_alias, stream
"""

    result = clean_triton_source(source, dynamic=False)

    assert "import math" not in result.source
    assert "import os" not in result.source
    assert result.source.count("import torch") == 1
    assert "from pathlib import Path" not in result.source
    assert "unused_alias" not in result.source
    assert "used_alias = torch.ops.aten" in result.source
    assert "unused_stream" not in result.source
    assert "get_raw_stream" in result.source
    assert "\n\n\n" not in result.source


def test_clean_triton_source_strips_standalone_benchmark_tail():
    source = """
def call(args):
    return args


def get_args():
    return []


def benchmark_compiled_module(args, times=10, repeat=10):
    return None


if __name__ == "__main__":
    print(get_args())
"""

    result = clean_triton_source(source, dynamic=False)

    assert "def call(args):" in result.source
    assert "def get_args():" not in result.source
    assert "benchmark_compiled_module" not in result.source
    assert "__main__" not in result.source


def test_clean_triton_source_does_not_strip_non_benchmark_get_args_helper():
    source = """
def call(args):
    return get_args(args)


def get_args(args):
    return args


def still_needed(args):
    return args
"""

    result = clean_triton_source(source, dynamic=False)

    assert "def get_args(args):" in result.source
    assert "def still_needed(args):" in result.source


def test_dynamic_clean_triton_source_patches_repeated_kernel_launches_by_occurrence():
    original_source = """
def call(args):
    s0 = args[0].size(0)
    s1 = args[1].size(0)
    triton_poi_fused_same_0.run(buf0, s0, stream=stream0)
    triton_poi_fused_same_0.run(buf1, s1, stream=stream0)
"""
    source = """
import triton

def call(args):
    s0 = args[0].size(0)
    s1 = args[1].size(0)
    triton_poi_fused_same_0[(1, 1, 1)](buf0, 4, XBLOCK=4)
    triton_poi_fused_same_0[(2, 1, 1)](buf1, 8, XBLOCK=4)
"""

    result = clean_triton_source(source, dynamic=True, original_source=original_source)

    assert (
        "triton.cdiv(s0, 4),\n        1,\n        1,\n    )](\n        buf0,\n        s0,"
        in result.source
    )
    assert (
        "triton.cdiv(s1, 4),\n        1,\n        1,\n    )](\n        buf1,\n        s1,"
        in result.source
    )


def test_dynamic_clean_triton_reduction(tmp_path):
    _requires_export_runtime()
    batch = torch.export.Dim("batch", min=1, max=16)
    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_dynamic_reduction.py"

    export_autograd_triton(
        sum_last_dim,
        [
            Specialization(
                args=(x,),
                dynamic_shapes={"x": {0: batch}},
                name="dynamic_batch_sum",
            )
        ],
        generated_path,
        source_backend="clean_triton",
    )
    module = _import_generated(generated_path)
    artifact_source = "\n".join(
        path.read_text()
        for path in generated_path.with_name("generated_dynamic_reduction_artifacts").glob("*.py")
    )
    assert "@triton.jit" in artifact_source
    assert "async_compile.triton" not in artifact_source
    assert "triton.cdiv(s" in artifact_source
    assert "[(2, 1, 1)]" not in artifact_source

    for batch_size in (1, 7, 16):
        dynamic_x = torch.randn(batch_size, 128, device="cuda", requires_grad=True)
        _compare_eager_and_compiled(
            sum_last_dim,
            module.sum_last_dim_compiled,
            (dynamic_x,),
            rtol=2e-4,
            atol=2e-4,
        )


def test_dynamic_clean_triton_matmul_dynamic_m(tmp_path):
    _requires_export_runtime()
    batch = torch.export.Dim("batch", min=1, max=64)
    x = torch.randn(16, 128, device="cuda", requires_grad=True)
    w = torch.randn(128, 256, device="cuda", requires_grad=True)
    generated_path = tmp_path / "generated_dynamic_matmul.py"

    export_autograd_triton(
        affine_activation,
        [
            Specialization(
                args=(x, w),
                kwargs={"activation": "relu"},
                dynamic_shapes={"x": {0: batch}},
                name="dynamic_batch_matmul",
            )
        ],
        generated_path,
        source_backend="clean_triton",
        max_autotune=True,
    )
    module = _import_generated(generated_path)
    artifact_source = "\n".join(
        path.read_text()
        for path in generated_path.with_name("generated_dynamic_matmul_artifacts").glob("*.py")
    )
    assert "@triton.jit" in artifact_source
    assert "async_compile.triton" not in artifact_source
    assert "triton_tem_" in artifact_source
    assert "[(\n" in artifact_source
    assert "((15 + s" in artifact_source
    assert "[(4, 1, 1)]" not in artifact_source

    for batch_size in (1, 7, 32):
        dynamic_x = torch.randn(batch_size, 128, device="cuda", requires_grad=True)
        _compare_eager_and_compiled(
            affine_activation,
            module.affine_activation_compiled,
            (dynamic_x, w),
            {"activation": "relu"},
            rtol=2e-4,
            atol=2e-4,
        )


def test_dynamic_clean_triton_source_rejects_one_dimensional_baked_launch():
    source = """
import triton

def call(args):
    s0 = args[0].size(0)
    triton_unknown_0_xnumel = s0
    triton_unknown_0[(1,)](buf0, 4, XBLOCK=4)
"""

    with pytest.raises(CleanTritonUnsupportedError, match="triton_unknown_0"):
        clean_triton_source(source, dynamic=True)


def test_dynamic_clean_triton_source_rejects_unsupported_baked_launch():
    source = """
import triton

def call(args):
    s0 = args[0].size(0)
    triton_red_fused_sum_0_xnumel = s0
    triton_red_fused_sum_0_rnumel = 128
    triton_red_fused_sum_0[(1, 1, 1)](buf0, buf1, 4, 128, XBLOCK=4, RBLOCK=128)
"""

    with pytest.raises(CleanTritonUnsupportedError, match='source_backend=\\"inductor\\"'):
        clean_triton_source(source, dynamic=True)


def test_dynamic_shape_limitations_are_explicitly_guarded(tmp_path):
    _requires_export_runtime()
    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    w = torch.randn(4, 5, device="cuda", requires_grad=True)

    with pytest.raises(NotImplementedError, match="forward-only"):
        export_autograd_triton(
            integer_tensor_output,
            [Specialization(args=(x,), dynamic_shapes={"x": {0: "batch"}})],
            tmp_path / "generated_dynamic_forward_only.py",
            source_backend="inductor",
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
