import pytest
from typer.testing import CliRunner

from transformer_nuggets.roofline import (
    RooflineSpec,
    RooflineWork,
    app,
    decorate_trace_file,
    decorate_trace_roofline,
    register_kernel_roofline_formula,
    register_roofline_formula,
)
from transformer_nuggets.utils.perfetto import read_trace, write_trace


@register_roofline_formula("test::scaled_square")
def scaled_square_work(inputs, output, kwargs):
    del output, kwargs
    x = inputs[0]
    return RooflineWork(
        logical_flops=2 * x.numel,
        read_bytes=x.nbytes,
        write_bytes=x.nbytes,
        model_kind="scaled_square",
        confidence="high",
    )


@register_kernel_roofline_formula("test::scaled_square", kernel_name=r"scaled_square_kernel")
def scaled_square_kernel_work(inputs, output, kwargs):
    del output, kwargs
    x = inputs[0]
    return RooflineWork(
        logical_flops=2 * x.numel,
        read_bytes=x.nbytes,
        write_bytes=x.nbytes,
        model_kind="scaled_square_kernel",
        confidence="high",
    )


def test_decorate_trace_roofline_uses_registered_formula():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "test::scaled_square",
                "ts": 1,
                "dur": 2,
                "args": {
                    "External id": 7,
                    "Input Dims": [[4, 8]],
                    "Input type": ["float"],
                    "Input Strides": [[8, 1]],
                    "Concrete Inputs": [""],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "scaled_square_kernel",
                "ts": 4,
                "dur": 10,
                "args": {"External id": 7},
            },
        ]
    }

    processed = decorate_trace_roofline(trace)
    kernel = processed["traceEvents"][1]

    assert kernel["args"]["transformer_nuggets.trace_roofline"]
    assert kernel["args"]["trace_roofline_work_scope"] == "kernel_specific"
    assert kernel["args"]["trace_roofline_cpu_op"] == "test::scaled_square"
    assert kernel["args"]["logical_flops"] == 64
    assert kernel["args"]["logical_read_bytes"] == 128
    assert kernel["args"]["logical_write_bytes"] == 128
    assert kernel["args"]["logical_bytes"] == 256
    assert kernel["args"]["achieved_logical_tflops"] == 64 / 10 / 1e6
    assert kernel["args"]["achieved_logical_tbps"] == 256 / 10 / 1e6
    assert kernel["args"]["trace_roofline_parent_op_logical_flops"] == 64
    roofline_boxes = [
        event
        for event in processed["traceEvents"]
        if event.get("cat") == "gpu_roofline_annotation"
    ]
    assert len(roofline_boxes) == 1
    assert "scaled_square" in roofline_boxes[0]["name"]
    cpu_op = processed["traceEvents"][0]
    assert cpu_op["args"]["inner_kernel_formula_count"] == 1
    assert cpu_op["args"]["inner_kernel_flop_coverage_ratio"] == 1
    assert trace["traceEvents"][1]["args"] == {"External id": 7}


def test_decorate_trace_roofline_color_codes_slow_kernel():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "test::scaled_square",
                "args": {
                    "External id": 11,
                    "Input Dims": [[1024]],
                    "Input type": ["float"],
                    "Input Strides": [[1]],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "scaled_square_kernel",
                "ts": 1,
                "dur": 100,
                "args": {"External id": 11},
            },
        ]
    }

    processed = decorate_trace_roofline(
        trace,
        roofline_spec=RooflineSpec("test", 1.0, 1.0),
    )
    findings = [
        event
        for event in processed["traceEvents"]
        if event.get("cat") == "gpu_roofline_annotation"
        and event.get("args", {}).get("is_optimization_finding")
    ]

    assert len(findings) == 1
    assert findings[0]["args"]["severity"] == "high"
    assert findings[0]["name"].startswith("🔴")


def test_decorate_trace_file_writes_default_output(tmp_path):
    input_path = tmp_path / "trace.json"
    write_trace(
        input_path,
        {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "test::scaled_square",
                    "args": {
                        "External id": 3,
                        "Input Dims": [[4]],
                        "Input type": ["float"],
                        "Input Strides": [[1]],
                    },
                },
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "kernel",
                    "dur": 5,
                    "args": {"External id": 3},
                },
            ]
        },
    )

    output_path = decorate_trace_file(input_path)

    assert output_path.name == "trace.roofline.json.gz"
    assert read_trace(output_path)["traceEvents"][1]["args"]["logical_flops"] == 8


def test_annotate_roofline_cli_writes_trace(tmp_path):
    input_path = tmp_path / "trace.json"
    output_path = tmp_path / "annotated.json.gz"
    write_trace(
        input_path,
        {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "aten::mul",
                    "ts": 0,
                    "dur": 1,
                    "args": {
                        "External id": 30,
                        "Input Dims": [[4], [4]],
                        "Input type": ["float", "float"],
                    },
                },
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "mul_kernel",
                    "ts": 2,
                    "dur": 4,
                    "args": {"External id": 30},
                },
            ]
        },
    )

    result = CliRunner().invoke(
        app,
        [str(input_path), "-o", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert read_trace(output_path)["traceEvents"][1]["args"]["logical_flops"] == 4


def test_decorate_trace_roofline_falls_back_to_input_traffic():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "unknown::operation",
                "ts": 1,
                "dur": 2,
                "args": {
                    "External id": 9,
                    "Input Dims": [[16]],
                    "Input type": ["c10::Half"],
                    "Input Strides": [[1]],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "unknown_kernel",
                "ts": 4,
                "dur": 8,
                "args": {"External id": 9},
            },
        ]
    }

    kernel = decorate_trace_roofline(trace)["traceEvents"][1]

    assert kernel["args"]["model_kind"] == "generic_trace_io"
    assert kernel["args"]["model_confidence"] == "unknown"
    assert kernel["args"]["logical_flops"] is None
    assert kernel["args"]["logical_read_bytes"] == 32
    assert kernel["args"]["logical_write_bytes"] is None
    assert kernel["args"]["achieved_logical_read_tbps"] == 32 / 8 / 1e6
    assert kernel["args"]["achieved_logical_tbps"] is None


def test_decorate_trace_roofline_tolerates_malformed_metadata():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "unknown::bad",
                "ts": "bad",
                "dur": "bad",
                "args": {
                    "External id": [1],
                    "Input Dims": [[None, "bad"]],
                    "Input type": 5,
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "bad_kernel",
                "ts": "bad",
                "dur": -1,
                "args": {"External id": [1]},
            },
        ]
    }

    processed = decorate_trace_roofline(trace)

    assert len(processed["traceEvents"]) == 2
    assert not processed["traceEvents"][1]["args"].get("transformer_nuggets.trace_roofline")


def test_decorate_trace_roofline_skips_reused_external_id():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::mul",
                "ts": 0,
                "dur": 1,
                "args": {"External id": 4, "Input Dims": [[4]], "Input type": ["float"]},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::sigmoid",
                "ts": 2,
                "dur": 1,
                "args": {"External id": 4, "Input Dims": [[4]], "Input type": ["float"]},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "ambiguous_kernel",
                "ts": 4,
                "dur": 2,
                "args": {"External id": 4},
            },
        ]
    }

    kernel = decorate_trace_roofline(trace)["traceEvents"][2]

    assert kernel["args"]["trace_roofline_attribution_error"] == "ambiguous_external_id"
    assert not kernel["args"].get("transformer_nuggets.trace_roofline")


def test_decorate_trace_roofline_is_idempotent():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "test::scaled_square",
                "ts": 0,
                "dur": 1,
                "args": {
                    "External id": 12,
                    "Input Dims": [[4]],
                    "Input type": ["float"],
                    "Input Strides": [[1]],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "scaled_square_kernel",
                "ts": 2,
                "dur": 4,
                "args": {"External id": 12},
            },
        ]
    }

    once = decorate_trace_roofline(trace)
    twice = decorate_trace_roofline(once)

    assert sum(event.get("cat") == "gpu_roofline_annotation" for event in once["traceEvents"]) == 1
    assert (
        sum(event.get("cat") == "gpu_roofline_annotation" for event in twice["traceEvents"]) == 1
    )


def test_decorate_trace_roofline_isolates_failing_formula():
    @register_roofline_formula("test::failing_formula")
    def failing_formula(inputs, outputs, kwargs):
        del inputs, outputs, kwargs
        raise RuntimeError("broken formula")

    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "test::failing_formula",
                "ts": 0,
                "dur": 1,
                "args": {
                    "External id": 13,
                    "Input Dims": [[4]],
                    "Input type": ["float"],
                    "Input Strides": [[1]],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "failing_kernel",
                "ts": 2,
                "dur": 100,
                "args": {"External id": 13},
            },
        ]
    }

    processed = decorate_trace_roofline(trace)
    cpu_op, kernel, roofline = processed["traceEvents"]

    assert "RuntimeError: broken formula" in cpu_op["args"]["formula_error"]
    assert kernel["args"]["trace_roofline_work_scope"] == "single_kernel_parent_op"
    assert roofline["args"]["severity"] == "unknown"
    assert not roofline["args"]["is_optimization_finding"]


def test_decorate_trace_roofline_parses_scalar_tensor():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "unknown::scalar",
                "ts": 0,
                "dur": 1,
                "args": {
                    "External id": 14,
                    "Input Dims": [[]],
                    "Input type": ["float"],
                    "Input Strides": [[]],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "scalar_kernel",
                "ts": 2,
                "dur": 4,
                "args": {"External id": 14},
            },
        ]
    }

    cpu_op = decorate_trace_roofline(trace)["traceEvents"][0]

    assert cpu_op["args"]["logical_read_bytes"] == 4


def test_multi_kernel_parent_work_is_not_kernel_specific():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::sigmoid",
                "ts": 0,
                "dur": 1,
                "args": {
                    "External id": 15,
                    "Input Dims": [[8]],
                    "Input type": ["float"],
                    "Input Strides": [[1]],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "part_a",
                "ts": 2,
                "dur": 3,
                "args": {"External id": 15},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "part_b",
                "ts": 6,
                "dur": 3,
                "args": {"External id": 15},
            },
        ]
    }

    processed = decorate_trace_roofline(trace)
    kernels = processed["traceEvents"][1:3]

    assert all(
        kernel["args"]["trace_roofline_work_scope"] == "parent_op_shared" for kernel in kernels
    )
    assert all("logical_flops" not in kernel["args"] for kernel in kernels)
    roofline_boxes = [
        event
        for event in processed["traceEvents"]
        if event.get("cat") == "gpu_roofline_annotation"
    ]
    assert all(box["args"]["severity"] == "unknown" for box in roofline_boxes)


def test_register_roofline_formula_rejects_accidental_collision():
    @register_roofline_formula("test::collision")
    def first(inputs, outputs, kwargs):
        del inputs, outputs, kwargs
        return RooflineWork(None, None, None, "first")

    def second(inputs, outputs, kwargs):
        del inputs, outputs, kwargs
        return RooflineWork(None, None, None, "second")

    with pytest.raises(ValueError, match="already registered"):
        register_roofline_formula("test::collision")(second)


def test_trace_incompatible_formula_falls_back_cleanly():
    @register_roofline_formula("test::fx_only", trace_compatible=False)
    def fx_only(inputs, outputs, kwargs):
        del inputs, kwargs
        return RooflineWork(1, outputs[0].nbytes, outputs[0].nbytes, "fx_only")

    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "test::fx_only",
                "ts": 0,
                "dur": 1,
                "args": {
                    "External id": 20,
                    "Input Dims": [[4]],
                    "Input type": ["float"],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "fx_only_kernel",
                "ts": 2,
                "dur": 4,
                "args": {"External id": 20},
            },
        ]
    }

    cpu_op = decorate_trace_roofline(trace)["traceEvents"][0]

    assert cpu_op["args"]["model_kind"] == "generic_trace_io"
    assert cpu_op["args"]["logical_flops"] is None


def test_low_confidence_kernel_is_needs_measurement_not_finding():
    @register_roofline_formula("test::low_parent")
    def parent(inputs, outputs, kwargs):
        del outputs, kwargs
        return RooflineWork(100, inputs[0].nbytes, inputs[0].nbytes, "parent")

    @register_kernel_roofline_formula("test::low_parent", kernel_name="low_kernel")
    def child(inputs, outputs, kwargs):
        del outputs, kwargs
        return RooflineWork(
            100,
            inputs[0].nbytes,
            inputs[0].nbytes,
            "low_kernel",
            confidence="low",
        )

    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "test::low_parent",
                "ts": 0,
                "dur": 1,
                "args": {
                    "External id": 21,
                    "Input Dims": [[1024]],
                    "Input type": ["float"],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "low_kernel",
                "ts": 2,
                "dur": 100,
                "args": {"External id": 21},
            },
        ]
    }

    processed = decorate_trace_roofline(
        trace,
        roofline_spec=RooflineSpec("test", 1.0, 1.0),
    )
    box = next(
        event
        for event in processed["traceEvents"]
        if event.get("cat") == "gpu_roofline_annotation"
    )

    assert box["args"]["severity"] == "needs_measurement"
    assert box["name"].startswith("🔵")
    assert not box["args"]["is_optimization_finding"]
    assert box["args"]["needs_measurement"]


def test_backward_stage_uses_cross_thread_fallback_and_interval_union():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "model/backward",
                "pid": 1,
                "tid": 10,
                "ts": 0,
                "dur": 20,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::sigmoid",
                "pid": 1,
                "tid": 11,
                "ts": 1,
                "dur": 2,
                "args": {
                    "External id": 22,
                    "Input Dims": [[4]],
                    "Input type": ["float"],
                },
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::sigmoid",
                "pid": 1,
                "tid": 11,
                "ts": 4,
                "dur": 2,
                "args": {
                    "External id": 23,
                    "Input Dims": [[4]],
                    "Input type": ["float"],
                },
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "a",
                "pid": 0,
                "tid": 7,
                "ts": 8,
                "dur": 5,
                "args": {"External id": 22},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "b",
                "pid": 0,
                "tid": 8,
                "ts": 10,
                "dur": 5,
                "args": {"External id": 23},
            },
        ]
    }

    stage = decorate_trace_roofline(trace)["traceEvents"][0]

    assert stage["args"]["stage_attribution_scope"] == "cross_thread_backward_inclusive"
    assert stage["args"]["stage_gpu_wall_us"] == 7
    assert stage["args"]["stage_gpu_busy_us"] == 7
    assert stage["args"]["stage_gpu_time_us"] == 10
