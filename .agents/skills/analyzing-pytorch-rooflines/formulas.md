# Formula Registration

## Parent operation

```python
from transformer_nuggets.roofline import RooflineWork, register_roofline_formula

@register_roofline_formula(torch.ops.my_ops.foo.default)
def foo_work(inputs, outputs, kwargs):
    x, weight = inputs[:2]
    result = outputs[0]
    return RooflineWork(
        logical_flops=...,
        read_bytes=x.nbytes + weight.nbytes,
        write_bytes=result.nbytes,
        model_kind="foo",
        confidence="high",
    )
```

Targets may be an `OpOverload`, `OpOverloadPacket`, or exact profiler event string. Registration rejects collisions unless `replace=True`.

Use `trace_compatible=False` when the formula requires FX output metadata. Raw traces pass `outputs=()` and `_concrete_inputs` in `kwargs`.

## Inner kernel

```python
from transformer_nuggets.roofline import register_kernel_roofline_formula

@register_kernel_roofline_formula(
    "my_ops::foo",
    kernel_name=r"foo_stage_1",
)
def foo_stage_1(inputs, outputs, kwargs):
    kernel_name = kwargs["kernel_name"]
    grid = kwargs["grid"]
    block = kwargs["block"]
    ...
```

Inner formulas are matched by parent op plus kernel-name regex. They are used only by raw-trace decoration.

## Work contract

- `logical_flops`: useful semantic FLOPs, or `None`.
- `read_bytes`: minimal required input traffic, or `None`.
- `write_bytes`: externally visible output traffic, or `None`.
- `model_kind`: stable short identifier.
- `confidence`: `high`, `medium`, `low`, or `unknown`.

Unknown values are `None`, never zero. Formulas must be pure shape/dtype models and must not launch tensors or read measured durations.

## Confidence

- `high`: direct operation equation and tensor footprints.
- `medium`: standard analytical approximation with limited ambiguity.
- `low`: useful only for selecting a measurement target.
- `unknown`: no defensible formula.

Low-confidence kernels are blue `needs measurement` entries, never red optimization findings.

## Coverage

For a multi-kernel parent op, the trace records:

```text
inner_kernel_formula_count
inner_kernel_flop_coverage_ratio
inner_kernel_byte_coverage_ratio
```

Coverage near one is only meaningful when inner formulas are independently derived. Fractions deliberately chosen to sum to the parent do not validate a model.

If no inner formula exists, kernels receive prefixed parent-op context with `trace_roofline_work_scope=parent_op_shared`; parent totals are not presented as kernel-specific work.
