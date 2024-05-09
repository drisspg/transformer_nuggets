import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest
import torch
from transformer_nuggets.utils.shape_trace import open_logs, ShapeLog
from transformer_nuggets.utils.tracing import NanInfDetect


def test_nan():
    a = torch.tensor(
        [
            0.0,
        ]
    )
    with pytest.raises(RuntimeError, match="returned a NaN"), NanInfDetect():
        print(torch.div(a, a))


def test_inf():
    a = torch.tensor(
        [
            1.0,
        ],
        dtype=torch.float16,
    )
    with pytest.raises(RuntimeError, match="returned an Inf"), NanInfDetect():
        print(torch.mul(a, 65537))


def test_breakpoint():
    a = torch.tensor(
        [
            0.0,
        ]
    )
    with (
        pytest.raises(RuntimeError, match="returned a NaN"),
        mock.patch("builtins.breakpoint") as mock_breakpoint,
        NanInfDetect(do_breakpoint=True),
    ):
        print(torch.div(a, a))
        mock_breakpoint.assert_called_once()


def test_shape_log():
    # Create an in-memory file-like object
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = Path(temp_file.name)

    aten = torch.ops.aten
    mode = ShapeLog(temp_path)

    with mode:
        torch.nn.functional.linear(torch.randn(3, 4), torch.randn(5, 4))
    logs = open_logs(temp_path)
    assert str(aten.mm.default) in logs.keys()
    assert str(aten.randn.default) in logs.keys()

    mm_ops = [aten.mm.default]
    with ShapeLog(temp_path, specific_ops=mm_ops):
        torch.nn.functional.linear(torch.randn(3, 4), torch.randn(5, 4))
    logs = open_logs(temp_path)
    assert str(aten.mm.default) in logs.keys()
    assert str(aten.randn.default) not in logs.keys()


if __name__ == "__main__":
    pytest.main([__file__])
