import unittest.mock as mock

import pytest
import torch
from transformer_nuggets.utils.tracing import NanInfDetect


def test_nan():
    a = torch.tensor(
        [
            0.0,
        ]
    )
    with pytest.raises(RuntimeError, match="returned a NaN"):
        with NanInfDetect():
            print(torch.div(a, a))


def test_inf():
    a = torch.tensor(
        [
            1.0,
        ],
        dtype=torch.float16,
    )
    with pytest.raises(RuntimeError, match="returned an Inf"):
        with NanInfDetect():
            print(torch.mul(a, 65537))


def test_breakpoint():
    a = torch.tensor(
        [
            0.0,
        ]
    )
    with pytest.raises(RuntimeError, match="returned a NaN"):
        with mock.patch("builtins.breakpoint") as mock_breakpoint:
            with NanInfDetect(do_breakpoint=True):
                print(torch.div(a, a))
            mock_breakpoint.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
