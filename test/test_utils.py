import ctypes
import os
import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest
import torch
from transformer_nuggets.utils.benchmark import (
    _get_max_sm_clock,
    _get_nvml_handle,
    _nvml,
    locked_clocks,
    benchmark_cuda_function_in_microseconds_triton,
)
from transformer_nuggets.utils.shape_trace import open_logs, ShapeLog
from transformer_nuggets.utils.tracing import NanInfDetect


requires_root = pytest.mark.skipif(os.geteuid() != 0, reason="Locking GPU clocks requires root")


def _read_sm_clock() -> int:
    nvml = _nvml()
    nvml.nvmlInit()
    handle = _get_nvml_handle(nvml)
    clk = ctypes.c_uint()
    nvml.nvmlDeviceGetClockInfo(handle, ctypes.c_uint(1), ctypes.byref(clk))
    val = clk.value
    nvml.nvmlShutdown()
    return val


@requires_root
def test_locked_clocks_context_manager():
    with locked_clocks() as target_mhz:
        assert isinstance(target_mhz, int)
        assert target_mhz > 0

        x = torch.randn(1024, 1024, device="cuda")
        for _ in range(10):
            x @ x
        torch.cuda.synchronize()

        assert _read_sm_clock() == target_mhz


@requires_root
def test_locked_clocks_custom_frequency():
    nvml = _nvml()
    nvml.nvmlInit()
    handle = _get_nvml_handle(nvml)
    max_clk = _get_max_sm_clock(nvml, handle)
    nvml.nvmlShutdown()

    with locked_clocks(clock_mhz=max_clk) as target_mhz:
        assert target_mhz == max_clk


@requires_root
def test_locked_clocks_resets_on_exception():
    with pytest.raises(ValueError, match="intentional"):
        with locked_clocks():
            raise ValueError("intentional")


@requires_root
def test_benchmark_lock_clocks_kwarg():
    x = torch.randn(1024, 1024, device="cuda")
    t = benchmark_cuda_function_in_microseconds_triton(lambda: x @ x, LOCK_CLOCKS=True)
    assert t > 0


def test_locked_clocks_no_permission():
    if os.geteuid() == 0:
        pytest.skip("Running as root, cannot test permission failure")
    with pytest.raises(RuntimeError, match="Requires root"):
        with locked_clocks():
            pass


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
