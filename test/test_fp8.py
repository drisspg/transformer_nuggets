import pytest
import torch

from transformer_nuggets.fp8.scaled_quant import (
    dynamic_scaled_quant,
    eager_dynamic_scaled_quant,
    eager_scaled_quant,
    scaled_quant,
)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_basic_quant(fp8_dtype):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    a = torch.rand(2**12, 2**12, dtype=torch.float32, device="cuda") * 9.6
    scale = torch.tensor([4.0], dtype=torch.float32, device="cuda")
    abs_max = torch.tensor([-1.0], dtype=torch.float32, device="cuda")
    output = scaled_quant(a, scale, abs_max, fp8_dtype)
    torch.testing.assert_close(output, (a * scale).to(fp8_dtype))
    torch.testing.assert_close(abs_max, torch.tensor([9.6], dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_saturated(fp8_dtype):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    a = torch.rand(2**12, 2**12, dtype=torch.float32, device="cuda")
    a[0, :100] = torch.finfo(fp8_dtype).max + 100
    scale = torch.tensor([4.0], dtype=torch.float32, device="cuda")
    abs_max = torch.tensor([-1.0], dtype=torch.float32, device="cuda")
    output = scaled_quant(a, scale, abs_max, fp8_dtype, saturated=True)
    eager_abs_max = torch.clone(abs_max)
    eager_output = eager_scaled_quant(a, scale, eager_abs_max, fp8_dtype, saturated=True)
    torch.testing.assert_close(output, eager_output)
    torch.testing.assert_close(
        abs_max,
        torch.tensor([torch.finfo(fp8_dtype).max + 100], dtype=torch.float32, device="cuda"),
    )


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dynamic_quant(fp8_dtype):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    a = torch.randn(2**12, 2**12, dtype=torch.float32, device="cuda") * 9.6
    output = eager_dynamic_scaled_quant(a, fp8_dtype)
    output_triton = dynamic_scaled_quant(a, fp8_dtype)
    torch.testing.assert_close(output.to(torch.float32), output_triton.to(torch.float32))


if __name__ == "__main__":
    pytest.main([__file__])
