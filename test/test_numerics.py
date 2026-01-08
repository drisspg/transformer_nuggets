import torch
import pytest
import tempfile
from pathlib import Path
from transformer_nuggets.numerics import (
    ulp_distance,
    compute_rmse,
    compute_error_stats,
    plot_abs_diff_distribution,
)


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return request.param


class TestULPDistance:
    def test_bfloat16_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        result = ulp_distance(a, b)
        assert torch.all(result == 0)

    def test_float16_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        result = ulp_distance(a, b)
        assert torch.all(result == 0)

    def test_float32_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = ulp_distance(a, b)
        assert torch.all(result == 0)

    def test_bfloat16_adjacent_values(self):
        a = torch.tensor([1.0], dtype=torch.bfloat16)
        a_int = a.view(torch.int16)
        b_int = a_int + 1
        b = b_int.view(torch.bfloat16)

        result = ulp_distance(a, b)
        assert result.item() == 1

    def test_float16_adjacent_values(self):
        a = torch.tensor([1.0], dtype=torch.float16)
        a_int = a.view(torch.int16)
        b_int = a_int + 1
        b = b_int.view(torch.float16)

        result = ulp_distance(a, b)
        assert result.item() == 1

    def test_float32_adjacent_values(self):
        a = torch.tensor([1.0], dtype=torch.float32)
        a_int = a.view(torch.int32)
        b_int = a_int + 1
        b = b_int.view(torch.float32)

        result = ulp_distance(a, b)
        assert result.item() == 1

    def test_sign_crossing_bfloat16(self):
        a = torch.tensor([1.0], dtype=torch.bfloat16)
        b = torch.tensor([-1.0], dtype=torch.bfloat16)
        result = ulp_distance(a, b)
        assert result.item() > 0

    def test_sign_crossing_float16(self):
        a = torch.tensor([1.0], dtype=torch.float16)
        b = torch.tensor([-1.0], dtype=torch.float16)
        result = ulp_distance(a, b)
        assert result.item() > 0

    def test_sign_crossing_float32(self):
        a = torch.tensor([1.0], dtype=torch.float32)
        b = torch.tensor([-1.0], dtype=torch.float32)
        result = ulp_distance(a, b)
        assert result.item() > 0

    def test_zero_handling(self):
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        for dtype in dtypes:
            a = torch.tensor([0.0], dtype=dtype)
            b = torch.tensor([0.0], dtype=dtype)
            result = ulp_distance(a, b)
            assert result.item() == 0

    def test_mixed_dtype_error(self):
        a = torch.tensor([1.0], dtype=torch.bfloat16)
        b = torch.tensor([1.0], dtype=torch.float16)

        with pytest.raises(ValueError, match="Tensor dtypes must match"):
            ulp_distance(a, b)

    def test_shape_mismatch_error(self):
        a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        b = torch.tensor([1.0], dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="Tensor shapes must match"):
            ulp_distance(a, b)

    def test_unsupported_dtype_error(self):
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="Unsupported dtype"):
            ulp_distance(a, b)

    def test_multidimensional_tensors(self):
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        for dtype in dtypes:
            a = torch.randn(3, 4, 5, dtype=dtype)
            b = a.clone()
            result = ulp_distance(a, b)
            assert result.shape == a.shape
            assert torch.all(result == 0)


class TestEdgeCases:
    def test_nan_handling(self):
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        for dtype in dtypes:
            a = torch.tensor([float("nan")], dtype=dtype)
            b = torch.tensor([1.0], dtype=dtype)

            result = ulp_distance(a, b)
            assert result.shape == a.shape

    def test_inf_handling(self):
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        for dtype in dtypes:
            a = torch.tensor([float("inf")], dtype=dtype)
            b = torch.tensor([1.0], dtype=dtype)

            result = ulp_distance(a, b)
            assert result.shape == a.shape

    def test_large_tensors(self):
        a = torch.randn(1000, dtype=torch.bfloat16)
        b = a.clone()

        result = ulp_distance(a, b)
        assert result.shape == a.shape
        assert torch.all(result == 0)

    def test_empty_tensors(self):
        a = torch.tensor([], dtype=torch.bfloat16)
        b = torch.tensor([], dtype=torch.bfloat16)

        result = ulp_distance(a, b)
        assert result.shape == a.shape
        assert result.numel() == 0


class TestComputeRMSE:
    def test_identical_tensors(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([1.0, 2.0, 3.0], device=device)
        rmse = compute_rmse(a, b)
        assert rmse == 0.0

    def test_different_tensors(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([2.0, 3.0, 4.0], device=device)
        rmse = compute_rmse(a, b)
        assert rmse == pytest.approx(1.0, rel=1e-5)

    def test_multidimensional(self, device):
        a = torch.randn(3, 4, 5, device=device)
        b = a.clone()
        rmse = compute_rmse(a, b)
        assert rmse == 0.0

    def test_numpy_input(self, device):
        import numpy as np

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])

        rmse = compute_rmse(a, b)
        assert rmse == pytest.approx(1.0, rel=1e-5)

    def test_mixed_dtypes(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        b = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float16, device=device)
        rmse = compute_rmse(a, b)
        assert isinstance(rmse, float)
        assert rmse > 0.0


class TestComputeErrorStats:
    def test_identical_tensors(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([1.0, 2.0, 3.0], device=device)
        stats = compute_error_stats(a, b)

        assert stats["mean"] == 0.0
        assert stats["max"] == 0.0
        assert stats["median"] == 0.0
        assert stats["std"] == 0.0

    def test_different_tensors(self, device):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        b = torch.tensor([2.0, 3.0, 4.0, 5.0], device=device)
        stats = compute_error_stats(a, b)

        assert stats["mean"] == pytest.approx(1.0, rel=1e-5)
        assert stats["max"] == pytest.approx(1.0, rel=1e-5)
        assert stats["median"] == pytest.approx(1.0, rel=1e-5)
        assert stats["std"] == pytest.approx(0.0, rel=1e-5)

    def test_varied_errors(self, device):
        a = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        stats = compute_error_stats(a, b)

        assert stats["mean"] == pytest.approx(2.5, rel=1e-5)
        assert stats["max"] == pytest.approx(4.0, rel=1e-5)
        assert stats["median"] == pytest.approx(2.5, rel=1e-5)
        assert stats["std"] > 0.0

    def test_multidimensional(self, device):
        a = torch.randn(3, 4, 5, device=device)
        b = a + 0.1
        stats = compute_error_stats(a, b)

        assert "mean" in stats
        assert "max" in stats
        assert "median" in stats
        assert "std" in stats
        assert all(isinstance(v, float) for v in stats.values())

    def test_numpy_input(self, device):
        import numpy as np

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])

        stats = compute_error_stats(a, b)

        assert stats["mean"] == pytest.approx(1.0, rel=1e-5)
        assert stats["max"] == pytest.approx(1.0, rel=1e-5)


class TestPlotAbsDiffDistribution:
    def test_creates_plot_file(self, device):
        a = torch.randn(100, device=device)
        b = a + torch.randn(100, device=device) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            plot_abs_diff_distribution(a, b, save_path, name="Test")
            assert save_path.exists()

    def test_with_different_bins(self, device):
        a = torch.randn(100, device=device)
        b = a + 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            plot_abs_diff_distribution(a, b, save_path, bins=20)
            assert save_path.exists()

    def test_with_auto_bins(self, device):
        a = torch.randn(100, device=device)
        b = a + 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            plot_abs_diff_distribution(a, b, save_path, bins="auto")
            assert save_path.exists()

    def test_multidimensional_tensors(self, device):
        a = torch.randn(10, 10, device=device)
        b = a + 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            plot_abs_diff_distribution(a, b, save_path)
            assert save_path.exists()

    def test_numpy_input(self, device):
        import numpy as np

        a = np.random.randn(100)
        b = a + 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"

            plot_abs_diff_distribution(a, b, save_path)
            assert save_path.exists()
