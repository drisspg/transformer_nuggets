import torch
import pytest
from transformer_nuggets.numerics import (
    ulp_distance,
    analyze_precision_differences,
    categorize_differences,
)


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


class TestAnalyzePrecisionDifferences:
    def test_identical_tensors(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        b = a.clone()

        results = analyze_precision_differences(a, b, print_results=False)

        assert results["total_elements"] == 3
        assert results["mismatch_count"] == 0
        assert results["exact_match_count"] == 3
        assert results["mismatch_percentage"] == 0.0
        assert results["dtype"] == torch.bfloat16

    def test_different_tensors(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        a_int = a.view(torch.int16)
        b_int = a_int + torch.tensor([1, 2, 3], dtype=torch.int16)
        b = b_int.view(torch.bfloat16)

        results = analyze_precision_differences(a, b, print_results=False)

        assert results["total_elements"] == 3
        assert results["mismatch_count"] == 3
        assert results["exact_match_count"] == 0
        assert results["mismatch_percentage"] == 100.0
        assert "ulp_min" in results
        assert "ulp_max" in results
        assert "ulp_mean" in results
        assert "ulp_categories" in results
        assert "ulp_histogram" in results

    def test_mixed_differences(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        b = a.clone()
        b[1] = torch.nextafter(b[1], b[1] + 1)

        results = analyze_precision_differences(a, b, print_results=False)

        assert results["total_elements"] == 4
        assert results["mismatch_count"] == 1
        assert results["exact_match_count"] == 3
        assert results["mismatch_percentage"] == 25.0

    def test_print_output(self, capsys):
        a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        b = a.clone()

        analyze_precision_differences(a, b, name="test analysis", print_results=True)

        captured = capsys.readouterr()
        assert "test analysis" in captured.out
        assert "Total elements: 2" in captured.out
        assert "Exact matches: 2" in captured.out


class TestCategorizeDifferences:
    def test_all_exact(self):
        ulp_distances = torch.zeros(5, dtype=torch.int32)
        result = categorize_differences(ulp_distances)

        assert result["exact"] == 5
        assert result["1_ulp"] == 0
        assert result["2_ulp"] == 0
        assert result["small_3_10"] == 0
        assert result["medium_11_100"] == 0
        assert result["large_over_100"] == 0

    def test_mixed_categories(self):
        ulp_distances = torch.tensor([0, 1, 2, 5, 50, 200], dtype=torch.int32)
        result = categorize_differences(ulp_distances)

        assert result["exact"] == 1
        assert result["1_ulp"] == 1
        assert result["2_ulp"] == 1
        assert result["small_3_10"] == 1
        assert result["medium_11_100"] == 1
        assert result["large_over_100"] == 1

    def test_boundary_values(self):
        ulp_distances = torch.tensor([3, 10, 11, 100, 101], dtype=torch.int32)
        result = categorize_differences(ulp_distances)

        assert result["small_3_10"] == 2
        assert result["medium_11_100"] == 2
        assert result["large_over_100"] == 1


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
