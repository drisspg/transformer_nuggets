import torch


def _ulp_distance_unified(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Unified ULP distance calculation for any floating point dtype"""
    dtype = a.dtype

    if dtype == torch.bfloat16 or dtype == torch.float16:
        int_dtype = torch.int16
        sign_shift = 15
        magnitude_mask = 0x7FFF
    elif dtype == torch.float32:
        int_dtype = torch.int32
        sign_shift = 31
        magnitude_mask = 0x7FFFFFFF
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    a_int = a.view(int_dtype)
    b_int = b.view(int_dtype)

    sign_a = (a_int >> sign_shift) & 1
    sign_b = (b_int >> sign_shift) & 1

    same_sign = sign_a == sign_b

    ulp_a = a_int & magnitude_mask
    ulp_b = b_int & magnitude_mask

    diff_sign_distance = ulp_a + ulp_b + 1
    same_sign_distance = (a_int - b_int).abs()

    result = torch.where(same_sign, same_sign_distance, diff_sign_distance)
    return result.to(torch.int32)


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate ULP (Unit in Last Place) distance between two tensors.

    Automatically detects the dtype and uses the appropriate ULP calculation.
    Supports bfloat16, float16, and float32.

    Args:
        a: First tensor
        b: Second tensor (must have same dtype as a)

    Returns:
        Tensor of ULP distances with same shape as input tensors

    Raises:
        ValueError: If dtypes don't match or are unsupported
    """
    if a.dtype != b.dtype:
        raise ValueError(f"Tensor dtypes must match: {a.dtype} vs {b.dtype}")

    if a.shape != b.shape:
        raise ValueError(f"Tensor shapes must match: {a.shape} vs {b.shape}")

    return _ulp_distance_unified(a, b)


def assert_close_with_ulp(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    equal_nan: bool = False,
    msg: str = "",
    verbose: bool = True,
    max_failures_to_show: int = 10,
) -> None:
    """
    Assert that two tensors are close, with detailed ULP analysis on failure.

    This function replicates PyTorch's torch.testing.assert_close algorithm exactly,
    but adds detailed ULP distance analysis when the assertion fails.

    Args:
        actual: Actual tensor
        expected: Expected tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        equal_nan: If True, NaN values in the same positions are considered equal
        msg: Optional message prefix for the assertion error
        verbose: If True, print detailed analysis on failure
        max_failures_to_show: Maximum number of failing elements to show details for

    Raises:
        AssertionError: If tensors are not close within the specified tolerances
    """
    assert actual.dtype == expected.dtype, "Dtypes of actual and expected must match"
    tol_dict = {
        torch.float16: (1e-3, 1e-5),
        torch.bfloat16: (1.6e-2, 1e-5),
        torch.float32: (1.3e-6, 1e-5),
        torch.float64: (1e-7, 1e-7),
    }

    rtol, atol = tol_dict[actual.dtype]

    matches = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)

    if torch.all(matches):
        return  # Test passes

    # Test fails - prepare detailed error message
    number_of_elements = matches.numel()
    total_mismatches = number_of_elements - int(torch.sum(matches))

    # Flatten tensors for analysis
    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    matches_flat = matches.flatten()

    # Step 2: Find maximum absolute difference (PyTorch's algorithm)
    abs_diff = torch.abs(actual_flat - expected_flat)
    abs_diff_for_max = abs_diff.clone()
    abs_diff_for_max[matches_flat] = 0
    max_abs_diff, max_abs_diff_flat_idx = torch.max(abs_diff_for_max, 0)

    # Step 3: Find maximum relative difference (PyTorch's algorithm)
    rel_diff = abs_diff / torch.abs(expected_flat).clamp(min=1e-20)
    rel_diff_for_max = rel_diff.clone()
    rel_diff_for_max[matches_flat] = 0
    max_rel_diff, max_rel_diff_flat_idx = torch.max(rel_diff_for_max, 0)

    # Unravel indices
    max_abs_diff_idx = torch.unravel_index(max_abs_diff_flat_idx, actual.shape)
    max_rel_diff_idx = torch.unravel_index(max_rel_diff_flat_idx, actual.shape)

    # Build error message
    error_parts = []
    if msg:
        error_parts.append(msg)

    error_parts.append("Tensor-likes are not close!")
    error_parts.append(
        f"\nMismatched elements: {total_mismatches} / {number_of_elements} ({total_mismatches / number_of_elements:.1%})"
    )

    max_abs_tuple = tuple(idx.item() for idx in max_abs_diff_idx)
    max_rel_tuple = tuple(idx.item() for idx in max_rel_diff_idx)

    error_parts.append(
        f"Greatest absolute difference: {max_abs_diff.item():.6g} at index {max_abs_tuple} (up to {atol:.6g} allowed)"
    )
    error_parts.append(
        f"Greatest relative difference: {max_rel_diff.item():.6g} at index {max_rel_tuple} (up to {rtol:.6g} allowed)"
    )

    if verbose:
        # Add ULP analysis
        error_parts.append("\n" + "=" * 60)
        error_parts.append("ULP Analysis of Failures:")
        error_parts.append("=" * 60)

        # Get all failing elements
        failing_mask = ~matches_flat
        failing_indices = torch.where(failing_mask)[0]

        if failing_indices.numel() > 0:
            failing_actual = actual_flat[failing_mask]
            failing_expected = expected_flat[failing_mask]
            failing_abs_diff = abs_diff[failing_mask]
            failing_rel_diff = rel_diff[failing_mask]

            # Calculate ULP distances for all failures
            failing_ulps = ulp_distance(failing_actual, failing_expected)

            # Summary statistics
            error_parts.append(f"\nTotal failures: {failing_indices.numel()}")
            error_parts.append(
                f"ULP distances: min={failing_ulps.min().item()}, max={failing_ulps.max().item()}, mean={failing_ulps.float().mean().item():.1f}"
            )

            # Sort by absolute difference to show worst cases
            sorted_indices = torch.argsort(failing_abs_diff, descending=True)

            error_parts.append(
                f"\nTop {min(max_failures_to_show, len(sorted_indices))} failures by absolute difference:"
            )
            error_parts.append(
                "  #  | Index"
                + " " * (max(10, len(str(actual.shape))) - 5)
                + " | Abs Diff    | Rel Diff    | ULP  | Expected     | Actual"
            )
            error_parts.append("-" * 100)

            for i in range(min(max_failures_to_show, len(sorted_indices))):
                fail_idx = sorted_indices[i]
                flat_idx = failing_indices[fail_idx]
                tensor_idx = torch.unravel_index(flat_idx, actual.shape)
                idx_tuple = tuple(idx.item() for idx in tensor_idx)

                exp_val = failing_expected[fail_idx].item()
                act_val = failing_actual[fail_idx].item()
                abs_d = failing_abs_diff[fail_idx].item()
                rel_d = failing_rel_diff[fail_idx].item()
                ulp_d = failing_ulps[fail_idx].item()

                idx_str = str(idx_tuple)
                error_parts.append(
                    f"  {i + 1:2d} | {idx_str:<{max(10, len(str(actual.shape)))}} | {abs_d:.6e} | {rel_d:.6e} | {ulp_d:4.0f} | {exp_val:12.6f} | {act_val:12.6f}"
                )

            # Check if the worst absolute and relative errors are at different locations
            if max_abs_tuple != max_rel_tuple:
                error_parts.append(
                    "\nNote: Maximum absolute and relative errors occur at different locations"
                )

                # Show ULP for both locations
                ulp_at_max_abs = ulp_distance(
                    actual_flat[max_abs_diff_flat_idx : max_abs_diff_flat_idx + 1],
                    expected_flat[max_abs_diff_flat_idx : max_abs_diff_flat_idx + 1],
                ).item()
                ulp_at_max_rel = ulp_distance(
                    actual_flat[max_rel_diff_flat_idx : max_rel_diff_flat_idx + 1],
                    expected_flat[max_rel_diff_flat_idx : max_rel_diff_flat_idx + 1],
                ).item()

                error_parts.append(
                    f"  Max abs diff location {max_abs_tuple}: {ulp_at_max_abs} ULP"
                )
                error_parts.append(
                    f"  Max rel diff location {max_rel_tuple}: {ulp_at_max_rel} ULP"
                )

    raise AssertionError("\n".join(error_parts))


__all__ = [
    "ulp_distance",
    "analyze_precision_differences",
    "categorize_differences",
    "assert_close_with_ulp",
]
