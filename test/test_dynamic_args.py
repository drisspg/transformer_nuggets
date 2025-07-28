#!/usr/bin/env python3
"""Test script for dynamic argument elementwise operations with alignment-aware caching"""

import torch
from operator import add
import pytest

from transformer_nuggets.cute.dynamic_args import elementwise_op_dynamic
from transformer_nuggets.cute.cache import get_cache_stats, clear_cute_cache
from transformer_nuggets.cute.utils import get_tensor_alignment


@pytest.mark.parametrize(
    "shape",
    [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (1000, 1000),
        (1234, 5678),
        (3333, 7777),
        (999, 1001),
    ],
)
def test_elementwise_op_dynamic_correctness(shape):
    """Test that dynamic elementwise operations produce correct results"""
    M, N = shape
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)

    # Test the dynamic API
    out = elementwise_op_dynamic(add, a, b)
    expected = a + b

    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_alignment_detection():
    """Test tensor alignment detection works correctly"""
    # Test aligned tensor
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    align_a = get_tensor_alignment(a, dim=-1)
    assert align_a >= 2  # Should be at least element size aligned

    # Test different dtypes
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    align_b = get_tensor_alignment(b, dim=-1)
    assert align_b >= 4  # float32 is 4 bytes


def test_cache_behavior():
    """Test that caching works correctly with alignment"""
    clear_cute_cache()

    # First call should be a cache miss
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)

    elementwise_op_dynamic(add, a, b)
    stats1 = get_cache_stats()
    assert stats1["misses"] >= 1

    # Second call with same shapes should be a cache hit
    c = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    d = torch.randn(512, 512, device="cuda", dtype=torch.float16)

    elementwise_op_dynamic(add, c, d)
    stats2 = get_cache_stats()
    assert stats2["hits"] >= 1


def test_different_sizes_use_different_configs():
    """Test that different tensor sizes use different thread/value configurations"""
    clear_cute_cache()

    shapes = [(256, 256), (2048, 2048), (8192, 8192)]

    for M, N in shapes:
        a = torch.randn(M, N, device="cuda", dtype=torch.float16)
        b = torch.randn(M, N, device="cuda", dtype=torch.float16)

        out = elementwise_op_dynamic(add, a, b)
        expected = a + b
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

    # Should have multiple cache entries for different configurations
    stats = get_cache_stats()
    assert stats["cache_size"] >= len(shapes)


if __name__ == "__main__":
    pytest.main([__file__])
