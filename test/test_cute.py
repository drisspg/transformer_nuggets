#!/usr/bin/env python3
"""Test script to demonstrate CUTE kernel caching and dynamic arguments"""

import torch
import pytest

# Skip entire module if CUDA is not available or CUTE cannot be imported
try:
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available", allow_module_level=True)
except ImportError:
    pytest.skip("CUTE not available", allow_module_level=True)

from operator import add, mul
from transformer_nuggets.cute import get_cache_stats, clear_cute_cache, auto_compile_and_cache
from transformer_nuggets.cute.dynamic_args import elementwise_op_dynamic
from transformer_nuggets.cute.utils import get_tensor_alignment


@cute.kernel
def simple_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # pyrefly: ignore  # not-iterable
    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

    # pyrefly: ignore  # unsupported-operation
    gC[mi, ni] = a_val + b_val


@cute.kernel
def simple_mul_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # pyrefly: ignore  # not-iterable
    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

    # pyrefly: ignore  # unsupported-operation
    gC[mi, ni] = a_val * b_val


def cached_elementwise(
    op,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # Define the kernel with @cute.jit
    @cute.jit
    def add_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        num_threads_per_block = 256
        # pyrefly: ignore  # not-iterable
        m, n = mA.shape

        # pyrefly: ignore  # missing-attribute, bad-argument-count
        simple_add_kernel(mA, mB, mC).launch(
            grid=((m * n) // num_threads_per_block, 1, 1), block=(num_threads_per_block, 1, 1)
        )

    @cute.jit
    def mul_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        num_threads_per_block = 256
        # pyrefly: ignore  # not-iterable
        m, n = mA.shape

        # pyrefly: ignore  # missing-attribute, bad-argument-count
        simple_mul_kernel(mA, mB, mC).launch(
            grid=((m * n) // num_threads_per_block, 1, 1), block=(num_threads_per_block, 1, 1)
        )

    # Use explicit caching based on operation
    if op == add:
        compiled_kernel = auto_compile_and_cache(add_kernel, mA, mB, mC)
        return compiled_kernel(mA, mB, mC)
    elif op == mul:
        compiled_kernel = auto_compile_and_cache(mul_kernel, mA, mB, mC)
        return compiled_kernel(mA, mB, mC)
    else:
        raise ValueError(f"Unsupported operation: {op}")


def test():
    print("CUTE Kernel Caching Demo")
    print("=" * 50)

    # Create test tensors
    M, N = 1024, 1024
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    # Convert to CUTE tensors
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    print("\n1. First call with 'add' - should compile:")
    cached_elementwise(add, a_, b_, c_)
    torch.testing.assert_close(c, a + b)
    print("✓ Addition result correct")

    print("\n2. Second call with 'add' - should hit cache:")
    c.zero_()
    cached_elementwise(add, a_, b_, c_)

    print("\n3. First call with 'mul' - should compile (different op):")
    c.zero_()
    cached_elementwise(mul, a_, b_, c_)
    torch.testing.assert_close(c, a * b)
    print("✓ Multiplication result correct")

    print("\n4. Different tensor sizes - should compile:")
    a2 = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b2 = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c2 = torch.zeros(512, 512, device="cuda", dtype=torch.float16)

    a2_ = from_dlpack(a2, assumed_align=16)
    b2_ = from_dlpack(b2, assumed_align=16)
    c2_ = from_dlpack(c2, assumed_align=16)

    cached_elementwise(add, a2_, b2_, c2_)

    print("\n5. Same size as #4 - should hit cache:")
    cached_elementwise(add, a2_, b2_, c2_)

    print("\n" + "=" * 50)
    print("Final Cache Statistics:")
    stats = get_cache_stats()
    print(f"  Total calls: {stats['total']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Unique kernels cached: {stats['cache_size']}")

    print("\nClearing cache...")
    clear_cute_cache()
    stats = get_cache_stats()
    print(f"  Cache size after clear: {stats['cache_size']}")


# Dynamic Args Tests
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
    # pyrefly: ignore  # bad-argument-type
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

    # pyrefly: ignore  # bad-argument-type
    elementwise_op_dynamic(add, a, b)
    stats1 = get_cache_stats()
    assert stats1["misses"] >= 1

    # Second call with same shapes should be a cache hit
    c = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    d = torch.randn(512, 512, device="cuda", dtype=torch.float16)

    # pyrefly: ignore  # bad-argument-type
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

        # pyrefly: ignore  # bad-argument-type
        out = elementwise_op_dynamic(add, a, b)
        expected = a + b
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

    # Should have multiple cache entries for different configurations
    stats = get_cache_stats()
    assert stats["cache_size"] >= len(shapes)


if __name__ == "__main__":
    pytest.main([__file__])
