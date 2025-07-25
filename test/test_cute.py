#!/usr/bin/env python3
"""Test script to demonstrate CUTE kernel caching"""

import torch
from operator import add, mul
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest

from transformer_nuggets.cute import cute_compile_and_cache, get_cache_stats, clear_cute_cache


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

    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

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

    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

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
        m, n = mA.shape

        simple_add_kernel(mA, mB, mC).launch(
            grid=((m * n) // num_threads_per_block, 1, 1), block=(num_threads_per_block, 1, 1)
        )

    @cute.jit
    def mul_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        num_threads_per_block = 256
        m, n = mA.shape

        simple_mul_kernel(mA, mB, mC).launch(
            grid=((m * n) // num_threads_per_block, 1, 1), block=(num_threads_per_block, 1, 1)
        )

    # Use explicit caching based on operation
    if op == add:
        compiled_kernel = cute_compile_and_cache(add_kernel, mA, mB, mC)
        return compiled_kernel(mA, mB, mC)
    elif op == mul:
        compiled_kernel = cute_compile_and_cache(mul_kernel, mA, mB, mC)
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


if __name__ == "__main__":
    pytest.main([__file__])
