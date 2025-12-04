import torch
from functools import partial
from operator import mul, add

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds
from transformer_nuggets.cute.cache import compile_and_cache, get_cache_stats
from transformer_nuggets.cute.base import CuteOp


class ElementwiseOp(CuteOp[[torch.Tensor, torch.Tensor, torch.Tensor], None]):
    """Elementwise operation using CUTE kernels."""

    def __init__(self, op: cutlass.Constexpr):
        super().__init__()
        self.op = op

    @cute.kernel
    # pyrefly: ignore  # bad-override
    def kernel(
        self,
        op: cutlass.Constexpr,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        blk_coord = ((None, None), bidx)

        blkA = gA[blk_coord]
        blkB = gB[blk_coord]
        blkC = gC[blk_coord]

        # pyrefly: ignore  # no-matching-overload
        tidfrgA = cute.composition(blkA, tv_layout)
        # pyrefly: ignore  # no-matching-overload
        tidfrgB = cute.composition(blkB, tv_layout)
        # pyrefly: ignore  # no-matching-overload
        tidfrgC = cute.composition(blkC, tv_layout)

        thr_coord = (tidx, None)

        thrA = tidfrgA[thr_coord]
        thrB = tidfrgB[thr_coord]
        thrC = tidfrgC[thr_coord]

        # pyrefly: ignore  # not-callable
        thrC[None] = op(thrA.load(), thrB.load())

    def get_key(self, *args, **kwargs) -> str:
        """Generate cache key including operation type and tensor properties."""
        op_name = getattr(self.op, "__name__", str(self.op))
        key_parts = [self.__class__.__name__, f"op={op_name}"]

        # Add tensor properties to key
        for arg in args:
            if isinstance(arg, cute.Tensor):
                key_parts.append(self._generate_tensor_key(arg))

        return "_".join(key_parts)

    def interface(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, *, assumed_align: int = 16
    ) -> None:
        mA = from_dlpack(a, assumed_align=assumed_align)
        mB = from_dlpack(b, assumed_align=assumed_align)
        mC = from_dlpack(c, assumed_align=assumed_align)
        return compile_and_cache(self, self.get_key(mA, mB, mC), self.op, mA, mB, mC)(mA, mB, mC)

    @cute.jit
    def __call__(self, op: cutlass.Constexpr, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        """JIT function that launches the elementwise kernel."""
        thr_layout = cute.make_layout((4, 32), stride=(32, 1))
        val_layout = cute.make_layout((4, 8), stride=(8, 1))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gA = cute.zipped_divide(mA, tiler_mn)
        gB = cute.zipped_divide(mB, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        # pyrefly: ignore  # missing-attribute, bad-argument-count
        self.kernel(op, gA, gB, gC, tv_layout).launch(
            grid=[cute.size(gC, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )


# Kernel definitions
@cute.kernel
def naive_elementwise_add_kernel(
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


@cute.jit
def naive_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    num_threads_per_block = 256
    # pyrefly: ignore  # not-iterable
    m, n = mA.shape

    # pyrefly: ignore  # bad-argument-count
    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    # pyrefly: ignore  # missing-attribute
    kernel.launch(
        grid=((m * n) // num_threads_per_block, 1, 1), block=(num_threads_per_block, 1, 1)
    )


# Public API function
def elementwise_op(
    op: cutlass.Constexpr,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    assumed_align: int = 16,
):
    """
    Apply an elementwise operation using CUTE kernels.

    Args:
        op: Cutlass constexpr operation (e.g., add, mul)
        a: Input tensor A (PyTorch tensor)
        b: Input tensor B (PyTorch tensor)
        c: Output tensor C (PyTorch tensor, will be modified in-place)
        assumed_align: Memory alignment assumption for dlpack conversion
    """
    # Convert PyTorch tensors to CUTE tensors
    return ElementwiseOp(op).interface(a, b, c, assumed_align=assumed_align)


def benchmark(callable, tensor_a):
    time = benchmark_cuda_function_in_microseconds(callable)
    avg_time = time / 1e3

    print(f"Average execution time: {avg_time:.4f} ms")

    total_bytes = 3 * tensor_a.numel() * tensor_a.element_size()
    throughput_gb_s = total_bytes / (avg_time / 1000) / 1e9
    print(f"Throughput: {throughput_gb_s:.2f} GB/s")
    print()


if __name__ == "__main__":
    M, N = 2048, 2048

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    # Test the new API that takes regular PyTorch tensors
    # pyrefly: ignore  # bad-argument-type
    elementwise_op(mul, a, b, c)
    torch.testing.assert_close(c, mul(a, b))
    print("✓ Multiplication test passed")

    c.zero_()

    # For naive kernel, we still need CUTE tensors
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    naive_elementwise_add_compiled = cute.compile(naive_elementwise_add, a_, b_, c_)

    # Note: elementwise_op is now cached automatically!
    # First call will compile, subsequent calls use cache

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50 + "\n")

    print("1. Naive elementwise add kernel:")
    benchmark(partial(naive_elementwise_add_compiled, a_, b_, c_), a)

    print("2. PyTorch add (baseline):")
    benchmark(partial(torch.add, a, b, out=c), a)

    print("3. Optimized elementwise add kernel (with caching):")
    benchmark(partial(elementwise_op, add, a, b, c, assumed_align=16), a)

    print("\nCache Statistics:")
    stats = get_cache_stats()
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Cache size: {stats['cache_size']}")
