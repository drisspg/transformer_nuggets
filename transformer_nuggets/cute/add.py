import torch
from functools import partial
from operator import mul, add

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds


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

    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

    gC[mi, ni] = a_val + b_val


@cute.kernel
def elementwise_apply_kernel(
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

    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    thr_coord = (tidx, None)

    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    thrC[None] = op(thrA.load(), thrB.load())


@cute.jit
def naive_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    num_threads_per_block = 256
    m, n = mA.shape

    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    kernel.launch(
        grid=((m * n) // num_threads_per_block, 1, 1), block=(num_threads_per_block, 1, 1)
    )


@cute.jit
def elementwise_op(
    op: cutlass.Constexpr,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    elementwise_apply_kernel(op, gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def benchmark(callable, *, num_warmups=5, num_iterations=200):
    time = benchmark_cuda_function_in_microseconds(callable)
    avg_time = time / 1e3

    print(f"Average execution time: {avg_time:.4f} ms")

    total_bytes = 3 * a.numel() * a.element_size()
    throughput_gb_s = total_bytes / (avg_time / 1000) / 1e9
    print(f"Throughput: {throughput_gb_s:.2f} GB/s")
    print()


if __name__ == "__main__":
    M, N = 2048, 2048

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    elementwise_op(mul, a_, b_, c_)
    torch.testing.assert_close(c, mul(a, b))
    print("✓ Multiplication test passed")

    c.zero_()

    naive_elementwise_add_compiled = cute.compile(naive_elementwise_add, a_, b_, c_)
    elementwise_add_compiled = cute.compile(elementwise_op, add, a_, b_, c_)

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50 + "\n")

    print("1. Naive elementwise add kernel:")
    benchmark(partial(naive_elementwise_add_compiled, a_, b_, c_))

    print("2. PyTorch add (baseline):")
    benchmark(partial(torch.add, a, b, out=c))

    print("3. Optimized elementwise add kernel:")
    benchmark(partial(elementwise_add_compiled, a_, b_, c_))
