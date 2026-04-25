from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def _symm_mem():
    import torch.distributed._symmetric_memory as symm_mem

    return symm_mem


@cute.kernel
def _all_reduce_simple_kernel(
    inputs: list[cute.Tensor],
    output: cute.Tensor,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    local_tile_out = output[blk_coord]
    local_tile_list = [tensor[blk_coord] for tensor in inputs]

    assert all(tensor.element_type == inputs[0].element_type for tensor in inputs)

    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        inputs[0].element_type,
    )
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        inputs[0].element_type,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    thr_tensor_list = [thr_copy.partition_S(tensor) for tensor in local_tile_list]
    thr_out = thr_copy.partition_D(local_tile_out)
    frg_tensor_list = [cute.make_fragment_like(tensor) for tensor in thr_tensor_list]
    frg_acc = cute.make_fragment_like(thr_out)
    frg_acc.fill(0.0)

    for thr_tensor, frg_tensor in zip(thr_tensor_list, frg_tensor_list):
        cute.copy(copy_atom_load, thr_tensor, frg_tensor)
        frg_acc.store(frg_tensor.load() + frg_acc.load())

    cute.copy(copy_atom_store, frg_acc, thr_out)


@cute.jit
def _all_reduce_simple(
    inputs: list[cute.Tensor],
    output: cute.Tensor,
):
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    divided_inputs = [cute.zipped_divide(tensor, tiler_mn) for tensor in inputs]
    divided_output = cute.zipped_divide(output, tiler_mn)
    _all_reduce_simple_kernel(
        divided_inputs,
        divided_output,
        thr_layout,
        val_layout,
    ).launch(
        grid=[cute.size(divided_output, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def symmetric_memory_peer_tensors(
    local_tensor: torch.Tensor,
    group: dist.ProcessGroup | str | None = None,
) -> tuple[object, list[torch.Tensor]]:
    """Rendezvous a PyTorch symmetric-memory tensor and return all peer views."""
    if group is None:
        group = dist.group.WORLD
    handle = _symm_mem().rendezvous(local_tensor, group=group)
    return handle, [
        handle.get_buffer(peer_rank, local_tensor.shape, local_tensor.dtype)
        for peer_rank in range(handle.world_size)
    ]


def compile_symmetric_memory_all_reduce(
    peer_tensors: list[torch.Tensor],
    output: torch.Tensor,
):
    """Compile the simple CuTeDSL peer-load all-reduce for the given tensor layouts."""
    if output.dtype != torch.float32 or any(
        tensor.dtype != torch.float32 for tensor in peer_tensors
    ):
        raise TypeError("symmetric_memory_all_reduce currently expects float32 tensors")
    return cute.compile(
        _all_reduce_simple,
        [from_dlpack(tensor) for tensor in peer_tensors],
        from_dlpack(output),
    )


def symmetric_memory_all_reduce(
    peer_tensors: list[torch.Tensor],
    output: torch.Tensor,
    compiled=None,
):
    """Run the simple CuTeDSL all-reduce over PyTorch symmetric-memory peer tensors."""
    if compiled is None:
        compiled = compile_symmetric_memory_all_reduce(peer_tensors, output)
    compiled(
        [from_dlpack(tensor) for tensor in peer_tensors],
        from_dlpack(output),
    )
    return output


def init_torchrun_process_group() -> None:
    """Initialize CUDA and torch.distributed for `torchrun` launched examples."""
    os.environ.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "1")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")


def run_symmetric_memory_all_reduce_example(
    m: int,
    n: int,
    *,
    warmup_iterations: int = 2,
    iterations: int = 10,
    skip_ref_check: bool = False,
    benchmark: bool = False,
) -> torch.Tensor:
    """Run a torchrun-friendly CuTeDSL all-reduce example using PyTorch symmetric memory."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    if rank == 0:
        print("\nRunning CuTeDSL symmetric-memory all-reduce with:")
        print(f"Tensor dimensions: [{m}, {n}]")
        print(f"GPU count: {world_size}")

    local_tensor = _symm_mem().empty((m, n), dtype=torch.float32, device=device)
    local_tensor.random_(0, 100)
    _, peer_tensors = symmetric_memory_peer_tensors(local_tensor)
    output = torch.zeros((m, n), device=device)

    if rank == 0:
        print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    compiled = compile_symmetric_memory_all_reduce(peer_tensors, output)
    if rank == 0:
        print(f"Compilation time: {time.time() - start_time:.4f} seconds")

    if not skip_ref_check:
        dist.barrier(device_ids=[device.index])
        symmetric_memory_all_reduce(peer_tensors, output, compiled)
        dist.barrier(device_ids=[device.index])
        torch.testing.assert_close(sum(tensor.cpu() for tensor in peer_tensors), output.cpu())
        if rank == 0:
            print("Results verified successfully!")

    if not benchmark:
        return output

    for _ in range(warmup_iterations):
        symmetric_memory_all_reduce(peer_tensors, output, compiled)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        symmetric_memory_all_reduce(peer_tensors, output, compiled)
    end.record()
    end.synchronize()
    avg_time_us = start.elapsed_time(end) * 1000 / iterations

    if rank == 0:
        bytes_moved = (world_size + 1) * output.numel() * output.element_size()
        print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")
        print(f"Achieved memory throughput: {bytes_moved / (avg_time_us / 1e6) / 1e9:.2f} GB/s")
        print(f"First few elements of result:\n{output[:3, :3]}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="simple CuTeDSL all-reduce using PyTorch symmetric memory"
    )
    parser.add_argument("--M", default=1024, type=int)
    parser.add_argument("--N", default=1024, type=int)
    parser.add_argument("--warmup_iterations", default=2, type=int)
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    init_torchrun_process_group()
    try:
        run_symmetric_memory_all_reduce_example(
            args.M,
            args.N,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            skip_ref_check=args.skip_ref_check,
            benchmark=args.benchmark,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
