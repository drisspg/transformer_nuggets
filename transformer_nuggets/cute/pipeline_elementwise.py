import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds
from transformer_nuggets.cute.cache import (
    compile_and_cache,
    get_cache_stats,
    print_cache,
    set_cache_hashing,
)
from transformer_nuggets.cute.base import CuteOp
from rich import print
from transformer_nuggets import init_logging
import logging


init_logging(logging.INFO)


def _pointer_to_int(pointer) -> int:
    """Convert cute pointer wrappers into raw integers for alignment checks."""
    if isinstance(pointer, int):
        return pointer
    if hasattr(pointer, "_pointer"):
        return pointer._pointer
    if hasattr(pointer, "value"):
        return pointer.value
    raise TypeError(f"Unsupported pointer type {type(pointer)!r}")


class DirectCopy(CuteOp[[torch.Tensor], torch.Tensor]):
    """Playing Around w/ TMA in cuteDsl"""

    def __init__(self, use_tma: bool = False, num_stages: int = 1):
        super().__init__()
        self.num_stages = 1

    @cute.kernel
    def kernel_tma(
        self,
        tma_tensor_load: cute.Tensor,
        tma_tensor_store: cute.Tensor,
        load_atom: cute.CopyAtom,
        store_atom: cute.CopyAtom,
        shared_storage: cutlass.Constexpr,
        tile_size: cutlass.Constexpr,
        dtype: cutlass.Constexpr,
    ):
        smem = cutlass.utils.SmemAllocator()

        storage = smem.allocate(shared_storage)
        smem_layout = cute.make_layout(tile_size)
        sA = storage.data.get_tensor(smem_layout)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tiler = cute.make_layout(tile_size)
        gA_tiled = cute.zipped_divide(tma_tensor_load, tiler)
        gB_tiled = cute.zipped_divide(tma_tensor_store, tiler)

        tAsA_load, tAgA = cpasync.tma_partition(
            load_atom,
            0,
            cute.make_layout(1),
            sA,
            gA_tiled[(None, bidx)],
        )
        tAsA_store, tBgB = cpasync.tma_partition(
            store_atom,
            0,
            cute.make_layout(1),
            sA,
            gB_tiled[(None, bidx)],
        )

        load_mbar_ptr = storage.load_mbar_ptr.data_ptr()

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(load_mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        tma_load_bytes = cute.size_in_bytes(dtype, smem_layout)

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(load_mbar_ptr, tma_load_bytes)

        if warp_idx == 0:
            cute.copy(load_atom, tAgA, tAsA_load, tma_bar_ptr=load_mbar_ptr)

        cute.arch.mbarrier_wait(load_mbar_ptr, 0)
        if warp_idx == 1:
            cute.copy(store_atom, tAsA_store, tBgB)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        copy_atoms: tuple[cute.CopyAtom, cute.CopyAtom],
        tv_layout: cute.Layout,
        crdB: cute.Tensor,
        shared_storage: cutlass.Constexpr,
        shape: cute.Shape,
        tiler_mn: cutlass.Constexpr,
    ):
        smem = cutlass.utils.SmemAllocator()

        storage = smem.allocate(shared_storage)
        smem_layout = cute.make_layout(tiler_mn)
        sA = storage.data.get_tensor(smem_layout)
        load_atom, store_atom = copy_atoms

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        cta_coord = ((None,), bidx)

        ctaA = gA[cta_coord]
        ctaB = gB[cta_coord]

        gmem_tiled_copy_load = cute.make_tiled_copy(load_atom, tv_layout, tiler_mn)
        gmem_thr_copy_load = gmem_tiled_copy_load.get_slice(tidx)

        tAgA = gmem_thr_copy_load.partition_S(ctaA)
        tAsA = gmem_thr_copy_load.partition_D(sA)

        gmem_tiled_copy_store = cute.make_tiled_copy(store_atom, tv_layout, tiler_mn)
        gmem_thr_copy_store = gmem_tiled_copy_store.get_slice(tidx)

        tBsA = gmem_thr_copy_store.partition_S(sA)
        tBgB = gmem_thr_copy_store.partition_D(ctaB)

        cute.copy(load_atom, tAgA, tAsA)
        cute.copy(store_atom, tBsA, tBgB)

    def _get_shared_struct(self, dtype: cutlass.Constexpr, num_elms: cutlass.Constexpr):
        @cute.struct
        class SharedStorage:
            load_mbar_ptr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, self.num_stages], 8
            ]
            store_mbar_ptr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, self.num_stages], 8
            ]
            data: cute.struct.Align[cute.struct.MemRange[dtype, num_elms], 128]

        return SharedStorage

    def get_key(self, *args, **kwargs) -> str:
        key_parts = [self.__class__.__name__]
        for arg in args:
            if isinstance(arg, cute.Tensor):
                key_parts.append(self._generate_tensor_key(arg))
        if "use_tma" in kwargs:
            key_parts.append(f"use_tma={kwargs['use_tma']}")
        return "_".join(key_parts)

    def interface(self, gA: torch.Tensor, *, use_tma: bool = False) -> torch.Tensor:
        destination = torch.empty_like(gA)

        gA_ptr = gA.data_ptr()
        if gA_ptr % 16 != 0:
            raise ValueError(f"Input tensor must be 16-byte aligned, got alignment {gA_ptr % 16}")
        destination_ptr = destination.data_ptr()
        if destination_ptr % 16 != 0:
            raise ValueError(
                f"Output tensor must be 16-byte aligned, got alignment {destination_ptr % 16}"
            )

        gA_cute = from_dlpack(gA, assumed_align=16)
        gB_cute = from_dlpack(destination, assumed_align=16)

        compile_and_cache(
            self, self.get_key(gA_cute, gB_cute, use_tma=use_tma), gA_cute, gB_cute, use_tma
        )(gA_cute, gB_cute)
        return destination

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, USE_TMA: cutlass.Constexpr = False):
        if cutlass.const_expr(USE_TMA):
            self.call_tma(mA, mB)
        else:
            self.call_tv(mA, mB)

    def get_tma_atoms(
        self, copy_op, gmem_tensor: cute.Tensor, smem_layout: cute.Layout, tiler_mn: cute.Layout
    ):
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            copy_op,
            gmem_tensor,
            smem_layout,
            tiler_mn,
        )
        return tma_atom, tma_tensor

    def get_copy_atoms(self, copy_op, tensor: cute.Tensor, tiler_mn: cute.Layout):
        copy_atom, copy_tensor = cute.make_tiled_copy_atom(
            copy_op,
            tensor,
            tiler_mn,
        )
        return copy_atom, copy_tensor

    @cute.jit
    def call_tma(self, mA: cute.Tensor, mB: cute.Tensor):
        num_warps = 2
        load_store_size = 8192 * 3
        num_threads = num_warps * 32
        total_size = mA.shape[0]
        # 1 block gets a load store size chunk
        num_blocks = cute.ceil_div(total_size, load_store_size)

        smem_layout_tma = cute.make_layout(load_store_size)
        # define the op, pass in global tensor, and flat smemlayout, tiler is 1d
        tma_atom_load, tma_tensor_load = self.get_tma_atoms(
            cpasync.CopyBulkTensorTileG2SOp(), mA, smem_layout_tma, load_store_size
        )
        tma_atom_store, tma_tensor_store = self.get_tma_atoms(
            cpasync.CopyBulkTensorTileS2GOp(), mB, smem_layout_tma, load_store_size
        )
        Shared = self._get_shared_struct(mA.element_type, load_store_size)
        dtype = mA.element_type
        self.kernel_tma(
            tma_tensor_load,
            tma_tensor_store,
            tma_atom_load,
            tma_atom_store,
            Shared,
            load_store_size,
            dtype,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[num_threads, 1, 1],
            cluster=[1, 1, 1],
        )

    @cute.jit
    def call_tv(self, mA: cute.Tensor, mB: cute.Tensor):
        # Simple 32 threads handle 4 contiguous elements
        thr_layout = cute.make_layout(256)
        val_layout = cute.make_layout(4)
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
        smem_layout = cute.make_layout(cute.size(tiler_mn))

        assert mA.shape[0] % val_layout.shape == 0, (
            f"Input size {mA.shape[0]} is not divisible by vector size {val_layout.shape}"
        )
        dtype = mA.element_type
        num_elms = cute.size(smem_layout, mode=[0])
        Shared = self._get_shared_struct(dtype, num_elms)

        num_threads = cute.size(tv_layout, mode=[0])
        bits_per_copy = mA.element_type.width
        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mA.element_type,
            num_bits_per_copy=bits_per_copy,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mB.element_type,
            num_bits_per_copy=bits_per_copy,
        )
        copy_atoms = (copy_atom_load, copy_atom_store)

        gA = cute.zipped_divide(mA, tiler_mn)
        gB = cute.zipped_divide(mB, tiler_mn)

        idB = cute.make_identity_tensor(mB.shape)
        crdB = cute.zipped_divide(idB, tiler_mn)

        self.kernel(gA, gB, copy_atoms, tv_layout, crdB, Shared, mB.shape, tiler_mn).launch(
            grid=[cute.size(gA, mode=[1]), 1, 1], block=[num_threads, 1, 1]
        )


def direct_copy(gA: torch.Tensor, *, use_tma: bool = False):
    return DirectCopy().interface(gA, use_tma=use_tma)


def main(use_tma: bool = False):
    set_cache_hashing(True)
    a = torch.randn(268435456, device="cuda", dtype=torch.float32)
    for _ in range(10):
        out = direct_copy(a, use_tma=use_tma)
        torch.testing.assert_close(out, a, atol=0.0, rtol=0.0)
    print("All tests passed")

    time_us = benchmark_cuda_function_in_microseconds(
        lambda: direct_copy(
            a,
            use_tma=use_tma,
        )
    )
    avg_time = time_us / 1e3
    print(f"Time: {avg_time:.3f} ms")
    bytes_moved = a.numel() * a.element_size() * 2
    bandwidth = bytes_moved / (avg_time / 1e3) / 1e9
    print(f"Bandwidth: {bandwidth:.2f} GB/s")

    stats = get_cache_stats()
    print(f"Cache stats: {stats}")
    print_cache()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
