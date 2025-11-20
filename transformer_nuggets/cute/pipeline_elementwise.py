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


class DirectCopy(CuteOp):
    """Playing Around w/ TMA in cuteDsl"""

    def __init__(self, debug: bool = False):
        super().__init__()
        self.DEBUG = debug

    @cute.kernel
    # pyrefly: ignore  # bad-override
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
        # pyrefly: ignore  # no-matching-overload
        storage = smem.allocate(shared_storage)
        smem_layout = cute.make_layout(tiler_mn)
        sA = storage.data.get_tensor(smem_layout)
        load_atom, store_atom = copy_atoms

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        cta_coord = ((None,), bidx)

        ctaA = gA[cta_coord]
        ctaB = gB[cta_coord]
        # ctacrdB = crdB[cta_coord]

        gmem_tiled_copy_load = cute.make_tiled_copy(load_atom, tv_layout, tiler_mn)
        gmem_thr_copy_load = gmem_tiled_copy_load.get_slice(tidx)

        tAgA = gmem_thr_copy_load.partition_S(ctaA)
        tAsA = gmem_thr_copy_load.partition_D(sA)

        gmem_tiled_copy_store = cute.make_tiled_copy(store_atom, tv_layout, tiler_mn)
        gmem_thr_copy_store = gmem_tiled_copy_store.get_slice(tidx)

        tBsA = gmem_thr_copy_store.partition_S(sA)
        tBgB = gmem_thr_copy_store.partition_D(ctaB)

        # tAcA = gmem_thr_copy_load.partition_S(ctacrdB)
        # tApA = cute.make_fragment(tAcA.shape, cutlass.Boolean)
        # for i in cutlass.range(cute.size(tApA), unroll=1):
        #     tApA[i] = cute.elem_less(tAcA[i], shape)

        cute.copy(
            load_atom,
            tAgA,
            tAsA,
        )
        # cute.arch.cp_async_commit_group()
        # cute.arch.cp_async_wait_group(0)
        # cute.arch.barrier()

        if cutlass.const_expr(self.DEBUG):
            if bidx == 0 and tidx == 0:
                cute.printf("Block {} loading tile into smem\n", bidx)
                cute.print_tensor(sA)

        cute.copy(store_atom, tBsA, tBgB)

    def _get_shared_struct(
        self, dtype: cutlass.Constexpr, num_elms: cutlass.Constexpr, USE_TMA: cutlass.Constexpr
    ):
        @cute.struct
        class Shared:
            # pyrefly: ignore  # bad-specialization, not-a-type
            bar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 16]
            # pyrefly: ignore  # bad-specialization
            data: cute.struct.Align[
                # pyrefly: ignore  # bad-specialization, not-a-type
                cute.struct.MemRange[dtype, num_elms],
                # pyrefly: ignore  # not-a-type
                16,
            ]

        return Shared

    # pyrefly: ignore  # bad-return
    def get_key(self, *args, **kwargs) -> str:
        pass

    @cute.jit
    # pyrefly: ignore  # bad-function-definition
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, USE_TMA: cutlass.Constexpr = False):
        # Simple 32 threads handle 4 contiguous elements
        thr_layout = cute.make_layout(256)
        val_layout = cute.make_layout(4)
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
        smem_layout = cute.make_layout(cute.size(tiler_mn))
        # pyrefly: ignore  # index-error
        assert mA.shape[0] % val_layout.shape == 0, (
            # pyrefly: ignore  # index-error
            f"Input size {mA.shape[0]} is not divisible by vector size {val_layout.shape}"
        )

        if cutlass.const_expr(USE_TMA):
            tma_atom_load, tma_tensor_load = self.get_tma_atoms(
                cpasync.CopyBulkTensorTileG2SOp(), mA, tiler_mn
            )
            tma_atom_store, tma_tensor_store = self.get_tma_atoms(
                cpasync.CopyBulkTensorTileS2GOp(), mB, tiler_mn
            )
            copy_atoms = (tma_atom_load, tma_atom_store)
        else:
            # pyrefly: ignore  # missing-attribute
            bits_per_copy = mA.element_type.width
            copy_atom_load = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                # cpasync.CopyG2SOp(),
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

        dtype = mA.element_type
        num_elms = cute.size(smem_layout, mode=[0])
        # pyrefly: ignore  # bad-argument-type
        Shared = self._get_shared_struct(dtype, num_elms, USE_TMA)

        if cutlass.const_expr(self.DEBUG):
            print(
                f"Launching a grid of {cute.size(gA, mode=[1])} and a block of {cute.size(tv_layout, mode=[0])}"
            )
            # pyrefly: ignore  # unbound-name
            print(f"Using copy atoms with {bits_per_copy}-bit copies")
            print("Load Atom:", copy_atoms[0])
            print("Store Atom:", copy_atoms[1])
            print("Smem layout:", smem_layout)
            print("Tiler MN:", tiler_mn)
            print("TV layout:", tv_layout)

        # pyrefly: ignore  # missing-attribute, bad-argument-count
        self.kernel(gA, gB, copy_atoms, tv_layout, crdB, Shared, mB.shape, tiler_mn).launch(
            grid=[cute.size(gA, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )

    def get_tma_atoms(self, copy_op, tensor: cute.Tensor, tiler_mn: cute.Layout):
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            copy_op,
            tensor,
            tiler_mn,
        )
        return tma_atom, tma_tensor

    def get_copy_atoms(self, copy_op, tensor: cute.Tensor, tiler_mn: cute.Layout):
        # pyrefly: ignore  # missing-attribute
        copy_atom, copy_tensor = cute.make_tiled_copy_atom(
            copy_op,
            tensor,
            tiler_mn,
        )
        return copy_atom, copy_tensor


def direct_copy(gA: torch.Tensor):
    destination = torch.empty_like(gA)

    gA_ptr = gA.data_ptr()
    if gA_ptr % 16 != 0:
        raise ValueError(f"Input tensor must be 16-byte aligned, got alignment {gA_ptr % 16}")
    destination_ptr = destination.data_ptr()
    if destination_ptr % 16 != 0:
        raise ValueError(
            f"Output tensor must be 16-byte aligned, got alignment {destination_ptr % 16}"
        )

    # pyrefly: ignore  # bad-assignment
    gA = from_dlpack(gA, assumed_align=16)
    gB = from_dlpack(destination, assumed_align=16)

    op = DirectCopy(False)

    cache_key = op.get_key(gA, gB)
    compiled_kernel = compile_and_cache(op, cache_key, gA, gB)
    compiled_kernel(gA, gB)
    return destination


if __name__ == "__main__":
    set_cache_hashing(True)
    a = torch.randn(268435456, device="cuda", dtype=torch.float32)
    for _ in range(10):
        out = direct_copy(a)
        torch.testing.assert_close(out, a, atol=0.0, rtol=0.0)

    from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

    time_us = benchmark_cuda_function_in_microseconds(lambda: direct_copy(a))
    avg_time = time_us / 1e3
    print(f"Time: {avg_time:.3f} ms")
    bytes_moved = a.numel() * a.element_size() * 2
    bandwidth = bytes_moved / (avg_time / 1e3) / 1e9
    print(f"Bandwidth: {bandwidth:.2f} GB/s")

    stats = get_cache_stats()
    print(f"Cache stats: {stats}")
    print_cache()
