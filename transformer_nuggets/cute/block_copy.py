from enum import IntEnum

import torch

import cutlass
import cutlass.cute as cute
from cutlass import pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.block import block_copy

from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.cache import compile_and_cache
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds


class TmaRowCopy(CuteOp[[torch.Tensor], torch.Tensor]):
    def __init__(self, num_stages: int = 1, rows_per_cta: int | None = None):
        super().__init__()
        self.num_stages = num_stages
        self.rows_per_cta = rows_per_cta or 4 * num_stages

    class WarpRoles(IntEnum):
        INIT = 0
        LOAD = INIT
        STORE = 1

    def _get_shared_struct(self, dtype: cutlass.Constexpr, num_elms: cutlass.Constexpr):
        @cute.struct
        class SharedStorage:
            load_mbar_ptr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages], 8
            ]
            data: cute.struct.Align[cute.struct.MemRange[dtype, num_elms], 128]

        return SharedStorage

    @cute.kernel
    def kernel(
        self,
        tma_tensor_load: cute.Tensor,
        tma_tensor_store: cute.Tensor,
        load_atom: cute.CopyAtom,
        store_atom: cute.CopyAtom,
        shared_storage: cutlass.Constexpr,
        tile_shape: cutlass.Constexpr,
        dtype: cutlass.Constexpr,
        row_count: cutlass.Constexpr,
    ):
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage)
        smem_layout = cute.make_ordered_layout(tile_shape, order=(2, 1, 0))
        staged_smem_layout = cute.make_ordered_layout(
            (self.num_stages, *tile_shape), order=(3, 2, 1, 0)
        )
        smem_tile = storage.data.get_tensor(staged_smem_layout)

        batch_idx, row_block_idx, _ = cute.arch.block_idx()
        row_base = row_block_idx * self.rows_per_cta
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        load_mbar_ptr = storage.load_mbar_ptr.data_ptr()

        src_tiles = cute.zipped_divide(tma_tensor_load, tile_shape)
        dst_tiles = cute.zipped_divide(tma_tensor_store, tile_shape)

        if warp_idx == self.WarpRoles.INIT:
            cpasync.prefetch_descriptor(load_atom)
            cpasync.prefetch_descriptor(store_atom)

        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=cute.size_in_bytes(dtype, smem_layout),
            barrier_storage=load_mbar_ptr,
        )

        if warp_idx == self.WarpRoles.LOAD:
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            for row_step in cutlass.range(self.rows_per_cta, unroll=1):
                load_pipeline.producer_acquire(producer_state)
                src_tile = src_tiles[(None, (batch_idx, row_base + row_step, 0))]
                stage_smem_tile = smem_tile[(producer_state.index, None, None, None)]
                block_copy(
                    load_atom,
                    cute.group_modes(src_tile, 0, 1),
                    cute.group_modes(stage_smem_tile, 0, 3),
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)

        if warp_idx == self.WarpRoles.STORE:
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            for row_step in cutlass.range(self.rows_per_cta, unroll=1):
                load_pipeline.consumer_wait(consumer_state)
                stage_smem_tile = smem_tile[(consumer_state.index, None, None, None)]
                # Tail CTAs may TMA-load full-OOB rows as zero-filled stages; skip their stores.
                if row_base + row_step < row_count:
                    dst_tile = dst_tiles[(None, (batch_idx, row_base + row_step, 0))]
                    block_copy(
                        store_atom,
                        cute.group_modes(stage_smem_tile, 0, 3),
                        cute.group_modes(dst_tile, 0, 1),
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

    @cute.jit
    def __call__(self, src: cute.Tensor, dst: cute.Tensor):
        tile_shape = (1, 1, src.shape[2])
        smem_layout = cute.make_ordered_layout(tile_shape, order=(2, 1, 0))
        load_atom, tma_tensor_load = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            src,
            smem_layout,
            tile_shape,
        )
        store_atom, tma_tensor_store = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dst,
            smem_layout,
            tile_shape,
        )
        shared_storage = self._get_shared_struct(
            src.element_type, self.num_stages * cute.size(smem_layout)
        )
        threads = 2 * cute.arch.WARP_SIZE
        self.kernel(
            tma_tensor_load,
            tma_tensor_store,
            load_atom,
            store_atom,
            shared_storage,
            tile_shape,
            src.element_type,
            src.shape[1],
        ).launch(
            grid=[src.shape[0], cute.ceil_div(src.shape[1], self.rows_per_cta), 1],
            block=[threads, 1, 1],
            cluster=[1, 1, 1],
        )

    def get_key(self, *args, **kwargs) -> str:
        key_parts = [
            self.__class__.__name__,
            f"stages_{self.num_stages}",
            f"rows_per_cta_{self.rows_per_cta}",
        ]
        for arg in args:
            if isinstance(arg, cute.Tensor):
                key_parts.append(self._generate_tensor_key(arg))
        return "_".join(key_parts)

    def interface(self, src: torch.Tensor) -> torch.Tensor:
        if src.ndim != 3:
            raise ValueError(f"Expected a 3D tensor, got rank {src.ndim}")
        if src.data_ptr() % 16 != 0:
            raise ValueError("Input tensor must be 16-byte aligned")
        if src.shape[2] * src.element_size() % 16 != 0:
            raise ValueError("Input innermost dimension must span a multiple of 16 bytes")
        dst = torch.empty_like(src)
        if dst.data_ptr() % 16 != 0:
            raise ValueError("Output tensor must be 16-byte aligned")

        src_cute = from_dlpack(src, assumed_align=16)
        dst_cute = from_dlpack(dst, assumed_align=16)
        compile_and_cache(
            self,
            self.get_key(src_cute, dst_cute),
            src_cute,
            dst_cute,
        )(src_cute, dst_cute)
        return dst


def tma_row_copy(
    src: torch.Tensor, num_stages: int = 1, rows_per_cta: int | None = None
) -> torch.Tensor:
    return TmaRowCopy(num_stages=num_stages, rows_per_cta=rows_per_cta).interface(src)


def main():
    M, K, N = 1024, 1024, 6 * 1024
    src = torch.arange(M * K * N, device="cuda", dtype=torch.float32).reshape(M, K, N)
    load_store_bytes = src.numel() * src.element_size() * 2
    rows_per_cta = 8
    for num_stages in (1, 2, 3, 4):
        dst = tma_row_copy(src, num_stages=num_stages, rows_per_cta=rows_per_cta)
        torch.testing.assert_close(dst, src)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda: tma_row_copy(src, num_stages=num_stages, rows_per_cta=rows_per_cta)
        )
        bandwidth_tbps = load_store_bytes / (time_us * 1e-6) / 1e12
        print(
            f"TMA block_copy stages={num_stages}, rows_per_cta={rows_per_cta}: "
            f"latency={time_us:.2f} us, bandwidth={bandwidth_tbps:.3f} TB/s"
        )

    torch_dst = torch.empty_like(src).copy_(src)
    torch.testing.assert_close(torch_dst, src)
    torch_time_us = benchmark_cuda_function_in_microseconds(
        lambda: torch.empty_like(src).copy_(src)
    )
    torch_bandwidth_tbps = load_store_bytes / (torch_time_us * 1e-6) / 1e12
    print(
        f"torch empty_like + copy_: latency={torch_time_us:.2f} us, "
        f"bandwidth={torch_bandwidth_tbps:.3f} TB/s"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("A Hopper or newer CUDA GPU is required for TMA block copies")
    main()
