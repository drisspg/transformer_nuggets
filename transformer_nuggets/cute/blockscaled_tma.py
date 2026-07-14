from __future__ import annotations

import operator
from enum import Enum, IntEnum

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blockscaled_layout as blockscaled_utils

from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.profiler.ops import profile_region


class BlockScaleLayout(str, Enum):
    """Describe caller-owned block-scale storage."""

    RAW = "raw"
    SWIZZLE_32_4_4 = "swizzle_32_4_4"


class GridScheduler(str, Enum):
    """Map logical output tiles onto the launched CTA grid."""

    STATIC = "static"
    PERSISTENT = "persistent"


class WarpRole(IntEnum):
    """Static warp roles within a block-scaled TMA CTA."""

    TMA_PRODUCER = 0


class ProfileTag(IntEnum):
    """Labeled regions emitted by the block-scaled TMA profiler."""

    TMA_PROLOGUE = 0
    TMA_REFILL = 1
    TMA_WAIT = 2
    TILE_COMPUTE = 3
    EPILOGUE = 4


BLOCKSCALED_TMA_PROFILE_TAGS = tuple(tag.name.lower() for tag in ProfileTag)
DEFAULT_PERSISTENT_CTAS_PER_SM = 8


class BlockscaledTmaGemv(CuteOp):
    """Schedule M=1 block-scaled GEMV independently of its low-precision format.

    ``enable_profiling=False`` emits no timer or profile-buffer instructions. A profiled
    specialization records statically indexed regions named by :attr:`profile_tags`.
    """

    profile_tags = BLOCKSCALED_TMA_PROFILE_TAGS
    format_name = "blockscaled"

    def __init__(
        self,
        n: int,
        k: int,
        block_n: int,
        num_stages: int,
        values_per_byte: int,
        scale_block_size: int,
        tile_k_u32: int,
        scale_copy_bits: int,
        weight_scale_copy_bits: int | None = None,
        enable_profiling: bool = False,
        num_compute_warps: int = 1,
        grid_scheduler: GridScheduler = GridScheduler.STATIC,
        num_persistent_ctas: int | None = None,
        use_global_scales: bool = False,
    ):
        super().__init__()
        physical_k_bytes = k // values_per_byte
        if k % values_per_byte != 0 or physical_k_bytes % (tile_k_u32 * 4) != 0:
            raise ValueError("k must contain a whole number of physical TMA tiles")
        if k % scale_block_size != 0:
            raise ValueError("k must be divisible by the scale block size")
        if n <= 0 or block_n <= 0 or n % block_n != 0 or block_n > 32:
            raise ValueError(
                "n must be positive and block_n must divide n and be between 1 and 32"
            )
        if num_stages not in (2, 3):
            raise ValueError("num_stages must be 2 or 3")
        if num_compute_warps not in (1, 2, 4) or block_n % num_compute_warps != 0:
            raise ValueError("num_compute_warps must be 1, 2, or 4 and divide block_n")
        self.m = 1
        self.n = n
        self.k = k
        self.sf_k = k // scale_block_size
        self.block_n = block_n
        self.num_stages = num_stages
        self.tile_k_u32 = tile_k_u32
        self.num_k_tiles = physical_k_bytes // (self.tile_k_u32 * 4)
        if self.num_k_tiles < num_stages:
            raise ValueError("TMA staging requires at least one K tile per stage")
        logical_values_per_tile = self.tile_k_u32 * 4 * values_per_byte
        self.words_per_lane = tile_k_u32 // 32
        logical_values_per_lane = logical_values_per_tile // 32
        if self.words_per_lane not in (4, 8):
            raise ValueError("each lane must own four or eight packed u32 values")
        if logical_values_per_lane % scale_block_size != 0:
            raise ValueError("each lane must own a whole number of scale blocks")
        self.scale_blocks_per_lane = logical_values_per_lane // scale_block_size
        self.scale_blocks_per_tile = logical_values_per_tile // scale_block_size
        self.scale_copy_bits = scale_copy_bits
        self.weight_scale_copy_bits = weight_scale_copy_bits or scale_copy_bits
        self.use_global_scales = use_global_scales
        self.enable_profiling = enable_profiling
        self.num_compute_warps = num_compute_warps
        self.rows_per_warp = block_n // num_compute_warps
        self.num_tiles = n // block_n
        self.grid_scheduler = GridScheduler(grid_scheduler)
        if self.grid_scheduler is GridScheduler.PERSISTENT:
            if num_persistent_ctas is None or num_persistent_ctas <= 0:
                raise ValueError(
                    "num_persistent_ctas must be positive for the persistent scheduler"
                )
            max_grid_ctas = min(self.num_tiles, num_persistent_ctas)
            self.grid_ctas = next(
                grid_ctas
                for grid_ctas in range(max_grid_ctas, 0, -1)
                if self.num_tiles % grid_ctas == 0
            )
        else:
            if num_persistent_ctas is not None:
                raise ValueError("num_persistent_ctas is only valid with the persistent scheduler")
            self.grid_ctas = self.num_tiles
        self.tiles_per_cta = self.num_tiles // self.grid_ctas
        self.max_profile_events_per_cta = 2 + 3 * self.num_k_tiles
        self.num_profile_units = self.num_tiles

    @cute.jit
    def make_blocked_scale_layout(self, scale_vector_size: cutlass.Constexpr):
        """Map logical matrix coordinates onto canonical blocked scale storage."""
        return blockscaled_utils.tile_atom_to_shape_SF(
            (((self.n + 127) // 128) * 128, self.k, 1),
            scale_vector_size,
        )

    def x_smem_layout(self):
        """Return the staged layout for one physical input tile."""
        return cute.make_ordered_layout(
            (self.m, self.tile_k_u32, self.num_stages), order=(1, 0, 2)
        )

    def w_smem_layout(self):
        """Return the staged layout for one physical weight-row tile."""
        return cute.make_ordered_layout(
            (self.block_n, self.tile_k_u32, self.num_stages), order=(1, 0, 2)
        )

    def profile_scope(
        self,
        prof_buf: cute.Tensor | None,
        tag: ProfileTag,
        pid_n: cutlass.Int32,
        event_idx,
        enable_profiling: cutlass.Constexpr,
    ):
        """Build a static-slot profiling context for one CTA region."""
        return profile_region(
            prof_buf,
            cutlass.Int32(self.max_profile_events_per_cta),
            cutlass.Int32(tag),
            pid_n,
            event_idx=cutlass.Int32(event_idx),
            bounds_check=False,
            enabled=enable_profiling,
        )

    @cute.jit
    def tma_producer_load_stage(
        self,
        producer: pipeline.PipelineProducer,
        tma_atom_w: cute.CopyAtom,
        tWgW: cute.Tensor,
        tWsW: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
    ) -> pipeline.PipelineProducer:
        """Acquire one pipeline stage and issue its weight and input TMA loads."""
        stage = producer.acquire_and_advance()
        cute.copy(
            tma_atom_w,
            tWgW[(None, stage.count)],
            tWsW[(None, stage.index)],
            tma_bar_ptr=stage.barrier,
        )
        cute.copy(
            tma_atom_x,
            tXgX[(None, stage.count)],
            tXsX[(None, stage.index)],
            tma_bar_ptr=stage.barrier,
        )
        stage.commit()
        return producer

    @cute.jit
    def decode_lane_values(self, raw_values: cute.Tensor):
        """Decode one lane's packed values into format-specific FP32 groups."""
        raise NotImplementedError

    @cute.jit
    def load_scale_values(
        self,
        scale_tensor: cute.Tensor,
        row,
        scale_k,
        scale_atom: cute.CopyAtom,
        scale_layout: cute.Layout,
    ):
        """Load the format-specific block scales for one lane."""
        raise NotImplementedError

    @cute.jit
    def prepare_weight_scale_values(
        self,
        scale_tensor: cute.Tensor,
        row_start,
        k_tile: cutlass.Constexpr,
        lane: cutlass.Int32,
        scale_atom: cute.CopyAtom,
        scale_layout: cute.Layout,
    ):
        """Optionally preload format-specific weight scales shared across row accumulation."""
        return None

    @cute.jit
    def load_prepared_weight_scale_values(
        self,
        prepared_scales,
        scale_tensor: cute.Tensor,
        row,
        scale_k,
        local_row: cutlass.Constexpr,
        lane: cutlass.Int32,
        scale_atom: cute.CopyAtom,
        scale_layout: cute.Layout,
    ):
        """Load one row's scales directly when the format has no prepared representation."""
        return self.load_scale_values(
            scale_tensor,
            row,
            scale_k,
            scale_atom,
            scale_layout,
        )

    @cute.jit
    def accumulate_scaled_products(
        self,
        accumulator,
        x_values,
        w_values,
        input_scales,
        weight_scales,
    ):
        """Apply block scales to one lane's products and update its accumulator."""
        raise NotImplementedError

    @cute.jit
    def load_lane_words(
        self,
        smem: cute.Tensor,
        row,
        lane: cutlass.Int32,
        stage,
        smem_atom: cute.CopyAtom,
        chunk_layout: cute.Layout,
    ):
        """Load one lane's packed 128-bit chunks into registers."""
        raw_values = cute.make_rmem_tensor((1, self.words_per_lane), cutlass.Uint32)
        if cutlass.const_expr(self.words_per_lane == 4):
            col = cute.assume(lane * 4, divby=4)
            cute.copy(
                smem_atom,
                cute.make_tensor(
                    smem.iterator + cute.assume(smem.layout((row, col, stage)), divby=4),
                    chunk_layout,
                ),
                cute.make_tensor(raw_values.iterator, chunk_layout),
            )
        else:
            chunk_a = lane & 4
            chunk_b = (lane ^ 4) & 4
            col_a = cute.assume(lane * 8 + chunk_a, divby=4)
            col_b = cute.assume(lane * 8 + chunk_b, divby=4)
            cute.copy(
                smem_atom,
                cute.make_tensor(
                    smem.iterator + cute.assume(smem.layout((row, col_a, stage)), divby=4),
                    chunk_layout,
                ),
                cute.make_tensor(
                    raw_values.iterator,
                    chunk_layout,
                ),
            )
            cute.copy(
                smem_atom,
                cute.make_tensor(
                    smem.iterator + cute.assume(smem.layout((row, col_b, stage)), divby=4),
                    chunk_layout,
                ),
                cute.make_tensor(
                    raw_values.iterator + 4,
                    chunk_layout,
                ),
            )
        return raw_values

    @cute.jit
    def load_lane_values(
        self,
        smem: cute.Tensor,
        row,
        lane: cutlass.Int32,
        stage,
        smem_atom: cute.CopyAtom,
        chunk_layout: cute.Layout,
    ):
        """Load and decode one lane's packed values."""
        return self.decode_lane_values(
            self.load_lane_words(smem, row, lane, stage, smem_atom, chunk_layout)
        )

    @cute.jit
    def load_weight_lane_values(
        self,
        smem: cute.Tensor,
        row,
        lane: cutlass.Int32,
        stage,
        smem_atom: cute.CopyAtom,
        chunk_layout: cute.Layout,
    ):
        """Load weight values using the format's default decode path."""
        return self.load_lane_values(smem, row, lane, stage, smem_atom, chunk_layout)

    @cute.jit
    def compute_warp_consume_stage(
        self,
        consumer: pipeline.PipelineConsumer,
        accumulators: list,
        k_tile: cutlass.Constexpr,
        lane: cutlass.Int32,
        owned_row_start: cutlass.Int32,
        n0: cutlass.Int32,
        pid_n: cutlass.Int32,
        sX: cute.Tensor,
        sW: cute.Tensor,
        mSFX: cute.Tensor,
        mSFW: cute.Tensor,
        smem_atom: cute.CopyAtom,
        input_scale_atom: cute.CopyAtom,
        weight_scale_atom: cute.CopyAtom,
        chunk_layout: cute.Layout,
        scale_layout: cute.Layout,
        prof_buf: cute.Tensor | None,
        enable_profiling: cutlass.Constexpr,
    ) -> tuple[pipeline.PipelineConsumer, list]:
        """Wait for one stage and accumulate the rows owned by this compute warp."""
        scale_k = lane * self.scale_blocks_per_lane + k_tile * self.scale_blocks_per_tile
        with self.profile_scope(
            prof_buf,
            ProfileTag.TMA_WAIT,
            pid_n,
            2 + 3 * k_tile,
            enable_profiling,
        ):
            input_scales = self.load_scale_values(
                mSFX,
                0,
                scale_k,
                input_scale_atom,
                scale_layout,
            )
            prepared_weight_scales = self.prepare_weight_scale_values(
                mSFW,
                n0 + owned_row_start,
                k_tile,
                lane,
                weight_scale_atom,
                scale_layout,
            )
            full = consumer.wait_and_advance()

        with self.profile_scope(
            prof_buf,
            ProfileTag.TILE_COMPUTE,
            pid_n,
            3 + 3 * k_tile,
            enable_profiling,
        ):
            x_values = self.load_lane_values(
                sX,
                0,
                lane,
                full.index,
                smem_atom,
                chunk_layout,
            )
            for local_row in cutlass.range_constexpr(self.rows_per_warp):
                cta_row = owned_row_start + local_row
                global_row = n0 + cta_row
                w_values = self.load_weight_lane_values(
                    sW,
                    cta_row,
                    lane,
                    full.index,
                    smem_atom,
                    chunk_layout,
                )
                weight_scales = self.load_prepared_weight_scale_values(
                    prepared_weight_scales,
                    mSFW,
                    global_row,
                    scale_k,
                    local_row,
                    lane,
                    weight_scale_atom,
                    scale_layout,
                )
                accumulators[local_row] = self.accumulate_scaled_products(
                    accumulators[local_row],
                    x_values,
                    w_values,
                    input_scales,
                    weight_scales,
                )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            full.release()
        return consumer, accumulators

    @cute.jit
    def compute_warp_store_output(
        self,
        accumulators: list,
        lane: cutlass.Int32,
        owned_row_start: cutlass.Int32,
        n0: cutlass.Int32,
        mGFW: cute.Tensor | None,
        mGFX: cute.Tensor | None,
        mO: cute.Tensor,
    ) -> None:
        """Reduce and store the output rows owned by this compute warp."""
        for local_row in cutlass.range_constexpr(self.rows_per_warp):
            accumulators[local_row] = cute.arch.warp_reduction(
                accumulators[local_row], operator.add
            )
        if lane == 0:
            output_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(self.use_global_scales):
                assert mGFX is not None and mGFW is not None
                output_scale = mGFX[0] * mGFW[0]
            for local_row in cutlass.range_constexpr(self.rows_per_warp):
                cta_row = owned_row_start + local_row
                mO[0, n0 + cta_row] = (accumulators[local_row] * output_scale).to(cutlass.BFloat16)

    @cute.kernel
    def kernel(
        self,
        tma_atom_w: cute.CopyAtom,
        mW_u32: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX_u32: cute.Tensor,
        mSFW: cute.Tensor,
        mSFX: cute.Tensor,
        mGFW: cute.Tensor | None,
        mGFX: cute.Tensor | None,
        mO: cute.Tensor,
        prof_buf: cute.Tensor | None,
        enable_profiling: cutlass.Constexpr,
    ):
        """Schedule logical N tiles across static or persistent CTA grids."""
        tidx, _, _ = cute.arch.thread_idx()
        physical_cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = tidx % 32
        owned_row_start = warp * self.rows_per_warp
        if cutlass.const_expr(enable_profiling):
            assert prof_buf is not None
        chunk_layout = cute.make_ordered_layout((1, 4), order=(1, 0))
        scale_layout = cute.make_layout(1)
        smem_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint32,
            num_bits_per_copy=128,
        )
        input_scale_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint8,
            num_bits_per_copy=self.scale_copy_bits,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_LAST,
        )
        weight_scale_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint8,
            num_bits_per_copy=self.weight_scale_copy_bits,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
        )

        smem = cutlass.utils.SmemAllocator()
        barriers = smem.allocate_array(cutlass.Int64, self.num_stages * 2, byte_alignment=8)
        sX = smem.allocate_tensor(cutlass.Uint32, self.x_smem_layout(), byte_alignment=128)
        sW = smem.allocate_tensor(cutlass.Uint32, self.w_smem_layout(), byte_alignment=128)

        if warp == WarpRole.TMA_PRODUCER:
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_x)
        producer, consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_compute_warps
            ),
            tx_count=(self.block_n + self.m) * self.tile_k_u32 * 4,
            barrier_storage=barriers,
            tidx=lane,
        ).make_participants()
        gX = cute.local_tile(mX_u32, (self.m, self.tile_k_u32), (0, None))
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )

        for tile_round in cutlass.range(self.tiles_per_cta, unroll=1):
            producer.reset()
            consumer.reset()
            pid_n = physical_cta + tile_round * self.grid_ctas
            n0 = pid_n * self.block_n
            gW = cute.local_tile(
                mW_u32,
                (self.block_n, self.tile_k_u32),
                (pid_n, None),
            )
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w,
                0,
                cute.make_layout(1),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(gW, 0, 2),
            )

            with self.profile_scope(
                prof_buf,
                ProfileTag.TMA_PROLOGUE,
                pid_n,
                0,
                enable_profiling,
            ):
                if warp == WarpRole.TMA_PRODUCER:
                    producer = self.tma_producer_load_stage(
                        producer,
                        tma_atom_w,
                        tWgW,
                        tWsW,
                        tma_atom_x,
                        tXgX,
                        tXsX,
                    )

            accumulators = [cutlass.Float32(0.0) for _ in range(self.rows_per_warp)]
            for k_tile in cutlass.range_constexpr(self.num_k_tiles):
                if warp == WarpRole.TMA_PRODUCER and k_tile < self.num_k_tiles - 1:
                    with self.profile_scope(
                        prof_buf,
                        ProfileTag.TMA_REFILL,
                        pid_n,
                        1 + 3 * k_tile,
                        enable_profiling,
                    ):
                        producer = self.tma_producer_load_stage(
                            producer,
                            tma_atom_w,
                            tWgW,
                            tWsW,
                            tma_atom_x,
                            tXgX,
                            tXsX,
                        )

                consumer, accumulators = self.compute_warp_consume_stage(
                    consumer,
                    accumulators,
                    k_tile,
                    lane,
                    owned_row_start,
                    n0,
                    pid_n,
                    sX,
                    sW,
                    mSFX,
                    mSFW,
                    smem_atom,
                    input_scale_atom,
                    weight_scale_atom,
                    chunk_layout,
                    scale_layout,
                    prof_buf,
                    enable_profiling,
                )

            with self.profile_scope(
                prof_buf,
                ProfileTag.EPILOGUE,
                pid_n,
                1 + 3 * self.num_k_tiles,
                enable_profiling,
            ):
                self.compute_warp_store_output(
                    accumulators,
                    lane,
                    owned_row_start,
                    n0,
                    mGFW,
                    mGFX,
                    mO,
                )

        if warp == WarpRole.TMA_PRODUCER:
            producer.tail()

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,
        mX: cute.Tensor,
        mSFW: cute.Tensor,
        mSFX: cute.Tensor,
        mGFW: cute.Tensor | None,
        mGFX: cute.Tensor | None,
        mO: cute.Tensor,
        prof_buf: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        """Construct TMA descriptors and launch the kernel."""
        mW_u32 = cute.recast_tensor(mW, cutlass.Uint32)
        mX_u32 = cute.recast_tensor(mX, cutlass.Uint32)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_op,
            mW_u32,
            cute.select(self.w_smem_layout(), mode=[0, 1]),
            (self.block_n, self.tile_k_u32),
        )
        tma_atom_x, tma_tensor_x = cpasync.make_tiled_tma_atom(
            tma_op,
            mX_u32,
            cute.select(self.x_smem_layout(), mode=[0, 1]),
            (self.m, self.tile_k_u32),
        )
        name = self.get_name()
        self.kernel(
            tma_atom_w,
            tma_tensor_w,
            tma_atom_x,
            tma_tensor_x,
            mSFW,
            mSFX,
            mGFW,
            mGFX,
            mO,
            prof_buf,
            self.enable_profiling,
            _name_prefix=name,
        ).launch(
            grid=[self.grid_ctas, 1, 1],
            block=[self.num_compute_warps * 32, 1, 1],
            stream=stream,
        )

    def get_key(self) -> str:
        """Return the static kernel specialization key."""
        return (
            f"{self.format_name}_{self.n}_{self.k}_{self.block_n}_{self.num_stages}"
            f"_cw={self.num_compute_warps}_scheduler={self.grid_scheduler.value}"
            f"_grid={self.grid_ctas}_global={self.use_global_scales}"
            f"_profile={self.enable_profiling}"
        )

    def get_name(self) -> str:
        """Return the compiled kernel name."""
        profile_suffix = "_profiled" if self.enable_profiling else ""
        global_suffix = "_global" if self.use_global_scales else ""
        return (
            f"{self.format_name}_tma_gemv_n{self.n}_k{self.k}"
            f"_bn{self.block_n}_s{self.num_stages}_cw{self.num_compute_warps}"
            f"_{self.grid_scheduler.value}_grid{self.grid_ctas}"
            f"{global_suffix}{profile_suffix}"
        )
