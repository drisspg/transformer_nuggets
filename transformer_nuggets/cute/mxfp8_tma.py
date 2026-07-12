"""TMA-staged MXFP8 GEMV for Blackwell GPUs."""

from __future__ import annotations

import operator
from functools import cache
from pathlib import Path

import torch
import typer

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import cpasync, tcgen05

from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.cache import compile_tvm_ffi_and_cache
from transformer_nuggets.cute.profiler import group_by_unit, profile_session
from transformer_nuggets.cute.profiler.ops import profile_region
from transformer_nuggets.cute.utils import fake_stream, make_fake_compact_tensor


TMA_PRODUCER_WARP = 0
TAG_TMA_PROLOGUE = 0
TAG_TMA_REFILL = 1
TAG_TMA_WAIT = 2
TAG_TILE_COMPUTE = 3
TAG_EPILOGUE = 4
MXFP8_TMA_PROFILE_TAGS = (
    "tma_prologue",
    "tma_refill",
    "tma_wait",
    "tile_compute",
    "epilogue",
)


@cute.jit
def combined_e8m0_to_f32(a, b):
    """Combine two E8M0 exponents before converting to float32."""
    exponent = cutlass.Int32(a) + cutlass.Int32(b) - 254
    bits = cutlass.Uint32(0)
    if exponent >= -126:
        bits = cutlass.Uint32(exponent + 127) << 23
        if exponent > 127:
            bits = cutlass.Uint32(0x7F800000)
    else:
        if exponent >= -149:
            bits = cutlass.Uint32(1) << cutlass.Uint32(exponent + 149)
    if a == 0xFF or b == 0xFF:
        bits = cutlass.Uint32(0x7F800001)
    return bits.bitcast(cutlass.Float32)


class Mxfp8TmaGemv(CuteOp):
    """Compute raw-layout MXFP8 GEMV with optional compile-time region profiling.

    ``enable_profiling=False`` emits no timer or profile-buffer instructions. A profiled
    specialization records statically indexed regions named by :attr:`profile_tags`.
    """

    profile_tags = MXFP8_TMA_PROFILE_TAGS

    def __init__(
        self,
        n: int,
        k: int,
        block_n: int,
        num_stages: int,
        enable_profiling: bool = False,
    ):
        super().__init__()
        m = 1
        if k % 1024 != 0:
            raise ValueError("k must be divisible by 1024")
        sf_k = k // 32
        if n <= 0 or block_n <= 0 or n % block_n != 0 or block_n > 32:
            raise ValueError(
                "n must be positive and block_n must divide n and be between 1 and 32"
            )
        if num_stages not in (2, 3):
            raise ValueError("num_stages must be 2 or 3")
        if sf_k < num_stages * 32:
            raise ValueError("TMA staging requires at least one K tile per stage")
        self.m = m
        self.n = n
        self.sf_k = sf_k
        self.block_n = block_n
        self.num_stages = num_stages
        self.tile_k_u32 = 256
        self.num_k_tiles = sf_k // 32
        self.enable_profiling = enable_profiling
        self.max_profile_events_per_cta = 2 + 3 * self.num_k_tiles
        self.num_profile_units = n // block_n

    def x_smem_layout(self):
        """Return the staged layout for M 1024-byte input tiles."""
        return cute.make_ordered_layout(
            (self.m, self.tile_k_u32, self.num_stages), order=(1, 0, 2)
        )

    def w_smem_layout(self):
        """Return the staged layout for one weight-row tile."""
        return cute.make_ordered_layout(
            (self.block_n, self.tile_k_u32, self.num_stages), order=(1, 0, 2)
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_w: cute.CopyAtom,
        mW_u32: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX_u32: cute.Tensor,
        mSFW: cute.Tensor,
        mSFX: cute.Tensor,
        mO: cute.Tensor,
        prof_buf: cute.Tensor | None,
        enable_profiling: cutlass.Constexpr,
    ):
        """Overlap TMA production with typed FP8 conversion and accumulation."""
        tidx, _, _ = cute.arch.thread_idx()
        pid_n, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = tidx % 32
        n0 = pid_n * self.block_n
        max_profile_events = cutlass.Int32(self.max_profile_events_per_cta)
        if cutlass.const_expr(enable_profiling):
            assert prof_buf is not None
        chunk_layout = cute.make_ordered_layout((1, 4), order=(1, 0))
        # Each lane owns u32s [8*lane, 8*lane+8) (one 32-value scale block) as
        # two 16-byte LDS.128 chunks. Loading them low-first for lanes 0-3 and
        # high-first for lanes 4-7 of each octet spreads every load across all
        # 32 banks, removing the 2x shared-load bank conflict.
        col_a = cute.assume(lane * 8 + (lane & 4), divby=4)
        col_b = cute.assume(lane * 8 + ((lane ^ 4) & 4), divby=4)
        scale_layout = cute.make_layout(1)
        smem_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint32,
            num_bits_per_copy=128,
        )
        input_scale_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint8,
            num_bits_per_copy=8,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_LAST,
        )
        weight_scale_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint8,
            num_bits_per_copy=8,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
        )

        smem = cutlass.utils.SmemAllocator()
        barriers = smem.allocate_array(cutlass.Int64, self.num_stages * 2, byte_alignment=8)
        sX = smem.allocate_tensor(cutlass.Uint32, self.x_smem_layout(), byte_alignment=128)
        sW = smem.allocate_tensor(cutlass.Uint32, self.w_smem_layout(), byte_alignment=128)

        if warp == TMA_PRODUCER_WARP:
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_x)
        producer, consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.m),
            tx_count=(self.block_n + self.m) * self.tile_k_u32 * 4,
            barrier_storage=barriers,
            tidx=lane,
        ).make_participants()

        gW = cute.local_tile(
            mW_u32,
            (self.block_n, self.tile_k_u32),
            (pid_n, None),
        )
        gX = cute.local_tile(mX_u32, (self.m, self.tile_k_u32), (0, None))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            cute.make_layout(1),
            cute.group_modes(sW, 0, 2),
            cute.group_modes(gW, 0, 2),
        )
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )

        with profile_region(
            prof_buf,
            max_profile_events,
            cutlass.Int32(TAG_TMA_PROLOGUE),
            pid_n,
            event_idx=cutlass.Int32(0),
            bounds_check=False,
            enabled=enable_profiling,
        ):
            if warp == TMA_PRODUCER_WARP:
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

        acc = [cutlass.Float32(0.0) for _ in range(self.block_n)]
        for k_tile in cutlass.range_constexpr(self.num_k_tiles):
            if warp == TMA_PRODUCER_WARP and k_tile < self.num_k_tiles - 1:
                with profile_region(
                    prof_buf,
                    max_profile_events,
                    cutlass.Int32(TAG_TMA_REFILL),
                    pid_n,
                    event_idx=cutlass.Int32(1 + 3 * k_tile),
                    bounds_check=False,
                    enabled=enable_profiling,
                ):
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

            with profile_region(
                prof_buf,
                max_profile_events,
                cutlass.Int32(TAG_TMA_WAIT),
                pid_n,
                event_idx=cutlass.Int32(2 + 3 * k_tile),
                bounds_check=False,
                enabled=enable_profiling,
            ):
                full = consumer.wait_and_advance()
            with profile_region(
                prof_buf,
                max_profile_events,
                cutlass.Int32(TAG_TILE_COMPUTE),
                pid_n,
                event_idx=cutlass.Int32(3 + 3 * k_tile),
                bounds_check=False,
                enabled=enable_profiling,
            ):
                scale_k = lane + k_tile * 32
                x_frag = cute.make_rmem_tensor((1, 8), cutlass.Uint32)
                cute.copy(
                    smem_atom,
                    cute.make_tensor(
                        sX.iterator + cute.assume(sX.layout((warp, col_a, full.index)), divby=4),
                        chunk_layout,
                    ),
                    cute.make_tensor(x_frag.iterator, chunk_layout),
                )
                cute.copy(
                    smem_atom,
                    cute.make_tensor(
                        sX.iterator + cute.assume(sX.layout((warp, col_b, full.index)), divby=4),
                        chunk_layout,
                    ),
                    cute.make_tensor(x_frag.iterator + 4, chunk_layout),
                )
                x_values = (
                    cute.recast_tensor(x_frag, cutlass.Float8E4M3FN)
                    .load()
                    .to(cutlass.Float32)
                    .reshape((2, 16))
                )
                input_scale = cute.make_rmem_tensor(1, cutlass.Uint8)
                cute.copy(
                    input_scale_atom,
                    cute.make_tensor(mSFX.iterator + mSFX.layout((warp, scale_k)), scale_layout),
                    input_scale,
                )
                sx = input_scale[0]

                for row in cutlass.range_constexpr(self.block_n):
                    w_frag = cute.make_rmem_tensor((1, 8), cutlass.Uint32)
                    cute.copy(
                        smem_atom,
                        cute.make_tensor(
                            sW.iterator
                            + cute.assume(sW.layout((row, col_a, full.index)), divby=4),
                            chunk_layout,
                        ),
                        cute.make_tensor(w_frag.iterator, chunk_layout),
                    )
                    cute.copy(
                        smem_atom,
                        cute.make_tensor(
                            sW.iterator
                            + cute.assume(sW.layout((row, col_b, full.index)), divby=4),
                            chunk_layout,
                        ),
                        cute.make_tensor(w_frag.iterator + 4, chunk_layout),
                    )
                    w_values = (
                        cute.recast_tensor(w_frag, cutlass.Float8E4M3FN)
                        .load()
                        .to(cutlass.Float32)
                        .reshape((2, 16))
                    )
                    weight_scale = cute.make_rmem_tensor(1, cutlass.Uint8)
                    cute.copy(
                        weight_scale_atom,
                        cute.make_tensor(
                            mSFW.iterator + mSFW.layout((n0 + row, scale_k)),
                            scale_layout,
                        ),
                        weight_scale,
                    )
                    product = (x_values * w_values).reduce(
                        cute.ReductionOp.ADD, cutlass.Float32(0.0), (None, 1)
                    )
                    acc[row] += (product[0] + product[1]) * combined_e8m0_to_f32(
                        sx, weight_scale[0]
                    )
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                full.release()

        with profile_region(
            prof_buf,
            max_profile_events,
            cutlass.Int32(TAG_EPILOGUE),
            pid_n,
            event_idx=cutlass.Int32(1 + 3 * self.num_k_tiles),
            bounds_check=False,
            enabled=enable_profiling,
        ):
            if warp == TMA_PRODUCER_WARP:
                producer.tail()
            for row in cutlass.range_constexpr(self.block_n):
                acc[row] = cute.arch.warp_reduction(acc[row], operator.add)
            if lane == 0:
                for row in cutlass.range_constexpr(self.block_n):
                    mO[warp, n0 + row] = acc[row].to(cutlass.BFloat16)

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,
        mX: cute.Tensor,
        mSFW: cute.Tensor,
        mSFX: cute.Tensor,
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
            mO,
            prof_buf,
            self.enable_profiling,
            _name_prefix=name,
        ).launch(
            grid=[self.n // self.block_n, 1, 1],
            block=[self.m * 32, 1, 1],
            stream=stream,
        )

    def get_key(self) -> str:
        """Return the static kernel specialization key."""
        return (
            f"{self.n}_{self.sf_k}_{self.block_n}_{self.num_stages}"
            f"_profile={self.enable_profiling}"
        )

    def get_name(self) -> str:
        """Return the compiled kernel name."""
        profile_suffix = "_profiled" if self.enable_profiling else ""
        return (
            f"mxfp8_tma_gemv_n{self.n}_k{self.sf_k * 32}"
            f"_bn{self.block_n}_s{self.num_stages}{profile_suffix}"
        )

    def interface(
        self,
        q_input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor | None = None,
        profile_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Launch raw-layout MXFP8 GEMV into a contiguous BF16 output.

        Args:
            q_input: Contiguous ``[1, K]`` FP8 E4M3 input.
            weight: Contiguous ``[N, K]`` FP8 E4M3 weight.
            input_scale: Raw ``[1, K / 32]`` E8M0 scale storage.
            weight_scale: Raw ``[N, K / 32]`` E8M0 scale storage.
            output: Optional contiguous ``[1, N]`` BF16 output tensor.
            profile_buffer: Legacy profiler buffer for a profiling-enabled specialization.

        Returns:
            The provided or newly allocated output tensor.

        Raises:
            ValueError: If tensor shapes or layouts do not match this specialization.
            TypeError: If tensor dtypes do not match the MXFP8 contract.
        """
        k = self.sf_k * 32
        expected_shapes = {
            "q_input": (1, k),
            "weight": (self.n, k),
            "input_scale": (1, self.sf_k),
            "weight_scale": (self.n, self.sf_k),
        }
        tensors = {
            "q_input": q_input,
            "weight": weight,
            "input_scale": input_scale,
            "weight_scale": weight_scale,
        }
        for name, tensor in tensors.items():
            if tensor.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} must have shape {expected_shapes[name]}, got {tuple(tensor.shape)}"
                )
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")
        if q_input.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
            raise TypeError("q_input and weight must use torch.float8_e4m3fn")
        if input_scale.element_size() != 1 or weight_scale.element_size() != 1:
            raise TypeError("input_scale and weight_scale must use one-byte E8M0 storage")
        if q_input.device != weight.device or any(
            tensor.device != q_input.device for tensor in (input_scale, weight_scale)
        ):
            raise ValueError("all inputs must be on the same CUDA device")
        if torch.cuda.get_device_capability(q_input.device) not in {(10, 0), (10, 3)}:
            raise ValueError("MXFP8 TMA GEMV requires SM100 or SM103")
        if output is None:
            output = torch.empty((1, self.n), dtype=torch.bfloat16, device=q_input.device)
        elif (
            output.shape != (1, self.n)
            or output.dtype != torch.bfloat16
            or output.device != q_input.device
            or not output.is_contiguous()
        ):
            raise ValueError("output must be a contiguous [1, N] BF16 tensor on the input device")

        expected_profile_numel = self.num_profile_units * (1 + 4 * self.max_profile_events_per_cta)
        if self.enable_profiling:
            if profile_buffer is None:
                raise ValueError("profile_buffer is required when enable_profiling=True")
            if (
                profile_buffer.dtype != torch.int64
                or profile_buffer.device != q_input.device
                or not profile_buffer.is_contiguous()
                or profile_buffer.numel() < expected_profile_numel
            ):
                raise ValueError(
                    "profile_buffer must be a contiguous CUDA int64 tensor with "
                    f"at least {expected_profile_numel} elements"
                )
        elif profile_buffer is not None:
            raise ValueError("profile_buffer requires an enable_profiling=True specialization")

        fake_profile_buffer = (
            make_fake_compact_tensor(cutlass.Int64, (expected_profile_numel,))
            if self.enable_profiling
            else None
        )
        compiled = compile_tvm_ffi_and_cache(
            self,
            self.get_name(),
            make_fake_compact_tensor(cutlass.Float8E4M3FN, (self.n, k)),
            make_fake_compact_tensor(cutlass.Float8E4M3FN, (1, k)),
            make_fake_compact_tensor(cutlass.Uint8, (self.n, self.sf_k)),
            make_fake_compact_tensor(cutlass.Uint8, (1, self.sf_k)),
            make_fake_compact_tensor(cutlass.BFloat16, (1, self.n)),
            fake_profile_buffer,
            fake_stream(),
            _name_prefix=self.get_name(),
        )
        compiled(
            weight,
            q_input,
            weight_scale.view(torch.uint8),
            input_scale.view(torch.uint8),
            output,
            profile_buffer,
        )
        return output


@cache
def get_mxfp8_tma_gemv(
    n: int,
    k: int,
    block_n: int,
    num_stages: int = 2,
    enable_profiling: bool = False,
) -> Mxfp8TmaGemv:
    """Return a cached MXFP8 TMA GEMV specialization."""
    return Mxfp8TmaGemv(n, k, block_n, num_stages, enable_profiling)


def mxfp8_tma_gemv(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    block_n: int,
    num_stages: int = 2,
    output: torch.Tensor | None = None,
    enable_profiling: bool = False,
    profile_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute raw-layout MXFP8 GEMV on prequantized inputs.

    Args:
        q_input: Contiguous ``[1, K]`` FP8 E4M3 input.
        weight: Contiguous ``[N, K]`` FP8 E4M3 weight.
        input_scale: Raw ``[1, K / 32]`` E8M0 scale storage.
        weight_scale: Raw ``[N, K / 32]`` E8M0 scale storage.
        block_n: Output rows computed by each CTA.
        num_stages: Number of shared-memory TMA pipeline stages.
        output: Optional caller-owned contiguous ``[1, N]`` BF16 output.
        enable_profiling: Compile a separate specialization with labeled region timing.
        profile_buffer: Buffer from ``profile_session`` for the profiled specialization.

    Returns:
        The provided or newly allocated output tensor.
    """
    if q_input.ndim != 2 or weight.ndim != 2:
        raise ValueError("q_input and weight must be rank-2 tensors")
    return get_mxfp8_tma_gemv(
        weight.shape[0],
        q_input.shape[1],
        block_n,
        num_stages,
        enable_profiling,
    ).interface(
        q_input,
        weight,
        input_scale,
        weight_scale,
        output,
        profile_buffer,
    )


def quantize_mxfp8_tensor(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create E4M3 values and raw E8M0 block scales for profiling inputs."""
    blocks = value.float().reshape(value.shape[0], -1, 32)
    max_abs = blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(max_abs / 448.0)).clamp(-126, 127)
    scale = torch.exp2(exponent).unsqueeze(-1)
    quantized = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.reshape_as(value), (exponent + 127).to(torch.uint8)


app = typer.Typer(help="Run the MXFP8 TMA GEMV with labeled intra-kernel profiling.")


@app.command()
def profile_mxfp8_tma(
    n: int = 4096,
    k: int = 8192,
    block_n: int = 4,
    num_stages: int = 2,
    output: Path = Path("mxfp8_tma.pftrace"),
    seed: int = 0,
    warmups: int = 1,
    device: str = "cuda",
) -> None:
    """Generate a Perfetto trace for one warm MXFP8 TMA GEMV launch."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise typer.BadParameter("device must name an available CUDA device")
    if warmups < 0:
        raise typer.BadParameter("warmups must be non-negative")

    torch.manual_seed(seed)
    q_input, input_scale = quantize_mxfp8_tensor(
        torch.randn((1, k), dtype=torch.bfloat16, device=torch_device)
    )
    weight, weight_scale = quantize_mxfp8_tensor(
        torch.randn((n, k), dtype=torch.bfloat16, device=torch_device)
    )
    op = get_mxfp8_tma_gemv(
        n,
        k,
        block_n,
        num_stages,
        enable_profiling=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with profile_session(
        max_events_per_unit=op.max_profile_events_per_cta,
        num_units=(op.num_profile_units, "CTA"),
        tag_names=list(op.profile_tags),
        trace_path=str(output),
        device=torch_device,
        post_process_events=group_by_unit,
    ) as (prof, _):
        result = torch.empty((1, n), dtype=torch.bfloat16, device=torch_device)
        for _ in range(warmups):
            op.interface(
                q_input,
                weight,
                input_scale,
                weight_scale,
                output=result,
                profile_buffer=prof.tensor,
            )
        torch.cuda.synchronize(torch_device)
        prof.tensor.zero_()
        op.interface(
            q_input,
            weight,
            input_scale,
            weight_scale,
            output=result,
            profile_buffer=prof.tensor,
        )

    typer.echo(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    app()
