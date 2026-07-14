from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Annotated

import cutlass
import cutlass.cute as cute
import torch
import typer

from transformer_nuggets.cute.blockscaled_tma import (
    BLOCKSCALED_TMA_PROFILE_TAGS,
    DEFAULT_PERSISTENT_CTAS_PER_SM,
    BlockScaleLayout,
    BlockscaledTmaGemv,
    GridScheduler,
    ProfileTag as ProfileTag,
)
from transformer_nuggets.cute.cache import compile_tvm_ffi_and_cache
from transformer_nuggets.cute.profiler import group_by_unit, profile_session
from transformer_nuggets.cute.utils import fake_stream, make_fake_compact_tensor


MXFP8_TMA_PROFILE_TAGS = BLOCKSCALED_TMA_PROFILE_TAGS


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


class Mxfp8TmaGemv(BlockscaledTmaGemv):
    """Compute MXFP8 GEMV from raw or canonical blocked E8M0 scales."""

    format_name = "mxfp8"

    def __init__(
        self,
        n: int,
        k: int,
        block_n: int,
        num_stages: int,
        enable_profiling: bool = False,
        num_compute_warps: int = 1,
        grid_scheduler: GridScheduler = GridScheduler.STATIC,
        num_persistent_ctas: int | None = None,
        block_scale_layout: BlockScaleLayout = BlockScaleLayout.RAW,
    ):
        self.block_scale_layout = BlockScaleLayout(block_scale_layout)
        rows_per_warp = block_n // num_compute_warps
        super().__init__(
            n,
            k,
            block_n,
            num_stages,
            values_per_byte=1,
            scale_block_size=32,
            tile_k_u32=256,
            scale_copy_bits=8,
            weight_scale_copy_bits=(
                32
                if self.block_scale_layout is BlockScaleLayout.SWIZZLE_32_4_4
                and rows_per_warp >= 2
                else 8
            ),
            enable_profiling=enable_profiling,
            num_compute_warps=num_compute_warps,
            grid_scheduler=grid_scheduler,
            num_persistent_ctas=num_persistent_ctas,
        )
        if self.block_scale_layout is BlockScaleLayout.SWIZZLE_32_4_4:
            self.format_name = "mxfp8_swizzled"

    @cute.jit
    def decode_lane_values(self, raw_values: cute.Tensor):
        """Decode 32 E4M3 values into two vector groups."""
        return (
            cute.recast_tensor(raw_values, cutlass.Float8E4M3FN)
            .load()
            .to(cutlass.Float32)
            .reshape((2, 16))
        )

    @cute.jit
    def load_scale_values(
        self,
        scale_tensor: cute.Tensor,
        row,
        scale_k,
        scale_atom: cute.CopyAtom,
        scale_layout: cute.Layout,
    ):
        """Load one E8M0 scale byte from raw or blocked storage."""
        scale = cute.make_rmem_tensor(1, cutlass.Uint8)
        if cutlass.const_expr(self.block_scale_layout is BlockScaleLayout.SWIZZLE_32_4_4):
            offset = self.make_blocked_scale_layout(32)((row, scale_k * 32, 0))
        else:
            offset = scale_tensor.layout((row, scale_k))
        cute.copy(
            scale_atom,
            cute.make_tensor(scale_tensor.iterator + offset, scale_layout),
            scale,
        )
        return scale[0]

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
        """Load up to four rows of blocked E8M0 scales per warp instruction."""
        if cutlass.const_expr(
            self.block_scale_layout is BlockScaleLayout.RAW or self.rows_per_warp == 1
        ):
            return None
        packed_row_groups = []
        for row_group in cutlass.range_constexpr((self.rows_per_warp + 3) // 4):
            rows_in_group = min(4, self.rows_per_warp - row_group * 4)
            lane_slot = lane % (rows_in_group * 8)
            row = row_start + row_group * 4 + lane_slot // 8
            col = k_tile * self.scale_blocks_per_tile + (lane_slot % 8) * 4
            scales = cute.make_rmem_tensor(4, cutlass.Uint8)
            cute.copy(
                scale_atom,
                cute.make_tensor(
                    scale_tensor.iterator.align(min_align=4)
                    + cute.assume(
                        self.make_blocked_scale_layout(32)((row, col * 32, 0)),
                        divby=4,
                    ),
                    cute.make_layout(4),
                ),
                scales,
            )
            packed_row_groups.append(cute.recast_tensor(scales, cutlass.Uint32).load()[0])
        return packed_row_groups

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
        """Shuffle one packed four-scale group to its four consuming lanes."""
        if cutlass.const_expr(
            self.block_scale_layout is BlockScaleLayout.RAW or self.rows_per_warp == 1
        ):
            return self.load_scale_values(
                scale_tensor,
                row,
                scale_k,
                scale_atom,
                scale_layout,
            )
        packed = cute.arch.shuffle_sync_op(
            prepared_scales[local_row // 4],
            (local_row % 4) * 8 + lane // 4,
        )
        return cutlass.Uint8((packed >> cutlass.Uint32((lane & 3) * 8)) & 0xFF)

    @cute.jit
    def accumulate_scaled_products(
        self,
        accumulator,
        x_values,
        w_values,
        input_scale,
        weight_scale,
    ):
        """Accumulate one 32-value E4M3 block with combined E8M0 scaling."""
        products = (x_values * w_values).reduce(
            cute.ReductionOp.ADD,
            cutlass.Float32(0.0),
            (None, 1),
        )
        return accumulator + (products[0] + products[1]) * combined_e8m0_to_f32(
            input_scale,
            weight_scale,
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
        """Launch MXFP8 GEMV from raw or blocked scales into contiguous BF16 output."""
        k = self.k
        for name, tensor, expected_shape in (
            ("q_input", q_input, (1, k)),
            ("weight", weight, (self.n, k)),
        ):
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
                )
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")

        if self.block_scale_layout is BlockScaleLayout.RAW:
            expected_scale_shapes = {
                "input_scale": (1, self.sf_k),
                "weight_scale": (self.n, self.sf_k),
            }
            for name, tensor in (("input_scale", input_scale), ("weight_scale", weight_scale)):
                if tensor.shape != expected_scale_shapes[name]:
                    raise ValueError(
                        f"{name} must have shape {expected_scale_shapes[name]}, "
                        f"got {tuple(tensor.shape)}"
                    )
        else:
            padded_sf_k = ((self.sf_k + 3) // 4) * 4
            expected_scale_numels = {
                "input_scale": 128 * padded_sf_k,
                "weight_scale": ((self.n + 127) // 128) * 128 * padded_sf_k,
            }
            for name, tensor in (("input_scale", input_scale), ("weight_scale", weight_scale)):
                if tensor.numel() != expected_scale_numels[name]:
                    raise ValueError(
                        f"{name} must contain {expected_scale_numels[name]} swizzled elements"
                    )
        for name, tensor in (("input_scale", input_scale), ("weight_scale", weight_scale)):
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
        weight_scale_shape = (
            (self.n, self.sf_k)
            if self.block_scale_layout is BlockScaleLayout.RAW
            else (expected_scale_numels["weight_scale"],)
        )
        input_scale_shape = (
            (1, self.sf_k)
            if self.block_scale_layout is BlockScaleLayout.RAW
            else (expected_scale_numels["input_scale"],)
        )
        compiled = compile_tvm_ffi_and_cache(
            self,
            self.get_name(),
            make_fake_compact_tensor(cutlass.Float8E4M3FN, (self.n, k)),
            make_fake_compact_tensor(cutlass.Float8E4M3FN, (1, k)),
            make_fake_compact_tensor(cutlass.Uint8, weight_scale_shape),
            make_fake_compact_tensor(cutlass.Uint8, input_scale_shape),
            None,
            None,
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
            None,
            None,
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
    num_compute_warps: int = 1,
    grid_scheduler: GridScheduler = GridScheduler.STATIC,
    num_persistent_ctas: int | None = None,
    block_scale_layout: BlockScaleLayout = BlockScaleLayout.RAW,
) -> Mxfp8TmaGemv:
    """Return a cached MXFP8 TMA GEMV specialization."""
    return Mxfp8TmaGemv(
        n,
        k,
        block_n,
        num_stages,
        enable_profiling,
        num_compute_warps,
        grid_scheduler,
        num_persistent_ctas,
        block_scale_layout,
    )


def select_mxfp8_tma_compute_warps(
    k: int,
    block_n: int,
    device: torch.device | str | int | None = None,
) -> int:
    """Select the B200-tuned consumer-warp count for a CTA row tile."""
    if torch.cuda.get_device_capability(device) != (10, 0):
        return 1
    target_rows_per_warp = 4 if k >= 8192 else 2
    for num_compute_warps in (4, 2, 1):
        if (
            block_n % num_compute_warps == 0
            and block_n // num_compute_warps >= target_rows_per_warp
        ):
            return num_compute_warps
    return 1


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
    num_compute_warps: int | None = None,
    grid_scheduler: GridScheduler = GridScheduler.STATIC,
    num_persistent_ctas: int | None = None,
    block_scale_layout: BlockScaleLayout = BlockScaleLayout.RAW,
) -> torch.Tensor:
    """Compute MXFP8 GEMV on prequantized inputs and caller-owned scales.

    Args:
        q_input: Contiguous ``[1, K]`` FP8 E4M3 input.
        weight: Contiguous ``[N, K]`` FP8 E4M3 weight.
        input_scale: Raw or ``SWIZZLE_32_4_4`` E8M0 input scales.
        weight_scale: Raw or ``SWIZZLE_32_4_4`` E8M0 weight scales.
        block_n: Output rows computed by each CTA.
        num_stages: Number of shared-memory TMA pipeline stages.
        output: Optional caller-owned contiguous ``[1, N]`` BF16 output.
        enable_profiling: Compile a separate specialization with labeled region timing.
        profile_buffer: Buffer from ``profile_session`` for the profiled specialization.
        num_compute_warps: Consumer warps sharing each CTA's output-row tile. ``None``
            selects the B200-tuned value and remains one warp on other architectures.
        grid_scheduler: Launch one CTA per output tile or reuse a persistent CTA grid.
        num_persistent_ctas: Maximum physical CTA count for the persistent scheduler.
            ``None`` requests eight CTAs per SM; the launch rounds down to a divisor
            of the logical tile count so every physical CTA executes equal work.
        block_scale_layout: Caller-owned raw or blocked scale storage layout.

    Returns:
        The provided or newly allocated output tensor.
    """
    if q_input.ndim != 2 or weight.ndim != 2:
        raise ValueError("q_input and weight must be rank-2 tensors")
    if num_compute_warps is None:
        num_compute_warps = select_mxfp8_tma_compute_warps(
            q_input.shape[1], block_n, q_input.device
        )
    grid_scheduler = GridScheduler(grid_scheduler)
    if grid_scheduler is GridScheduler.PERSISTENT and num_persistent_ctas is None:
        num_persistent_ctas = (
            DEFAULT_PERSISTENT_CTAS_PER_SM
            * torch.cuda.get_device_properties(q_input.device).multi_processor_count
        )
    return get_mxfp8_tma_gemv(
        weight.shape[0],
        q_input.shape[1],
        block_n,
        num_stages,
        enable_profiling,
        num_compute_warps,
        grid_scheduler,
        num_persistent_ctas,
        block_scale_layout,
    ).interface(
        q_input,
        weight,
        input_scale,
        weight_scale,
        output,
        profile_buffer,
    )


def mxfp8_tma_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    block_n: int,
    output: torch.Tensor | None = None,
    num_stages: int = 2,
    num_compute_warps: int | None = None,
    grid_scheduler: GridScheduler = GridScheduler.STATIC,
    num_persistent_ctas: int | None = None,
) -> torch.Tensor:
    """Run the M=1 subset using scaled_mm's blocked-scale and transposed-B convention."""
    if mat_b.ndim != 2:
        raise ValueError("mat_b must be rank-2")
    weight = mat_b.t()
    if not weight.is_contiguous():
        raise ValueError("mat_b must be a metadata transpose of contiguous [N, K] storage")
    return mxfp8_tma_gemv(
        mat_a,
        weight,
        scale_a,
        scale_b,
        block_n=block_n,
        num_stages=num_stages,
        output=output,
        num_compute_warps=num_compute_warps,
        grid_scheduler=grid_scheduler,
        num_persistent_ctas=num_persistent_ctas,
        block_scale_layout=BlockScaleLayout.SWIZZLE_32_4_4,
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
    n: int = 14336,
    k: int = 4096,
    block_n: int = 8,
    num_stages: int = 2,
    num_compute_warps: Annotated[
        int | None,
        typer.Option(help="Compute warps per CTA; defaults to the architecture-tuned value."),
    ] = None,
    grid_scheduler: GridScheduler = GridScheduler.STATIC,
    num_persistent_ctas: Annotated[
        int | None,
        typer.Option(help="Maximum persistent CTA count; defaults to eight CTAs per SM."),
    ] = None,
    output: Path = Path("mxfp8_tma.pftrace"),
    seed: int = 0,
    warmups: int = 1,
    device: str = "cuda",
) -> None:
    """Profile a Llama 3.1 8B-sized MLP gate/up projection by default."""
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
    if num_compute_warps is None:
        num_compute_warps = select_mxfp8_tma_compute_warps(k, block_n, torch_device)
    if grid_scheduler is GridScheduler.PERSISTENT and num_persistent_ctas is None:
        num_persistent_ctas = (
            DEFAULT_PERSISTENT_CTAS_PER_SM
            * torch.cuda.get_device_properties(torch_device).multi_processor_count
        )
    op = get_mxfp8_tma_gemv(
        n,
        k,
        block_n,
        num_stages,
        enable_profiling=True,
        num_compute_warps=num_compute_warps,
        grid_scheduler=grid_scheduler,
        num_persistent_ctas=num_persistent_ctas,
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
