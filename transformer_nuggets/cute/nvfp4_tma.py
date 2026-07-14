from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch

from transformer_nuggets.cute.blockscaled_tma import (
    BLOCKSCALED_TMA_PROFILE_TAGS,
    DEFAULT_PERSISTENT_CTAS_PER_SM,
    BlockscaledTmaGemv,
    GridScheduler,
)
from transformer_nuggets.cute.cache import compile_tvm_ffi_and_cache
from transformer_nuggets.cute.utils import fake_stream, make_fake_compact_tensor


NVFP4_TMA_PROFILE_TAGS = BLOCKSCALED_TMA_PROFILE_TAGS


@cute.jit
def blocked_scale_offset(row, col, col_blocks: cutlass.Constexpr):
    """Map a logical scale coordinate to SWIZZLE_32_4_4 storage."""
    row_block = row // 128
    row_in_block = row % 128
    return (
        (((row_block * col_blocks + col // 4) * 32 + row_in_block % 32) * 4) + row_in_block // 32
    ) * 4 + col % 4


class Nvfp4TmaGemv(BlockscaledTmaGemv):
    """Compute M=1 NVFP4 GEMV using scaled_mm-compatible storage."""

    format_name = "nvfp4"

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
        use_global_scales: bool = False,
    ):
        super().__init__(
            n,
            k,
            block_n,
            num_stages,
            values_per_byte=2,
            scale_block_size=16,
            tile_k_u32=128,
            scale_copy_bits=16,
            enable_profiling=enable_profiling,
            num_compute_warps=num_compute_warps,
            grid_scheduler=grid_scheduler,
            num_persistent_ctas=num_persistent_ctas,
            use_global_scales=use_global_scales,
        )

    @cute.jit
    def decode_lane_values(self, raw_values: cute.Tensor):
        """Decode 32 packed E2M1 values into two 16-value groups."""
        return (
            cute.recast_tensor(raw_values, cutlass.Float4E2M1FN)
            .load()
            .to(cutlass.Float32)
            .reshape((16, 2))
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
        """Load two contiguous E4M3 block scales from swizzled storage."""
        scales = cute.make_rmem_tensor(2, cutlass.Uint8)
        cute.copy(
            scale_atom,
            cute.make_tensor(
                scale_tensor.iterator.align(min_align=2)
                + cute.assume(
                    blocked_scale_offset(row, scale_k, self.sf_k // 4),
                    divby=2,
                ),
                cute.make_layout(2),
            ),
            scales,
        )
        return (
            scales[0].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32),
            scales[1].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32),
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
        """Accumulate two independently scaled 16-value E2M1 blocks."""
        products = (x_values * w_values).reduce(
            cute.ReductionOp.ADD,
            cutlass.Float32(0.0),
            (1, None),
        )
        for scale_idx in cutlass.range_constexpr(2):
            accumulator += products[scale_idx] * input_scales[scale_idx] * weight_scales[scale_idx]
        return accumulator

    def interface(
        self,
        q_input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        input_global_scale: torch.Tensor | None = None,
        weight_global_scale: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        profile_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Launch packed NVFP4 GEMV into a contiguous BF16 output."""
        expected_shapes = {
            "q_input": (1, self.k // 2),
            "weight": (self.n, self.k // 2),
        }
        for name, tensor in (("q_input", q_input), ("weight", weight)):
            if tensor.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} must have shape {expected_shapes[name]}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != torch.float4_e2m1fn_x2:
                raise TypeError(f"{name} must use torch.float4_e2m1fn_x2")
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")

        expected_scale_numels = {
            "input_scale": 128 * self.sf_k,
            "weight_scale": ((self.n + 127) // 128) * 128 * self.sf_k,
        }
        for name, tensor in (("input_scale", input_scale), ("weight_scale", weight_scale)):
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name} must use torch.float8_e4m3fn")
            if tensor.numel() != expected_scale_numels[name]:
                raise ValueError(
                    f"{name} must contain {expected_scale_numels[name]} swizzled elements"
                )
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")

        if any(tensor.device != q_input.device for tensor in (weight, input_scale, weight_scale)):
            raise ValueError("all inputs must be on the same CUDA device")
        if torch.cuda.get_device_capability(q_input.device) not in {(10, 0), (10, 3)}:
            raise ValueError("NVFP4 TMA GEMV requires SM100 or SM103")

        global_scales = (input_global_scale, weight_global_scale)
        if self.use_global_scales:
            if any(scale is None for scale in global_scales):
                raise ValueError("both global scales are required by this specialization")
            for name, scale in zip(
                ("input_global_scale", "weight_global_scale"),
                global_scales,
                strict=True,
            ):
                assert scale is not None
                if (
                    scale.shape != (1,)
                    or scale.dtype != torch.float32
                    or scale.device != q_input.device
                    or not scale.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be a contiguous CUDA float32 tensor with shape [1]"
                    )
        elif any(scale is not None for scale in global_scales):
            raise ValueError("global scales require a global-scale specialization")

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
        fake_global_scale = (
            make_fake_compact_tensor(cutlass.Float32, (1,)) if self.use_global_scales else None
        )
        compiled = compile_tvm_ffi_and_cache(
            self,
            self.get_name(),
            make_fake_compact_tensor(cutlass.Uint8, (self.n, self.k // 2)),
            make_fake_compact_tensor(cutlass.Uint8, (1, self.k // 2)),
            make_fake_compact_tensor(
                cutlass.Uint8,
                (expected_scale_numels["weight_scale"],),
            ),
            make_fake_compact_tensor(
                cutlass.Uint8,
                (expected_scale_numels["input_scale"],),
            ),
            fake_global_scale,
            fake_global_scale,
            make_fake_compact_tensor(cutlass.BFloat16, (1, self.n)),
            fake_profile_buffer,
            fake_stream(),
            _name_prefix=self.get_name(),
        )
        compiled(
            weight.view(torch.uint8),
            q_input.view(torch.uint8),
            weight_scale.view(torch.uint8),
            input_scale.view(torch.uint8),
            weight_global_scale,
            input_global_scale,
            output,
            profile_buffer,
        )
        return output


@cache
def get_nvfp4_tma_gemv(
    n: int,
    k: int,
    block_n: int,
    num_stages: int = 2,
    enable_profiling: bool = False,
    num_compute_warps: int = 1,
    grid_scheduler: GridScheduler = GridScheduler.STATIC,
    num_persistent_ctas: int | None = None,
    use_global_scales: bool = False,
) -> Nvfp4TmaGemv:
    """Return a cached NVFP4 TMA GEMV specialization."""
    return Nvfp4TmaGemv(
        n,
        k,
        block_n,
        num_stages,
        enable_profiling,
        num_compute_warps,
        grid_scheduler,
        num_persistent_ctas,
        use_global_scales,
    )


def nvfp4_tma_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    block_n: int,
    global_scale_a: torch.Tensor | None = None,
    global_scale_b: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    num_stages: int = 2,
    num_compute_warps: int | None = None,
    grid_scheduler: GridScheduler = GridScheduler.STATIC,
    num_persistent_ctas: int | None = None,
) -> torch.Tensor:
    """Run the M=1 subset using F.scaled_mm's transposed-B convention."""
    if mat_b.ndim != 2:
        raise ValueError("mat_b must be rank-2")
    weight = mat_b.t()
    if not weight.is_contiguous():
        raise ValueError("mat_b must be a metadata transpose of contiguous [N, K/2] storage")
    return nvfp4_tma_gemv(
        mat_a,
        weight,
        scale_a,
        scale_b,
        block_n=block_n,
        num_stages=num_stages,
        input_global_scale=global_scale_a,
        weight_global_scale=global_scale_b,
        output=output,
        num_compute_warps=num_compute_warps,
        grid_scheduler=grid_scheduler,
        num_persistent_ctas=num_persistent_ctas,
    )


def select_nvfp4_tma_compute_warps(block_n: int) -> int:
    """Select the largest warp count that preserves at least two rows per warp."""
    for num_compute_warps in (4, 2, 1):
        if block_n % num_compute_warps == 0 and block_n // num_compute_warps >= 2:
            return num_compute_warps
    return 1


def nvfp4_tma_gemv(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    block_n: int,
    num_stages: int = 2,
    input_global_scale: torch.Tensor | None = None,
    weight_global_scale: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    enable_profiling: bool = False,
    profile_buffer: torch.Tensor | None = None,
    num_compute_warps: int | None = None,
    grid_scheduler: GridScheduler = GridScheduler.STATIC,
    num_persistent_ctas: int | None = None,
) -> torch.Tensor:
    """Compute M=1 NVFP4 GEMV with optional tensorwise decode scales."""
    if q_input.ndim != 2 or weight.ndim != 2:
        raise ValueError("q_input and weight must be rank-2 tensors")
    k = q_input.shape[1] * 2
    if num_compute_warps is None:
        num_compute_warps = select_nvfp4_tma_compute_warps(block_n)
    grid_scheduler = GridScheduler(grid_scheduler)
    if grid_scheduler is GridScheduler.PERSISTENT and num_persistent_ctas is None:
        num_persistent_ctas = (
            DEFAULT_PERSISTENT_CTAS_PER_SM
            * torch.cuda.get_device_properties(q_input.device).multi_processor_count
        )
    use_global_scales = input_global_scale is not None or weight_global_scale is not None
    return get_nvfp4_tma_gemv(
        weight.shape[0],
        k,
        block_n,
        num_stages,
        enable_profiling,
        num_compute_warps,
        grid_scheduler,
        num_persistent_ctas,
        use_global_scales,
    ).interface(
        q_input,
        weight,
        input_scale,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        output,
        profile_buffer,
    )
