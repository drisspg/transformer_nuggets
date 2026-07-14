from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
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


@dsl_user_op
def decode_e2m1x8_words(
    packed: cutlass.Uint32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32]:
    """Decode one packed E2M1 word into four FP16x2 registers."""
    converted = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [packed.ir_value(loc=loc, ip=ip)],
        """{
            .reg .b8 b0, b1, b2, b3;
            mov.b32 {b0, b1, b2, b3}, $4;
            cvt.rn.f16x2.e2m1x2 $0, b0;
            cvt.rn.f16x2.e2m1x2 $1, b1;
            cvt.rn.f16x2.e2m1x2 $2, b2;
            cvt.rn.f16x2.e2m1x2 $3, b3;
        }""",
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), converted, [word], loc=loc, ip=ip))
        for word in range(4)
    )


@dsl_user_op
def dot_e2m1x8(
    x0: cutlass.Uint32,
    x1: cutlass.Uint32,
    x2: cutlass.Uint32,
    x3: cutlass.Uint32,
    packed_weight: cutlass.Uint32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float32:
    """Decode one weight word and reduce its dot product with decoded input values."""
    result = llvm.inline_asm(
        T.f32(),
        [
            x0.ir_value(loc=loc, ip=ip),
            x1.ir_value(loc=loc, ip=ip),
            x2.ir_value(loc=loc, ip=ip),
            x3.ir_value(loc=loc, ip=ip),
            packed_weight.ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .b8 b0, b1, b2, b3;
            .reg .b32 w0, w1, w2, w3;
            .reg .b32 p0, p1, p2, p3;
            .reg .b16 lo, hi;
            .reg .f16 total;
            mov.b32 {b0, b1, b2, b3}, $5;
            cvt.rn.f16x2.e2m1x2 w0, b0;
            cvt.rn.f16x2.e2m1x2 w1, b1;
            cvt.rn.f16x2.e2m1x2 w2, b2;
            cvt.rn.f16x2.e2m1x2 w3, b3;
            mul.rn.f16x2 p0, $1, w0;
            mul.rn.f16x2 p1, $2, w1;
            mul.rn.f16x2 p2, $3, w2;
            mul.rn.f16x2 p3, $4, w3;
            add.rn.f16x2 p0, p0, p1;
            add.rn.f16x2 p2, p2, p3;
            add.rn.f16x2 p0, p0, p2;
            mov.b32 {lo, hi}, p0;
            add.rn.f16 total, lo, hi;
            cvt.f32.f16 $0, total;
        }""",
        "=f,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


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
            weight_scale_copy_bits=(
                32
                if block_n // num_compute_warps >= 2 and (block_n // num_compute_warps) % 2 == 0
                else 16
            ),
            enable_profiling=enable_profiling,
            num_compute_warps=num_compute_warps,
            grid_scheduler=grid_scheduler,
            num_persistent_ctas=num_persistent_ctas,
            use_global_scales=use_global_scales,
        )
        self.use_streaming_decode = self.num_k_tiles <= 4 or self.num_k_tiles >= 16

    @cute.jit
    def decode_lane_values(self, raw_values: cute.Tensor):
        """Decode 32 packed E2M1 values into register-resident FP16 groups."""
        if cutlass.const_expr(self.use_streaming_decode):
            return (
                decode_e2m1x8_words(raw_values[0, 0]),
                decode_e2m1x8_words(raw_values[0, 1]),
                decode_e2m1x8_words(raw_values[0, 2]),
                decode_e2m1x8_words(raw_values[0, 3]),
            )
        return (
            cute.recast_tensor(raw_values, cutlass.Float4E2M1FN)
            .load()
            .to(cutlass.Float16)
            .reshape((16, 2))
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
        """Keep selected packed weights raw until each immediate dot reduction."""
        if cutlass.const_expr(self.use_streaming_decode):
            return self.load_lane_words(smem, row, lane, stage, smem_atom, chunk_layout)
        return self.load_lane_values(smem, row, lane, stage, smem_atom, chunk_layout)

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
                    self.make_blocked_scale_layout(16)((row, scale_k * 16, 0)),
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
    def prepare_weight_scale_values(
        self,
        scale_tensor: cute.Tensor,
        row_start,
        k_tile: cutlass.Constexpr,
        lane: cutlass.Int32,
        scale_atom: cute.CopyAtom,
        scale_layout: cute.Layout,
    ):
        """Load two rows of four-column scale groups per warp instruction."""
        if cutlass.const_expr(self.rows_per_warp < 2 or self.rows_per_warp % 2 != 0):
            return None
        packed_row_pairs = []
        for row_pair in cutlass.range_constexpr(self.rows_per_warp // 2):
            scales = cute.make_rmem_tensor(4, cutlass.Uint8)
            row = row_start + row_pair * 2 + lane // 16
            col = k_tile * self.scale_blocks_per_tile + (lane % 16) * 4
            cute.copy(
                scale_atom,
                cute.make_tensor(
                    scale_tensor.iterator.align(min_align=4)
                    + cute.assume(
                        self.make_blocked_scale_layout(16)((row, col * 16, 0)),
                        divby=4,
                    ),
                    cute.make_layout(4),
                ),
                scales,
            )
            packed_row_pairs.append(cute.recast_tensor(scales, cutlass.Uint32).load()[0])
        return packed_row_pairs

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
        """Shuffle one packed four-scale group to its two consuming lanes."""
        if cutlass.const_expr(self.rows_per_warp < 2 or self.rows_per_warp % 2 != 0):
            return self.load_scale_values(
                scale_tensor,
                row,
                scale_k,
                scale_atom,
                scale_layout,
            )
        packed = cute.arch.shuffle_sync_op(
            prepared_scales[local_row // 2],
            (local_row % 2) * 16 + lane // 2,
        )
        scales = packed >> cutlass.Uint32((lane & 1) * 16)
        return (
            cutlass.Uint8(scales & 0xFF).bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32),
            cutlass.Uint8((scales >> 8) & 0xFF).bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32),
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
        if cutlass.const_expr(self.use_streaming_decode):
            partials = tuple(
                dot_e2m1x8(
                    x_values[word][0],
                    x_values[word][1],
                    x_values[word][2],
                    x_values[word][3],
                    w_values[0, word],
                )
                for word in range(4)
            )
            products = (
                partials[0] + partials[1],
                partials[2] + partials[3],
            )
        else:
            products = (
                (x_values * w_values)
                .reshape((2, 8, 2))
                .reduce(
                    cute.ReductionOp.ADD,
                    cutlass.Float16(0.0),
                    (1, None, None),
                )
            )
            products = products.to(cutlass.Float32).reduce(
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
    block_n: int | None = None,
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


def select_nvfp4_tma_compute_warps(
    k: int,
    block_n: int,
    device: torch.device | str | int | None = None,
) -> int:
    """Select the B200-tuned NVFP4 compute-warp count."""
    if torch.cuda.get_device_capability(device) != (10, 0):
        return 1
    target_rows_per_warp = 1 if k >= 12288 else 2
    for num_compute_warps in (4, 2, 1):
        if (
            block_n % num_compute_warps == 0
            and block_n // num_compute_warps >= target_rows_per_warp
        ):
            return num_compute_warps
    return 1


def select_nvfp4_tma_config(
    n: int,
    k: int,
    device: torch.device | str | int | None = None,
) -> tuple[int, int, int]:
    """Select the B200-tuned block, stage, and compute-warp configuration."""
    if torch.cuda.get_device_capability(device) != (10, 0):
        block_n = next(candidate for candidate in (8, 4, 2, 1) if n % candidate == 0)
        return block_n, 2, 1
    preferred_block_n = 16 if n >= 12288 and k < 12288 else 8
    block_n = next(
        candidate for candidate in (preferred_block_n, 8, 4, 2, 1) if n % candidate == 0
    )
    return block_n, 2, select_nvfp4_tma_compute_warps(k, block_n, device)


def nvfp4_tma_gemv(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    block_n: int | None = None,
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
    if block_n is None:
        block_n, num_stages, selected_compute_warps = select_nvfp4_tma_config(
            weight.shape[0],
            k,
            q_input.device,
        )
        if num_compute_warps is None:
            num_compute_warps = selected_compute_warps
    elif num_compute_warps is None:
        num_compute_warps = select_nvfp4_tma_compute_warps(k, block_n, q_input.device)
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
