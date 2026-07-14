import pytest
import torch


if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
    pytest.skip("NVFP4 TMA GEMV requires SM100 or SM103", allow_module_level=True)
if not hasattr(torch, "float4_e2m1fn_x2"):
    pytest.skip("FP4 dtype is unavailable", allow_module_level=True)

try:
    from transformer_nuggets.cute import (
        NVFP4_TMA_PROFILE_TAGS,
        GridScheduler,
        get_nvfp4_tma_gemv,
        nvfp4_tma_gemv,
        nvfp4_tma_scaled_mm,
        select_nvfp4_tma_compute_warps,
        select_nvfp4_tma_config,
        select_nvfp4_tma_split_k,
        select_nvfp4_tma_stage_weight_scales,
    )
    from transformer_nuggets.cute.profiler import profile_session
    from transformer_nuggets.cute.profiler.host import decode_events
except ImportError:
    pytest.skip("CuTe DSL not available", allow_module_level=True)


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


@pytest.mark.parametrize(
    ("k", "block_n", "expected"),
    [(8192, 4, 2), (8192, 8, 4), (8192, 16, 4), (14336, 4, 4)],
)
def test_select_nvfp4_tma_compute_warps(k, block_n, expected):
    """Use one row per warp only for the measured very-long-K regime."""
    assert select_nvfp4_tma_compute_warps(k, block_n) == expected


@pytest.mark.parametrize(
    ("n", "k", "expected"),
    [
        (4608, 8192, (8, 2, 4)),
        (8192, 2048, (8, 2, 4)),
        (14336, 4096, (16, 2, 4)),
        (4096, 14336, (8, 2, 4)),
        (8192, 14336, (8, 3, 4)),
        (8192, 28672, (8, 3, 4)),
        (12288, 16384, (8, 2, 4)),
        (14336, 16384, (8, 3, 4)),
        (24576, 24576, (8, 2, 4)),
        (32768, 32768, (8, 3, 4)),
        (16384, 8192, (8, 2, 4)),
    ],
)
def test_select_nvfp4_tma_config(n, k, expected):
    """Bake in the measured B200 shape families."""
    assert select_nvfp4_tma_config(n, k) == expected


@pytest.mark.parametrize(
    ("n", "k", "expected"),
    [
        (1024, 16384, 4),
        (1024, 17408, 1),
        (2048, 16384, 1),
        (2048, 24576, 4),
        (4096, 32768, 1),
    ],
)
def test_select_nvfp4_tma_split_k(n, k, expected):
    """Use split-K only for measured underfilled long-K grids."""
    if torch.cuda.get_device_capability() != (10, 0):
        expected = 1
    assert select_nvfp4_tma_split_k(n, k) == expected


@pytest.mark.parametrize(
    ("n", "k", "block_n", "num_compute_warps", "grid_scheduler", "split_k", "expected"),
    [
        (16384, 6144, 8, 4, GridScheduler.STATIC, 1, True),
        (32768, 8192, 8, 4, GridScheduler.STATIC, 1, True),
        (14336, 8192, 8, 4, GridScheduler.STATIC, 1, False),
        (16384, 12288, 8, 4, GridScheduler.STATIC, 1, False),
        (16384, 8192, 16, 4, GridScheduler.STATIC, 1, False),
        (16384, 8192, 8, 2, GridScheduler.STATIC, 1, False),
        (16384, 8192, 8, 4, GridScheduler.PERSISTENT, 1, False),
        (16384, 8192, 8, 4, GridScheduler.STATIC, 2, False),
    ],
)
def test_select_nvfp4_tma_stage_weight_scales(
    n, k, block_n, num_compute_warps, grid_scheduler, split_k, expected
):
    """Stage physical scale subsets only in the measured B200 regime."""
    if torch.cuda.get_device_capability() != (10, 0):
        expected = False
    assert (
        select_nvfp4_tma_stage_weight_scales(
            n,
            k,
            block_n,
            num_compute_warps,
            grid_scheduler,
            split_k,
        )
        is expected
    )


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack low-nibble-first E2M1 codes into the PyTorch FP4 shell dtype."""
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert natural [rows, K/16] scales to SWIZZLE_32_4_4 storage."""
    rows, cols = scales.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    padded = torch.zeros(
        (padded_rows, padded_cols),
        dtype=scales.dtype,
        device=scales.device,
    )
    padded[:rows, :cols] = scales
    return (
        padded.view(padded_rows // 128, 128, padded_cols // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
        .contiguous()
    )


def make_case(n: int, k: int):
    """Create packed values, nonuniform scales, and an FP32 reference."""
    generator = torch.Generator(device="cuda").manual_seed(n + k)
    input_codes = torch.randint(
        0, 16, (1, k), dtype=torch.uint8, device="cuda", generator=generator
    )
    weight_codes = torch.randint(
        0, 16, (n, k), dtype=torch.uint8, device="cuda", generator=generator
    )
    scale_choices = torch.tensor([0.5, 1.0, 2.0, 3.0], device="cuda")
    input_scale_ids = torch.randint(
        0,
        4,
        (1, k // 16),
        device="cuda",
        generator=generator,
    )
    weight_scale_ids = torch.randint(
        0,
        4,
        (n, k // 16),
        device="cuda",
        generator=generator,
    )
    input_scales = scale_choices[input_scale_ids].to(torch.float8_e4m3fn)
    weight_scales = scale_choices[weight_scale_ids].to(torch.float8_e4m3fn)
    input_values = FP4_VALUES.to("cuda")[input_codes.long()]
    weight_values = FP4_VALUES.to("cuda")[weight_codes.long()]
    input_dequant = input_values * input_scales.float().repeat_interleave(16, dim=1)
    weight_dequant = weight_values * weight_scales.float().repeat_interleave(16, dim=1)
    expected_fp32 = input_dequant @ weight_dequant.T
    return (
        pack_fp4(input_codes),
        pack_fp4(weight_codes),
        swizzle_scales(input_scales),
        swizzle_scales(weight_scales),
        expected_fp32.bfloat16(),
        expected_fp32,
    )


@pytest.mark.parametrize("num_compute_warps", [1, 2, 4])
def test_nvfp4_tma_matches_reference(num_compute_warps):
    """Match independently decoded NVFP4 values with distinct block scales."""
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(128, 2048)
    actual = nvfp4_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=4,
        num_compute_warps=num_compute_warps,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.05)


@pytest.mark.parametrize(("block_n", "num_compute_warps"), [(8, 4), (6, 2)])
def test_nvfp4_tma_scale_layout_and_paired_loads(block_n, num_compute_warps):
    """Match random scales across blocked-layout boundaries and odd row ownership."""
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(264, 2048)
    actual = nvfp4_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=block_n,
        num_compute_warps=num_compute_warps,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.05)


def test_nvfp4_tma_compact_scale_tma_matches_blocked_layout():
    """Match exact scales across physical 128-row atoms with compact TMA staging."""
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(264, 6144)
    actual = nvfp4_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=8,
        num_stages=3,
        num_compute_warps=4,
        stage_weight_scales=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_nvfp4_tma_fp16_partial_reduction_preserves_extreme_values():
    """Keep exact FP16 partial sums before accumulating them in FP32."""
    n, k = 128, 2048
    input_codes = torch.full((1, k), 7, dtype=torch.uint8, device="cuda")
    weight_codes = torch.full((n, k), 7, dtype=torch.uint8, device="cuda")
    scale_values = torch.tensor([0.5, 1.0, 2.0, 448.0], device="cuda")
    input_scale = scale_values[torch.arange(k // 16, device="cuda") % 4].reshape(1, -1)
    weight_scale = scale_values[torch.arange(n * (k // 16), device="cuda") * 3 % 4].reshape(n, -1)
    expected = (
        (torch.full((1, k), 6.0, device="cuda") * input_scale.repeat_interleave(16, 1))
        @ (torch.full((n, k), 6.0, device="cuda") * weight_scale.repeat_interleave(16, 1)).T
    ).bfloat16()
    actual = nvfp4_tma_gemv(
        pack_fp4(input_codes),
        pack_fp4(weight_codes),
        swizzle_scales(input_scale.to(torch.float8_e4m3fn)),
        swizzle_scales(weight_scale.to(torch.float8_e4m3fn)),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.01)


@pytest.mark.parametrize("k", [8192, 16384])
def test_nvfp4_tma_streaming_decode_matches_reference_bitwise(k):
    """Preserve the exact BF16 result across middle- and long-K streaming decode."""
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(128, k)
    actual = nvfp4_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=8,
        num_compute_warps=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_nvfp4_tma_uses_tuned_default_config():
    """Run the architecture-tuned block and warp selection when unspecified."""
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(128, 2048)
    actual = nvfp4_tma_gemv(q_input, weight, input_scale, weight_scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.05)


def test_nvfp4_tma_applies_global_scales_in_epilogue():
    """Multiply the completed FP32 accumulator by both tensorwise scales."""
    q_input, weight, input_scale, weight_scale, _, expected_fp32 = make_case(128, 2048)
    input_global_scale = torch.tensor([1.5], dtype=torch.float32, device="cuda")
    weight_global_scale = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    actual = nvfp4_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=4,
        input_global_scale=input_global_scale,
        weight_global_scale=weight_global_scale,
        num_compute_warps=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual,
        (expected_fp32 * input_global_scale * weight_global_scale).bfloat16(),
        atol=2.0,
        rtol=0.05,
    )


def test_nvfp4_tma_profiles_common_pipeline_regions():
    """Use the shared block-scaled pipeline's static profiling slots."""
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(128, 4096)
    op = get_nvfp4_tma_gemv(
        128,
        4096,
        4,
        enable_profiling=True,
        num_compute_warps=4,
    )
    with profile_session(
        max_events_per_unit=op.max_profile_events_per_cta,
        num_units=(op.num_profile_units, "CTA"),
        tag_names=list(NVFP4_TMA_PROFILE_TAGS),
        device="cuda",
    ) as (prof, tags):
        actual = op.interface(
            q_input,
            weight,
            input_scale,
            weight_scale,
            profile_buffer=prof.tensor,
        )
    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.05)
    events = decode_events(prof, tags)
    assert len(events) == op.num_profile_units * (3 * op.num_k_tiles + 1)
    assert {event.tag_name for event in events} == set(NVFP4_TMA_PROFILE_TAGS)


def test_nvfp4_tma_matches_scaled_mm_global_scale_contract():
    """Match F.scaled_mm blockwise and tensorwise NVFP4 scaling semantics."""
    from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

    q_input, weight, input_scale, weight_scale, _, _ = make_case(128, 2048)
    input_global_scale = torch.tensor([1.5], dtype=torch.float32, device="cuda")
    weight_global_scale = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    actual = nvfp4_tma_scaled_mm(
        q_input,
        weight.t(),
        input_scale,
        weight_scale,
        block_n=4,
        global_scale_a=input_global_scale,
        global_scale_b=weight_global_scale,
        num_compute_warps=4,
    )
    expected = scaled_mm(
        q_input,
        weight.t(),
        scale_a=[input_scale, input_global_scale],
        scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        scale_b=[weight_scale, weight_global_scale],
        scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        output_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.05)


def test_nvfp4_tma_split_k_requires_partial_output():
    """Require caller-owned FP32 workspace for allocation-free split-K replay."""
    q_input, weight, input_scale, weight_scale, _, _ = make_case(128, 8192)
    with pytest.raises(ValueError, match="partial_output is required"):
        nvfp4_tma_gemv(
            q_input,
            weight,
            input_scale,
            weight_scale,
            block_n=8,
            num_compute_warps=4,
            split_k=2,
        )


def test_nvfp4_tma_split_k_applies_global_scales_after_reduction():
    """Apply tensorwise scales after cancelling FP32 K partitions."""
    n, k, split_k = 8, 4096, 2
    input_codes = torch.full((1, k), 7, dtype=torch.uint8, device="cuda")
    weight_codes = torch.full((n, k), 7, dtype=torch.uint8, device="cuda")
    weight_codes[:, k // 2 :] = 15
    scales = torch.full((1, k // 16), 448.0, device="cuda").to(torch.float8_e4m3fn)
    weight_scales = scales.expand(n, -1).contiguous()
    global_scale = torch.tensor([1e30], dtype=torch.float32, device="cuda")
    output = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    partial_output = torch.empty((split_k, n), dtype=torch.float32, device="cuda")
    nvfp4_tma_gemv(
        pack_fp4(input_codes),
        pack_fp4(weight_codes),
        swizzle_scales(scales),
        swizzle_scales(weight_scales),
        block_n=8,
        num_compute_warps=4,
        split_k=split_k,
        input_global_scale=global_scale,
        weight_global_scale=torch.ones_like(global_scale),
        output=output,
        partial_output=partial_output,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


def test_nvfp4_tma_split_k_accepts_scalar_aligned_workspace():
    """Accept contiguous FP32 workspaces without imposing vector alignment."""
    n, k, split_k = 128, 8192, 2
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(n, k)
    storage = torch.empty(split_k * n + 1, dtype=torch.float32, device="cuda")
    partial_output = storage[1:].view(split_k, n)
    output = torch.empty_like(expected)
    nvfp4_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=8,
        num_compute_warps=4,
        split_k=split_k,
        output=output,
        partial_output=partial_output,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=2.0, rtol=0.05)


@pytest.mark.parametrize(
    ("k", "split_k", "num_stages", "stage_weight_scales"),
    [(8192, 2, 2, True), (16384, 4, 3, False)],
)
def test_nvfp4_tma_split_k_cuda_graph(k, split_k, num_stages, stage_weight_scales):
    """Reduce parallel K partitions in FP32 before the final BF16 conversion."""
    n = 128
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(n, k)
    output = torch.empty_like(expected)
    partial_output = torch.empty((split_k, n), dtype=torch.float32, device="cuda")
    kwargs = {
        "block_n": 8,
        "num_compute_warps": 4,
        "num_stages": num_stages,
        "split_k": split_k,
        "stage_weight_scales": stage_weight_scales,
        "output": output,
        "partial_output": partial_output,
    }
    nvfp4_tma_gemv(q_input, weight, input_scale, weight_scale, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nvfp4_tma_gemv(q_input, weight, input_scale, weight_scale, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=2.0, rtol=0.05)


def test_nvfp4_tma_compact_scale_tma_persistent_reuse():
    """Reuse compact scale stages across persistent output tiles."""
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(256, 2048)
    actual = nvfp4_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=8,
        num_compute_warps=4,
        grid_scheduler=GridScheduler.PERSISTENT,
        num_persistent_ctas=4,
        stage_weight_scales=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.05)


def test_nvfp4_tma_persistent_cuda_graph():
    """Replay persistent NVFP4 GEMV into caller-owned output."""
    q_input, weight, input_scale, weight_scale, expected, _ = make_case(256, 2048)
    output = torch.empty_like(expected)
    nvfp4_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=4,
        output=output,
        num_compute_warps=4,
        grid_scheduler=GridScheduler.PERSISTENT,
        num_persistent_ctas=4,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nvfp4_tma_gemv(
            q_input,
            weight,
            input_scale,
            weight_scale,
            block_n=4,
            output=output,
            num_compute_warps=4,
            grid_scheduler=GridScheduler.PERSISTENT,
            num_persistent_ctas=4,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=2.0, rtol=0.05)
