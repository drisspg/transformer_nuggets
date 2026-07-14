import pytest
import torch
from typer.testing import CliRunner


if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
    pytest.skip("MXFP8 TMA GEMV requires SM100 or SM103", allow_module_level=True)

try:
    from transformer_nuggets.cute import (
        BlockScaleLayout,
        GridScheduler,
        MXFP8_TMA_PROFILE_TAGS,
        get_mxfp8_tma_gemv,
        mxfp8_tma_gemv,
        mxfp8_tma_scaled_mm,
        select_mxfp8_tma_compute_warps,
    )
    from transformer_nuggets.cute.mxfp8_tma import app
    from transformer_nuggets.cute.profiler import profile_session
    from transformer_nuggets.cute.profiler.host import decode_events
except ImportError:
    pytest.skip("CuTe DSL not available", allow_module_level=True)


def quantize_mxfp8(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows into E4M3 values with raw E8M0 block scales."""
    blocks = value.float().reshape(value.shape[0], -1, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(dim=-1) / 448.0)).clamp(-126, 127)
    scale = torch.exp2(exponent).unsqueeze(-1)
    quantized = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.reshape_as(value), (exponent + 127).to(torch.uint8)


@pytest.mark.parametrize(
    ("k", "block_n", "expected_sm100"),
    [(8192, 4, 1), (8192, 8, 2), (8192, 16, 4), (4096, 4, 2), (4096, 8, 4)],
)
def test_select_mxfp8_tma_compute_warps(k, block_n, expected_sm100):
    """Bake in the measured B200 rows-per-warp crossover."""
    expected = expected_sm100 if torch.cuda.get_device_capability() == (10, 0) else 1
    assert select_mxfp8_tma_compute_warps(k, block_n) == expected


def swizzle_mxfp8_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert natural E8M0 scales to canonical SWIZZLE_32_4_4 storage."""
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


def dequantize_mxfp8(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize raw MXFP8 storage to float32."""
    expanded_scale = scale.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
    return value.float() * expanded_scale


@pytest.mark.parametrize("num_compute_warps", [1, 2, 4])
@pytest.mark.parametrize(
    ("k", "block_n", "num_stages"),
    [(2048, 4, 2), (4096, 8, 3)],
)
def test_mxfp8_tma_gemv_matches_reference(k, block_n, num_stages, num_compute_warps):
    """Match independently dequantized float32 matmul."""
    torch.manual_seed(k)
    q_input, input_scale = quantize_mxfp8(torch.randn((1, k), dtype=torch.bfloat16, device="cuda"))
    weight, weight_scale = quantize_mxfp8(
        torch.randn((128, k), dtype=torch.bfloat16, device="cuda")
    )
    expected = (
        dequantize_mxfp8(q_input, input_scale) @ dequantize_mxfp8(weight, weight_scale).T
    ).bfloat16()
    output = torch.empty_like(expected)

    actual = mxfp8_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=block_n,
        num_stages=num_stages,
        output=output,
        num_compute_warps=num_compute_warps,
    )
    torch.cuda.synchronize()

    assert actual is output
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.05)


@pytest.mark.parametrize(("block_n", "num_compute_warps"), [(4, 4), (8, 4), (6, 2)])
def test_mxfp8_tma_swizzled_scales_match_raw(block_n, num_compute_warps):
    """Read canonical blocked E8M0 scales across row-layout boundaries."""
    n, k = 264, 2048
    q_input, input_scale = quantize_mxfp8(torch.randn((1, k), dtype=torch.bfloat16, device="cuda"))
    weight, weight_scale = quantize_mxfp8(torch.randn((n, k), dtype=torch.bfloat16, device="cuda"))
    expected = mxfp8_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=block_n,
        num_compute_warps=num_compute_warps,
    )
    actual = mxfp8_tma_gemv(
        q_input,
        weight,
        swizzle_mxfp8_scales(input_scale),
        swizzle_mxfp8_scales(weight_scale),
        block_n=block_n,
        num_compute_warps=num_compute_warps,
        block_scale_layout=BlockScaleLayout.SWIZZLE_32_4_4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.05)


def test_mxfp8_tma_scaled_mm_adapter_matches_scaled_mm():
    """Match scaled_mm's blocked E8M0 scale and transposed-weight contract."""
    from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

    n, k = 128, 2048
    q_input, input_scale = quantize_mxfp8(torch.randn((1, k), dtype=torch.bfloat16, device="cuda"))
    weight, weight_scale = quantize_mxfp8(torch.randn((n, k), dtype=torch.bfloat16, device="cuda"))
    input_scale = swizzle_mxfp8_scales(input_scale).view(torch.float8_e8m0fnu)
    weight_scale = swizzle_mxfp8_scales(weight_scale).view(torch.float8_e8m0fnu)
    actual = mxfp8_tma_scaled_mm(
        q_input,
        weight.t(),
        input_scale,
        weight_scale,
        block_n=8,
        num_compute_warps=4,
    )
    expected = scaled_mm(
        q_input,
        weight.t(),
        scale_a=[input_scale],
        scale_recipe_a=[ScalingType.BlockWise1x32],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4],
        scale_b=[weight_scale],
        scale_recipe_b=[ScalingType.BlockWise1x32],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4],
        output_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.05)


def test_mxfp8_tma_gemv_persistent_grid_matches_reference():
    """Reuse a bounded physical CTA grid across all logical output tiles."""
    k = 2048
    q_input, input_scale = quantize_mxfp8(torch.randn((1, k), dtype=torch.bfloat16, device="cuda"))
    weight, weight_scale = quantize_mxfp8(
        torch.randn((128, k), dtype=torch.bfloat16, device="cuda")
    )

    output = torch.empty((1, 128), dtype=torch.bfloat16, device="cuda")
    actual = mxfp8_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=4,
        output=output,
        num_compute_warps=4,
        grid_scheduler=GridScheduler.PERSISTENT,
        num_persistent_ctas=3,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mxfp8_tma_gemv(
            q_input,
            weight,
            input_scale,
            weight_scale,
            block_n=4,
            output=output,
            num_compute_warps=4,
            grid_scheduler=GridScheduler.PERSISTENT,
            num_persistent_ctas=3,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert actual is output
    expected = (
        dequantize_mxfp8(q_input, input_scale) @ dequantize_mxfp8(weight, weight_scale).T
    ).bfloat16()
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.05)


@pytest.mark.parametrize("num_compute_warps", [1, 2, 4])
def test_mxfp8_tma_gemv_cuda_graph_replay(num_compute_warps):
    """Replay into caller-owned output without hidden allocation or copies."""
    k = 2048
    q_input, input_scale = quantize_mxfp8(torch.randn((1, k), dtype=torch.bfloat16, device="cuda"))
    weight, weight_scale = quantize_mxfp8(
        torch.randn((128, k), dtype=torch.bfloat16, device="cuda")
    )
    output = torch.empty((1, 128), dtype=torch.bfloat16, device="cuda")
    mxfp8_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=4,
        output=output,
        num_compute_warps=num_compute_warps,
    )
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        mxfp8_tma_gemv(
            q_input,
            weight,
            input_scale,
            weight_scale,
            block_n=4,
            output=output,
            num_compute_warps=num_compute_warps,
        )
    graph.replay()
    torch.cuda.synchronize()

    expected = (
        dequantize_mxfp8(q_input, input_scale) @ dequantize_mxfp8(weight, weight_scale).T
    ).bfloat16()
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.05)


@pytest.mark.parametrize(
    ("grid_scheduler", "num_persistent_ctas"),
    [(GridScheduler.STATIC, None), (GridScheduler.PERSISTENT, 3)],
)
def test_mxfp8_tma_gemv_profiles_labeled_regions(grid_scheduler, num_persistent_ctas):
    """Record each compile-time-enabled region with static event slots."""
    k = 2048
    q_input, input_scale = quantize_mxfp8(torch.randn((1, k), dtype=torch.bfloat16, device="cuda"))
    weight, weight_scale = quantize_mxfp8(
        torch.randn((128, k), dtype=torch.bfloat16, device="cuda")
    )
    op = get_mxfp8_tma_gemv(
        128,
        k,
        4,
        enable_profiling=True,
        num_compute_warps=4,
        grid_scheduler=grid_scheduler,
        num_persistent_ctas=num_persistent_ctas,
    )

    with profile_session(
        max_events_per_unit=op.max_profile_events_per_cta,
        num_units=(op.num_profile_units, "CTA"),
        tag_names=list(MXFP8_TMA_PROFILE_TAGS),
        device=q_input.device,
    ) as (prof, tags):
        actual = op.interface(
            q_input,
            weight,
            input_scale,
            weight_scale,
            profile_buffer=prof.tensor,
        )

    expected = (
        dequantize_mxfp8(q_input, input_scale) @ dequantize_mxfp8(weight, weight_scale).T
    ).bfloat16()
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.05)
    events = decode_events(prof, tags)
    assert len(events) == op.num_profile_units * (3 * op.num_k_tiles + 1)
    assert {event.tag_name for event in events} == set(MXFP8_TMA_PROFILE_TAGS)
    assert {event.unit_id for event in events} == set(range(op.num_profile_units))


@pytest.mark.parametrize(("input_byte", "weight_byte"), [(254, 0), (0, 254)])
def test_mxfp8_tma_gemv_combines_cancelling_scales(input_byte, weight_byte):
    """Avoid an infinite intermediate when E8M0 scale exponents cancel."""
    k = 2048
    q_input = torch.ones((1, k), dtype=torch.float8_e4m3fn, device="cuda")
    weight = torch.ones((128, k), dtype=torch.float8_e4m3fn, device="cuda")
    input_scale = torch.full((1, k // 32), input_byte, dtype=torch.uint8, device="cuda")
    weight_scale = torch.full((128, k // 32), weight_byte, dtype=torch.uint8, device="cuda")

    actual = mxfp8_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=4,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, torch.full_like(actual, k), rtol=0, atol=0)


def test_mxfp8_tma_cli_writes_pftrace(tmp_path):
    """Run the module CLI and write a nonempty native Perfetto trace."""
    trace_path = tmp_path / "mxfp8_tma.pftrace"
    result = CliRunner().invoke(
        app,
        [
            "--n",
            "128",
            "--k",
            "2048",
            "--block-n",
            "4",
            "--num-compute-warps",
            "4",
            "--grid-scheduler",
            "persistent",
            "--num-persistent-ctas",
            "3",
            "--output",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert trace_path.stat().st_size > 0


def test_mxfp8_tma_gemv_preserves_nan_scale():
    """Decode the reserved E8M0 byte as NaN rather than infinity."""
    k = 2048
    q_input = torch.ones((1, k), dtype=torch.float8_e4m3fn, device="cuda")
    weight = torch.ones((128, k), dtype=torch.float8_e4m3fn, device="cuda")
    input_scale = torch.full((1, k // 32), 127, dtype=torch.uint8, device="cuda")
    weight_scale = torch.full((128, k // 32), 127, dtype=torch.uint8, device="cuda")
    input_scale[:, 0] = 0xFF

    actual = mxfp8_tma_gemv(
        q_input,
        weight,
        input_scale,
        weight_scale,
        block_n=4,
    )
    torch.cuda.synchronize()

    assert torch.isnan(actual).all()
