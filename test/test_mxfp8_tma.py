import pytest
import torch
from typer.testing import CliRunner


if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
    pytest.skip("MXFP8 TMA GEMV requires SM100 or SM103", allow_module_level=True)

try:
    from transformer_nuggets.cute import (
        MXFP8_TMA_PROFILE_TAGS,
        get_mxfp8_tma_gemv,
        mxfp8_tma_gemv,
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


def dequantize_mxfp8(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize raw MXFP8 storage to float32."""
    expanded_scale = scale.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
    return value.float() * expanded_scale


@pytest.mark.parametrize(
    ("k", "block_n", "num_stages"),
    [(2048, 4, 2), (4096, 8, 3)],
)
def test_mxfp8_tma_gemv_matches_reference(k, block_n, num_stages):
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
    )
    torch.cuda.synchronize()

    assert actual is output
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.05)


def test_mxfp8_tma_gemv_cuda_graph_replay():
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
        )
    graph.replay()
    torch.cuda.synchronize()

    expected = (
        dequantize_mxfp8(q_input, input_scale) @ dequantize_mxfp8(weight, weight_scale).T
    ).bfloat16()
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.05)


def test_mxfp8_tma_gemv_profiles_labeled_regions():
    """Record each compile-time-enabled region with static event slots."""
    k = 2048
    q_input, input_scale = quantize_mxfp8(torch.randn((1, k), dtype=torch.bfloat16, device="cuda"))
    weight, weight_scale = quantize_mxfp8(
        torch.randn((128, k), dtype=torch.bfloat16, device="cuda")
    )
    op = get_mxfp8_tma_gemv(128, k, 4, enable_profiling=True)

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
