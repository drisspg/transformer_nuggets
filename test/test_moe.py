import pytest
import torch

from transformer_nuggets.moe.single_device import DeviceMOE
from transformer_nuggets.moe.unrolled import LoopMoE


def _is_b200() -> bool:
    return torch.cuda.is_available() and "B200" in torch.cuda.get_device_name(0).upper()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="CUDA bfloat16 is not available")
@pytest.mark.skipif(not _is_b200(), reason="Test is only enabled on B200")
@torch.no_grad()
def test_device_moe_matches_loop_moe_bfloat16():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    hidden_size = 64
    intermediate_size = 32
    num_experts = 8
    topk = 2

    loop_moe = LoopMoE(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
    ).to(device="cuda", dtype=torch.bfloat16)
    device_moe = DeviceMOE(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
    ).to(device="cuda", dtype=torch.bfloat16)
    device_moe.load_state_dict(loop_moe.state_dict())

    hidden_states = torch.randn(2, 4, hidden_size, device="cuda", dtype=torch.bfloat16)

    loop_out = loop_moe(hidden_states)
    device_out = device_moe(hidden_states)

    assert loop_out.dtype is torch.bfloat16
    assert device_out.dtype is torch.bfloat16
    torch.testing.assert_close(device_out, loop_out, atol=5e-2, rtol=5e-2)


def _has_flex_gemm_sm100() -> bool:
    try:
        from torch._higher_order_ops import flex_gemm  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.skipif(not _has_flex_gemm_sm100(), reason="requires FlexGEMM and an SM100 GPU")
@pytest.mark.parametrize("mode", ["fused", "save_h", "recompute"])
@pytest.mark.parametrize("glu_name", ["situ_glu", "swiglu"])
def test_grouped_glu_experts_matches_reference(glu_name, mode):
    from transformer_nuggets.moe import flex_gemm_experts as fge

    torch.manual_seed(0)
    glu = getattr(fge, glu_name)
    experts = fge.GroupedGLUExperts(256, 192, 8, glu=glu, mode=mode)
    experts = experts.to(device="cuda", dtype=torch.bfloat16)
    # Ragged experts, including an empty one.
    offs = torch.tensor([200, 13, 0, 301, 77, 150, 99, 184]).cumsum(0).to("cuda", torch.int32)
    x = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores = torch.rand(1024, device="cuda", requires_grad=True)
    dy = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)

    def run(fn, *leaves):
        for leaf in leaves:
            leaf.grad = None
        out = fn(*leaves)
        out.backward(dy.to(out.dtype))
        return [out.detach(), *(leaf.grad for leaf in leaves)]

    leaves = (x, experts.w13, experts.w2, scores)
    actual = run(lambda x, w13, w2, s: experts(x, offs, s), *leaves)
    eager = run(lambda *t: fge.reference_grouped_glu(*t[:3], offs, t[3], glu), *leaves)
    fp32 = [t.detach().float().requires_grad_() for t in leaves]
    expected = run(lambda *t: fge.reference_grouped_glu(*t[:3], offs, t[3], glu), *fp32)
    for name, a, e, ref in zip(("y", "dx", "dw13", "dw2", "dscores"), actual, eager, expected):
        rel = lambda t: ((t.float() - ref).norm() / ref.norm()).item()  # noqa: B023
        assert rel(a) <= 1.5 * rel(e) + 1e-3, f"{name}: flex {rel(a):.2e} vs eager {rel(e):.2e}"


if __name__ == "__main__":
    pytest.main([__file__])
