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


if __name__ == "__main__":
    pytest.main([__file__])
