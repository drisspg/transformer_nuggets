import pytest
import torch

from transformer_nuggets.mx.to_blocked import (
    _to_blocked_single,
    _to_blocked_single_manual,
    to_blocked,
    to_blocked_v2,
    to_blocked_manual,
    to_blocked_manual_v2,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_individual(device):
    scales = torch.randint(256, size=(128, 4), device="cuda", dtype=torch.uint8)
    single = _to_blocked_single(scales)
    single_vmap = _to_blocked_single_manual(scales)
    assert torch.equal(single, single_vmap)


@pytest.mark.parametrize("shape", [(128, 4), (256, 8), (300, 9)])
def test_rearrange(shape):
    scales = torch.randint(256, size=shape, device="cuda", dtype=torch.uint8)
    eager = to_blocked(scales)
    manual = to_blocked_manual(scales)
    assert torch.equal(eager, manual)


@pytest.mark.parametrize("shape", [(128, 4), (256, 8), (300, 9)])
def test_rearrange_v2(shape):
    scales = torch.randint(256, size=shape, device="cuda", dtype=torch.uint8)
    eager = to_blocked_v2(scales)
    manual = to_blocked_manual_v2(scales)
    assert torch.equal(eager, manual)
