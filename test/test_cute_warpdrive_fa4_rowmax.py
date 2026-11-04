"""Hardware row-max loads preserve the unmasked FA4 example's numerical contract."""

import pytest
import torch


@pytest.fixture
def requires_sm103():
    if not torch.cuda.is_available():
        pytest.skip("requires an SM103-family GPU")
    major, minor = torch.cuda.get_device_capability()
    if major != 10 or minor < 3:
        pytest.skip("requires an SM103-family GPU")


@pytest.mark.parametrize("mode", ["single", "pair_clc", "pair_static"])
def test_fa4_hardware_rowmax_matches_software(requires_sm103, mode):
    pytest.importorskip("flash_attn.cute")
    from test_cute_warpdrive import fa4_inputs

    from transformer_nuggets.cute.warpdrive import (
        ClusterLaunchControl,
        CtaPair,
        StaticPersistent,
    )
    from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import fa4_fwd

    kwargs = {}
    if mode != "single":
        kwargs = {
            "cluster": CtaPair(),
            "schedule": ClusterLaunchControl() if mode == "pair_clc" else StaticPersistent(),
            "split_p": True,
            "max_ctas": 4,
        }
    # Ragged Q, growing K-block magnitudes, and distinct head/tile scales force rescaling.
    # The small persistent cap makes the same CTAs wrap their channel phases across items.
    for b, h, sq, sk in [(1, 2, 640, 384), (2, 3, 1024, 512)]:
        q, k, v = fa4_inputs(b, h, sq, sk, nonuniform=True)
        expected, expected_lse = fa4_fwd(q, k, v, **kwargs)
        for _ in range(3):
            actual, actual_lse = fa4_fwd(q, k, v, use_ldred_rowmax=True, **kwargs)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual_lse, expected_lse, atol=0, rtol=0)


def test_fa4_hardware_rowmax_rejects_sm100(monkeypatch):
    pytest.importorskip("flash_attn.cute")
    from cutlass.base_dsl import BaseDSL
    from cutlass.base_dsl.enums import Arch

    from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import FlashAttentionForward

    monkeypatch.setattr(BaseDSL._get_dsl(), "get_arch_enum", lambda: Arch.sm_100a)
    with pytest.raises(ValueError, match="SM103-family"):
        FlashAttentionForward(use_ldred_rowmax=True)
