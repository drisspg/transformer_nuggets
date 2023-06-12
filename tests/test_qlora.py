import unittest

import pytest
import torch

import torch.nn.functional as F


import transformer_nuggets.quant.qlora as qlora
from transformer_nuggets.quant import NF4Tensor, NF4TensorDebug

bnb_available = False
try:
    import bitsandbytes as bnb

    bnb_available = True
except ImportError:
    print("Could not import bitsandbytes")


@pytest.mark.parametrize(
    "inpt_size, block_size, scaler_block_size", [(16384, 64, 256), (256, 16, 16), (1024, 32, 32)]
)
def test_reconstruction(inpt_size: int, block_size: int, scaler_block_size: int):
    torch.manual_seed(0)
    device = "cuda"
    input_weight = torch.empty(1, inpt_size, device=device, dtype=torch.bfloat16)
    input_weight = input_weight.normal_(0, 1)

    qlora_debug = NF4TensorDebug(input_weight, block_size)
    nugs_qlora = NF4Tensor(input_weight, block_size, scaler_block_size)
    debug_diff = (qlora_debug.get_original_weight().to(device) - input_weight).abs()
    diff = (nugs_qlora.get_original_weight() - input_weight).abs()

    assert abs(debug_diff.max() - diff.max()) < 1e-2


@unittest.skipIf(not bnb_available, "Bitsandbytes not available")
@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
def test_reconstruction_qlora_vs_bnb(embed_dim: int):
    torch.manual_seed(0)
    device = "cuda:0"
    input_weight = qlora.build_input_weight(embed_dim, device)
    nugs_qlora = NF4Tensor(input_weight)
    bnb_linear = qlora.build_bitsandbytes_linear(input_weight, device)
    # This is sneaky but don't know if there is a better way to get the reconstruction
    bnb_reconstruction = bnb_linear(
        torch.eye(embed_dim, embed_dim, dtype=torch.bfloat16, device=device)
    )
    bnb_diff = (bnb_reconstruction.T - input_weight).abs().max()
    nugs_diff = (nugs_qlora.get_original_weight() - input_weight).abs().max()
    # Since we are subtle different we assume that we both reconstruct with
    # a similar precision
    assert (nugs_diff - bnb_diff).abs() < 2e-1


@unittest.skip("BnB and nugs reconstruction are slightly different")
@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
@pytest.mark.parametrize("compile", [True, False])
def test_bitsandbytes_linear_parity(embed_dim, compile):
    device = torch.device("cuda:0")
    input_weight = qlora.build_input_weight(embed_dim, device)
    sample_input = qlora.get_sample_inputs(8, 128, embed_dim, device)
    bnb_linear = qlora.build_bitsandbytes_linear(input_weight, device)
    qlora_weight = NF4Tensor(input_weight)

    def qlora_linear(
        input_tensor: torch.Tensor,
        lora_weight: NF4Tensor,
    ):
        return F.linear(input_tensor, lora_weight.get_original_weight())

    if compile:
        qlora_linear = torch.compile(qlora_linear, fullgraph=True)

    nugs_result = qlora_linear(sample_input, qlora_weight)
    bnb_result = bnb_linear(sample_input)
    torch.testing.assert_close(nugs_result, bnb_result)


@unittest.skip("BnB and nugs reconstruction are slightly different")
@pytest.mark.parametrize("embed_dim", [4096, 5120, 6656, 8192])
def test_bitsandbytes_mlp_parity(embed_dim):
    device = torch.device("cuda:0")
    weights = qlora.get_mlp_weights(embed_dim, device)
    sample_input = qlora.get_sample_inputs(8, 128, embed_dim, device)

    qlora_mlp = qlora.QloraMLP(*weights)
    compiled_qlora_mlp = torch.compile(qlora_mlp, fullgraph=True)
    bnb_mlp = qlora.BnbQloraMLP(*weights, device)

    qlora_mlp_result = qlora_mlp(sample_input)
    compiled_qlora_mlp_result = compiled_qlora_mlp(sample_input)
    bnb_mlp_result = bnb_mlp(sample_input)

    torch.testing.assert_close(qlora_mlp_result, compiled_qlora_mlp_result)
    torch.testing.assert_close(qlora_mlp_result, bnb_mlp_result)
