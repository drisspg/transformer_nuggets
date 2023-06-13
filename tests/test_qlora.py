import unittest

import pytest
import torch

import torch.nn.functional as F


import transformer_nuggets.quant.qlora as qlora
from transformer_nuggets.quant import NF4Tensor, NF4TensorDebug, linear_nf4

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
    nugs_qlora = NF4Tensor.from_tensor(input_weight, block_size, scaler_block_size)
    debug_diff = (qlora_debug.get_original_weight().to(device) - input_weight).abs()
    diff = (nugs_qlora.get_original_weight() - input_weight).abs()

    assert abs(debug_diff.max() - diff.max()) < 1e-2


@unittest.skipIf(not bnb_available, "Bitsandbytes not available")
@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
def test_reconstruction_qlora_vs_bnb(embed_dim: int):
    torch.manual_seed(0)
    device = "cuda:0"
    input_weight = qlora.build_input_weight(embed_dim, device)
    nugs_qlora = NF4Tensor.from_tensor(input_weight)
    bnb_linear = qlora.build_bitsandbytes_linear(input_weight, device)
    # This is sneaky but don't know if there is a better way to get the reconstruction
    bnb_reconstruction = bnb_linear(
        torch.eye(embed_dim, embed_dim, dtype=torch.bfloat16, device=device)
    )
    bnb_diff = (bnb_reconstruction.T - input_weight).abs().max()
    nugs_diff = (nugs_qlora.get_original_weight() - input_weight).abs().max()
    # Since we are subtle different we assume that we both reconstruct with
    # a similar precision
    assert bnb_diff < 1
    assert nugs_diff < 1
    assert (nugs_diff - bnb_diff).abs() < 2e-1


@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("requires_grad", [True, False])
def test_autograd_func_to_eager(embed_dim: int, compile: bool, requires_grad: bool):
    torch.manual_seed(0)
    device = "cuda:0"
    input_weight = qlora.build_input_weight(embed_dim, device)
    sample_input = qlora.get_sample_inputs(8, 128, embed_dim, device, requires_grad=requires_grad)
    nugs_qlora = NF4Tensor.from_tensor(input_weight)
    if compile:
        func = torch.compile(linear_nf4, fullgraph=True)
    else:
        func = linear_nf4
    out = func(sample_input, nugs_qlora)
    if requires_grad:
        out.sum().backward()


@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
@pytest.mark.parametrize("compile", [True, False])
def test_bitsandbytes_linear_parity(embed_dim, compile):
    device = torch.device("cuda:0")
    input_weight = qlora.build_input_weight(embed_dim, device)
    sample_input = qlora.get_sample_inputs(8, 128, embed_dim, device)
    bnb_linear = qlora.build_bitsandbytes_linear(input_weight, device)
    qlora_weight = NF4Tensor.from_tensor(input_weight)

    def qlora_linear(
        input_tensor: torch.Tensor,
        lora_weight: NF4Tensor,
    ):
        return F.linear(input_tensor, lora_weight.get_original_weight())

    if compile:
        qlora_linear = torch.compile(qlora_linear, fullgraph=True)

    original_result = F.linear(sample_input, input_weight)
    nugs_result = qlora_linear(sample_input, qlora_weight)
    bnb_result = bnb_linear(sample_input)
    nugs_difference = (original_result - nugs_result).abs()
    bnb_difference = (original_result - bnb_result).abs()
    assert nugs_difference.max() < 0.5 * embed_dim
    assert bnb_difference.max() < 0.5 * embed_dim


@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
@pytest.mark.parametrize("compile", [True, False])
def test_bitsandbytes_mlp_parity(embed_dim, compile):
    device = torch.device("cuda:0")
    weights = qlora.get_mlp_weights(embed_dim, device)
    sample_input = qlora.get_sample_inputs(8, 128, embed_dim, device)

    qlora_mlp = qlora.NF4MLP(*weights)
    bnb_mlp = qlora.BnbQloraMLP(*weights, device)
    mlp = qlora.MLP(*weights)

    nugs_mlp = qlora_mlp
    if compile:
        nugs_mlp = torch.compile(qlora_mlp, fullgraph=True)

    original_result = mlp(sample_input)
    nugs_result = nugs_mlp(sample_input)
    bnb_mlp_result = bnb_mlp(sample_input)

    nugs_difference = (original_result - nugs_result).abs()
    bnb_difference = (original_result - bnb_mlp_result).abs()

    assert nugs_difference.max() < (0.5 * embed_dim) ** 2
    assert bnb_difference.max() < (0.5 * embed_dim) ** 2
