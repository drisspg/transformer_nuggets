import unittest
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

import transformer_nuggets.quant as quant
from transformer_nuggets.quant import QLoRAWeight, QLoRAWeightDebug

bnb_available = False
try:
    import bitsandbytes as bnb

    bnb_available = True
except ImportError:
    print("Could not import bitsandbytes")


def build_input_weight(embed_dim: int, device: torch.device):
    torch.manual_seed(0)
    input_weight = torch.empty(embed_dim, embed_dim, device=device, dtype=torch.bfloat16)
    input_weight.normal_(0, 1)
    return input_weight


def build_bitsandbytes_linear(input_weight: torch.Tensor, device: torch.device):
    param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4").cuda(device)
    bnb_linear = bnb.nn.LinearNF4(input_weight.size(0), input_weight.size(1), bias=False)
    bnb_linear.weight = param
    bnb_linear.to(device)
    return bnb_linear


def get_sample_inputs(bsz: int, seqlen: int, embed_dim: int, device: torch.device):
    sample_input = torch.rand(bsz, seqlen, embed_dim, device=device, dtype=torch.bfloat16)
    sample_input = sample_input.view(bsz * seqlen, embed_dim)
    return sample_input


def get_mlp_weights(
    embed_dim: int, device: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    def find_multiple(n: int, k: int) -> int:
        if n % k == 0:
            return n
        return n + k - (n % k)

    hidden_dim = 4 * embed_dim
    n_hidden = int(2 * hidden_dim / 3)
    n_hidden = find_multiple(n_hidden, 256)
    weight1 = torch.empty((n_hidden, embed_dim), dtype=torch.bfloat16, device=device).normal_(0, 1)
    weight2 = torch.empty((n_hidden, embed_dim), dtype=torch.bfloat16, device=device).normal_(0, 1)
    weight3 = torch.empty((embed_dim, n_hidden), dtype=torch.bfloat16, device=device).normal_(0, 1)

    return weight1, weight2, weight3


class MLP(nn.Module):
    def __init__(self, weight1, weight2, weight3) -> None:
        super().__init__()
        self.w1, self.w2, self.w3 = weight1, weight2, weight3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(F.linear(x, self.w1)) * F.linear(x, self.w2)
        x = F.linear(x, self.w3)
        return x


class QloraMLP(nn.Module):
    def __init__(self, weight1, weight2, weight3) -> None:
        super().__init__()
        self.w1 = QLoRAWeight(weight1)
        self.w2 = QLoRAWeight(weight2)
        self.w3 = QLoRAWeight(weight3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(F.linear(x, self.w1.get_original_weight())) * F.linear(
            x, self.w2.get_original_weight()
        )
        x = F.linear(x, self.w3.get_original_weight())
        return x


class BnbQloraMLP(nn.Module):
    def __init__(self, weight1, weight2, weight3, device) -> None:
        super().__init__()
        self.w1 = build_bitsandbytes_linear(weight1, device)
        self.w2 = build_bitsandbytes_linear(weight2, device)
        self.w3 = build_bitsandbytes_linear(weight3, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        return x


@pytest.mark.parametrize("embed_dim", [4096, 5120, 6656, 8192])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("scaler_block_size", [256])
def test_single_to_double_quantization(embed_dim: int, block_size: int, scaler_block_size: int):
    torch.manual_seed(0)
    input_weight = torch.empty(embed_dim, embed_dim, device="cuda", dtype=torch.bfloat16)
    input_weight = input_weight.normal_(0, 1)

    qlora = QLoRAWeight(input_weight, block_size)
    single_quantization = quant.get_block_absmax(input_weight.flatten(), block_size)
    double_quantization = qlora.dequantize_scalers(
        qlora.quantized_scalers, qlora.quantization_factor, scaler_block_size
    )

    assert qlora.quantized_scalers.dtype == torch.int8
    assert single_quantization.dtype == input_weight.dtype

    assert_close(single_quantization, double_quantization, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "inpt_size, block_size, scaler_block_size", [(16384, 64, 256), (256, 16, 16), (1024, 32, 32)]
)
def test_reconstruction(inpt_size: int, block_size: int, scaler_block_size: int):
    torch.manual_seed(0)
    device = "cuda"
    input_weight = torch.empty(1, inpt_size, device=device, dtype=torch.bfloat16)
    input_weight = input_weight.normal_(0, 1)

    qlora_debug = QLoRAWeightDebug(input_weight, block_size)
    qlora = QLoRAWeight(input_weight, block_size, scaler_block_size)
    debug_diff = (qlora_debug.get_original_weight().to(device) - input_weight).abs()
    diff = (qlora.get_original_weight() - input_weight).abs()

    assert abs(debug_diff.max() - diff.max()) < 1e-2


@unittest.skipIf(not bnb_available, "Bitsandbytes not available")
@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
@pytest.mark.parametrize("compile", [True, False])
def test_bitsandbytes_linear_parity(embed_dim, compile):
    device = torch.device("cuda:0")
    input_weight = build_input_weight(embed_dim, device)
    sample_input = get_sample_inputs(8, 128, embed_dim, device)
    bnb_linear = build_bitsandbytes_linear(input_weight, device)
    qlora_weight = QLoRAWeight(input_weight)

    def qlora_linear(
        input_tensor: torch.Tensor,
        lora_weight: QLoRAWeight,
    ):
        return F.linear(input_tensor, lora_weight.get_original_weight())

    if compile:
        qlora_linear = torch.compile(qlora_linear, fullgraph=True)

    nugs_result = qlora_linear(sample_input, qlora_weight)
    bnb_result = bnb_linear(sample_input)
    torch.testing.assert_close(nugs_result, bnb_result)


@unittest.skipIf(not bnb_available, "Bitsandbytes not available")
@pytest.mark.parametrize("embed_dim", [4096, 5120, 6656, 8192])
def test_bitsandbytes_mlp_parity(embed_dim):
    device = torch.device("cuda:0")
    weights = get_mlp_weights(embed_dim, device)
    sample_input = get_sample_inputs(8, 128, embed_dim, device)

    qlora_mlp = QloraMLP(*weights)
    compiled_qlora_mlp = torch.compile(qlora_mlp, fullgraph=True)
    bnb_mlp = BnbQloraMLP(*weights, device)

    qlora_mlp_result = qlora_mlp(sample_input)
    compiled_qlora_mlp_result = compiled_qlora_mlp(sample_input)
    bnb_mlp_result = bnb_mlp(sample_input)

    torch.testing.assert_close(qlora_mlp_result, compiled_qlora_mlp_result)
    torch.testing.assert_close(qlora_mlp_result, bnb_mlp_result)
