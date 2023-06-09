import unittest

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

import transformer_nuggets as nugs
import transformer_nuggets.quant as quant
from transformer_nuggets.quant import QLoRAWeight, QLoRAWeightDebug

bnb_available = False
try:
    import bitsandbytes as bnb

    bnb_available = True
except ImportError:
    print("Could not import bitsandbytes")


@pytest.mark.parametrize("scaler_block_size", [256])
@pytest.mark.parametrize("block_size", [64, 32])
def test_single_to_double_quantization(block_size: int, scaler_block_size: int):
    torch.manual_seed(0)
    input_weight = torch.empty(1, 16384, device="cuda", dtype=torch.bfloat16)
    input_weight = input_weight.normal_(0, 1)

    qlora = QLoRAWeight(input_weight, block_size)
    single_quantization = quant.get_block_absmax(input_weight.flatten(), block_size)
    double_quantization = qlora.dequantize_scalers(
        qlora.quantized_scalers, qlora.quantization_factor, scaler_block_size
    )

    assert qlora.quantized_scalers.dtype == torch.int8
    assert qlora.scalers.dtype == input_weight.dtype

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


def build_input_weight(input_shape: int, output_shape: int, device: torch.device):
    torch.manual_seed(0)
    input_weight = torch.empty(input_shape, output_shape, device=device, dtype=torch.bfloat16)
    input_weight.normal_(0, 1)
    return input_weight


def build_bitsandbytes_linear(input_weight: torch.Tensor, device: torch.device):
    param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4").cuda(device)
    bnb_linear = bnb.nn.LinearNF4(input_weight.size(0), input_weight.size(1), bias=False)
    bnb_linear.weight = param
    bnb_linear.to(device)
    return bnb_linear


def get_sample_inputs(bsz: int, seqlen: int, n_heads: int, head_dim: int, device: torch.device):
    sample_input = torch.empty(bsz, seqlen, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    sample_input = sample_input.view(bsz * seqlen, n_heads * head_dim)
    return sample_input


@unittest.skipIf(not bnb_available, "Bitsandbytes not available")
@pytest.mark.parametrize("input_shape, output_shape", [(4096, 4096)])
@pytest.mark.parametrize("compile", [True, False])
def test_bitsandbytes_parity(input_shape, output_shape, compile):
    device = torch.device("cuda:0")
    input_weight = build_input_weight(input_shape, output_shape, device)
    sample_input = get_sample_inputs(8, 128, 32, 128, device)
    bnb_linear = build_bitsandbytes_linear(input_weight, device)
    qlora_weight = QLoRAWeight(input_weight)

    def dequant_matmul(lora_weight: QLoRAWeight, input_tensor: torch.Tensor):
        return F.linear(input_tensor, lora_weight.get_original_weight())

    if compile:
        dequant_matmul = torch.compile(dequant_matmul, fullgraph=True)

    nugs_result = dequant_matmul(qlora_weight, sample_input)
    bnb_result = bnb_linear(sample_input)
    torch.testing.assert_close(nugs_result, bnb_result)
