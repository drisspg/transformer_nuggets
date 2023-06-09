import torch
from transformer_nuggets.quant import QLoRAWeight
from torch.testing import assert_close


def test_single_to_double_quantization():
    torch.manual_seed(0)
    input_weight = torch.empty(1, 16384, device="cuda", dtype=torch.bfloat16)
    input_weight = input_weight.normal_(0, 1)

    qlora = QLoRAWeight(input_weight, 64, scaler_block_size=256)
    single_quantization = qlora.scalers
    double_quantization = qlora.dequantize_scalers(qlora.quantized_scalers, qlora.quantization_factor)

    assert qlora.quantized_scalers.dtype == torch.int8
    assert qlora.scalers.dtype == input_weight.dtype

    assert_close(single_quantization, double_quantization, atol=2e-2, rtol=2e-2)