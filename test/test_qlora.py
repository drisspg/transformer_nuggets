import unittest

import pytest
import torch

import torch.nn.functional as F

import transformer_nuggets.quant.qlora as qlora
from transformer_nuggets.quant.qlora import linear_nf4
from transformer_nuggets.quant.dequant_kernel import dequant_nf4_tensor
from transformer_nuggets.quant.nf4_tensor import NF4Tensor
from transformer_nuggets.quant.qlora_debug import NF4TensorDebug

bnb_available = False
try:
    import bitsandbytes as bnb

    bnb_available = True
except ImportError:
    print("Could not import bitsandbytes")


@pytest.mark.parametrize(
    "inpt_size, block_size, scaler_block_size",
    [(16384, 64, 256), (256, 16, 16), (1024, 32, 32)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_reconstruction(
    inpt_size: int, block_size: int, scaler_block_size: int, dtype: torch.dtype
):
    torch.manual_seed(0)
    device = "cuda"
    input_weight = torch.empty(1, inpt_size, device=device, dtype=dtype)
    input_weight = input_weight.normal_(0, 1)

    qlora_debug = NF4TensorDebug(input_weight, block_size)
    nugs_qlora = NF4Tensor.from_tensor(input_weight, block_size, scaler_block_size)
    debug_diff = (qlora_debug.get_original_weight().to(device) - input_weight).abs()
    diff = (nugs_qlora.get_original_weight() - input_weight).abs()

    assert abs(debug_diff.max() - diff.max()) < 1e-2


@pytest.mark.parametrize(
    "inpt_size, block_size, scaler_block_size",
    [(16384, 64, 256), (256, 16, 16), (1024, 32, 32)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_reconstruction_triton_kernel(
    inpt_size: int, block_size: int, scaler_block_size: int, dtype: torch.dtype
):
    torch.manual_seed(0)
    device = "cuda"
    input_weight = torch.empty(1, inpt_size, device=device, dtype=torch.bfloat16)
    input_weight = input_weight.normal_(0, 1)

    nugs_qlora = NF4Tensor.from_tensor(input_weight, block_size, scaler_block_size)
    pytorch_diff = (nugs_qlora.get_original_weight() - input_weight).abs()
    triton_diff = (dequant_nf4_tensor(nugs_qlora) - input_weight).abs()

    assert abs(pytorch_diff.max() - triton_diff.max()) < 1e-2
    assert (pytorch_diff - triton_diff).abs().max() < 1e-2


@unittest.skipIf(not bnb_available, "Bitsandbytes not available")
@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not bnb_available,
    reason="CUDA is not available or bitsandbytes not available",
)
def test_reconstruction_qlora_vs_bnb(embed_dim: int, dtype: torch.dtype):
    torch.manual_seed(0)
    device = "cuda"
    input_weight = qlora.build_input_weight(embed_dim, device, dtype)
    nugs_qlora = NF4Tensor.from_tensor(input_weight)
    bnb_linear = qlora.build_bitsandbytes_linear(input_weight, device)
    # This is sneaky but don't know if there is a better way to get the reconstruction
    bnb_reconstruction = bnb_linear(torch.eye(embed_dim, embed_dim, dtype=dtype, device=device))
    bnb_diff = (bnb_reconstruction.T - input_weight).abs().max()
    nugs_diff = (nugs_qlora.get_original_weight() - input_weight).abs().max()
    # Since we are subtle different we assume that we both reconstruct with
    # a similar precision
    assert bnb_diff < 1
    assert nugs_diff < 1
    assert (nugs_diff - bnb_diff).abs() < 2e-1


@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not bnb_available,
    reason="CUDA is not available or bitsandbytes not available",
)
def test_binning_distribution(embed_dim: int):
    device = "cuda"
    input_weight = qlora.build_input_weight(embed_dim, device)
    nugs_qlora = NF4Tensor.from_tensor(input_weight)
    first_elements = (nugs_qlora.quantized_data >> 4).to(torch.long)
    second_elements = (nugs_qlora.quantized_data & 0b1111).to(torch.long)

    bnb_param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4").cuda(device)
    bnb_data = bnb_param.data

    bnb_first_elements = (bnb_data >> 4).to(torch.long)
    bnb_second_elements = (bnb_data & 0b1111).to(torch.long)

    bnb_first_counts = torch.unique(bnb_first_elements, return_counts=True)[1]  # noqa: F841
    bnb_second_counts = torch.unique(bnb_second_elements, return_counts=True)[1]  # noqa: F841

    first_counts = torch.unique(first_elements, return_counts=True)[1]  # noqa: F841
    second_counts = torch.unique(second_elements, return_counts=True)[1]  # noqa: F841

    # Why are these normally distributed and not uniform?


@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_autograd_func_to_eager(
    embed_dim: int, compile: bool, requires_grad: bool, dtype: torch.dtype
):
    torch._dynamo.reset()
    torch.manual_seed(0)
    device = "cuda"
    input_weight = qlora.build_input_weight(embed_dim, device, dtype)
    sample_input = qlora.get_sample_inputs(
        8, 128, embed_dim, device, requires_grad=requires_grad, dtype=dtype
    )
    nugs_qlora = NF4Tensor.from_tensor(input_weight)

    if compile:
        func = torch.compile(qlora.linear_nf4, fullgraph=True)
    else:
        func = qlora.linear_nf4
    out = func(sample_input, nugs_qlora)
    if requires_grad:
        out.sum().backward()


@pytest.mark.parametrize("embed_dim", [256, 4096, 5120, 6656, 8192])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not bnb_available,
    reason="CUDA is not available or bitsandbytes not available",
)
def test_bitsandbytes_linear_parity(embed_dim, compile, dtype):
    device = torch.device("cuda:0")
    input_weight = qlora.build_input_weight(embed_dim, device, dtype)
    sample_input = qlora.get_sample_inputs(8, 128, embed_dim, device)
    bnb_linear = qlora.build_bitsandbytes_linear(input_weight, device)
    qlora_weight = NF4Tensor.from_tensor(input_weight)

    def qlora_linear(
        input_tensor: torch.Tensor,
        lora_weight: NF4Tensor,
    ):
        return linear_nf4(input_tensor, lora_weight)

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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not bnb_available,
    reason="CUDA is not available or bitsandbytes not available",
)
def test_bitsandbytes_mlp_parity(embed_dim, compile, dtype):
    device = torch.device("cuda:0")
    weights = qlora.get_mlp_weights(embed_dim, device, dtype)
    sample_input = qlora.get_sample_inputs(8, 128, embed_dim, device, dtype)

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


@pytest.mark.parametrize("embed_dim", [256, 4096])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("r", [1, 2])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("run_backward", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qlora_linear(
    embed_dim: int,
    compile: bool,
    r: int,
    dropout: float,
    run_backward: bool,
    dtype: torch.dtype,
):
    torch._dynamo.reset()
    torch.manual_seed(0)
    device = "cuda:0"
    # Analog for replacing first linear in MLP
    weight = qlora.get_mlp_weights(embed_dim, device, dtype)[0]
    n_hidden = weight.size(0)  # hardcode llama 7b
    nugs_qlora_linear = qlora.QloraLinear(embed_dim, n_hidden, weight, r, lora_dropout=dropout)
    func = nugs_qlora_linear
    if compile:
        func = torch.compile(nugs_qlora_linear, fullgraph=True)
    sample_input = qlora.get_sample_inputs(8, 128, embed_dim, device, dtype=dtype)
    out = func(sample_input)
    if run_backward:
        out.sum().backward()
        assert nugs_qlora_linear.lora_A.grad is not None
        assert nugs_qlora_linear.lora_B.grad is not None
