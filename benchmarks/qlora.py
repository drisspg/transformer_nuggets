import torch
import torch.nn.functional as F
import transformer_nuggets as nugs
from transformer_nuggets.quant import QLoRAWeight, QLoRAWeightDebug


bnb_available = False
try:
    import bitsandbytes as bnb

    bnb_available = True
except ImportError:
    print("Could not import bitsandbytes")


def debug_sanity_check(device):
    torch.manual_seed(0)
    input_weight = torch.empty(1, 16384, device=device, dtype=torch.bfloat16)
    input_weight = input_weight.normal_(0, 1)

    qlora_debug = QLoRAWeightDebug(input_weight, 64)
    qlora = QLoRAWeight(input_weight, 64)
    max_abs_debug = (qlora_debug.get_original_weight().to(device) - input_weight).abs().max()
    max_abs = (qlora.get_original_weight() - input_weight).abs().max()

    print(f"Max abs diff for QLoRADebug: {max_abs_debug}")
    print(f"Max abs diff for QLoRA: {max_abs}")

    
def build_llama_7b(device):
    # Lets do an actual llama size 7b
    # Number of parameters	dimension	n heads	n layers	Learn rate	Batch size	n tokens
    # 7B	                        4096	32	        32	        3.0E-04	    4M	        1T#
    # For the in projection of llama 7b thats like
    # 4096, 32*128 so just 4096 x 4096

    torch.manual_seed(0)
    input_weight = torch.empty(4096, 4096, device=device, dtype=torch.bfloat16)
    input_weight = input_weight.normal_(0, 1)
    qlora_weght = QLoRAWeight(input_weight, 64)

    # Arbitrary input
    bsz = 8
    seqlen = 128
    n_heads = 32
    head_dim = 128

    sample_input = torch.empty(bsz, seqlen, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    sample_input = sample_input.view(bsz * seqlen, n_heads * head_dim)

    return qlora_weght, input_weight, sample_input


def main(debug=True):
    device = "cuda"
    if debug:
        debug_sanity_check(device)

    qlora_weight, input_weight, sample_input = build_llama_7b(device)

    def dequant_matmul(lora_weight, input_tensor):
        return F.linear(input_tensor, lora_weight.get_original_weight())

    compile_dequant_matmul = torch.compile(dequant_matmul, fullgraph=True)
    eager_time = nugs.utils.benchmark_torch_function_in_microseconds(
        dequant_matmul, qlora_weight, sample_input
    )
    matmul_time = nugs.utils.benchmark_torch_function_in_microseconds(
        F.linear, sample_input, input_weight
    )
    # warmup
    for _ in range(3):
        compile_dequant_matmul(qlora_weight, sample_input)

    compiled_time = nugs.utils.benchmark_torch_function_in_microseconds(
        compile_dequant_matmul, qlora_weight, sample_input
    )

    print(f"Time in eager for full matmul: {matmul_time} us")
    print(f"Time in eager for dequant_matmul: {eager_time} us")
    print(f"Time for compiled dequant_matmul : {compiled_time} us")

    # Compare against bits and bytes:
    if device == "cuda" and bnb_available:
        param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4").cuda(0)
        bnb_linear = bnb.nn.LinearNF4(input_weight.size(0), input_weight.size(1))
        bnb_linear.weight = param
        bnb_linear.to(device)
        # warmup
        for _ in range(3):
            bnb_linear(sample_input)
        bnb_time = nugs.utils.benchmark_torch_function_in_microseconds(bnb_linear, sample_input)
        print(f"Time for bnb linear: {bnb_time} us")

  
    # correctness
    eager_result = dequant_matmul(qlora_weight, sample_input)
    compiled_result = compile_dequant_matmul(qlora_weight, sample_input)
    bnb_result = bnb_linear(sample_input)
    print(
        f"Max abs diff between eager and compiled: {(eager_result - compiled_result).abs().max()}"
    )
    print(f"Max abs diff between eager and bnb: {(eager_result - bnb_result).abs().max()}")


if __name__ == "__main__":
    main()
