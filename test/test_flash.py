import pytest
import torch

from torch.nn.attention import sdpa_kernel, SDPBackend
from transformer_nuggets.flash import attention, BiasMode, build_rel_mask


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(6, 8, 256, 16)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("bias_choice", [BiasMode.rel_pos, BiasMode.none, BiasMode.alibi])
@pytest.mark.parametrize("sm_scale", [None, 1])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_flash_all(Z, H, N_CTX, D_HEAD, causal, bias_choice, sm_scale, dtype=torch.float16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    if sm_scale is None:
        sm_scale = 1 / (D_HEAD**0.5)
    dout = torch.randn_like(q)

    # reference implementation
    if bias_choice == BiasMode.none:
        attn_bias = None
    else:
        attn_bias = build_rel_mask(N_CTX, N_CTX, H, bias_choice, causal=causal)
        attn_bias = attn_bias.expand(Z, H, N_CTX, N_CTX).to(q.device).to(q.dtype)

    is_causal = causal if (bias_choice == BiasMode.none) else False
    with sdpa_kernel(SDPBackend.MATH):
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale, is_causal=is_causal, attn_mask=attn_bias
        )
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out, mask = attention(q, k, v, causal, sm_scale, bias_choice, True)
    tri_out.half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # Check attn_bias equivalence
    if bias_choice != BiasMode.none:
        BLOCK_M = 128
        BLOCK_N = 64
        mask = mask.half()
        if N_CTX > BLOCK_M and causal:
            # Since the kernel will not iterate over all seq_len_kv when causal
            # We will only check the minimum rectangular block
            attn_bias = attn_bias[:, :, :, :BLOCK_M]
            mask = mask[:, :, :, :BLOCK_M]
        torch.testing.assert_close(attn_bias, mask, atol=4e-2, rtol=0)

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=5.8e-2, rtol=0)
    if bias_choice != BiasMode.none:
        fudge_factor = 6.1
    else:
        fudge_factor = 1
    atol = 2e-2 * fudge_factor
    if bias_choice == BiasMode.rel_pos and not causal:
        atol *= 4.5
    torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=0)


def test_flash_masked_block(dtype=torch.float16):
    torch.manual_seed(20)
    Z, H, N_CTX, D_HEAD = (6, 8, 256, 16)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    sm_scale = 1 / (D_HEAD**0.5)

    temp_mask = torch.ones((Z, H, N_CTX, N_CTX)).tril_(-1).bool()
    ref_mask = torch.zeros_like(temp_mask, dtype=torch.float32)
    ref_mask.masked_fill_(temp_mask, float("-inf"))
    ref_mask = ref_mask.to(q.device).to(q.dtype)
    dout = torch.randn_like(q)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale, is_causal=False, attn_mask=ref_mask
        )

    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    
    tri_out, mask = attention(q, k, v, False, sm_scale, BiasMode.inverse_causal, True)  # type: ignore

    tri_out.half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # Check attn_bias equivalence
    atol = 2e-2 * 6
    torch.testing.assert_close(ref_out, tri_out, atol=5.8e-2, rtol=0)
    torch.testing.assert_close(ref_mask, mask.half(), atol=4e-2, rtol=0)
    breakpoint()
    torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=0)

if __name__ == "__main__":
    pytest.main([__file__])
