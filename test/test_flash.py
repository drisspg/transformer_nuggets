import torch
import pytest
import triton
from transformer_nuggets.flash import BiasMode, build_alibi_mask, attention


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(6, 8, 128, 16)])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('bias_choice', [BiasMode.rel_pos,  BiasMode.none, BiasMode.alibi])
def test_op(Z, H, N_CTX, D_HEAD, causal, bias_choice, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()

    sm_scale = 1
    dout = torch.randn_like(q)
   
    # reference implementation
    if bias_choice == BiasMode.rel_pos:
        attn_bias = build_alibi_mask(N_CTX, N_CTX, H, scale=1, causal=causal)
        attn_bias = attn_bias.expand(Z, H, N_CTX, N_CTX).to(q.device).to(q.dtype) 
    elif bias_choice == BiasMode.alibi:
        attn_bias = build_alibi_mask(N_CTX, N_CTX, H, scale=None, causal=causal)
        attn_bias = attn_bias.expand(Z, H, N_CTX, N_CTX).to(q.device).to(q.dtype)
    elif bias_choice == BiasMode.none:
        attn_bias = None
    is_causal = causal if (bias_choice == BiasMode.none) else False
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale,is_causal=is_causal, attn_mask=attn_bias)
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
        torch.testing.assert_close(attn_bias, mask.half(), atol=4e-2, rtol=0)
    
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=4e-2, rtol=0)
    if bias_choice != BiasMode.none:
        fudge_factor = 5
    else:
        fudge_factor = 1
    atol = 1e-2 * fudge_factor
    if bias_choice == BiasMode.rel_pos and not causal:
        atol *= 3
    torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=0)


if __name__ == '__main__':
    pytest.main([__file__])