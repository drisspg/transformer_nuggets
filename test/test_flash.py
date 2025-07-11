import importlib.util
import pytest
import torch

if importlib.util.find_spec("triton") is None:
    pytest.skip("Triton is not available", allow_module_level=True)

from transformer_nuggets.flash import (
    attention,
    BiasMode,
    build_causal_mask,
    build_rel_mask,
)


def clone_grad_and_reset(tensor):
    cloned_grad = tensor.grad.clone()
    tensor.grad = None
    return cloned_grad


def clone_grad_and_reset_all(*tensors):
    return (clone_grad_and_reset(tensor) for tensor in tensors)


def maybe_grab_upper_section(mask, N_CTX, causal):
    BLOCK_M = 128
    if N_CTX > BLOCK_M and causal:
        # Since the kernel will not iterate over all seq_len_kv when causal
        # We will only check the minimum rectangular block
        mask = mask[:, :, :, :BLOCK_M]
    return mask


def check_bias(bias_choice, causal, attn_bias, mask, N_CTX):
    if bias_choice != BiasMode.none:
        mask = maybe_grab_upper_section(mask, N_CTX, causal)
        attn_bias = maybe_grab_upper_section(attn_bias, N_CTX, causal)
        torch.testing.assert_close(attn_bias, mask.to(attn_bias.dtype), atol=4e-2, rtol=0)


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(1, 8, 128, 16)])
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize(
    "bias_choice", [BiasMode.rel_pos, BiasMode.none, BiasMode.alibi, BiasMode.causal]
)
@pytest.mark.parametrize("sm_scale", [None, 1])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_flash_specific_masks(
    Z, H, N_CTX, D_HEAD, is_causal, bias_choice, sm_scale, dtype=torch.float16
):
    from torch.nn.attention import sdpa_kernel, SDPBackend

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
    is_causal = False
    attn_bias = None
    if bias_choice in {BiasMode.causal}:
        attn_bias = (
            build_causal_mask(N_CTX, N_CTX)
            .to(device=q.device, dtype=q.dtype)
            .expand(Z, H, N_CTX, N_CTX)
        )
    elif bias_choice in {BiasMode.rel_pos, BiasMode.alibi}:
        attn_bias = build_rel_mask(N_CTX, N_CTX, H, bias_choice, causal=is_causal)
        attn_bias = attn_bias.expand(Z, H, N_CTX, N_CTX).to(q.device).to(q.dtype)
    elif bias_choice == BiasMode.none:
        pass
    else:
        raise ValueError(f"Invalid bias_choice: {bias_choice}")

    with sdpa_kernel(SDPBackend.MATH):
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale, is_causal=is_causal, attn_mask=attn_bias
        )
    ref_out.backward(dout)
    ref_dq, ref_dk, ref_dv = clone_grad_and_reset_all(q, k, v)
    # triton implementation
    tri_out, mask = attention(q, k, v, is_causal, sm_scale, bias_choice, True)
    tri_out.half()
    tri_out.backward(dout)
    tri_dq, tri_dk, tri_dv = clone_grad_and_reset_all(q, k, v)

    # compare
    check_bias(bias_choice, is_causal, attn_bias, mask, N_CTX)

    torch.testing.assert_close(ref_out, tri_out, atol=5.8e-2, rtol=0)
    if bias_choice != BiasMode.none:
        fudge_factor = 6.1
    else:
        fudge_factor = 1
    atol = 2e-2 * fudge_factor
    if bias_choice == BiasMode.rel_pos and not is_causal:
        atol *= 4.5
    torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=0)


@pytest.mark.xfail(reason="This test is failing due to a bug in the implementation")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_flash_masked_block(dtype=torch.float16):
    from torch.nn.attention import sdpa_kernel, SDPBackend

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
    with sdpa_kernel(SDPBackend.MATH):
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale, is_causal=False, attn_mask=ref_mask
        )

    ref_out.backward(dout)
    ref_dq, ref_dk, ref_dv = clone_grad_and_reset_all(q, k, v)

    tri_out, mask = attention(q, k, v, False, sm_scale, BiasMode.inverse_causal, True)  # type: ignore

    tri_out.half()
    tri_out.backward(dout)
    tri_dq, tri_dk, tri_dv = clone_grad_and_reset_all(q, k, v)
    # Check attn_bias equivalence
    atol = 2e-2 * 6
    # compare
    check_bias(BiasMode.inverse_causal, False, ref_mask, mask, N_CTX)

    torch.testing.assert_close(ref_out, tri_out, atol=5.8e-2, rtol=0)

    torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__])
