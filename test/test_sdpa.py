from functools import partial
from typing import List, Optional, Tuple, Union

import pytest

import torch
from torch.nn.functional import scaled_dot_product_attention
from transformer_nuggets.sdpa import sdpa_prototype
from transformer_nuggets.sdpa.attn_mask import (CausalMask, CausalVariant,
                                                LambdaMask, TensorMask)


def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
):
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype

    query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


def rand_sdpa_tensor(
    shape: Tuple[Union[int, List[int]]],
    device: str,
    dtype: torch.dtype,
    type: str,
    requires_grad: bool = False,
    packed: bool = False,
) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        type (str): Nested or Dense
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    batch, seq_len, num_heads, head_dim = shape
    if type == "nested":
        if isinstance(seq_len, list):

            def _size(i):
                return (
                    (seq_len[i], num_heads, head_dim)
                    if not packed
                    else (seq_len[i], 3 * num_heads * head_dim)
                )

            return torch.nested.nested_tensor(
                [
                    torch.randn(_size(i), device=device, dtype=dtype, requires_grad=requires_grad)
                    for i in range(batch)
                ]
            )
        else:
            size = (
                (seq_len, num_heads, head_dim)
                if not packed
                else (seq_len, 3 * num_heads * head_dim)
            )
            return torch.nested.nested_tensor(
                [
                    torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)
                    for _ in range(batch)
                ]
            )
    else:
        assert isinstance(seq_len, int)
        size = (
            (batch, seq_len, num_heads, head_dim)
            if not packed
            else (batch, seq_len, 3 * num_heads * head_dim)
        )
        return torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)


def test_base_case():
    # Bsz, num_heads, seq_len, head_dim
    shape = (16, 16, 128, 16)
    make_tensor = partial(
        rand_sdpa_tensor, shape, "cuda", torch.float16, "dense", requires_grad=True
    )
    query, key, value = make_tensor(), make_tensor(), make_tensor()
    query_prototype, key_prototype, value_prototype = query_key_value_clones(query, key, value)

    pytorch_output = scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )
    sdpa_output = sdpa_prototype(
        query_prototype, key_prototype, value_prototype, None, False, dropout_p=0.0
    )

    dOut = torch.randn_like(pytorch_output)
    pytorch_output.backward(dOut)
    sdpa_output.backward(dOut)

    torch.testing.assert_close(pytorch_output, sdpa_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(query.grad, query_prototype.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(key.grad, key_prototype.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(value.grad, value_prototype.grad, rtol=1e-5, atol=1e-5)


def test_materialized_case():
    # Bsz, num_heads, seq_len, head_dim
    bsz = 16
    num_heads = 16
    seq_len = 128
    head_dim = 16
    shape = (bsz, num_heads, seq_len, head_dim)
    make_tensor = partial(
        rand_sdpa_tensor, shape, "cuda", torch.float16, "dense", requires_grad=True
    )
    query, key, value = make_tensor(), make_tensor(), make_tensor()
    query_prototype, key_prototype, value_prototype = query_key_value_clones(query, key, value)
    mask = torch.rand(bsz, num_heads, seq_len, seq_len, dtype=torch.float16, device="cuda")
    attn_mask = TensorMask(mask)

    pytorch_output = scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    sdpa_output = sdpa_prototype(
        query_prototype,
        key_prototype,
        value_prototype,
        attn_mask=attn_mask,
        scale=None,
        causal=False,
        dropout_p=0.0,
    )

    dOut = torch.randn_like(pytorch_output)
    pytorch_output.backward(dOut)
    sdpa_output.backward(dOut)

    torch.testing.assert_close(pytorch_output, sdpa_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(query.grad, query_prototype.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(key.grad, key_prototype.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(value.grad, value_prototype.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("causal_variant", [CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT])
@pytest.mark.parametrize("shapes", [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)])
def test_causal_variants(causal_variant: CausalVariant, shapes: List[Tuple[int]]):
    # Bsz, num_heads, seq_len, head_dim
    bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shapes
    if causal_variant == CausalVariant.LOWER_RIGHT and seq_len_q > seq_len_kv:
        pytest.skip("Lower right causal mask will produce NaNs in the output when seq_len_q > seq_len_kv!")
    device = torch.device("cuda")
    make_q_tensor = partial(
        rand_sdpa_tensor, (bsz, num_heads, seq_len_q, head_dim), device, torch.float16, "dense", requires_grad=True
    )
    make_kv_tensor = partial(
        rand_sdpa_tensor, (bsz, num_heads, seq_len_kv, head_dim), device, torch.float16, "dense", requires_grad=True
    )
    query, key, value = make_q_tensor(), make_kv_tensor(), make_kv_tensor()
    query_prototype, key_prototype, value_prototype = query_key_value_clones(query, key, value)
    attn_mask = CausalMask(causal_variant, seq_len_q, seq_len_kv)

    pytorch_output = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask.materialize(device), dropout_p=0.0, is_causal=False
    )
    sdpa_output = sdpa_prototype(
        query_prototype,
        key_prototype,
        value_prototype,
        attn_mask=attn_mask,
        scale=None,
        causal=False,
        dropout_p=0.0,
    )

    dOut = torch.randn_like(pytorch_output)
    pytorch_output.backward(dOut)
    sdpa_output.backward(dOut)
    atol, rtol = 5e-4, 5e-4
    grad_atol, grad_rtol = 5e-3, 5e-3
    torch.testing.assert_close(pytorch_output, sdpa_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(query.grad, query_prototype.grad, atol=grad_atol, rtol=grad_rtol)
    torch.testing.assert_close(key.grad, key_prototype.grad, atol=grad_atol, rtol=grad_rtol)
    torch.testing.assert_close(value.grad, value_prototype.grad, atol=grad_atol, rtol=grad_rtol)


@pytest.mark.parametrize("shape", [(16, 16, 128, 16), (16, 16, 52, 32)])
def test_tensor_mask(shape: List[Tuple[int]]):
    bsz, num_heads, seq_len, head_dim = shape
    device = torch.device("cuda")
    make_tensor = partial(
        rand_sdpa_tensor, shape, device, torch.float16, "dense", requires_grad=True
    )
    query, key, value = make_tensor(), make_tensor(), make_tensor()
    query_prototype, key_prototype, value_prototype = query_key_value_clones(query, key, value)
    attn_mask = TensorMask(torch.rand(bsz, num_heads, seq_len, seq_len, dtype=torch.float16, device=device))

    pytorch_output = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask.materialize(device), dropout_p=0.0, is_causal=False
    )
    sdpa_output = sdpa_prototype(
        query_prototype,
        key_prototype,
        value_prototype,
        attn_mask=attn_mask,
        scale=None,
        causal=False,
        dropout_p=0.0,
    )

    dOut = torch.randn_like(pytorch_output)
    pytorch_output.backward(dOut)
    sdpa_output.backward(dOut)
    atol, rtol = 5e-4, 5e-4
    grad_atol, grad_rtol = 5e-3, 5e-3
    torch.testing.assert_close(pytorch_output, sdpa_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(query.grad, query_prototype.grad, atol=grad_atol, rtol=grad_rtol)
    torch.testing.assert_close(key.grad, key_prototype.grad, atol=grad_atol, rtol=grad_rtol)
    torch.testing.assert_close(value.grad, value_prototype.grad, atol=grad_atol, rtol=grad_rtol)

if __name__ == "__main__":
    pytest.main([__file__])