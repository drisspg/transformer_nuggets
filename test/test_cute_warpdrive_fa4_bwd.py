"""The FA4 SM100 backward port: dK/dV bit for bit against FA4's 1-CTA backward on the same
inputs, dQ to the TMA reduce-add order, all three against an fp32 reference, and the wrapper's
input contract. Small shapes; the benchmark covers the large ones."""

import pytest
import torch

pytest.importorskip("cutlass.cute")
pytest.importorskip("flash_attn.cute")

from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

from transformer_nuggets.cute.warpdrive import StaticPersistent, WorkStream
from transformer_nuggets.cute.warpdrive.examples.fa4_bwd import (
    FlashAttentionBackward,
    compile_fa4_bwd,
    fa4_bwd,
    fa4_config,
)

D = 128


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("warpdrive tests require SM100-family GPUs")


def inputs(b, h, sq, sk, scale, causal, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b, sq, h, D, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(b, sk, h, D, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    dout = torch.randn_like(q)
    out, lse = _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal, return_lse=True)[:2]
    return q, k, v, out, dout, lse


def reference(q, k, v, dout, scale, causal):
    """fp32 attention backward through autograd (rows without keys give zero gradients)."""
    qf, kf, vf = (t.float().transpose(1, 2).requires_grad_() for t in (q, k, v))
    s = qf @ kf.transpose(-1, -2) * scale
    if causal:
        sq, sk = s.shape[-2:]
        keep = torch.ones(sq, sk, dtype=torch.bool, device=s.device).tril(sk - sq)
        s = s.masked_fill(~keep, float("-inf"))
    (s.softmax(-1).nan_to_num() @ vf).backward(dout.float().transpose(1, 2))
    return [t.grad.transpose(1, 2) for t in (qf, kf, vf)]


SHAPES = [
    (1, 2, 128, 128),  # one Q tile: no pipelined loop iteration
    (1, 2, 384, 256),  # Q ring wraps; two key tiles
    (2, 3, 200, 330),  # ragged Q and K tails
    (1, 2, 1000, 64),  # one partial key tile, eight Q tiles
]


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_matches_fa4_and_fp32(requires_blackwell, shape, causal):
    scale = 0.3  # not the default 1/sqrt(d)
    q, k, v, out, dout, lse = inputs(*shape, scale, causal)
    ours = fa4_bwd(q, k, v, out, dout, lse, scale, causal=causal)
    theirs = _flash_attn_bwd(
        q, k, v, out, dout, lse, scale, causal, config=fa4_config(q, k, causal, True)
    )[:3]
    ref = reference(q, k, v, dout, scale, causal)
    assert torch.equal(ours[1], theirs[1]) and torch.equal(ours[2], theirs[2])
    for name, a, b, r in zip(("dq", "dk", "dv"), ours, theirs, ref):
        assert torch.isfinite(a).all(), name
        err, fa4_err = ((t.float() - r).abs().max().item() for t in (a, b))
        # dQ adds the tiles' partial sums in TMA reduce order; within FA4's own error.
        assert err <= 1.1 * fa4_err + 2e-3 * r.abs().max().item(), (name, err, fa4_err)


def test_repeated_launches_and_shapes(requires_blackwell):
    """One compiled kernel across shapes; dK/dV are deterministic, dQ is not bitwise (the
    reduce-adds of different key tiles race), but stays within reduce-order rounding."""
    scale = D**-0.5
    fa4_bwd(*inputs(1, 1, 128, 128, scale, False), scale)
    before = compile_fa4_bwd.cache_info()
    for shape in [(1, 2, 256, 384), (2, 1, 130, 129)]:
        args = inputs(*shape, scale, False)
        first, again = (fa4_bwd(*args, scale) for _ in range(2))
        assert torch.equal(first[1], again[1]) and torch.equal(first[2], again[2])
        assert torch.allclose(first[0].float(), again[0].float(), atol=2e-3, rtol=0)
    after = compile_fa4_bwd.cache_info()
    assert after.currsize == before.currsize
    assert after.hits + after.misses == before.hits + before.misses + 4


def test_one_dq_staging_buffer(requires_blackwell):
    """One dQ staging buffer computes the same dK/dV bits (dQ: the same tile sums)."""
    scale = D**-0.5
    args = inputs(2, 2, 300, 260, scale, False)
    base, knob = fa4_bwd(*args, scale), fa4_bwd(*args, scale, dq_stages=1)
    assert torch.equal(base[1], knob[1]) and torch.equal(base[2], knob[2])
    assert torch.allclose(base[0].float(), knob[0].float(), atol=2e-3, rtol=0)
    with pytest.raises(ValueError, match="dq_stages"):
        FlashAttentionBackward(dq_stages=4)


def test_describe_shows_the_derived_protocol():
    op = FlashAttentionBackward()
    text = op.describe()
    for line in (
        "0-3    reduce      152",
        "4-11   compute     136",
        "tmem P   [0, 64) (inside S)",
        "tmem dQ  [256, 384) (inside dP)",
    ):
        assert line in text, text
    assert op.s.scope.empty_counts == {0: [8]} and op.dq.scope.empty_counts == {0: [4]}
    causal = FlashAttentionBackward(causal=True).describe()
    assert "0-3    reduce      136" in causal and "4-11   compute     144" in causal


def test_rejects_unsupported_inputs(requires_blackwell):
    q, k, v, out, dout, lse = inputs(1, 2, 128, 128, D**-0.5, False)
    with pytest.raises(ValueError, match="expected"):
        fa4_bwd(q[..., :64], k[..., :64], v[..., :64], out[..., :64], dout[..., :64], lse)
    with pytest.raises(ValueError, match="no GQA"):
        fa4_bwd(q, k[:, :, :1], v[:, :, :1], out, dout, lse)
    with pytest.raises(TypeError, match="bf16"):
        fa4_bwd(q.half(), k, v, out, dout, lse)
    with pytest.raises(ValueError, match="contiguous"):
        fa4_bwd(q.transpose(1, 2).contiguous().transpose(1, 2), k, v, out, dout, lse)
    with pytest.raises(ValueError, match="lse"):
        fa4_bwd(q, k, v, out, dout, lse.transpose(1, 2))
    with pytest.raises(ValueError, match="one work item"):
        op = FlashAttentionBackward()
        op.work = WorkStream(schedule=StaticPersistent())
        compile_fa4_bwd(op)
    empty = torch.empty(1, 0, 2, D, device="cuda", dtype=torch.bfloat16)
    dq, dk, dv = fa4_bwd(empty, k, v, empty, empty, lse[..., :0])
    assert dq.shape == empty.shape and not dk.any() and not dv.any()
