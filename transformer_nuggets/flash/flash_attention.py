"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import enum

import torch

import triton
import triton.language as tl


class BiasMode(enum.Enum):
    none = 0
    rel_pos = 1
    alibi = 2


def build_causal_mask(seq_len_q, seq_len_kv):
    temp_mask = torch.ones((seq_len_q, seq_len_kv)).tril_().bool()
    mask = torch.zeros_like(temp_mask, dtype=torch.float32)
    mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    return mask


def build_rel_mask(
    n_queries: int,
    n_keys: int,
    n_heads: int,
    mode: BiasMode,
    causal=True,
):
    """Builds torch equivalent mask
    Args:
        n_queries: Number of queries.
        n_keys: Number of keys.
        n_heads: Number of attention heads.
        mode: Bias mode for the attention mask.
        causal: Whether to include causal mask. Defaults to True.

    Returns:
        torch.Tensor: The alibi attention mask.
    """
    if mode == BiasMode.alibi:
        assert n_heads % 8 == 0
    m_0 = 2.0 ** (-8.0 / n_heads)
    slopes = torch.pow(m_0, torch.arange(1, 1 + n_heads))[:, None, None]
    base = -1 * (torch.arange(n_queries)[:, None] - torch.arange(n_keys)[None, :])
    mask = base
    mask = mask * slopes if mode == BiasMode.alibi else mask
    mask = mask.expand(n_heads, n_queries, n_keys)
    if causal:
        causal_mask = build_causal_mask(n_queries, n_keys)
        causal_mask = causal_mask.expand(n_heads, n_queries, n_keys)
        full_mask = mask + causal_mask
    else:
        full_mask = mask
    return full_mask


@triton.jit
def rel_attention_triton(cur, m, n, head_num, num_heads):
    bias = n - m
    cur = cur + bias
    return cur


@triton.jit
def alibi_attention_triton(cur, m, n, head_num, num_heads):
    # 0 Indexing
    alibi_scale = tl.math.exp2(-((head_num + 1) * 8.0 / num_heads))
    bias = n - m
    cur = cur + (alibi_scale * bias)
    return cur


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def score_modification(
    qk,
    offs_m,
    start_n,
    offs_n,
    off_hz,
    H,
    q,
    k,
    mask_block_ptr,
    BIAS_CHOICE: tl.constexpr,
    DEBUG_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # ~~~~~~~~~~~~~~~~~~~ This is all mask stuff ~~~~~~~~~~~~~~~~~~~
    if BIAS_CHOICE == 1:
        qk = rel_attention_triton(qk, offs_m[:, None], (start_n + offs_n[None, :]), off_hz % H, H)
    elif BIAS_CHOICE == 2:
        qk = alibi_attention_triton(
            qk, offs_m[:, None], (start_n + offs_n[None, :]), off_hz % H, H
        )
    if DEBUG_MASK and BIAS_CHOICE != BiasMode.none:
        mask = qk - tl.dot(q, k)
        if IS_CAUSAL:
            mask = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), mask, float("-inf"))
        tl.store(mask_block_ptr, mask)
    # ~~~~~~~~~~~~~~~~~~~ This is the end of mask stuff ~~~~~~~~~~~~~~~~~~~
    return qk


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    """When STAGE == 3 no casual attention will happpen"""
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    Out,
    mask_scratch_space,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_CHOICE: tl.constexpr,
    DEBUG_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_batch_head_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_batch_head_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_batch_head_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh

    out_batch_head_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_batch_head_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_batch_head_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_batch_head_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + out_batch_head_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    if DEBUG_MASK and BIAS_CHOICE != BiasMode.none:
        mask_block_ptr = tl.make_block_ptr(
            base=mask_scratch_space + off_hz * N_CTX * N_CTX,
            shape=(N_CTX, N_CTX),
            strides=(N_CTX, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # TODO fix this
    # qk_scale = sm_scale * 1.44269504
    qk_scale = sm_scale
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = score_modification(
            qk,
            offs_m,
            start_n,
            offs_n,
            off_hz,
            H,
            q,
            k,
            mask_block_ptr,
            BIAS_CHOICE,
            DEBUG_MASK,
            IS_CAUSAL,
        )
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        # TODO FIX ME
        # alpha = tl.math.exp2(m_i - m_i_new)
        # p = tl.math.exp2(qk - m_i_new[:, None])
        alpha = tl.math.exp(m_i - m_i_new)
        p = tl.math.exp(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        # ~~~~~~~~~~~~~ Output Debug Mask ~~~~~~~~~~~~~~~
        if DEBUG_MASK and BIAS_CHOICE != BiasMode.none:
            mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))
    # write back l and m
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    # TODO fix me
    # tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    tl.store(l_ptrs, m_i + tl.math.log(l_i))
    # write back O
    tl.store(O_block_ptr, acc.to(tl.float16))


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    # Batch x Head postion * BLOCK_M
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load a slice of rows from O and do
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute dot prod
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    BIAS_CHOICE: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    # TODO fix me
    # qk_scale = sm_scale * 1.44269504
    qk_scale = sm_scale
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        if CAUSAL:
            lo = start_n * BLOCK_M
        else:
            lo = 0
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            if CAUSAL:
                qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.0), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk *= qk_scale
            # ~~~~~~~~~~~~~~~~~~~ This is all mask stuff ~~~~~~~~~~~~~~~~~~~
            if BIAS_CHOICE == 1:
                qk = rel_attention_triton(
                    qk, offs_m_curr[:, None], (offs_n[None, :]), off_hz % H, H
                )
            elif BIAS_CHOICE == 2:
                qk = alibi_attention_triton(
                    qk, offs_m_curr[:, None], (offs_n[None, :]), off_hz % H, H
                )
            # ~~~~~~~~~~~~~~~~~~~ This is the end of mask stuff ~~~~~~~~~~~~~~~~~~~
            l_i = tl.load(l_ptrs + offs_m_curr)
            # TODO fix me
            # p = tl.math.exp2(qk - l_i[:, None])
            p = tl.math.exp(qk - l_i[:, None])
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            # compute dq
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, bias_choice: BiasMode, debug_mask=False):
        # shape constraints
        batch_size, num_heads, seq_len_qv, d_head = q.shape
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        scratch_space = None
        if debug_mask:
            scratch_space = torch.zeros(
                (batch_size, num_heads, seq_len_qv, seq_len_qv),
                device=q.device,
                dtype=torch.float32,
            )

        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            L,
            o,
            scratch_space,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            BIAS_CHOICE=bias_choice.value,
            DEBUG_MASK=debug_mask,
            num_warps=num_warps,
            num_stages=4,
        )

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.bias_choice = bias_choice
        ctx.debug_mask = debug_mask
        return o, scratch_space

    @staticmethod
    def backward(ctx, do, dmask=None):
        BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        # Launch Grid of Batch * Num_heads
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1],)](
            o,
            do,
            delta,
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            do,
            dq,
            dk,
            dv,
            L,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            num_warps=8,
            CAUSAL=ctx.causal,
            BIAS_CHOICE=ctx.bias_choice.value,
            num_stages=1,
        )
        return dq, dk, dv, None, None, None, None


attention = _attention.apply
