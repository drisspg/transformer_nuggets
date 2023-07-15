"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import pytest
import torch

import triton
import triton.language as tl

import torch
import enum

def build_causal_mask(seq_len_q, seq_len_kv):
    temp_mask = (
        torch.ones((seq_len_q, seq_len_kv))
        .tril_()
        .bool()
    )
    mask = torch.zeros_like(temp_mask, dtype=torch.float32)
    mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    return mask

def build_alibi_mask(n_queries, n_keys, n_heads, scale=None, causal=True):
    if scale is None:
        assert n_heads%8 == 0
    m_0  = 2.0 ** (-8.0 / n_heads)
    slopes = torch.pow(m_0, torch.arange(1, 1 + n_heads))[:, None, None]
    base = -1 * (torch.arange(n_queries)[:, None] - torch.arange(n_keys)[None, :])
    if scale is not None:
        alibi_base = base * scale
    else:
        alibi_base = base * slopes
    alibi_base = alibi_base.expand(n_heads, n_queries, n_keys)
    if causal:
        causal_mask = build_causal_mask(n_queries, n_keys)
        causal_mask = causal_mask.expand(n_heads, n_queries, n_keys)
        full_mask = alibi_base + causal_mask
    else:
        full_mask = alibi_base
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

class BiasMode(enum.Enum):
    none = 0
    rel_pos = 1
    alibi = 2

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M,
    Out, mask_scratch_space,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MODE: tl.constexpr,
    BIAS_CHOICE: tl.constexpr,
    DEBUG_MASK: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    if DEBUG_MASK and BIAS_CHOICE != BiasMode.none:
        mask_block_ptr = tl.make_block_ptr(
            base = mask_scratch_space + off_hz*N_CTX*N_CTX,
            shape=(N_CTX, N_CTX),
            strides=(N_CTX, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # causal check on every loop iteration can be expensive
    # and peeling the last iteration of the loop does not work well with ptxas
    # so we have a mode to do the causal check in a separate kernel entirely
    if MODE == 0:  # entire non-causal attention
        lo, hi = 0, N_CTX
    if MODE == 1:  # entire causal attention
        lo, hi = 0, (start_m + 1) * BLOCK_M
    if MODE == 2:  # off band-diagonal
        lo, hi = 0, start_m * BLOCK_M
    if MODE == 3:  # on band-diagonal
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        m_i = tl.load(m_ptrs)
        l_i = tl.load(l_ptrs)
        acc += tl.load(O_block_ptr).to(tl.float32)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # advance block pointers to first iteration of the loop
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BIAS_CHOICE == BiasMode.rel_pos:
            qk = rel_attention_triton(qk, offs_m[:, None], (start_n + offs_n[None, :]), off_hz%H, H)
        elif BIAS_CHOICE == BiasMode.alibi:
            qk = alibi_attention_triton(qk, offs_m[:, None], (start_n + offs_n[None, :]), off_hz%H, H)
        if DEBUG_MASK and BIAS_CHOICE != BiasMode.none:
            mask = qk - tl.dot(q,k)
            if MODE == 1 or MODE == 3:
                mask = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), mask, float("-inf"))
            tl.store(mask_block_ptr, mask)

        if MODE == 1 or MODE == 3:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.math.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp(m_i - m_i_new)
        beta = tl.math.exp(m_ij - m_i_new)
        l_i *= alpha
        l_i_new = l_i + beta * l_ij
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if DEBUG_MASK and BIAS_CHOICE != BiasMode.none:
            mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # write back O
    tl.store(O_block_ptr, acc.to(tl.float16))

@triton.jit
def _bwd_preprocess(
    Out, DO, L,
    NewDO, Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    # This jumps to a block of attention 
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # O and DO is a block of output embeddings 
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # get the seqlen_q block from L
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO,
    DQ, DK, DV,
    L, M,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    num_block,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MODE: tl.constexpr,
    BIAS_CHOICE: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    #  Removed and switch to exp. Could probably
    # use the same trick with the lambda funcs to change base
    # qk_scale = sm_scale * 1.44269504
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        if MODE == 0:
            # if non_causal
            lo = 0
        else:
            # Causal
            lo = start_n * BLOCK_M
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
        m_ptrs = M + off_hz * N_CTX
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
            # NOTE: `do` is pre-divided by `l`; no normalization here
            # if MODE == 1:

            if MODE == 1:
                qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                
            # do the bias shenangians
            if BIAS_CHOICE == BiasMode.rel_pos:
                qk = rel_attention_triton(qk, offs_m_curr[:, None], (offs_n[None, :]), off_hz%H, H)
            elif BIAS_CHOICE == BiasMode.alibi:
                qk = alibi_attention_triton(qk, offs_m_curr[:, None], (offs_n[None, :]), off_hz%H, H)
            
            qk += tl.dot(q, tl.trans(k))
            # qk *= qk_scale
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.math.exp(qk - m[:, None])
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
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        batch_size, num_heads, seq_len_qv, d_head = q.shape
        # Bsize, n_heads, seq_len_q, d_head = q.shape
        # Grid is (ceil_div(seq_len_q, 128), Bsize * n_heads, 1))
        grid = (triton.cdiv(seq_len_qv, 128), batch_size * num_heads, 1)
        L = torch.empty((batch_size * num_heads, seq_len_qv), device=q.device, dtype=torch.float32)
        m = torch.empty((batch_size * num_heads, seq_len_qv), device=q.device, dtype=torch.float32)

        num_warps = 4 if Lk <= 64 else 8
        if causal:
            modes = [1] if seq_len_qv <= 2048 else [2, 3]
        else:
            modes = [0]
        # TODO delete when we are good
        if debug_mask:
            scratch_space = torch.zeros((batch_size, num_heads, seq_len_qv, seq_len_qv), device=q.device, dtype=torch.float32)
        else:
            scratch_space = None

        for mode in modes:
            _fwd_kernel[grid](
                q, k, v, sm_scale,
                L, m,
                o, scratch_space,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                batch_size, num_heads, seq_len_qv,
                BLOCK_M=128, BLOCK_N=BLOCK, BLOCK_DMODEL=Lk,
                MODE=mode,
                BIAS_CHOICE=bias_choice.value,
                DEBUG_MASK=debug_mask,
                num_warps=num_warps,
                num_stages=2)

        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.bias_choice = bias_choice
        ctx.debug_mask = debug_mask
        return o, scratch_space

    @staticmethod
    def backward(ctx, do, dmask):
        BLOCK = 128
        q, k, v, o, l, m = ctx.saved_tensors
        do = do.contiguous()
        # Higher precision, weird?
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        if ctx.causal:
            mode = 1
        else:
            mode = 0
        # launch kernel to pre process
        # grid = (triton.cdiv(seq_len_qv, 128), batch_size * num_heads, 1)
        # is full flattened
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do, l,
            do_scaled, delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        # Launch over batch_size * num_heads
        # Num_blocks (blocks to cover seq_len_qv is passed in)
        _bwd_kernel[(ctx.grid[1],)](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dq, dk, dv,
            l, m,
            delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            MODE=mode,
            BIAS_CHOICE=ctx.bias_choice.value,
            num_stages=1,
        )
        return dq, dk, dv, None, None, None, None


attention = _attention.apply