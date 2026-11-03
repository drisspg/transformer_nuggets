"""Cut-down FlashAttention-4 forward (SM100) written with ``warp_roles``.

Scope: bf16 in/out, fp32 accumulation, head_dim 128, non-causal, no masking, work items of
(256 query rows, head, batch) scheduled by a ``TileStream`` (``OneShot`` by default, or a
persistent policy), 1-CTA MMAs. ``seqlen_k`` must be a multiple
of 128 (no key masking); ``seqlen_q`` may be any length because TMA zero-fills the loads of
the last Q tile and clips its O store. Returns O and the log-sum-exp (natural log).

The schedule mirrors FA4's non-ping-pong path (``flash_attn/cute/flash_fwd_sm100.py``):
two 128-row Q tiles per CTA (``q_stage=2``), 128-column K/V tiles, warps 0-3 softmax for
tile 0, 4-7 softmax for tile 1, 8-11 correction, 12 MMA, 13 epilogue, 14 load, 15 idle.

Dependency graph (channels, all mbarriers; slot j = Q tile j):

    load --q[j] (TMA)--> mma            Q held for the whole KV loop
    load --kv ring (TMA)--> mma         K_i and V_i alternate in one 3-stage ring
    mma --s[j] (UMMA)--> softmax[j] + correction   fan-out on S_j(i) full; the empty side is
                                        the join "P_j(i) written AND O_j rescaled" (256 arrivals)
    softmax[j] --stats[j] (SMEM payload)--> correction   per-row rescale factor, then row sum/max
    mma --s[j] final commit--> correction + softmax[j]   final PV done (O_j ready to store);
                                        its release lets the next work item reuse S_j and O_j
    correction --epi[j] (sO)--> epilogue  TMA store of O
    softmax[0] <-> softmax[1] turnstile around exp2 (FA4's s0_s1_sequence; off by default,
                                        as in FA4)

TMEM (512 columns): S0 [0,128), S1 [128,256), O0 [256,384), O1 [384,512); P_j is bf16 packed
two per column into the upper half of S_j ([64,128) of each S slot).
"""

import math
from types import SimpleNamespace

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.tensor_utils import LayoutEnum
from flash_attn.cute.softmax import SoftmaxSm100
from quack import copy_utils, layout_utils

from transformer_nuggets.cute.cache import compile_tvm_ffi_and_cache
from transformer_nuggets.cute.utils import make_fake_compact_tensor
from transformer_nuggets.cute.warp_roles import (
    Consumer,
    Instance,
    OneShot,
    Producer,
    Raster,
    Ring,
    Slots,
    Threads,
    Tiles,
    TileStream,
    Tma,
    Tmem,
    TmemPlan,
    Turn,
    Turnstile,
    Umma,
    WarpSpecialized,
    role,
)

M_BLOCK = 128
N_BLOCK = 128
HEAD_DIM = 128
Q_STAGE = 2
TILE_BYTES = M_BLOCK * HEAD_DIM * 2  # one bf16 128x128 Q, K, or V tile
RESCALE_THRESHOLD = 8.0  # FA4 bf16: skip rescaling O unless the row max grew by > 2^8
LOG2_E = math.log2(math.e)
LN2 = math.log(2.0)


def umma(tiled_mma, acc, a, b, accumulate):
    """Issue one tile MMA as K-block tcgen05.mma instructions; ``accumulate`` may be dynamic."""
    atom = cute.make_mma_atom(tiled_mma.op)
    for k in range(cute.size(a.shape[2])):
        atom.set(tcgen05.Field.ACCUMULATE, accumulate if k == 0 else True)
        cute.gemm(atom, acc, a[None, None, k], b[None, None, k], acc)


class FlashAttentionForward(WarpSpecialized):
    def __init__(
        self, kv_stages: int = 3, turnstile: bool = False, trace: bool = False, policy=None
    ):
        self.kv_stages = kv_stages
        self.q = Slots(Q_STAGE, Tma(TILE_BYTES), Umma, producer_role="load")
        self.kv = Ring(kv_stages, Tma(TILE_BYTES), Umma)
        self.s = Slots(
            Q_STAGE,
            Umma,
            Threads,
            producer_role="mma",
            consumer_roles=("softmax", "correction"),
            release_fence="tmem_store",  # P (softmax) and rescaled O (correction) via tcgen05.st
        )
        self.stats = Slots(Q_STAGE, Threads, Threads, payload=(Float32, (M_BLOCK, 2)))
        self.epi = Slots(Q_STAGE, Threads, Threads, commit_fence="smem_async")
        self.mufu = Turnstile(enabled=turnstile)
        self.tmem_plan = (
            TmemPlan()
            .region("S", N_BLOCK, Q_STAGE)
            .region("O", HEAD_DIM, Q_STAGE)
            .alias("P", of="S", offset=N_BLOCK // 2, cols=N_BLOCK // 2)
        )
        # Work items: (256-row Q block, head, batch); all Q blocks of a head run back to back so
        # its K/V stays in L2.
        self.tiles = TileStream(order=Raster(along="m"), policy=policy or OneShot())
        self.trace_protocol = trace

    def get_name(self) -> str:
        ts = "_ts" if self.mufu.enabled else ""
        return f"warp_roles_fa4_fwd_kv{self.kv_stages}{ts}_{self.tiles.get_key()}"

    def get_key(self) -> str:
        return self.get_name()

    # ------------------------------------------------------------------ roles

    @role(warps=14, regs=48)
    @cute.jit
    def load(self, q: Producer, kv: Producer, tiles: Tiles, c):
        thr_qk, thr_pv = c.mma_qk.get_slice(0), c.mma_pv.get_slice(0)
        while tiles.valid:
            m_blk, head, batch = tiles.work
            gQ = cute.local_tile(
                c.mQ[None, None, head, batch], (Q_STAGE * M_BLOCK, HEAD_DIM), (m_blk, 0)
            )
            gQ = layout_utils.select(cute.flat_divide(gQ, (M_BLOCK,)), mode=[0, 2, 1])
            load_q, _, _ = copy_utils.tma_get_copy_fn(
                c.tma_q, 0, cute.make_layout(1), thr_qk.partition_A(gQ), c.sQ
            )
            gK = cute.local_tile(c.mK[None, None, head, batch], (N_BLOCK, HEAD_DIM), (None, 0))
            gV = cute.local_tile(c.mV[None, None, head, batch], (HEAD_DIM, N_BLOCK), (0, None))
            tKsK, tKgK = cpasync.tma_partition(
                c.tma_k,
                0,
                cute.make_layout(1),
                cute.group_modes(c.sK, 0, 3),
                cute.group_modes(thr_qk.partition_B(gK), 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                c.tma_v,
                0,
                cute.make_layout(1),
                cute.group_modes(c.sV, 0, 3),
                cute.group_modes(thr_pv.partition_B(gV), 0, 3),
            )
            # FA4 order: K0, Q0, Q1, V0, then (K_i, V_i).
            with kv.produce() as h:
                cute.copy(c.tma_k, tKgK[None, 0], tKsK[None, h.index], tma_bar_ptr=h.barrier)
            for j in cutlass.range_constexpr(Q_STAGE):
                with q.produce(j) as h:
                    load_q(src_idx=j, dst_idx=j, tma_bar_ptr=h.barrier)
            with kv.produce() as h:
                cute.copy(c.tma_v, tVgV[None, 0], tVsV[None, h.index], tma_bar_ptr=h.barrier)
            for i in cutlass.range(1, c.n_blocks, unroll=1):
                with kv.produce() as h:
                    cute.copy(c.tma_k, tKgK[None, i], tKsK[None, h.index], tma_bar_ptr=h.barrier)
                with kv.produce() as h:
                    cute.copy(c.tma_v, tVgV[None, i], tVsV[None, h.index], tma_bar_ptr=h.barrier)
            tiles.advance()

    @role(warps=12, regs=48)
    @cute.jit
    def mma(self, q: Consumer, kv: Consumer, s: Producer, tiles: Tiles, tmem: Tmem, c):
        thr_qk, thr_pv = c.mma_qk.get_slice(0), c.mma_pv.get_slice(0)
        tSrQ, tSrK = thr_qk.make_fragment_A(c.sQ), thr_qk.make_fragment_B(c.sK)
        tOrV = thr_pv.make_fragment_B(c.sV)
        tS = [cute.make_tensor(tmem.ptr("S", j), c.s_layout) for j in range(Q_STAGE)]
        tO = [cute.make_tensor(tmem.ptr("O", j), c.o_layout) for j in range(Q_STAGE)]
        # P as the TMEM A operand of PV: bf16 view of the S columns (2 elements per column).
        tOrP0 = thr_pv.make_fragment_A(cute.make_tensor(tmem.base, c.p_layout.outer))
        tOrP0 = tOrP0[None, None, None, 0]
        tOrP = [
            cute.make_tensor(tOrP0.iterator + tmem.cols("P", j) * 2, tOrP0.layout)
            for j in range(Q_STAGE)
        ]
        while tiles.valid:
            # S_j(0) = Q_j K_0^T. Q handles stay open across the KV loop.
            k0 = kv.wait()
            qh = []
            for j in cutlass.range_constexpr(Q_STAGE):
                qh.append(q.wait(j))
                sp = s.acquire(j)  # previous item's final S_j released: O_j stored, P_j read
                umma(
                    c.mma_qk,
                    tS[j],
                    tSrQ[None, None, None, j],
                    tSrK[None, None, None, k0.index],
                    False,
                )
                sp.commit()
            k0.release()

            # Block i: O_j += P_j(i-1) V_(i-1); S_j(i) = Q_j K_i^T. V(i-1) is released after
            # the second PV but before K_i is: overlapping, non-nested handle lifetimes.
            for i in cutlass.range(c.n_blocks - 1, unroll=1):
                v = kv.wait()
                for j in cutlass.range_constexpr(Q_STAGE):
                    sl = s.acquire(j)  # P_j written and O_j rescaled (softmax[j], correction)
                    umma(c.mma_pv, tO[j], tOrP[j], tOrV[None, None, None, v.index], i > 0)
                    if cutlass.const_expr(j == 0):
                        k = kv.wait()
                    if cutlass.const_expr(j == Q_STAGE - 1):
                        v.release()
                    # Overwrites S_j (and the P_j alias) after PV_j read P_j: tcgen05.mma
                    # issued by one thread executes in order, so no extra barrier is needed.
                    umma(
                        c.mma_qk,
                        tS[j],
                        tSrQ[None, None, None, j],
                        tSrK[None, None, None, k.index],
                        False,
                    )
                    # No O_full commit here: correction rescales O_j only after waiting S_j(i)
                    # full, and this commit completes after every earlier MMA, including PV_j.
                    sl.commit()
                k.release()
            for h in qh:
                h.release()

            # Final PV. Its S_j commit means "O_j final": correction stores O_j, and both S
            # consumers release it, which is what lets the next item's QK_j overwrite S_j/P_j
            # and its first PV_j overwrite O_j. One commit per acquire keeps S_j balanced
            # across items (n + 1 commits, n + 1 waits per consumer).
            v_last = kv.wait()
            for j in cutlass.range_constexpr(Q_STAGE):
                with s.produce(j):
                    umma(
                        c.mma_pv,
                        tO[j],
                        tOrP[j],
                        tOrV[None, None, None, v_last.index],
                        c.n_blocks > 1,
                    )
            v_last.release()
            tiles.advance()

    @role(warps=[range(4), range(4, 8)], regs=192)
    @cute.jit
    def softmax(
        self,
        stage: Instance,
        s: Consumer[Instance],
        stats: Producer[Instance],
        mufu: Turn,
        tiles: Tiles,
        tmem: Tmem,
        c,
    ):
        tidx = cute.arch.thread_idx()[0] % 128
        thr_qk = c.mma_qk.get_slice(0)
        tSAcc = cute.make_tensor(tmem.ptr("S", stage), c.s_layout)[(None, None), 0, 0]
        tStP = cute.make_tensor(
            tmem.ptr("P", stage),
            cute.composition(tSAcc.layout, cute.make_layout((M_BLOCK, N_BLOCK // 2))),
        )
        tScS = thr_qk.partition_C(cute.make_identity_tensor((M_BLOCK, N_BLOCK)))[
            (None, None), 0, 0
        ]
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        thr_ld = tcgen05.make_tmem_copy(ld_atom, tSAcc).get_slice(tidx)
        st_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_st = tcgen05.make_tmem_copy(st_atom, tStP).get_slice(tidx)
        t = SimpleNamespace(
            tidx=tidx,
            thr_ld=thr_ld,
            thr_st=thr_st,
            tStS=thr_ld.partition_S(tSAcc),
            tStP=thr_st.partition_D(tStP),
            rS_shape=thr_ld.partition_D(tScS).shape,
            rP_shape=thr_st.partition_S(cute.make_identity_tensor((M_BLOCK, N_BLOCK // 2))).shape,
        )
        while tiles.valid:
            sm = SoftmaxSm100.create(c.scale_log2, rescale_threshold=RESCALE_THRESHOLD)
            sm.reset()
            s, stats, mufu = self.softmax_step(stage, s, stats, mufu, sm, t, True)
            for _ in cutlass.range(c.n_blocks - 1, unroll=1):
                s, stats, mufu = self.softmax_step(stage, s, stats, mufu, sm, t, False)
            with stats.produce(stage) as st:
                st.data[tidx, 0] = sm.row_sum[0]
                st.data[tidx, 1] = sm.row_max[0]
            with s.consume(stage):  # the final "O_j done" commit; nothing to read
                pass
            tiles.advance()

    def softmax_step(self, stage, s, stats, mufu, sm, t, is_first):
        """One KV block for one Q tile (plain Python: inlined into the role at trace time).

        Returns the ends so the caller's loop carries their phases.
        """
        sh = s.wait(stage)
        rS = cute.make_rmem_tensor(t.rS_shape, Float32)
        cute.copy(t.thr_ld, t.tStS, rS)
        row_max, acc_scale = sm.update_row_max(rS.load(), is_first)
        with stats.produce(stage) as st:
            st.data[t.tidx, 0] = Float32(1.0) if is_first else acc_scale
        sm.scale_subtract_rowmax(rS, row_max)
        rP32 = cute.make_rmem_tensor(t.rP_shape, Float32)
        rP = cute.make_tensor(cute.recast_ptr(rP32.iterator, dtype=cutlass.BFloat16), rS.layout)
        with mufu.turn():
            sm.apply_exp2_convert(rS, rP)
        for i in range(cute.size(t.tStP.shape[2])):  # plain Python: no preprocessor here
            cute.copy(t.thr_st, rP32[None, None, i], t.tStP[None, None, i])
        sh.release()  # layer fences the tcgen05.st of P before arriving
        sm.update_row_sum(rS.load(), acc_scale, is_first)
        return s, stats, mufu

    @role(warps=range(8, 12), regs=80)
    @cute.jit
    def correction(self, s: Consumer, stats: Consumer, epi: Producer, tiles: Tiles, tmem: Tmem, c):
        tidx = cute.arch.thread_idx()[0] % 128
        thr_pv = c.mma_pv.get_slice(0)
        tO = [cute.make_tensor(tmem.ptr("O", j), c.o_layout) for j in range(Q_STAGE)]
        while tiles.valid:
            m_blk, head, batch = tiles.work
            for _ in cutlass.range(c.n_blocks, unroll=1):
                for j in cutlass.range_constexpr(Q_STAGE):
                    with stats.consume(j) as st:
                        scale = st.data[tidx, 0]
                    # S_j(i) full: QK_j(i) was issued after PV_j(i-1), so O_j holds PV_j(i-1).
                    sh = s.wait(j)
                    if cute.arch.vote_ballot_sync(scale < 1.0) != 0:
                        self.rescale_o(thr_pv, tO[j], tidx, scale)
                    sh.release()  # joins softmax[j]'s P_j release on the same barrier
            for j in cutlass.range_constexpr(Q_STAGE):
                with stats.consume(j) as st:
                    row_sum = st.data[tidx, 0]
                    row_max = st.data[tidx, 1]
                empty_row = row_sum == 0.0 or row_sum != row_sum  # noqa: PLR0124 (NaN)
                inv_sum = cute.arch.rcp_approx(row_sum if not empty_row else 1.0)
                with s.consume(j), epi.produce(j):  # final S_j commit: O_j holds the last PV
                    self.store_o_smem(thr_pv, tO[j], tidx, inv_sum, c.sO[None, None, j])
                row = (m_blk * Q_STAGE + j) * M_BLOCK + tidx
                if row < c.seqlen_q:
                    lse = (row_max * c.scale_log2 + cute.math.log2(row_sum, fastmath=True)) * LN2
                    c.mLSE[batch, head, row] = lse if not empty_row else -Float32.inf
            tiles.advance()

    @cute.jit
    def rescale_o(self, thr_pv, tO, tidx, scale):
        """O_j *= scale for this thread's row, 16 columns at a time (FA4 correction_rescale)."""
        cols = 16
        tOcO = thr_pv.partition_C(cute.make_identity_tensor((M_BLOCK, HEAD_DIM)))
        ld = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(cols)), Float32)
        st = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(cols)), Float32)
        tO_i = cute.composition(tO, cute.make_layout((M_BLOCK, cols)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((M_BLOCK, cols)))
        thr_ld = tcgen05.make_tmem_copy(ld, tO_i).get_slice(tidx)
        thr_st = tcgen05.make_tmem_copy(st, tO_i).get_slice(tidx)
        tOtO_t2r = thr_ld.partition_S(tO_i)
        tOtO_r2t = thr_st.partition_D(tO_i)
        r_shape = thr_ld.partition_D(tOcO_i).shape
        for i in cutlass.range_constexpr(HEAD_DIM // cols):
            r = cute.make_rmem_tensor(r_shape, Float32)
            cute.copy(thr_ld, cute.make_tensor(tOtO_t2r.iterator + i * cols, tOtO_t2r.layout), r)
            for e in cutlass.range(0, cute.size(r), 2, unroll_full=True):
                r[e], r[e + 1] = cute.arch.mul_packed_f32x2((r[e], r[e + 1]), (scale, scale))
            cute.copy(thr_st, r, cute.make_tensor(tOtO_r2t.iterator + i * cols, tOtO_r2t.layout))

    @cute.jit
    def store_o_smem(self, thr_pv, tO, tidx, scale, sO):
        """sO = bf16(O_j * scale) (FA4 correction_epilogue, TMA-O path)."""
        cols = 16
        tOsO = thr_pv.partition_C(sO)
        tOcO = thr_pv.partition_C(cute.make_identity_tensor((M_BLOCK, HEAD_DIM)))
        tO_i = cute.logical_divide(tO, cute.make_layout((M_BLOCK, cols)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((M_BLOCK, cols)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((M_BLOCK, cols)))
        ld_atom = sm100_utils.get_tmem_load_op(
            (M_BLOCK, HEAD_DIM, N_BLOCK),
            LayoutEnum.ROW_MAJOR,
            cutlass.BFloat16,
            Float32,
            (M_BLOCK, cols),
            use_2cta_instrs=False,
        )
        tiled_ld = tcgen05.make_tmem_copy(ld_atom, tO_i[(None, None), 0])
        thr_ld = tiled_ld.get_slice(tidx)
        st_atom = sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, cutlass.BFloat16, Float32, tiled_ld
        )
        tiled_st = cute.make_tiled_copy_D(st_atom, tiled_ld)
        tOtO_t2r = thr_ld.partition_S(tO_i[(None, None), None])
        tOsO_r2s = copy_utils.partition_D_position_independent(thr_ld, tOsO_i[(None, None), None])
        tOcO_t2r = thr_ld.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range_constexpr(HEAD_DIM // cols):
            r = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, Float32)
            cute.copy(tiled_ld, tOtO_t2r[None, 0, 0, i], r)
            for e in cutlass.range(0, cute.size(r), 2, unroll_full=True):
                r[e], r[e + 1] = cute.arch.mul_packed_f32x2((r[e], r[e + 1]), (scale, scale))
            copy_utils.cvt_copy(tiled_st, r, tOsO_r2s[None, 0, 0, i])

    @role(warps=13, regs=48)
    @cute.jit
    def epilogue(self, epi: Consumer, tiles: Tiles, c):
        while tiles.valid:
            m_blk, head, batch = tiles.work
            gO = cute.local_tile(
                c.mO[None, None, head, batch], (Q_STAGE * M_BLOCK, HEAD_DIM), (m_blk, 0)
            )
            gO = layout_utils.select(cute.flat_divide(gO, (M_BLOCK,)), mode=[0, 2, 1])
            store_o, _, _ = copy_utils.tma_get_copy_fn(c.tma_o, 0, cute.make_layout(1), c.sO, gO)
            eh = []
            for j in cutlass.range_constexpr(Q_STAGE):
                eh.append(epi.wait(j))
                store_o(src_idx=j, dst_idx=j)
                cute.arch.cp_async_bulk_commit_group()
            for j in cutlass.range_constexpr(Q_STAGE):
                cute.arch.cp_async_bulk_wait_group(Q_STAGE - 1 - j, read=True)
                eh[j].release()
            tiles.advance()

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self,
        mma_qk,
        mma_pv,
        tma_q,
        mQ,
        tma_k,
        mK,
        tma_v,
        mV,
        tma_o,
        mO,
        mLSE,
        scale_log2,
        sQ_layout,
        sK_layout,
        sV_layout,
        p_layout,
        sO_layout,
        work,
    ):
        smem = self.begin_roles(tiles=work)
        sO = smem.allocate_tensor(cutlass.BFloat16, sO_layout.outer, 1024, sO_layout.inner)
        sQ = smem.allocate_tensor(cutlass.BFloat16, sQ_layout.outer, 1024, sQ_layout.inner)
        sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout.outer, 1024, sK_layout.inner)
        # K and V stages share one buffer (same stage size at head_dim 128).
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        thr_qk, thr_pv = mma_qk.get_slice(0), mma_pv.get_slice(0)
        ctx = SimpleNamespace(
            mma_qk=mma_qk,
            mma_pv=mma_pv,
            tma_q=tma_q,
            tma_k=tma_k,
            tma_v=tma_v,
            tma_o=tma_o,
            mQ=mQ,
            mK=mK,
            mV=mV,
            mO=mO,
            mLSE=mLSE,
            sQ=sQ,
            sK=sK,
            sV=sV,
            sO=sO,
            p_layout=p_layout,
            scale_log2=scale_log2,
            s_layout=thr_qk.make_fragment_C(thr_qk.partition_shape_C((M_BLOCK, N_BLOCK))).layout,
            o_layout=thr_pv.make_fragment_C(thr_pv.partition_shape_C((M_BLOCK, HEAD_DIM))).layout,
            n_blocks=mK.shape[0] // N_BLOCK,
            seqlen_q=mQ.shape[0],
        )
        self.run_roles(ctx)

    # ------------------------------------------------------------------- host

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        scale_log2: Float32,
        workspace: cute.Tensor,
        max_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        bf16 = cutlass.BFloat16
        # (b, s, h, d) -> (s, d, h, b); V -> (d, s, h, b) so it is the MN-major B operand of PV.
        mQ, mK, mV, mO = (
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0]))
            for t in (q, k, v, o)
        )
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[1, 0, 2, 3]))
        major = cute.nvgpu.OperandMajorMode
        mma_qk = sm100_utils.make_trivial_tiled_mma(
            bf16, bf16, major.K, major.K, Float32, tcgen05.CtaGroup.ONE, (M_BLOCK, N_BLOCK)
        )
        mma_pv = sm100_utils.make_trivial_tiled_mma(
            bf16,
            bf16,
            major.K,
            major.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            (M_BLOCK, HEAD_DIM),
            tcgen05.OperandSource.TMEM,
        )
        tiler_qk, tiler_pv = (M_BLOCK, N_BLOCK, HEAD_DIM), (M_BLOCK, HEAD_DIM, N_BLOCK)
        sQ_layout = sm100_utils.make_smem_layout_a(mma_qk, tiler_qk, bf16, Q_STAGE)
        sK_layout = sm100_utils.make_smem_layout_b(mma_qk, tiler_qk, bf16, self.kv_stages)
        p_layout = sm100_utils.make_smem_layout_a(mma_pv, tiler_pv, bf16, Q_STAGE)
        sV_layout = sm100_utils.make_smem_layout_b(mma_pv, tiler_pv, bf16, self.kv_stages)
        sO_layout = sm100_utils.make_smem_layout_epi(
            bf16, LayoutEnum.ROW_MAJOR, (M_BLOCK, HEAD_DIM), Q_STAGE
        )
        load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_A(
            load_op, mQ, cute.select(sQ_layout, mode=[0, 1, 2]), tiler_qk, mma_qk
        )
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_B(
            load_op, mK, cute.select(sK_layout, mode=[0, 1, 2]), tiler_qk, mma_qk
        )
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(
            load_op, mV, cute.select(sV_layout, mode=[0, 1, 2]), tiler_pv, mma_pv
        )
        tma_o, mO_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mO,
            cute.select(sO_layout, mode=[0, 1]),
            (M_BLOCK, HEAD_DIM),
        )
        m_blocks = cute.ceil_div(mQ.shape[0], Q_STAGE * M_BLOCK)
        work = self.tiles.problem((m_blocks, mQ.shape[2], mQ.shape[3]), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            mma_qk,
            mma_pv,
            tma_q,
            mQ_t,
            tma_k,
            mK_t,
            tma_v,
            mV_t,
            tma_o,
            mO_t,
            lse,
            scale_log2,
            sQ_layout,
            sK_layout,
            sV_layout,
            p_layout,
            sO_layout,
            work,
        ).launch(
            grid=work.grid(), block=(self.num_threads, 1, 1), stream=stream, min_blocks_per_mp=1
        )


def compile_fa4_fwd(op: FlashAttentionForward):
    """Compile once with dynamic batch, heads, and sequence lengths."""
    b, h = cute.sym_int(), cute.sym_int()
    sq, sk = cute.sym_int(), cute.sym_int(divisibility=N_BLOCK)
    bf16 = cutlass.BFloat16
    qo = (b, sq, h, HEAD_DIM)
    kv = (b, sk, h, HEAD_DIM)
    return compile_tvm_ffi_and_cache(
        op,
        op.get_key(),
        make_fake_compact_tensor(bf16, qo),
        make_fake_compact_tensor(bf16, kv),
        make_fake_compact_tensor(bf16, kv),
        make_fake_compact_tensor(bf16, qo),
        make_fake_compact_tensor(Float32, (b, h, sq), assumed_align=4),
        Float32(1.0),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        cutlass.Int32(1),
        name=op.get_name(),
    )


def fa4_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    kv_stages: int = 3,
    turnstile: bool = False,
    policy=None,
    max_ctas: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention forward for ``(batch, seqlen, heads, 128)`` bf16 tensors; returns (O, LSE).

    ``LSE`` is ``(batch, heads, seqlen_q)`` fp32 (natural log), as in FA4. ``policy``: work
    scheduling (default ``OneShot()``; ``StaticPersistent()``, ``DynamicPersistent()``,
    ``Clc()`` run several (Q block, head, batch) items per CTA). ``max_ctas`` caps the
    static/dynamic persistent grid (default: the SM count).
    """
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError("fa4_fwd expects bfloat16 q, k, v")
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM or k.shape != v.shape:
        raise ValueError(f"expected (b, s, h, {HEAD_DIM}) tensors with k.shape == v.shape")
    if k.shape[0] != q.shape[0] or k.shape[2] != q.shape[2]:
        raise ValueError("q and k/v must share batch and head counts (no GQA)")
    if k.shape[1] == 0 or k.shape[1] % N_BLOCK:
        raise ValueError(f"seqlen_k must be a positive multiple of {N_BLOCK} (no key masking)")
    if any(not t.is_contiguous() or t.data_ptr() % 16 for t in (q, k, v)):
        raise ValueError("q, k, v must be contiguous and 16-byte aligned")
    scale = softmax_scale if softmax_scale is not None else HEAD_DIM**-0.5
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], q.shape[2], q.shape[1], device=q.device, dtype=torch.float32)
    key = (kv_stages, turnstile, policy)
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = FlashAttentionForward(kv_stages, turnstile, policy=policy)
    fn = compile_fa4_fwd(op)
    workspace, sms = op.tiles.launch_args(q.device)
    fn(q, k, v, out, lse, scale * LOG2_E, workspace, max_ctas or sms)
    return out, lse


_OPS: dict = {}
