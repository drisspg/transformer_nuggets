"""Persistent BF16 grouped GEMM ``out[rows_g] = A[rows_g] @ B[g]^T`` on SM100 (MoE experts).

``A: (total_M, K)`` holds the groups' rows back to back, ``B: (G, N, K)`` one weight per group,
``offsets: (G,)`` int32 on the device gives each group's end row (``torch._grouped_mm``'s
``offs``), so the host knows only ``G``, ``total_M``, ``N`` and ``K``. The work items are
``(group, m_tile, n_tile)`` tiles decoded on the device by ``GroupedTiles`` from the offsets;
roles, channels and the epilogue are those of ``persistent_gemm.py`` with the tile's rows taken
from the item instead of a tile index. One TMA descriptor serves every group: A and ``out`` are
addressed at the group's row (``cute.domain_offset``) and B is a 3-D ``(N, K, G)`` tensor
with the group as a coordinate, so no per-group descriptors have to be built or patched on the
device.

A group's last m-tile is ragged (its rows end inside the tile) unless it is also the last rows
of ``A``. TMA stores clip only at the tensor's edge, so such a tile is stored with predicated
register stores and every other tile through SMEM with TMA (``epilogue``). The loads of the
ragged tile read the next group's rows into the unused part of the A stage, which only
affects accumulator rows that are never stored.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from transformer_nuggets.cute.jit_cache import jit_cache
from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    AtomicPersistent,
    Consumer,
    Producer,
    Ring,
    RoleConfig,
    SingleCta,
    SmemPlan,
    StageView,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkMap,
    WorkStream,
    role,
)
from transformer_nuggets.cute.warpdrive.examples.persistent_gemm import (
    CTA_TILE_M,
    EPI_BARRIER_ID,
    EPI_TILE_NS,
    MAX_C_STAGES,
    TILE_K,
)


@cute.jit
def _locate_tile(mt: Int32, offsets: cute.Tensor, tile_m: int):
    """The group of global m-tile ``mt``, that tile's first row and its valid rows:
    ``(found, group, row0, rows)``.

    Group g's tiles start at m-tile ``ceil(P_g / tile_m)`` where ``P_g = offsets[g-1] +
    g * tile_m``: the padded row space gives every group a gap of one tile, so its real tiles
    fit in its range of m-tiles and at most one slot per group is unused (``found`` is false
    there). ``P_g`` is strictly increasing, so a binary search over the offsets finds the group
    in ``log2(G)`` loads instead of a scan.
    """
    G = offsets.shape[0]
    step = Int32(1)
    while step < G:
        step = step * 2
    lo = Int32(0)  # invariant: group lo starts at or before mt
    while step > 1:
        step = step // 2
        cand = lo + step
        fits = (cand < G) & (offsets[cutlass.min(cand, G) - 1] <= (mt - cand) * tile_m)
        lo = cutlass.select_(fits, cand, lo)
    start_row = cutlass.select_(lo > 0, offsets[cutlass.max(lo, 1) - 1], Int32(0))
    first_tile = (start_row + lo * tile_m + tile_m - 1) // tile_m
    row0 = start_row + (mt - first_tile) * tile_m
    end = offsets[lo]
    return row0 < end, lo, row0, cutlass.min(end - row0, tile_m)


@dataclass(frozen=True)
class GroupedTiles(WorkMap):
    """``(group, row0, rows, n_tile)`` items of a grouped GEMM, decoded on the device.

    The space is ``(offsets, total_m, n_tiles)``. The host cannot count the groups' m-tiles
    (``sum(ceil(size_g / tile_m))``), so the index space covers ``ceil(total_m / tile_m) + G``
    m-tiles, an upper bound; the unused slot of a group (see ``_locate_tile``) is dropped. n is
    the fast axis: consecutive items share a tile's rows of A.
    """

    tile_m: int = CTA_TILE_M
    words = 4
    skips = True

    def bind(self, space):
        offsets, total_m, n_tiles = space
        return offsets, Int32(total_m), Int32(cutlass.max(n_tiles, 0))

    def count64(self, space):
        offsets, total_m, n_tiles = space
        m_tiles = (total_m + self.tile_m - 1) // self.tile_m + offsets.shape[0]
        return Int64(m_tiles) * Int64(n_tiles)

    def decode(self, index, space):
        offsets, _, n_tiles = space
        found, group, row0, rows = _locate_tile(index // n_tiles, offsets, self.tile_m)
        return found, (group, row0, rows, index % n_tiles)

    def get_key(self) -> str:
        return f"grouped{self.tile_m}"

    def describe(self) -> str:
        return f"GroupedTiles(tile_m={self.tile_m})"


@dataclass(frozen=True)
class GroupedGemmParams:
    """Kernel-prologue values the roles read; built once in ``GroupedGemm.kernel``."""

    tiled_mma: cute.TiledMma
    tma_a: cute.CopyAtom
    tma_b: cute.CopyAtom
    mA: cute.Tensor  # TMA coordinate tensor of A (total_M, K)
    sA: cute.Tensor  # the A stages, for the per-item TMA partition
    tAsA: StageView  # TMA partitions of the A and B stages: (TMA, stage)
    tBsB: StageView
    tBgB: cute.Tensor  # whole-problem B tiles: (TMA, n_tiles, k_tiles, G)
    tCrA: StageView  # MMA fragments over the SMEM stages: (MMA, MMA_M, MMA_K, stage)
    tCrB: StageView
    num_k_tiles: cutlass.Int32
    acc_layout: cute.Layout  # all accumulator stages
    epi_tile: cute.Tile
    tma_c: cute.CopyAtom
    mC_tma: cute.Tensor  # TMA coordinate tensor of the output (total_M, N)
    mC: cute.Tensor  # the output itself, for predicated stores
    cC: cute.Tensor  # its coordinates
    sC: cute.Tensor  # epilogue staging buffers
    total_m: cutlass.Int32


class GroupedGemm(WarpSpecialized):
    def __init__(
        self,
        schedule=None,
        ab_stages: int | None = None,
        acc_stages: int = 2,
        trace: bool = False,
        cluster=None,
        tile_n: int = 256,
        c_stages: int = 2,
        epi_tile_n: int = 32,
        roles: dict[str, RoleConfig] | None = None,
    ):
        """``ab_stages``: A/B ring depth (default: what fills ~192 KB). ``acc_stages``: TMEM
        accumulators. ``tile_n``: output tile width. ``c_stages`` SMEM buffers of
        ``128 x epi_tile_n`` stage the TMA-store epilogue. ``roles``: placement and register
        budgets by role name over the ``@role`` defaults (40/40/160)."""
        self.cluster = cluster or SingleCta()
        cs = self.cluster.pair_size  # CTAs per MMA
        if self.cluster.size != cs:
            raise ValueError(f"GroupedGemm runs on one CTA or a CTA pair, got {self.cluster}")
        if epi_tile_n not in EPI_TILE_NS:
            raise ValueError(f"epi_tile_n must be one of {EPI_TILE_NS}, got {epi_tile_n}")
        if tile_n % epi_tile_n or not 0 < tile_n <= 256 or tile_n // cs % 8:
            raise ValueError(
                f"tile_n must be a multiple of epi_tile_n={epi_tile_n} in (0, 256], got {tile_n}"
            )
        if not 1 <= c_stages <= MAX_C_STAGES:
            raise ValueError(f"c_stages must be in [1, {MAX_C_STAGES}], got {c_stages}")
        self.tile_n = tile_n
        self.epi_tile = (CTA_TILE_M, epi_tile_n)
        self.c_stages = c_stages
        self.roles = roles
        cta_tile = (CTA_TILE_M, tile_n, TILE_K)
        self.mma_tile = (cta_tile[0] * cs, cta_tile[1], cta_tile[2])
        self.dtype = cutlass.BFloat16
        # Attributes read by the CUTLASS epilogue partition helpers.
        self.c_dtype, self.acc_dtype = self.dtype, cutlass.Float32
        self.cta_tile_shape_mnk = cta_tile
        self.c_layout = LayoutEnum.ROW_MAJOR
        self.acc_stages = acc_stages
        stage_bytes = (cta_tile[0] + cta_tile[1] // cs) * cta_tile[2] * 2  # this CTA's copies
        if ab_stages is None:  # fill ~192 KB of SMEM
            ab_stages = 192 * 1024 // stage_bytes
        self.smem_plan = SmemPlan()
        a = self.smem_plan.buffer("A", self.dtype, align=1024)
        b = self.smem_plan.buffer("B", self.dtype, align=1024)
        self.smem_plan.buffer("C", self.dtype, align=128)
        self.ab = Ring(ab_stages, Tma(stage_bytes), Umma, buffer={"A": a, "B": b})
        self.acc = Ring(acc_stages, Umma, Threads)
        self.tmem_plan = TmemPlan().region("acc", cta_tile[1], acc_stages)
        # Decoding an item reads the offsets (a binary search, ~log2(G) dependent loads). A
        # broadcast schedule does that once per item in the scheduler warp, ahead of the roles;
        # a local one (StaticPersistent, OneShot) repeats it in every role between tiles,
        # which measured ~30% slower at G=128, K=2048.
        self.work = WorkStream(
            mapping=GroupedTiles(tile_m=self.mma_tile[0]), schedule=schedule or AtomicPersistent()
        )
        self.trace_protocol = trace

    def get_name(self) -> str:
        return (
            f"warpdrive_ggemm_{self.cluster.get_key()}_n{self.tile_n}_ab{self.ab.n}_"
            f"acc{self.acc_stages}_c{self.c_stages}x{self.epi_tile[1]}_"
            f"{self.work.get_key()}{self.iket_key()}{self.trace_key()}"
        )

    # ------------------------------------------------------------------ roles

    @role(warps=4, regs=40)
    @cute.jit
    def load(self, params: GroupedGemmParams, *, ab: Producer, work: WorkCursor):
        ab = ab.views(A=params.tAsA, B=params.tBsB)
        thr_mma = params.tiled_mma.get_slice(self.cluster.coords(self.cluster.rank())[0])
        while work.valid:
            g, row0, _, nt = work.item
            # This item's rows of A, tiled along K: (TM, TK, k_tiles) -> (TMA, k_tiles).
            gA = cute.local_tile(
                cute.domain_offset((row0, 0), params.mA),
                self.mma_tile,
                (0, None, None),
                proj=(1, None, 1),
            )
            _, tAgA = cpasync.tma_partition(
                params.tma_a,
                0,
                cute.make_layout(1),
                cute.group_modes(params.sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBgB = params.tBgB[None, nt, None, g]
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.produce() as s:
                    a, b = s.view("A"), s.view("B")
                    cute.copy(params.tma_a, tAgA[None, k], a, tma_bar_ptr=s.barrier)
                    cute.copy(params.tma_b, tBgB[None, k], b, tma_bar_ptr=s.barrier)
            work.advance()
        return ab, work

    @role(warps=5, regs=40, cta="leader")
    @cute.jit
    def mma(
        self,
        params: GroupedGemmParams,
        *,
        ab: Consumer,
        acc: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        tiled_mma = params.tiled_mma
        tCtAcc = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        ab = ab.views(A=params.tCrA, B=params.tCrB)
        while work.valid:
            with acc.produce() as a:
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range(params.num_k_tiles, unroll=1):
                    with ab.consume() as s:
                        for kb in cutlass.range_constexpr(cute.size(params.tCrA.tensor, mode=[2])):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc[None, None, None, a.index],
                                s.view("A", None, None, kb),
                                s.view("B", None, None, kb),
                                tCtAcc[None, None, None, a.index],
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            work.advance()
        return ab, acc, work

    @role(warps=range(4), regs=160)
    @cute.jit
    def epilogue(self, params: GroupedGemmParams, *, acc: Consumer, work: WorkCursor, tmem: Tmem):
        """TMEM -> registers -> swizzled SMEM -> TMA store, one ``epi_tile`` subtile at a time
        through ``c_stages`` SMEM buffers, for a tile whose rows all belong to its group (or end
        at the tensor's edge, where TMA clips). A tile that ends inside ``A`` before its group's
        last row is stored from registers instead, 16 bytes at a time, skipping the rows of the
        next group and the columns past N."""
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = self.cluster.coords(self.cluster.rank())[0]  # this CTA's rows of the MMA tile
        thr_mma = params.tiled_mma.get_slice(rank)
        tAcc = transform_partitioned_tensor_layout(
            cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        )
        # Whole-problem tiles only size the register fragment; items address their own rows.
        gC_all = transform_partitioned_tensor_layout(
            thr_mma.partition_C(
                cute.local_tile(params.mC, self.mma_tile, (None, None, None), proj=(1, 1, None))
            )
        )
        gC_l = cute.make_tensor(gC_all.iterator, cute.append(gC_all.layout, cute.make_layout(1)))
        t2r, tTR_tAcc, tTR_rAcc = epilogue_tmem_copy_and_partition(
            self, tidx, tAcc, gC_l, params.epi_tile, self.cluster.pair
        )
        thr_t2r = t2r.get_slice(tidx)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.dtype)
        r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(self, t2r, tTR_rC, tidx, params.sC)
        bSG_sC, _ = cpasync.tma_partition(
            params.tma_c,
            0,
            cute.make_layout(1),
            cute.group_modes(params.sC, 0, 2),
            cute.group_modes(cute.flat_divide(gC_all, params.epi_tile), 0, 2),
        )
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc) - 1)  # (..., SUB, STAGE)
        subtiles = cute.size(tTR_tAcc, mode=[3])
        # A thread's fragment is one row of 32 consecutive columns: store it as 16-byte vectors.
        vec = cute.make_layout((8, cute.size(tTR_rC) // 8))
        rC_v = cute.make_tensor(tTR_rC.iterator, vec)
        n_cols = params.mC.shape[1]
        # Buffers rotate across tiles too: with a subtile count that is not a multiple of
        # c_stages (tile_n=224: 7), a tile's first subtile alternates buffers.
        rotate = subtiles % self.c_stages != 0
        first = Int32(0)  # the buffer of this tile's first subtile
        while work.valid:
            _, row0, rows, nt = work.item
            row0 = row0 + CTA_TILE_M * rank
            rows = cutlass.max(cutlass.min(rows - CTA_TILE_M * rank, CTA_TILE_M), 0)
            row_end = row0 + rows
            # TMA clips at the tensor's edge, not at the group's.
            tma_ok = (rows == CTA_TILE_M) | (row_end == params.total_m)
            gC, cC, gC_tma = (
                cute.flat_divide(
                    transform_partitioned_tensor_layout(
                        thr_mma.partition_C(
                            cute.local_tile(
                                cute.domain_offset((row0 - CTA_TILE_M * rank, 0), t),
                                self.mma_tile[:2],
                                (0, nt),
                            )
                        )
                    ),
                    params.epi_tile,
                )
                for t in (params.mC, params.cC, params.mC_tma)
            )
            tTR_gC, tTR_cC = (thr_t2r.partition_D(t) for t in (gC, cC))
            _, bSG_gC = cpasync.tma_partition(
                params.tma_c,
                0,
                cute.make_layout(1),
                cute.group_modes(params.sC, 0, 2),
                cute.group_modes(gC_tma, 0, 2),
            )
            with acc.consume() as a:
                for i in cutlass.range_constexpr(subtiles):
                    buf = (first + i) % self.c_stages if rotate else i % self.c_stages
                    cute.copy(t2r, tTR_tAcc[None, None, None, i, a.index], tTR_rAcc)
                    tTR_rC.store(tTR_rAcc.load().to(self.dtype))
                    if tma_ok:
                        cute.copy(r2s, tRS_rC, tRS_sC[None, None, None, buf])
                        cute.arch.fence_view_async_shared()  # st.shared -> TMA store
                        cute.arch.barrier(barrier_id=EPI_BARRIER_ID, number_of_threads=128)
                        if warp == 0:
                            cute.copy(params.tma_c, bSG_sC[None, buf], bSG_gC[None, 0, i])
                            cute.arch.cp_async_bulk_commit_group()
                            # Buffer `buf` is rewritten c_stages subtiles later; keep at most
                            # c_stages - 1 stores still reading SMEM.
                            cute.arch.cp_async_bulk_wait_group(self.c_stages - 1, read=True)
                        cute.arch.barrier(barrier_id=EPI_BARRIER_ID, number_of_threads=128)
                    else:
                        g = cute.make_tensor(tTR_gC[None, None, None, 0, i].iterator, vec)
                        crd = tTR_cC[None, None, None, 0, i]
                        for v in cutlass.range_constexpr(cute.size(vec, mode=[1])):
                            row, col = crd[8 * v]
                            if (row < row_end) & (col + 8 <= n_cols):
                                cute.autovec_copy(rC_v[None, v], g[None, v])
            if cutlass.const_expr(rotate):
                first = (first + subtiles) % self.c_stages
            work.advance()
        if warp == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        return acc, work

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self,
        tiled_mma,
        tma_a,
        mA,
        tma_b,
        mB,
        tma_c,
        mC_tma,
        mC,
        a_layout,
        b_layout,
        c_layout,
        total_m,
        bound,
    ):
        self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout, "C": c_layout})
        sA, sB, sC = self.buffer("A"), self.buffer("B"), self.buffer("C")
        mma_rank = self.cluster.coords(self.cluster.rank())[0]
        thr_mma = tiled_mma.get_slice(mma_rank)
        # Whole-problem B tiles (BN, BK, n_tiles, k_tiles, G); A tiles are built per item.
        gB = cute.local_tile(mB, self.mma_tile[1:], (None, None, None))
        gA0 = cute.local_tile(mA, self.mma_tile, (0, None, None), proj=(1, None, 1))
        tAsA, _ = cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA0), 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tile[:2])
        acc_staged = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        params = GroupedGemmParams(
            tiled_mma=tiled_mma,
            tma_a=tma_a,
            tma_b=tma_b,
            mA=mA,
            sA=sA,
            tAsA=StageView(tAsA, axis=1),
            tBsB=StageView(tBsB, axis=1),
            tBgB=tBgB,
            tCrA=StageView(tiled_mma.make_fragment_A(sA), axis=3),
            tCrB=StageView(tiled_mma.make_fragment_B(sB), axis=3),
            num_k_tiles=cute.size(gB, mode=[3]),
            acc_layout=acc_staged.layout,
            epi_tile=self.epi_tile,
            tma_c=tma_c,
            mC_tma=mC_tma,
            mC=mC,
            cC=cute.make_identity_tensor(mC.shape),
            sC=sC,
            total_m=total_m,
        )
        self.run_roles(params)

    # ------------------------------------------------------------------- host

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        offsets: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        tile, group = self.mma_tile, self.cluster.cta_group
        op = tcgen05.MmaF16BF16Op(
            self.dtype,
            cutlass.Float32,
            (tile[0], tile[1], 16),
            group,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, tile, self.dtype, self.ab.n)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, tile, self.dtype, self.ab.n)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(group)
        vmnk = self.cluster.layout_vmnk().shape
        tma_a, mA = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, a, cute.select(a_layout, mode=[0, 1, 2]), tile, tiled_mma, vmnk
        )
        # B as (N, K, G): the group is the batch coordinate of one descriptor.
        b_nkg = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
        tma_b, mB = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, b_nkg, cute.select(b_layout, mode=[0, 1, 2]), tile, tiled_mma, vmnk
        )
        c_layout = sm100_utils.make_smem_layout_epi(
            self.dtype, self.c_layout, self.epi_tile, self.c_stages
        )
        tma_c, mC_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, cute.select(c_layout, mode=[0, 1]), self.epi_tile
        )
        total_m = Int32(a.shape[0])
        n_tiles = cute.ceil_div(c.shape[1], tile[1])
        bound = self.work.bind((offsets, total_m, n_tiles), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tiled_mma,
            tma_a,
            mA,
            tma_b,
            mB,
            tma_c,
            mC_tma,
            c,
            a_layout,
            b_layout,
            c_layout,
            total_m,
            bound,
        ).launch(**self.launch_config(bound), stream=stream)


@jit_cache(uncached=lambda op: op.trace_protocol)
def compile_grouped_gemm(op: GroupedGemm):
    """Compile once with dynamic total_M, G, N, K (K and N multiples of 8 for 16-byte TMA
    strides). Cached in memory and on disk under ``op.__cute_cache_key__()``."""
    m, g = cute.sym_int(), cute.sym_int()
    n, k = cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)
    bf16 = cutlass.BFloat16
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (m, k)),
        make_fake_compact_tensor(bf16, (g, n, k)),
        make_fake_compact_tensor(bf16, (m, n)),
        make_fake_compact_tensor(cutlass.Int32, (g,)),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


def check_grouped_gemm_args(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Check the operands and return ``out`` (allocated when ``None``). ``offsets`` holds
    the groups' end rows; its values are read only on the device."""
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("grouped_gemm expects bfloat16 operands")
    if offsets.dtype != torch.int32:
        raise TypeError(f"offsets must be int32, got {offsets.dtype}")
    if a.ndim != 2 or b.ndim != 3 or a.shape[1] != b.shape[2]:
        raise ValueError(f"incompatible shapes {tuple(a.shape)}, {tuple(b.shape)}")
    if offsets.ndim != 1 or offsets.shape[0] != b.shape[0]:
        raise ValueError(f"offsets must have shape ({b.shape[0]},), got {tuple(offsets.shape)}")
    if not (a.is_contiguous() and b.is_contiguous() and offsets.is_contiguous()):
        raise ValueError("a, b and offsets must be contiguous")
    if a.data_ptr() % 16 or b.data_ptr() % 16:
        raise ValueError("a and b must be 16-byte aligned")
    if a.shape[1] % 8 or b.shape[1] % 8:
        raise ValueError("K and N must be multiples of 8 (16-byte TMA strides)")
    if b.device != a.device or offsets.device != a.device:
        raise ValueError("a, b and offsets must be on one device")
    shape = (a.shape[0], b.shape[1])
    if out is None:
        return a.new_empty(shape)
    if out.dtype != torch.bfloat16:
        raise TypeError(f"out must be bfloat16, got {out.dtype}")
    if out.shape != shape or out.device != a.device:
        raise ValueError(
            f"out must have shape {shape} on {a.device}, got {tuple(out.shape)} on {out.device}"
        )
    if not out.is_contiguous() or out.data_ptr() % 16:
        raise ValueError("out must be contiguous and 16-byte aligned")
    return out


_OPS: dict = {}


def _frozen_roles(roles):
    return None if roles is None else tuple(sorted(roles.items()))


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor | None = None,
    schedule=None,
    max_ctas: int | None = None,
    cluster=None,
    **config,
) -> torch.Tensor:
    """``out[s_g:e_g] = a[s_g:e_g] @ b[g].T`` for row-major bf16 ``a: (total_M, K)``,
    ``b: (G, N, K)`` and int32 ``offsets: (G,)`` of group end rows (``e_g = offsets[g]``,
    ``s_0 = 0``). Rows past ``offsets[-1]`` are left as they are. ``schedule`` defaults to
    ``AtomicPersistent()``; ``max_ctas`` caps the persistent grid; ``config``:
    ``GroupedGemm``'s other keyword options (``tile_n``, ``ab_stages``, ``acc_stages``,
    ``c_stages``, ``epi_tile_n``, ``roles``)."""
    out = check_grouped_gemm_args(a, b, offsets, out)
    if out.numel() == 0:
        return out
    if a.shape[1] == 0:  # no K tiles: the MMA would publish an unwritten accumulator
        return out.zero_()
    frozen = {k: _frozen_roles(v) if k == "roles" else v for k, v in config.items()}
    key = (schedule, cluster, *sorted(frozen.items()))
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = GroupedGemm(schedule=schedule, cluster=cluster, **config)
    fn = compile_grouped_gemm.by_static_key(key, lambda: compile_grouped_gemm(op))
    workspace, sms = op.work.launch_args(a.device)
    fn(a, b, out, offsets, workspace, max_ctas or sms)
    return out


def grouped_gemm_reference(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor):
    """Per-group fp32 matmuls; rows past the last offset are zero."""
    out = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=torch.float32)
    start = 0
    for g, end in enumerate(offsets.tolist()):
        out[start:end] = a[start:end].float() @ b[g].float().T
        start = end
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    g, n, k = 4, 512, 256
    sizes = [300, 0, 128, 77]
    offsets = torch.tensor(sizes, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    a = torch.randn(sum(sizes), k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(g, n, k, device="cuda", dtype=torch.bfloat16)
    out = grouped_gemm(a, b, offsets)
    err = (out.float() - grouped_gemm_reference(a, b, offsets)).abs().max().item()
    print(f"G={g} sizes={sizes} N={n} K={k}: max abs err {err:.4g}")
    print(GroupedGemm().describe())
