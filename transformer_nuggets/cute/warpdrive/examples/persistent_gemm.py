"""Persistent BF16 ``D = A @ B^T`` (``F.linear`` layout) on SM100 with a ``WorkStream``.

The roles of ``gemm.py`` wrapped in a tile loop: warps 0-3 epilogue, 4 TMA load, 5 MMA, and
(for the broadcast schedules) the layer's scheduler warp in padding warp 6. Two TMEM accumulator
stages (2 x 256 columns) let the MMA warp run tile i+1's mainloop while the epilogue drains
tile i; the epilogue stores through SMEM with TMA, which also clips ragged edges. The schedule,
mapping and cluster are constructor arguments::

    PersistentGemm(schedule=StaticPersistent())  # or OneShot(), AtomicPersistent()
    PersistentGemm(schedule=ClusterLaunchControl())  # or your own IndexSchedule
    PersistentGemm(mapping=CutlassRaster(swizzle=8))  # any WorkMap, or an adapter
    PersistentGemm(cluster=CtaPair())  # 2-CTA: 256x224 MMA tiles, CtaGroup.TWO
    PersistentGemm(cluster=Cluster(2, 2, pair=True), tile_n=224)  # two pairs share A tiles

With ``CtaPair`` the MMA tile is 256 x tile_n x 64: each CTA of the pair loads its 128 rows of A
and its tile_n / 2 rows of B, the leader's MMA warp issues 2-CTA MMAs reading both CTAs' SMEM,
and each CTA's epilogue stores its 128 x tile_n half from its own TMEM. The only source
differences are the ``self.cluster`` declaration, ``cta="leader"`` on the MMA role (a no-op for
one CTA), and the tile shapes derived from the cluster; the layer derives every cross-CTA
barrier.

``tile_n`` defaults to 224 for a CTA pair and 256 otherwise (``default_tile_n``), from the
measurements in docs/examples.md. Which width wins is mostly wave quantization of the tile count
against the resident CTAs, so pass ``tile_n`` for a shape you care about.

A larger ``Cluster`` runs one MMA tile per pair (per CTA without pairs) and multicasts the loads
its CTAs share: pairs side by side along N need the same rows of A, so each CTA loads a part of
its A tile and multicasts it (``Tma(..., multicast="n")``); pairs along M share B the same way.
A work item is then a cluster tile, and each pair offsets it by its position in the cluster.

One compile serves every M, N, K: tile counts, the persistent grid cap, and the atomic counter
are runtime values; schedule, mapping, cluster, and stage counts are compile-time.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from transformer_nuggets.cute.cache import compile_tvm_ffi_and_cache
from transformer_nuggets.cute.utils import make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    Consumer,
    Producer,
    Raster,
    Ring,
    SingleCta,
    SmemPlan,
    StageView,
    StaticPersistent,
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
from transformer_nuggets.cute.warpdrive.examples.gemm import check_gemm_args

CTA_TILE_M, TILE_K = 128, 64  # per-CTA output rows and K step; tile N is a constructor argument
EPI_TILE = (128, 32)  # one TMA store box; tile_n // 32 per CTA output tile
C_STAGES = 2  # SMEM staging buffers for the TMA-store epilogue
EPI_BARRIER_ID = 3  # named barrier for the 4 epilogue warps (layer owns 1 and 2)


def default_tile_n(cluster) -> int:
    """224 for a CTA pair, 256 for one CTA and for larger clusters (measured defaults)."""
    return 224 if cluster.size == cluster.pair_size == 2 else 256


@dataclass(frozen=True)
class PersistentGemmParams:
    """Kernel-prologue values the roles read; built once in ``PersistentGemm.kernel``."""

    tiled_mma: cute.TiledMma
    tma_a: cute.CopyAtom
    tma_b: cute.CopyAtom
    mcast_a: object  # TMA multicast masks of A and B (None: not multicast)
    mcast_b: object
    tile_m: cutlass.Int32  # this CTA's MMA tile within a work item's cluster tile
    tile_n: cutlass.Int32
    tAsA: StageView  # TMA partitions of the A and B stages: (TMA, stage)
    tAgA: cute.Tensor  # whole-problem A tiles: (TMA, m_tiles, k_tiles)
    tBsB: StageView
    tBgB: cute.Tensor  # whole-problem B tiles: (TMA, n_tiles, k_tiles)
    tCrA: StageView  # MMA fragments over the SMEM stages: (MMA, MMA_M, MMA_K, stage)
    tCrB: StageView
    num_k_tiles: cutlass.Int32
    acc_layout: cute.Layout  # all accumulator stages
    epi_tile: cute.Tile
    tma_c: cute.CopyAtom
    mC_tma: cute.Tensor
    sC: cute.Tensor  # epilogue staging buffers


class PersistentGemm(WarpSpecialized):
    def __init__(
        self,
        schedule=None,
        mapping: WorkMap | None = None,
        ab_stages: int | None = None,
        acc_stages: int = 2,
        trace: bool = False,
        cluster=None,
        tile_n: int | None = None,
    ):
        self.cluster = cluster or SingleCta()
        cs = self.cluster.pair_size  # CTAs per MMA
        if tile_n is None:
            tile_n = default_tile_n(self.cluster)
        if tile_n % EPI_TILE[1] or not 0 < tile_n <= 256 or tile_n // cs % 8:
            raise ValueError(
                f"tile_n must be a multiple of {EPI_TILE[1]} in (0, 256], got {tile_n}"
            )
        self.tile_n = tile_n
        cta_tile = (CTA_TILE_M, tile_n, TILE_K)
        # The MMA tile spans the cluster along M; each CTA holds 128 rows of A and N / cs rows
        # of B per stage (2-CTA MMAs read B from both CTAs).
        self.mma_tile = (cta_tile[0] * cs, cta_tile[1], cta_tile[2])
        self.dtype = cutlass.BFloat16
        # Attributes read by the CUTLASS epilogue partition helpers.
        self.c_dtype, self.acc_dtype = self.dtype, cutlass.Float32
        self.cta_tile_shape_mnk = cta_tile
        self.c_layout = LayoutEnum.ROW_MAJOR
        self.acc_stages = acc_stages
        stage_bytes = (cta_tile[0] + cta_tile[1] // cs) * cta_tile[2] * 2  # this CTA's copies
        if ab_stages is None:  # fill ~192 KB of SMEM (48 KB stages for 1 CTA, 32 KB for a pair)
            ab_stages = 192 * 1024 // stage_bytes
        # A and B stages for the mainloop ring, C staging buffers for the TMA-store epilogue.
        self.smem_plan = SmemPlan()
        a = self.smem_plan.buffer("A", self.dtype, align=1024)
        b = self.smem_plan.buffer("B", self.dtype, align=1024)
        self.smem_plan.buffer("C", self.dtype, align=128)
        # Pairs (CTAs) side by side along N share A tiles; along M, B tiles.
        self.pairs_mn = (self.cluster.m // cs, self.cluster.n)
        for count, operand in zip(self.pairs_mn, "BA"):
            # Each CTA loads 1/count of the shared tile; the TMA partition needs that to divide
            # the tile's power-of-two extents (3 compiles, then faults at runtime).
            if count & (count - 1):
                raise ValueError(
                    f"{self.cluster} multicasts {operand} across {count} CTAs; PersistentGemm "
                    "needs a power-of-two count of MMA tiles along each cluster mode"
                )
        multicast = "n" * (self.pairs_mn[1] > 1) + "m" * (self.pairs_mn[0] > 1)
        self.ab = Ring(
            ab_stages, Tma(stage_bytes, multicast=multicast), Umma, buffer={"A": a, "B": b}
        )
        self.acc = Ring(acc_stages, Umma, Threads)
        self.tmem_plan = TmemPlan().region("acc", cta_tile[1], acc_stages)
        self.work = WorkStream(
            mapping=mapping or Raster(swizzle=8), schedule=schedule or StaticPersistent()
        )
        self.trace_protocol = trace

    def get_name(self) -> str:
        return (
            f"warpdrive_pgemm_{self.cluster.get_key()}_n{self.tile_n}_ab{self.ab.n}_"
            f"acc{self.acc_stages}_"
            f"{self.work.get_key()}{self.iket_key()}{self.trace_key()}"
        )

    def get_key(self) -> str:
        return self.get_name()

    # ------------------------------------------------------------------ roles

    @role(warps=4, regs=40)
    @cute.jit
    def load(self, params: PersistentGemmParams, *, ab: Producer, work: WorkCursor):
        ab = ab.views(A=params.tAsA, B=params.tBsB)
        mcast_a = {} if params.mcast_a is None else {"mcast_mask": params.mcast_a}
        mcast_b = {} if params.mcast_b is None else {"mcast_mask": params.mcast_b}
        while work.valid:
            m, n = self.mma_tile_mn(params, work.item)
            tAgA = params.tAgA[None, m, None]
            tBgB = params.tBgB[None, n, None]
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.produce() as s:
                    a, b = s.view("A"), s.view("B")
                    cute.copy(params.tma_a, tAgA[None, k], a, tma_bar_ptr=s.barrier, **mcast_a)
                    cute.copy(params.tma_b, tBgB[None, k], b, tma_bar_ptr=s.barrier, **mcast_b)
            work.advance()
        return ab, work

    @role(warps=5, regs=40, cta="leader")
    @cute.jit
    def mma(
        self,
        params: PersistentGemmParams,
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
    def epilogue(
        self, params: PersistentGemmParams, *, acc: Consumer, work: WorkCursor, tmem: Tmem
    ):
        """TMEM -> registers -> swizzled SMEM -> TMA store, one 128x32 subtile at a time through
        ``C_STAGES`` SMEM buffers. TMA clips the ragged edges, so partial tiles cost the same."""
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # This CTA's rows of the tile.
        thr_mma = params.tiled_mma.get_slice(self.cluster.coords(self.cluster.rank())[0])
        tAcc = transform_partitioned_tensor_layout(
            cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        )
        # (MMA, MMA_M, MMA_N, m_tiles, n_tiles) -> ((M), (N), m_tiles, n_tiles)
        gC = transform_partitioned_tensor_layout(
            thr_mma.partition_C(
                cute.local_tile(
                    params.mC_tma, self.mma_tile, (None, None, None), proj=(1, 1, None)
                )
            )
        )
        gC_l = cute.make_tensor(gC.iterator, cute.append(gC.layout, cute.make_layout(1)))
        t2r, tTR_tAcc, tTR_rAcc = epilogue_tmem_copy_and_partition(
            self, tidx, tAcc, gC_l, params.epi_tile, self.cluster.pair
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.dtype)
        r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(self, t2r, tTR_rC, tidx, params.sC)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            params.tma_c,
            0,
            cute.make_layout(1),
            cute.group_modes(params.sC, 0, 2),
            cute.group_modes(cute.flat_divide(gC, params.epi_tile), 0, 2),
        )
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc) - 1)  # (..., SUB, STAGE)
        subtiles = cute.size(tTR_tAcc, mode=[3])
        # Buffers rotate across tiles too: with a subtile count that is not a multiple of
        # C_STAGES (tile_n=224: 7), a tile's first subtile alternates buffers.
        rotate = subtiles % C_STAGES != 0
        first = Int32(0)  # the buffer of this tile's first subtile
        while work.valid:
            m, n = self.mma_tile_mn(params, work.item)
            dst = cute.group_modes(bSG_gC[None, None, None, m, n], 1, 3)  # (TMA, SUB)
            with acc.consume() as a:
                for i in cutlass.range_constexpr(subtiles):
                    buf = (first + i) % C_STAGES if rotate else i % C_STAGES
                    cute.copy(t2r, tTR_tAcc[None, None, None, i, a.index], tTR_rAcc)
                    tRS_rC.store(r2s.retile(tTR_rAcc).load().to(self.dtype))
                    cute.copy(r2s, tRS_rC, tRS_sC[None, None, None, buf])
                    cute.arch.fence_view_async_shared()  # st.shared -> TMA store
                    cute.arch.barrier(barrier_id=EPI_BARRIER_ID, number_of_threads=128)
                    if warp == 0:
                        cute.copy(params.tma_c, bSG_sC[None, buf], dst[None, i])
                        cute.arch.cp_async_bulk_commit_group()
                        # Buffer `buf` is rewritten C_STAGES subtiles later; keep at most
                        # C_STAGES - 1 stores still reading SMEM.
                        cute.arch.cp_async_bulk_wait_group(C_STAGES - 1, read=True)
                    cute.arch.barrier(barrier_id=EPI_BARRIER_ID, number_of_threads=128)
            if cutlass.const_expr(rotate):
                first = (first + subtiles) % C_STAGES
            work.advance()
        if warp == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        return acc, work

    def mma_tile_mn(self, params: PersistentGemmParams, item):
        """This CTA's MMA tile of a work item (a cluster tile of ``pairs_mn`` MMA tiles)."""
        m, n, _ = item
        pairs_m, pairs_n = self.pairs_mn
        if cutlass.const_expr(pairs_m > 1):
            m = m * pairs_m + params.tile_m
        if cutlass.const_expr(pairs_n > 1):
            n = n * pairs_n + params.tile_n
        return m, n

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self, tiled_mma, tma_a, mA, tma_b, mB, tma_c, mC_tma, a_layout, b_layout, c_layout, bound
    ):
        self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout, "C": c_layout})
        sA, sB, sC = self.buffer("A"), self.buffer("B"), self.buffer("C")
        # This CTA's slice of the (possibly 2-CTA) MMA: its rows of A and of B.
        mma_rank, pair_m, pair_n = self.cluster.coords(self.cluster.rank())
        thr_mma = tiled_mma.get_slice(mma_rank)
        # Whole-problem tile views: (BM, BK, m_tiles, k_tiles) and (BN, BK, n_tiles, k_tiles).
        gA = cute.local_tile(mA, self.mma_tile, (None, None, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self.mma_tile, (None, None, None), proj=(None, 1, 1))
        # Each CTA loads its part of the tiles it shares with the CTAs along N (A) or M (B).
        share_a, share_b = self.cluster.multicast("n"), self.cluster.multicast("m")
        tAsA, tAgA = cpasync.tma_partition(
            tma_a,
            share_a.coord,
            share_a.layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_b,
            share_b.coord,
            share_b.layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tile[:2])
        acc_staged = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        params = PersistentGemmParams(
            tiled_mma=tiled_mma,
            tma_a=tma_a,
            tma_b=tma_b,
            mcast_a=share_a.mask,
            mcast_b=share_b.mask,
            tile_m=pair_m,
            tile_n=pair_n,
            tAsA=StageView(tAsA, axis=1),
            tAgA=tAgA,
            tBsB=StageView(tBsB, axis=1),
            tBgB=tBgB,
            tCrA=StageView(tiled_mma.make_fragment_A(sA), axis=3),
            tCrB=StageView(tiled_mma.make_fragment_B(sB), axis=3),
            num_k_tiles=cute.size(gA, mode=[3]),
            acc_layout=acc_staged.layout,
            epi_tile=EPI_TILE,
            tma_c=tma_c,
            mC_tma=mC_tma,
            sC=sC,
        )
        self.run_roles(params)

    # ------------------------------------------------------------------- host

    def load_op(self, sharing_ctas: int):
        """The TMA load op for an operand shared by ``sharing_ctas`` CTAs of the cluster."""
        if sharing_ctas > 1:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(self.cluster.cta_group)
        return cpasync.CopyBulkTensorTileG2SOp(self.cluster.cta_group)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
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
        # 2-CTA loads (CtaGroup.TWO) signal the leader's full barrier from both CTAs; loads
        # shared along a cluster mode are multicast.
        vmnk = self.cluster.layout_vmnk().shape
        tma_a, mA = cute.nvgpu.make_tiled_tma_atom_A(
            self.load_op(self.pairs_mn[1]),
            a,
            cute.select(a_layout, mode=[0, 1, 2]),
            tile,
            tiled_mma,
            vmnk,
        )
        tma_b, mB = cute.nvgpu.make_tiled_tma_atom_B(
            self.load_op(self.pairs_mn[0]),
            b,
            cute.select(b_layout, mode=[0, 1, 2]),
            tile,
            tiled_mma,
            vmnk,
        )
        c_layout = sm100_utils.make_smem_layout_epi(self.dtype, self.c_layout, EPI_TILE, C_STAGES)
        tma_c, mC_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, cute.select(c_layout, mode=[0, 1]), EPI_TILE
        )
        # One work item per cluster tile; a ragged last cluster tile's extra MMA tiles fall past
        # the edge, where TMA loads zeros and clips the stores.
        cluster_tile = (self.mma_tile[0] * self.pairs_mn[0], self.mma_tile[1] * self.pairs_mn[1])
        m_tiles, n_tiles = cute.ceil_div(c.layout.shape, cluster_tile)
        bound = self.work.bind((m_tiles, n_tiles, 1), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tiled_mma, tma_a, mA, tma_b, mB, tma_c, mC_tma, a_layout, b_layout, c_layout, bound
        ).launch(**self.launch_config(bound), stream=stream)


def compile_persistent_gemm(op: PersistentGemm):
    """Compile once with dynamic M, N, K (K and N multiples of 8 for 16-byte TMA strides)."""
    m = cute.sym_int()
    n, k = cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)
    bf16 = cutlass.BFloat16
    return compile_tvm_ffi_and_cache(
        op,
        op.get_name(),
        make_fake_compact_tensor(bf16, (m, k)),
        make_fake_compact_tensor(bf16, (n, k)),
        make_fake_compact_tensor(bf16, (m, n)),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


_OPS: dict = {}


def persistent_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    schedule=None,
    mapping: WorkMap | None = None,
    max_ctas: int | None = None,
    cluster=None,
    tile_n: int | None = None,
) -> torch.Tensor:
    """``a @ b.T`` for row-major bf16 ``a: (M, K)``, ``b: (N, K)``. ``schedule`` defaults to
    ``StaticPersistent()``; ``max_ctas`` caps the persistent grid in CTAs (default: the CTAs of
    the clusters that fit at once, the SM count for one CTA or a pair); ``cluster=CtaPair()``
    runs 2-CTA 256 x ``tile_n`` tiles, ``Cluster(2, 2, pair=True)`` two such pairs sharing A.
    ``tile_n`` defaults to ``default_tile_n(cluster)``: 224 for a pair, 256 otherwise."""
    out = check_gemm_args(a, b, out)
    if out.numel() == 0:
        return out
    if a.shape[1] == 0:  # no K tiles: the MMA would publish an unwritten accumulator
        return out.zero_()
    if tile_n is None:
        tile_n = default_tile_n(cluster or SingleCta())
    key = (schedule, mapping, cluster, tile_n)
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = PersistentGemm(
            schedule=schedule, mapping=mapping, cluster=cluster, tile_n=tile_n
        )
    fn = compile_persistent_gemm(op)
    workspace, sms = op.work.launch_args(a.device)
    fn(a, b, out, workspace, max_ctas or sms)
    return out


if __name__ == "__main__":
    from transformer_nuggets.cute.warpdrive import (
        AtomicPersistent,
        Cluster,
        ClusterLaunchControl,
        CtaPair,
        OneShot,
    )

    for cluster in (None, CtaPair(), Cluster(2, 2, pair=True)):
        for schedule in (
            OneShot(),
            StaticPersistent(),
            AtomicPersistent(),
            ClusterLaunchControl(),
        ):
            for m, n, k in [(512, 512, 512), (300, 520, 136), (4097, 2056, 4104)]:
                a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                ref = torch.nn.functional.linear(a.float(), b.float())
                out = persistent_gemm(a, b, schedule=schedule, cluster=cluster)
                err = (out.float() - ref).abs().max().item()
                print(f"{cluster} {schedule.describe():<28} M={m} N={n} K={k}: err {err:.4g}")
    print(PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair()).describe())
    print(PersistentGemm(cluster=Cluster(2, 2, pair=True), tile_n=224).describe())
