"""CuTeDSL probes that turn a tcgen05 variant into one repeated-instruction kernel.

Every probe launches ``blocks`` clusters, zero-fills its SMEM and TMEM operands once, crosses a
CTA or cluster barrier, and then issues only the measured instruction in its repeated region.
One timer thread per CTA writes ``(cycles, smid)`` into an ``int64[2 * ctas]`` output, where
``cycles`` is the ``clock64`` delta around the repeated region and completion wait.
"""

from __future__ import annotations

import re

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import cute, pipeline, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.memory import compute_tmem_cols_from_layout

from tcgen05_bench.variants import LDST_REGISTER_FACTOR, MmaVariant, TmemVariant

MMA_WARP = 0
TIMER_WARP = 1
MMA_THREADS = 128
TMEM_COLUMNS = 512
CP_SOURCE_BYTES = 8192
CIRCULAR_SF_COLUMNS = 32
SPARSE_METADATA_COLUMNS = 32
# (K chunk of 16 B, ring stage, next ring stage) for the eight K=96 MMAs that cover 3 x 128 B,
# following CUTLASS's sm103 blockscaled GEMM mainloop.
CIRCULAR_K96_SEQUENCE = (
    (0, 0, 0),
    (3, 0, 0),
    (6, 0, 1),
    (1, 1, 1),
    (4, 1, 1),
    (7, 1, 2),
    (2, 2, 2),
    (5, 2, 2),
)

DTYPES = {
    "bf16": cutlass.BFloat16,
    "f16": cutlass.Float16,
    "tf32": cutlass.TFloat32,
    "f32": cutlass.Float32,
    "e4m3": cutlass.Float8E4M3FN,
    "e5m2": cutlass.Float8E5M2,
    "e2m3": cutlass.Float6E2M3FN,
    "e3m2": cutlass.Float6E3M2FN,
    "e2m1": cutlass.Float4E2M1FN,
    "s8": cutlass.Int8,
    "u8": cutlass.Uint8,
    "s32": cutlass.Int32,
    "ue8m0": cutlass.Float8E8M0FNU,
    "ue4m3": cutlass.Float8E4M3FN,
}


def kernel_name(variant: MmaVariant | TmemVariant) -> str:
    """Return a CuTeDSL-legal artifact name for a variant."""
    return "tc_" + re.sub(r"[^a-z0-9]+", "_", variant.name.lower()).strip("_")


# --------------------------------------------------------------------------------------------
# Inline PTX helpers for instructions CuTeDSL does not expose directly.
# --------------------------------------------------------------------------------------------


def _asm(asm: str, operands: list, constraints: str, *, loc=None, ip=None, results=None):
    return llvm.inline_asm(
        results,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


def _ir(value, *, loc=None, ip=None):
    return value.ir_value(loc=loc, ip=ip)


@dsl_user_op
def tcgen05_fence_before(*, loc=None, ip=None) -> None:
    """Emit ``tcgen05.fence::before_thread_sync``."""
    _asm("tcgen05.fence::before_thread_sync;", [], "", loc=loc, ip=ip)


@dsl_user_op
def tcgen05_fence_after(*, loc=None, ip=None) -> None:
    """Emit ``tcgen05.fence::after_thread_sync``."""
    _asm("tcgen05.fence::after_thread_sync;", [], "", loc=loc, ip=ip)


@dsl_user_op
def tmem_store_zero(taddr: Int32, zero: Int32, *, loc=None, ip=None) -> None:
    """Store one 32-bit value into eight columns of this warp's 32 TMEM lanes."""
    regs = ", ".join(["$1"] * 8)
    _asm(
        f"tcgen05.st.sync.aligned.32x32b.x8.b32 [$0], {{{regs}}};",
        [_ir(Int32(taddr), loc=loc, ip=ip), _ir(Int32(zero), loc=loc, ip=ip)],
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_sparse(
    d_taddr: Int32,
    a_desc: Int64,
    b_desc: Int64,
    e_taddr: Int32,
    idesc: int,
    kind: str,
    cta_group: int,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one accumulating ``tcgen05.mma.sp`` from an elected thread."""
    with cute.arch.elect_one(loc=loc, ip=ip):
        _asm(
            "{\n\t.reg .pred p;\n\tsetp.eq.u32 p, 1, 1;\n\t"
            f"tcgen05.mma.sp.cta_group::{cta_group}.kind::{kind} [$0], $1, $2, [$3], $4, p;\n\t}}",
            [
                _ir(Int32(d_taddr), loc=loc, ip=ip),
                _ir(Int64(a_desc), loc=loc, ip=ip),
                _ir(Int64(b_desc), loc=loc, ip=ip),
                _ir(Int32(e_taddr), loc=loc, ip=ip),
                _ir(Int32(idesc), loc=loc, ip=ip),
            ],
            "r,l,l,r,r",
            loc=loc,
            ip=ip,
        )


def sparse_instruction_descriptor(variant: MmaVariant) -> int:
    """Return the SM100 ``kind::f16``/``kind::f8f6f4`` instruction descriptor with sparse set."""
    formats = {"bf16": 1, "f16": 0, "e4m3": 0, "e5m2": 1}
    c_format = {"f16": 0, "f32": 1}[variant.kind.acc]
    return (
        (1 << 2)
        | (c_format << 4)
        | (formats[variant.kind.a] << 7)
        | (formats[variant.kind.b] << 10)
        | ((variant.n >> 3) << 17)
        | ((variant.m >> 4) << 24)
    )


@dsl_user_op
def tmem_wait_st(*, loc=None, ip=None) -> None:
    """Wait for this thread's outstanding ``tcgen05.st``."""
    _asm("tcgen05.wait::st.sync.aligned;", [], "", loc=loc, ip=ip)


@dsl_user_op
def tmem_ld_batch(addresses: list, shape: str, num: int, *, loc=None, ip=None) -> Int32:
    """Issue one ``tcgen05.ld`` per address, then one ``tcgen05.wait::ld``.

    Every load gets its own destination registers inside one asm block, so the batch is truly in
    flight together. Returns the XOR of each load's first register; callers fold it into a live
    value so ptxas keeps every load.
    """
    regs = num * LDST_REGISTER_FACTOR[shape]
    count = len(addresses)
    lines = []
    for index in range(count):
        outputs = ", ".join(f"${index * regs + i}" for i in range(regs))
        lines.append(
            f"tcgen05.ld.sync.aligned.{shape}.x{num}.b32 {{{outputs}}}, [${count * regs + index}];"
        )
    lines.append("tcgen05.wait::ld.sync.aligned;")
    result = _asm(
        "\n\t".join(lines),
        [_ir(Int32(address), loc=loc, ip=ip) for address in addresses],
        ",".join(["=r"] * (count * regs) + ["r"] * count),
        results=llvm.StructType.get_literal([T.i32()] * (count * regs)),
        loc=loc,
        ip=ip,
    )
    folded = Int32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip))
    for index in range(1, count):
        folded = folded ^ Int32(llvm.extractvalue(T.i32(), result, [index * regs], loc=loc, ip=ip))
    return folded


@dsl_user_op
def tmem_st_batch(addresses: list, value: Int32, shape: str, num: int, *, loc=None, ip=None):
    """Issue one ``tcgen05.st`` of a repeated register per address, then ``tcgen05.wait::st``."""
    regs = num * LDST_REGISTER_FACTOR[shape]
    inputs = ", ".join(["$0"] * regs)
    lines = [
        f"tcgen05.st.sync.aligned.{shape}.x{num}.b32 [${index + 1}], {{{inputs}}};"
        for index in range(len(addresses))
    ]
    lines.append("tcgen05.wait::st.sync.aligned;")
    _asm(
        "\n\t".join(lines),
        [_ir(Int32(value), loc=loc, ip=ip)] + [_ir(Int32(a), loc=loc, ip=ip) for a in addresses],
        ",".join(["r"] * (len(addresses) + 1)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_cp(taddr: Int32, desc: Int64, shape: str, *, loc=None, ip=None) -> None:
    """Issue one single-thread ``tcgen05.cp.cta_group::1`` SMEM->TMEM copy."""
    _asm(
        f"tcgen05.cp.cta_group::1.{shape} [$0], $1;",
        [_ir(Int32(taddr), loc=loc, ip=ip), _ir(Int64(desc), loc=loc, ip=ip)],
        "r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_shift_down(taddr: Int32, *, loc=None, ip=None) -> None:
    """Issue one single-thread ``tcgen05.shift.cta_group::1.down``."""
    _asm(
        "tcgen05.shift.cta_group::1.down [$0];",
        [_ir(Int32(taddr), loc=loc, ip=ip)],
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_alloc_dealloc(smem_addr: Int32, columns: int, *, loc=None, ip=None) -> None:
    """Allocate ``columns`` TMEM columns, read the address back, and free them (warp-wide)."""
    _asm(
        "{\n\t.reg .b32 taddr;\n\t"
        f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {columns};\n\t"
        "tcgen05.fence::before_thread_sync;\n\t"
        "bar.warp.sync -1;\n\t"
        "tcgen05.fence::after_thread_sync;\n\t"
        "ld.shared.b32 taddr, [$0];\n\t"
        f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, {columns};\n\t"
        "}",
        [_ir(Int32(smem_addr), loc=loc, ip=ip)],
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_relinquish(*, loc=None, ip=None) -> None:
    """Emit ``tcgen05.relinquish_alloc_permit.cta_group::1``."""
    _asm("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", [], "", loc=loc, ip=ip)


def _pointer_int(pointer, *, loc=None, ip=None) -> Int32:
    return Int32(pointer.toint(loc=loc, ip=ip))


@cute.jit
def zero_tmem(tmem_pointer, warp, columns: int) -> None:
    """Zero every allocated column of the calling warp's 32 lanes."""
    base = _pointer_int(tmem_pointer) + ((warp % 4) * 32 << 16)
    for column in cutlass.range_constexpr(0, columns, 8):
        tmem_store_zero(base + column, Int32(0))
    tmem_wait_st()


@cute.jit
def zero_smem(pointer, num_bytes: int, thread, threads: int) -> None:
    """Zero ``num_bytes`` of SMEM with 32-bit stores from every thread."""
    words = cute.recast_ptr(pointer, dtype=cutlass.Uint32)
    word = thread
    while word < num_bytes // 4:
        words[word] = cutlass.Uint32(0)
        word += threads


@cute.jit
def write_sample(output: cute.Tensor, cta: Int32, cycles: Int64) -> None:
    """Store the CTA's timed cycles and SM id."""
    output[2 * cta] = cycles
    output[2 * cta + 1] = Int64(cute.arch.smid())


# --------------------------------------------------------------------------------------------
# tcgen05.mma probe
# --------------------------------------------------------------------------------------------


def make_mma_op(variant: MmaVariant, collector_op: str = "DISCARD"):
    """Construct the CuTeDSL op for an MMA variant (raises when the arch rejects it).

    ``collector_op`` applies to the operand named by ``variant.collector`` on sm_107 ops.
    """
    kind = variant.kind
    collectors = {}
    if variant.collector:
        collectors[f"{variant.collector}_collector_op"] = tcgen05.CollectorOp[collector_op]
    shape = (variant.m, variant.n, kind.k)
    cta_group = tcgen05.CtaGroup.ONE if variant.cta_group == 1 else tcgen05.CtaGroup.TWO
    source = tcgen05.OperandSource.SMEM if variant.a_src == "smem" else tcgen05.OperandSource.TMEM
    major = tcgen05.OperandMajorMode.K
    a, b, acc = DTYPES[kind.a], DTYPES[kind.b], DTYPES[kind.acc]
    sf = DTYPES[kind.sf] if kind.sf else None
    op = getattr(tcgen05, kind.op)
    if kind.op == "MmaF16BF16Op":
        return op(a, acc, shape, cta_group, source, major, major)
    if kind.op == "MmaTF32Op":
        return op(shape, cta_group, source, major, major)
    if kind.op == "SM107MmaFP8Op":
        return op(a, b, acc, shape, cta_group, source, major, major, **collectors)
    if kind.op == "MmaF8F6F4Op":
        return op(a, b, acc, shape, cta_group, source, major, major)
    if kind.op == "MmaI8Op":
        return op(a, shape, cta_group, source, major, major)
    if kind.op == "MmaMXF8F6F4Op":
        return op(a, b, shape, cta_group, source, major, major)
    if kind.op == "SM107BlockScaledMmaMXF8F6F4Op":
        return op(a, b, sf, kind.sf_vec, shape, cta_group, source, major, major, **collectors)
    if kind.op in ("MmaMXF4Op", "SM103MmaMXF4Op"):
        return op(shape, cta_group, source)
    if kind.op in ("MmaMXF4NVF4Op", "SM103MmaMXF4NVF4Op"):
        return op(sf, shape, cta_group, source)
    if kind.op == "SM107MmaMXF4NVF4Op":
        return op(sf, kind.sf_vec, shape, cta_group, source, **collectors)
    raise ValueError(f"unknown tcgen05 op {kind.op}")


class MmaProbe:
    """Repeat one tcgen05.mma instruction form over a fixed per-CTA envelope."""

    def __init__(self, variant: MmaVariant) -> None:
        self.variant = variant
        self.kind = variant.kind
        self.two_cta = variant.cta_group == 2
        self.ts = variant.a_src == "tmem"
        self.latency = variant.mode == "latency"
        self.tile = (variant.m, variant.n, self.kind.k_env)
        self.a_type = DTYPES[self.kind.a]
        self.b_type = DTYPES[self.kind.b]
        self.acc_type = DTYPES[self.kind.acc]
        self.sf_type = DTYPES[self.kind.sf] if self.kind.sf else None
        # sm_103 K=96 FP4 reads 48 B of K per instruction across a 3x128 B circular SMEM ring.
        self.circular = self.kind.k == 96
        self.sparse = variant.sparse

    def get_name(self) -> str:
        """Return the stable artifact and kernel prefix."""
        return kernel_name(self.variant)

    def circular_layout(self, mma, extent: int):
        """Return the sm_103 K=96 operand layout: (rows, 16 B) x 8 chunks x 3 ring stages."""
        rows = extent // cute.size(mma.thr_id.shape)
        return tcgen05.tile_to_mma_shape(
            tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Uint8),
            ((rows, 16), 1, 8, 3),
            order=(0, 1, 2),
        )

    def operand_bytes(self, dtype, layout) -> int:
        """Return SMEM bytes for one operand, padding sub-byte types to 8-bit containers."""
        num_bytes = int(cute.size_in_bytes(dtype, layout.outer))
        if self.kind.container_bits == 8 and dtype.width < 8:
            num_bytes = num_bytes * 8 // dtype.width
        return (num_bytes + 1023) // 1024 * 1024

    @cute.jit
    def __call__(self, output: cute.Tensor, repeats: Int32, blocks: Int32, stream):
        mma = cute.make_tiled_mma(cute.make_mma_atom(make_mma_op(self.variant)))
        mma_fill = mma
        mma_last = mma
        if cutlass.const_expr(self.variant.collector):
            mma_fill = cute.make_tiled_mma(cute.make_mma_atom(make_mma_op(self.variant, "FILL")))
            mma_last = cute.make_tiled_mma(
                cute.make_mma_atom(make_mma_op(self.variant, "LASTUSE"))
            )
        if cutlass.const_expr(self.circular):
            a_layout = self.circular_layout(mma, self.variant.m)
            b_layout = self.circular_layout(mma, self.variant.n)
            self.a_type = self.b_type = cutlass.Uint8
        else:
            a_layout = sm100_utils.make_smem_layout_a(mma, self.tile, self.a_type, 1)
            b_layout = sm100_utils.make_smem_layout_b(mma, self.tile, self.b_type, 1)
        self.a_bytes = self.operand_bytes(self.a_type, a_layout)
        self.b_bytes = self.operand_bytes(self.b_type, b_layout)
        acc_layout = mma.make_fragment_C(mma.partition_shape_C(self.tile[:2])).layout
        self.acc_columns = compute_tmem_cols_from_layout(acc_layout, self.acc_type)
        columns = self.variant.m_panels * self.variant.n_panels * self.acc_columns

        sfa_layout = None
        sfb_layout = None
        self.sfa_offset = columns
        self.sfb_offset = columns
        if cutlass.const_expr(self.circular):
            # Scale factors are zero-filled TMEM; every K=96 MMA reads one fixed SF window.
            columns += 2 * CIRCULAR_SF_COLUMNS
            self.sfb_offset = self.sfa_offset + CIRCULAR_SF_COLUMNS
        elif cutlass.const_expr(self.kind.block_scaled):
            vec = self.kind.sf_vec
            sfa_smem = blockscaled_utils.make_smem_layout_sfa(mma, self.tile, vec, 1)
            sfb_smem = blockscaled_utils.make_smem_layout_sfb(mma, self.tile, vec, 1)
            sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
                mma, self.tile, vec, cute.slice_(sfa_smem, (None, None, None, 0))
            )
            sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
                mma, self.tile, vec, cute.slice_(sfb_smem, (None, None, None, 0))
            )
            columns += compute_tmem_cols_from_layout(sfa_layout, self.sf_type)
            self.sfb_offset = columns
            columns += compute_tmem_cols_from_layout(sfb_layout, self.sf_type)
        self.e_offset = columns
        if cutlass.const_expr(self.sparse):
            columns += SPARSE_METADATA_COLUMNS
        self.a_offset = columns
        if cutlass.const_expr(self.ts):
            columns += self.kind.k_env * self.kind.container_bits // 32
        if cutlass.const_expr(columns > TMEM_COLUMNS):
            raise ValueError(f"{self.variant.name} needs {columns} TMEM columns")
        self.tmem_columns = max(32, 1 << (columns - 1).bit_length())

        @cute.struct
        class SharedStorage:
            completion: cute.struct.MemRange[Int64, 1]
            tmem_dealloc: Int64
            tmem_holder: Int32
            a: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_bytes], 1024]
            b: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.b_bytes], 1024]

        self.shared_type = SharedStorage
        self.smem_bytes = SharedStorage.size_in_bytes()
        cluster = 2 if self.two_cta else 1
        self.kernel(
            mma,
            mma_fill,
            mma_last,
            a_layout,
            b_layout,
            acc_layout,
            sfa_layout,
            sfb_layout,
            output,
            repeats,
        ).launch(
            grid=(blocks * cluster, 1, 1),
            block=(MMA_THREADS, 1, 1),
            cluster=(cluster, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.jit
    def issue_collector_pairs(
        self, mma_fill, mma_last, tmem_pointer, a_fragment, b_fragment, acc_layout, sfa, sfb
    ):
        """Issue each K step as a collector fill/lastuse pair sharing the kept operand."""
        for k_block in cutlass.range_constexpr(self.variant.k_instructions):
            for pair in cutlass.range_constexpr(2):
                atom = mma_fill if pair == 0 else mma_last
                accumulator = cute.make_tensor(
                    cute.recast_ptr(tmem_pointer + pair * self.acc_columns, dtype=self.acc_type),
                    acc_layout,
                )
                a = a_fragment[None, None, k_block]
                b = b_fragment[None, None, k_block]
                if cutlass.const_expr(self.kind.block_scaled):
                    cute.gemm(
                        atom,
                        accumulator,
                        [a, sfa[None, None, k_block]],
                        [b, sfb[None, None, k_block]],
                        accumulator,
                    )
                else:
                    cute.gemm(atom, accumulator, a, b, accumulator)

    @cute.jit
    def issue(self, mma, tmem_pointer, a_fragment, b_fragment, acc_layout, sfa, sfb, panels: int):
        """Issue ``panels`` accumulator panels of the variant's K extent."""
        for panel in cutlass.range_constexpr(panels):
            accumulator = cute.make_tensor(
                cute.recast_ptr(tmem_pointer + panel * self.acc_columns, dtype=self.acc_type),
                acc_layout,
            )
            if cutlass.const_expr(self.sparse):
                # Sparse A holds K/2 elements, so sparse step j reads dense A k-block j and the
                # two dense B k-blocks 2j, 2j+1 (contiguous within the swizzle atom).
                for step in cutlass.range_constexpr(self.variant.k_instructions):
                    mma_sparse(
                        _pointer_int(accumulator.iterator),
                        tcgen05.smem_descriptor_to_int(a_fragment[None, None, step].iterator),
                        tcgen05.smem_descriptor_to_int(b_fragment[None, None, 2 * step].iterator),
                        _pointer_int(tmem_pointer + self.e_offset),
                        sparse_instruction_descriptor(self.variant),
                        self.kind.kind,
                        self.variant.cta_group,
                    )
            elif cutlass.const_expr(self.circular):
                count = 1 if self.latency else len(CIRCULAR_K96_SEQUENCE)
                for k_chunk, stage, next_stage in CIRCULAR_K96_SEQUENCE[:count]:
                    a_desc = tcgen05.make_umma_smem_desc(
                        a_fragment[None, 0, k_chunk, stage].iterator,
                        a_fragment[None, 0, k_chunk, stage].layout,
                        "k",
                        next_src=a_fragment[None, 0, 0, next_stage].iterator,
                    )
                    b_desc = tcgen05.make_umma_smem_desc(
                        b_fragment[None, 0, k_chunk, stage].iterator,
                        b_fragment[None, 0, k_chunk, stage].layout,
                        "k",
                        next_src=b_fragment[None, 0, 0, next_stage].iterator,
                    )
                    view = cute.make_layout(1, stride=0)
                    cute.mma_atom_call(
                        mma,
                        accumulator[None, 0, 0],
                        cute.make_tensor(a_desc, view),
                        cute.make_tensor(b_desc, view),
                        accumulator[None, 0, 0],
                    )
            elif cutlass.const_expr(self.kind.block_scaled):
                cute.gemm(mma, accumulator, [a_fragment, sfa], [b_fragment, sfb], accumulator)
            else:
                cute.gemm(mma, accumulator, a_fragment, b_fragment, accumulator)

    @cute.kernel
    def kernel(
        self,
        mma: cute.TiledMma,
        mma_fill: cute.TiledMma,
        mma_last: cute.TiledMma,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        acc_layout: cute.Layout,
        sfa_layout,
        sfb_layout,
        output: cute.Tensor,
        repeats: Int32,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        thread, _, _ = cute.arch.thread_idx()
        cta = cute.arch.block_idx()[0]
        rank = Int32(0)
        if cutlass.const_expr(self.two_cta):
            rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        shared = utils.SmemAllocator().allocate(self.shared_type)
        barrier = shared.completion.data_ptr()

        zero_smem(shared.a.data_ptr(), self.a_bytes, thread, MMA_THREADS)
        zero_smem(shared.b.data_ptr(), self.b_bytes, thread, MMA_THREADS)
        if warp == MMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier, 1)
            cute.arch.mbarrier_init_fence()
        tmem = utils.TmemAllocator(
            shared.tmem_holder.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=MMA_THREADS),
            allocator_warp_id=MMA_WARP,
            is_two_cta=self.two_cta,
            two_cta_tmem_dealloc_mbar_ptr=shared.tmem_dealloc.ptr,
        )
        tmem.allocate(self.tmem_columns)
        tmem.wait_for_alloc()
        tmem_pointer = tmem.retrieve_ptr(cutlass.Float32)
        zero_tmem(tmem_pointer, warp, self.tmem_columns)
        if cutlass.const_expr(self.sparse):
            # 2:4 metadata nibble 0b0100 selects elements (0, 1) of every group of four.
            base = _pointer_int(tmem_pointer) + ((warp % 4) * 32 << 16) + self.e_offset
            for column in cutlass.range_constexpr(0, SPARSE_METADATA_COLUMNS, 8):
                tmem_store_zero(base + column, Int32(0x44444444))
            tmem_wait_st()

        thr_mma = mma.get_slice(rank)
        sa = cute.make_tensor(
            cute.recast_ptr(shared.a.data_ptr(), a_layout.inner, dtype=self.a_type), a_layout.outer
        )
        sb = cute.make_tensor(
            cute.recast_ptr(shared.b.data_ptr(), b_layout.inner, dtype=self.b_type), b_layout.outer
        )
        if cutlass.const_expr(self.circular):
            a_fragment = sa
            b_fragment = sb
            mma.set(
                tcgen05.Field.SFA,
                cute.recast_ptr(tmem_pointer + self.sfa_offset, dtype=self.sf_type),
            )
            mma.set(
                tcgen05.Field.SFB,
                cute.recast_ptr(tmem_pointer + self.sfb_offset, dtype=self.sf_type),
            )
        elif cutlass.const_expr(self.ts):
            a_tmem = thr_mma.make_fragment_A(cute.make_tensor(tmem_pointer, a_layout.outer))
            a_fragment = cute.make_tensor(
                cute.recast_ptr(tmem_pointer + self.a_offset, dtype=self.a_type),
                a_tmem.layout,
            )[None, None, None, 0]
        else:
            a_fragment = thr_mma.make_fragment_A(sa)[None, None, None, 0]
        if cutlass.const_expr(not self.circular):
            b_fragment = thr_mma.make_fragment_B(sb)[None, None, None, 0]
        sfa = None
        sfb = None
        if cutlass.const_expr(self.kind.block_scaled and not self.circular):
            sfa = cute.make_tensor(
                cute.recast_ptr(tmem_pointer + self.sfa_offset, dtype=self.sf_type), sfa_layout
            )
            sfb = cute.make_tensor(
                cute.recast_ptr(tmem_pointer + self.sfb_offset, dtype=self.sf_type), sfb_layout
            )
        if cutlass.const_expr(self.latency and not self.circular):
            a_fragment = a_fragment[None, None, 0]
            b_fragment = b_fragment[None, None, 0]
            if cutlass.const_expr(self.kind.block_scaled):
                sfa = sfa[None, None, 0]
                sfb = sfb[None, None, 0]

        cute.arch.fence_view_async_shared()
        tcgen05_fence_before()
        if cutlass.const_expr(self.two_cta):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            pipeline.NamedBarrier(barrier_id=2, num_threads=MMA_THREADS).arrive_and_wait()
        tcgen05_fence_after()

        cta_group = tcgen05.CtaGroup.TWO if self.two_cta else tcgen05.CtaGroup.ONE
        mask = 0b11 if self.two_cta else None
        mma.set(tcgen05.Field.ACCUMULATE, True)
        mma_fill.set(tcgen05.Field.ACCUMULATE, True)
        mma_last.set(tcgen05.Field.ACCUMULATE, True)
        if cutlass.const_expr(self.latency):
            if warp == MMA_WARP and rank == 0:
                start = cute.arch.clock64()
                for repeat in cutlass.range(repeats, unroll=1):
                    self.issue(mma, tmem_pointer, a_fragment, b_fragment, acc_layout, sfa, sfb, 1)
                    with cute.arch.elect_one():
                        tcgen05.commit(barrier, mask, cta_group)
                    cute.arch.mbarrier_wait(barrier, repeat % 2)
                cycles = cute.arch.clock64() - start
                if thread == 0:
                    write_sample(output, cta, cycles)
            elif warp == TIMER_WARP and rank != 0:
                if thread == TIMER_WARP * 32:
                    write_sample(output, cta, Int64(0))
        else:
            if warp == MMA_WARP and rank == 0:
                panels = self.variant.m_panels * self.variant.n_panels
                for _ in cutlass.range(repeats, unroll=1):
                    if cutlass.const_expr(self.variant.collector):
                        self.issue_collector_pairs(
                            mma_fill,
                            mma_last,
                            tmem_pointer,
                            a_fragment,
                            b_fragment,
                            acc_layout,
                            sfa,
                            sfb,
                        )
                    else:
                        self.issue(
                            mma, tmem_pointer, a_fragment, b_fragment, acc_layout, sfa, sfb, panels
                        )
                with cute.arch.elect_one():
                    tcgen05.commit(barrier, mask, cta_group)
            elif warp == TIMER_WARP:
                start = cute.arch.clock64()
                cute.arch.mbarrier_wait(barrier, 0)
                cycles = cute.arch.clock64() - start
                if thread == TIMER_WARP * 32:
                    write_sample(output, cta, cycles)

        tcgen05_fence_before()
        pipeline.NamedBarrier(barrier_id=3, num_threads=MMA_THREADS).arrive_and_wait()
        if cutlass.const_expr(self.two_cta):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        tcgen05_fence_after()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_pointer)


# --------------------------------------------------------------------------------------------
# Non-MMA tcgen05 probe
# --------------------------------------------------------------------------------------------


class TmemProbe:
    """Repeat one tcgen05.ld/st/cp/shift/commit/fence/alloc form."""

    def __init__(self, variant: TmemVariant) -> None:
        self.variant = variant
        self.op = variant.op
        self.threads = 32 * variant.warps
        self.latency = variant.mode == "latency"
        self.owns_tmem = variant.op != "alloc"

    def get_name(self) -> str:
        """Return the stable artifact and kernel prefix."""
        return kernel_name(self.variant)

    def columns_per_instruction(self) -> int:
        """Return TMEM columns touched by one ld/st/cp/shift (used to rotate addresses)."""
        if self.op in ("ld", "st"):
            factor = 1 if self.variant.shape == "32x32b" else 2
            return factor * self.variant.registers
        return 8

    @cute.jit
    def __call__(self, output: cute.Tensor, repeats: Int32, blocks: Int32, stream):
        @cute.struct
        class SharedStorage:
            completion: cute.struct.MemRange[Int64, 1]
            tmem_holder: Int32
            scratch: Int32
            source: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, CP_SOURCE_BYTES], 1024]

        self.shared_type = SharedStorage
        self.kernel(output, repeats).launch(
            grid=(blocks, 1, 1),
            block=(self.threads, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    def batch_addresses(self, base, batch: int) -> list:
        """Return the TMEM addresses of one in-flight ld/st batch."""
        step = self.columns_per_instruction()
        k = self.variant.inflight
        slots = TMEM_COLUMNS // step
        return [base + ((batch * k + i) % slots) * step for i in range(k)]

    @cute.jit
    def load_repeat(self, tmem_base, lane_base, live: Int32) -> Int32:
        """Issue one repeat of tcgen05.ld batches and fold one register of each into ``live``."""
        for batch in cutlass.range_constexpr(self.variant.unroll // self.variant.inflight):
            live = live ^ tmem_ld_batch(
                self.batch_addresses(tmem_base + lane_base, batch),
                self.variant.shape,
                self.variant.num,
            )
        return live

    @cute.jit
    def body(self, tmem_base, lane_base, desc, barrier, phase, scratch_addr, warp):
        """Issue one repeat of the measured instruction (all ops except tcgen05.ld)."""
        variant = self.variant
        step = self.columns_per_instruction()
        if cutlass.const_expr(self.op == "st"):
            for batch in cutlass.range_constexpr(variant.unroll // variant.inflight):
                tmem_st_batch(
                    self.batch_addresses(tmem_base + lane_base, batch),
                    Int32(0),
                    variant.shape,
                    variant.num,
                )
        elif cutlass.const_expr(self.op in ("cp", "shift", "commit")):
            if warp == 0:
                with cute.arch.elect_one():
                    for u in cutlass.range_constexpr(variant.unroll):
                        if cutlass.const_expr(self.op == "cp"):
                            tmem_cp(tmem_base + u * step, desc, variant.shape)
                        elif cutlass.const_expr(self.op == "shift"):
                            tmem_shift_down(tmem_base + u * step)
                    if cutlass.const_expr(self.latency):
                        tcgen05.commit(barrier)
                if cutlass.const_expr(self.latency):
                    cute.arch.mbarrier_wait(barrier, phase)
        elif cutlass.const_expr(self.op == "fence"):
            if cutlass.const_expr(variant.shape == "before_thread_sync"):
                tcgen05_fence_before()
            else:
                tcgen05_fence_after()
        elif cutlass.const_expr(self.op == "alloc"):  # noqa: SIM102 - const_expr stays separate
            if warp == 0:
                tmem_alloc_dealloc(scratch_addr, variant.num)

    @cute.kernel
    def kernel(self, output: cute.Tensor, repeats: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        thread, _, _ = cute.arch.thread_idx()
        cta = cute.arch.block_idx()[0]
        shared = utils.SmemAllocator().allocate(self.shared_type)
        barrier = shared.completion.data_ptr()
        zero_smem(shared.source.data_ptr(), CP_SOURCE_BYTES, thread, self.threads)
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier, 1)
            cute.arch.mbarrier_init_fence()

        tmem_base = Int32(0)
        tmem_pointer = None
        tmem = None
        if cutlass.const_expr(self.owns_tmem):
            tmem = utils.TmemAllocator(
                shared.tmem_holder.ptr,
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads),
                allocator_warp_id=0,
            )
            tmem.allocate(TMEM_COLUMNS)
            tmem.wait_for_alloc()
            tmem_pointer = tmem.retrieve_ptr(cutlass.Float32)
            if warp < 4:
                zero_tmem(tmem_pointer, warp, TMEM_COLUMNS)
            tmem_base = _pointer_int(tmem_pointer)

        # SM100 SMEM matrix descriptor: start>>4, LBO=128 B, SBO=256 B, version 1, no swizzle.
        source = _pointer_int(shared.source.data_ptr())
        desc = (
            Int64((source >> 4) & 0x3FFF)
            | Int64((128 >> 4) << 16)
            | (Int64(256 >> 4) << 32)
            | (Int64(1) << 46)
        )
        scratch_addr = _pointer_int(shared.scratch.ptr)
        lane_base = ((warp % 4) * 32) << 16

        cute.arch.fence_view_async_shared()
        tcgen05_fence_before()
        pipeline.NamedBarrier(barrier_id=2, num_threads=self.threads).arrive_and_wait()
        tcgen05_fence_after()

        start = cute.arch.clock64()
        if cutlass.const_expr(self.op == "ld"):
            live = Int32(0)
            for _ in cutlass.range(repeats, unroll=1):
                live = self.load_repeat(tmem_base, lane_base, live)
            shared.scratch = live
        else:
            for repeat in cutlass.range(repeats, unroll=1):
                self.body(tmem_base, lane_base, desc, barrier, repeat % 2, scratch_addr, warp)
        if cutlass.const_expr(self.op in ("cp", "shift") and not self.latency):  # noqa: SIM102
            if warp == 0:
                with cute.arch.elect_one():
                    tcgen05.commit(barrier)
                cute.arch.mbarrier_wait(barrier, 0)
        pipeline.NamedBarrier(barrier_id=3, num_threads=self.threads).arrive_and_wait()
        cycles = cute.arch.clock64() - start
        if thread == 0:
            write_sample(output, cta, cycles)

        if cutlass.const_expr(self.owns_tmem):
            tcgen05_fence_after()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_pointer)
        else:
            if warp == 0:
                tmem_relinquish()


def make_probe(variant: MmaVariant | TmemVariant):
    """Return the probe object for a variant."""
    if isinstance(variant, MmaVariant):
        return MmaProbe(variant)
    return TmemProbe(variant)
