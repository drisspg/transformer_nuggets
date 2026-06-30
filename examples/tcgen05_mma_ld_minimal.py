"""Minimal Blackwell tcgen05 MMA -> completion barrier -> fence -> tcgen05.ld flow.

Run on SM100 with PTX dumping enabled:
    CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=agent_space/tcgen05_ptx python examples/tcgen05_mma_ld_minimal.py

The generated PTX should show the non-pipelined ordering sequence around the
accumulator handoff:
    tcgen05.mma
    tcgen05.commit.mbarrier::arrive
    mbarrier.try_wait / wait
    tcgen05.fence::after_thread_sync
    tcgen05.ld
"""

import argparse
from pathlib import Path

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def tcgen05_mma_f16_128x128x16(
    tmem_ptr: cute.Pointer,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    accumulate: cutlass.Int32,
    instruction_descriptor: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    tmem_addr = tmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    a_desc = tcgen05.smem_descriptor_to_int(a_frag.iterator, loc=loc, ip=ip).ir_value(
        loc=loc, ip=ip
    )
    b_desc = tcgen05.smem_descriptor_to_int(b_frag.iterator, loc=loc, ip=ip).ir_value(
        loc=loc, ip=ip
    )
    with cute.arch.elect_one(loc=loc, ip=ip):
        llvm.inline_asm(
            None,
            [
                tmem_addr,
                a_desc,
                b_desc,
                cutlass.Int32(accumulate).ir_value(loc=loc, ip=ip),
                cutlass.Int32(instruction_descriptor).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .u32 idesc, zero;\n\t"
            ".reg .pred pred_accumulate;\n\t"
            "mov.u32 idesc, $4;\n\t"
            "mov.u32 zero, 0;\n\t"
            "setp.ne.u32 pred_accumulate, $3, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, idesc, "
            "{zero, zero, zero, zero}, pred_accumulate;\n"
            "}",
            "r,l,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def tcgen05_commit_arrive(mbar_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    with cute.arch.elect_one(loc=loc, ip=ip):
        llvm.inline_asm(
            None,
            [mbar_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];",
            "r",
            has_side_effects=True,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def nanosleep_skew(tidx: cutlass.Int32, salt: cutlass.Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(tidx).ir_value(loc=loc, ip=ip),
            cutlass.Int32(salt).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .u32 ns;\n\t"
        "xor.b32 ns, $0, $1;\n\t"
        "mul.lo.u32 ns, ns, 1103515245;\n\t"
        "shr.u32 ns, ns, 20;\n\t"
        "and.b32 ns, ns, 1023;\n\t"
        "add.u32 ns, ns, 1;\n\t"
        "nanosleep.u32 ns;\n"
        "}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tcgen05_fence_after_thread_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tcgen05_ld_32x32b_x32_store_bf16(
    tmem_addr: cutlass.Int32,
    out: cute.Tensor,
    row: cutlass.Int32,
    col: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    elem_offset = cutlass.Int64(row) * cutlass.Int64(out.stride[0]) + cutlass.Int64(col)
    byte_offset = elem_offset * cutlass.Int64(2)
    llvm.inline_asm(
        None,
        [
            out.iterator.llvm_ptr,
            byte_offset.ir_value(loc=loc, ip=ip),
            cutlass.Int32(tmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
        ".reg .b32 r8, r9, r10, r11, r12, r13, r14, r15;\n\t"
        ".reg .b32 r16, r17, r18, r19, r20, r21, r22, r23;\n\t"
        ".reg .b32 r24, r25, r26, r27, r28, r29, r30, r31;\n\t"
        ".reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .f32 f8, f9, f10, f11, f12, f13, f14, f15;\n\t"
        ".reg .f32 f16, f17, f18, f19, f20, f21, f22, f23;\n\t"
        ".reg .f32 f24, f25, f26, f27, f28, f29, f30, f31;\n\t"
        ".reg .b32 o0, o1, o2, o3, o4, o5, o6, o7;\n\t"
        ".reg .b32 o8, o9, o10, o11, o12, o13, o14, o15;\n\t"
        ".reg .u64 dst;\n\t"
        "add.u64 dst, $0, $1;\n\t"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{r0, r1, r2, r3, r4, r5, r6, r7, "
        "r8, r9, r10, r11, r12, r13, r14, r15, "
        "r16, r17, r18, r19, r20, r21, r22, r23, "
        "r24, r25, r26, r27, r28, r29, r30, r31}, [$2];\n\t"
        "mov.b32 f0, r0; mov.b32 f1, r1; mov.b32 f2, r2; mov.b32 f3, r3;\n\t"
        "mov.b32 f4, r4; mov.b32 f5, r5; mov.b32 f6, r6; mov.b32 f7, r7;\n\t"
        "mov.b32 f8, r8; mov.b32 f9, r9; mov.b32 f10, r10; mov.b32 f11, r11;\n\t"
        "mov.b32 f12, r12; mov.b32 f13, r13; mov.b32 f14, r14; mov.b32 f15, r15;\n\t"
        "mov.b32 f16, r16; mov.b32 f17, r17; mov.b32 f18, r18; mov.b32 f19, r19;\n\t"
        "mov.b32 f20, r20; mov.b32 f21, r21; mov.b32 f22, r22; mov.b32 f23, r23;\n\t"
        "mov.b32 f24, r24; mov.b32 f25, r25; mov.b32 f26, r26; mov.b32 f27, r27;\n\t"
        "mov.b32 f28, r28; mov.b32 f29, r29; mov.b32 f30, r30; mov.b32 f31, r31;\n\t"
        "cvt.rn.bf16x2.f32 o0, f1, f0; cvt.rn.bf16x2.f32 o1, f3, f2;\n\t"
        "cvt.rn.bf16x2.f32 o2, f5, f4; cvt.rn.bf16x2.f32 o3, f7, f6;\n\t"
        "cvt.rn.bf16x2.f32 o4, f9, f8; cvt.rn.bf16x2.f32 o5, f11, f10;\n\t"
        "cvt.rn.bf16x2.f32 o6, f13, f12; cvt.rn.bf16x2.f32 o7, f15, f14;\n\t"
        "cvt.rn.bf16x2.f32 o8, f17, f16; cvt.rn.bf16x2.f32 o9, f19, f18;\n\t"
        "cvt.rn.bf16x2.f32 o10, f21, f20; cvt.rn.bf16x2.f32 o11, f23, f22;\n\t"
        "cvt.rn.bf16x2.f32 o12, f25, f24; cvt.rn.bf16x2.f32 o13, f27, f26;\n\t"
        "cvt.rn.bf16x2.f32 o14, f29, f28; cvt.rn.bf16x2.f32 o15, f31, f30;\n\t"
        "st.global.v8.b32 [dst], {o0, o1, o2, o3, o4, o5, o6, o7};\n\t"
        "st.global.v8.b32 [dst+32], {o8, o9, o10, o11, o12, o13, o14, o15};\n"
        "}",
        "l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


class MinimalTcgen05MmaLd:
    def __init__(
        self,
        io_dtype=cutlass.BFloat16,
        use_fence: bool = True,
        sleep_skew: bool = False,
    ):
        self.io_dtype = io_dtype
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape_mnk = (128, 128, 16)
        self.mma_tiler_mnk = (128, 128, 16)
        self.threads_per_cta = 128
        self.ab_stages = 1
        self.acc_stages = 1
        self.num_tmem_cols = 512
        self.use_fence = use_fence
        self.sleep_skew = sleep_skew
        if io_dtype is not cutlass.BFloat16:
            raise ValueError(
                "This raw inline-PTX example currently hard-codes the BF16 descriptor"
            )
        self.mma_idesc = 136316048

        self.mma_op = tcgen05.MmaF16BF16Op(
            self.io_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32

        self.SharedStorage = SharedStorage

    def __call__(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        k = a.shape[1]
        n = b.shape[0]
        assumed_align = 32
        a_cute = (
            from_dlpack(a, assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=k)
        )
        b_cute = (
            from_dlpack(b, assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=k)
        )
        out_cute = (
            from_dlpack(out, assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=n)
        )
        self._launch(a_cute, b_cute, out_cute)

    @cute.jit
    def _launch(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
        tiled_mma = cute.make_tiled_mma(self.mma_op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler_mnk,
            a.element_type,
            self.ab_stages,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler_mnk,
            b.element_type,
            self.ab_stages,
        )
        a_smem_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
        b_smem_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

        tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op,
            a,
            a_smem_one_stage,
            self.mma_tiler_mnk,
            tiled_mma,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op,
            b,
            b_smem_one_stage,
            self.mma_tiler_mnk,
            tiled_mma,
        )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            out,
            a_smem_layout,
            b_smem_layout,
            _name_prefix="minimal_tcgen05_mma_ld",
        ).launch(grid=(1, 1, 1), block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.SharedStorage)
        sA = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier
        )
        tmem.allocate(self.num_tmem_cols)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        num_tma_copy_bytes = cute.size_in_bytes(
            self.io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(self.io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.ab_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=num_tma_copy_bytes,
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        ).make_participants()

        mma_coord_mnk = (0, 0, None)
        gA = cute.local_tile(mA_mkl, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_nkl, self.mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        tCtAcc = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2]))

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        tmem_thread_offset = (tidx << 16) & 6291456

        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA[(None, 0)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            ab_full = ab_consumer.wait_and_advance()
            tcgen05_mma_f16_128x128x16(
                tmem_ptr,
                tCrA[(None, None, 0, ab_full.index)],
                tCrB[(None, None, 0, ab_full.index)],
                cutlass.Int32(0),
                cutlass.Int32(self.mma_idesc),
            )
            ab_full.release()
            tcgen05_commit_arrive(acc_empty.barrier)

        tmem.relinquish_alloc_permit()
        if const_expr(self.sleep_skew):
            nanosleep_skew(tidx, cutlass.Int32(17))
        acc_full = acc_consumer.wait_and_advance()
        if const_expr(self.sleep_skew):
            nanosleep_skew(tidx, cutlass.Int32(29))
        if const_expr(self.use_fence):
            tcgen05_fence_after_thread_sync()
        if const_expr(self.sleep_skew):
            nanosleep_skew(tidx, cutlass.Int32(43))
        for tile_idx in cutlass.range_constexpr(4):
            tcgen05_ld_32x32b_x32_store_bf16(
                tmem_ptr.toint() + tmem_thread_offset + tile_idx * 32,
                mC_mnl,
                tidx,
                tile_idx * 32,
            )
        acc_full.release()

        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)


def make_input(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randint(-2, 3, (rows, cols), device="cuda", dtype=torch.int32).to(dtype)


def run(use_fence: bool = True, sleep_skew: bool = False) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this example requires an SM100 CUDA GPU")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("tcgen05 requires Blackwell / SM100 or newer")

    io_dtype = cutlass.BFloat16
    torch_dtype = cutlass_torch.dtype(io_dtype)
    a = make_input(128, 16, torch_dtype)
    b = make_input(128, 16, torch_dtype)
    out = torch.empty((128, 128), dtype=torch_dtype, device="cuda")

    MinimalTcgen05MmaLd(io_dtype, use_fence=use_fence, sleep_skew=sleep_skew)(a, b, out)
    torch.testing.assert_close(out, a @ b.T, rtol=2e-2, atol=2e-2)
    print("tcgen05 MMA -> fence -> LD correctness check passed")


def print_dump_hint(dump_dir: str | None) -> None:
    if dump_dir is None:
        return
    for path in sorted(Path(dump_dir).glob("*.ptx")):
        print(f"PTX: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--no-fence", action="store_true")
    parser.add_argument("--sleep-skew", action="store_true")
    args = parser.parse_args()

    run(use_fence=not args.no_fence, sleep_skew=args.sleep_skew)
    print_dump_hint(args.dump_dir)


if __name__ == "__main__":
    main()
