"""Annotated Blackwell CuTeDSL GEMM example.

This script is a cleaned-up Python version of
`~/meta/my_scripts/cutey/gemm.ipynb`. It keeps the notebook notes close to the
code, but moves execution behind `main()` so importing the file does not compile
or launch kernels.

Run with:
    python examples/cutedsl_blackwell_gemm.py --m 8192 --n 8192 --k 8192 --benchmark

The kernel is written for Blackwell / SM100 CuTeDSL (`tcgen05`, TMEM, TMA).
"""

from abc import ABC, abstractmethod

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass import Constexpr, const_expr

from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

"""
### Gemms
Different params:

* IO dtype for A/B what is input to the MMA OP
* Acc Dtype: FP32
* C -> convert acc dtype to C before write
* mma_inst_shape_mnk: The shape of one tcgen05 mma instruction can deal with. See more details in [PTX Document 9.7.16.2.1. Matrix Shape](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape). From beginning, we choose the biggest one as it's easy to reach SOL. Driss: 128xNxK is basically requried but 256 should be good
* MMA_TILER_MKN: This is the logical blocking or tiling of C (the output) and thus tiling of A, B, C. Could be two cta but not here
* threads_per_cta, at least 128. Not sure why it NEEDs to be 1 warp group especially if tcgen can be issued via 1 thread in a warp
* ab_stages: The number demonstrates how many blocks that TMA can load before each block's computation. It's usually limited by the smem capacity. For mma_tiler_mnk (128, 256, 64), we can set it as 4 at most.
* acc_stage: As each CTA only computes one block of acc and stores out, the number is 1.


Lets do some Math, the smem size on blackwell is 227 KB, lets assume bf16
$$A smem: (128 * 64) * dtype\\_size * NUM\\_STAGES = (128 * 64) * 2 * 4 = 65,536$$
$$B smem: (256 * 64) * dtype\\_size * NUM\\_STAGES = (256 * 64) * 2 * 4 = 131,072$$

Adding them together is 196,608 or ~192 KB. If we chose 5 we would have 245,760 and would use too much smem!
"""

"""
## Layouts

* Tiled MMA. The tiled MMA helps calculate GEMM for one mma tile. We configurate it as tcgen05 MMA. I think this is a helper for turning A MNK cta shape and decomposing into a series of tcgenMMA instrctuions that can be used to build the bigger tile

* Smem layous for A and B. They must match the post-partitioned (CTA-local) shapes expected by the MMA instructions. `sm100_utils` provides functions to determine the post-partitioned shape. These functions take the tiled MMA and the MMA tiler as inputs and return a shape that is at least rank-3: mode 0 has the same extent as the raw MMA instruction, mode 1 counts how many times we replicate that MMA across the CTA's M/N tile, and mode 2 iterates across the CTA's K blocking. When we pipeline additional stages we effectively tack on another axis so each stage owns its own swizzled tile. The helpers also bake in the shared-memory swizzle/stride order so the cp.async/TMA copies land exactly where the `tcgen05` instructions read, keeping bank conflicts low and producer/consumer warps in lockstep.

* TMA descriptors for A & B. We've already know A, B tensors' info in both GMEM (global memory) & SMEM (shared memory). We take those to generate TMA descriptors & tme tensors. They helps load a block of A & B from GMEM to SMEM.
"""


class CuteOP(ABC):
    @abstractmethod
    def kernel(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def interface(self, *args, **kwargs):
        pass


class GemmFake(CuteOP):
    def __init__(
        self,
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        mma_tiler_mnk,
        ab_stages: int,
        acc_stage: int,
        threads_per_cta: int,
        **kwargs,
    ):
        self.io_dtype = io_dtype
        self.acc_dtype = acc_dtype
        self.mma_inst_shape_mnk = mma_inst_shape_mnk
        self.mma_tiler_mnk = mma_tiler_mnk
        self.ab_stages = ab_stages
        self.acc_stage = acc_stage
        self.threads_per_cta = threads_per_cta

        self.op = tcgen05.MmaF16BF16Op(
            io_dtype,
            acc_dtype,
            mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )

    def interface(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, aligned: bool = False):
        """This is the top dog takes in torch inputs spits out torch outputs.
        It's the interface that user will see.
        """
        # By marking 1 as leading dim, we let M, N vary but ensure that K dim is always contiguous
        # We are outputing in K-major order or Row Major
        k = a.shape[1]
        n = b.shape[0]
        assumed_align = 32 if aligned else None
        a_tensor = (
            from_dlpack(a, assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=k)
        )
        b_tensor = (
            from_dlpack(b, assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=k)
        )
        c_tensor = (
            from_dlpack(c, assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=n)
        )
        return self(a_tensor, b_tensor, c_tensor)

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
        pass

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        # Construct tiled MMA
        tiled_mma = cute.make_tiled_mma(self.op)

        """ Weird this baisscally is 1 thread owns the full slice for this MMA op shape, maybe that make sense w/ new tcgen ops
        Tiled MMA
        Thr Layout VMNK: (1,1,1,1):(0,0,0,0)
        Permutation MNK: (_,_,_)
        MMA Atom
        ThrID:           1:0
        Shape MNK:       (128,256,16)  # we chose k = 16 in the op
        TV Layout A:     (1,(128,16)):(128,(1,128))
        TV Layout B:     (1,(256,16)):(256,(1,256))
        TV Layout C:     (1,(128,256)):(128,(1,128))
        """

        # Construct SMEM layouts for A and B
        # A -> S<3,4,3> o 0 o ((128,16),1,4,4):((64,1),0,16,8192)
        # B -> S<3,4,3> o 0 o ((256,16),1,4,4):((64,1),0,16,16384)
        """ This is a composed layout by the looks of it:

        We have the inner func which is a 343 swizzler, then the offset
        which is 0, and then the outer layout which is the base MMA tile (128, 16) for A
        and 256, 16 for B. This is the base op shape

        mma_tiler_mnk = (128, 256, 64) this is what we passed in, and we are going to need 4 mma calls
        to build up an entire CTA worth. Hence the first 4 on mode 2. This has a stride of 16 because we jump of the
        K major size. Then 4 for AB stages and we need to jump over the full MN X 16 X 4 = 128 * 16 * 4 = 8192

        Dittoish logic for B smem layout
        """
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
        # The trailing dim is the AB_stages mode so we just ingore that here for loading 1 stage
        a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
        b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

        # Construct TMA load atoms G2S = global to shared
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            op,
            a,
            a_smem_layout_one_stage,
            self.mma_tiler_mnk,
            tiled_mma,
        )
        """
        TMA issued by 1 thread and it holds all it owns the M X K and N X K data
        Copy Atom
            ThrID:         1:0
            TV Layout Src: (1,8192):(0,1)
            TV Layout Dst: (1,8192):(0,1)
            Value type:    f16
        tensor<(0,0) o (?,?):(1@1,1@0)>
        """
        b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            op,
            b,
            b_smem_layout_one_stage,
            self.mma_tiler_mnk,
            tiled_mma,
        )

        # Launch the kernel
        """
        This will be gM x gN x 1
        And the tiler will be the mn shapes: (128, 256)
        """
        grid_shape = cute.ceil_div((*c.layout.shape, 1), self.mma_tiler_mnk[:2])
        self.kernel(
            tiled_mma,
            a_tma_atom,
            a_tma_tensor,
            b_tma_atom,
            b_tma_tensor,
            c,
            a_smem_layout,
            b_smem_layout,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
        )


"""
## How to write the kernel

1. Prologue: everything before running the main reduction loop
2. The Main loop: brrrrrrr
3. Epilogue: cleanup and then store the results to global memory



### Prologue
* We figure out which MN tile in C we are working on
* mma-coord_mnk, actualy above is not right but we convert the block_idx to the mma_coord
* tidx, we slice the input partition(logical tile of data) to align w/ what is needed for TMEM
* we gett the warp idx since only single threads need to invoke some ops like the actual tcgen.mma

```text
Tensor Memory (TMEM) Layout for .32x32b with .num=2:
┌─────────────────────────────────────────────────────────────────┐
│                    Tensor Memory Lanes                          │
├────────┬────────┬────────┬────────┬─────┬────────┬────────┬─────┤
│ Lane 0 │ Lane 1 │ Lane 2 │ Lane 3 │ ... │Lane 30 │Lane 31 │     │
├────────┼────────┼────────┼────────┼─────┼────────┼────────┼─────┤
│ 64bits │ 64bits │ 64bits │ 64bits │ ... │ 64bits │ 64bits │     │
└────────┴────────┴────────┴────────┴─────┴────────┴────────┴─────┘
    ↑        ↑        ↑        ↑             ↑        ↑
    │        │        │        │             │        │
┌───┴───┐┌───┴───┐┌───┴───┐┌───┴───┐    ┌────┴───┐┌───┴───┐
│ T0    ││ T1    ││ T2    ││ T3    │    │ T30    ││ T31   │  ← Threads
│ r0,r1 ││ r0,r1 ││ r0,r1 ││ r0,r1 │    │ r0,r1  ││ r0,r1 │  ← Registers
└───────┘└───────┘└───────┘└───────┘    └────────┘└───────┘

Thread i owns:  r0 = bits [0:31]   of TMEM lane i
                r1 = bits [32:63]  of TMEM lane i
```

* we got to setup both smem for TMA and TMEM for the MMA ops
* TMA (global to shared) AND the tcgen.mma are async. We gotta set up a pipleine. Where TMA is the producer and once all bytes have arrived in smem we signal that MMA can consume the buffer and mma
* Also need to know when mma is done and the acc is read
* Usual partitoning
* cpasync.prefetch_descriptor -> shortency latency for async tma = we should mesauree

### mainloop
* This is the meat and potatoes. We have a loop that proccess all k tiles.
* We try prefetch A and B into our buffers to keep the pipeline busy
* Do MMA


### Epilogue
* get the view of the epilogue tiler, I think this is quite similar to A,B
* Load from tmem to registers
* We need this because at least in our case the fp32 accum is not the C outdtype. So we need to convert
* Do some tmem cleanup for next kerenels
* Finally store to C w/ tma or st.global
* Dellacote Tmem

** Subtiling the acumulator **

Naively since we own (128, 256) and 1 thread will own 256 values. we split this to increase ILP
"""


class GEMM(GemmFake):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stages * 2]
            # barrier for coordinating something..
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.acc_stage * 2]
            # This is the smem buffer that will hold the tmem pointer
            tmem_holding_buf: cutlass.Int32

        self.SharedStorage = SharedStorage
        self.NUM_TMEM_COLS = 512
        """
        So what this flag does is that we want to be able to call the tma loads before we start the hot loop of mmas
        for i in range(prefetch):
            acquire_and_wait()
            tma_load()
        for k in range(num_k_tiles):
            if k >= prefetch:
                acquire_and_wait()
                mma()
            else:
                mma()

        """
        self.PREFETCH: Constexpr = kwargs.get("prefetch", True)

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
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        bidx, bidy, _ = cute.arch.block_idx()
        mma_coord_mnk = (bidx, bidy, None)

        ## Now we know where we are lets setup the memory

        # Setup the shared storage
        smem = cutlass.utils.SmemAllocator()
        # bars
        storage = smem.allocate(self.SharedStorage)
        sA = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=a_smem_layout.outer,  # this is the base layout
            byte_alignment=128,  # should probably check :0
            swizzle=a_smem_layout.inner,  # the layouts inner has this
        )
        # ditto
        sB = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # Lets setup tmem, hmm all threads participate in this
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier
        )
        # has to be less than 512 and multiple of 32
        tmem.allocate(self.NUM_TMEM_COLS)

        # TODO lets see if we can measure the diff here. Also do all threads need to participate in this?
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        # Now lets make the pipeline
        # pass in dtype and layout, we tma load 1 stage at a time so drop the last dim
        num_tma_copy_bytes = cute.size_in_bytes(
            self.io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(self.io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.ab_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            ),  # 1 thread is the producer issuing the tma load
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            ),  # 1 thread is the consummer issuing the mma
            tx_count=num_tma_copy_bytes,  # Number of bytes expected to be written to the transaction barrier for one stage
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        ).make_participants()

        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            ),  # 1 thread is the producer issuing the tmem to smem
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta
            ),  # threads per cta consume the smem to registers and convert + write to global
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        ).make_participants()

        # okay memory is setup, pipelines are built now lets figure out what data we are working on
        """ Cheat sheet naming Conventions:
        mX = full tensor in global memory
        gX = tiled view of global memory
        sX = tiled view of smem
        Pattern: tXyZ
        tXyZ = thread view, partitioned by X's layout, in y memory space, tensor Z
        t  C  g  A
        │  │  │  └─ Tensor A
        │  │  └──── in Global memory
        │  └─────── partitioned according to C's thread layout
        └────────── thread view
        """
        # Basically we split A w/ the tiler and give me the [x, y] block positions view, since
        # we ar eusing MNK we need to skip M w/ A and we use proj for that
        # For Proj = None it means ignore basically
        # (bM, bK, RestK)
        gA = cute.local_tile(mA_mkl, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_nkl, self.mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC_mnl, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

        # This is the 128, 256, 64 mma slice we setup and by getting slice 0 we something..
        thr_mma = tiled_mma.get_slice(0)

        # This is the thread partitioned global tensor why the C?
        tCgA = thr_mma.partition_A(gA)  # (MMA, MMA_N, MMA_K)
        tCgB = thr_mma.partition_B(gB)  # (MMA, MMA_M, MMA_N)
        tCgC = thr_mma.partition_C(gC)  # (MMA, MMA_M, MMA_N)

        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K)
        tCrB = tiled_mma.make_fragment_B(sB)  # (MMA, MMA_N, MMA_K)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])  # (MMA, MMA_M, MMA_N)
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,  # cluster coord
            cute.make_layout(1),  # cluster Coord we are not using this feature
            cute.group_modes(sA, 0, 3),  # Flatten all modes but the stage mode
            cute.group_modes(tCgA, 0, 3),  # Flatten all modes but the stage mode
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # CTA-wide sync before retrieving the pointer to the start of the allocated TMEM
        # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        # Swap the pointer in tCtAcc
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        SUBTILE_COUNT = 4
        # this size + mode is baiscally the M atom and N atom size so (128, 256 // 4) = (128, 64)
        epi_tile_shape = (
            cute.size(tCtAcc, mode=[0, 0]),
            cute.size(tCtAcc, mode=[0, 1]) // SUBTILE_COUNT,
        )
        epi_tiler = (epi_tile_shape,)
        tCtAcc_epi = cute.zipped_divide(
            tCtAcc, epi_tiler
        )  # do the actual divison ((128, 64), 1, 4)
        gC_epi = cute.zipped_divide(tCgC, epi_tiler)

        # hmmm this makes me think that this has each thread do 64 * 32bits or 2048 bits of data per copy
        tmem_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
            cutlass.Float32,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

        tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)  # (TmemCpy,NumTmemCpy,NumTiles)
        tDgC = tmem_thr_copy.partition_D(gC_epi)  # (TmemCpy,NumTmemCpy,NumTiles)

        tCrAcc = cute.make_rmem_tensor(
            tDgC[None, None, 0].shape, self.acc_dtype
        )  # (TmemCpy,NumTmemCpy)
        tCrC = cute.make_rmem_tensor(
            tDgC[None, None, 0].shape, self.io_dtype
        )  # (TmemCpy,NumTmemCpy)

        # Main loop time baby
        num_k_tiles = cute.size(gA, mode=[2])

        # its a little surprpisng you can issue both of these ops from 1 warp tbh but cool
        if warp_idx == 0:
            # Wait for a empty accumulator buffer
            acc_empty = acc_producer.acquire_and_advance()
            prefetch_stages = self.ab_stages - 2 if const_expr(self.PREFETCH) else None
            for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=prefetch_stages):
                # Issue TMA loads
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, ab_empty.count)],  # walk tiles of k
                    tAsA[(None, ab_empty.index)],  # put into index in circular buffer
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )

                # Execute one K-block worth of MMA instructions
                ab_full = ab_consumer.wait_and_advance()
                num_k_blocks = cute.size(tCrA, mode=[2])
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, ab_full.index)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                # Signal that the A/B buffers have been consumed and are ready for the next load
                ab_full.release()

            # Signal that the accumulator is fully computed
            acc_empty.commit()

        # Release TMEM allocation lock
        tmem.relinquish_alloc_permit()

        # Wait for the accumulator buffer to be full
        acc_full = acc_consumer.wait_and_advance()

        # TMEM -> RMEM -> GEMM
        # Sub-tiling for better instruction-level parallelism
        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
            tCrC.store(tCrAcc.load().to(self.io_dtype))
            cute.autovec_copy(tCrC, tDgC[None, None, i])
        acc_full.release()

        # Deallocate TMEM
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)


def make_random_k_major_tensor(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(rows, cols, dtype=torch.int32).random_(-2, 2).to(dtype=dtype, device="cuda")


def benchmark_gemm(callable, label: str, m: int, n: int, k: int, peak_tflops: float | None):
    avg_time_us = benchmark_cuda_function_in_microseconds(callable)
    achieved_tflops = (m * n * k * 2) / (avg_time_us * 1_000_000)

    print(f"Performance Metrics for: {label}".center(80, "-"))
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Compute throughput: {achieved_tflops:.2f} TFLOP/s")
    if peak_tflops is not None:
        print(
            f"Peak utilization: {100 * achieved_tflops / peak_tflops:.1f}% of {peak_tflops:.1f} TFLOP/s"
        )


def run(
    m: int = 8192,
    n: int = 8192,
    k: int = 8192,
    benchmark: bool = False,
    aligned: bool = False,
    peak_tflops: float | None = None,
):
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a Blackwell GPU.")
        return

    io_dtype = cutlass.Float16
    acc_dtype = cutlass.Float32
    mma_inst_shape_mnk = (128, 256, 16)
    mma_tiler_mnk = (128, 256, 64)
    threads_per_cta = 128
    ab_stages = 4
    acc_stage = 1

    a = make_random_k_major_tensor(m, k, cutlass_torch.dtype(io_dtype))
    b = make_random_k_major_tensor(n, k, cutlass_torch.dtype(io_dtype))
    c = torch.empty((m, n), dtype=cutlass_torch.dtype(io_dtype), device="cuda")

    gemm_op = GEMM(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        mma_tiler_mnk,
        ab_stages,
        acc_stage,
        threads_per_cta,
        prefetch=True,
    )
    gemm_op.interface(a, b, c, aligned=aligned)
    torch.testing.assert_close(c, a @ b.T)
    print("Correctness check passed.")

    if benchmark:
        benchmark_gemm(
            lambda: gemm_op.interface(a, b, c, aligned=aligned),
            f"PREFETCH: {True}, ALIGNED: {aligned}",
            m,
            n,
            k,
            peak_tflops,
        )

        no_prefetch_op = GEMM(
            io_dtype,
            acc_dtype,
            mma_inst_shape_mnk,
            mma_tiler_mnk,
            ab_stages,
            acc_stage,
            threads_per_cta,
            prefetch=False,
        )
        benchmark_gemm(
            lambda: no_prefetch_op.interface(a, b, c, aligned=aligned),
            f"PREFETCH: {False}, ALIGNED: {aligned}",
            m,
            n,
            k,
            peak_tflops,
        )

        if not aligned:
            benchmark_gemm(
                lambda: gemm_op.interface(a, b, c, aligned=True),
                f"PREFETCH: {True}, ALIGNED: {True}",
                m,
                n,
                k,
                peak_tflops,
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--peak-tflops", type=float, default=None)
    args = parser.parse_args()
    run(
        m=args.m,
        n=args.n,
        k=args.k,
        benchmark=args.benchmark,
        aligned=args.aligned,
        peak_tflops=args.peak_tflops,
    )


if __name__ == "__main__":
    main()
