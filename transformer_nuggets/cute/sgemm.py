# Portions adapted from NVIDIA CUTLASS's CuTeDSL Ampere SGEMM example.
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Full-FP32 SIMT GEMM (SGEMM) examples in CuTeDSL.

These kernels compute ``C = A @ B`` with FP32 inputs, FP32 accumulation, and FP32
output using CUDA-core FFMAs (``MmaUniversalOp``), not tensor cores. The comparison
point is cuBLAS FP32 SGEMM (``torch.mm`` with TF32 disabled).

* ``naive_sgemm``: one thread per output element, reading A and B straight from
  global memory. It exists as a readable baseline.
* ``sgemm`` / ``Sgemm``: Blackwell warp-specialized CTA-tiled kernel. A producer
  warp streams A and B into a shared-memory ring with TMA. Consumer warps use
  CuTe's universal-MMA partitioning and packed full-FP32 ``FFMA2`` arithmetic.

Set ``SGEMM_PROFILE=1`` and pass a compact profile buffer to compile in producer
and consumer-warp-0 timing. The default specialization has no profiling storage
or timer instructions. See ``examples/cute_sgemm_profile.py``.

All four compact input layouts are supported, not arbitrary strided views.
M, N, and K must be positive multiples of four and tensor offsets must fit int32.
Use one compilation target per process; the shared compilation cache does not
support switching between heterogeneous GPU architectures.

Layout convention: the kernels view A as ``(M, K)``, B as ``(N, K)``, and C as
``(M, N)``. For ``torch.mm(a, b)`` with ``a: (M, K)`` and ``b: (K, N)``, the kernel's B
is ``b.t()``. All four row-/column-major input combinations are accepted directly;
transpose views select the specialization without a copy. Unlike BLAS, this API
has no N/T flags: pass tensors whose logical shapes already describe ``a @ b``.
These examples provide forward computation only, without an autograd formula.
"""

import os
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Literal

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import cute, pipeline
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.nvgpu.warpgroup import SmemLayoutAtomKind

from transformer_nuggets.cute.base import CuteOp
from transformer_nuggets.cute.cache import compile_tvm_ffi_and_cache
from transformer_nuggets.cute.profiler import (
    compact_flush_smem_to_gmem,
    compact_prepare_smem,
    compact_profile_region,
)
from transformer_nuggets.cute.profiler.ops import ENABLE_PROFILING_CONST
from transformer_nuggets.cute.utils import get_device_properties, make_fake_compact_tensor


class SgemmProfileTag(IntEnum):
    PRODUCER_ACQUIRE = 0
    PRODUCER_ISSUE = 1
    CONSUMER_WAIT = 2
    CONSUMER_COMPUTE = 3


SGEMM_PROFILE_TILES = 32
SGEMM_PROFILE_EVENTS = len(SgemmProfileTag) * SGEMM_PROFILE_TILES
SGEMM_PROFILE_TAGS = tuple(tag.name.lower() for tag in SgemmProfileTag)


Major = Literal["mn", "k"]
"""Contiguous mode of an operand: the M/N mode or the K mode."""

VECTOR_ELEMENTS = 4
"""FP32 elements per 16 bytes: the TMA stride granule and one LDS.128."""

K_MAJOR_SWIZZLE = {
    8: SmemLayoutAtomKind.K_SW32,
    16: SmemLayoutAtomKind.K_SW64,
    32: SmemLayoutAtomKind.K_SW128,
}
"""K-major smem swizzle whose row width is ``tile_k`` FP32 elements."""


@dataclass(frozen=True)
class SgemmConfig:
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 16
    num_stages: int = 4
    num_threads: int = 256
    """Consumer (FFMA) threads; one extra warp issues the TMA loads."""
    threads_n: int = 16
    """Consumer threads along N; the rest spread along M."""
    k_group: int = 4
    """k values loaded into registers per smem read; K-major reads are this wide."""

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int:
                raise TypeError(f"{field.name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field.name} must be positive")
        if self.num_threads + 32 > 1024:
            raise ValueError("num_threads plus the TMA warp must not exceed 1024")
        if self.num_threads % 32:
            raise ValueError("num_threads must be a multiple of 32")
        if self.tile_k not in K_MAJOR_SWIZZLE:
            raise ValueError(f"tile_k must be one of {sorted(K_MAJOR_SWIZZLE)}")
        if self.num_stages < 2:
            raise ValueError("num_stages must be at least 2")
        if self.threads_n & (self.threads_n - 1):
            raise ValueError("threads_n must be a power of two")
        if self.tile_m % 8 or self.tile_n % 8:
            raise ValueError("tile_m and tile_n must be multiples of eight for the TMA swizzle")
        if self.k_group not in (1, 2, 4) or self.num_threads % self.threads_n:
            raise ValueError("k_group must be 1, 2, or 4 and threads_n must divide num_threads")
        m_granule = self.threads_m * VECTOR_ELEMENTS
        n_granule = self.threads_n * VECTOR_ELEMENTS
        if self.tile_m % m_granule or self.tile_n % n_granule:
            raise ValueError(
                f"tile_m must be a multiple of {m_granule} and tile_n a multiple of "
                f"{n_granule} for {self.num_threads} threads"
            )
        if max(self.tile_m, self.tile_n) > 256:
            raise ValueError("TMA boxes are limited to 256 elements per mode")

    @property
    def threads_m(self) -> int:
        return self.num_threads // self.threads_n

    @property
    def name(self) -> str:
        return (
            f"{self.tile_m}x{self.tile_n}x{self.tile_k}_s{self.num_stages}_t{self.num_threads}"
            f"_n{self.threads_n}_g{self.k_group}"
        )


class Sgemm(CuteOp[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
    """Warp-specialized FP32 SIMT GEMM with a TMA producer warp.

    One warp streams ``(tile_mn, tile_k)`` boxes of A ``(M, K)`` and B ``(N, K)`` into a
    ``num_stages`` smem ring; consumer warps use a ``MmaUniversalOp(Float32)`` tiled
    MMA to partition the work, then issue packed FFMAs on pairs of N columns.

    Smem keeps each operand's contiguous mode (TMA cannot transpose), and the register
    fragment is ordered along that mode. With the default four-element K group,
    both majors can use LDS.128:

    * MN-major: each thread owns runs of four consecutive M (or N) indices.
    * K-major: each thread owns strided M (or N) indices and loads four consecutive k
      per index. Rows are swizzled so the eight rows one 8-lane LDS.128 phase reads
      fall in distinct banks.

    TMA zero-fills out-of-bounds box elements, so ragged M, N, and K need no load
    predicates. C is row-major ``(M, N)``.
    """

    def __init__(
        self,
        config: SgemmConfig | None = None,
        *,
        a_major: Major = "k",
        b_major: Major = "mn",
        enable_profiling: bool | None = None,
    ):
        super().__init__()
        self.config = config or SgemmConfig()
        if a_major not in ("mn", "k") or b_major not in ("mn", "k"):
            raise ValueError("a_major and b_major must be mn or k")
        self.enable_profiling = (
            os.environ.get("SGEMM_PROFILE", "0") == "1"
            if enable_profiling is None
            else enable_profiling
        )
        if self.enable_profiling and not ENABLE_PROFILING_CONST:
            raise ValueError("SGEMM_PROFILE requires TNUGGETS_ENABLE_PROFILING enabled at import")
        self.a_major = a_major
        self.b_major = b_major
        self.cta_tiler = (self.config.tile_m, self.config.tile_n, self.config.tile_k)

    def get_name(self) -> str:
        suffix = "_profile" if self.enable_profiling else ""
        return f"sgemm_{self.config.name}_a{self.a_major}_b{self.b_major}{suffix}"

    def get_key(self) -> str:
        return self.get_name()

    def make_smem_layout(self, major: Major, tile_mn: int) -> cute.Layout | cute.ComposedLayout:
        cfg = self.config
        shape = (tile_mn, cfg.tile_k, cfg.num_stages)
        if major == "mn":
            return cute.make_layout(shape)
        atom = warpgroup.make_smem_layout_atom(K_MAJOR_SWIZZLE[cfg.tile_k], cutlass.Float32)
        return cute.tile_to_shape(atom, shape, (0, 1, 2))

    @staticmethod
    def make_permutation(major: Major, num_threads: int) -> cute.Layout | None:
        """Give each thread four consecutive MN indices when MN is contiguous."""
        if major == "mn":
            return cute.make_layout((num_threads, VECTOR_ELEMENTS), stride=(VECTOR_ELEMENTS, 1))
        return None

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        profile_buffer: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        cfg = self.config
        sA_layout = self.make_smem_layout(self.a_major, cfg.tile_m)
        sB_layout = self.make_smem_layout(self.b_major, cfg.tile_n)
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_a = cpasync.make_tiled_tma_atom(
            tma_op, mA, cute.slice_(sA_layout, (None, None, 0)), (cfg.tile_m, cfg.tile_k)
        )
        tma_atom_b, tma_b = cpasync.make_tiled_tma_atom(
            tma_op, mB, cute.slice_(sB_layout, (None, None, 0)), (cfg.tile_n, cfg.tile_k)
        )

        atoms_layout = cute.make_layout(
            (cfg.threads_m, cfg.threads_n, 1), stride=(cfg.threads_n, 1, 0)
        )
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.MmaUniversalOp(cutlass.Float32),
            atoms_layout,
            permutation_mnk=(
                self.make_permutation(self.a_major, cfg.threads_m),
                self.make_permutation(self.b_major, cfg.threads_n),
                None,
            ),
        )

        @cute.struct
        class SharedStorage:
            barriers: cute.struct.MemRange[cutlass.Int64, cfg.num_stages * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sA_layout)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sB_layout)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            mC,
            sA_layout,
            sB_layout,
            tiled_mma,
            profile_buffer,
        ).launch(
            grid=(cute.size(cute.ceil_div(mC.shape, (cfg.tile_m, cfg.tile_n))), 1, 1),
            block=[cfg.num_threads + 32, 1, 1],
            smem=SharedStorage.size_in_bytes()
            + (8 * (1 + SGEMM_PROFILE_EVENTS) if self.enable_profiling else 0),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout | cute.ComposedLayout,
        sB_layout: cute.Layout | cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        profile_buffer: cute.Tensor | None,
    ):
        cfg = self.config
        num_consumer_warps = cfg.num_threads // 32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_idx, _, _ = cute.arch.block_idx()
        # Flatten the tile grid to avoid CUDA's 65,535-block grid-y limit.
        tiles_m = cute.ceil_div(mC.shape[0], cfg.tile_m)
        bidx, bidy = block_idx % tiles_m, block_idx // tiles_m
        tiler_coord = (bidx, bidy, None)

        if warp_idx == num_consumer_warps:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        record = (bidx == 0) & (bidy == 0)
        if cutlass.const_expr(self.enable_profiling):
            profile_records = smem.allocate_tensor(
                cutlass.Int64, cute.make_layout((1 + SGEMM_PROFILE_EVENTS,)), 8
            )
            compact_prepare_smem(profile_records, cutlass.Int32(SGEMM_PROFILE_EVENTS), record)
        else:
            profile_records = None
        sA = self.get_smem_tensor(storage.sA, sA_layout)
        sB = self.get_smem_tensor(storage.sB, sB_layout)
        tx_count = cute.size_in_bytes(
            cutlass.Float32, cute.slice_(sA_layout, (None, None, 0))
        ) + cute.size_in_bytes(cutlass.Float32, cute.slice_(sB_layout, (None, None, 0)))
        producer, consumer = pipeline.PipelineTmaAsync.create(
            num_stages=cfg.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            # One elected lane per consumer warp releases each stage.
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_consumer_warps),
            tx_count=tx_count,
            barrier_storage=storage.barriers.data_ptr(),
        ).make_participants()

        # (BLK_MN, BLK_K, k) tiles of the TMA coordinate tensors.
        gA = cute.local_tile(mA, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=self.cta_tiler, coord=tiler_coord, proj=(None, 1, 1))
        k_tile_count = cute.size(gA, mode=[2])

        if warp_idx == num_consumer_warps:
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB, 0, 2),
            )
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                sample = record & (k_tile < SGEMM_PROFILE_TILES)
                with self.profile_scope(
                    profile_records,
                    SgemmProfileTag.PRODUCER_ACQUIRE,
                    k_tile,
                    num_consumer_warps,
                    sample,
                ):
                    empty = producer.acquire_and_advance()
                with self.profile_scope(
                    profile_records,
                    SgemmProfileTag.PRODUCER_ISSUE,
                    k_tile,
                    num_consumer_warps,
                    sample,
                ):
                    cute.copy(
                        tma_atom_a,
                        tAgA[None, k_tile],
                        tAsA[None, empty.index],
                        tma_bar_ptr=empty.barrier,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[None, k_tile],
                        tBsB[None, empty.index],
                        tma_bar_ptr=empty.barrier,
                    )
            producer.tail()
        else:
            thr_mma = tiled_mma.get_slice(tidx)
            gC = cute.local_tile(mC, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, 1, None))
            tCgC = thr_mma.partition_C(gC)
            # N fastest, so each FFMA2 accumulator pair is adjacent in registers.
            tCrC = cute.make_rmem_tensor(
                cute.make_ordered_layout(tCgC.shape, (2, 1, 0)), cutlass.Float32
            )
            tCrC.fill(0.0)

            # Group K for register loads; the default group of four permits LDS.128
            # for K-major operands as well as MN-major operands.
            k_group = (None, None, cfg.k_group, None)
            tCsA = cute.logical_divide(thr_mma.partition_A(sA), k_group)
            tCsB = cute.logical_divide(thr_mma.partition_B(sB), k_group)
            tCrA = self.make_fragment(self.a_major, tCsA)
            tCrB = self.make_fragment(self.b_major, tCsB)
            num_k_groups = cfg.tile_k // cfg.k_group

            for k_tile in cutlass.range(k_tile_count, unroll=1):
                sample = record & (k_tile < SGEMM_PROFILE_TILES)
                with self.profile_scope(
                    profile_records,
                    SgemmProfileTag.CONSUMER_WAIT,
                    k_tile,
                    0,
                    sample,
                ):
                    full = consumer.wait_and_advance()
                with self.profile_scope(
                    profile_records,
                    SgemmProfileTag.CONSUMER_COMPUTE,
                    k_tile,
                    0,
                    sample,
                ):
                    for k_group_idx in cutlass.range_constexpr(num_k_groups):
                        cute.autovec_copy(tCsA[None, None, (None, k_group_idx), full.index], tCrA)
                        cute.autovec_copy(tCsB[None, None, (None, k_group_idx), full.index], tCrB)
                        if k_group_idx == num_k_groups - 1:
                            # All lanes finish their generic-proxy reads before the
                            # elected lane releases this stage for a TMA overwrite.
                            cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            full.release()
                        for k in cutlass.range_constexpr(cfg.k_group):
                            self.outer_product_ffma2(
                                tCrC, tCrA[None, None, k], tCrB[None, None, k]
                            )

            tCcC = thr_mma.partition_C(cute.make_identity_tensor(gC.shape))
            residue_mn = (
                mC.shape[0] - cutlass.Int32(cfg.tile_m) * bidx,
                mC.shape[1] - cutlass.Int32(cfg.tile_n) * bidy,
            )
            predC = cute.make_rmem_tensor(tCrC.layout, cutlass.Boolean)
            for i in cutlass.range(cute.size(tCrC), unroll_full=True):
                predC[i] = cute.elem_less(tCcC[i], residue_mn)
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
            cute.copy(atom, tCrC, tCgC, pred=predC)

        if cutlass.const_expr(self.enable_profiling):  # noqa: SIM102
            if record:
                compact_flush_smem_to_gmem(
                    profile_records,
                    profile_buffer,
                    cutlass.Int32(0),
                    cutlass.Int32(SGEMM_PROFILE_EVENTS),
                )

    def profile_scope(self, records, tag, tile, target_warp, record):
        return compact_profile_region(
            records,
            cutlass.Int32(SGEMM_PROFILE_EVENTS),
            cutlass.Int32(tag),
            cutlass.Int32(0),
            tile * len(SgemmProfileTag) + int(tag),
            target_warp=cutlass.Int32(target_warp),
            enabled=self.enable_profiling,
            record=record,
            smem=True,
        )

    @staticmethod
    def outer_product_ffma2(tCrC: cute.Tensor, tCrA: cute.Tensor, tCrB: cute.Tensor):
        """``C[m, n] += A[m] * B[n]`` with packed FFMA2, two N columns per instruction.

        Same per-element rounding as ``cute.gemm`` on ``MmaUniversalOp``, with half
        as many arithmetic instructions. Explicit packing is needed: the observed
        CuTeDSL 4.8.1 scalar universal-MMA lowering did not generate FFMA2.
        """
        for m in range(cute.size(tCrA)):
            for n in range(0, cute.size(tCrB), 2):
                tCrC[0, m, n], tCrC[0, m, n + 1] = cute.arch.fma_packed_f32x2(
                    (tCrA[m], tCrA[m]), (tCrB[n], tCrB[n + 1]), (tCrC[0, m, n], tCrC[0, m, n + 1])
                )

    @staticmethod
    def get_smem_tensor(buffer, layout: cute.Layout | cute.ComposedLayout) -> cute.Tensor:
        if isinstance(layout, cute.ComposedLayout):
            return buffer.get_tensor(layout.outer, swizzle=layout.inner)
        return buffer.get_tensor(layout)

    @staticmethod
    def make_fragment(major: Major, tCsX: cute.Tensor) -> cute.Tensor:
        """Registers for one k group, contiguous along the operand's contiguous mode."""
        shape = cute.slice_(tCsX, (None, None, (None, 0), 0)).shape
        order = (0, 1, 2) if major == "mn" else (1, 2, 0)
        return cute.make_rmem_tensor(cute.make_ordered_layout(shape, order), cutlass.Float32)

    def interface(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor,
        profile_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_sgemm_operands(a, b)
        validate_sgemm_output(a, b, out)
        if operand_major(a) != self.a_major or operand_major(b.t()) != self.b_major:
            raise ValueError("operand layouts do not match the Sgemm specialization")
        if get_device_properties(a.device).major < 10:
            raise ValueError("TMA/FFMA2 sgemm requires Blackwell or newer")
        if self.enable_profiling:
            if profile_buffer is None:
                raise ValueError(
                    "SGEMM_PROFILE=1 requires profile_buffer; see examples/cute_sgemm_profile.py"
                )
            if (
                profile_buffer.dtype != torch.int64
                or profile_buffer.device != a.device
                or profile_buffer.shape != (1 + SGEMM_PROFILE_EVENTS,)
                or not profile_buffer.is_contiguous()
                or profile_buffer.data_ptr() % 8
            ):
                raise ValueError(
                    "profile_buffer must be a compact aligned int64 (129,) tensor on the input device"
                )
            if any(tensors_overlap(profile_buffer, tensor) for tensor in (a, b, out)):
                raise ValueError("profile_buffer must not overlap inputs or out")
        elif profile_buffer is not None:
            raise ValueError("profile_buffer requires SGEMM_PROFILE=1")
        with torch.cuda.device(a.device):
            compile_sgemm(self)(a, b.t(), out, profile_buffer)
        return out


def operand_major(tensor: torch.Tensor) -> Major:
    """Return the contiguous mode of a compact 2-D ``(MN, K)`` operand."""
    if tensor.stride(1) == 1 and tensor.stride(0) == tensor.shape[1]:
        return "k"
    if tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]:
        return "mn"
    raise ValueError(f"operand must be compact in one mode, got strides {tensor.stride()}")


def compile_sgemm(op: Sgemm):
    """Compile ``op`` once per layout specialization with dynamic M, N, K."""
    m, n, k = (cute.sym_int(divisibility=VECTOR_ELEMENTS) for _ in range(3))

    def fake_operand(mn: cute.SymInt, major: Major):
        stride_order = (1, 0) if major == "k" else (0, 1)
        return make_fake_compact_tensor(cutlass.Float32, (mn, k), stride_order=stride_order)

    return compile_tvm_ffi_and_cache(
        op,
        op.get_key(),
        fake_operand(m, op.a_major),
        fake_operand(n, op.b_major),
        make_fake_compact_tensor(cutlass.Float32, (m, n)),
        (
            make_fake_compact_tensor(cutlass.Int64, (1 + SGEMM_PROFILE_EVENTS,), assumed_align=8)
            if op.enable_profiling
            else None
        ),
        name=op.get_name(),
    )


def validate_sgemm_operands(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("sgemm expects float32 operands")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes {tuple(a.shape)} @ {tuple(b.shape)}")
    if not a.is_cuda or a.device != b.device:
        raise ValueError("a and b must be CUDA tensors on the same device")
    if any(dim <= 0 for dim in (*a.shape, b.shape[1])):
        raise ValueError("M, N, and K must be positive")
    if any(dim % VECTOR_ELEMENTS for dim in (*a.shape, b.shape[1])):
        raise ValueError(f"M, N, and K must be multiples of {VECTOR_ELEMENTS}")
    if a.data_ptr() % 16 or b.data_ptr() % 16:
        raise ValueError("a and b must be 16-byte aligned")
    operand_major(a)
    operand_major(b.t())
    if max(a.numel(), b.numel(), a.shape[0] * b.shape[1]) >= 2**31:
        raise ValueError("sgemm currently requires 32-bit element offsets")


def validate_sgemm_output(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if out.dtype != torch.float32:
        raise TypeError("out must be float32")
    if out.device != a.device:
        raise ValueError("out must be on the input device")
    if out.shape != (a.shape[0], b.shape[1]) or not out.is_contiguous() or out.data_ptr() % 16:
        raise ValueError("out must be a contiguous, 16-byte aligned (M, N) tensor")
    if any(tensors_overlap(out, tensor) for tensor in (a, b)):
        raise ValueError("out must not overlap either input")


def tensors_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Check actual byte-range overlap after compactness/device validation."""
    return (
        left.data_ptr() < right.data_ptr() + right.numel() * right.element_size()
        and right.data_ptr() < left.data_ptr() + left.numel() * left.element_size()
    )


def sgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    config: SgemmConfig | None = None,
    *,
    profile_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``a @ b`` in full FP32 with the TMA-fed CuTeDSL SIMT kernel.

    ``a`` is ``(M, K)`` and ``b`` is ``(K, N)``; each may be row- or column-major.
    ``out`` must be a row-major ``(M, N)`` float32 tensor if provided.
    """
    validate_sgemm_operands(a, b)
    if out is None:
        out = a.new_empty(a.shape[0], b.shape[1])
    op = Sgemm(config, a_major=operand_major(a), b_major=operand_major(b.t()))
    return op.interface(a, b, out, profile_buffer)


@cute.kernel
def naive_sgemm_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    """One thread per C element; adjacent threads cover adjacent N columns."""
    tidx, tidy, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    tiles_n = cute.ceil_div(mC.shape[1], bdimx)
    n = (block_idx % tiles_n) * bdimx + tidx
    m = (block_idx // tiles_n) * bdimy + tidy
    if m < mC.shape[0] and n < mC.shape[1]:
        acc = cutlass.Float32(0.0)
        for k in range(mA.shape[1]):
            acc += mA[m, k] * mB[n, k]
        mC[m, n] = acc


@cute.jit
def naive_sgemm_launch(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream: cuda.CUstream):
    block = (32, 8, 1)
    naive_sgemm_kernel.set_name_prefix("naive_sgemm")
    naive_sgemm_kernel(mA, mB, mC).launch(
        grid=(cute.ceil_div(mC.shape[1], block[0]) * cute.ceil_div(mC.shape[0], block[1]), 1, 1),
        block=block,
        stream=stream,
    )


def naive_sgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ b`` with the naive kernel; ``a`` and ``b`` must be row-major."""
    validate_sgemm_operands(a, b)
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("naive_sgemm expects row-major operands")
    m, n, k = (cute.sym_int(divisibility=VECTOR_ELEMENTS) for _ in range(3))
    with torch.cuda.device(a.device):
        compiled = compile_tvm_ffi_and_cache(
            naive_sgemm_launch,
            "fp32_row_major",
            make_fake_compact_tensor(cutlass.Float32, (m, k)),
            make_fake_compact_tensor(cutlass.Float32, (n, k), stride_order=(0, 1)),
            make_fake_compact_tensor(cutlass.Float32, (m, n)),
            name="naive_sgemm",
        )
        out = a.new_empty(a.shape[0], b.shape[1])
        compiled(a, b.t(), out)
    return out
