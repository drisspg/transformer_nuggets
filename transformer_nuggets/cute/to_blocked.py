import torch


import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from transformer_nuggets.cute.base import CuteOp


class ToBlocked(CuteOp):
    """Convert a mx/nv tensor to blocked format"""

    # Number of K-blocks (each 128x4) to process per thread block
    K_BLOCKS_PER_TB = 32

    def __init__(self):
        super().__init__()

    @cute.kernel()
    def kernel(
        self,
        gI: cute.Tensor,
        gO_blocked: cute.Tensor,
        tv_layout_swizzle: cute.Layout,
        tv_layout_linear: cute.Layout,
        linear_layout: cute.Layout,
        smem_swizzle: cute.Swizzle,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()

        blk_coord = ((None, None), bidx)
        gI_blk: cute.Tensor = gI[blk_coord]
        gO_blk: cute.Tensor = gO_blocked[blk_coord]

        tXgI_tv = cute.composition(gI_blk, tv_layout_swizzle)
        smem_layout_swizzled = cute.make_composed_layout(smem_swizzle, 0, gO_blk.layout)
        sBlk = smem.allocate_tensor(
            element_type=gO_blk.element_type,
            layout=smem_layout_swizzled,
            byte_alignment=128,
        )
        tXs_tv = cute.composition(sBlk, tv_layout_swizzle)

        tXgI = tXgI_tv[(tidx, None)]
        tXsO = tXs_tv[(tidx, None)]

        tXrI = tXgI.load()
        tXsO.store(tXrI)

        cute.arch.barrier()

        linear_layout_swizzled = cute.make_composed_layout(smem_swizzle, 0, linear_layout)
        s_linear = cute.make_tensor(sBlk.iterator, linear_layout_swizzled)
        g_linear = cute.make_tensor(gO_blk.iterator, linear_layout)

        s_tv = cute.composition(s_linear, tv_layout_linear)
        g_tv = cute.composition(g_linear, tv_layout_linear)

        s_thr = s_tv[(tidx, None)]
        g_thr = g_tv[(tidx, None)]

        tXO_ssa = s_thr.load()
        g_thr.store(tXO_ssa)

    @cutlass.dsl_user_op
    def get_block_swizzled_atom(self, *, loc=None, ip=None):
        """The atom layout for a single 128x4 block (512 elements)"""
        atom_shape = ((32, 4), 4)
        atom_stride = ((16, 4), 1)
        return cute.make_layout(atom_shape, stride=atom_stride)

    @cute.jit()
    def __call__(self, input: cute.Tensor, output: cute.Tensor):
        block_scale_atom = self.get_block_swizzled_atom()

        # Tile the atom layout to the full output shape
        block_scale_layout = cute.tile_to_shape(block_scale_atom, cute.shape(output), (2, 1))

        # === Blocked layout ===
        thread_layout = cute.make_ordered_layout((128, 1), order=(1, 0))
        val_layout = cute.make_ordered_layout((1, 4 * self.K_BLOCKS_PER_TB), order=(1, 0))
        tiler_mn, tv_layout_swizzle = cute.make_layout_tv(thread_layout, val_layout)

        # === TV layout for linear store (smem → output) ===
        num_threads = cute.size(thread_layout)
        num_values = cute.size(val_layout)
        tile_size = cute.size(tiler_mn)

        thr_layout_linear = cute.make_layout(num_threads)
        val_layout_linear = cute.make_layout(num_values)
        _, tv_layout_linear = cute.make_layout_tv(thr_layout_linear, val_layout_linear)
        linear_layout = cute.make_layout(tile_size)

        smem_swizzle = cute.make_swizzle(5, 2, 5)

        gI = cute.zipped_divide(input, tiler_mn)
        output_blocked = cute.make_tensor(output.iterator, block_scale_layout)
        gO_blocked = cute.zipped_divide(output_blocked, tiler_mn)

        self.kernel(
            gI, gO_blocked, tv_layout_swizzle, tv_layout_linear, linear_layout, smem_swizzle
        ).launch(
            grid=[cute.size(gI, mode=[1]), 1, 1],
            block=[cute.size(thread_layout), 1, 1],
        )

    def interface(self, scales: torch.Tensor):
        output = torch.empty_like(scales)

        assumed_align = 512  # Scales are 128*4 byte aligned
        # K dimension must be divisible by 4 * K_BLOCKS_PER_TB
        k_divisibility = 4 * self.K_BLOCKS_PER_TB
        input_cute = (
            from_dlpack(scales, assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=0, divisibility=128)
            .mark_compact_shape_dynamic(mode=1, divisibility=k_divisibility)
        )
        output_cute = (
            from_dlpack(output, assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=0, divisibility=128)
            .mark_compact_shape_dynamic(mode=1, divisibility=k_divisibility)
        )
        self(input_cute, output_cute)
        return output.view(-1, 32, 16)


Blocked = ToBlocked()


def to_blocked_mx(scales: torch.Tensor):
    return Blocked.interface(scales)


if __name__ == "__main__":
    from jsonargparse import CLI

    from torchao.prototype.mx_formats.utils import to_blocked as to_blocked_ao

    def main(trace: bool = False):
        M_chunks = 8192
        K_chunks = 128

        def bytes_tb_per_second(scales: torch.Tensor, time_us: float):
            return 2 * (scales.numel() * scales.element_size()) / time_us * 1e-6

        scales = (
            torch.randint(0, 256, size=(M_chunks * 128 * K_chunks * 4,), device="cuda")
            .to(torch.uint8)
            .view(M_chunks * 128, K_chunks * 4)
        )
        out_cute = to_blocked_mx(scales)
        out_ao = to_blocked_ao(scales, use_triton_kernel=True).view(-1, 32, 16)
        torch.testing.assert_close(out_cute, out_ao, atol=0, rtol=0)
        if not trace:
            from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

            time = benchmark_cuda_function_in_microseconds(to_blocked_mx, scales)
            print(
                f"Cute time taken: {time} microseconds, IO: {bytes_tb_per_second(scales, time)} TB/s"
            )
            time_ao = benchmark_cuda_function_in_microseconds(
                to_blocked_ao, scales, use_triton_kernel=True
            )
            print(
                f"AO triton time taken: {time_ao} microseconds, IO: {bytes_tb_per_second(scales, time_ao)} TB/s"
            )
        print(out_cute.shape)

    CLI(main)
