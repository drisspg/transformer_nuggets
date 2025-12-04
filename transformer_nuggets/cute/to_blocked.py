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
        gO: cute.Tensor,
        tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        blk_coord = ((None, None), bidx)
        gI_blk = gI[blk_coord]
        gO_blk = gO[blk_coord]

        # pyrefly: ignore  # no-matching-overload
        tXgI_tv = cute.composition(gI_blk, tv_layout)  # (Threads, Values) → gmem addr
        # pyrefly: ignore  # no-matching-overload
        tXgO_tv = cute.composition(gO_blk, tv_layout)  # (Threads, Values) → gmem addr

        tXgI = tXgI_tv[(tidx, None)]
        tXgO = tXgO_tv[(tidx, None)]

        tXrI = tXgI.load()
        tXgO.store(tXrI)

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

        # Each thread block handles K_BLOCKS_PER_TB worth of K (128 x 16 elements)
        # 128 threads, each handles 16 elements (4 elements per K-block × 4 K-blocks)
        thread_layout = cute.make_ordered_layout((128, 1), order=(1, 0))
        val_layout = cute.make_ordered_layout((1, 4 * self.K_BLOCKS_PER_TB), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thread_layout, val_layout)

        gI = cute.zipped_divide(input, tiler_mn)
        output_blocked = cute.make_tensor(output.iterator, block_scale_layout)
        gO = cute.zipped_divide(output_blocked, tiler_mn)

        self.kernel(gI, gO, tv_layout).launch(
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

    def main(trace: bool = False):
        M_chunks = 8192
        K_chunks = 128

        def bytes_tb_per_second(scales: torch.Tensor, time_us: float):
            return 2 * (scales.numel() * scales.element_size()) / time_us * 1e-6

        scales = torch.arange(
            (M_chunks * 128 * K_chunks * 4), device="cuda", dtype=torch.uint8
        ).view(M_chunks * 128, K_chunks * 4)
        out_cute = to_blocked_mx(scales)
        if not trace:
            from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

            time = benchmark_cuda_function_in_microseconds(to_blocked_mx, scales)
            print(f"Time taken: {time} microseconds, IO: {bytes_tb_per_second(scales, time)} TB/s")
            # pyrefly: ignore  # import-error
            from torchao.prototype.mx_formats.utils import to_blocked as to_blocked_ao

            time_ao = benchmark_cuda_function_in_microseconds(
                to_blocked_ao, scales, use_triton_kernel=True
            )
            print(
                f"Time taken: {time_ao} microseconds, IO: {bytes_tb_per_second(scales, time_ao)} TB/s"
            )
        print(out_cute.shape)

    CLI(main)
