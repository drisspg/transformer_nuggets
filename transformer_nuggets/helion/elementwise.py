import torch

import helion
import helion.language as hl

from transformer_nuggets.cute.dynamic_args import cute_add


@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # match pytorch broadcasting rules
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        # match type promotion of torch.add
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    # tile will be a tuple of blocks
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def main() -> None:
    M, N = 8192, 8192
    x = torch.randn([M, N], device="cuda", dtype=torch.float16).transpose(0, 1)
    y = torch.randn([M, N], device="cuda", dtype=torch.float16).transpose(0, 1)

    out = add(x, y)
    out_cute = cute_add(x, y)
    out_eager = torch.add(x, y)
    torch.testing.assert_close(out, out_cute)
    torch.testing.assert_close(out, out_eager)
    # time = benchmark_cuda_function_in_microseconds_triton(add, x, y)
    # mem_ops = M * N * 3 * x.element_size()
    # print(f"Helion Bandwidth (GB/s): {mem_ops / time / 1e3}")

    # time_cute = benchmark_cuda_function_in_microseconds_triton(cute_add, x, y)
    # print(f"Cute Bandwidth (GB/s): {mem_ops / time_cute / 1e3}")

    # time_eager = benchmark_cuda_function_in_microseconds_triton(torch.add, x, y)
    # print(f"PyTorch Eager Bandwidth (GB/s): {mem_ops / time_eager / 1e3}")


if __name__ == "__main__":
    main()
