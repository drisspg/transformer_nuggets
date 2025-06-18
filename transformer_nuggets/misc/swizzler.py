import torch
import matplotlib.pyplot as plt


def swizzle_functor(mask_bits, least_sig_bits, shift_distance):
    """Creates a functor that applies the swizzle pattern.

    Args:
        mask_bits: The number of bits to mask.
        least_sig_bits: The number of least significant bits to keep.
        shift_distance: The distance to shift the bits.
    """
    assert least_sig_bits >= 0
    assert mask_bits >= 0
    assert abs(shift_distance) >= mask_bits

    bit_mask = (1 << mask_bits) - 1
    yy_mask = bit_mask << (least_sig_bits + max(0, shift_distance))

    def swizzler(index):
        return index ^ ((index & yy_mask) >> shift_distance)

    return swizzler


def plot_bank_pattern(indices: torch.Tensor, title, vec_size=1, figsize=(6, 6)):
    """Plot the shared memory bank access pattern.

    Args:
        indices: 2D array of bank indices
        title: Plot title
        figsize: Figure size tuple
    """
    banks = indices % (32 / vec_size)
    banks = banks.numpy()

    plt.figure(figsize=figsize)
    plt.imshow(banks, cmap="tab10")

    # Add text annotations
    for i in range(banks.shape[0]):
        for j in range(banks.shape[1]):
            plt.text(j, i, str(banks[i, j]), ha="center", va="center", color="black")

    plt.colorbar(label="Bank Index")
    plt.title(f"{title}\n__shared__ {indices.shape[0]}x{indices.shape[1]}")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    # Example from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#example-matrix-transpose
    # And https://leimao.github.io/blog/CuTe-Swizzle/

    # How to create your swizzler, lets do an example
    # We are going to assume transaction size is 16 bytes int4 4*4 this means we have 8 groups
    # of 4 banks. Effectively 8 transactions before we hit a bank conflict

    # So how do we set up the swizzler to avoid this?
    # Lets first get enough bits for all the effective banks
    # log2(8) = 3
    # We dont need to preserve any ordering between consective transatcitons so we set
    # least_sig_bits = 0
    # We know that consectutive rows differ by N = 8, We want to ensure that

    # We have a 8x8 matrix in shmem but we aren't doing Vectorized memory access
    # least_sig = log2(N) : N is the size of the vecotrized load
    # mask_bits = log2(32 * 4 / elem_size) - least_sig : Total number of elements that make up a full bank
    # shift_distance = log2(fast_dim of shmem - least_sig)
    # In this case
    # Elem_size = 4
    # Vecotr size = 4
    # Fast dim of shsmem is 8
    # So least_sig = log2(4) = 2
    # mask_bits = log2(32*4/4) - 2 = log2(32) = 3
    # shift_distance = log2(8 - 0) = 3 (8 since our shemem fast dim is 8)
    # assert get_optimal_params(4, 4, 8) == (3, 2, 3), f"Got {get_optimal_params(4, 4, 8)}"
    swizzler = swizzle_functor(3, 2, 3)
    indices = torch.arange(64 * 4).view(32, 64)
    print(indices)
    print(swizzler(indices))

    title = "No Swizzle"
    plt = plot_bank_pattern(indices, title)
    plt.savefig(title)
