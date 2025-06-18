from typing import Optional
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class FlexAttentionKernelArgs:
    """Arguments that can be passed to control the behavior of the FlexAttention kernels.
    These arguments get converted to a dictionary of string-value pairs when passed to the kernels.
    """

    # Performance tuning options
    num_warps: int | None = None
    """Number of warps to use in the CUDA kernel."""

    num_stages: int | None = None
    """Number of pipeline stages to use in the CUDA kernel."""

    BLOCK_M: int | None = None
    """Thread block size across the seqlen dim of Q."""

    BLOCK_N: int | None = None
    """Block size to iterate over across the seqlen dim of K/V in each thread block."""

    # Numerical behavior options
    PRESCALE_QK: bool = False
    """Whether to pre-scale QK by 1/sqrt(d) and change of base. About 20% more numerical error but slightly faster."""

    ROWS_GUARANTEED_SAFE: bool = False
    """Is it guaranteed that at least one value in each row is not masked out? Allows skipping safety checks."""

    BLOCKS_ARE_CONTIGUOUS: bool = False
    """Is it guaranteed that all blocks in the mask are contiguous? Allows optimizing block traversal."""

    WRITE_DQ: bool = True
    """Controls whether gradients are computed in the DQ iteration loop of backwards pass."""

    FORCE_USE_FLEX_ATTENTION: bool = False
    """If this flag is set this disallows the use of flex-decoding kernel"""

    def asdict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
