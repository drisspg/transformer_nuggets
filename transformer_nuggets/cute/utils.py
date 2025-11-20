import torch
import hashlib
from typing import Any
import cutlass.cute as cute
from pathlib import Path


def get_tensor_alignment(tensor: torch.Tensor, dim: int) -> int:
    """Calculate the maximum alignment for a tensor assuming a specific dimension is contiguous.

    Args:
        tensor: The tensor to check
        dim: The dimension assumed to be contiguous (negative indexing supported)

    Returns:
        Maximum alignment in bytes that divides both the pointer and the contiguous region size
    """
    # Handle negative indexing
    if dim < 0:
        dim = tensor.ndim + dim

    # Get the size of the assumed contiguous dimension
    contiguous_elements = tensor.shape[dim]

    # Convert to bytes
    element_size = tensor.element_size()
    contiguous_bytes = contiguous_elements * element_size

    # Get pointer
    ptr = tensor.data_ptr()

    # Find the best alignment that divides both pointer and size
    max_align = 128

    while max_align > 1:
        if ptr % max_align == 0 and contiguous_bytes % max_align == 0:
            break
        max_align //= 2

    return max_align


def generate_tensor_cache_key(tensor: cute.Tensor) -> str:
    """Generate a cache key component for a CUTE tensor.

    Args:
        tensor: CUTE tensor to generate key for

    Returns:
        String representation suitable for cache key
    """
    tensor_str = str(tensor)
    if " o " in tensor_str and ")>" in tensor_str:
        # Extract everything after ' o ' and before '>'
        inner_part = tensor_str.split(" o ")[1].rstrip(">")
        # pyrefly: ignore  # missing-attribute
        return f"tensor_{inner_part}_dtype={tensor._dtype}"
    else:
        # Fallback if format is different
        # pyrefly: ignore  # missing-attribute
        return f"tensor_shape={tensor.shape}_dtype={tensor._dtype}"


def hash_cache_key(key_parts: list | tuple, use_sha256: bool = True) -> str:
    """Hash cache key components into a fixed-length string.

    Args:
        key_parts: List or tuple of cache key components
        use_sha256: If True, use SHA256 hash; otherwise join with underscores

    Returns:
        Hashed or joined cache key
    """
    key_str = "_".join(str(part) for part in key_parts)

    if use_sha256:
        return hashlib.sha256(key_str.encode()).hexdigest()
    else:
        return key_str


def extract_tensor_properties(tensor: torch.Tensor) -> dict[str, Any]:
    """Extract relevant properties from a PyTorch tensor for caching.

    Args:
        tensor: PyTorch tensor

    Returns:
        Dictionary of tensor properties
    """
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": tuple(tensor.stride()),
        "is_contiguous": tensor.is_contiguous(),
        "data_ptr": tensor.data_ptr(),
    }


def _visualize_tv_layout_impl(
    tiler_mn: tuple[int, int],
    shape: tuple,
    stride: tuple,
    save_path: str,
    *,
    thread_layout: tuple[tuple, tuple] | None = None,
    value_layout: tuple[tuple, tuple] | None = None,
    font_size: int = 32,
    cell_px: int = 200,
    grid_lw: float = 2.5,
    color_fn=None,
    DEBUG: bool = False,
):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if isinstance(shape[0], int):
        n_thr = shape[0]
    else:
        n_thr = math.prod(shape[0])
    if isinstance(shape[1], int):
        n_val = shape[1]
    else:
        n_val = math.prod(shape[1])

    M, N = tiler_mn
    thr_ids = np.full((M, N), -1, dtype=int)
    val_ids = np.full((M, N), -1, dtype=int)
    filled = np.zeros((M, N), dtype=bool)

    if DEBUG:
        if thread_layout is not None:
            print(f"Thread layout: {thread_layout}")
        if value_layout is not None:
            print(f"Value layout: {value_layout}")
        print(f"Tiler (M, N): {tiler_mn}")
        print(f"TV Layout shape: {shape}, stride: {stride}")
        print(f"Total threads: {n_thr}, total values: {n_val}")

    for tid in range(n_thr):
        for vid in range(n_val):

            @cute.jit
            def g():
                tv_layout = cute.make_layout(shape, stride=stride)
                return tv_layout((tid, vid))

            # pyrefly: ignore  # missing-argument
            pos = g()
            # pyrefly: ignore  # unsupported-operation
            n = pos // M
            # pyrefly: ignore  # unsupported-operation
            m = pos % M
            if DEBUG:
                print(f"tid={tid}, vid={vid} -> pos={pos} -> (m,n)=({m},{n})")
            if filled[m, n]:
                continue
            thr_ids[m, n] = tid
            val_ids[m, n] = vid
            filled[m, n] = True

    if color_fn is None:
        # pyrefly: ignore  # missing-attribute
        pastel = plt.cm.Set3.colors
        cmap = (pastel * ((n_thr // 12) + 1))[:n_thr]
        color_fn = lambda t, v: cmap[t % len(cmap)]

    bg_rgb = np.zeros((M, N, 3))
    for m in range(M):
        for n in range(N):
            tid = thr_ids[m, n]
            if tid >= 0:
                bg_rgb[m, n] = mcolors.to_rgb(color_fn(tid, val_ids[m, n]))

    fig_w, fig_h = N * cell_px / 100, M * cell_px / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(bg_rgb, interpolation="none")

    for m in range(M):
        for n in range(N):
            if thr_ids[m, n] >= 0:
                ax.text(
                    n,
                    m,
                    f"T{thr_ids[m, n]}\nV{val_ids[m, n]}",
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    weight="bold",
                )

    ax.set_xticks(np.arange(N + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(M + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linewidth=grid_lw)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(M - 0.5, -0.5)

    @cute.jit()
    def get_tv_layout_str():
        return str(cute.make_layout(shape, stride=stride))

    tv_layout_str = get_tv_layout_str()
    if thread_layout is not None and value_layout is not None:
        thread_str = f"{thread_layout[0]} : {thread_layout[1]}"
        value_str = f"{value_layout[0]} : {value_layout[1]}"
        title = f"Thread: {thread_str}\nValue: {value_str} \ntv_layout {tv_layout_str}"
    else:
        title = f"TV Layout: {shape} : {stride}\n{tv_layout_str}"
    ax.set_title(title, fontsize=font_size + 2, pad=12)

    plt.tight_layout()
    path = Path(save_path).with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(f"Saved to {path}")


def visualize_tv_layout(
    thread_layout: tuple[tuple, tuple],
    value_layout: tuple[tuple, tuple],
    save_path: str,
    *,
    font_size: int = 32,
    cell_px: int = 200,
    grid_lw: float = 2.5,
    color_fn=None,
    DEBUG: bool = False,
):
    """Visualize a T/V layout from thread and value layouts.

    Args:
        thread_layout: (shape, stride) tuple for thread layout
        value_layout: (shape, stride) tuple for value layout
        save_path: Path to save the SVG file
        font_size: Font size for text labels
        cell_px: Cell size in pixels
        grid_lw: Grid line width
        color_fn: Optional function (tid, vid) -> color
    """
    if isinstance(thread_layout, cute.Layout):
        # pyrefly: ignore  # bad-assignment
        thread_layout = (thread_layout.shape, thread_layout.stride)
    if isinstance(value_layout, cute.Layout):
        # pyrefly: ignore  # bad-assignment
        value_layout = (value_layout.shape, value_layout.stride)

    @cute.jit
    def get_tv_layout():
        thread_cute_layout = cute.make_layout(thread_layout[0], stride=thread_layout[1])
        value_cute_layout = cute.make_layout(value_layout[0], stride=value_layout[1])
        tiler_mn, tv_layout = cute.make_layout_tv(thread_cute_layout, value_cute_layout)
        return tiler_mn, tv_layout.shape, tv_layout.stride

    # pyrefly: ignore  # not-iterable, missing-argument
    tiler_mn, shape, stride = get_tv_layout()

    return _visualize_tv_layout_impl(
        tiler_mn,
        shape,
        stride,
        save_path,
        thread_layout=thread_layout,
        value_layout=value_layout,
        font_size=font_size,
        cell_px=cell_px,
        grid_lw=grid_lw,
        color_fn=color_fn,
        DEBUG=DEBUG,
    )


def visualize_tv_layout_direct(
    tv_layout: tuple[tuple, tuple],
    tiler_mn: tuple[int, int],
    save_path: str,
    *,
    font_size: int = 32,
    cell_px: int = 200,
    grid_lw: float = 2.5,
    color_fn=None,
    DEBUG: bool = False,
):
    """Visualize a T/V layout directly from tv_layout and tiler_mn.

    Args:
        tv_layout: (shape, stride) tuple for the combined TV layout
        tiler_mn: (M, N) tuple for the tiler dimensions
        save_path: Path to save the SVG file
        font_size: Font size for text labels
        cell_px: Cell size in pixels
        grid_lw: Grid line width
        color_fn: Optional function (tid, vid) -> color
    """
    shape, stride = tv_layout

    return _visualize_tv_layout_impl(
        tiler_mn,
        shape,
        stride,
        save_path,
        font_size=font_size,
        cell_px=cell_px,
        grid_lw=grid_lw,
        color_fn=color_fn,
        DEBUG=DEBUG,
    )
