from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda

_VALID_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
TMA_ALIGNMENT_BYTES = 16


def _contains_torch_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_torch_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_torch_tensor(key) or _contains_torch_tensor(item)
            for key, item in value.items()
        )
    return False


def torch_dtype_to_cute_dtype(dtype: torch.dtype):
    """Map a supported Torch dtype to its CuTeDSL numeric type."""
    match dtype:
        case torch.float16:
            return cutlass.Float16
        case torch.bfloat16:
            return cutlass.BFloat16
        case torch.float32:
            return cutlass.Float32
        case torch.uint8:
            return cutlass.Uint8
        case torch.int8:
            return cutlass.Int8
        case torch.int32:
            return cutlass.Int32
        case _:
            raise TypeError(f"Unsupported torch dtype for CuTeDSL: {dtype}")


def current_cuda_stream() -> cuda.CUstream:
    """Return the current Torch CUDA stream as a CUDA Python handle."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@lru_cache(maxsize=8)
def get_device_properties(device: torch.device) -> Any:
    """Return cached CUDA properties for a device."""
    return torch.cuda.get_device_properties(device)


def requires_int64_abi(*tensors: torch.Tensor | None) -> bool:
    """Check reachable offsets and every stride exposed through the CuTe ABI."""
    for tensor in tensors:
        if tensor is None:
            continue
        strides = tensor.stride()
        if any(abs(stride) > 2**31 - 1 for stride in strides):
            return True
        if (
            tensor.numel()
            and 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, strides)) > 2**31
        ):
            return True
    return False


def tensor_supports_contiguous_dim(
    tensor: torch.Tensor,
    *,
    dim: int = -1,
    alignment_bytes: int = 1,
) -> bool:
    """Return whether one mode is contiguous with aligned slice origins."""
    if tensor.ndim == 0 or not -tensor.ndim <= dim < tensor.ndim or alignment_bytes < 1:
        return False
    dim %= tensor.ndim
    element_size = tensor.element_size()
    return (
        tensor.stride(dim) == 1
        and tensor.data_ptr() % alignment_bytes == 0
        and all(
            index == dim or stride * element_size % alignment_bytes == 0
            for index, stride in enumerate(tensor.stride())
        )
    )


def tensor_supports_tma(tensor: torch.Tensor) -> bool:
    """Return whether a CUDA tensor has a TMA-compatible aligned row layout."""
    return tensor.is_cuda and tensor_supports_contiguous_dim(
        tensor,
        alignment_bytes=TMA_ALIGNMENT_BYTES,
    )


def make_fake_strided_tensor(
    dtype: Any,
    shape: tuple[Any, ...],
    *,
    contiguous_dim: int = -1,
    stride_divisibility: int = 1,
    assumed_align: int | None = None,
    use_int64_strides: bool = True,
) -> Any:
    """Create a fake tensor with one contiguous mode and dynamic other strides."""
    if not shape:
        raise ValueError("make_fake_strided_tensor requires at least one dimension")
    if not -len(shape) <= contiguous_dim < len(shape):
        raise ValueError(f"contiguous_dim is out of range for rank {len(shape)}")
    if stride_divisibility < 1:
        raise ValueError("stride_divisibility must be positive")
    if assumed_align is not None and assumed_align < 1:
        raise ValueError("assumed_align must be positive")
    contiguous_dim %= len(shape)
    sym_int = cute.sym_int64 if use_int64_strides else cute.sym_int
    strides = tuple(
        1 if index == contiguous_dim else sym_int(divisibility=stride_divisibility)
        for index in range(len(shape))
    )
    if assumed_align is None:
        alignment_bits = stride_divisibility * dtype.width
        if alignment_bits % 8:
            raise ValueError("sub-byte fake tensors require an explicit assumed_align")
        assumed_align = max(1, alignment_bits // 8)
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=strides,
        assumed_align=assumed_align,
    )


def make_fake_tensor(dtype, shape: tuple, divisibility: int = 1, leading_dim: int = -1):
    """Compatibility wrapper for :func:`make_fake_strided_tensor`."""
    return make_fake_strided_tensor(
        dtype,
        shape,
        contiguous_dim=leading_dim,
        stride_divisibility=divisibility,
    )


def make_fake_compact_tensor(
    dtype, shape: tuple, stride_order: tuple[int, ...] | None = None, assumed_align: int = 16
):
    """Create a fake compact tensor for a TVM-FFI compile signature."""
    if stride_order is None:
        stride_order = tuple(reversed(range(len(shape))))
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=stride_order,
        assumed_align=assumed_align,
    )


def compile_tvm_ffi(
    entrypoint: Any,
    *compile_args: Any,
    name: str | None = None,
) -> Any:
    """Compile fake arguments with the canonical typed TVM-FFI stream ABI."""
    if name is None:
        get_name = getattr(entrypoint, "get_name", None)
        if not callable(get_name):
            raise TypeError("compile_tvm_ffi() requires entrypoint.get_name() or name=")
        name = get_name()
    if not isinstance(name, str) or _VALID_NAME.fullmatch(name) is None:
        raise ValueError(
            "CuTeDSL compile names must start with a lowercase letter and contain "
            f"only lowercase letters, digits, and underscores; got {name!r}"
        )
    if any(_contains_torch_tensor(arg) for arg in compile_args):
        raise TypeError("compile_tvm_ffi() accepts fake CuTe tensors, not runtime Torch tensors")

    jit_wrapper = entrypoint if hasattr(entrypoint, "set_name_prefix") else entrypoint.__call__
    jit_wrapper.set_name_prefix(name)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile[cute.EnableTVMFFI](entrypoint, *compile_args, stream)


def get_tensor_alignment(tensor: torch.Tensor, dim: int) -> int:
    """Return the largest power-of-two alignment shared by all slices along ``dim``."""
    if tensor.ndim == 0 or not -tensor.ndim <= dim < tensor.ndim:
        return 1
    dim %= tensor.ndim
    if tensor.stride(dim) != 1:
        return 1
    element_size = tensor.element_size()
    for alignment in (128, 64, 32, 16, 8, 4, 2):
        if tensor.data_ptr() % alignment == 0 and all(
            index == dim or stride * element_size % alignment == 0
            for index, stride in enumerate(tensor.stride())
        ):
            return alignment
    return 1


def get_max_power_of_two_divisibility(value: int, cap: int = 128) -> int:
    """Return the largest power-of-two divisor of value, capped by cap."""
    divisibility = 1
    while divisibility < cap and value % (divisibility * 2) == 0:
        divisibility *= 2
    return divisibility


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

        return f"tensor_{inner_part}_dtype={tensor._dtype}"
    else:
        # Fallback if format is different

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

            pos = g()

            n = pos // M

            m = pos % M
            if DEBUG:
                print(f"tid={tid}, vid={vid} -> pos={pos} -> (m,n)=({m},{n})")
            if filled[m, n]:
                continue
            thr_ids[m, n] = tid
            val_ids[m, n] = vid
            filled[m, n] = True

    if color_fn is None:
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
        thread_layout = (thread_layout.shape, thread_layout.stride)
    if isinstance(value_layout, cute.Layout):
        value_layout = (value_layout.shape, value_layout.stride)

    @cute.jit
    def get_tv_layout():
        thread_cute_layout = cute.make_layout(thread_layout[0], stride=thread_layout[1])
        value_cute_layout = cute.make_layout(value_layout[0], stride=value_layout[1])
        tiler_mn, tv_layout = cute.make_layout_tv(thread_cute_layout, value_cute_layout)
        return tiler_mn, tv_layout.shape, tv_layout.stride

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
