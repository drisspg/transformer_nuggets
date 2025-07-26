import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any
from collections.abc import Callable

import cutlass.cute as cute


logger = logging.getLogger(__name__)


class CuteKernelCache:
    def __init__(self, max_size: int = 1000, use_hashing: bool = False):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._lock = threading.Lock()
        self._use_hashing = use_hashing

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._stats["hits"] += 1
                return self._cache[key]
            self._stats["misses"] += 1
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                # Update existing entry and move to end
                self._cache[key] = value
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = value
                # Evict oldest if over limit
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)  # Remove oldest (FIFO)
                    self._stats["evictions"] += 1

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_stats(self) -> dict[str, int | float]:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0
            return {
                **self._stats,
                "total": total,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "max_size": self._max_size,
                "use_hashing": self._use_hashing,
            }


_kernel_cache = CuteKernelCache()


def _generate_cache_key(*args, use_hashing: bool = False, **kwargs) -> str:
    key_parts = []

    for arg in args:
        if isinstance(arg, cute.Tensor):
            # Get string representation and extract the shape:stride pattern
            tensor_str = str(arg)
            # Format is: Tensor<address@mem o (shape):(stride)>
            # We want just the (shape):(stride) part

            if " o " in tensor_str and ")>" in tensor_str:
                # Extract everything after ' o ' and before '>'
                inner_part = tensor_str.split(" o ")[1].rstrip(">")
                # inner_part should be like "(?,?):(?,1)"
                key_parts.append(f"tensor_{inner_part}_dtype={arg._dtype}")
            else:
                # Fallback if format is different
                key_parts.append(f"tensor_shape={arg.shape}_dtype={arg._dtype}")
        elif hasattr(arg, "__name__"):
            key_parts.append(f"op={arg.__name__}")
        else:
            key_parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    key_str = "_".join(key_parts)
    logger.debug(f"Generated cache key: {key_str}")

    if use_hashing:
        return hashlib.sha256(key_str.encode()).hexdigest()
    else:
        return key_str


def cute_compile_and_cache(func: Callable, *args, cache_extra=None, **kwargs):
    """
    Compile a @cute.jit decorated function and cache the result.

    The cache key is generated from the function name, tensor shapes/strides/dtypes,
    and any additional data provided via cache_extra. This allows caching different
    compilations for the same kernel when memory layout properties (like alignment)
    affect performance.

    Args:
        func: A function decorated with @cute.jit
        *args: Arguments to pass to cute.compile and for cache key generation.
               Typically includes cute.Tensor objects that define the kernel's
               input/output layout.
        cache_extra: Optional extra data to include in cache key. Useful for
                    caching based on properties not captured by tensor metadata
                    alone (e.g., memory alignment, vectorization hints).
                    Can be a single value or tuple/list of values.
        **kwargs: Keyword arguments to pass to cute.compile (e.g., block_size,
                 num_warps, etc.)

    Returns:
        Compiled kernel that can be executed with the same tensor arguments

    Example:
        @cute.jit
        def my_kernel(a, b, c):
            # kernel implementation
            ...

        # Basic usage - cache based on tensor shapes/strides
        compiled_kernel = cute_compile_and_cache(my_kernel, tensor_a, tensor_b, tensor_c)
        result = compiled_kernel(tensor_a, tensor_b, tensor_c)

        # Advanced usage - cache different compilations based on alignment
        align_a = get_tensor_alignment(tensor_a, dim=-1)
        align_b = get_tensor_alignment(tensor_b, dim=0)

        compiled_kernel = cute_compile_and_cache(
            my_kernel,
            tensor_a, tensor_b, tensor_c,
            cache_extra=(align_a, align_b),  # Different kernels for different alignments
            block_size=256  # Compilation parameter
        )

        # The same tensor shapes but different alignment will result in a cache miss
        # and a new compilation, allowing optimization for specific memory layouts
    """
    # Generate cache key from function and arguments
    cache_key = _generate_cache_key(*args, use_hashing=_kernel_cache._use_hashing, **kwargs)

    # Add extra cache key data if provided
    if cache_extra is not None:
        extra_str = (
            str(cache_extra)
            if not isinstance(cache_extra, (list, tuple))
            else "_".join(str(x) for x in cache_extra)
        )
        cache_key = f"{cache_key}_extra_{extra_str}"

    cache_key = f"{func.__name__}_{cache_key}"

    # Check cache
    compiled_kernel = _kernel_cache.get(cache_key)
    if compiled_kernel is not None:
        logger.debug(f"Cache hit for {func.__name__} (key: {cache_key})")
        return compiled_kernel

    logger.debug(f"Cache miss for {func.__name__} (key: {cache_key}) - Compiling...")
    compiled_kernel = cute.compile(func, *args, **kwargs)
    _kernel_cache.set(cache_key, compiled_kernel)
    return compiled_kernel


def clear_cute_cache():
    _kernel_cache.clear()


def get_cache_stats():
    return _kernel_cache.get_stats()


def set_cache_size(max_size: int):
    """Set the maximum cache size. If current cache exceeds new size, oldest entries are evicted."""
    global _kernel_cache
    with _kernel_cache._lock:
        _kernel_cache._max_size = max_size
        # Evict oldest entries if current cache exceeds new limit
        while len(_kernel_cache._cache) > max_size:
            _kernel_cache._cache.popitem(last=False)
            _kernel_cache._stats["evictions"] += 1


def set_cache_hashing(use_hashing: bool):
    """Enable or disable hashing of cache keys.

    When disabled (default), full string keys are used which are more readable.
    When enabled, SHA256 hashes are used which are shorter but less readable.
    """
    global _kernel_cache
    with _kernel_cache._lock:
        _kernel_cache._use_hashing = use_hashing


def print_cache():
    """Print cache entries in a nice table format."""
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title="Cute Kernel Cache")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Cache Key", style="cyan")

    with _kernel_cache._lock:
        if not _kernel_cache._cache:
            console.print("[yellow]Cache is empty[/yellow]")
            return

        for i, key in enumerate(_kernel_cache._cache.keys(), 1):
            table.add_row(str(i), key)

    console.print(table)

    # Print stats too
    stats = get_cache_stats()
    console.print(
        f"\n[bold]Cache Stats:[/bold] {stats['cache_size']}/{stats['max_size']} entries, "
        f"{stats['hits']} hits, {stats['misses']} misses, "
        f"{stats['hit_rate']:.1%} hit rate"
    )
