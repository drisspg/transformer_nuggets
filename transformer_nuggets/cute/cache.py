import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any
from collections.abc import Callable

import cutlass.cute as cute


logger = logging.getLogger(__name__)


class CuteKernelCache:
    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._lock = threading.Lock()

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
            }


_kernel_cache = CuteKernelCache()


def _generate_cache_key(*args, **kwargs) -> str:
    key_parts = []

    for arg in args:
        if isinstance(arg, cute.Tensor):
            key_parts.append(f"tensor_shape={arg.shape}_dtype={arg._dtype}")
        elif hasattr(arg, "__name__"):
            key_parts.append(f"op={arg.__name__}")
        else:
            key_parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    key_str = "_".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def cute_compile_and_cache(func: Callable, *args, **kwargs):
    """
    Compile a @cute.jit decorated function and cache the result.

    Args:
        func: A function decorated with @cute.jit
        *args: Arguments to pass to cute.compile and for cache key generation
        **kwargs: Keyword arguments to pass to cute.compile and for cache key generation

    Returns:
        Compiled kernel that can be executed

    Example:
        @cute.jit
        def my_kernel(a, b, c):
            # kernel implementation
            ...

        # Cache the compilation
        compiled_kernel = cute_compile_and_cache(my_kernel, tensor_a, tensor_b, tensor_c)
        result = compiled_kernel(tensor_a, tensor_b, tensor_c)
    """
    # Generate cache key from function and arguments
    cache_key = _generate_cache_key(*args, **kwargs)
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
