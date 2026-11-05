"""Persistent in-memory and on-disk cache for TVM-FFI CuTeDSL compilations.

See ``cache.jit_cache`` for the contract and ``_key`` for what the key covers.
"""

from transformer_nuggets.cute.jit_cache._key import (
    canonical_key,
    function_cache_key,
    module_closure,
    module_source_digest,
)
from transformer_nuggets.cute.jit_cache.cache import (
    CacheInfo,
    CacheLockError,
    cache_enabled,
    get_cache_path,
    jit_cache,
)
from transformer_nuggets.cute.jit_cache.compile import compile_many, precompile_many
from transformer_nuggets.cute.jit_cache.target import (
    CompileTarget,
    detect_compile_target,
    get_compile_target,
    set_compile_target,
)

__all__ = [
    "CacheInfo",
    "CacheLockError",
    "CompileTarget",
    "cache_enabled",
    "canonical_key",
    "compile_many",
    "detect_compile_target",
    "function_cache_key",
    "get_cache_path",
    "get_compile_target",
    "jit_cache",
    "module_closure",
    "module_source_digest",
    "precompile_many",
    "set_compile_target",
]
