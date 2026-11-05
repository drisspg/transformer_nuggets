# jit_cache

A persistent compile cache for TVM-FFI CuTeDSL kernels, ported from Attention Gym
(`attn_gym/_backends/cute/cache.py` and `_key.py`) with the same keys and invalidation rules,
plus trace records and `uncached=` (below).
Decorate a module-level compile function; its arguments are static values that decide the
generated code:

```python
from transformer_nuggets.cute.jit_cache import jit_cache
from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor


@jit_cache
def compile_gemm(op: Gemm):
    m, n, k = cute.sym_int(), cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)
    bf16 = cutlass.BFloat16
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (m, k)),
        make_fake_compact_tensor(bf16, (n, k)),
        make_fake_compact_tensor(bf16, (m, n)),
    )


compile_gemm(Gemm())(a, b, out)  # compiles once; later calls and later processes load it
```

The first call compiles and exports the kernel as an object file; a second process with the
same key loads that file instead of compiling. Sizes that only feed bounds, grids or loop trip
counts should stay `cute.sym_int()` so every shape shares one entry.

## What the key covers

- The compile function's arguments, encoded structurally: plain values, tuples, dataclasses,
  named tuples, enums and functions (by source, closure and referenced globals). An object can
  define `__cute_cache_key__()` to name its own key; warpdrive kernels do (see
  [roles](../warpdrive/docs/roles.md#compile-keys)). Anything else must pickle.
- The compile target (device type, capability, `CUTE_DSL_ARCH`, SM count).
- Python, cutlass, tvm_ffi, torch and CUDA versions, a few cutlass compiler sources, and the
  codegen variables `CUTE_DSL_COMPILER_OPT`, `CUTE_DSL_ENABLE_ASSERTIONS`,
  `CUTE_DSL_ENABLE_TVM_FFI`, `CUTE_DSL_LIBS` and `CUTE_DSL_LINEINFO`.
- The source of the compile function's module and of every `transformer_nuggets` module it
  imports, directly or transitively (`module_closure`). Editing one of them recompiles;
  editing an unrelated module does not.

Imports are found statically, so keep the code a kernel traces reachable by import. Ordinary
`import`/`from` statements count, function-local ones included, and so does a module named as a
string literal (`importlib.import_module("transformer_nuggets.x")`). `import pkg.sub.mod` does
not add `pkg/sub/__init__.py`. Pass `extra_sources=(...)` only for what import analysis
cannot see: files read while tracing, or code outside `transformer_nuggets` (other than the
versioned dependencies above). A class passed as an argument is keyed by its module and name;
if the compile module does not import its module, list it in `extra_sources` or give the class
a `__cute_cache_key__` that includes `module_source_digest(cls.__module__)`.

## Trace results

A kernel loaded from disk was never traced, so anything the trace left on the Python
objects is missing. An argument that defines `__cute_trace_record__()` (a picklable record
of that state) and `__cute_restore_trace__(record)` gets the record of the build back on
every call served from memory or disk (`by_static_key` hits excepted: they reuse the compiled
arguments). warpdrive kernels use this for `describe()`, the TMA
and alias reports and their warnings. `jit_cache(uncached=predicate)` compiles the calls the
predicate selects without either cache, for when the trace itself is the point (warpdrive's
`trace_protocol`).

## Using it

- `fn.cache_info()`: process-local `hits`, `misses` and `currsize` (distinct keys). A disk load
  is a hit. Assert on these in tests rather than disabling the cache.
- `fn.cache_clear()`: drop this function's in-memory entries; the disk stays.
- `fn.by_static_key(key, lambda: fn(...))`: a hot path's launch cache keyed by your own
  hashable value, skipping the full key on a hit (and the trace-record restore).
- `compile_many(fn, [(args...), ...], workers=4)`: compile variants in forked workers of a fresh
  compiler process, then load them in the caller. Arguments must pickle.

## The disk cache

Entries live under `$TRANSFORMER_NUGGETS_CUTE_CACHE_DIR`, else
`$XDG_CACHE_HOME/transformer_nuggets/cute`, else `~/.cache/transformer_nuggets/cute`. Each
directory there is one source/version fingerprint; each entry is `<key>.o` (the exported
kernel), an optional `<key>.trace` (pickled trace records), and a `<key>.lock`. Entries are
published atomically under a file lock, so parallel processes (pytest-xdist workers) compile
each key once. Labels are package-relative, so checkouts with identical sources share entries.

Nothing is ever evicted. To clear it:

```bash
rm -rf "${TRANSFORMER_NUGGETS_CUTE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/transformer_nuggets/cute}"
```

`CUTE_DSL_NO_CACHE=1` turns the disk cache off (memory caching stays); it forces cold compiles,
so use it only where a test asserts on tracing itself.
