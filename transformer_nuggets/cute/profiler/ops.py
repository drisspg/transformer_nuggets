"""NVIDIA Intra-Kernel Profiling Device Operations (CUTE DSL).

This module provides device-side helpers for intra-kernel profiling on NVIDIA GPUs
(Hopper/Blackwell). These are CUTE `dsl_user_ops` that wrap inline PTX instructions
for reading the global timer and recording profiling events.

Static Allocation Buffer Layout (all int64 for alignment):
    For each unit u (0 <= u < num_units):
        buf[u * slice_size + 0] = event_count for unit u
        buf[u * slice_size + 1 + 4*i + 0] = start_ns
        buf[u * slice_size + 1 + 4*i + 1] = dur_ns
        buf[u * slice_size + 1 + 4*i + 2] = tag_id
        buf[u * slice_size + 1 + 4*i + 3] = tid

    slice_size = 1 + 4 * max_events_per_unit

Usage in kernels (using guarded helpers - recommended):
    unit_id = bidx
    event_idx = Int32(0)

    start_ns = warp_start(prof_buf, unit_id, event_idx, max_events_per_unit, Int32(0))
    warp_stop(prof_buf, unit_id, event_idx, start_ns, TAG_X, tid, max_events_per_unit, Int32(0))
    warp_flush(prof_buf, unit_id, event_idx, max_events_per_unit, Int32(0))

Low-level usage (manual guards):
    if warp_idx == 0 and lane_idx == 0:
        start_ns = static_start(prof_buf, unit_id, event_idx, max_events_per_unit)
        static_stop(prof_buf, unit_id, event_idx, start_ns, TAG_X, tid, max_events_per_unit)
        static_flush(prof_buf, unit_id, event_idx, max_events_per_unit)
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


__all__ = [
    "read_globaltimer",
    "static_start",
    "static_stop",
    "warp_atomic_alloc",
    "warp_start",
    "warp_stop",
    "profile_region",
    "ENABLE_PROFILING",
    "ENABLE_PROFILING_CONST",
]


def _env_flag(name: str, default: bool = True) -> bool:
    """Parse a boolean-like environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "off", "no", "")


# Global kill-switch for profiling generation. Evaluated at kernel staging time,
# so toggling the env var and re-staging the kernel removes all profiling code.
ENABLE_PROFILING: bool = _env_flag("TNUGGETS_ENABLE_PROFILING", default=True)
# Use cutlass.const_expr so the branch is compile-time in staged kernels.
# If const_expr is missing in the installed cutlass, fall back to plain bool.
ENABLE_PROFILING_CONST = getattr(cutlass, "const_expr", lambda x: x)(ENABLE_PROFILING)
_PROFILE_DISABLED_PRINTED = False


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the GPU global timer (nanoseconds on Hopper/Blackwell).

    Uses inline PTX: mov.u64 $0, %globaltimer;

    Note: %globaltimer is available on sm70+ and returns nanoseconds on recent
    NVIDIA architectures (Hopper/Blackwell). Unlike clock/clock64, globaltimer
    is synchronized across all SMs, making it suitable for comparing timings
    across different blocks.

    Returns:
        Int64: Current global timer value in nanoseconds.
    """
    result = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(result)


@dsl_user_op
def _store_i64(ptr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """Store an int64 value to a global memory address using cache-streaming store."""
    llvm.inline_asm(
        None,
        [ptr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.global.cs.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atomic_add_i64_ptr(ptr: Int64, val: Int64, *, loc=None, ip=None) -> Int64:
    """Atomically add val to the int64 at ptr and return the old value.

    Uses inline PTX atom.add.u64 instruction.
    """
    result = llvm.inline_asm(
        T.i64(),
        [ptr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "atom.add.u64 $0, [$1], $2;",
        "=l,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(result)


@cute.jit
def static_start(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    max_events_per_unit: Int32,
) -> Int64:
    """Read start timestamp for an event within a unit's static buffer slice.

    No atomics or memory traffic are performed here. The caller must pass the
    returned timestamp to static_stop.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is (e.g., block index, warp index).
        event_idx: Event index within this unit (0, 1, 2, ... tracked in registers).
        max_events_per_unit: Maximum events per unit (determines slice size).

    Returns:
        Int64: Start timestamp (0 when event_idx >= max_events_per_unit).
    """
    start_ns = Int64(0)
    if event_idx < max_events_per_unit:
        start_ns = read_globaltimer()
    return start_ns


@cute.jit
def static_stop(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    start_ns: Int64,
    tag: Int32,
    tid: Int32,
    max_events_per_unit: Int32,
) -> None:
    """Stop profiling an event, recording duration, tag, and tid.

    Computes duration from the provided start timestamp and writes event data.
    No atomics are used.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is (must match static_start).
        event_idx: Event index within this unit (must match static_start).
        start_ns: Start timestamp returned by static_start.
        tag: Tag ID for this event (from TagTable on host).
        tid: Thread/block identifier for Perfetto visualization.
        max_events_per_unit: Maximum events per unit (determines slice size).
    """
    if event_idx < max_events_per_unit:
        end_ns = read_globaltimer()

        base_ptr = buf.iterator.toint()

        slice_size = Int64(1 + 4 * max_events_per_unit)
        base_offset = Int64(unit_id) * slice_size + Int64(1 + 4 * event_idx)
        byte_offset = base_offset * Int64(8)

        dur_ns = end_ns - start_ns

        _store_i64(base_ptr + byte_offset, start_ns)
        _store_i64(base_ptr + byte_offset + Int64(8), dur_ns)
        _store_i64(base_ptr + byte_offset + Int64(16), Int64(tag))
        _store_i64(base_ptr + byte_offset + Int64(24), Int64(tid))


@cute.jit
def static_flush(
    buf: cute.Tensor,
    unit_id: Int32,
    count: Int32,
    max_events_per_unit: Int32,
) -> None:
    """Write the final event count for a unit (DEPRECATED).

    Note: This function is deprecated. The decode_events function now scans
    all slots, so explicit flushing is no longer required.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is.
        count: Number of events recorded by this unit.
        max_events_per_unit: Maximum events per unit (determines slice size).
    """
    base_ptr = buf.iterator.toint()

    slice_size = Int64(1 + 4 * max_events_per_unit)
    byte_offset = Int64(unit_id) * slice_size * Int64(8)

    _store_i64(base_ptr + byte_offset, Int64(count))


@cute.jit
def warp_atomic_alloc(
    buf: cute.Tensor,
    unit_id: Int32,
    max_events_per_unit: Int32,
    target_warp: Int32,
) -> Int32:
    """Atomically allocate event index, guarded to lane 0 of target_warp.

    Only lane 0 of target_warp actually allocates. Other threads get -1.
    This is the safe version to use in profile_region.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is.
        max_events_per_unit: Maximum events per unit.
        target_warp: Which warp should allocate.

    Returns:
        Int32: The allocated event index (or -1 for non-allocating threads).
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    event_idx = Int32(-1)
    if warp_idx == target_warp and lane_idx == 0:
        base_ptr = buf.iterator.toint()
        slice_size = Int64(1 + 4 * max_events_per_unit)
        counter_ptr = base_ptr + Int64(unit_id) * slice_size * Int64(8)
        old_count = _atomic_add_i64_ptr(counter_ptr, Int64(1))
        event_idx = Int32(old_count)
    return event_idx


@cute.jit
def warp_start(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    max_events_per_unit: Int32,
    target_warp: Int32,
) -> Int64:
    """Start profiling, but only for lane 0 of the specified warp.

    This is a convenience wrapper around static_start that automatically
    guards the call so only one thread per block/unit executes it.

    For simple per-block profiling, use target_warp=Int32(0).
    For warp-specialized kernels, use the appropriate warp index.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is (e.g., block index, or bidx * NUM_WARPS + warp_idx).
        event_idx: Event index within this unit (0, 1, 2, ... tracked in registers).
        max_events_per_unit: Maximum events per unit (determines slice size).
        target_warp: Which warp should profile (0, 1, 2, ...).

    Returns:
        Int64: Start timestamp for this event (0 for non-participating lanes).
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    start_ns = Int64(0)
    if warp_idx == target_warp and lane_idx == 0:
        start_ns = static_start(buf, unit_id, event_idx, max_events_per_unit)
    return start_ns


@cute.jit
def warp_stop(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    start_ns: Int64,
    tag: Int32,
    tid: Int32,
    max_events_per_unit: Int32,
    target_warp: Int32,
) -> None:
    """Stop profiling, but only for lane 0 of the specified warp.

    This is a convenience wrapper around static_stop that automatically
    guards the call so only one thread per block/unit executes it.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is (must match warp_start).
        event_idx: Event index within this unit (must match warp_start).
        start_ns: Start timestamp returned by warp_start.
        tag: Tag ID for this event (from TagTable on host).
        tid: Thread/block identifier for Perfetto visualization.
        max_events_per_unit: Maximum events per unit (determines slice size).
        target_warp: Which warp should profile (must match warp_start).
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    if warp_idx == target_warp and lane_idx == 0:
        static_stop(buf, unit_id, event_idx, start_ns, tag, tid, max_events_per_unit)


@cute.jit
def warp_flush(
    buf: cute.Tensor,
    unit_id: Int32,
    count: Int32,
    max_events_per_unit: Int32,
    target_warp: Int32,
) -> None:
    """Flush event count, but only for lane 0 of the specified warp.

    This is a convenience wrapper around static_flush that automatically
    guards the call so only one thread per block/unit executes it.

    Call this at the end of the kernel to record how many events were logged.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is.
        count: Number of events recorded by this unit.
        max_events_per_unit: Maximum events per unit (determines slice size).
        target_warp: Which warp should flush (0, 1, 2, ...).
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    if warp_idx == target_warp and lane_idx == 0:
        static_flush(buf, unit_id, count, max_events_per_unit)


class _ProfileRegionContext:
    """DSL-level context manager for profiling regions in CUTE kernels.

    This works at IR generation time, not runtime. When you use:
        with profile_region(prof_buf, max_events, tag, unit_id):
            compute_something()

    The __enter__ generates warp_start IR, __exit__ generates warp_stop IR.

    Two modes:
    - Atomic mode (event_idx=None): Allocates event index at runtime via atomic.
    - Static mode (event_idx provided): Uses the given index, no atomics.
    """

    def __init__(self, buf, max_events_per_unit, tag, unit_id, target_warp, event_idx):
        self._buf = buf
        self._max_events_per_unit = max_events_per_unit
        self._tag = tag
        self._unit_id = unit_id
        self._target_warp = target_warp
        self._event_idx = event_idx
        self._use_atomic = event_idx is None

    def __enter__(self):
        if cutlass.const_expr(self._use_atomic):
            self._event_idx = warp_atomic_alloc(
                self._buf, self._unit_id, self._max_events_per_unit, self._target_warp
            )

        self._start_ns = warp_start(
            self._buf,
            self._unit_id,
            self._event_idx,
            self._max_events_per_unit,
            self._target_warp,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        warp_stop(
            self._buf,
            self._unit_id,
            self._event_idx,
            self._start_ns,
            self._tag,
            self._unit_id,
            self._max_events_per_unit,
            self._target_warp,
        )
        return False


class _NoOpProfileRegion:
    """No-op context used when profiling is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def profile_region(buf, max_events_per_unit, tag, unit_id, target_warp=None, event_idx=None):
    """Create a context manager for profiling a code region in CUTE DSL kernels.

    This enables clean `with` statement syntax inside @cute.kernel functions.

    **Atomic mode (recommended for simplicity)**:

    When event_idx is omitted, event indices are allocated atomically at runtime.
    No manual index tracking or flushing needed:

        for i in cutlass.range(4):
            with profile_region(prof_buf, max_events, TAG_COMPUTE):
                compute_something()
            with profile_region(prof_buf, max_events, TAG_STORE):
                store_something()
        # Done! No flush needed.

    **Static mode (for maximum performance)**:

    When event_idx is provided, no atomics are used. Use runtime expressions
    with the loop variable to get unique indices:

        for i in cutlass.range(4):
            with profile_region(..., event_idx=Int32(0) + i * Int32(2)):
                ...
            with profile_region(..., event_idx=Int32(1) + i * Int32(2)):
                ...
        # No flush needed.

    Args:
        buf: Profile buffer tensor (cute.Tensor).
        max_events_per_unit: Maximum events per unit (Int32).
        tag: Tag ID for this region (Int32).
        unit_id: Which profiling unit (buffer slice) to use (Int32). Defaults to bidx.
        target_warp: Which warp should profile (Int32). Defaults to Int32(0).
        event_idx: Event index (Int32). If None, uses atomic allocation.

    Returns:
        Context manager for use with `with` statement.
    """
    if not cutlass.const_expr(bool(ENABLE_PROFILING_CONST)):
        global _PROFILE_DISABLED_PRINTED
        if not _PROFILE_DISABLED_PRINTED:
            print("profiling disabled: TNUGGETS_ENABLE_PROFILING=0, profile_region is no-op")
            _PROFILE_DISABLED_PRINTED = True
        return _NoOpProfileRegion()

    if target_warp is None:
        target_warp = Int32(0)

    return _ProfileRegionContext(buf, max_events_per_unit, tag, unit_id, target_warp, event_idx)
