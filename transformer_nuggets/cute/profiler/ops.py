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
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


__all__ = [
    "read_globaltimer",
    "read_globaltimer_lo32",
    "static_start",
    "static_stop",
    "warp_atomic_alloc",
    "warp_start",
    "warp_stop",
    "profile_region",
    "RegionToken",
    "region_start",
    "region_end",
    "raw_event_stop",
    "compact_event_stop",
    "compact_anchor_init",
    "ENABLE_PROFILING",
    "ENABLE_PROFILING_CONST",
]


@cute.jit
def raw_event_stop(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    start_ns: Int64,
    tag: Int32,
    tid: Int32,
    max_events_per_unit: Int32,
) -> None:
    """Read the end timer and store an event record without a warp guard.

    Pair with :func:`read_globaltimer` for the start timestamp. Only lane 0 of
    the calling warp performs the stores; the caller is responsible for any
    outer warp election. The inner ``if lane_idx == 0:`` yields no value, so
    this primitive can be used in deeply nested control flow where
    :func:`profile_region` would trigger ``cf.br`` lowering failures.
    """
    lane_idx = cute.arch.lane_idx()
    end_ns = read_globaltimer()
    if lane_idx == 0:
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
def compact_anchor_init(
    buf: cute.Tensor,
    unit_id: Int32,
    max_events_per_unit: Int32,
) -> None:
    """Write a full 64-bit globaltimer anchor to slot 0 of this unit's buffer.

    Compact mode stores each event as a single packed int64 with only the low
    32 bits of the timer. The decoder reconstructs full 64-bit timestamps by
    combining each record's ts_lo32 with the upper 32 bits of this anchor (and
    handling wraparound). Call once per CTA at the top of ``@cute.kernel``,
    BEFORE any nested warp guards.

    Only lane 0 of warp 0 performs the store.
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    if warp_idx == 0 and lane_idx == 0:
        ts = read_globaltimer()
        base_ptr = buf.iterator.toint()
        slice_size = Int64(1 + max_events_per_unit)
        byte_offset = Int64(unit_id) * slice_size * Int64(8)
        _store_i64(base_ptr + byte_offset, ts)


@cute.jit
def compact_event_stop(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    start_ts32: Int32,
    tag: Int32,
    max_events_per_unit: Int32,
) -> None:
    """Record one event in compact format: single packed int64 store, no warp guard.

    Pair with :func:`read_globaltimer_lo32` for the start timestamp. The
    packed record layout is::

        bits  63..56  55..32     31..0
              tag(8)  dur_ns(24) ts_lo32(32)

    ``dur_ns`` saturates at ~16.7 ms (2^24 ns); regions longer than that
    overflow silently — use the legacy 4xi64 path (:func:`raw_event_stop`) for
    coarse outer regions if needed. ``tag`` is masked to 8 bits.

    Only lane 0 of the calling warp performs the store; the caller is
    responsible for any outer warp guard. The inner ``if lane_idx == 0:``
    yields no value, so this primitive is safe in deeply nested control flow.
    """
    lane_idx = cute.arch.lane_idx()
    end_ts32 = read_globaltimer_lo32()
    if lane_idx == 0:
        dur32 = end_ts32 - start_ts32
        ts_u = Int64(start_ts32) & Int64(0xFFFFFFFF)
        dur_u = Int64(dur32) & Int64(0xFFFFFF)
        tag_u = Int64(tag) & Int64(0xFF)
        packed = ts_u | (dur_u << 32) | (tag_u << 56)
        base_ptr = buf.iterator.toint()
        slice_size = Int64(1 + max_events_per_unit)
        byte_offset = (Int64(unit_id) * slice_size + Int64(1 + event_idx)) * Int64(8)
        _store_i64(base_ptr + byte_offset, packed)


class RegionToken(NamedTuple):
    """Pairing token returned by :func:`region_start` and consumed by :func:`region_end`.

    Captures everything :func:`region_end` needs to close the region (slot
    index, start timestamp, unit id, target warp), so callers cannot mismatch
    those fields between start and end. Implemented as ``NamedTuple`` so the
    DSL can thread it through ``scf.for`` as a loop-carried value.
    """

    unit_id: Int32
    event_idx: Int32
    start_ns: Int64
    target_warp: Int32


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
def read_globaltimer_lo32(*, loc=None, ip=None) -> Int32:
    """Read the low 32 bits of ``%globaltimer`` (PTX ``mov.u32 ... %globaltimer_lo;``).

    Lowers to a single ``CS2R.32 Rn, SR_GLOBALTIMERLO`` on Blackwell. One-third
    the register pressure of :func:`read_globaltimer` and avoids the 64-bit
    packing overhead, at the cost of ~4 s wraparound. The host decoder rebases
    against a per-unit 64-bit anchor (see :func:`compact_anchor_init`).

    Returns:
        Int32: Low 32 bits of the current global timer in nanoseconds.
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        "mov.u32 $0, %globaltimer_lo;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


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
    bounds_check: cutlass.Constexpr = True,
) -> Int64:
    """Read start timestamp for an event within a unit's static buffer slice.

    No atomics or memory traffic are performed here. The caller must pass the
    returned timestamp to static_stop.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is (e.g., block index, warp index).
        event_idx: Event index within this unit (0, 1, 2, ... tracked in registers).
        max_events_per_unit: Maximum events per unit (determines slice size).
        bounds_check: Compile-time flag. When ``True`` (default), gate the timer
            read on ``event_idx < max_events_per_unit``. Setting ``False`` drops
            the inner ``scf.if`` and is required when calling from deeply nested
            control flow (e.g. inside ``if warp_idx == 0: for ...:`` in a
            warp-specialized kernel) where the extra conditional has been seen
            to trip ``cf.br`` lowering or LLVM dominance verification.

    Returns:
        Int64: Start timestamp (0 when ``bounds_check`` is True and the slot
        is out of range).
    """
    if cutlass.const_expr(bounds_check):
        start_ns = Int64(0)
        if event_idx < max_events_per_unit:
            start_ns = read_globaltimer()
        return start_ns
    return read_globaltimer()


@cute.jit
def static_stop(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    start_ns: Int64,
    tag: Int32,
    tid: Int32,
    max_events_per_unit: Int32,
    bounds_check: cutlass.Constexpr = True,
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
        bounds_check: Forwarded to :func:`static_start`.
    """
    if cutlass.const_expr(bounds_check):
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
    else:
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
    bounds_check: cutlass.Constexpr = True,
) -> Int64:
    """Start profiling, but only for lane 0 of the specified warp.

    Convenience wrapper around :func:`static_start` that adds a
    ``warp_idx == target_warp and lane_idx == 0`` guard so only one thread
    records the event.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is.
        event_idx: Event index within this unit.
        max_events_per_unit: Maximum events per unit.
        target_warp: Which warp should profile.
        bounds_check: Forwarded to :func:`static_start`.

    Returns:
        Int64: Start timestamp (0 for non-participating lanes).
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    start_ns = Int64(0)
    if warp_idx == target_warp and lane_idx == 0:
        start_ns = static_start(
            buf, unit_id, event_idx, max_events_per_unit, bounds_check=bounds_check
        )
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
    bounds_check: cutlass.Constexpr = True,
) -> None:
    """Stop profiling, but only for lane 0 of the specified warp.

    Args:
        buf: Profile buffer tensor (int64).
        unit_id: Which profiling unit this is.
        event_idx: Event index within this unit.
        start_ns: Start timestamp returned by :func:`warp_start`.
        tag: Tag ID for this event.
        tid: Per-event tid.
        max_events_per_unit: Maximum events per unit.
        target_warp: Which warp should profile.
        bounds_check: Forwarded to :func:`static_stop`.
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    if warp_idx == target_warp and lane_idx == 0:
        static_stop(
            buf,
            unit_id,
            event_idx,
            start_ns,
            tag,
            tid,
            max_events_per_unit,
            bounds_check=bounds_check,
        )


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

    Generated at IR-staging time: ``__enter__`` emits the warp_start IR,
    ``__exit__`` emits the warp_stop IR. Pairing is structural (Python
    ``with``), so there is no way to start without ending. ``event_idx=None``
    selects atomic slot allocation; a caller-supplied ``event_idx`` selects
    static (no-atomic) allocation.
    """

    def __init__(
        self, buf, max_events_per_unit, tag, unit_id, tid, target_warp, event_idx, bounds_check
    ):
        self._buf = buf
        self._max_events_per_unit = max_events_per_unit
        self._tag = tag
        self._unit_id = unit_id
        self._tid = tid
        self._target_warp = target_warp
        self._event_idx = event_idx
        self._use_atomic = event_idx is None
        self._bounds_check = bounds_check

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
            bounds_check=self._bounds_check,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        warp_stop(
            self._buf,
            self._unit_id,
            self._event_idx,
            self._start_ns,
            self._tag,
            self._tid,
            self._max_events_per_unit,
            self._target_warp,
            bounds_check=self._bounds_check,
        )
        return False


class _NoOpProfileRegion:
    """No-op context used when profiling is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def profile_region(
    buf,
    max_events_per_unit,
    tag,
    unit_id,
    target_warp=None,
    event_idx=None,
    tid=None,
    bounds_check=True,
):
    """Context manager that profiles a code region in a CUTE DSL kernel.

    Atomic mode (default, ``event_idx=None``)::

        for i in cutlass.range(4):
            with profile_region(prof_buf, max_events, TAG_COMPUTE, bidx):
                compute_something()

    Static mode (``event_idx`` provided, no atomics)::

        for i in cutlass.range(4):
            with profile_region(..., event_idx=Int32(0) + i * Int32(2)):
                ...

    See :func:`region_start`/:func:`region_end` for explicit token pairing
    across Python scopes or loop iterations.

    Args:
        buf: Profile buffer tensor.
        max_events_per_unit: Slots reserved per unit (Int32).
        tag: Tag ID for the region (Int32).
        unit_id: Profiling unit (buffer slice) to write to (Int32).
        target_warp: Which warp records the event. Defaults to ``Int32(0)``.
        event_idx: Optional explicit slot index. ``None`` enables atomic mode.
        tid: Per-event ``tid`` written into the slot and surfaced as the
            Perfetto thread id. Defaults to ``unit_id`` for backward
            compatibility with the single-warp case. For warp-specialized
            kernels pass ``cute.arch.warp_idx()`` (or any per-warp value) so
            each warp gets its own Perfetto lane.
        bounds_check: Forwarded to :func:`static_start`/:func:`static_stop`.

    Returns:
        Context manager for use with ``with``.
    """
    if not cutlass.const_expr(bool(ENABLE_PROFILING_CONST)):
        global _PROFILE_DISABLED_PRINTED
        if not _PROFILE_DISABLED_PRINTED:
            print("profiling disabled: TNUGGETS_ENABLE_PROFILING=0, profile_region is no-op")
            _PROFILE_DISABLED_PRINTED = True
        return _NoOpProfileRegion()

    if target_warp is None:
        target_warp = Int32(0)
    if tid is None:
        tid = unit_id

    return _ProfileRegionContext(
        buf, max_events_per_unit, tag, unit_id, tid, target_warp, event_idx, bounds_check
    )


def region_start(
    buf: cute.Tensor,
    unit_id: Int32,
    max_events_per_unit: Int32,
    target_warp: Int32 | None = None,
    event_idx: Int32 | None = None,
) -> RegionToken:
    """Open a region and return a :class:`RegionToken` for explicit pairing.

    Use this when ``with profile_region(...)`` does not fit, e.g. when a region
    spans two Python scopes or needs to be loop-carried across
    ``cutlass.range`` iterations.

    Args:
        buf: Profile buffer tensor.
        unit_id: Profiling unit (buffer slice) to write to.
        max_events_per_unit: Slots reserved per unit.
        target_warp: Which warp records the event. Defaults to ``Int32(0)``.
        event_idx: Optional explicit slot index. ``None`` enables atomic
            allocation via :func:`warp_atomic_alloc`.

    Returns:
        Token to pass to :func:`region_end`.
    """
    if target_warp is None:
        target_warp = Int32(0)
    if event_idx is None:
        event_idx = warp_atomic_alloc(buf, unit_id, max_events_per_unit, target_warp)
    start_ns = warp_start(buf, unit_id, event_idx, max_events_per_unit, target_warp)
    return RegionToken(
        unit_id=unit_id, event_idx=event_idx, start_ns=start_ns, target_warp=target_warp
    )


def region_end(
    buf: cute.Tensor,
    tag: Int32,
    token: RegionToken,
    max_events_per_unit: Int32,
    tid: Int32 | None = None,
) -> None:
    """Close a region opened by :func:`region_start`.

    Slot index, start timestamp, unit id, and target warp all come from
    ``token``, so the start/end pair can't drift apart on those fields.

    Args:
        buf: Profile buffer tensor.
        tag: Tag ID for the region.
        token: Token returned by :func:`region_start`.
        max_events_per_unit: Slots reserved per unit.
        tid: Per-event ``tid``. Defaults to the token's ``unit_id``.
    """
    if tid is None:
        tid = token.unit_id
    warp_stop(
        buf,
        token.unit_id,
        token.event_idx,
        token.start_ns,
        tag,
        tid,
        max_events_per_unit,
        token.target_warp,
    )
