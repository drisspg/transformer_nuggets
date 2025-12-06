"""NVIDIA Intra-Kernel Profiling Device Operations (CUTE DSL).

This module provides device-side helpers for intra-kernel profiling on NVIDIA GPUs
(Hopper/Blackwell). These are CUTE `dsl_user_ops` that wrap inline PTX instructions
for reading the global timer and recording profiling events.

Buffer Layout (all int64 for alignment):
    buf[0] = event_count (int64)
    For each event i (0 <= i < max_events):
        buf[1 + 4*i + 0] = start_ns (int64)
        buf[1 + 4*i + 1] = dur_ns (int64)
        buf[1 + 4*i + 2] = tag_id (int64)
        buf[1 + 4*i + 3] = tid (int64)

Usage in kernels:
    Guard profiling with constexpr and restrict to warp0/lane0 to minimize contention:

    if const_expr(PROFILE):
        if cute.arch.warp_idx() == 0 and cute.arch.lane_idx() == 0:
            eid = profile_start(buf, max_events)
    # ... code region ...
    if const_expr(PROFILE):
        if cute.arch.warp_idx() == 0 and cute.arch.lane_idx() == 0:
            profile_stop(buf, eid, TAG_X, blockIdx.x * WARPS_PER_BLOCK + warp_id, max_events)

Or use the convenience helpers:
    eid = lane0_warp0_start(buf, max_events)
    # ... code region ...
    lane0_warp0_stop(buf, eid, TAG_X, tid, max_events)
"""

import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


__all__ = [
    "read_globaltimer",
    "profile_start",
    "profile_stop",
    # Convenience wrappers
    "lane0_warp0_start",
    "lane0_warp0_stop",
    # Configurable wrappers
    "elected_start",
    "elected_stop",
    "warp_start",
    "warp_stop",
    # Helper class
    "ProfileRegion",
    # DSL context manager (use with `with` statement in kernels)
    "profile_region",
]


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the GPU global timer (nanoseconds on Hopper/Blackwell).

    Uses inline PTX: mov.u64 $0, %globaltimer;

    Note: %globaltimer is available on sm70+ and returns nanoseconds on recent
    NVIDIA architectures (Hopper/Blackwell). For older architectures, consider
    using clock64() scaled by host-provided ns_per_cycle.

    Returns:
        Int64: Current global timer value in nanoseconds.
    """
    result = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(result)


@dsl_user_op
def _atomic_add_i64_ptr(ptr: Int64, val: Int64, *, loc=None, ip=None) -> Int64:
    """Atomically add val to the int64 at ptr (as raw address) and return the old value.

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


@dsl_user_op
def _store_i64(ptr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """Store an int64 value to a global memory address.

    Uses inline PTX st.global.u64 instruction.
    """
    llvm.inline_asm(
        None,
        [ptr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_i64(ptr: Int64, *, loc=None, ip=None) -> Int64:
    """Load an int64 value from a global memory address.

    Uses inline PTX ld.global.u64 instruction.
    """
    result = llvm.inline_asm(
        T.i64(),
        [ptr.ir_value(loc=loc, ip=ip)],
        "ld.global.u64 $0, [$1];",
        "=l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(result)


@cute.jit
def profile_start(buf: cute.Tensor, max_events: Int32) -> Int32:
    """Start profiling an event, returning the event ID.

    Atomically increments buf[0] (event count) and stores the start timestamp
    at buf[1 + 4*eid] if eid < max_events.

    Args:
        buf: Profile buffer tensor (int64, length 1 + 4*max_events).
        max_events: Maximum number of events the buffer can hold.

    Returns:
        Int32: Event ID (eid). May be >= max_events if buffer is full.
               Caller should pass this to profile_stop.
    """
    # Get base pointer as int64 (raw address)
    base_ptr = buf.iterator.toint()

    # Atomically increment count at buf[0] and get old value as eid
    eid_i64 = _atomic_add_i64_ptr(base_ptr, Int64(1))
    eid = Int32(eid_i64)

    # If within bounds, store start timestamp at buf[1 + 4*eid]
    if eid < max_events:
        start_ns = read_globaltimer()
        # Each element is 8 bytes (int64)
        offset = Int64(1 + 4 * eid) * Int64(8)
        _store_i64(base_ptr + offset, start_ns)

    return eid


@cute.jit
def profile_stop(
    buf: cute.Tensor,
    eid: Int32,
    tag: Int32,
    tid: Int32,
    max_events: Int32,
) -> None:
    """Stop profiling an event, recording duration, tag, and tid.

    If eid < max_events, computes duration and writes:
        buf[1 + 4*eid + 1] = dur_ns (end - start)
        buf[1 + 4*eid + 2] = tag_id
        buf[1 + 4*eid + 3] = tid

    Args:
        buf: Profile buffer tensor (int64, length 1 + 4*max_events).
        eid: Event ID returned by profile_start.
        tag: Tag ID for this event (from TagTable on host).
        tid: Thread/block identifier (e.g., blockIdx.x * warps_per_block + warp_id).
        max_events: Maximum number of events the buffer can hold.
    """
    if eid < max_events:
        end_ns = read_globaltimer()

        # Get base pointer as int64 (raw address)
        base_ptr = buf.iterator.toint()

        # Base offset for this event: buf[1 + 4*eid]
        base_offset = Int64(1 + 4 * eid) * Int64(8)

        # Read start timestamp
        start_ns = _load_i64(base_ptr + base_offset)
        dur_ns = end_ns - start_ns

        # Write dur_ns at buf[1 + 4*eid + 1]
        _store_i64(base_ptr + base_offset + Int64(8), dur_ns)
        # Write tag at buf[1 + 4*eid + 2]
        _store_i64(base_ptr + base_offset + Int64(16), Int64(tag))
        # Write tid at buf[1 + 4*eid + 3]
        _store_i64(base_ptr + base_offset + Int64(24), Int64(tid))


@cute.jit
def lane0_warp0_start(buf: cute.Tensor, max_events: Int32) -> Int32:
    """Start profiling, but only for warp 0, lane 0.

    Returns -1 for non-participating threads (other warps/lanes).
    This reduces contention by limiting profiling to one lane per block.

    Args:
        buf: Profile buffer tensor (int64, length 1 + 4*max_events).
        max_events: Maximum number of events the buffer can hold.

    Returns:
        Int32: Event ID if warp0/lane0, else -1.
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    eid = Int32(-1)
    if warp_idx == 0 and lane_idx == 0:
        eid = profile_start(buf, max_events)

    return eid


@cute.jit
def lane0_warp0_stop(
    buf: cute.Tensor,
    eid: Int32,
    tag: Int32,
    tid: Int32,
    max_events: Int32,
) -> None:
    """Stop profiling, but only for warp 0, lane 0 with valid eid.

    Args:
        buf: Profile buffer tensor (int64, length 1 + 4*max_events).
        eid: Event ID returned by lane0_warp0_start (-1 means skip).
        tag: Tag ID for this event (from TagTable on host).
        tid: Thread/block identifier (e.g., blockIdx.x * warps_per_block + warp_id).
        max_events: Maximum number of events the buffer can hold.
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    if warp_idx == 0 and lane_idx == 0 and eid >= Int32(0):
        profile_stop(buf, eid, tag, tid, max_events)


# =============================================================================
# Configurable profiling helpers
# =============================================================================


@cute.jit
def elected_start(buf: cute.Tensor, max_events: Int32) -> Int32:
    """Start profiling using elect_one (one lane per warp).

    Uses cute.arch.elect_one() to select exactly one thread per warp.
    This is useful when you want to profile from all warps but only
    one lane per warp to minimize overhead.

    Args:
        buf: Profile buffer tensor (int64, length 1 + 4*max_events).
        max_events: Maximum number of events the buffer can hold.

    Returns:
        Int32: Event ID if elected, else -1.
    """
    eid = Int32(-1)
    with cute.arch.elect_one():
        eid = profile_start(buf, max_events)
    return eid


@cute.jit
def elected_stop(
    buf: cute.Tensor,
    eid: Int32,
    tag: Int32,
    tid: Int32,
    max_events: Int32,
) -> None:
    """Stop profiling using elect_one (one lane per warp).

    Args:
        buf: Profile buffer tensor (int64, length 1 + 4*max_events).
        eid: Event ID returned by elected_start (-1 means skip).
        tag: Tag ID for this event (from TagTable on host).
        tid: Thread/block identifier.
        max_events: Maximum number of events the buffer can hold.
    """
    if eid >= Int32(0):
        with cute.arch.elect_one():
            profile_stop(buf, eid, tag, tid, max_events)


@cute.jit
def warp_start(
    buf: cute.Tensor,
    max_events: Int32,
    target_warp: Int32,
) -> Int32:
    """Start profiling only from a specific warp (lane 0).

    Useful for profiling warp-specialized kernels where different warps
    have different roles (e.g., warp 0 = TMA producer, warp 1 = consumer).

    Args:
        buf: Profile buffer tensor (int64, length 1 + 4*max_events).
        max_events: Maximum number of events the buffer can hold.
        target_warp: Which warp should profile (0, 1, 2, ...).

    Returns:
        Int32: Event ID if this is target_warp lane 0, else -1.
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    eid = Int32(-1)
    if warp_idx == target_warp and lane_idx == 0:
        eid = profile_start(buf, max_events)
    return eid


@cute.jit
def warp_stop(
    buf: cute.Tensor,
    eid: Int32,
    tag: Int32,
    tid: Int32,
    max_events: Int32,
    target_warp: Int32,
) -> None:
    """Stop profiling only from a specific warp (lane 0).

    Args:
        buf: Profile buffer tensor (int64, length 1 + 4*max_events).
        eid: Event ID returned by warp_start (-1 means skip).
        tag: Tag ID for this event (from TagTable on host).
        tid: Thread/block identifier.
        max_events: Maximum number of events the buffer can hold.
        target_warp: Which warp should profile (must match warp_start).
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    if warp_idx == target_warp and lane_idx == 0 and eid >= Int32(0):
        profile_stop(buf, eid, tag, tid, max_events)


# =============================================================================
# Paired start/stop helper (cleaner than separate calls)
# =============================================================================


class ProfileRegion:
    """Helper for pairing profile start/stop calls with pre-configured parameters.

    This reduces boilerplate by pre-binding the buffer, max_events, tag, and tid.
    Use the returned start/stop functions inside the kernel.

    Example in @cute.jit or @cute.kernel:
        # Create region config (can be done outside kernel if params are known)
        compute_region = ProfileRegion(prof_buf, max_events, TAG_COMPUTE, bidx)

        # Use in kernel - much cleaner than passing all args twice
        eid = compute_region.start()
        # ... code to profile ...
        compute_region.stop(eid)

    For warp-specific profiling:
        producer_region = ProfileRegion(prof_buf, max_events, TAG_PRODUCER, bidx, target_warp=Int32(0))
        eid = producer_region.start()
        # ... producer code (only warp 0 profiles) ...
        producer_region.stop(eid)

    Note: This is a Python-level helper that generates the appropriate CUTE DSL
    calls. True 'with' statement context managers don't work inside kernels
    because the code is compiled to GPU IR.
    """

    def __init__(
        self,
        buf,
        max_events,
        tag,
        tid,
        target_warp=None,
    ):
        """Initialize a profile region with pre-bound parameters.

        Args:
            buf: Profile buffer tensor (cute.Tensor).
            max_events: Maximum events the buffer can hold (Int32).
            tag: Tag ID for this region (Int32).
            tid: Thread/block identifier for this region (Int32).
            target_warp: If set, only this warp profiles (Int32). If None, uses warp 0.
        """
        self._buf = buf
        self._max_events = max_events
        self._tag = tag
        self._tid = tid
        self._target_warp = target_warp

    def start(self) -> Int32:
        """Start profiling. Call this at region entry. Returns eid."""
        if self._target_warp is not None:
            return warp_start(self._buf, self._max_events, self._target_warp)
        else:
            return lane0_warp0_start(self._buf, self._max_events)

    def stop(self, eid: Int32) -> None:
        """Stop profiling. Call this at region exit with eid from start()."""
        if self._target_warp is not None:
            warp_stop(self._buf, eid, self._tag, self._tid, self._max_events, self._target_warp)
        else:
            lane0_warp0_stop(self._buf, eid, self._tag, self._tid, self._max_events)


# =============================================================================
# DSL-level context manager (works with `with` statement in kernels)
# =============================================================================


class _ProfileRegionContext:
    """DSL-level context manager for profiling regions in CUTE kernels.

    This works at IR generation time, not runtime. When you use:
        with profile_region(buf, max_events, tag, tid):
            # profiled code

    The __enter__ generates profile_start IR, __exit__ generates profile_stop IR.
    """

    def __init__(self, buf, max_events, tag, tid, target_warp=None):
        self._buf = buf
        self._max_events = max_events
        self._tag = tag
        self._tid = tid
        self._target_warp = target_warp
        self._eid = None

    def __enter__(self):
        """Generate profile_start IR and capture eid."""
        if self._target_warp is not None:
            self._eid = warp_start(self._buf, self._max_events, self._target_warp)
        else:
            self._eid = lane0_warp0_start(self._buf, self._max_events)
        return self._eid

    def __exit__(self, exc_type, exc_value, traceback):
        """Generate profile_stop IR with the captured eid."""
        if self._eid is not None:
            if self._target_warp is not None:
                warp_stop(
                    self._buf, self._eid, self._tag, self._tid, self._max_events, self._target_warp
                )
            else:
                lane0_warp0_stop(self._buf, self._eid, self._tag, self._tid, self._max_events)
        return False  # Don't suppress exceptions


def profile_region(buf, max_events, tag, tid, target_warp=None):
    """Create a context manager for profiling a code region in CUTE DSL kernels.

    This enables clean `with` statement syntax inside @cute.kernel functions:

        with profile_region(prof_buf, max_events, Int32(TAG_COMPUTE), bidx):
            # code to profile - only warp 0, lane 0 records timing
            compute_something()

    For warp-specific profiling:

        with profile_region(prof_buf, max_events, Int32(TAG_PRODUCER), bidx, target_warp=Int32(0)):
            # only warp 0 profiles this region
            producer_work()

    The context manager:
    - On enter: calls profile_start (or warp_start) and captures eid
    - On exit: calls profile_stop (or warp_stop) with the captured eid

    ⚠️ KNOWN LIMITATION: Variables defined outside the `with` block are NOT visible
    inside due to CUTE DSL's IR generation model. Use explicit start/stop if you
    need to access outer variables, or define all variables inside the `with` block.

        # ❌ Does NOT work:
        val = output[idx]
        with profile_region(...):
            output[idx] = val + 1.0  # 'val' not visible!

        # ✅ Works:
        with profile_region(...):
            val = output[idx]  # define inside
            output[idx] = val + 1.0

    Args:
        buf: Profile buffer tensor (cute.Tensor).
        max_events: Maximum events the buffer can hold (Int32).
        tag: Tag ID for this region (Int32).
        tid: Thread/block identifier for this region (Int32).
        target_warp: If set, only this warp profiles (Int32). If None, uses warp 0.

    Returns:
        Context manager that yields the eid (can be ignored).
    """
    return _ProfileRegionContext(buf, max_events, tag, tid, target_warp)
