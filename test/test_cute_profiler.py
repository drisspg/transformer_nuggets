"""GPU tests for the CuTeDSL intra-kernel profiler.

Covers the four recording shapes (legacy 4xi64, compact 1xi64 gmem,
compact 1xi64 smem, token-paired) end-to-end: record events on the device,
decode on the host, and assert the events are well-formed.
"""

import pytest
import torch

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available", allow_module_level=True)
except ImportError:
    pytest.skip("CUTE not available", allow_module_level=True)

from transformer_nuggets.cute.profiler import (
    compact_anchor_init,
    compact_anchor_init_smem,
    compact_event_stop,
    compact_event_stop_smem,
    compact_flush_smem_to_gmem,
    profile_region,
    profile_session,
    raw_event_stop,
    read_globaltimer,
    read_globaltimer_lo32,
    region_end,
    region_start,
)
from transformer_nuggets.cute.profiler.host import decode_events


NUM_BLOCKS = 4
NUM_ITERS = 8
THREADS = 64


# ---------------------------------------------------------------------------
# Atomic legacy mode (profile_region)
# ---------------------------------------------------------------------------


@cute.kernel
def _kernel_atomic(prof_buf: cute.Tensor, max_events: cutlass.Int32):
    bidx, _, _ = cute.arch.block_idx()
    for i in cutlass.range(NUM_ITERS):
        with profile_region(prof_buf, max_events, Int32(i & 0xF), bidx):
            pass


@cute.jit()
def _launch_atomic(prof_buf, max_events):
    _kernel_atomic(prof_buf, max_events).launch(grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1))


def test_legacy_atomic_roundtrip():
    """profile_region (legacy, atomic-mode) records events that decode cleanly."""
    with profile_session(
        max_events_per_unit=NUM_ITERS + 4,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=[f"t{i}" for i in range(NUM_ITERS)],
    ) as (prof, tags):
        _launch_atomic(from_dlpack(prof.tensor), Int32(prof.max_events_per_unit))

    events = decode_events(prof, tags)
    assert len(events) == NUM_BLOCKS * NUM_ITERS
    units_seen = {e.unit_id for e in events}
    assert units_seen == set(range(NUM_BLOCKS))
    tags_seen = {e.tag_id for e in events}
    assert tags_seen == set(range(NUM_ITERS))
    for e in events:
        assert e.start_ns > 0
        assert e.dur_ns >= 0


# ---------------------------------------------------------------------------
# Legacy raw_event_stop (no warp guard; deeply-nested call site)
# ---------------------------------------------------------------------------


@cute.kernel
def _kernel_raw_legacy(prof_buf: cute.Tensor, max_events: cutlass.Int32):
    bidx, _, _ = cute.arch.block_idx()
    if cute.arch.warp_idx() == 0:
        for i in cutlass.range(NUM_ITERS):
            start = read_globaltimer()
            raw_event_stop(prof_buf, bidx, Int32(i), start, Int32(i & 0xF), bidx, max_events)


@cute.jit()
def _launch_raw_legacy(prof_buf, max_events):
    _kernel_raw_legacy(prof_buf, max_events).launch(grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1))


def test_legacy_raw_event_stop_roundtrip():
    """raw_event_stop records events even from inside a warp guard."""
    with profile_session(
        max_events_per_unit=NUM_ITERS,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=[f"t{i}" for i in range(NUM_ITERS)],
    ) as (prof, tags):
        _launch_raw_legacy(from_dlpack(prof.tensor), Int32(prof.max_events_per_unit))

    events = decode_events(prof, tags)
    assert len(events) == NUM_BLOCKS * NUM_ITERS
    for e in events:
        assert e.start_ns > 0


# ---------------------------------------------------------------------------
# Token-paired mode
# ---------------------------------------------------------------------------


@cute.kernel
def _kernel_token(prof_buf: cute.Tensor, max_events: cutlass.Int32):
    bidx, _, _ = cute.arch.block_idx()
    outer = region_start(prof_buf, bidx, max_events)
    for i in cutlass.range(NUM_ITERS):
        inner = region_start(prof_buf, bidx, max_events)
        region_end(prof_buf, Int32(i & 0xF), inner, max_events)
    region_end(prof_buf, Int32(NUM_ITERS), outer, max_events)


@cute.jit()
def _launch_token(prof_buf, max_events):
    _kernel_token(prof_buf, max_events).launch(grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1))


def test_token_pairing_roundtrip():
    """region_start/region_end token pairing records nested events."""
    with profile_session(
        max_events_per_unit=NUM_ITERS + 4,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=[f"t{i}" for i in range(NUM_ITERS + 1)],
    ) as (prof, tags):
        _launch_token(from_dlpack(prof.tensor), Int32(prof.max_events_per_unit))

    events = decode_events(prof, tags)
    # NUM_ITERS inner + 1 outer per block.
    assert len(events) == NUM_BLOCKS * (NUM_ITERS + 1)
    outer_events = [e for e in events if e.tag_id == NUM_ITERS]
    inner_events = [e for e in events if e.tag_id < NUM_ITERS]
    assert len(outer_events) == NUM_BLOCKS
    assert len(inner_events) == NUM_BLOCKS * NUM_ITERS
    # Outer should be the slowest per block (it wraps everything inner).
    for outer in outer_events:
        same_block_inner = [e for e in inner_events if e.unit_id == outer.unit_id]
        assert outer.dur_ns >= max(e.dur_ns for e in same_block_inner)


# ---------------------------------------------------------------------------
# Compact 1xi64 gmem mode
# ---------------------------------------------------------------------------


@cute.kernel
def _kernel_compact_gmem(prof_buf: cute.Tensor, max_events: cutlass.Int32):
    bidx, _, _ = cute.arch.block_idx()
    compact_anchor_init(prof_buf, bidx, max_events)
    if cute.arch.warp_idx() == 0:
        for i in cutlass.range(NUM_ITERS):
            start = read_globaltimer_lo32()
            compact_event_stop(prof_buf, bidx, Int32(i), start, Int32(i & 0xF), max_events)


@cute.jit()
def _launch_compact_gmem(prof_buf, max_events):
    _kernel_compact_gmem(prof_buf, max_events).launch(
        grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1)
    )


def test_compact_gmem_roundtrip():
    """compact_event_stop records events that decode with reconstructed 64-bit timestamps."""
    with profile_session(
        max_events_per_unit=NUM_ITERS + 1,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=[f"t{i}" for i in range(NUM_ITERS)],
        compact=True,
    ) as (prof, tags):
        _launch_compact_gmem(from_dlpack(prof.tensor), Int32(prof.max_events_per_unit))

    events = decode_events(prof, tags)
    assert len(events) == NUM_BLOCKS * NUM_ITERS
    for e in events:
        # Anchors put all events near the same 64-bit timer band.
        assert e.start_ns > (1 << 40), "start_ns should be a reconstructed 64-bit globaltimer"


# ---------------------------------------------------------------------------
# Compact 1xi64 smem mode (the cheap-hot-path variant)
# ---------------------------------------------------------------------------


PROF_SMEM_SLOTS = NUM_ITERS + 1


@cute.kernel
def _kernel_compact_smem(prof_gmem: cute.Tensor, max_events: cutlass.Int32):
    @cute.struct
    class S:
        prof_smem: cute.struct.MemRange[cutlass.Int64, PROF_SMEM_SLOTS]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(S)
    smem_tensor = storage.prof_smem.get_tensor(cute.make_layout((PROF_SMEM_SLOTS,)))

    bidx, _, _ = cute.arch.block_idx()
    compact_anchor_init_smem(smem_tensor)

    if cute.arch.warp_idx() == 0:
        for i in cutlass.range(NUM_ITERS):
            start = read_globaltimer_lo32()
            compact_event_stop_smem(smem_tensor, Int32(i), start, Int32(i & 0xF))

    compact_flush_smem_to_gmem(smem_tensor, prof_gmem, bidx, max_events)


@cute.jit()
def _launch_compact_smem(prof_gmem, max_events):
    _kernel_compact_smem(prof_gmem, max_events).launch(
        grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1)
    )


def test_compact_smem_roundtrip():
    """Smem-staged compact path flushes to gmem and decodes identically to the gmem path."""
    with profile_session(
        max_events_per_unit=NUM_ITERS,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=[f"t{i}" for i in range(NUM_ITERS)],
        compact=True,
    ) as (prof, tags):
        _launch_compact_smem(from_dlpack(prof.tensor), Int32(prof.max_events_per_unit))

    events = decode_events(prof, tags)
    assert len(events) == NUM_BLOCKS * NUM_ITERS
    units_seen = {e.unit_id for e in events}
    assert units_seen == set(range(NUM_BLOCKS))
    tags_seen = {e.tag_id for e in events}
    assert tags_seen == set(range(NUM_ITERS))
    for e in events:
        assert e.start_ns > (1 << 40)


# ---------------------------------------------------------------------------
# Decoder overflow detection
# ---------------------------------------------------------------------------


def test_compact_decode_detects_event_idx_overflow():
    """If a caller's event_idx overflows past max_events_per_unit, the decoder yells."""
    # Hand-craft a buffer that mimics overflow: unit 0 anchor is sane, unit 1 anchor
    # was clobbered by a stray event store (so its value is wildly different).
    from transformer_nuggets.cute.profiler.host import ProfileBuf

    max_events = 4
    slice_size = 1 + max_events
    num_units = 2
    flat = torch.zeros(num_units * slice_size, dtype=torch.int64, device="cuda")
    # Unit 0: real anchor + one event.
    flat[0] = 1_000_000_000_000_000_000  # plausible ns timestamp
    flat[1] = 1 | (10 << 32) | (3 << 56)  # ts_lo=1, dur=10, tag=3
    # Unit 1: anchor slot clobbered by a packed record (low bits != a real timer hi).
    flat[slice_size + 0] = 5 | (20 << 32) | (4 << 56)  # looks like an event

    from transformer_nuggets.cute.profiler.host import TagTable

    tags = TagTable([f"t{i}" for i in range(8)])
    buf = ProfileBuf(
        tensor=flat,
        max_events_per_unit=max_events,
        num_units=num_units,
        unit_name="Block",
        compact=True,
    )
    with pytest.raises(RuntimeError, match="diverge from the median"):
        decode_events(buf, tags)
