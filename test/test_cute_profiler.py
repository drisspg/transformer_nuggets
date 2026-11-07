"""GPU tests for the CuTeDSL intra-kernel profiler.

Covers the three recording shapes (full 4xi64 records, compact 1xi64 gmem, compact 1xi64
smem) end-to-end: record events on the device, decode on the host, and assert the events are
well-formed.
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
    compact_event_stop,
    compact_event_stop_smem,
    compact_flush_smem_to_gmem,
    compact_prepare_smem,
    event_stop,
    profile_session,
    read_globaltimer,
    read_globaltimer_lo32,
)
from transformer_nuggets.cute.profiler.host import decode_events, events_to_perfetto


NUM_BLOCKS = 4
NUM_ITERS = 8
THREADS = 64


# ---------------------------------------------------------------------------
# Full records (event_stop), with and without payloads
# ---------------------------------------------------------------------------


@cute.kernel
def _kernel_full(prof_buf: cute.Tensor, max_events: cutlass.Int32):
    bidx, _, _ = cute.arch.block_idx()
    if cute.arch.warp_idx() == 1 and cute.arch.lane_idx() == 0:
        for i in cutlass.range(NUM_ITERS):
            start = read_globaltimer()
            # Alternate plain records with payloads, some negative.
            if i % 2 == 0:
                event_stop(prof_buf, bidx, Int32(i), start, Int32(1), Int32(1), max_events)
            else:
                payload = (Int32(i) - 4) * Int32(1000) + bidx
                event_stop(
                    prof_buf, bidx, Int32(i), start, Int32(2), Int32(1), max_events, payload
                )


@cute.jit()
def _launch_full(prof_buf, max_events):
    _kernel_full(prof_buf, max_events).launch(grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1))


def test_full_record_roundtrip(tmp_path):
    """Full records keep their slot, warp and tag; a payload decodes as Event.payload (and
    Perfetto args), and plain records in the same buffer keep payload None."""
    with profile_session(
        max_events_per_unit=NUM_ITERS,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=["unused", "plain", "item"],
        trace_path=str(tmp_path / "payload.json"),
        trace_format="chrome_json",
    ) as session:
        _launch_full(from_dlpack(session.prof.tensor), Int32(session.prof.max_events_per_unit))

    events = session.events
    assert len(events) == NUM_BLOCKS * NUM_ITERS
    for e in events:
        assert e.start_ns > 0 and e.tid == 1
        if e.event_idx % 2 == 0:
            assert (e.tag_name, e.payload) == ("plain", None)
        else:
            assert (e.tag_name, e.payload) == ("item", (e.event_idx - 4) * 1000 + e.unit_id)
    payloads = [
        ev["args"].get("payload") for ev in session.trace["traceEvents"] if ev.get("ph") == "X"
    ]
    assert sorted(p for p in payloads if p is not None) == sorted(
        e.payload for e in events if e.payload is not None
    )


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


def test_compact_gmem_roundtrip(tmp_path):
    """Compact events populate the session result and preserve their static slots."""
    trace_path = tmp_path / "compact_gmem.pftrace"
    with profile_session(
        max_events_per_unit=NUM_ITERS + 1,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=[f"t{i}" for i in range(NUM_ITERS)],
        compact=True,
        expected_events_per_unit=NUM_ITERS,
        trace_path=str(trace_path),
    ) as session:
        _launch_compact_gmem(
            from_dlpack(session.prof.tensor), Int32(session.prof.max_events_per_unit)
        )

    events = session.events
    assert session.trace is not None
    assert trace_path.exists()
    assert len(events) == NUM_BLOCKS * NUM_ITERS
    for unit_id in range(NUM_BLOCKS):
        assert {e.event_idx for e in events if e.unit_id == unit_id} == set(range(NUM_ITERS))
    for e in events:
        # Anchors put all events near the same 64-bit timer band.
        assert e.start_ns > (1 << 40), "start_ns should be a reconstructed 64-bit globaltimer"

    trace = events_to_perfetto([events[0]], split_overlaps=False)
    duration_event = next(event for event in trace["traceEvents"] if event.get("ph") == "X")
    assert duration_event["args"]["event_idx"] == events[0].event_idx


def test_expected_event_count_validation_rejects_missing_events():
    """profile_session validates raw decoded counts before post-processing."""
    with pytest.raises(RuntimeError, match=f"Expected {NUM_ITERS + 1} events"):
        with profile_session(
            max_events_per_unit=NUM_ITERS + 1,
            num_units=(NUM_BLOCKS, "Block"),
            tag_names=[f"t{i}" for i in range(NUM_ITERS)],
            compact=True,
            expected_events_per_unit=NUM_ITERS + 1,
        ) as session:
            _launch_compact_gmem(
                from_dlpack(session.prof.tensor), Int32(session.prof.max_events_per_unit)
            )


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
    compact_prepare_smem(smem_tensor, max_events)

    if cute.arch.warp_idx() == 0:
        for i in cutlass.range(NUM_ITERS):
            start = read_globaltimer_lo32()
            compact_event_stop_smem(smem_tensor, Int32(i), start, Int32(i & 0xF), max_events)

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
    ) as session:
        _launch_compact_smem(
            from_dlpack(session.prof.tensor), Int32(session.prof.max_events_per_unit)
        )

    events = session.events
    assert len(events) == NUM_BLOCKS * NUM_ITERS
    units_seen = {e.unit_id for e in events}
    assert units_seen == set(range(NUM_BLOCKS))
    tags_seen = {e.tag_id for e in events}
    assert tags_seen == set(range(NUM_ITERS))
    for unit_id in range(NUM_BLOCKS):
        assert {e.event_idx for e in events if e.unit_id == unit_id} == set(range(NUM_ITERS))
    for e in events:
        assert e.start_ns > (1 << 40)


# ---------------------------------------------------------------------------
# Out-of-range slots are dropped, not written over neighboring records
# ---------------------------------------------------------------------------


@cute.kernel
def _kernel_compact_slots(prof_buf: cute.Tensor, max_events: cutlass.Int32):
    bidx, _, _ = cute.arch.block_idx()
    compact_anchor_init(prof_buf, bidx, max_events)
    cute.arch.barrier()
    if cute.arch.warp_idx() == 0:
        # Block 0's slots -1 and max_events would overwrite its anchor and block 1's.
        for slot in (Int32(0) - bidx, max_events * bidx):
            start = read_globaltimer_lo32()
            compact_event_stop(prof_buf, Int32(1) - bidx, slot, start, Int32(0), max_events)


@cute.jit()
def _launch_compact_slots(prof_buf, max_events):
    _kernel_compact_slots(prof_buf, max_events).launch(grid=(2, 1, 1), block=(THREADS, 1, 1))


def test_compact_out_of_range_slots_are_dropped():
    with profile_session(
        max_events_per_unit=2, num_units=2, tag_names=["t"], compact=True
    ) as session:
        _launch_compact_slots(from_dlpack(session.prof.tensor), Int32(2))
    # Block 0 records slot 0 of unit 1 twice; block 1's slots -1 and 2 of unit 0 are dropped.
    assert [(e.unit_id, e.event_idx) for e in session.events] == [(1, 0)]
    anchors = session.prof.tensor.view(2, 3)[:, 0]
    assert (anchors > 1 << 50).all(), "an out-of-range record overwrote an anchor"


SENTINEL = 0x5E17


@cute.kernel
def _kernel_compact_smem_slot(prof_gmem: cute.Tensor, out: cute.Tensor):
    @cute.struct
    class S:
        prof_smem: cute.struct.MemRange[cutlass.Int64, 2]
        neighbor: cute.struct.MemRange[cutlass.Int64, 1]

    storage = cutlass.utils.SmemAllocator().allocate(S)
    smem_tensor = storage.prof_smem.get_tensor(cute.make_layout((2,)))
    neighbor = storage.neighbor.get_tensor(cute.make_layout((1,)))
    if cute.arch.thread_idx()[0] == 0:
        neighbor[0] = cutlass.Int64(SENTINEL)
    compact_prepare_smem(smem_tensor, Int32(1))
    if cute.arch.warp_idx() == 0:
        start = read_globaltimer_lo32()
        compact_event_stop_smem(smem_tensor, Int32(1), start, Int32(0), Int32(1))  # past the end
    compact_flush_smem_to_gmem(smem_tensor, prof_gmem, Int32(0), Int32(1))
    if cute.arch.thread_idx()[0] == 0:
        out[0] = neighbor[0]


@cute.jit()
def _launch_compact_smem_slot(prof_gmem, out):
    _kernel_compact_smem_slot(prof_gmem, out).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


def test_compact_smem_slot_past_the_slice_is_dropped():
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    with profile_session(
        max_events_per_unit=1, num_units=1, tag_names=["t"], compact=True
    ) as session:
        _launch_compact_smem_slot(from_dlpack(session.prof.tensor), from_dlpack(out))
    assert session.events == []
    assert out.item() == SENTINEL


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
