"""Declared named barriers (``NamedBarrier``): ID allocation, collision and count checks, and a
runtime toy of a one-shot flag between roles and a sync inside one role."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from transformer_nuggets.cute.jit_cache import jit_cache
from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    Consumer,
    CtaPair,
    NamedBarrier,
    Producer,
    ProtocolError,
    Ring,
    Threads,
    WarpSpecialized,
    named_barrier,
    role,
)


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("warpdrive tests require SM100-family GPUs")


def _raises_protocol(match: str, fn):
    with pytest.raises(Exception, match=match) as info:
        fn()
    assert isinstance(info.value, ProtocolError) or "ProtocolError" in repr(info.value)


class FlagToy(WarpSpecialized):
    """publisher (warp 0, raw) -> done -> waiters (warps 1-2), then pair_sync inside the waiters.

    The publisher stalls for ``delay`` cycles, writes ``value + lane`` to SMEM and arrives on
    the one-shot ``done`` flag; the waiters sync on it, so they read the value rather than the
    sentinel. Waiter warp 0 then stalls and doubles the value into a second SMEM row before the
    waiters' ``pair_sync``; waiter warp 1 reads that row after it. ``mode`` injects errors."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.done = NamedBarrier(threads=96)
        self.pair_sync = NamedBarrier(threads=96 if mode == "short_users" else 64)
        if mode == "foreign":
            other = FlagToy()
            other.describe()  # allocates other.done the same ID as self.done
            self.borrowed = (other.done,)

    @role(warps=0, raw=True)
    @cute.jit
    def publisher(self, ctx):
        lane = cute.arch.lane_idx()
        start = cute.arch.clock64()
        while cute.arch.clock64() - start < ctx.delay:
            pass
        ctx.smem[0, lane] = ctx.value + lane
        self.done.arrive()
        if cutlass.const_expr(self.mode == "collision"):
            named_barrier(self.done.id, 96)
        if cutlass.const_expr(self.mode == "out_of_range"):
            named_barrier(16, 32)
        if cutlass.const_expr(self.mode == "foreign"):
            self.borrowed[0].sync()

    @role(warps=range(1, 3))
    @cute.jit
    def waiters(self, ctx):
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 1
        self.done.sync()
        value = ctx.smem[0, lane]
        if warp == 0:
            start = cute.arch.clock64()
            while cute.arch.clock64() - start < ctx.delay:
                pass
            ctx.smem[1, lane] = value * 2
        self.pair_sync.sync()
        if warp == 1:
            ctx.out[lane, 0] = value
            ctx.out[lane, 1] = ctx.smem[1, lane]

    @cute.kernel
    def kernel(self, out, value: cutlass.Int32, delay: cutlass.Int64):
        smem = self.begin_roles()
        buf = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2, 32), stride=(32, 1)), 16)
        tidx = cute.arch.thread_idx()[0]
        if tidx < 64:
            buf[tidx // 32, tidx % 32] = -1  # run_roles' CTA sync publishes the sentinel
        self.run_roles(SimpleNamespace(out=out, smem=buf, value=value, delay=delay))

    @cute.jit
    def __call__(self, out: cute.Tensor, value: cutlass.Int32, delay: cutlass.Int64, stream):
        self.kernel.set_name_prefix(f"warpdrive_named_barrier_toy_{self.mode}")
        self.kernel(out, value, delay).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_flag_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (32, 2), assumed_align=4)
    return compile_tvm_ffi(
        op, out, cutlass.Int32(0), cutlass.Int64(0), name=f"named_barrier_toy_{op.mode}"
    )


@pytest.mark.parametrize("value, delay", [(7, 0), (1000, 20000)])
def test_one_shot_flag_and_intra_role_sync(requires_blackwell, value, delay):
    """The waiters see the publisher's write through the one-shot flag, and waiter warp 1 sees
    warp 0's write through the intra-role sync, even when the writers stall."""
    out = torch.full((32, 2), -5, device="cuda", dtype=torch.int32)
    compile_flag_toy(FlagToy())(out, value, delay)
    expected = torch.arange(32, device="cuda", dtype=torch.int32) + value
    torch.testing.assert_close(out[:, 0], expected)
    torch.testing.assert_close(out[:, 1], expected * 2)


def test_describe_lists_named_barriers_and_their_users():
    """A compile lists each barrier's users; a second instance served from the cache restores
    them from the trace record."""
    expected = [
        [
            "done",
            "id",
            "3",
            "96",
            "threads",
            "arrive",
            "by",
            "publisher;",
            "sync",
            "by",
            "waiters",
        ],
        ["pair_sync", "id", "4", "64", "threads", "sync", "by", "waiters"],
    ]
    for op in (FlagToy(), FlagToy()):
        assert "users known after a compile" in op.describe()
        compile_flag_toy(op)
        rows = [line.split() for line in op.describe().splitlines()]
        assert all(row in rows for row in expected)
    assert compile_flag_toy.cache_info().hits >= 1


@pytest.mark.parametrize(
    "mode, message",
    [
        ("collision", r"named_barrier\(3\) at .* uses a barrier ID owned by NamedBarrier done"),
        ("out_of_range", r"named_barrier\(16\) at .*: barrier IDs are 0-15"),
        ("foreign", r"publisher: NamedBarrier done at .* belongs to another kernel"),
        (
            "short_users",
            r"NamedBarrier pair_sync counts 96 threads, but the roles using it \(waiters\) have "
            r"only 64; it can never complete",
        ),
    ],
)
def test_named_barrier_trace_errors(mode, message):
    _raises_protocol(message, lambda: compile_flag_toy(FlagToy(mode)))


# --------------------------------------------------------------------------- #
# Allocation and declaration checks (host only)
# --------------------------------------------------------------------------- #


def _kernel(attrs, roles, cluster=None):
    """A kernel with the given attribute factories and role functions."""
    cls = type("Probe", (WarpSpecialized,), dict(roles))

    def init(self):
        for name, make in attrs.items():
            setattr(self, name, make())
        if cluster is not None:
            self.cluster = cluster

    cls.__init__ = init
    return cls()


@role(warps=0)
@cute.jit
def _prod(self, ctx, *, a: Producer):
    return a


@role(warps=1)
@cute.jit
def _cons(self, ctx, *, a: Consumer):
    return a


PAIR = {"prod": _prod, "cons": _cons}


def test_named_barriers_allocate_upward_and_leave_channel_ids():
    ring = {"a": lambda: Ring(2, Threads, Threads, sync="named")}
    bars = {"x": lambda: NamedBarrier(threads=64), "y": lambda: NamedBarrier(threads=32)}
    plain = _kernel(ring, PAIR)
    op = _kernel({**ring, **bars}, PAIR)
    assert op.named_barrier_ids == plain.named_barrier_ids == {"a": ([12, 13], [14, 15])}
    op.describe()
    assert (op.x.id, op.y.id) == (3, 4)


def test_named_barrier_ids_are_exhaustible():
    # 6 channel stages take 12 IDs; two NamedBarriers make 14 of the 13 free IDs (3-15).
    attrs = {
        "a": lambda: Ring(6, Threads, Threads, sync="named"),
        "x": lambda: NamedBarrier(threads=64),
        "y": lambda: NamedBarrier(threads=64),
    }
    _raises_protocol("need 14 barrier IDs; 13 are free", _kernel(attrs, PAIR).describe)
    del attrs["y"]
    _kernel(attrs, PAIR).describe()  # 13 fit exactly


@pytest.mark.parametrize("threads", [0, 48, 2048, 64.0, True])
def test_named_barrier_counts_whole_warps(threads):
    with pytest.raises(ValueError, match="int multiple of 32 in"):
        NamedBarrier(threads=threads)


def test_named_barrier_count_must_fit_the_kernel():
    attrs = {"a": lambda: Ring(1, Threads, Threads), "x": lambda: NamedBarrier(threads=96)}
    op = _kernel(attrs, PAIR)
    _raises_protocol("counts 96 threads, but the kernel launches 64", op.describe)


def test_named_barrier_declarations_are_distinct():
    shared = NamedBarrier(threads=64)
    attrs = {"a": lambda: Ring(1, Threads, Threads), "x": lambda: shared, "y": lambda: shared}
    op = _kernel(attrs, PAIR)
    _raises_protocol("attributes 'x' and 'y' use the same declaration", op.describe)


def test_named_barrier_id_needs_a_kernel():
    _raises_protocol("has no ID yet", lambda: NamedBarrier(threads=64).id)


class PairProbe(WarpSpecialized):
    """A CTA pair whose ``x`` is synced by a leader-only warp and a warp in both CTAs."""

    cluster = CtaPair()

    def __init__(self, threads: int):
        self.x = NamedBarrier(threads=threads)

    @role(warps=0, cta="leader")
    @cute.jit
    def lead(self, ctx):
        self.x.sync()

    @role(warps=1)
    @cute.jit
    def both(self, ctx):
        self.x.sync()

    @cute.kernel
    def kernel(self, out):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out))

    @cute.jit
    def __call__(self, out: cute.Tensor, stream):
        self.kernel.set_name_prefix(f"warpdrive_named_barrier_pair_{self.x.threads}")
        self.kernel(out).launch(**self.launch_config(grid=(2, 1, 1)), stream=stream)


@jit_cache
def compile_pair_probe(op):
    out = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    return compile_tvm_ffi(op, out, name=f"named_barrier_pair_{op.x.threads}")


def test_named_barrier_users_are_counted_per_cta():
    """The peer CTA runs only ``both``: 32 threads can never complete a 64-thread barrier
    there, though the leader CTA has 64."""
    _raises_protocol(
        r"NamedBarrier x counts 64 threads, but the roles using it in CTA rank 1 \(both\) have "
        r"only 32",
        lambda: compile_pair_probe(PairProbe(64)),
    )
    compile_pair_probe(PairProbe(32))
