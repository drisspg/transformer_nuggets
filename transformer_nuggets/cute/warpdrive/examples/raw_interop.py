"""Managed and raw synchronization on the same barriers (``docs/dropping_down.md``).

``producer`` (warp 0) is fully managed: it fills stage ``s`` of the ``buf`` SMEM buffer with
epoch ``e``'s words ``e * 64 + t`` and commits each stage on ``ring``.
``consumer`` (warps 1-2) takes its end of ``ring`` but drives the barriers itself: it waits on
the full barrier by pointer with the end's live ``PipelineState``, orders its 64 threads'
reads with the ``read`` named barrier and releases each stage with one arrival from its first
thread, so ``override={"empty_counts": 1}`` replaces the derived 64. ``watcher`` (warp 3) is
a raw role: it waits for the consumer's managed ``done.arrive()`` with a native ``bar.sync``
on ``done.id`` and reduces the consumer's per-thread results into ``out``.

``describe()`` shows what this leaves unverified: the overridden empty count next to the
derived one, and ``done``'s participation (the native waiter supplies 32 of its 96 threads).
"""

from types import SimpleNamespace

import cutlass
import torch
from cutlass import cute

from transformer_nuggets.cute.jit_cache import jit_cache
from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    Consumer,
    NamedBarrier,
    Producer,
    Ring,
    SmemPlan,
    Threads,
    WarpSpecialized,
    role,
)

WORDS = 64  # one word per consumer thread per stage
CONSUMER_FIRST_THREAD = 32  # warp 1's lane 0


class RawInterop(WarpSpecialized):
    def __init__(self, stages: int = 2):
        self.stages = stages
        self.ring = Ring(stages, Threads, Threads, override={"empty_counts": 1})
        self.read = NamedBarrier(threads=64)  # the consumer's threads
        self.done = NamedBarrier(threads=96)  # the consumer arrives, the watcher syncs
        self.smem_plan = SmemPlan()
        self.smem_plan.buffer("buf", cutlass.Int32)  # (stage, word)
        self.smem_plan.buffer("results", cutlass.Int32)  # (wrong words | sum, thread)

    @role(warps=0)
    @cute.jit
    def producer(self, ctx, *, ring: Producer):
        lane = cute.arch.lane_idx()
        for e in cutlass.range(ctx.n, unroll=1):
            with ring.produce() as h:
                ctx.buf[h.index, lane] = e * WORDS + lane
                ctx.buf[h.index, lane + 32] = e * WORDS + lane + 32
        return ring

    @role(warps=range(1, 3))
    @cute.jit
    def consumer(self, ctx, *, ring: Consumer):
        t = cute.arch.thread_idx()[0] - CONSUMER_FIRST_THREAD
        bad, total = cutlass.Int32(0), cutlass.Int32(0)
        state = ring.state  # the live PipelineState: this body advances it by hand
        for e in cutlass.range(ctx.n, unroll=1):
            cute.arch.mbarrier_wait(ring.full_barrier(state.index), state.phase)
            word = ctx.buf[state.index, t]
            bad += word % WORDS != t  # this thread's word; the sum checks the epochs
            total += word
            self.read.sync()  # every thread has read the stage before the one release
            if t == 0:
                cute.arch.mbarrier_arrive(ring.empty_barrier(state.index))
            state.advance()
        ring.state = state
        ctx.results[0, t] = bad
        ctx.results[1, t] = total
        self.done.arrive()  # publishes the results to the watcher
        return ring

    @role(warps=3, raw=True)
    @cute.jit
    def watcher(self, ctx):
        cute.arch.barrier(barrier_id=self.done.id, number_of_threads=self.done.threads)
        lane = cute.arch.lane_idx()
        if lane < 2:
            acc = cutlass.Int32(0)
            for t in cutlass.range(WORDS, unroll=1):
                acc += ctx.results[lane, t]
            ctx.out[lane] = acc

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        layouts = {
            "buf": cute.make_layout((self.stages, WORDS)),
            "results": cute.make_layout((2, WORDS)),
        }
        self.begin_roles(layouts=layouts)
        ctx = SimpleNamespace(out=out, n=n, buf=self.buffer("buf"), results=self.buffer("results"))
        self.run_roles(ctx)

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix("warpdrive_raw_interop")
        self.kernel(out, n).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def compile_raw_interop(op: RawInterop):
    out = make_fake_compact_tensor(cutlass.Int32, (2,), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), name="warpdrive_raw_interop")


def raw_interop(n: int, stages: int = 2) -> torch.Tensor:
    """``[wrong words, sum of every word]`` after ``n`` epochs through the ring."""
    out = torch.full((2,), -1, device="cuda", dtype=torch.int32)
    compile_raw_interop(RawInterop(stages))(out, n)
    return out
