"""Explicit role results keep final phases out of Python frame introspection."""

import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    Clc,
    Consumer,
    CtaPair,
    DynamicPersistent,
    Producer,
    Slots,
    StaticPersistent,
    Threads,
    Tiles,
    TileStream,
    Umma,
    WarpSpecialized,
    role,
)


@pytest.fixture
def requires_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires a Blackwell GPU")


class ReturnedProducer(WarpSpecialized):
    def __init__(self, mode="renamed"):
        self.mode = mode
        self.cluster = CtaPair()
        self.ch = Slots(1, Threads, Umma)

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        live = ch
        ch = None  # a role may rename its argument; teardown follows the returned end
        for _ in cutlass.range(ctx.n, unroll=1):
            with live.produce(0):
                pass
        if cutlass.const_expr(self.mode == "missing"):
            return None
        if cutlass.const_expr(self.mode == "duplicate"):
            return live, live
        if cutlass.const_expr(self.mode == "wrong_type"):
            return 7
        return live

    @role(warps=1, cta="leader")
    @cute.jit
    def consume(self, ch: Consumer, ctx):
        count = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume(0):
                count += 1
        if cute.arch.lane_idx() == 0:
            ctx.out[0] = count

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_returned_producer_" + self.mode)
        self.kernel(out, n).launch(**self.launch_config(grid=(2, 1, 1)), stream=stream)


def compile_producer(op):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4),
        cutlass.Int32(1),
        name="warpdrive_explicit_producer_" + op.mode,
    )


@pytest.mark.parametrize("n", [0, 1, 2, 9, 64])
def test_returned_renamed_producer_drains_final_state(requires_blackwell, n):
    fn = compile_producer(ReturnedProducer())
    out = torch.full((1,), -1, device="cuda", dtype=torch.int32)
    fn(out, n)
    assert out.item() == n


@pytest.mark.parametrize(
    "mode, message",
    [
        ("missing", "return the final producer end for ch"),
        ("duplicate", "returned the same endpoint"),
        ("wrong_type", "return final producer ends and tile cursors"),
    ],
)
def test_bad_role_result_rejected_before_launch(requires_blackwell, mode, message):
    with pytest.raises(Exception, match=message):
        compile_producer(ReturnedProducer(mode))


class ReturnedCursor(WarpSpecialized):
    def __init__(self, policy, *, missing=False):
        self.tiles = TileStream(policy=policy)
        self.missing = missing

    @role(warps=0)
    @cute.jit
    def count(self, tiles: Tiles, ctx):
        live = tiles
        tiles = None
        while live.valid:
            i, _, _ = live.work
            if cute.arch.lane_idx() == 0:
                ctx.out[i] = i + 1
            live = self.next_item(live)
        if cutlass.const_expr(self.missing):
            return None
        return live

    @cute.jit
    def next_item(self, tiles):
        tiles.advance()
        return tiles

    @cute.kernel
    def kernel(self, out: cute.Tensor, work):
        self.begin_roles(tiles=work)
        self.run_roles(SimpleNamespace(out=out))

    @cute.jit
    def __call__(self, out: cute.Tensor, workspace: cute.Tensor, stream: cuda.CUstream):
        work = self.tiles.problem((out.shape[0], 1, 1), workspace, 2)
        self.kernel.set_name_prefix("warpdrive_returned_cursor")
        self.kernel(out, work).launch(**self.launch_config(work), stream=stream)


def compile_cursor(op):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        name="warpdrive_explicit_cursor",
    )


@pytest.mark.parametrize("policy", [StaticPersistent(), DynamicPersistent(), Clc()])
def test_returned_cursor_from_helper(requires_blackwell, policy):
    op = ReturnedCursor(policy)
    fn = compile_cursor(op)
    workspace, _ = op.tiles.launch_args("cuda")
    for n in (1, 19, 33):
        out = torch.zeros(n, device="cuda", dtype=torch.int32)
        fn(out, workspace)
        torch.testing.assert_close(out, torch.arange(1, n + 1, device="cuda", dtype=torch.int32))


@pytest.mark.parametrize("iket", [False, True])
def test_cursor_return_required_independently_of_tracing(requires_blackwell, iket):
    op = ReturnedCursor(StaticPersistent(), missing=True).enable_iket(iket)
    with pytest.raises(Exception, match="return the final tile cursor for tiles"):
        compile_cursor(op)


def test_no_python_profiler_needed(requires_blackwell, monkeypatch):
    def forbidden(*args):
        pytest.fail("role dispatch must not inspect Python frames through setprofile")

    monkeypatch.setattr(sys, "setprofile", forbidden)
    compile_producer(ReturnedProducer())
    compile_cursor(ReturnedCursor(StaticPersistent()).enable_iket())
