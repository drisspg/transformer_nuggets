"""Explicit role results keep final phases out of Python frame introspection."""

import sys
from dataclasses import dataclass
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
    Instance,
    Producer,
    Slots,
    StaticPersistent,
    Threads,
    Tiles,
    TileStream,
    Tmem,
    Turn,
    Turnstile,
    Umma,
    WarpSpecialized,
    role,
)
from transformer_nuggets.cute.warpdrive.examples.gemm import Gemm, GemmParams, compile_gemm
from transformer_nuggets.cute.warpdrive.examples.persistent_gemm import (
    PersistentGemm,
    PersistentGemmParams,
    compile_persistent_gemm,
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
        if cutlass.const_expr(self.mode == "missing_consumer"):
            return None
        return ch

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
        ("missing", "return all warpdrive state arguments; missing ch"),
        ("missing_consumer", "return all warpdrive state arguments; missing ch"),
        ("duplicate", "returned the same state argument"),
        ("wrong_type", "return channel endpoints, tile cursors and turnstile seats"),
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
    with pytest.raises(Exception, match="return all warpdrive state arguments; missing tiles"):
        compile_cursor(op)


@dataclass(frozen=True)
class TurnParams:
    log: cute.Tensor
    iterations: cutlass.Int32


class ReturnedTurn(WarpSpecialized):
    def __init__(self, missing=False):
        self.turns = Turnstile()
        self.missing = missing

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def worker(self, params: TurnParams, seat: Instance, turns: Turn):
        current = turns
        turns = None
        for _ in cutlass.range(params.iterations, unroll=1):
            with current.turn():
                if cute.arch.lane_idx() == 0:
                    pos = params.log[0]
                    params.log[pos + 1] = seat
                    params.log[0] = pos + 1
                    cute.arch.fence_acq_rel_cta()
        if cutlass.const_expr(self.missing):
            return None
        return current

    @cute.kernel
    def kernel(self, log: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(params=TurnParams(log, n))

    @cute.jit
    def __call__(self, log: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_returned_turn")
        self.kernel(log, n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


def compile_turn(op):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4),
        cutlass.Int32(1),
        name="warpdrive_explicit_turn",
    )


def test_returned_turnstile_state(requires_blackwell):
    n = 7
    log = torch.zeros(2 * n + 1, dtype=torch.int32, device="cuda")
    compile_turn(ReturnedTurn())(log, n)
    assert log.cpu().tolist() == [2 * n] + [0, 1] * n


def test_turnstile_must_be_returned(requires_blackwell):
    with pytest.raises(Exception, match="return all warpdrive state arguments; missing turns"):
        compile_turn(ReturnedTurn(missing=True))


class ReorderedGemm(Gemm):
    @role(warps=5, regs=40)
    @cute.jit
    def mma(self, params: GemmParams, ab: Consumer, acc: Producer, tiles: Tiles, tmem: Tmem):
        ab, acc, tiles = Gemm.mma(self, params, ab, acc, tiles, tmem)
        return tiles, acc, ab  # state identity, not tuple position, controls cleanup


def test_return_order_does_not_change_bindings(requires_blackwell):
    torch.manual_seed(0)
    a, b = (torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    expected = torch.empty(256, 256, device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    compile_gemm(Gemm())(a, b, expected)
    compile_gemm(ReorderedGemm())(a, b, actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class ReorderedPersistentGemm(PersistentGemm):
    @role(warps=4, regs=40)
    @cute.jit
    def load(self, params: PersistentGemmParams, ab: Producer, tiles: Tiles):
        ab, tiles = PersistentGemm.load(self, params, ab, tiles)
        return tiles, ab


def test_reordered_state_drives_pair_producer_tail(requires_blackwell):
    # One cluster processes 16 output tiles x 4 K tiles: repeated wraps and a partial last
    # round of the six-stage AB ring. The returned AB state is actually used by its pair tail.
    torch.manual_seed(0)
    a = torch.randn(900, 200, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1000, 200, device="cuda", dtype=torch.bfloat16)
    outputs = []
    for cls in (PersistentGemm, ReorderedPersistentGemm):
        op = cls(cluster=CtaPair(), policy=StaticPersistent())
        fn = compile_persistent_gemm(op)
        workspace, _ = op.tiles.launch_args(a.device)
        out = torch.full((900, 1000), float("nan"), device="cuda", dtype=torch.bfloat16)
        fn(a, b, out, workspace, 2)
        torch.cuda.synchronize()
        outputs.append(out)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_no_python_profiler_needed(requires_blackwell, monkeypatch):
    def forbidden(*args):
        pytest.fail("role dispatch must not inspect Python frames through setprofile")

    monkeypatch.setattr(sys, "setprofile", forbidden)
    compile_producer(ReturnedProducer())
    compile_cursor(ReturnedCursor(StaticPersistent()).enable_iket())
