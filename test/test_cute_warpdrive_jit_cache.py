"""warpdrive kernels as jit_cache compile arguments: what the key covers, reuse across runtime
shapes, invalidation by source edits, and pickling for parallel compilation."""

import importlib
import pickle
import sys
import textwrap

import pytest
import torch

pytest.importorskip("cutlass.cute")

from transformer_nuggets.cute.jit_cache import CacheInfo, CompileTarget, compile_many
from transformer_nuggets.cute.jit_cache import _key as cute_key
from transformer_nuggets.cute.warpdrive import ClusterLaunchControl, CtaPair, StaticPersistent
from transformer_nuggets.cute.warpdrive.examples.gemm import Gemm, compile_gemm, gemm
from transformer_nuggets.cute.warpdrive.examples.persistent_gemm import (
    PersistentGemm,
    persistent_gemm,
)
from transformer_nuggets.cute.warpdrive.examples.persistent_gemm import (
    compile_persistent_gemm as compile_pgemm,
)


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires an SM100-family GPU")


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSFORMER_NUGGETS_CUTE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    return tmp_path


def key_of(op, fn=compile_gemm) -> str:
    return cute_key.make_key(fn, (op,), {}, CompileTarget("test"))


class ReorderedGemm(Gemm):
    """Same name and constructor as Gemm: only the class tells the two apart."""


def test_key_names_each_compile_time_choice():
    """Equal configurations share a key; every knob that changes the kernel or its trace-time
    check gets its own (compile only)."""
    base = key_of(Gemm())
    assert key_of(Gemm()) == base
    strict = Gemm()
    strict.alias_check = "strict"
    changed = [
        Gemm(ab_stages=4),
        Gemm(acc_stages=2),
        Gemm().enable_iket(),
        Gemm().enable_iket(waits=False),
        strict,
        ReorderedGemm(),
    ]
    keys = {key_of(op) for op in changed}
    assert base not in keys and len(keys) == len(changed)
    pairs = [PersistentGemm(), PersistentGemm(schedule=StaticPersistent())]  # the default
    assert key_of(pairs[0], compile_pgemm) == key_of(pairs[1], compile_pgemm)
    others = [
        PersistentGemm(schedule=ClusterLaunchControl()),
        PersistentGemm(cluster=CtaPair()),
        PersistentGemm(tile_n=224),
    ]
    keys = {key_of(op, compile_pgemm) for op in [pairs[0], *others]}
    assert len(keys) == len(others) + 1


def test_pickled_kernel_keeps_its_key():
    """A kernel pickles as its constructor call and host settings, so compile_many can ship it
    to a fresh compiler process (compile only)."""
    op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair(), tile_n=224)
    op.enable_iket(waits=False).alias_check = "off"
    copy = pickle.loads(pickle.dumps(op))
    assert type(copy) is PersistentGemm and copy is not op
    assert (copy.iket, copy.iket_waits, copy.alias_check) == (True, False, "off")
    assert key_of(copy, compile_pgemm) == key_of(op, compile_pgemm)


# A kernel module outside the package, as a test or user file would define one: its own source
# is hashed for its compile function and, through module_source_digest, for its kernel class.
FILL_MODULE = '''
import cutlass
from cutlass import cute

from transformer_nuggets.cute.jit_cache import jit_cache
from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import WarpSpecialized, role

VALUE = {value}


class Fill(WarpSpecialized):
    def get_name(self):
        return "warpdrive_jit_cache_fill"

    @role(warps=1)
    @cute.jit
    def fill(self, out):
        out[cute.arch.lane_idx()] = VALUE

    @cute.kernel
    def kernel(self, out: cute.Tensor):
        self.begin_roles()
        self.run_roles(out)

    @cute.jit
    def __call__(self, out: cute.Tensor, stream):
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(out).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


@jit_cache
def compile_fill(op):
    return compile_tvm_ffi(op, make_fake_compact_tensor(cutlass.Int32, (32,), assumed_align=4))
'''


def test_editing_the_kernel_module_recompiles(requires_blackwell, isolated_cache, tmp_path_factory):
    """A warm process loads the kernel from disk; editing its module compiles a new entry, and
    restoring the source finds the old one again."""
    directory = tmp_path_factory.mktemp("kernels")
    source = directory / "warpdrive_fill_kernel.py"
    sys.path.insert(0, str(directory))
    try:

        def load(value: int):
            source.write_text(textwrap.dedent(FILL_MODULE.format(value=value)))
            cute_key.source_fingerprint.cache_clear()
            cute_key.module_source_digest.cache_clear()
            if "warpdrive_fill_kernel" in sys.modules:
                return importlib.reload(sys.modules["warpdrive_fill_kernel"])
            return importlib.import_module("warpdrive_fill_kernel")

        def run(module) -> list[int]:
            out = torch.zeros(32, device="cuda", dtype=torch.int32)
            module.compile_fill(module.Fill())(out)
            return out.tolist()

        module = load(7)
        assert run(module) == [7] * 32
        assert module.compile_fill.cache_info() == CacheInfo(hits=0, misses=1, currsize=1)
        module.compile_fill.cache_clear()  # a warm process: the entry is on disk
        assert run(module) == [7] * 32
        assert module.compile_fill.cache_info() == CacheInfo(hits=1, misses=0, currsize=1)

        module = load(8)
        assert run(module) == [8] * 32
        assert module.compile_fill.cache_info() == CacheInfo(hits=0, misses=1, currsize=1)
        assert len(list(isolated_cache.rglob("*.o"))) == 2

        module = load(7)
        assert run(module) == [7] * 32
        assert module.compile_fill.cache_info() == CacheInfo(hits=1, misses=0, currsize=1)
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop("warpdrive_fill_kernel", None)


# M: 1, a multiple of 16, odd, a tile edge, several waves; N and K: multiples of 8 on both sides
# of 16. Each is a runtime value, so every shape reuses the one compiled kernel.
SHAPES = [(1, 8, 8), (16, 256, 64), (129, 264, 136), (128, 520, 24), (4097, 2056, 520)]


@pytest.mark.parametrize(
    "entry, compile_fn",
    [(gemm, compile_gemm), (persistent_gemm, compile_pgemm)],
    ids=["gemm", "persistent_gemm"],
)
def test_entry_points_reuse_one_kernel_across_shapes(requires_blackwell, entry, compile_fn):
    compile_fn.cache_clear()
    for m, n, k in SHAPES:
        torch.manual_seed(m)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(a.float(), b.float())
        out = entry(a, b)
        assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item() + 1e-2
    info = compile_fn.cache_info()  # a disk load counts as a hit, a compile as a miss
    assert info.currsize == 1 and info.hits + info.misses == len(SHAPES)


def test_compile_many_builds_kernels_in_a_fresh_process(requires_blackwell, isolated_cache):
    """Pickled kernels compile in forked workers of a fresh compiler process; the caller only
    loads the artifacts."""
    ops = [Gemm(ab_stages=2), Gemm(ab_stages=4)]
    compile_gemm.cache_clear()
    fns = compile_many(compile_gemm, [(op,) for op in ops], workers=2)
    assert compile_gemm.cache_info() == CacheInfo(hits=2, misses=0, currsize=2)
    a = torch.randn(300, 136, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(520, 136, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    for fn in fns:
        out = torch.empty(300, 520, device="cuda", dtype=torch.bfloat16)
        fn(a, b, out)
        assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item() + 1e-2
