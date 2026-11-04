"""Constructor-selected roles: binding, validation, specialization reuse, and real handoffs."""

from dataclasses import FrozenInstanceError, dataclass, field, replace
from operator import attrgetter

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from transformer_nuggets.cute.cache import (
    clear_cute_cache,
    compile_tvm_ffi_and_cache,
    get_cache_stats,
)
from transformer_nuggets.cute.utils import make_fake_compact_tensor
from transformer_nuggets.cute.warpdrive import (
    Consumer,
    CtaPair,
    Instance,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    Slots,
    Threads,
    WarpSpecialized,
    role,
)


@dataclass(frozen=True)
class CopyRoles:
    load: RoleConfig = field(default_factory=lambda: RoleConfig(0, regs=40))
    consume: RoleConfig = field(default_factory=lambda: RoleConfig(range(4, 8), regs=160))


@dataclass(frozen=True)
class CopyCtx:
    out: cute.Tensor
    iterations: cutlass.Int32


def load_config(kernel: "ConfiguredCopy") -> RoleConfig:
    # Named functions and lambdas have the same host-only selector contract.
    return kernel.roles.load


class ConfiguredCopy(WarpSpecialized):
    def __init__(self, roles: CopyRoles | None = None):
        self.roles = roles if roles is not None else CopyRoles()
        self.data = Ring(2, Threads, Threads, payload=(cutlass.Int32, (32,)))

    def get_name(self) -> str:
        # Deliberately independent of roles: the shared compile cache must distinguish them.
        return "warpdrive_configured_copy"

    def get_key(self) -> str:
        return self.get_name()

    @role(config=load_config)
    @cute.jit
    def load(self, data: Producer, ctx: CopyCtx):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.iterations, unroll=1):
            with data.produce() as h:
                h.data[lane] = i * 32 + lane

    @role(config=lambda kernel: kernel.roles.consume)
    @cute.jit
    def consume(self, data: Consumer, ctx: CopyCtx):
        lane = cute.arch.lane_idx()
        value = cutlass.Int32(0)
        for _ in cutlass.range(ctx.iterations, unroll=1):
            with data.consume() as h:
                value += h.data[lane]
        ctx.out[cute.arch.thread_idx()[0]] = value

    @cute.kernel
    def kernel(self, out: cute.Tensor, iterations: cutlass.Int32):
        self.begin_roles()
        self.run_roles(CopyCtx(out, iterations))

    @cute.jit
    def __call__(self, out: cute.Tensor, iterations: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(out, iterations).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


def compile_copy(op: ConfiguredCopy, bound_method: bool = False):
    return compile_tvm_ffi_and_cache(
        op.__call__ if bound_method else op,
        op.get_key(),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4),
        cutlass.Int32(1),
        name=op.get_name(),
    )


class FixedCopy(ConfiguredCopy):
    @role(warps=0, regs=40)
    @cute.jit
    def load(self, data: Producer, ctx: CopyCtx):
        pass

    @role(warps=range(4, 8), regs=160)
    @cute.jit
    def consume(self, data: Consumer, ctx: CopyCtx):
        pass


def test_config_matches_shorthand_and_normalizes_warps():
    mutable_warps = [range(4), range(4, 8)]
    config = RoleConfig(mutable_warps, regs=192)
    mutable_warps.clear()
    assert config.warps == (range(4), range(4, 8))
    assert RoleConfig(0) == RoleConfig(range(1)) == RoleConfig([range(1)])
    with pytest.raises(FrozenInstanceError):
        config.regs = 184
    assert ConfiguredCopy().role_config_key() == FixedCopy().role_config_key()
    assert (
        ConfiguredCopy().describe().replace("ConfiguredCopy", "FixedCopy")
        == FixedCopy().describe()
    )


def test_resolution_is_per_instance_and_frozen_after_collection():
    selections = []

    def select(kernel):
        selections.append(kernel)
        return kernel.setting

    class Selected(WarpSpecialized):
        def __init__(self, setting):
            self.setting = setting

        @role(config=select)
        @cute.jit
        def worker(self, ctx):
            pass

    first, second = Selected(RoleConfig(0)), Selected(RoleConfig(4, regs=40))
    key = first.role_config_key()
    assert selections == [first]
    first.setting = RoleConfig(2)
    assert first.role_config_key() == key  # snapshot, not a second selector invocation
    second.describe()
    assert selections == [first, second]
    assert second.role_config_key() != key


@pytest.mark.parametrize("selector", [attrgetter("setting"), lambda kernel: kernel.setting])
def test_callable_object_and_lambda_instances(selector):
    class Instanced(WarpSpecialized):
        def __init__(self):
            self.setting = RoleConfig([range(1), range(1, 2)])
            self.data = Slots(2, Threads, Threads)

        @role(config=selector)
        @cute.jit
        def load(self, index: Instance, data: Producer[Instance], ctx):
            pass

        @role(warps=2)
        @cute.jit
        def consume(self, data: Consumer, ctx):
            pass

    op = Instanced()
    op.describe()
    assert op.data.scope.full_counts == {0: [32, 32]}
    assert op.data.scope.empty_counts == {0: [32, 32]}


def test_resolved_placement_drives_pair_scope():
    op = ConfiguredCopy(
        CopyRoles(RoleConfig(0, regs=40, cta="leader"), RoleConfig(range(4, 8), regs=160))
    )
    op.cluster = CtaPair()
    # Pair payloads aren't yet supported: use the same graph as a pure handoff.
    op.data = Ring(2, Threads, Threads)
    op.describe()
    assert op.data.scope.full_counts == {0: [32, 32], 1: [32, 32]}
    assert op.data.scope.empty_counts == {0: [256, 256], 1: [1, 1]}


def test_resolved_graph_validation_and_no_implicit_budget():
    with pytest.raises(ValueError, match="claimed by"):
        ConfiguredCopy(CopyRoles(consume=RoleConfig(0, regs=40))).describe()
    with pytest.raises(ValueError, match="mixed register budgets"):
        ConfiguredCopy(CopyRoles(RoleConfig(0), RoleConfig(1, regs=40))).describe()
    op = ConfiguredCopy(CopyRoles(RoleConfig(0), RoleConfig(1)))
    assert op.num_threads == 64  # no setmaxnreg-driven warpgroup padding
    assert all(entry[4] is None for entry in op.role_config_key())


@pytest.mark.parametrize("value", [-1, 32, [], range(0, 4, 2)])
def test_invalid_selected_warps(value):
    with pytest.raises(ValueError, match="role warps"):
        RoleConfig(value)


@pytest.mark.parametrize("value", [True, 0, 23, 41, 264, 40.0])
def test_invalid_selected_register_budget(value):
    with pytest.raises(ValueError, match="role regs"):
        RoleConfig(0, regs=value)


@pytest.mark.parametrize("kwargs", [{"warps": 0}, {"regs": 40}, {"cta": "leader"}, {"raw": True}])
def test_cannot_mix_selector_with_fixed_placement(kwargs):
    with pytest.raises(ValueError, match="cannot be combined"):
        role(config=lambda kernel: kernel.setting, **kwargs)


def test_bad_selector_diagnostics():
    with pytest.raises(TypeError, match="callable returning RoleConfig"):
        role(config=RoleConfig(0))

    class Bad(WarpSpecialized):
        @role(config=lambda kernel: None)
        @cute.jit
        def worker(self, ctx):
            pass

    with pytest.raises(ProtocolError, match="role worker: config selector must return RoleConfig"):
        Bad().describe()


def test_raw_role_config_and_key():
    class Raw(WarpSpecialized):
        def __init__(self, raw):
            self.setting = RoleConfig(0, raw=raw)

        @role(config=lambda kernel: kernel.setting)
        @cute.jit
        def worker(self, ctx):
            pass

    assert Raw(True).role_config_key() != Raw(False).role_config_key()
    assert Raw(True).num_threads == 32


@pytest.fixture
def requires_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires a Blackwell GPU")


@pytest.mark.parametrize("bound_method", [False, True])
def test_configured_handoffs_and_cache_isolation(requires_blackwell, bound_method):
    clear_cute_cache()
    configs = [
        CopyRoles(),
        CopyRoles(RoleConfig(0, regs=48), RoleConfig(range(4, 8), regs=152)),
        CopyRoles(RoleConfig(0, regs=40), RoleConfig(range(8, 12), regs=160)),
    ]
    compiled = [compile_copy(ConfiguredCopy(config), bound_method) for config in configs]
    assert get_cache_stats()["misses"] == len(configs)
    assert len({id(fn) for fn in compiled}) == len(configs)
    assert compile_copy(ConfiguredCopy(CopyRoles()), bound_method) is compiled[0]
    assert get_cache_stats()["hits"] == 1
    for config, fn in zip(configs, compiled):
        for n in (1, 7, 19):  # first use and ring wraps, without recompiling
            out = torch.full((384,), -1, device="cuda", dtype=torch.int32)
            fn(out, n)
            expected = torch.full_like(out, -1)
            for warp in config.consume.warps[0]:
                lanes = torch.arange(32, device="cuda", dtype=torch.int32)
                expected[warp * 32 : (warp + 1) * 32] = 32 * n * (n - 1) // 2 + n * lanes
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert get_cache_stats()["misses"] == len(configs)


def test_omitted_registers_emit_no_adjustment(requires_blackwell, monkeypatch):
    def unexpected_adjustment(*args, **kwargs):
        pytest.fail("regs=None must not emit setmaxnreg")

    monkeypatch.setattr(cute.arch, "setmaxregister_increase", unexpected_adjustment)
    monkeypatch.setattr(cute.arch, "setmaxregister_decrease", unexpected_adjustment)
    op = ConfiguredCopy(CopyRoles(RoleConfig(0), RoleConfig(1)))
    clear_cute_cache()
    compile_copy(op)


@pytest.mark.parametrize("tvm_ffi", [False, True])
@pytest.mark.parametrize("bound_method", [False, True])
def test_both_cache_entrypoints_key_the_resolved_config(monkeypatch, tvm_ffi, bound_method):
    from transformer_nuggets.cute import cache

    cache.clear_cute_cache()
    built = []

    def compile_marker(op, *args, **kwargs):
        owner = op.__self__ if bound_method else op
        built.append(owner.role_config_key())
        return object()

    monkeypatch.setattr(cache.cute, "compile", compile_marker)
    monkeypatch.setattr(cache, "compile_tvm_ffi", compile_marker)
    compile_fn = cache.compile_tvm_ffi_and_cache if tvm_ffi else cache.compile_and_cache

    def cached(roles=None):
        op = ConfiguredCopy(roles)
        return compile_fn(op.__call__ if bound_method else op, "same_key")

    first = cached()
    changed = CopyRoles(consume=RoleConfig(range(4, 8), regs=152))
    second = cached(changed)
    assert first is not second
    assert cached() is first
    assert len(built) == 2


def test_fa4_register_bucket(requires_blackwell):
    pytest.importorskip("flash_attn.cute")
    from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import Fa4Roles, fa4_fwd

    torch.manual_seed(0)
    q = torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    base = Fa4Roles()
    tuned = replace(
        base,
        softmax=replace(base.softmax, regs=184),
        correction=replace(base.correction, regs=96),
    )
    expected, expected_lse = fa4_fwd(q, k, v, roles=base)
    actual, actual_lse = fa4_fwd(q, k, v, roles=tuned)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)
