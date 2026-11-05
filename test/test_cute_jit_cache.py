from __future__ import annotations

import importlib
import multiprocessing
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pytest

from transformer_nuggets.cute.jit_cache import _key as cute_key
from transformer_nuggets.cute.jit_cache import cache as cute_cache
from transformer_nuggets.cute.jit_cache import compile as cute_compile
from transformer_nuggets.cute.jit_cache import target as cute_target

_DRIVER_LOG_ENV = "TRANSFORMER_NUGGETS_TEST_CUTE_DRIVER_LOG"
_DRIVER_READY_ENV = "TRANSFORMER_NUGGETS_TEST_CUTE_DRIVER_READY"
_DRIVER_EXPECTED_ENV = "TRANSFORMER_NUGGETS_TEST_CUTE_DRIVER_EXPECTED"


class CompileConfig(NamedTuple):
    """Typed static config used across the compiler-process boundary."""

    variant: str
    timing: float


class FakeCompiled:
    """Minimal compiled callable that exports a loadable test artifact."""

    def __init__(self, value: str):
        self.value = value

    def __call__(self) -> str:
        return self.value

    def export_to_c(self, object_file_path: str, function_name: str) -> None:
        Path(object_file_path).write_text(f"{function_name}:{self.value}")


def load_fake_compiled(path: Path) -> FakeCompiled:
    """Load the test artifact format written by ``FakeCompiled``."""
    function_name, separator, value = path.read_text().partition(":")
    if separator != ":" or function_name != cute_cache.EXPORT_FUNCTION_NAME:
        raise RuntimeError("corrupt test artifact")
    return FakeCompiled(value)


@cute_cache.jit_cache
def compile_test_variant(config: CompileConfig) -> FakeCompiled:
    """Importable fake compiler used to exercise the fresh compiler process."""
    ready_directory = os.getenv(_DRIVER_READY_ENV)
    expected = int(os.getenv(_DRIVER_EXPECTED_ENV, "0"))
    if ready_directory and expected:
        ready_path = Path(ready_directory)
        ready_path.mkdir(parents=True, exist_ok=True)
        (ready_path / f"{config.variant}-{os.getpid()}.ready").touch()
        deadline = time.monotonic() + 10
        while len(list(ready_path.glob("*.ready"))) < expected:
            if time.monotonic() >= deadline:
                raise RuntimeError("compiler workers did not run concurrently")
            time.sleep(0.01)

    log_path = os.getenv(_DRIVER_LOG_ENV)
    if log_path:
        with Path(log_path).open("a") as log:
            log.write(
                f"{os.getpid()},{os.getppid()},{config.variant},{os.getenv('CUTE_DSL_ARCH', '')}\n"
            )
    return FakeCompiled(config.variant)


def wait_for_processes(processes, timeout: float = 15.0) -> None:
    """Join child processes and fail without leaving a process behind."""
    deadline = time.monotonic() + timeout
    try:
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        failures = [(process.pid, process.exitcode) for process in processes if process.exitcode]
        assert failures == []
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


@pytest.fixture(autouse=True)
def reset_compile_target():
    cute_target.set_compile_target(None)
    yield
    cute_target.set_compile_target(None)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Use deterministic fingerprints and an isolated persistent cache."""
    cache_directory = tmp_path / "cache"
    monkeypatch.setenv("TRANSFORMER_NUGGETS_CUTE_CACHE_DIR", str(cache_directory))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    monkeypatch.setattr(
        cute_cache,
        "_source_fingerprint",
        lambda _fn, _extra_sources=(): "test-source",
    )
    monkeypatch.setattr(
        cute_cache,
        "get_compile_target",
        lambda: cute_target.CompileTarget("test"),
    )
    monkeypatch.setattr(cute_cache, "_load_compiled", load_fake_compiled)
    return cache_directory


def test_cache_directory_can_be_overridden(tmp_path, monkeypatch):
    configured_path = tmp_path / "custom-cache"
    monkeypatch.setenv("TRANSFORMER_NUGGETS_CUTE_CACHE_DIR", str(configured_path))

    assert cute_cache.get_cache_path() == configured_path


def test_runtime_cache_skips_persistent_key_for_immutable_arguments(isolated_cache, monkeypatch):
    make_key_calls = 0
    original_make_key = cute_cache._make_key

    def counting_make_key(*args, **kwargs):
        nonlocal make_key_calls
        make_key_calls += 1
        return original_make_key(*args, **kwargs)

    monkeypatch.setattr(cute_cache, "_make_key", counting_make_key)

    @cute_cache.jit_cache
    def compile_kernel(config: CompileConfig) -> FakeCompiled:
        return FakeCompiled(config.variant)

    assert compile_kernel(CompileConfig("shared", 1.0))() == "shared"
    assert compile_kernel(CompileConfig("shared", 1.0))() == "shared"
    assert make_key_calls == 1
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=1, currsize=1)


@pytest.mark.parametrize(
    ("first", "second"),
    [(True, 1), (1, 1.0), (0.0, -0.0)],
)
def test_runtime_cache_preserves_scalar_representation(isolated_cache, first, second):
    def describe(value) -> str:
        if isinstance(value, float):
            return f"float:{value.hex()}"
        return f"{type(value).__name__}:{value}"

    @cute_cache.jit_cache
    def compile_kernel(value) -> FakeCompiled:
        return FakeCompiled(describe(value))

    assert compile_kernel(first)() == describe(first)
    assert compile_kernel(second)() == describe(second)
    assert compile_kernel(first)() == describe(first)
    assert compile_kernel(second)() == describe(second)
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=2, misses=2, currsize=2)


def test_runtime_cache_keeps_nan_payloads_distinct(isolated_cache):
    first_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    second_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0]

    @cute_cache.jit_cache
    def compile_kernel(value: float) -> FakeCompiled:
        return FakeCompiled(struct.pack("!d", value).hex())

    assert compile_kernel(first_nan)() == "7ff8000000000001"
    assert compile_kernel(second_nan)() == "7ff8000000000002"
    assert compile_kernel.cache_info().currsize == 2


def test_explicit_cache_key_is_shared_by_memory_and_disk(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache(cache_key=lambda _value: "shared")
    def compile_kernel(value: str) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(value)

    assert compile_kernel("first")() == "first"
    assert compile_kernel("second")() == "first"
    compile_kernel.cache_clear()
    assert compile_kernel("third")() == "first"
    assert compile_count == 1
    assert len(list(isolated_cache.rglob("*.o"))) == 1


def test_explicit_cache_key_is_shared_by_precompile_and_lookup(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache(cache_key=lambda _value: "shared")
    def compile_kernel(value: str) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(value)

    assert not compile_kernel.is_cached("first")
    compile_kernel.precompile("first")
    assert compile_kernel.is_cached("second")
    assert compile_kernel("second")() == "first"
    assert compile_count == 1
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=0, currsize=1)


def test_explicit_cache_key_must_be_hashable():
    @cute_cache.jit_cache(cache_key=lambda _value: [])
    def compile_kernel(value: str) -> FakeCompiled:
        return FakeCompiled(value)

    with pytest.raises(TypeError, match="cache_key must return a hashable value"):
        compile_kernel("value")


def test_runtime_cache_includes_compile_target(isolated_cache, monkeypatch):
    compile_count = 0
    target = cute_target.CompileTarget("first")
    monkeypatch.setattr(cute_cache, "get_compile_target", lambda: target)

    @cute_cache.jit_cache
    def compile_kernel(value: str) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(cute_cache.get_compile_target().device_type)

    assert compile_kernel("shared")() == "first"
    target = cute_target.CompileTarget("second")
    assert compile_kernel("shared")() == "second"
    target = cute_target.CompileTarget("first")
    assert compile_kernel("shared")() == "first"
    assert compile_count == 2
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=2, currsize=2)


def test_runtime_key_fuzz_matches_persistent_key_equivalence(isolated_cache):
    same_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    same_nan_copy = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    other_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0]
    calls = [
        ((True,), {}),
        ((1,), {}),
        ((1.0,), {}),
        ((0.0,), {}),
        ((-0.0,), {}),
        ((same_nan,), {}),
        ((same_nan_copy,), {}),
        ((other_nan,), {}),
        ((CompileConfig("shared", 1.0),), {}),
        ((["nested", {"value": True}],), {}),
        ((["nested", {"value": 1}],), {}),
        (({same_nan: "value"},), {}),
        (({same_nan: "value", same_nan_copy: "value"},), {}),
        ((), {"first": 1, "second": 2}),
        ((), {"second": 2, "first": 1}),
    ]
    target = cute_target.CompileTarget("test")

    def compile_kernel(value=None, **kwargs):
        return value, kwargs

    runtime_keys = [cute_cache._make_runtime_key(*call, target) for call in calls]
    persistent_keys = [cute_cache._make_key(compile_kernel, *call, target) for call in calls]
    for left in range(len(calls)):
        for right in range(len(calls)):
            if runtime_keys[left] == runtime_keys[right]:
                assert persistent_keys[left] == persistent_keys[right]

    assert runtime_keys[5] == runtime_keys[6]
    assert runtime_keys[5] != runtime_keys[7]
    assert runtime_keys[-2] == runtime_keys[-1]

    bool_target = cute_target.CompileTarget("test", sm_count=True)
    int_target = cute_target.CompileTarget("test", sm_count=1)
    assert cute_cache._make_runtime_key((), {}, bool_target) != cute_cache._make_runtime_key(
        (), {}, int_target
    )


def test_runtime_cache_snapshots_mutable_arguments(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache
    def compile_kernel(values: list[str]) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(values[0])

    values = ["first"]
    assert compile_kernel(values)() == "first"
    assert compile_kernel(["first"])() == "first"
    values[0] = "second"
    assert compile_kernel(values)() == "second"
    assert compile_kernel(["second"])() == "second"
    compile_kernel.cache_clear()
    assert compile_kernel(["first"])() == "first"
    assert compile_count == 2


def test_same_key_compiles_once_across_threads(isolated_cache):
    compile_count = 0
    count_lock = threading.Lock()

    @cute_cache.jit_cache
    def compile_kernel(key: str) -> FakeCompiled:
        nonlocal compile_count
        with count_lock:
            compile_count += 1
        time.sleep(0.1)
        return FakeCompiled(key)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(compile_kernel, ["shared"] * 8))

    assert compile_count == 1
    assert [result() for result in results] == ["shared"] * 8
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=7, misses=1, currsize=1)
    assert len(list(isolated_cache.rglob("*.o"))) == 1


def test_distinct_keys_serialize_compilation_within_process(isolated_cache):
    active_compiles = 0
    max_active_compiles = 0
    count_lock = threading.Lock()

    @cute_cache.jit_cache
    def compile_kernel(key: str) -> FakeCompiled:
        nonlocal active_compiles, max_active_compiles
        with count_lock:
            active_compiles += 1
            max_active_compiles = max(max_active_compiles, active_compiles)
        time.sleep(0.1)
        with count_lock:
            active_compiles -= 1
        return FakeCompiled(key)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(compile_kernel, ["first", "second"]))

    assert {result() for result in results} == {"first", "second"}
    assert max_active_compiles == 1
    assert compile_kernel.cache_info().misses == 2


def test_corrupt_artifact_is_recompiled(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache
    def compile_kernel(key: str) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(key)

    assert compile_kernel("key")() == "key"
    compile_kernel.cache_clear()
    artifact = next(isolated_cache.rglob("*.o"))
    artifact.write_bytes(b"corrupt")

    assert compile_kernel("key")() == "key"
    assert compile_count == 2
    assert artifact.read_text() == f"{cute_cache.EXPORT_FUNCTION_NAME}:key"


def test_failed_export_does_not_publish_partial_artifact(isolated_cache):
    class FailedExport(FakeCompiled):
        def export_to_c(self, object_file_path: str, function_name: str) -> None:
            Path(object_file_path).write_bytes(b"partial")
            raise RuntimeError("export failed")

    @cute_cache.jit_cache
    def compile_kernel() -> FailedExport:
        return FailedExport("value")

    assert compile_kernel()() == "value"
    assert not list(isolated_cache.rglob("*.o"))


def test_persistent_cache_can_be_disabled_with_an_option(tmp_path, monkeypatch):
    cache_directory = tmp_path / "cache"
    monkeypatch.setenv("TRANSFORMER_NUGGETS_CUTE_CACHE_DIR", str(cache_directory))
    monkeypatch.setattr(
        cute_cache,
        "get_compile_target",
        lambda: cute_target.CompileTarget("test"),
    )
    monkeypatch.setattr(
        cute_cache,
        "_source_fingerprint",
        lambda _fn, _extra_sources=(): "test-source",
    )
    compile_count = 0

    @cute_cache.jit_cache(persistent=False)
    def compile_kernel() -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled("value")

    compile_kernel()
    compile_kernel.cache_clear()
    compile_kernel()

    assert compile_count == 2
    assert not list(cache_directory.rglob("*.o"))


def test_bad_fork_check_precedes_cached_target_lookup(monkeypatch):
    """Reject a child that inherited a target cached by its parent."""
    import torch

    class Properties:
        major = 10
        minor = 0
        name = "test-gpu"
        multi_processor_count = 100

    bad_fork = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: Properties())
    monkeypatch.setattr(torch.cuda, "_is_in_bad_fork", lambda: bad_fork, raising=False)
    cute_target._query_compile_target.cache_clear()
    try:
        assert cute_target.detect_compile_target() == cute_target.CompileTarget(
            device_type="cuda",
            capability=(10, 0),
            name="test-gpu",
            sm_count=100,
        )
        bad_fork = True
        with pytest.raises(RuntimeError, match="forked CUDA child"):
            cute_target.detect_compile_target()
    finally:
        cute_target._query_compile_target.cache_clear()


@pytest.mark.parametrize(
    ("configured_arch", "physical", "expected"),
    (
        ("sm_90a", (10, 0), (9, 0)),
        ("sm_100a", (9, 0), (10, 0)),
        ("sm_103a", (10, 0), (10, 3)),
        (None, (10, 3), (10, 3)),
        (None, None, None),
    ),
)
def test_compile_target_effective_capability(configured_arch, physical, expected):
    """Prefer an explicit code-generation architecture over the physical GPU."""
    target = cute_target.CompileTarget(
        device_type="cuda",
        configured_arch=configured_arch,
        capability=physical,
    )
    assert target.effective_capability == expected


def test_compile_target_rejects_invalid_configured_architecture():
    target = cute_target.CompileTarget(device_type="cuda", configured_arch="native")
    with pytest.raises(ValueError, match="configured CUDA architecture"):
        _ = target.effective_capability


def test_explicit_target_avoids_device_discovery(monkeypatch):
    target = cute_target.CompileTarget(
        device_type="cuda",
        capability=(10, 0),
        name="test-gpu",
        sm_count=100,
    )

    def unexpected_discovery():
        raise AssertionError("CUDA discovery was called")

    monkeypatch.setattr(cute_target, "detect_compile_target", unexpected_discovery)
    cute_target.set_compile_target(target)

    assert cute_target.get_compile_target() == target


@pytest.mark.skipif(sys.platform == "win32", reason="fork is required")
def test_fresh_compile_driver_forks_parallel_workers(tmp_path, monkeypatch):
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")
    cache_directory = tmp_path / "cache"
    log_path = tmp_path / "compiles.log"
    ready_directory = tmp_path / "ready"
    configs = tuple(CompileConfig(str(index), float(index + 1)) for index in range(2))
    shared = CompileConfig("shared", 0.5)
    monkeypatch.setenv("TRANSFORMER_NUGGETS_CUTE_CACHE_DIR", str(cache_directory))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    monkeypatch.setenv(_DRIVER_LOG_ENV, str(log_path))
    monkeypatch.setenv(_DRIVER_READY_ENV, str(ready_directory))
    monkeypatch.setenv(_DRIVER_EXPECTED_ENV, str(len(configs)))
    monkeypatch.setattr(cute_cache, "_load_compiled", load_fake_compiled)
    compile_test_variant.cache_clear()
    target = cute_target.CompileTarget(
        device_type="cuda",
        capability=(10, 0),
        name="test-gpu",
        sm_count=100,
    )
    calls = [(config,) for config in configs + (shared,) * 2]

    compiled = cute_compile.compile_many(
        compile_test_variant, calls, workers=len(configs), target=target
    )

    assert [fn() for fn in compiled] == [config.variant for (config,) in calls]
    records = [line.split(",") for line in log_path.read_text().splitlines()]
    variant_records = [record for record in records if record[2] != "shared"]
    worker_pids = {int(record[0]) for record in variant_records}
    driver_pids = {int(record[1]) for record in records}
    assert {record[2] for record in variant_records} == {config.variant for config in configs}
    assert sum(record[2] == "shared" for record in records) == 1
    assert len(worker_pids) == len(configs)
    assert len(driver_pids) == 1
    assert os.getpid() not in driver_pids
    assert {record[3] for record in records} == {"sm_100a"}
    assert len(list(cache_directory.rglob("*.o"))) == len(configs) + 1

    def unexpected_driver(*args, **kwargs):
        raise AssertionError("a fully warm compile_many started a compiler process")

    compile_test_variant.cache_clear()
    monkeypatch.setattr(cute_compile.subprocess, "run", unexpected_driver)
    warm = cute_compile.compile_many(
        compile_test_variant, calls[:-1], workers=len(configs), target=target
    )
    assert [fn() for fn in warm] == [config.variant for (config,) in calls[:-1]]
    assert compile_test_variant.cache_info() == cute_cache.CacheInfo(
        hits=len(calls) - 1, misses=0, currsize=len(configs) + 1
    )


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl.flock and fork are required")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded.*:DeprecationWarning")
def test_same_key_compiles_once_across_processes(isolated_cache):
    context = multiprocessing.get_context("fork")
    process_count = 2
    compile_count = context.Value("i", 0)
    start = context.Event()

    @cute_cache.jit_cache
    def compile_kernel(key: str) -> FakeCompiled:
        with compile_count.get_lock():
            compile_count.value += 1
        time.sleep(0.2)
        return FakeCompiled(key)

    def worker() -> None:
        start.wait()
        assert compile_kernel("shared")() == "shared"

    processes = [context.Process(target=worker) for _ in range(process_count)]
    for process in processes:
        process.start()
    start.set()
    wait_for_processes(processes)

    assert compile_count.value == 1
    assert len(list(isolated_cache.rglob("*.o"))) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl.flock and fork are required")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded.*:DeprecationWarning")
@pytest.mark.parametrize("process_count", [1, 2])
def test_parallel_population_then_sequential_reload(isolated_cache, process_count):
    """Workers compile distinct variants before the parent reloads them sequentially."""
    context = multiprocessing.get_context("fork")
    compile_count = context.Value("i", 0)
    compile_barrier = context.Barrier(process_count, timeout=10)

    @cute_cache.jit_cache
    def compile_kernel(variant: str) -> FakeCompiled:
        with compile_count.get_lock():
            compile_count.value += 1
        compile_barrier.wait()
        return FakeCompiled(variant)

    def run_wave() -> None:
        start = context.Event()

        def worker(variant: int) -> None:
            start.wait()
            expected = str(variant)
            assert compile_kernel(expected)() == expected

        processes = [
            context.Process(target=worker, args=(variant,)) for variant in range(process_count)
        ]
        for process in processes:
            process.start()
        start.set()
        wait_for_processes(processes)

    run_wave()
    assert compile_count.value == process_count
    assert len(list(isolated_cache.rglob("*.o"))) == process_count

    compile_kernel.cache_clear()
    for variant in range(process_count):
        expected = str(variant)
        assert compile_kernel(expected)() == expected

    assert compile_count.value == process_count
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(
        hits=process_count,
        misses=0,
        currsize=process_count,
    )


# below, through each import form the source key must follow.
SYNTHETIC_PACKAGE = {
    "__init__.py": "",
    "entry.py": """
import pkg.absolute
import pkg.aliased as aliased
import pkg.sub.leaf
from . import relative
from pkg.lazymap import thing
from pkg.nested import VALUE
from pkg.reexport import NAME
from pkg.starred import *

from transformer_nuggets.cute.jit_cache import jit_cache


@jit_cache
def compile_kernel():
    from pkg import local
""",
    "absolute.py": "from pkg import cycle\n",
    "cycle.py": "from pkg import absolute\n",
    "aliased.py": "VALUE = 1\n",
    "starred.py": "VALUE = 1\n",
    "relative.py": "VALUE = 1\n",
    "local.py": "VALUE = 1\n",
    # Excluded: ``import pkg.sub.leaf`` runs this ``__init__`` only implicitly.
    "sub/__init__.py": "from pkg import unrelated\n",
    "sub/leaf.py": "from ..relative import VALUE\n",
    "nested/__init__.py": "from . import child\nfrom .. import upper\nVALUE = 1\n",
    "nested/child.py": "VALUE = 1\n",
    "upper.py": "VALUE = 1\n",
    "reexport/__init__.py": "from .impl import NAME\n",
    "reexport/impl.py": "NAME = 1\n",
    # Lazy exports name their owners only as strings .
    "lazymap.py": """
EXPORTS = {"thing": "pkg.lazy_target"}


def __getattr__(name):
    import importlib

    if name not in EXPORTS:
        raise AttributeError(name)
    return getattr(importlib.import_module(EXPORTS[name]), name)
""",
    "lazy_target.py": "thing = 1\n",
    # Excluded: nothing imports it.
    "unrelated.py": "VALUE = 1\n",
}


@pytest.fixture
def synthetic_package(tmp_path, monkeypatch):
    """Import ``SYNTHETIC_PACKAGE`` as ``pkg`` and treat it as the package the key follows."""
    root = tmp_path / "pkg"
    for name, source in SYNTHETIC_PACKAGE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(cute_key, "_PACKAGE_ROOT", root)
    yield root
    for name in [name for name in sys.modules if name == "pkg" or name.startswith("pkg.")]:
        del sys.modules[name]


@pytest.mark.parametrize("module", sorted(SYNTHETIC_PACKAGE))
def test_cache_namespace_changes_only_with_imported_sources(synthetic_package, module):
    """Editing a module the compile function transitively imports recompiles; others do not."""
    pytest.importorskip("cutlass")
    compile_kernel = importlib.import_module("pkg.entry").compile_kernel
    baseline = compile_kernel.cache_namespace()
    with (synthetic_package / module).open("a") as source:
        source.write("EDITED = 1\n")
    cute_key.source_fingerprint.cache_clear()

    changed = compile_kernel.cache_namespace() != baseline
    assert changed == (module not in {"sub/__init__.py", "unrelated.py"})


def test_callers_outside_the_package_key_on_their_package_imports(
    synthetic_package, tmp_path_factory, monkeypatch
):
    """A compile function outside the package keys on its own file and package imports."""
    pytest.importorskip("cutlass")
    caller = tmp_path_factory.mktemp("outside") / "outside_kernels.py"
    caller.write_text(
        "import pkg.sub.leaf\nfrom transformer_nuggets.cute.jit_cache import jit_cache\n\n"
        "@jit_cache\ndef compile_kernel():\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(caller.parent))
    module = importlib.import_module("outside_kernels")
    monkeypatch.setitem(sys.modules, "outside_kernels", module)  # Removed at teardown.
    compile_kernel = module.compile_kernel

    def namespace_after_editing(path: Path) -> str:
        with path.open("a") as source:
            source.write("EDITED = 1\n")
        cute_key.source_fingerprint.cache_clear()
        return compile_kernel.cache_namespace()

    baseline = compile_kernel.cache_namespace()
    assert namespace_after_editing(synthetic_package / "unrelated.py") == baseline
    assert namespace_after_editing(synthetic_package / "relative.py") != baseline
    assert namespace_after_editing(caller) != baseline


def test_kernel_layer_modules_are_part_of_the_kernel_key():
    """An example's key covers the warpdrive layer it inlines, not unrelated kernels."""
    package = cute_key._PACKAGE_ROOT
    closure = cute_key.module_closure(package / "cute/warpdrive/examples/gemm.py")
    assert package / "cute/warpdrive/kernel.py" in closure
    assert package / "cute/warpdrive/channels.py" in closure
    assert package / "cute/warpdrive/examples/fa4_fwd.py" not in closure
    assert package / "cute/sgemm.py" not in closure
