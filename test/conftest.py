import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_roofline_cache(tmp_path_factory):
    """Keep measured roofline specs out of the user's cache during tests."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(
            "TRANSFORMER_NUGGETS_ROOFLINE_CACHE",
            str(tmp_path_factory.mktemp("roofline") / "specs.json"),
        )
        yield


@pytest.fixture
def traced_compiles(monkeypatch):
    """Keep this test's compiles out of the jit_cache disk cache (``CUTE_DSL_NO_CACHE``).

    For tests that assert on tracing itself (IKET ranges emitted, monkeypatched trace hooks).
    A cached warpdrive kernel restores what its trace reported (describe(), alias findings, TMA
    reports, warnings), but nothing is traced. The in-memory cache still applies: clear the
    compile function (``cache_clear()``) when an earlier test in the process may have compiled
    the same key.
    """
    monkeypatch.setenv("CUTE_DSL_NO_CACHE", "1")
