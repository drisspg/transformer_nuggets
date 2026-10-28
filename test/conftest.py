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
