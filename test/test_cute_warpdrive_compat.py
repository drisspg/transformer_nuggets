import importlib.metadata
import importlib.util

import pytest

pytest.importorskip("cutlass.cute")

from transformer_nuggets.cute.warpdrive import _compat


def test_import_rejects_cutedsl_below_the_floor(monkeypatch):
    """Below 4.6.2 warpdrive fails at import with the reason, not later with a GPU hang."""
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "4.6.1")
    spec = importlib.util.spec_from_file_location("warpdrive_compat_probe", _compat.__file__)
    with pytest.raises(ImportError, match=r"nvidia-cutlass-dsl>=4\.6\.2, found 4\.6\.1.*elect"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
