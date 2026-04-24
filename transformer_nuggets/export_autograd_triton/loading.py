from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_exported_module(path: str | Path) -> ModuleType:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generated module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
