"""The root API stays lightweight until its utility subpackages are requested."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "imports",
    [
        "utils, numerics = package.utils, package.numerics",
        "from transformer_nuggets import utils, numerics",
        "from transformer_nuggets import *; assert Path is package.Path and logging is package.logging",
    ],
    ids=["attributes", "from", "star"],
)
def test_root_package_lazily_exports_subpackages(imports: str) -> None:
    # A fresh interpreter avoids subpackages already loaded by other tests or plugins.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
import sys
import transformer_nuggets as package

assert not any(name.startswith(("transformer_nuggets.utils", "transformer_nuggets.numerics"))
               for name in sys.modules)
assert {{"utils", "numerics", "DATA_DIR", "init_logging", "vscode_breakpoint"}} <= set(dir(package))
assert package.DATA_DIR.name == "data"
assert callable(package.init_logging) and callable(package.vscode_breakpoint)
try:
    package.missing_export
except AttributeError as error:
    assert "missing_export" in str(error)
else:
    raise AssertionError("unknown package attribute did not raise")

{imports}
assert utils is package.utils is importlib.import_module("transformer_nuggets.utils")
assert numerics is package.numerics is importlib.import_module("transformer_nuggets.numerics")
assert callable(utils.benchmark_cuda_function_in_microseconds)
assert callable(numerics.ulp_distance)
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
