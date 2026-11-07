"""Import-time compatibility across the supported CuTeDSL versions (4.6.2 through 4.8).

Use version predicates when selecting implementations on the host, not inside a kernel or
launch hot path. Keep renamed APIs here so kernels use stable names. The module mirrors
attention-gym's ``attn_gym/_backends/cute/compat.py``.
"""

from importlib.metadata import version

from cutlass import cute
from packaging.specifiers import SpecifierSet
from packaging.version import Version

CUTEDSL_VERSION = Version(version("nvidia-cutlass-dsl"))
# 4.6.0 and 4.6.1 elect inside a bulk copy, so one issued in elect_one() (as 4.8 requires)
# runs a nested elect.sync and hangs.
MIN_CUTEDSL_VERSION = "4.6.2"


def cutedsl_version_matches(specifier: str) -> bool:
    """Match a PEP 440 range or exact version, including qualifying prereleases.

    For example, ``>=4.8`` excludes ``4.8.0rc1`` but includes ``4.9.0.dev1``.
    Invalid specifiers raise ``packaging.specifiers.InvalidSpecifier``.
    """
    return SpecifierSet(specifier).contains(CUTEDSL_VERSION, prereleases=True)


if not cutedsl_version_matches(f">={MIN_CUTEDSL_VERSION}"):
    raise ImportError(
        f"warpdrive requires nvidia-cutlass-dsl>={MIN_CUTEDSL_VERSION}, found {CUTEDSL_VERSION}: "
        "earlier versions elect inside a cp.async.bulk copy, so a bulk copy issued in "
        "cute.arch.elect_one() runs a nested elect.sync and hangs"
    )


if cutedsl_version_matches(">=4.8"):
    from cutlass.memory import SmemAllocator, TmemAllocator
    from cutlass.tensor_utils import LayoutEnum
else:
    from cutlass.utils import LayoutEnum, SmemAllocator, TmemAllocator

if cutedsl_version_matches(">=4.7"):
    from cutlass.base_dsl import Arch

    # The legacy FastDivmodDivisor warns from 4.7; the divmod arithmetic is the same.
    FastDivmodDivisor = cute.FastDivmodDivisorV2
else:
    from cutlass.base_dsl.arch import Arch

    FastDivmodDivisor = cute.FastDivmodDivisor


__all__ = [
    "CUTEDSL_VERSION",
    "Arch",
    "FastDivmodDivisor",
    "LayoutEnum",
    "SmemAllocator",
    "TmemAllocator",
    "cutedsl_version_matches",
]
