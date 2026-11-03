"""warp_roles: a thin warp-role layer over CuTeDSL mbarriers, TMEM, and ``setmaxnreg``.

A kernel declares channels (``Ring`` for rotating stages, ``Slots`` for caller-chosen slots),
turnstiles, and a ``TmemPlan`` as attributes, and writes roles as methods decorated
``@role(warps, regs)`` directly above ``@cute.jit``. Role parameters receive channel ends
by name and annotation (``Producer``, ``Consumer``, ``Consumer[Instance]``), the TMEM view
(``Tmem``), the instance index (``Instance``), or a turnstile seat (``Turn``).

Ends are DSL values: dynamic loops carry their phases when the end is a local name that the
loop body uses through a method call or reassignment. Helpers that take ends must return them
and the caller must reassign (``s, stats = step(s, stats)``), as with any CuTeDSL value.
See ``examples/`` for a GEMM and an FA4-style attention forward.
"""

from transformer_nuggets.cute.warp_roles.channels import (
    Channel,
    Handle,
    Ring,
    Slots,
    Threads,
    Tma,
    Turnstile,
    Umma,
)
from transformer_nuggets.cute.warp_roles.kernel import (
    Consumer,
    Instance,
    Producer,
    Tmem,
    Turn,
    WarpSpecialized,
    role,
)
from transformer_nuggets.cute.warp_roles.protocol import ProtocolError, ProtocolLog
from transformer_nuggets.cute.warp_roles.tmem import TmemPlan, TmemView

__all__ = [
    "Channel",
    "Consumer",
    "Handle",
    "Instance",
    "Producer",
    "ProtocolError",
    "ProtocolLog",
    "Ring",
    "Slots",
    "Threads",
    "Tma",
    "Tmem",
    "TmemPlan",
    "TmemView",
    "Turn",
    "Turnstile",
    "Umma",
    "WarpSpecialized",
    "role",
]
