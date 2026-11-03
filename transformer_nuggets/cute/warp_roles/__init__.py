"""warp_roles: a thin warp-role layer over CuTeDSL mbarriers, TMEM, and ``setmaxnreg``.

A kernel declares channels (``Ring`` for rotating stages, ``Slots`` for caller-chosen slots),
turnstiles, and a ``TmemPlan`` as attributes, and writes roles as methods decorated
``@role(warps, regs)`` directly above ``@cute.jit``. Role parameters receive channel ends
by name and annotation (``Producer``, ``Consumer``, ``Consumer[Instance]``), the TMEM view
(``Tmem``), the instance index (``Instance``), or a turnstile seat (``Turn``).

Ends are DSL values: dynamic loops carry their phases when the end is a local name that the
loop body uses through a method call or reassignment. Helpers that take ends must return them
and the caller must reassign (``s, stats = step(s, stats)``), as with any CuTeDSL value.
Work scheduling (``scheduling.py``): declare ``self.tiles = TileStream(order=..., policy=...)``
and give roles ``tiles: Tiles``; the policy (``OneShot``, ``StaticPersistent``,
``DynamicPersistent``, ``Clc``) is a one-line switch. Clusters (``cluster.py``): declare
``self.cluster = CtaPair()`` and pin the MMA role with ``@role(..., cta="leader")``; the layer
derives every cross-CTA barrier, arrival count, multicast mask, and teardown step. See ``examples/`` for a GEMM, a persistent
GEMM, and an FA4-style attention forward.
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
from transformer_nuggets.cute.warp_roles.cluster import CtaPair, SingleCta
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
from transformer_nuggets.cute.warp_roles.scheduling import (
    Clc,
    DynamicPersistent,
    OneShot,
    PdlWait,
    Raster,
    StaticPersistent,
    TileCursor,
    TileProblem,
    Tiles,
    TileStream,
    WorkMapping,
)
from transformer_nuggets.cute.warp_roles.tmem import TmemPlan, TmemView

__all__ = [
    "Channel",
    "Clc",
    "Consumer",
    "CtaPair",
    "DynamicPersistent",
    "Handle",
    "Instance",
    "OneShot",
    "PdlWait",
    "Producer",
    "ProtocolError",
    "ProtocolLog",
    "Raster",
    "Ring",
    "SingleCta",
    "Slots",
    "StaticPersistent",
    "Threads",
    "TileCursor",
    "TileProblem",
    "TileStream",
    "Tiles",
    "Tma",
    "Tmem",
    "TmemPlan",
    "TmemView",
    "Turn",
    "Turnstile",
    "Umma",
    "WarpSpecialized",
    "WorkMapping",
    "role",
]
