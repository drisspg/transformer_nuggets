"""warp_roles: authoring code for warp-specialized CuTeDSL kernels.

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
``self.cluster = CtaPair()`` and pin the MMA role with ``@role(..., cta="leader")``. See
``examples/`` for a GEMM, a persistent GEMM, and an FA4-style attention forward.

Design principles
-----------------
The layer exists so that agents and people can write and iterate on warp-specialized kernels
quickly. Explicit code is preferred over clever inference. The layer hides only what has one
correct, checkable answer, removes the footguns around it, and always lets the author drill down
to the underlying CuTeDSL/cutlass objects. State moves through explicit arguments and returns.

The layer derives: arrival counts per slot and per CTA; phase and stage bookkeeping; armed TMA
bytes; barrier storage, initialization, init fences, and teardown; TMEM allocation and free;
``setmaxnreg`` budgets and warpgroup padding; scheduler plumbing; cross-CTA targets and
multicast masks; and pairing/protocol checks (handle lifetimes, TMA atoms vs channel scope and
bytes). ``describe()`` prints every derived value.

The author decides: tiling, layouts, stage counts, the role split, schedules and work order,
which synchronizations to skip, and the math. Each is a constructor argument or a declaration,
so trying another one is a one-line change that the checks re-verify.

Dropping down, per edge:

* Raw state: ``end.state``/``index``/``phase(slot)``/``full_barrier(i)``/``empty_barrier(i)``/
  ``targets``/``scope`` (per-CTA counts, masks, ``tx``) and ``handle.index``/``phase``/
  ``full_barrier``/``empty_barrier``. ``handle.commit(signal=False)``/``release(signal=False)``
  close a handle while the role signals with raw ``cute.arch`` calls.
* ``Channel(override={...})`` replaces derived counts, ``tx``, or masks; ``describe()`` shows
  derived vs overridden values and flags differences.
* ``Barriers(n, counts=None)`` reserves mbarriers that user code (or a ``cutlass.pipeline``
  built on ``self.barrier_ptr(name)``) drives; ``@role(..., raw=True)`` reserves warps and a
  register budget for a function the layer only dispatches. Both still count in SMEM, padding,
  init fences, and teardown.
* ``with unchecked(check, reason=...)`` turns off one trace-time check at one site;
  ``describe()`` lists every opt-out with its reason, line, and what it let through.
* Raw ``cute.copy(..., tma_bar_ptr=p)`` with a pointer the layer did not hand out stays legal;
  ``describe()`` marks that channel's TMA check unverified.

Tracing: ``kernel.enable_iket()`` before compiling makes the layer emit IKET ranges named after
the declarations: ``<role>`` per work item (or per role body), ``<channel>.acquire``/``.wait``/
``.tail`` around blocking waits, ``<turnstile>.turn``, and ``with self.trace_region(name)`` user
regions (``enable_iket(waits=False)``: role ranges only). Disabled, it emits nothing.
``plan_from_warp_roles(kernel, cta=0)`` exports the matching ``PipelinePlan``, so a ``run-iket``
capture feeds ``load_iket_capture``, ``analyze_pipeline``, and ``write_pipeline_perfetto`` from
``transformer_nuggets.cute.profiler.pipeline`` with no hand annotations (``tracing.py``).

Cluster memory scope: thread arrivals that cross CTAs default to ``remote_scope="cta"`` (what
CUTLASS and FA4 use), which counts the arrival but does not order ordinary memory. A pair-scoped
channel whose signal publishes ordinary memory (DSMEM or global) that the other CTA reads
outside a layer payload must declare ``remote_scope="cluster"``. The layer cannot detect this
case. The cluster scope costs about 30% on FA4's hot S channel.
"""

from transformer_nuggets.cute.warp_roles.channels import (
    Barriers,
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
from transformer_nuggets.cute.warp_roles.pipeline_plan import plan_from_warp_roles
from transformer_nuggets.cute.warp_roles.protocol import ProtocolError, ProtocolLog, unchecked
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
    "Barriers",
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
    "plan_from_warp_roles",
    "role",
    "unchecked",
]
