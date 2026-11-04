"""Work scheduling for warp-role kernels: which CTA runs which work item, in what order.

Three independent, compile-time choices:

* **policy**: how a CTA obtains its next *linear work index*. Policies know nothing about
  tiles.

  ============================  ========================================  =======================
  policy                        next index                                launch grid
  ============================  ========================================  =======================
  ``OneShot()``                 none (one item per CTA)                   ``total`` (see below)
  ``StaticPersistent()``        ``index + gridDim.x``                     ``min(total, max_ctas)``
  ``DynamicPersistent(stages)`` ``gridDim.x + atomicAdd(counter, 1)``     ``min(total, max_ctas)``
  ``Clc(stages)``               ``clusterlaunchcontrol.try_cancel``       ``total``
  ============================  ========================================  =======================

* **order** (a work mapping): ``index -> work item``, a tuple of Int32 words that ``item()``
  turns into whatever the roles consume. ``Raster`` is the tile mapping (row/column raster, L2
  swizzle bands, reversed axes for longest-processing-time-first). The mapping owns its runtime
  ``space`` (``Raster``: tile extents), the item count, and an optional ``skip(item)`` predicate
  for empty items.
* **launch grid**: ``OneShot`` with the default ``Raster()`` (along m, no swizzle or reversal)
  launches the tile space itself, ``(m_tiles, n_tiles, l)`` CTAs (times the cluster size along
  x), and reads the item from ``blockIdx`` with no division. A launch whose ``n_tiles`` or ``l``
  exceeds the 65535 limit of ``gridDim.y``/``z`` falls back to the 1-D grid; the kernel sees a
  1-D launch (``gridDim.y = gridDim.z = 1`` with more than one ``(n, l)`` pair) and decodes the
  index in a branch of the prologue that tile-grid launches skip. The branch costs about
  25-50 ns on 512^3 to 2048^2 GEMMs against reading ``blockIdx`` unconditionally; decoding on
  every launch cost about 200 ns (GB300, CuTeDSL 4.8.0, CUDA-graph replay of
  examples/gemm.py). A hand-built ``(total, 1, 1)`` grid is also correct, just not
  division-free. Every other combination launches a 1-D grid and decodes the index.
  An empty space (an extent <= 0) launches one work unit; a space whose indices or grid would
  exceed Int32 gets a zero-wide grid that CUDA rejects (``TileProblem.grid``).
* **distribution**, derived by the layer from the policy. Index sequences every role can compute
  (``OneShot``, ``StaticPersistent``) are decoded locally in each role with no synchronization.
  Claimed indices (``DynamicPersistent``, ``Clc``) are claimed by a layer-owned scheduler warp,
  decoded once, and broadcast as ``[index, item words]`` over a ``Ring`` whose consumers are
  every role that takes ``Tiles``. The layer derives the ring's arrival counts, payload, fences,
  and the scheduler's warp (the lowest warp no role claims).

Wherever decoding happens (each role in local mode, the scheduler warp in broadcast mode) the
stream first runs its readiness hook once (``ready=PdlWait()``) and steps past items the mapping
skips, so every role sees the same item sequence.

A kernel declares ``self.tiles = TileStream(order=Raster(swizzle=8), policy=Clc())``; roles
take ``tiles: Tiles`` and loop::

    while tiles.valid:
        m, n, l = tiles.work
        ...
        tiles.advance()

Tracing checks that each role calls ``advance()`` exactly once, directly in the body of a
``while tiles.valid:`` loop (not under a branch or nested loop), so every role steps through
the same sequence.

Clusters (``CtaPair``): a work item belongs to a cluster. Indices count clusters (``blockIdx.x //
2``, ``gridDim.x // 2``), the launch grid is two CTAs per cluster, local policies compute the same
sequence in both CTAs, and broadcast policies claim on the leader only: its scheduler warp
publishes each decoded item into the work-ring slot of both CTAs with ``st.async`` (the bytes
complete each CTA's full barrier), and the consumers of both CTAs release the leader's empty
barrier. CLC cancels whole clusters, so its response decodes to ``ctaid.x // 2``.

Counter lifetime (``DynamicPersistent``): the workspace holds ``[claims, finished]``. Each CTA's
scheduler reports once after its last claim and the last CTA to report zeroes both words, so
launches ordered on one stream (including CUDA-graph replays) reuse one workspace with no host
reset. A workspace must not be shared by concurrently running launches.

Custom schedules subclass the same public protocols as the built-ins: ``Policy`` (next index),
``WorkMapping`` (index -> item, optional ``skip``), and ``Readiness`` (wait before the first
decode). Their docstrings say which methods run on the host (in the ``@cute.jit`` entry point)
and which on the device, and what each must guarantee. ``adapters.py`` wraps external schedulers
(CUTLASS, FA4) as mappings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, is_dataclass

import cutlass
from cutlass import Int32, Int64, cute

from transformer_nuggets.cute.warpdrive.channels import Ring, Slots, Threads, Tma
from transformer_nuggets.cute.warpdrive.protocol import ProtocolError, current_block, user_site
from transformer_nuggets.cute.warpdrive.tracing import pop, push_if

AXES = ("m", "n", "l")
INT32_MAX = 2**31 - 1


class Tiles:
    """Annotation: the role receives a ``TileCursor`` over the stream with the parameter's name."""


def _field_key(obj, skip: tuple[str, ...] = ()) -> str:
    """``_<name><value>`` for each dataclass field: the default compile key of a protocol object
    with constructor arguments (override ``get_key`` for values that are not short tokens)."""
    if not is_dataclass(obj):
        return ""
    return "".join(
        f"_{f.name}" + re.sub(r"[^0-9a-z]+", "_", str(getattr(obj, f.name)).lower())
        for f in fields(obj)
        if f.name not in skip
    )


def _min(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return cutlass.min(a, b)


# --------------------------------------------------------------------------- #
# Work mappings: linear index -> work item
# --------------------------------------------------------------------------- #


class WorkMapping:
    """Public protocol for ``TileStream(order=...)``: linear work index -> work item.

    Subclass it (``Raster`` and the ``adapters.py`` mappings do) and set or override:

    * ``words`` (class attribute, positive int): Int32 words per item. Broadcast policies ship
      ``[index, *words]`` through the work ring, so keep it small.
    * ``bind(space)``, host (the ``@cute.jit`` entry point, inside ``TileStream.problem``):
      normalize the runtime space and return the bound space, any DSL value
      (``extract_mlir_values`` must work on it: tuples of ``Int32``, tensors, CuTeDSL params
      objects). It is passed to the kernel and to every method below. ``bind`` must make an
      empty space safe (``Raster`` stores ``(1, 1, 0)``).
    * ``size(space)``, host and device: the item count, Int32. Items are indices
      ``[0, size)``.
    * ``size64(space)``, host: the Int64 count for the launch's range check
      (``TileProblem.grid``): exact up to ``2**31``, it may saturate there but must not wrap.
      The default widens ``size``, so a mapping whose count can exceed Int32 overrides it.
    * ``decode(index, space)``, device: ``words`` Int32-convertible values. It may be called with
      ``index >= size`` (the result is then ignored), including on an empty space, and must not
      fault or divide by zero there.
    * ``item(words)``, device: what ``tiles.work`` returns (default: the tuple of words).
    * ``skips`` and ``skip(item, space)`` (optional), device: with ``skips = True``, items whose
      ``skip`` is true are removed permanently for every role. Only called for valid indices; it
      is not a readiness wait. It must give the same answer in every role and CTA.
    * ``get_key()``, host: compile-key fragment naming every compile-time choice (never runtime
      extents), in lowercase letters, digits and underscores (examples put it in kernel names).
      ``describe()`` (optional): text for ``describe()``.

    Every method sees the bound space. ``decode`` and ``skip`` must be pure functions of their
    arguments (plus tables written before the launch, or before ``ready`` returns), so every role
    that decodes an index gets the same item.
    """

    words: int = 1
    skips = False

    def bind(self, space):
        return space

    def size(self, space):
        raise NotImplementedError

    def size64(self, space):
        size = self.size(space)
        return size if isinstance(size, int) else Int64(size)

    def decode(self, index, space):
        raise NotImplementedError

    def item(self, words):
        return tuple(words)

    def skip(self, item, space):
        raise NotImplementedError

    def get_key(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Raster(WorkMapping):
    """Tile mapping over ``space = (m_tiles, n_tiles, l)``; ``l`` is always the slowest axis.

    ``along``: the axis consecutive indices advance along ("m": m fastest, as a 2-D launch grid
    with ``blockIdx.x`` over m). ``swizzle``: cut the other axis into bands of this many tiles
    and finish one band before the next; inside a band its tiles are innermost, so ``k * swizzle``
    consecutive indices touch ``swizzle`` operand panels on the banded axis instead of
    ``k * swizzle`` (CUTLASS raster + swizzle). The last band may be narrower. ``reverse``: axes
    visited high to low (``reverse="m"`` is longest-processing-time-first for causal attention).

    ``decode`` uses only ``+ - * // %`` and ``min``, so it runs on Python ints (tests) and on
    device ``Int32`` values alike. Its intermediates are bounded by ``max(index, item count)``,
    so it does not overflow for any index of a space the launch admits (``TileProblem.grid``).

    ``bind`` treats a negative extent as zero and maps an empty space to ``(1, 1, 0)``: no
    items, and every divisor in ``decode`` is 1. It runs in the host part of the jit entry
    point, so kernels pay nothing for it.
    """

    along: str = "m"
    swizzle: int = 1
    reverse: tuple[str, ...] = ()
    words = 3

    def __post_init__(self):
        if self.along not in ("m", "n"):
            raise ValueError(f"Raster.along must be 'm' or 'n', got {self.along!r}")
        if type(self.swizzle) is not int or self.swizzle < 1:
            raise ValueError(f"Raster.swizzle must be a positive int, got {self.swizzle!r}")
        if isinstance(self.reverse, str):
            object.__setattr__(self, "reverse", (self.reverse,))
        bad = [a for a in self.reverse if a not in AXES]
        if bad:
            raise ValueError(f"Raster.reverse axes must be among {AXES}, got {bad}")

    def bind(self, space):
        if len(space) != 3:
            raise ValueError(f"Raster space is (m_tiles, n_tiles, l), got {space!r}")
        if all(isinstance(e, int) for e in space):  # static extents: the same rule on the host
            if min(space) <= 0:
                return (1, 1, 0)
            return tuple(min(e, INT32_MAX) for e in space)
        em, en, el = (Int32(e) for e in space)
        # INT32_MAX when every extent is positive, else 0 (clamping the space to (1, 1, 0)).
        cap = cutlass.max(cutlass.min(cutlass.min(cutlass.min(em, en), el), 1), 0) * INT32_MAX
        return tuple(cutlass.max(cutlass.min(e, cap), lo) for e, lo in ((em, 1), (en, 1), (el, 0)))

    def size(self, space):
        em, en, el = space
        return em * en * el

    def size64(self, space):
        # Saturates at 2**31 so neither this nor the launch check's arithmetic wraps Int64.
        em, en, el = space
        limit = INT32_MAX + 1
        if all(isinstance(e, int) for e in space):  # static extents: plain host arithmetic
            return min(min(em * en, limit) * el, limit)
        return cutlass.min(cutlass.min(Int64(em) * en, limit) * el, limit)

    def decode(self, index, space):
        em, en, el = space
        per_l = em * en
        li = index // per_l
        r = index - li * per_l
        e_along, e_band = (em, en) if self.along == "m" else (en, em)
        if self.swizzle == 1:
            b = r // e_along
            a = r - b * e_along
        else:
            # A band never holds more than the whole layer, so this stays within the count.
            band_tiles = _min(self.swizzle, e_band) * e_along
            start = (r // band_tiles) * self.swizzle
            width = _min(self.swizzle, e_band - start)
            local = r - start * e_along
            a = local // width
            b = start + (local - a * width)
        m, n = (a, b) if self.along == "m" else (b, a)
        if "m" in self.reverse:
            m = em - 1 - m
        if "n" in self.reverse:
            n = en - 1 - n
        if "l" in self.reverse:
            li = el - 1 - li
        return m, n, li

    def get_key(self) -> str:
        return f"{self.along}{self.swizzle}" + (
            "r" + "".join(self.reverse) if self.reverse else ""
        )

    def describe(self) -> str:
        parts = [f"along={self.along}"]
        if self.swizzle > 1:
            parts.append(f"swizzle={self.swizzle}")
        if self.reverse:
            parts.append(f"reverse={self.reverse}")
        return f"Raster({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# Readiness hooks
# --------------------------------------------------------------------------- #


class Readiness:
    """Public protocol for ``TileStream(ready=...)``: ``wait()`` runs on the device once,
    wherever the stream decodes (each role in local mode, the scheduler warp in broadcast mode),
    before the first ``decode``/``skip``. It must return in every warp that calls it.
    ``get_key()`` (host) names it in the compile key."""

    def wait(self):
        raise NotImplementedError

    def get_key(self) -> str:
        return type(self).__name__.lower() + _field_key(self)


@dataclass(frozen=True)
class PdlWait(Readiness):
    """Readiness hook: ``griddepcontrol.wait`` before the first mapping-dependent read, for
    mappings that read tables written by the preceding kernel under programmatic dependent
    launch."""

    def wait(self):
        cute.arch.griddepcontrol_wait()

    def get_key(self) -> str:
        return "pdl"


# --------------------------------------------------------------------------- #
# Policies: how a CTA gets its next linear index
# --------------------------------------------------------------------------- #


class Policy:
    """Public protocol for ``TileStream(policy=...)``: how a work unit gets its next linear index.

    A work unit is a CTA, or a cluster of a ``CtaPair``. Policies never see tiles: they return
    indices that the mapping decodes. Every index a policy returns is a valid index in
    ``[0, problem.total)`` or a terminal value ``>= problem.total`` (never negative); once a unit
    returns a terminal index its stream has ended.

    Class attributes:

    * ``persistent``: a unit may run more than one item, so every managed role must loop. With
      ``persistent = False`` a unit runs at most one item: ``step`` must return a terminal index
      (the cursor keeps the first item's words), and raw roles may use ``work_item()``.
    * ``broadcast``: ``False`` computes indices locally, in every role, with ``first``/``step``;
      ``True`` claims them on a layer-owned scheduler warp with ``claim`` and broadcasts the
      decoded items (leader CTA only in a pair). The layer owns the ring, its counts, the drain,
      and the publication to both CTAs of a pair.
    * ``stages``: work-ring depth of a broadcast policy (decoded items buffered ahead of the
      slowest role); a positive int.
    * ``uses_max_ctas``: ``grid_size`` reads the ``max_ctas`` cap (``TileProblem.grid`` then
      requires one).
    * ``uses_workspace``: device code reads ``problem.workspace`` (the int32 tensor from
      ``launch_args``).

    Host methods (the ``@cute.jit`` entry point; ``total`` is the item count):

    * ``grid_size(total, max_units)``: launched work units; at least 1. ``max_units`` is the cap
      in work units, or ``None`` without ``uses_max_ctas``.
    * ``reach(total, units, cs)``: the largest grid x extent or index the launch computes, from
      the Int64 ``total`` (``WorkMapping.size64``); ``TileProblem.grid`` launches a zero-wide
      grid (which CUDA rejects) when it exceeds Int32. Override it when the default
      (``total + units - 1``, a strided terminal index) does not bound your policy.
    * ``get_key()``: compile-key fragment naming every compile-time field, in lowercase letters,
      digits and underscores. The default is the lowercased class name, ``stages`` and each other
      dataclass field as ``_<name><value>``; override it when a value is not a short token.
      ``describe()``.
    * ``fetch_channels()`` (broadcast): ``{name: Channel}`` of extra layer-owned channels whose
      producer and consumer ends the scheduler warp both hold (CLC's response slot).

    Device methods (``problem`` is the kernel's ``TileProblem``: ``total``, ``space``,
    ``workspace``, ``cs``, ``units()``, ``first_unit()``):

    * ``first(problem)``: this unit's first index. It runs in every role (and the scheduler), so
      it may depend only on the launch (``blockIdx``, ``gridDim``) and ``problem``. Default:
      ``problem.first_unit()``.
    * ``step(index, problem)`` (local): the next index after a valid ``index``; the same pure
      function in every role.
    * ``claim(fetch, problem, index)`` (broadcast): claim the next index on the scheduler warp.
      ``index`` is the unit's previous index (valid or skipped), ``fetch`` maps each
      ``fetch_channels`` name to its ``(producer end, consumer end)``. Called only when a
      ring slot is free and the previous claim was valid.
    * ``finish(fetch, problem)`` (broadcast): once, after the unit's last claim.

    All values are Int32. Extents past Int32 are rejected at launch through ``reach``.
    """

    broadcast = False
    persistent = True
    stages = 0
    uses_max_ctas = True
    uses_workspace = False

    def grid_size(self, total, max_units):
        if max_units is None:  # uses_max_ctas = False: one unit per item
            return cutlass.max(total, 1)
        return cutlass.max(cutlass.min(total, max_units), 1)

    def reach(self, total, units, cs: int):
        """Largest launch-grid x extent or linear index the launch computes, from the Int64 item
        count ``total`` and ``units`` launched work units: the last unit's terminal index
        overshoots ``total`` by less than the stride (``units``)."""
        return total + units - 1

    def get_key(self) -> str:
        return f"{type(self).__name__.lower()}{self.stages}" + _field_key(self, skip=("stages",))

    def describe(self) -> str:
        return type(self).__name__ + (f"(stages={self.stages})" if self.broadcast else "()")

    def fetch_channels(self) -> dict:
        """Extra layer-owned channels the scheduler warp holds both ends of."""
        return {}

    def first(self, problem):
        return problem.first_unit()

    def step(self, index, problem):
        raise NotImplementedError

    def claim(self, fetch, problem, index):
        raise NotImplementedError

    def finish(self, fetch, problem):
        pass


@dataclass(frozen=True)
class OneShot(Policy):
    """One item per CTA, ``grid = total`` (the classic non-persistent launch)."""

    persistent = False
    uses_max_ctas = False

    def grid_size(self, total, max_units):
        return cutlass.max(total, 1)

    def reach(self, total, units, cs: int):
        return total * cs  # the 1-D grid; the terminal index is ``total``

    def step(self, index, problem):
        return problem.total


@dataclass(frozen=True)
class StaticPersistent(Policy):
    """``min(total, max_ctas)`` CTAs; CTA ``b`` runs ``b, b + grid, b + 2 grid, ...``."""

    def step(self, index, problem):
        return index + problem.units()


@cute.jit
def _claim_atomic(workspace, units):
    old = Int32(0)
    if cute.arch.lane_idx() == 0:
        old = cute.arch.atomic_add(workspace.iterator, Int32(1), scope="gpu")
    return cute.arch.shuffle_sync(old, 0) + units


@cute.jit
def _report_finished(workspace, units):
    # The same lane made every claim of this CTA. The first fence orders those claims before
    # the report; the last reporter's fence orders all reports before its reset, so the reset
    # is coherence-ordered after every claim of this launch.
    if cute.arch.lane_idx() == 0:
        claims = workspace.iterator
        cute.arch.fence_acq_rel_gpu()
        done = cute.arch.atomic_add(claims + 1, Int32(1), scope="gpu")
        grid = units  # one reporting scheduler per work unit (CTA or cluster leader)
        if done == grid - 1:
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_exch(claims, Int32(0), scope="gpu")
            cute.arch.atomic_exch(claims + 1, Int32(0), scope="gpu")
        elif done >= grid:  # workspace dirty or shared with a concurrent launch
            cute.arch.inline_ptx("trap;", read_only_args=[])


@dataclass(frozen=True)
class DynamicPersistent(Policy):
    """``min(total, max_ctas)`` CTAs; after its first item (``blockIdx.x``) a CTA claims
    ``gridDim.x + atomicAdd(workspace[0], 1)``. ``stages``: decoded items buffered ahead of the
    slowest role. Needs an int32 ``[2]`` zeroed workspace (see the module docstring)."""

    stages: int = 2
    broadcast = True
    uses_workspace = True

    def claim(self, fetch, problem, index):
        return _claim_atomic(problem.workspace, problem.units())

    def finish(self, fetch, problem):
        _report_finished(problem.workspace, problem.units())


@dataclass(frozen=True)
class Clc(Policy):
    """Blackwell cluster launch control: ``grid = total``; a running CTA cancels a not-yet-
    launched CTA with ``clusterlaunchcontrol.try_cancel`` and runs its item instead. The hardware
    balances load and yields SMs to other kernels, with no counter to manage. ``stages``: decoded
    items buffered ahead of the slowest role. In a ``CtaPair`` the leader cancels whole
    clusters (the response is not multicast; the decoded item is published to both CTAs)."""

    stages: int = 2
    broadcast = True
    uses_max_ctas = False

    def grid_size(self, total, max_units):
        return cutlass.max(total, 1)

    def reach(self, total, units, cs: int):
        return total * cs  # the grid; a failed cancel returns the terminal index ``total``

    def fetch_channels(self) -> dict:
        # The scheduler's private response slot: try_cancel completes 16 bytes on the full
        # barrier (like a TMA load). The response is read through the generic proxy and the next
        # try_cancel overwrites it through the async proxy, so the release is fenced.
        clc = Slots(1, Tma(16), Threads, payload=(Int32, (4,)), release_fence="smem_async")
        clc.internal = True  # completed by try_cancel, not by TMA copies
        return {"clc": clc}

    def claim(self, fetch, problem, index):
        prod, cons = fetch["clc"]
        h = prod.acquire(0)
        with cute.arch.elect_one():
            cute.arch.issue_clc_query(h.barrier, h.data.iterator, multicast=False)
        h.commit()  # TMA-like producer: completed by the response bytes
        r = cons.wait(0)
        x, _, _, ok = cute.arch.clc_response(r.data.iterator)
        r.release()
        if cutlass.const_expr(problem.cs > 1):
            x = x // problem.cs  # first CTA of the canceled cluster -> cluster index
        return x * ok + problem.total * (1 - ok)

    def finish(self, fetch, problem):
        # Every query was waited, and none follows a failed one (PTX forbids that), so nothing
        # is in flight when the scheduler warp exits.
        pass


# --------------------------------------------------------------------------- #
# Declaration and runtime problem
# --------------------------------------------------------------------------- #


@dataclass
class TileStream:
    """Declaration: ``self.tiles = TileStream(order=Raster(swizzle=8), policy=Clc())``.

    ``ready``: readiness hook run once before the first decode wherever decoding happens.
    ``scheduler_warp``: warp for the scheduler role of broadcast policies (default: the lowest
    warp no role claims, usually a warpgroup padding warp).
    ``record``: when > 0, every role (and the scheduler) logs ``(index, %globaltimer_lo)`` for
    its first ``record`` items per CTA into the workspace (see ``timeline``).
    """

    order: WorkMapping = field(default_factory=Raster)
    policy: Policy = field(default_factory=OneShot)
    ready: Readiness | None = None
    scheduler_warp: int | None = None
    record: int = 0
    name: str = field(default="", init=False)
    roles: list = field(default_factory=list, init=False)  # recording order (set by the layer)
    # The declaring kernel (set on attribute assignment); its cluster fixes CTAs per work unit.
    owner: object = field(default=None, init=False, repr=False, compare=False)
    _workspaces: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        for value, protocol, arg in (
            (self.order, WorkMapping, "order"),
            (self.policy, Policy, "policy"),
            (self.ready, (Readiness, type(None)), "ready"),
        ):
            if not isinstance(value, protocol):
                names = getattr(protocol, "__name__", "Readiness or None")
                raise TypeError(
                    f"TileStream.{arg} must be a {names} (subclass the public protocol), "
                    f"got {type(value).__name__}"
                )
        policy = type(self.policy)
        needed = "claim" if self.policy.broadcast else "step"
        if getattr(policy, needed) is getattr(Policy, needed):
            mode = "broadcast" if self.policy.broadcast else "local"
            raise TypeError(f"{policy.__name__} is a {mode} policy but does not define {needed}()")
        if type(self.order.words) is not int or self.order.words < 1:
            raise ValueError("work items need at least one word (a positive int)")
        if self.scheduler_warp is not None and (
            type(self.scheduler_warp) is not int or not 0 <= self.scheduler_warp < 32
        ):
            raise ValueError(
                f"TileStream.scheduler_warp must be an int in [0, 32), got {self.scheduler_warp!r}"
            )
        if type(self.record) is not int or self.record < 0:
            raise ValueError(f"TileStream.record must be a nonnegative int, got {self.record!r}")
        if self.policy.broadcast and (
            type(self.policy.stages) is not int or self.policy.stages <= 0
        ):
            raise ValueError("broadcast policy stages must be a positive int")

    @property
    def cluster_size(self) -> int:
        """CTAs per work unit: 2 when the owning kernel declares ``CtaPair``."""
        return 1 if self.owner is None else self.owner.cluster.size

    @property
    def payload_words(self) -> int:
        """Broadcast payload per stage: ``[index, item words]`` padded to 16 bytes."""
        return -(-(1 + self.order.words) // 4) * 4

    def channel(self) -> Ring:
        """The broadcast ring. In a pair the leader's scheduler fills both CTAs' slots with
        ``st.async`` (TMA-like: one armed arrival plus payload bytes per CTA)."""
        payload = (Int32, (self.payload_words,))
        if self.cluster_size == 1:
            return Ring(self.policy.stages, Threads, Threads, payload=payload)
        ring = Ring(self.policy.stages, Tma(4 * self.payload_words), Threads, payload=payload)
        ring.internal = True  # leader-only producer filling consumer CTAs (no multicast TMA)
        return ring

    def describe(self) -> str:
        ready = f", ready={type(self.ready).__name__}" if self.ready else ""
        desc = getattr(self.order, "describe", lambda: type(self.order).__name__)()
        return f"{self.policy.describe()}, {desc}{ready}"

    def get_key(self) -> str:
        """Compile-key fragment: only compile-time choices (never extents or SM counts)."""
        key = f"{self.policy.get_key()}_{self.order.get_key()}"
        ready = f"_{self.ready.get_key()}" if self.ready else ""
        return key + ready + (f"_rec{self.record}" if self.record else "")

    # ---- host, inside the @cute.jit entry point ---------------------------- #

    def problem(self, space, workspace=None, max_ctas=None) -> TileProblem:
        """Bind the runtime space (``Raster``: ``(m_tiles, n_tiles, l)``). ``workspace``: int32
        ``[2]`` zeroed tensor, read only by ``DynamicPersistent``. ``max_ctas``: persistent grid
        cap (runtime, typically the SM count)."""
        if workspace is None and self.policy.uses_workspace:
            raise ValueError(f"{self.policy.describe()} needs a workspace tensor")
        if workspace is None and self.record:
            raise ValueError("TileStream(record=...) needs a workspace tensor")
        return TileProblem(self, self.order.bind(space), workspace, max_ctas)

    # ---- device, inside a raw role ------------------------------------------ #

    def work_item(self):
        """``valid, item = self.tiles.work_item()``: this CTA's work item for a raw role, which
        takes no ``Tiles`` cursor. Only for non-persistent policies (``OneShot``), where a unit
        has at most one item. It runs the readiness hook and the skip predicate like a managed
        role and reads the item the way the cursor does, so it stays correct on the 1-D
        fallback grid, a hand-built ``(total, 1, 1)`` grid, a ``CtaPair``, and custom
        mappings. ``valid`` is false for an empty space or a skipped item. Call it after
        ``begin_roles``."""
        if self.policy.persistent or self.policy.broadcast:
            raise ProtocolError(
                f"TileStream {self.name or '?'}: work_item() needs a non-persistent local policy "
                f"(OneShot), not {self.policy.describe()}; a role that runs several items takes "
                "`tiles: Tiles` and loops"
            )
        problems = getattr(self.owner, "_ws_problems", None) or {}
        if self.name not in problems:
            raise ProtocolError(
                f"TileStream {self.name or '?'}: call work_item() inside a role, after "
                "begin_roles()"
            )
        problem = problems[self.name]
        if self.ready is not None:
            self.ready.wait()
        index = Int32(self.policy.first(problem))
        if self.order.skips:
            index = Int32(_skip_local(self, index, problem))
        words = problem.first_words if _tile_grid(self) else _decode(self, index, problem)
        return index < problem.total, self.order.item(list(words))

    # ---- host, plain Python ------------------------------------------------ #

    def launch_args(self, device, max_ctas: int | None = None):
        """``(workspace, max_ctas)`` for the compiled entry point. One cached workspace per
        (device, current stream): launches on one stream run in order, so they may share it.
        With ``record``, ``max_ctas`` must cover the launch grid in CTAs (CLC/OneShot: the item
        count times the cluster size); it is rounded up to whole clusters."""
        import torch

        device = torch.device(device)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        words = 4
        if self.record:
            if not self.roles:
                raise ValueError(
                    "TileStream(record=...) sizes its log by role count: compile the kernel "
                    "(or call describe()) before launch_args()"
                )
            cs = self.cluster_size
            ctas = -(-(max_ctas or sms) // cs) * cs  # the grid launches whole clusters
            words += len(self.roles) * ctas * self.record * 2
        stream = torch.cuda.current_stream(device).cuda_stream
        ws = self._workspaces.get((device, stream))
        if ws is None or ws.numel() < words:
            ws = torch.zeros(words, dtype=torch.int32, device=device)
            self._workspaces[(device, stream)] = ws
        if self.record:
            ws[4:].fill_(-1)
        return ws, max_ctas or sms

    def timeline(self, workspace, ctas: int) -> dict:
        """Decode a ``record`` workspace: ``{role: [[(index, t_ns), ...] per CTA]}`` with times
        relative to the earliest event (``%globaltimer_lo`` wraps every ~4.3 s)."""
        rec = (
            workspace[4 : 4 + len(self.roles) * ctas * self.record * 2]
            .view(len(self.roles), ctas, self.record, 2)
            .cpu()
            .long()
        )
        valid = rec[..., 0] >= 0
        t = rec[..., 1] & 0xFFFFFFFF
        t0 = t[valid].min() if valid.any() else 0
        t = (t - t0) % (1 << 32)
        out = {}
        for r, name in enumerate(self.roles):
            out[name] = [
                [
                    (int(rec[r, b, k, 0]), int(t[r, b, k]))
                    for k in range(self.record)
                    if valid[r, b, k]
                ]
                for b in range(ctas)
            ]
        return out


class TileProblem:
    """Runtime work space, passed to the kernel as a DSL value: the mapping's space and, for
    ``DynamicPersistent``, the workspace. ``max_ctas`` stays on the host."""

    def __init__(self, stream: TileStream, space, workspace, max_ctas):
        self.stream, self.space, self.workspace, self.max_ctas = stream, space, workspace, max_ctas
        self.total = stream.order.size(space)
        self.cs = stream.cluster_size  # CTAs per work unit

    def units(self):
        """Device-side: launched work units along x (CTAs, or clusters of a pair); the
        persistent policies' stride. A ``OneShot`` tile grid has more units along y and z."""
        grid = cute.arch.grid_dim()[0]
        return grid if self.cs == 1 else grid // self.cs

    def first_unit(self):
        """Device-side: this CTA's work unit (the cluster index of a pair), as a linear index."""
        bidx, bidy, bidz = cute.arch.block_idx()
        unit = bidx if self.cs == 1 else bidx // self.cs
        if _tile_grid(self.stream):
            em, en, _ = self.space
            unit = unit + em * (bidy + en * bidz)
        return unit

    def start(self):
        """Device-side, in the kernel prologue (``begin_roles``): the first item's words of a
        ``_tile_grid`` stream, shared by every role so the fallback branch stays off their
        paths."""
        self.first_words = self.grid_item_words() if _tile_grid(self.stream) else None

    def grid_item_words(self):
        """Device-side: this CTA's item words for a ``_tile_grid`` stream, from ``blockIdx``
        when the launch is the tile grid, decoded from the 1-D index when it fell back."""
        bidx, bidy, bidz = cute.arch.block_idx()
        unit = bidx if self.cs == 1 else bidx // self.cs
        return list(
            _grid_item(self.stream.order, self.space, Int32(unit), Int32(bidy), Int32(bidz))
        )

    def _uses_workspace(self) -> bool:
        return self.stream.policy.uses_workspace or bool(self.stream.record)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.space)
        self._n_space = len(values)
        if self._uses_workspace():
            values += cutlass.extract_mlir_values(self.workspace)
        return values

    def __new_from_mlir_values__(self, values):
        n = getattr(self, "_n_space", len(cutlass.extract_mlir_values(self.space)))
        space = cutlass.new_from_mlir_values(self.space, values[:n])
        ws = None
        if self._uses_workspace():
            ws = cutlass.new_from_mlir_values(self.workspace, values[n:])
        new = TileProblem(self.stream, space, ws, None)
        # Prologue values (begin_roles) dominate every later region, so they carry as is.
        new.first_words = getattr(self, "first_words", None)
        return new

    def grid(self):
        """Launch grid for this policy: ``(ctas, 1, 1)``, or the tile space for a ``OneShot``
        over plain ``Raster()`` (``tile_grid_dims``); ``max_ctas`` caps CTAs, so a pair launches
        at most ``max_ctas // 2`` clusters. An empty space launches one work unit. A space whose
        ``policy.reach`` exceeds Int32 gets a zero-wide grid, which CUDA rejects at launch."""
        policy = self.stream.policy
        cap = None
        if policy.uses_max_ctas:
            if self.max_ctas is None:
                raise ValueError(f"{policy.describe()} needs max_ctas")
            cap = self.max_ctas if self.cs == 1 else cutlass.max(self.max_ctas // self.cs, 1)
        units = policy.grid_size(self.total, cap)
        if _tile_grid(self.stream):
            x, y, z = tile_grid_dims(self.space, self.cs)
        else:
            x, y, z = units * self.cs, 1, 1
        # Every grid extent and index must fit Int32. Past that the grid is zero-wide, so CUDA
        # rejects the launch instead of the kernel covering a wrapped space. This is host code
        # in the jit entry point: kernels pay nothing for the check.
        size = self.stream.order.size64(self.space)
        if isinstance(size, int) and isinstance(units, int):  # static extents: plain host ints
            return (x if policy.reach(size, units, self.cs) <= INT32_MAX else 0), y, z
        reach = policy.reach(size, Int64(units), self.cs)
        return Int32(cutlass.select_(reach <= INT32_MAX, x, Int32(0))), y, z


# --------------------------------------------------------------------------- #
# Device side: decoding, the per-role cursor, and the scheduler role
# --------------------------------------------------------------------------- #


def _decode(stream, index, problem):
    return [Int32(w) for w in stream.order.decode(index, problem.space)]


@cute.jit
def _skipped(stream, index, problem):
    """``skip(item)`` for a valid index; false past the end (skip may read per-item tables)."""
    skipped = cutlass.Boolean(False)
    if index < problem.total:
        order = stream.order
        skipped = cutlass.Boolean(
            order.skip(order.item(_decode(stream, index, problem)), problem.space)
        )
    return skipped


@cute.jit
def _skip_local(stream, index, problem):
    """First index at or after ``index`` (in this CTA's local sequence) the mapping keeps."""
    step = stream.policy.step  # a method call on `stream` in the loop would carry it
    while _skipped(stream, index, problem):
        index = Int32(step(index, problem))
    return index


@cute.jit
def _record(workspace, slot, index, total, count, cap: int):
    role, warp = slot
    first_lane = (cute.arch.lane_idx() == 0) & (cute.arch.warp_idx() == warp)
    if first_lane & (index < total) & (count < cap):
        bidx, _, _ = cute.arch.block_idx()
        off = 4 + ((role * cute.arch.grid_dim()[0] + bidx) * cap + count) * 2
        workspace[off] = index
        workspace[off + 1] = cute.arch.globaltimer_lo()


@dataclass
class _CursorRecord:
    """Trace-time record shared by loop-carried copies of one cursor."""

    role: str
    advances: list = field(default_factory=list)  # (site, block)
    valid_reads: list = field(default_factory=list)  # (block, valid SSA value identity)
    trace: object = None

    def check(self):
        if not self.advances:
            raise ProtocolError(
                f"{self.role}: takes Tiles but never calls advance(); loop with "
                "`while tiles.valid: ...; tiles.advance()`"
            )
        if len(self.advances) > 1:
            sites = ", ".join(s for s, _ in self.advances)
            raise ProtocolError(
                f"{self.role}: advance() traced {len(self.advances)} times ({sites}); call it "
                "exactly once, unconditionally, at the end of the tile loop body"
            )
        site, block = self.advances[0]
        conditions = [
            (cond, value) for cond, value in self.valid_reads if _is_while_body(block, cond)
        ]
        if not conditions:
            raise ProtocolError(
                f"{self.role}: advance() at {site} must run once per iteration of a "
                "`while tiles.valid:` loop: call it directly in that loop's body, not under a "
                "nested branch or loop"
            )
        if not any(hash(cond.operations[-1].operands[0]) == value for cond, value in conditions):
            raise ProtocolError(
                f"{self.role}: the tile-loop condition must be exactly tiles.valid; "
                "an extra condition can leave the stream undrained"
            )


def _is_while_body(block, cond_block) -> bool:
    """``block`` is the body (after region) of the ``scf.while`` whose condition (before
    region) is ``cond_block``."""
    op = block.owner
    return (
        op.name == "scf.while"
        and block == op.regions[1].blocks[0]
        and cond_block == op.regions[0].blocks[0]
    )


MAX_GRID_YZ = 65535  # gridDim.y and gridDim.z limit


def tile_grid_dims(space, cs: int = 1):
    """Launch grid of a ``_tile_grid`` stream over ``space = (m_tiles, n_tiles, l)``: the tile
    space when ``n_tiles`` and ``l`` fit ``gridDim.y``/``z``, else the 1-D grid of ``total``
    items; one work unit when the space is empty (an extent <= 0). Works on Python ints and on
    DSL extents."""
    em, en, el = space
    if all(isinstance(e, int) for e in space):
        if min(space) <= 0:
            return (cs, 1, 1)
        if en <= MAX_GRID_YZ and el <= MAX_GRID_YZ:
            return (em * cs, en, el)
        return (em * en * el * cs, 1, 1)
    nonempty = (em > 0) & (en > 0) & (el > 0)
    tiles = nonempty & (en <= MAX_GRID_YZ) & (el <= MAX_GRID_YZ)
    one = Int32(1)
    return (
        Int32(cutlass.select_(tiles, em, cutlass.select_(nonempty, em * en * el, one))) * cs,
        Int32(cutlass.select_(tiles, en, one)),
        Int32(cutlass.select_(tiles, el, one)),
    )


@cute.jit
def _grid_item(order, space, m, n, li):
    """Item words of a ``_tile_grid`` stream: ``blockIdx`` as is on a tile-space grid, or the
    decoded index on any 1-D grid (the fallback, or a hand-built ``(total, 1, 1)`` launch, where
    ``m`` is the linear index and ``n = l = 0``). The launch shape decides, not the extents, so
    both launches are correct; with ``n_tiles = l = 1`` the two grids coincide."""
    decode = order.decode  # a method call on `order` in the branch would carry it
    _, en, el = space
    _, gdy, gdz = cute.arch.grid_dim()
    if (gdy * gdz == 1) & (en * el > 1):
        dm, dn, dl = decode(m, space)
        m, n, li = Int32(dm), Int32(dn), Int32(dl)
    return m, n, li


def _tile_grid(stream) -> bool:
    """One item per CTA in plain raster order: launch the tile space as the grid and take the item
    from ``blockIdx``. Decoding a 1-D index divides by the extents, which sits on the load warp's
    path to its first copy (about 0.2 us on a 512^3 GEMM)."""
    order = stream.order
    return (
        type(stream.policy) is OneShot  # a subclass may override first()
        and type(order) is Raster
        and order.along == "m"
        and order.swizzle == 1
        and not order.reverse
        and not stream.record  # the recording log indexes CTAs by blockIdx.x
    )


def _hooked(stream) -> bool:
    """Hooks can move a CTA's first item off ``policy.first``."""
    return stream.ready is not None or stream.order.skips


class TileCursor:
    """One role's position in the work stream: ``valid``, ``work``, ``index``, ``advance()``.

    A DSL value carrying the linear index, the decoded item words, and, in broadcast mode, the
    role's consumer end of the work ring.
    """

    def __init__(self, stream, problem, end, index, words, rec, count=None, slot=None, iket=None):
        self._stream, self._problem, self._end, self._rec = stream, problem, end, rec
        self.index, self._words = index, words
        self._count, self._slot = count, slot  # recording: items seen, (role ordinal, warp)
        self._iket = iket  # IKET range name of one work item (the role name), or None

    @staticmethod
    def create(stream, problem, end, trace, *, scheduler: bool = False, slot=None):
        """Local mode: run the hooks and decode here. Broadcast mode: start at ``policy.first``
        decoded locally, or, when hooks may move the first item, receive it from the scheduler
        (which publishes it in ``scheduler_role``)."""
        rec = _CursorRecord(trace.role_name, trace=trace)
        trace.checks.append(rec.check)
        count = Int32(0) if stream.record else None
        iket = trace.role_name if trace.iket else None
        cur = TileCursor(
            stream, problem, end, Int32(stream.policy.first(problem)), [], rec, count, slot, iket
        )
        if end is None:
            if stream.ready is not None:
                stream.ready.wait()
            if stream.order.skips:
                cur.index = Int32(_skip_local(stream, cur.index, problem))
        elif _hooked(stream):
            if scheduler:
                # scheduler_role runs the hooks, then publishes and receives the first item
                # (and opens its range through _first_item()).
                cur._words = [Int32(0)] * stream.order.words
                return cur
            cur._receive()
            cur._first_item()
            return cur
        if end is None and _tile_grid(stream):
            cur._words = list(problem.first_words)
        else:
            cur._words = _decode(stream, cur.index, problem)
        cur._first_item()
        return cur

    def _first_item(self):
        self._log()
        if self._iket is not None:
            push_if(self.index < self._problem.total, self._iket)

    def _log(self):
        if self._count is not None:
            _record(
                self._problem.workspace,
                self._slot,
                self.index,
                self._problem.total,
                self._count,
                self._stream.record,
            )
            self._count = self._count + 1

    def __extract_mlir_values__(self):
        values = [self.index.ir_value()] + [w.ir_value() for w in self._words]
        if self._count is not None:
            values.append(self._count.ir_value())
        if self._end is not None:
            values += self._end.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values):
        n = 1 + len(self._words)
        words = [Int32(v) for v in values[1:n]]
        count = None
        if self._count is not None:
            count, n = Int32(values[n]), n + 1
        end = None if self._end is None else self._end.__new_from_mlir_values__(values[n:])
        return TileCursor(
            self._stream,
            self._problem,
            end,
            Int32(values[0]),
            words,
            self._rec,
            count,
            self._slot,
            self._iket,
        )

    @property
    def valid(self):
        valid = self.index < self._problem.total
        # Keep SSA identity opaque: a raw Value in a shared dataclass becomes loop-carried
        # state during CuTeDSL serialization, rather than trace-only bookkeeping.
        self._rec.valid_reads.append((current_block(), hash(valid.ir_value())))
        return valid

    @property
    def work(self):
        """The current work item (``Raster``: ``(m, n, l)`` tile coordinates)."""
        block = current_block()
        while block is not None:
            if any(block == advanced for _, advanced in self._rec.advances):
                raise ProtocolError(
                    f"{self._rec.role}: tiles.work read after advance() in the same iteration "
                    f"at {user_site()}; read the current item before advancing"
                )
            owner = block.owner
            block = owner.operation.block if owner.name != "builtin.module" else None
        return self._stream.order.item(self._words)

    def advance(self):
        """Move to this CTA's next work item (``valid`` turns false when there is none)."""
        self._rec.advances.append((user_site(), current_block()))
        self._rec.trace.item_block = current_block()
        if self._iket is not None:
            pop()  # this item's range
        stream, problem = self._stream, self._problem
        if self._end is None:
            self.index = Int32(stream.policy.step(self.index, problem))
            if stream.order.skips:
                self.index = Int32(_skip_local(stream, self.index, problem))
            if stream.policy.persistent:  # OneShot: the stream has ended, keep the words
                self._words = _decode(stream, self.index, problem)
        else:
            self._receive()
        self._log()
        if self._iket is not None:
            push_if(self.index < problem.total, self._iket)

    def _receive(self):
        h = self._end.wait()
        self.index = Int32(h.data[0])
        self._words = [Int32(h.data[1 + i]) for i in range(self._stream.order.words)]
        h.release()


class _Fetch:
    """The scheduler's private policy channels (``{name: (producer end, consumer end)}``) as one
    DSL value, so a single ``fetch.claim()`` call in a loop carries every phase."""

    def __init__(self, policy, ends: dict, problem):
        self._policy, self._ends, self._problem = policy, ends, problem

    def __extract_mlir_values__(self):
        return [v for pc in self._ends.values() for e in pc for v in e.__extract_mlir_values__()]

    def __new_from_mlir_values__(self, values):
        ends = {}
        for k, pair in self._ends.items():
            new = []
            for e in pair:
                n = len(e.__extract_mlir_values__())
                new.append(e.__new_from_mlir_values__(values[:n]))
                values = values[n:]
            ends[k] = tuple(new)
        return _Fetch(self._policy, ends, self._problem)

    def claim(self, index):
        return Int32(self._policy.claim(self._ends, self._problem, index))

    def finish(self):
        self._policy.finish(self._ends, self._problem)


def _publish(h, index, words, total):
    chan = h._end._chan
    if isinstance(chan.producer, Tma):  # a cluster's ring: the bytes complete each full barrier
        targets = chan.scope.full_ctas if chan.scope.pair else (0,)  # local: leader-only users
        _publish_pair(h.data.iterator, h.barrier, [index, *words], targets)
    else:
        _publish_local(h, index, words, total)


@cute.jit
def _publish_local(h, index, words, total):
    with cute.arch.elect_one():
        h.data[0] = index
        if index < total:
            for i in cutlass.range_constexpr(len(words)):
                h.data[1 + i] = words[i]


@cute.jit
def _publish_pair(slot_ptr, full, values, targets: cutlass.Constexpr):
    """Leader: ``st.async`` the item into the slot of every consumer CTA; the stores complete
    the bytes the producer armed on each CTA's full barrier (``Tma(16 * chunks)``)."""
    words = list(values) + [Int32(0)] * (-len(values) % 4)
    base = slot_ptr.toint()
    with cute.arch.elect_one():
        for c in cutlass.range_constexpr(len(targets)):
            for q in cutlass.range_constexpr(len(words) // 4):
                dst = cute.make_ptr(Int32, base + 16 * q, cute.AddressSpace.smem, assumed_align=16)
                cute.arch.store_async_dsmem(dst, tuple(words[4 * q : 4 * q + 4]), full, targets[c])


@cute.jit
def scheduler_role(kernel, work, tiles: TileCursor, fetch: _Fetch):
    """Layer-owned scheduler warp for broadcast policies: claim, decode, publish, repeat.

    A claim is made only once a ring slot is free, so a CTA reserves at most ``stages`` items
    beyond what its slowest role has started, and only after the previous claim came back valid
    (CLC forbids querying after a failed query). The scheduler consumes its own ring like every
    other role, which keeps it in step with the CTA.
    """
    stream = tiles._stream
    problem = tiles._problem
    order = stream.order
    if cutlass.const_expr(_hooked(stream)):
        # Hooks may move the first item off blockIdx.x, so publish it like a claim.
        if cutlass.const_expr(stream.ready is not None):
            stream.ready.wait()
        with work.produce() as h0:
            first = tiles.index
            if cutlass.const_expr(order.skips):
                while _skipped(stream, first, problem):
                    first = fetch.claim(first)
            _publish(h0, first, _decode(stream, first, problem), problem.total)
        tiles._receive()
        tiles._first_item()
    while tiles.valid:
        with work.produce() as h:
            index = fetch.claim(tiles.index)
            if cutlass.const_expr(order.skips):
                while _skipped(stream, index, problem):
                    index = fetch.claim(index)
            _publish(h, index, _decode(stream, index, problem), problem.total)
        tiles.advance()
    fetch.finish()
    return work, tiles
