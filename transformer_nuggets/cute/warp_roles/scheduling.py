"""Work scheduling for warp-role kernels: which CTA runs which work item, in what order.

Three independent, compile-time choices:

* **policy**: how a CTA obtains its next *linear work index*. Policies know nothing about
  tiles.

  ============================  ========================================  =======================
  policy                        next index                                launch grid
  ============================  ========================================  =======================
  ``OneShot()``                 none (one item per CTA)                   ``total``
  ``StaticPersistent()``        ``index + gridDim.x``                     ``min(total, max_ctas)``
  ``DynamicPersistent(stages)`` ``gridDim.x + atomicAdd(counter, 1)``     ``min(total, max_ctas)``
  ``Clc(stages)``               ``clusterlaunchcontrol.try_cancel``       ``total``
  ============================  ========================================  =======================

* **order** (a work mapping): ``index -> work item``, a tuple of Int32 words that ``item()``
  turns into whatever the roles consume. ``Raster`` is the tile mapping (row/column raster, L2
  swizzle bands, reversed axes for longest-processing-time-first). The mapping owns its runtime
  ``space`` (``Raster``: tile extents), the item count, and an optional ``skip(item)`` predicate
  for empty items.
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

Counter lifetime (``DynamicPersistent``): the workspace holds ``[claims, finished]``. Each CTA's
scheduler reports once after its last claim and the last CTA to report zeroes both words, so
launches ordered on one stream (including CUDA-graph replays) reuse one workspace with no host
reset. A workspace must not be shared by concurrently running launches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cutlass
from cutlass import Int32, cute

from transformer_nuggets.cute.warp_roles.channels import Ring, Slots, Threads, Tma
from transformer_nuggets.cute.warp_roles.protocol import ProtocolError, current_block, user_site

AXES = ("m", "n", "l")


class Tiles:
    """Annotation: the role receives a ``TileCursor`` over the stream with the parameter's name."""


def _min(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return cutlass.min(a, b)


# --------------------------------------------------------------------------- #
# Work mappings: linear index -> work item
# --------------------------------------------------------------------------- #


class WorkMapping:
    """Protocol for ``order=``. ``words``: Int32 words per item. ``bind(space)`` normalizes the
    runtime space (a DSL value), ``size(space)`` counts items, ``decode(index, space)`` returns
    ``words`` Int32 values, and ``item(words)`` builds what ``tiles.work`` returns.

    ``decode`` may be called with ``index >= size`` (its result is then ignored) and must not
    fault there. ``skip(item, space)`` (optional, only called for valid indices) permanently
    removes items no role should see; it is not a readiness wait.
    """

    words: int = 1
    skips = False

    def bind(self, space):
        return space

    def size(self, space):
        raise NotImplementedError

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
    device ``Int32`` values alike.
    """

    along: str = "m"
    swizzle: int = 1
    reverse: tuple[str, ...] = ()
    words = 3

    def __post_init__(self):
        if self.along not in ("m", "n"):
            raise ValueError(f"Raster.along must be 'm' or 'n', got {self.along!r}")
        if not isinstance(self.swizzle, int) or self.swizzle < 1:
            raise ValueError(f"Raster.swizzle must be a positive int, got {self.swizzle!r}")
        if isinstance(self.reverse, str):
            object.__setattr__(self, "reverse", (self.reverse,))
        bad = [a for a in self.reverse if a not in AXES]
        if bad:
            raise ValueError(f"Raster.reverse axes must be among {AXES}, got {bad}")

    def bind(self, space):
        if len(space) != 3:
            raise ValueError(f"Raster space is (m_tiles, n_tiles, l), got {space!r}")
        return tuple(Int32(e) for e in space)

    def size(self, space):
        em, en, el = space
        return em * en * el

    def decode(self, index, space):
        em, en, el = space
        per_l = em * en
        l = index // per_l
        r = index - l * per_l
        e_along, e_band = (em, en) if self.along == "m" else (en, em)
        if self.swizzle == 1:
            b = r // e_along
            a = r - b * e_along
        else:
            band_tiles = self.swizzle * e_along
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
            l = el - 1 - l
        return m, n, l

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


@dataclass(frozen=True)
class PdlWait:
    """Readiness hook: ``griddepcontrol.wait`` before the first decode, for mappings that read
    tables written by the preceding kernel under programmatic dependent launch."""

    def wait(self):
        cute.arch.griddepcontrol_wait()


# --------------------------------------------------------------------------- #
# Policies: how a CTA gets its next linear index
# --------------------------------------------------------------------------- #


class _Policy:
    """``broadcast``: indices are claimed by one scheduler warp and broadcast to every role."""

    broadcast = False
    persistent = True
    stages = 0

    def grid_size(self, total, max_ctas):
        return cutlass.max(cutlass.min(total, max_ctas), 1)

    def describe(self) -> str:
        return type(self).__name__ + (f"(stages={self.stages})" if self.broadcast else "()")

    def fetch_channels(self) -> dict:
        """Extra layer-owned channels the scheduler warp holds both ends of."""
        return {}


@dataclass(frozen=True)
class OneShot(_Policy):
    """One item per CTA, ``grid = total`` (the classic non-persistent launch)."""

    persistent = False

    def grid_size(self, total, max_ctas):
        return cutlass.max(total, 1)

    def step(self, index, problem):
        return problem.total


@dataclass(frozen=True)
class StaticPersistent(_Policy):
    """``min(total, max_ctas)`` CTAs; CTA ``b`` runs ``b, b + grid, b + 2 grid, ...``."""

    def step(self, index, problem):
        return index + cute.arch.grid_dim()[0]


@cute.jit
def _claim_atomic(workspace):
    old = Int32(0)
    if cute.arch.lane_idx() == 0:
        old = cute.arch.atomic_add(workspace.iterator, Int32(1), scope="gpu")
    return cute.arch.shuffle_sync(old, 0) + cute.arch.grid_dim()[0]


@cute.jit
def _report_finished(workspace):
    # The same lane made every claim of this CTA. The first fence orders those claims before
    # the report; the last reporter's fence orders all reports before its reset, so the reset
    # is coherence-ordered after every claim of this launch.
    if cute.arch.lane_idx() == 0:
        claims = workspace.iterator
        cute.arch.fence_acq_rel_gpu()
        done = cute.arch.atomic_add(claims + 1, Int32(1), scope="gpu")
        grid = cute.arch.grid_dim()[0]
        if done == grid - 1:
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_exch(claims, Int32(0), scope="gpu")
            cute.arch.atomic_exch(claims + 1, Int32(0), scope="gpu")
        elif done >= grid:  # workspace dirty or shared with a concurrent launch
            cute.arch.inline_ptx("trap;", read_only_args=[])


@dataclass(frozen=True)
class DynamicPersistent(_Policy):
    """``min(total, max_ctas)`` CTAs; after its first item (``blockIdx.x``) a CTA claims
    ``gridDim.x + atomicAdd(workspace[0], 1)``. ``stages``: decoded items buffered ahead of the
    slowest role. Needs an int32 ``[2]`` zeroed workspace (see the module docstring)."""

    stages: int = 2
    broadcast = True

    def claim(self, fetch, problem):
        return _claim_atomic(problem.workspace)

    def finish(self, fetch, problem):
        _report_finished(problem.workspace)


@dataclass(frozen=True)
class Clc(_Policy):
    """Blackwell cluster launch control: ``grid = total``; a running CTA cancels a not-yet-
    launched CTA with ``clusterlaunchcontrol.try_cancel`` and runs its item instead. The hardware
    balances load and yields SMs to other kernels, with no counter to manage. ``stages``: decoded
    items buffered ahead of the slowest role. 1-CTA clusters only (see REPORT.md)."""

    stages: int = 2
    broadcast = True

    def grid_size(self, total, max_ctas):
        return cutlass.max(total, 1)

    def fetch_channels(self) -> dict:
        # The scheduler's private response slot: try_cancel completes 16 bytes on the full
        # barrier (like a TMA load). The response is read through the generic proxy and the next
        # try_cancel overwrites it through the async proxy, so the release is fenced.
        return {
            "clc": Slots(1, Tma(16), Threads, payload=(Int32, (4,)), release_fence="smem_async")
        }

    def claim(self, fetch, problem):
        prod, cons = fetch["clc"]
        h = prod.acquire(0)
        with cute.arch.elect_one():
            cute.arch.issue_clc_query(h.barrier, h.data.iterator, multicast=False)
        h.commit()  # TMA-like producer: completed by the response bytes
        r = cons.wait(0)
        x, _, _, ok = cute.arch.clc_response(r.data.iterator)
        r.release()
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
    policy: _Policy = field(default_factory=OneShot)
    ready: PdlWait | None = None
    scheduler_warp: int | None = None
    record: int = 0
    name: str = field(default="", init=False)
    roles: list = field(default_factory=list, init=False)  # recording order (set by the layer)
    _workspaces: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.order.words < 1:
            raise ValueError("work items need at least one word")

    @property
    def payload_words(self) -> int:
        """Broadcast payload per stage: ``[index, item words]`` padded to 16 bytes."""
        return -(-(1 + self.order.words) // 4) * 4

    def channel(self) -> Ring:
        return Ring(self.policy.stages, Threads, Threads, payload=(Int32, (self.payload_words,)))

    def describe(self) -> str:
        ready = f", ready={type(self.ready).__name__}" if self.ready else ""
        desc = getattr(self.order, "describe", lambda: type(self.order).__name__)()
        return f"{self.policy.describe()}, {desc}{ready}"

    def get_key(self) -> str:
        """Compile-key fragment: only compile-time choices (never extents or SM counts)."""
        p = self.policy
        key = f"{type(p).__name__.lower()}{p.stages}_{self.order.get_key()}"
        return key + ("_pdl" if self.ready else "") + (f"_rec{self.record}" if self.record else "")

    # ---- host, inside the @cute.jit entry point ---------------------------- #

    def problem(self, space, workspace=None, max_ctas=None) -> TileProblem:
        """Bind the runtime space (``Raster``: ``(m_tiles, n_tiles, l)``). ``workspace``: int32
        ``[2]`` zeroed tensor, read only by ``DynamicPersistent``. ``max_ctas``: persistent grid
        cap (runtime, typically the SM count)."""
        if (isinstance(self.policy, DynamicPersistent) or self.record) and workspace is None:
            raise ValueError("DynamicPersistent and record need a workspace tensor")
        return TileProblem(self, self.order.bind(space), workspace, max_ctas)

    # ---- host, plain Python ------------------------------------------------ #

    def launch_args(self, device, max_ctas: int | None = None):
        """``(workspace, max_ctas)`` for the compiled entry point. One cached workspace per
        (device, current stream): launches on one stream run in order, so they may share it.
        With ``record``, ``max_ctas`` must cover the launch grid (CLC/OneShot: the item count)."""
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
            words += len(self.roles) * (max_ctas or sms) * self.record * 2
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

    def _uses_workspace(self) -> bool:
        return isinstance(self.stream.policy, DynamicPersistent) or bool(self.stream.record)

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
        return TileProblem(self.stream, space, ws, None)

    def grid(self):
        """Launch grid for this policy: ``(ctas, 1, 1)``."""
        policy = self.stream.policy
        uses_cap = isinstance(policy, (StaticPersistent, DynamicPersistent))
        if uses_cap and self.max_ctas is None:
            raise ValueError(f"{policy.describe()} needs max_ctas")
        return (policy.grid_size(self.total, self.max_ctas), 1, 1)


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
    root: object
    advances: list = field(default_factory=list)  # (site, block)
    read_valid: bool = False

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
        if block == self.root or not self.read_valid:
            raise ProtocolError(
                f"{self.role}: advance() at {site} is not inside a `while tiles.valid:` loop"
            )


def _hooked(stream) -> bool:
    """Hooks can move a CTA's first item off ``blockIdx.x``."""
    return stream.ready is not None or stream.order.skips


class TileCursor:
    """One role's position in the work stream: ``valid``, ``work``, ``index``, ``advance()``.

    A DSL value carrying the linear index, the decoded item words, and, in broadcast mode, the
    role's consumer end of the work ring.
    """

    def __init__(self, stream, problem, end, index, words, rec, count=None, slot=None):
        self._stream, self._problem, self._end, self._rec = stream, problem, end, rec
        self.index, self._words = index, words
        self._count, self._slot = count, slot  # recording: items seen, (role ordinal, warp)

    @staticmethod
    def create(stream, problem, end, trace, *, scheduler: bool = False, slot=None):
        """Local mode: run the hooks and decode here. Broadcast mode: start at ``blockIdx.x``
        decoded locally, or, when hooks may move the first item, receive it from the scheduler
        (which publishes it in ``scheduler_role``)."""
        rec = _CursorRecord(trace.role_name, current_block())
        trace.checks.append(rec.check)
        bidx, _, _ = cute.arch.block_idx()
        count = Int32(0) if stream.record else None
        cur = TileCursor(stream, problem, end, Int32(bidx), [], rec, count, slot)
        if end is None:
            if stream.ready is not None:
                stream.ready.wait()
            if stream.order.skips:
                cur.index = Int32(_skip_local(stream, cur.index, problem))
        elif _hooked(stream):
            if scheduler:
                # scheduler_role runs the hooks, then publishes and receives the first item.
                cur._words = [Int32(0)] * stream.order.words
                return cur
            cur._receive()
            cur._log()
            return cur
        cur._words = _decode(stream, cur.index, problem)
        cur._log()
        return cur

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
            self._stream, self._problem, end, Int32(values[0]), words, self._rec, count, self._slot
        )

    @property
    def valid(self):
        self._rec.read_valid = True
        return self.index < self._problem.total

    @property
    def work(self):
        """The current work item (``Raster``: ``(m, n, l)`` tile coordinates)."""
        return self._stream.order.item(self._words)

    def advance(self):
        """Move to this CTA's next work item (``valid`` turns false when there is none)."""
        self._rec.advances.append((user_site(), current_block()))
        stream, problem = self._stream, self._problem
        if self._end is None:
            self.index = Int32(stream.policy.step(self.index, problem))
            if stream.order.skips:
                self.index = Int32(_skip_local(stream, self.index, problem))
            self._words = _decode(stream, self.index, problem)
        else:
            self._receive()
        self._log()

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

    def claim(self):
        return Int32(self._policy.claim(self._ends, self._problem))

    def finish(self):
        self._policy.finish(self._ends, self._problem)


@cute.jit
def _publish(h, index, words, total):
    with cute.arch.elect_one():
        h.data[0] = index
        if index < total:
            for i in cutlass.range_constexpr(len(words)):
                h.data[1 + i] = words[i]


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
                    first = fetch.claim()
            _publish(h0, first, _decode(stream, first, problem), problem.total)
        tiles._receive()
        tiles._log()
    while tiles.valid:
        with work.produce() as h:
            index = fetch.claim()
            if cutlass.const_expr(order.skips):
                while _skipped(stream, index, problem):
                    index = fetch.claim()
            _publish(h, index, _decode(stream, index, problem), problem.total)
        tiles.advance()
    fetch.finish()
