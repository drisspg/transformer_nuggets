"""Cluster declarations and the derivation of cross-CTA channel scopes.

A kernel declares its cluster once: ``self.cluster = CtaPair()`` (the default is
``SingleCta()``), or ``Cluster(m, n, pair=...)`` for any other shape. A cluster is ``m x n`` CTAs
(``m`` along the output's M, ``n`` along N) that share each work item. With ``pair=True`` each two
consecutive CTAs along M form an *MMA pair* running ``tcgen05`` ``CtaGroup.TWO`` MMAs: the pair's
leader (its even rank) issues every MMA, which reads A and B from the shared memory of both CTAs
and writes each CTA's half of the accumulator into that CTA's tensor memory. ``CtaPair()`` is
``Cluster(2, 1, pair=True)``.

CTAs share loads through TMA multicast. A ``Tma(bytes, multicast="n")`` producer's copies may
land in every CTA with this CTA's coordinates except along N (the CTAs that need the same rows
of A), ``"m"`` likewise along M (the same rows of B). ``Cluster.multicast(mode)`` gives the
partition coordinate and the copy mask of one mode.

Roles run in every CTA of the cluster unless pinned: ``cta="leader"`` runs in the leader of each
MMA pair (every CTA when the cluster has no pairs), ``cta="rank0"`` in cluster rank 0 only. From
that placement the layer derives each channel's *scope*:

* **local**: every participant runs in the same CTAs and no copy is multicast. Each CTA has its
  own independent copy of the channel.
* **cross-CTA** (*pair-scoped* in a CTA pair): participants run in different CTAs, or copies are
  multicast. A barrier lives in every CTA that has a role waiting on it, each signal reaches the
  barrier of every target CTA (remote ``mbarrier.arrive``, ``tcgen05.commit`` multicast, or a TMA
  transaction completing on the pair leader's barrier), and arrival counts and transaction bytes
  sum the contributions of every signalling CTA. A channel spans each MMA pair (each CTA without
  pairs), extended to the CTAs its multicast copies reach; with a ``rank0`` participant it spans
  the whole cluster.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import cutlass
from cutlass import Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from transformer_nuggets.cute.warpdrive.protocol import ProtocolError

ALL = "all"
LEADER = "leader"
RANK0 = "rank0"
PLACEMENTS = (ALL, LEADER, RANK0)
MULTICAST_MODES = ("m", "n")
MAX_CLUSTER_CTAS = 8  # the portable cluster size limit
# Derived ``Scope`` fields a channel may override (``Channel(override={...})``).
SCOPE_OVERRIDES = ("full_counts", "empty_counts", "tx", "full_mask", "empty_mask")


@dataclass(frozen=True)
class TmaMulticast:
    """One multicast mode of this CTA: ``cpasync.tma_partition(atom, coord, layout, ...)``
    splits each tile among the CTAs that share it, and ``cute.copy(..., mcast_mask=mask)``
    delivers this CTA's part to all of them."""

    coord: object
    layout: cute.Layout
    mask: object
    ctas: int  # CTAs sharing each tile (1: no multicast)


@dataclass(frozen=True)
class Cluster:
    """``m x n`` CTAs sharing each work item; with ``pair``, consecutive CTAs along M form 2-CTA
    MMA pairs (``CtaGroup.TWO``).

    Rank ``r`` (``block_idx_in_cluster``) has coordinates ``(v, mi, ni)`` with
    ``r = v + V * (mi + (m // V) * ni)``, ``V = 2`` with pairs and 1 without: the ``(V, M, N, K)``
    cluster layout of the CUTLASS SM100 GEMMs. The launch cluster is ``(m * n, 1, 1)`` CTAs, whose
    hardware ranks follow the same linear order, so a work unit stays a run of consecutive
    ``blockIdx.x``.
    """

    m: int = 1
    n: int = 1
    pair: bool = False

    def __post_init__(self):
        for name in ("m", "n"):
            extent = getattr(self, name)
            if type(extent) is not int or extent <= 0:
                raise ValueError(f"Cluster.{name} must be a positive int, got {extent!r}")
        if type(self.pair) is not bool:
            raise ValueError(f"Cluster.pair must be a bool, got {self.pair!r}")
        if self.pair and self.m % 2:
            raise ValueError(f"a cluster of MMA pairs needs an even m, got m={self.m}")
        if self.m * self.n > MAX_CLUSTER_CTAS:
            raise ValueError(
                f"clusters have at most {MAX_CLUSTER_CTAS} CTAs, got {self.m} x {self.n}"
            )

    @property
    def size(self) -> int:
        return self.m * self.n

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.m, self.n, 1)

    @property
    def pair_size(self) -> int:
        """CTAs per MMA (``V``): 2 with pairs, else 1."""
        return 2 if self.pair else 1

    @property
    def cta_group(self):
        return tcgen05.CtaGroup.TWO if self.pair else tcgen05.CtaGroup.ONE

    # ---- host-side geometry ------------------------------------------------ #

    def coords(self, rank):
        """``(v, mi, ni)`` of ``rank`` (a Python int, or a device rank): its CTA in the MMA pair
        and the pair's M/N position. Coordinates the shape fixes are Python ints."""
        pairs_m = self.m // self.pair_size
        v = rank if self.size == self.pair_size else rank % self.pair_size
        pair = 0 if self.size == self.pair_size else rank // self.pair_size
        return v, pair % pairs_m if pairs_m > 1 else 0, pair // pairs_m if self.n > 1 else 0

    def ctas(self, placement: str) -> tuple[int, ...]:
        """Ranks a role with this ``cta=`` placement runs in."""
        match placement:
            case "all":
                return tuple(range(self.size))
            case "leader":
                return tuple(range(0, self.size, self.pair_size))
            case "rank0":
                return (0,)
        raise ValueError(f"role cta must be one of {PLACEMENTS}, got {placement!r}")

    def group(self, rank: int) -> tuple[int, ...]:
        """The MMA pair of ``rank`` (``rank`` alone without pairs)."""
        first = rank - rank % self.pair_size
        return tuple(range(first, first + self.pair_size))

    def line(self, rank: int, mode: str) -> tuple[int, ...]:
        """Ranks with ``rank``'s coordinates except along ``mode`` ("m" or "n"): the CTAs one
        multicast copy along that mode reaches."""
        v, mi, ni = self.coords(rank)
        return tuple(
            r
            for r in range(self.size)
            if self.coords(r)[0] == v
            and (mode == "m" or self.coords(r)[1] == mi)
            and (mode == "n" or self.coords(r)[2] == ni)
        )

    def reach(self, rank: int, modes: tuple[str, ...]) -> tuple[int, ...]:
        """CTAs whose SMEM the copies of ``rank`` write: itself and its line along each mode."""
        return tuple(sorted({rank}.union(*(self.line(rank, mode) for mode in modes))))

    def launch_cluster(self):
        return None if self.size == 1 else (self.size, 1, 1)

    def get_key(self) -> str:
        if self.size == 1:
            return "cta1"
        if (self.m, self.n, self.pair) == (2, 1, True):
            return "pair"
        return f"m{self.m}n{self.n}" + ("_pairs" if self.pair else "")

    def describe(self) -> str:
        if self.size == 1:
            return "single CTA, tcgen05 CtaGroup.ONE"
        if (self.m, self.n, self.pair) == (2, 1, True):
            return "CTA pair (2, 1, 1), tcgen05 CtaGroup.TWO, leader = rank 0"
        group = "TWO, leaders = even ranks" if self.pair else "ONE"
        return f"cluster {self.m} x {self.n} (M x N), tcgen05 CtaGroup.{group}"

    # ---- device-side ------------------------------------------------------- #

    def layout_vmnk(self):
        """Cluster layout ``(V, M, N, K)`` for TMA atoms and partitioning."""
        return cute.tiled_divide(cute.make_layout(self.shape), (self.pair_size,))

    def rank(self):
        """This CTA's rank (``block_idx_in_cluster``, hinted warp-uniform; 0 for one CTA). Call it
        from converged code (for example at the top of a role): the uniform hint is a warp-wide
        operation, so calling it inside a lane-predicated branch hangs the warp."""
        if self.size == 1:
            return 0
        return cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    def is_leader(self):
        """True in the CTAs that issue MMAs (the leader of each pair)."""
        return rank_in(self.rank(), self.ctas(LEADER), self.size)

    def coord_vmnk(self):
        """This CTA's ``(v, mi, ni, 0)`` cluster coordinate (converged code, like ``rank()``)."""
        return self.layout_vmnk().get_flat_coord(self.rank())

    def multicast(self, mode: str) -> TmaMulticast:
        """Partition coordinate, layout and copy mask that share each tile among the CTAs along
        ``mode``: "n" for A (CTAs with the same rows of A), "m" for B. With one CTA along the
        mode, the copy is not multicast (``mask`` is None). Call after ``begin_roles`` from
        converged code; the layer's TMA check recognizes the mask then."""
        from transformer_nuggets.cute.warpdrive.tma import record_multicast_mask

        if mode not in MULTICAST_MODES:
            raise ValueError(f"multicast mode must be one of {MULTICAST_MODES}, got {mode!r}")
        axis = 1 if mode == "m" else 2
        vmnk = self.layout_vmnk()
        ctas = len(self.line(0, mode))
        if ctas == 1:
            return TmaMulticast(0, cute.make_layout(1), None, 1)
        coord = self.coord_vmnk()
        layout = cute.make_layout(
            cute.slice_(vmnk, (0, None, 0, 0) if axis == 1 else (0, 0, None, 0)).shape
        )
        mask = cpasync.create_tma_multicast_mask(vmnk, coord, mcast_mode=axis)
        record_multicast_mask(mask, mode, ctas)
        return TmaMulticast(coord[axis], layout, mask, ctas)


class SingleCta(Cluster):
    """No cluster: one CTA per work item, ``CtaGroup.ONE`` MMAs."""

    def __init__(self):
        super().__init__(1, 1, False)


class CtaPair(Cluster):
    """Cluster ``(2, 1, 1)``: two CTAs along M share each work item, ``CtaGroup.TWO`` MMAs
    issued by the leader (rank 0). Per-CTA tiles are half the MMA tile along M."""

    def __init__(self):
        super().__init__(2, 1, True)


def rank_in(rank, ctas, size: int):
    """Whether ``rank`` is one of ``ctas`` (of a ``size``-CTA cluster): True when they are every
    CTA, else the shortest exact device test."""
    ctas = tuple(sorted(ctas))
    if len(ctas) == size:
        return True
    if len(ctas) == 1:
        return rank == ctas[0]
    stride = ctas[1] - ctas[0]
    if ctas == tuple(range(ctas[0], size, stride)) and ctas[0] < stride:
        return rank % stride == ctas[0]
    hit = rank == ctas[0]
    for c in ctas[1:]:
        hit = hit | (rank == c)
    return hit


def rank_value(values: dict, rank):
    """``values[rank]`` for this CTA's ``rank``, as the cheapest exact device form over the
    listed ranks: a constant when they agree, else ``a + b * rank`` or ``k << rank``, else a
    select chain. ``rank`` may be a callable, evaluated only when the value varies."""
    ranks = sorted(values)
    first = values[ranks[0]]
    if all(values[r] == first for r in ranks):
        return first
    if callable(rank):
        rank = rank()
    r0, r1 = ranks[:2]
    slope, rem = divmod(values[r1] - first, r1 - r0)
    base = first - slope * r0
    if rem == 0 and all(base + slope * r == values[r] for r in ranks):
        return base + slope * rank
    if first % (1 << r0) == 0 and all(values[r] == (first >> r0) << r for r in ranks):
        return Int32(first >> r0) << rank
    out = Int32(first)
    for r in ranks[1:]:
        out = Int32(cutlass.select_(rank == r, values[r], out))
    return out


# --------------------------------------------------------------------------- #
# Scope derivation (host side, from role placement)
# --------------------------------------------------------------------------- #


@dataclass
class Scope:
    """Where one channel's barriers live and how each side signals them.

    ``pair``: the channel crosses CTAs (pair-scoped/cross-CTA); False for a CTA-local channel.
    ``full_ctas`` are the CTAs with a consumer (they wait on full barriers); ``empty_ctas`` the
    CTAs with a producer (both empty when local). Counts are per CTA rank, per slot. ``tx``: bytes
    armed on each full barrier for a TMA-like producer, which completes once the copies of
    ``tx_ctas`` CTAs land (2 in an MMA pair: the leader arms for both CTAs).
    ``full_targets``/``empty_targets``: for each rank where that side signals, the ranks whose
    full/empty barrier its signal reaches (a TMA producer: the full barriers it arms).
    ``full_mask``/``empty_mask``: ``tcgen05.commit`` multicast masks (bit = CTA rank), one int when
    every signalling CTA uses the same mask, else ``{rank: mask}``. ``tail``: producers drain the
    empty barriers before exit, because a consumer in another CTA signals them asynchronously
    (``tcgen05.commit`` multicast), and a CTA must not exit with such arrivals in flight.
    """

    pair: bool
    full_ctas: tuple[int, ...]
    empty_ctas: tuple[int, ...]
    full_counts: dict
    empty_counts: dict
    tx: int | None = None
    tail: bool = False
    full_mask: int | dict = 1
    empty_mask: int | dict = 1
    tx_ctas: int = 1
    full_targets: dict = field(default_factory=dict)
    empty_targets: dict = field(default_factory=dict)

    def describe(self) -> str:
        if not self.pair:
            return "local"
        names = len(set(self.full_ctas) | set(self.empty_ctas) | {0, 1}) <= 2

        def ctas(cs):
            if names:
                return "+".join("leader" if c == 0 else "peer" for c in cs)
            return "+".join(f"rank {c}" for c in cs)

        kind = "pair" if names else "cluster"
        return f"{kind}: full in {ctas(self.full_ctas)}, empty in {ctas(self.empty_ctas)}"


def _masks(targets: dict) -> int | dict:
    masks = {r: sum(1 << c for c in cs) for r, cs in targets.items()}
    values = set(masks.values())
    return values.pop() if len(values) == 1 else masks


def derive_scope(chan, cluster, *, internal: bool = False) -> Scope:
    """Derive ``chan``'s scope from the placement of its bound role instances."""
    from transformer_nuggets.cute.warpdrive.channels import Threads, Tma, Umma

    bound = [*chan.producers, *chan.consumers]
    ranks = range(cluster.size)
    placed = {ri.name: cluster.ctas(ri.cta) for ri, _ in bound}
    modes = chan.producer.multicast if isinstance(chan.producer, Tma) else ()
    reach = {r: cluster.reach(r, modes) for r in ranks}
    multicast = any(len(cs) > 1 for cs in reach.values())
    same = len(set(placed.values())) == 1
    if same and not multicast:
        full = chan.slot_counts("producer")
        empty = chan.slot_counts("consumer")
        tx = chan.producer.tx_bytes if isinstance(chan.producer, Tma) else None
        self_targets = {c: (c,) for c in ranks}

        def local_mask(mech, signalers):
            if mech is not Umma:
                return 1
            return _masks({c: (c,) for ri, _ in signalers for c in placed[ri.name]})

        return Scope(
            False,
            (),
            (),
            {c: full for c in ranks},
            {c: empty for c in ranks},
            tx,
            tail=False,
            full_mask=local_mask(chan.producer, chan.producers),
            empty_mask=local_mask(chan.consumer, chan.consumers),
            full_targets=self_targets,
            empty_targets=dict(self_targets),
        )

    if chan.payload is not None and not internal:
        raise ProtocolError(
            f"channel {chan.name}: its layer-owned payload is per-CTA SMEM, but the channel is "
            "pair-scoped (its roles run in different CTAs), so a consumer could read the other "
            "CTA's unwritten copy; keep payload channels CTA-local (same placement on both sides)"
        )
    if multicast and RANK0 in {ri.cta for ri, _ in bound}:
        raise ProtocolError(
            f"channel {chan.name}: a multicast TMA channel cannot have a rank0 role; its CTAs "
            "load and consume their tiles per MMA pair"
        )
    full_ctas = tuple(sorted({c for ri, _ in chan.consumers for c in placed[ri.name]}))
    empty_ctas = tuple(sorted({c for ri, _ in chan.producers for c in placed[ri.name]}))
    # The CTAs one signal can reach: the CTA itself when every participant runs in the same
    # CTAs (only multicast crosses), the MMA pair when every participant runs in each pair,
    # else the whole cluster.
    if same:
        unit = {r: (r,) for r in ranks}
    elif all(any(c in cs for c in cluster.group(r)) for cs in placed.values() for r in ranks):
        unit = {r: cluster.group(r) for r in ranks}
    else:
        unit = {r: tuple(ranks) for r in ranks}
    full_targets = {p: tuple(c for c in full_ctas if c in unit[p]) for p in empty_ctas}
    # A consumer releases every producing CTA whose copies or signals reach the CTAs it reads.
    empty_targets = {
        c: tuple(p for p in empty_ctas if set(reach[p]) & set(unit[c])) for c in full_ctas
    }

    def counts(mech, signalers, targets):
        out = {}
        for c in ranks:
            per_slot = []
            for j in range(chan.n):
                insts = [ri for ri, slot in signalers if slot in (None, j)]
                senders = [
                    (ri, s) for ri in insts for s in placed[ri.name] if c in targets.get(s, ())
                ]
                if mech is Threads:
                    per_slot.append(sum(ri.num_threads for ri, _ in senders))
                else:  # Umma commit multicast or one TMA arm: one arrival per signalling CTA
                    per_slot.append(len({s for _, s in senders}))
            # Never waited in a CTA no signal reaches.
            out[c] = per_slot if any(per_slot) else [1] * chan.n
        return out

    tx, tx_ctas = None, 1
    if isinstance(chan.producer, Tma):
        tx = chan.producer.tx_bytes
        leaders = cluster.ctas(LEADER)
        if internal or (same and multicast):
            pass  # the producer arms each full barrier it targets for that CTA's own bytes
        elif cluster.pair and empty_ctas == tuple(ranks) and full_ctas == leaders:
            # 2-CTA copies of both CTAs of a pair complete on the leader's barrier, which only
            # the leader arms.
            tx_ctas = cluster.pair_size
            tx *= tx_ctas
            full_targets = {p: (p,) for p in full_ctas}
        elif set(empty_ctas) <= set(leaders) and len(empty_ctas) < len(ranks):
            raise ProtocolError(
                f"channel {chan.name}: a leader-only TMA producer feeding consumers in other "
                "CTAs would need multicast TMA into the peer CTA, armed there by the leader, "
                "which the layer does not derive; run the producer in both CTAs (each loads its "
                "own half)"
            )
        elif cluster.pair:
            raise ProtocolError(
                f"channel {chan.name}: TMA producers in both CTAs can only feed leader-only "
                f"consumers (2-CTA operands complete on the leader's barrier); consumers are "
                f"{[ri.name + '@' + ri.cta for ri, _ in chan.consumers]}"
            )
        else:
            raise ProtocolError(
                f"channel {chan.name}: a TMA copy completes on its own CTA's barrier without "
                "MMA pairs, so producers and consumers must run in the same CTAs; consumers "
                f"are {[ri.name + '@' + ri.cta for ri, _ in chan.consumers]}"
            )
    if isinstance(chan.consumer, Tma):
        raise ProtocolError(f"channel {chan.name}: TMA is a producer mechanism")
    tail = chan.consumer is Umma and any(p != c for c, ps in empty_targets.items() for p in ps)
    return Scope(
        True,
        full_ctas,
        empty_ctas,
        counts(chan.producer, chan.producers, full_targets),
        counts(chan.consumer, chan.consumers, empty_targets),
        tx,
        tail=tail,
        full_mask=_masks(full_targets),
        empty_mask=_masks(empty_targets),
        tx_ctas=tx_ctas,
        full_targets=full_targets,
        empty_targets=empty_targets,
    )


def apply_overrides(chan, scope: Scope, cluster) -> Scope:
    """``scope`` with the channel's ``override`` values (user values win)."""
    if not chan.override:
        return scope
    changes = {}
    for key, value in chan.override.items():
        if key in ("full_counts", "empty_counts"):
            changes[key] = _per_rank_counts(chan, key, value, cluster)
        elif type(value) is not int or value < 0:
            raise ProtocolError(f"channel {chan.name}: override {key} must be an int >= 0")
        else:
            changes[key] = value
    if "tx" in changes and scope.tx is None:
        raise ProtocolError(f"channel {chan.name}: override tx on a channel without TMA bytes")
    return dataclasses.replace(scope, **changes)


def _per_rank_counts(chan, key: str, value, cluster) -> dict:
    ranks = range(cluster.size)
    if isinstance(value, dict):
        if any(type(rank) is not int for rank in value) or set(value) != set(ranks):
            raise ProtocolError(f"channel {chan.name}: override {key} needs ranks {list(ranks)}")
        per_rank = value
    else:
        per_rank = dict.fromkeys(ranks, value)
    out = {}
    for rank, counts in per_rank.items():
        if type(counts) is int:
            counts = [counts] * chan.n
        else:
            try:
                counts = list(counts)
            except TypeError as exc:
                raise ProtocolError(
                    f"channel {chan.name}: override {key} needs {chan.n} positive counts, got {counts!r}"
                ) from exc
        if len(counts) != chan.n or any(type(c) is not int or c <= 0 for c in counts):
            raise ProtocolError(
                f"channel {chan.name}: override {key} needs {chan.n} positive counts, got {counts}"
            )
        out[rank] = counts
    return out


def check_atoms(ctx_values, cluster):
    """Trace-time check that MMA atoms in the role context match the cluster's MMA pairs, and
    that a kernel has no TMA load its cluster cannot complete: 2-CTA loads need pairs, multicast
    loads more than one CTA. A pair kernel may hold both kinds of TMA load: ``CtaGroup.TWO`` for
    pair-scoped channels (bytes complete on the leader's barrier) and ``CtaGroup.ONE`` for
    CTA-local ones."""
    for name, v in ctx_values.items():
        if isinstance(v, cute.TiledMma):
            size = cute.size(v.thr_id.shape)
            if isinstance(size, int) and size != cluster.pair_size:
                raise ProtocolError(
                    f"context {name}: tiled MMA spans {size} CTA(s) (CtaGroup."
                    f"{'TWO' if size == 2 else 'ONE'}) but the kernel declares "
                    f"{type(cluster).__name__} ({cluster.pair_size} CTA(s) per MMA); build it "
                    "with self.cluster.cta_group"
                )
        op = getattr(v, "op", None) if isinstance(v, cute.CopyAtom) else None
        kind = type(op).__name__
        if "G2S" not in kind:
            continue
        if getattr(op, "cta_group", None) == tcgen05.CtaGroup.TWO and not cluster.pair:
            raise ProtocolError(
                f"context {name}: 2-CTA TMA load atom (CtaGroup.TWO) in a "
                f"{type(cluster).__name__} kernel; its bytes would complete on a peer's barrier"
            )
        if "Multicast" in kind and cluster.size == 1:
            raise ProtocolError(
                f"context {name}: multicast TMA load atom in a {type(cluster).__name__} kernel; "
                "multicast needs a cluster (declare self.cluster = Cluster(m, n, ...))"
            )
