"""Cluster declarations and the derivation of cross-CTA channel scopes.

A kernel declares its cluster once: ``self.cluster = CtaPair()`` (the default is
``SingleCta()``). In a CTA pair, two CTAs along M form a cluster ``(2, 1, 1)`` and run
``tcgen05`` ``CtaGroup.TWO`` MMAs: the leader CTA (rank 0) issues every MMA, which reads A and B
from the shared memory of both CTAs and writes each CTA's half of the accumulator into that CTA's
tensor memory.

Roles run in every CTA of the cluster unless pinned with ``@role(..., cta="leader")``. From that
placement the layer derives each channel's *scope*:

* **local**: every participant has the same placement. Each CTA has its own independent copy of
  the channel (in a pair kernel, a channel between two roles that run in both CTAs pairs the
  roles of the same CTA).
* **pair**: participants have different placements (for example a leader-only MMA role and a
  TMA load or epilogue role that runs in both CTAs). The pair shares one logical channel: a
  barrier lives in every CTA that has a role waiting on it, each signal reaches the barrier of
  every such CTA (remote ``mbarrier.arrive``, ``tcgen05.commit`` multicast, or a TMA
  transaction completing on the leader's barrier), and arrival counts and transaction bytes sum
  the contributions of both CTAs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from cutlass import cute
from cutlass.cute.nvgpu import tcgen05

from transformer_nuggets.cute.warpdrive.protocol import ProtocolError

ALL = "all"
LEADER = "leader"
PLACEMENTS = (ALL, LEADER)
# Derived ``Scope`` fields a channel may override (``Channel(override={...})``).
SCOPE_OVERRIDES = ("full_counts", "empty_counts", "tx", "full_mask", "empty_mask")


@dataclass(frozen=True)
class SingleCta:
    """No cluster: one CTA per work item, ``CtaGroup.ONE`` MMAs."""

    size = 1
    shape = (1, 1, 1)
    cta_group = tcgen05.CtaGroup.ONE

    def layout_vmnk(self):
        """Cluster layout ``(V, M, N, K)`` for TMA atoms and partitioning."""
        return cute.tiled_divide(cute.make_layout(self.shape), (1,))

    def rank(self):
        """This CTA's rank in the cluster (the MMA ``V`` coordinate)."""
        return 0

    def is_leader(self):
        """True in the CTA that issues the MMAs."""
        return True

    def launch_cluster(self):
        return None

    def get_key(self) -> str:
        return "cta1"

    def describe(self) -> str:
        return "single CTA"


@dataclass(frozen=True)
class CtaPair(SingleCta):
    """Cluster ``(2, 1, 1)``: two CTAs along M share each work item, ``CtaGroup.TWO`` MMAs
    issued by the leader (rank 0). Per-CTA tiles are half the MMA tile along M."""

    size = 2
    shape = (2, 1, 1)
    cta_group = tcgen05.CtaGroup.TWO

    def layout_vmnk(self):
        return cute.tiled_divide(cute.make_layout(self.shape), (2,))

    def rank(self):
        """This CTA's rank (``block_idx_in_cluster``, hinted warp-uniform). Call it from
        converged code (for example at the top of a role): the uniform hint is a warp-wide
        operation, so calling it inside a lane-predicated branch hangs the warp."""
        return cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    def is_leader(self):
        return self.rank() == 0

    def launch_cluster(self):
        return self.shape

    def get_key(self) -> str:
        return "pair"

    def describe(self) -> str:
        return "CTA pair, cluster (2, 1, 1), tcgen05 CtaGroup.TWO, leader = rank 0"


# --------------------------------------------------------------------------- #
# Scope derivation (host side, from role placement)
# --------------------------------------------------------------------------- #


@dataclass
class Scope:
    """Where one channel's barriers live and how each side signals them.

    ``full_ctas`` are the CTAs with a consumer (they wait on full barriers); ``empty_ctas`` the
    CTAs with a producer. Counts are per CTA rank, per slot. ``tx``: bytes armed on each full
    barrier for a TMA-like producer (in a pair, by the leader, for every CTA's copies).
    ``full_mask``/``empty_mask``: multicast masks (bit = CTA rank) of a ``tcgen05.commit``
    signalling the full/empty barriers. ``tail``: producers drain the empty barriers before exit,
    because the consumer signals them asynchronously from another CTA (``tcgen05.commit``
    multicast), and a CTA must not exit with such arrivals in flight.
    """

    pair: bool
    full_ctas: tuple[int, ...]
    empty_ctas: tuple[int, ...]
    full_counts: dict
    empty_counts: dict
    tx: int | None = None
    tail: bool = False
    full_mask: int = 1
    empty_mask: int = 1

    def describe(self) -> str:
        if not self.pair:
            return "local"

        def ctas(cs):
            return "+".join("leader" if c == 0 else "peer" for c in cs)

        return f"pair: full in {ctas(self.full_ctas)}, empty in {ctas(self.empty_ctas)}"


def _ctas(ri, cluster) -> tuple[int, ...]:
    return (0,) if ri.cta == LEADER else tuple(range(cluster.size))


def derive_scope(chan, cluster, *, internal: bool = False) -> Scope:
    """Derive ``chan``'s scope from the placement of its bound role instances."""
    from transformer_nuggets.cute.warpdrive.channels import Threads, Tma, Umma

    bound = [*chan.producers, *chan.consumers]
    local = cluster.size == 1 or len({ri.cta for ri, _ in bound}) == 1
    if local:
        full = chan.slot_counts("producer")
        empty = chan.slot_counts("consumer")
        tx = chan.producer.tx_bytes if isinstance(chan.producer, Tma) else None
        ranks = range(cluster.size)
        return Scope(
            False, (), (), {c: full for c in ranks}, {c: empty for c in ranks}, tx, tail=False
        )

    if chan.payload is not None and not internal:
        raise ProtocolError(
            f"channel {chan.name}: its layer-owned payload is per-CTA SMEM, but the channel is "
            "pair-scoped (its roles run in different CTAs), so a consumer could read the other "
            "CTA's unwritten copy; keep payload channels CTA-local (same placement on both sides)"
        )
    full_ctas = tuple(sorted({c for ri, _ in chan.consumers for c in _ctas(ri, cluster)}))
    empty_ctas = tuple(sorted({c for ri, _ in chan.producers for c in _ctas(ri, cluster)}))

    def counts(mech, signalers, targets):
        out = {}
        for c in range(cluster.size):
            if c not in targets:
                out[c] = [1] * chan.n  # never waited in this CTA
                continue
            per_slot = []
            for j in range(chan.n):
                insts = [ri for ri, slot in signalers if slot in (None, j)]
                if mech is Threads:
                    per_slot.append(sum(ri.num_threads * len(_ctas(ri, cluster)) for ri in insts))
                else:  # Umma commit multicast or one TMA arm: one arrival per target barrier
                    per_slot.append(1 if insts else 0)
            out[c] = per_slot
        return out

    tx = None
    if isinstance(chan.producer, Tma):
        prod_ctas = empty_ctas
        if prod_ctas == tuple(range(cluster.size)) and full_ctas == (0,):
            tx = chan.producer.tx_bytes * cluster.size  # both halves complete on the leader
        elif prod_ctas == (0,) and (internal or full_ctas == (0,)):
            tx = chan.producer.tx_bytes  # the leader fills every consumer CTA itself
        elif prod_ctas == (0,):
            raise ProtocolError(
                f"channel {chan.name}: a leader-only TMA producer feeding consumers in both CTAs "
                "would need multicast TMA into the peer CTA, which the layer does not support; "
                "run the producer in both CTAs (each loads its own half)"
            )
        else:
            raise ProtocolError(
                f"channel {chan.name}: TMA producers in both CTAs can only feed leader-only "
                f"consumers (2-CTA operands complete on the leader's barrier); consumers are "
                f"{[ri.name + '@' + ri.cta for ri, _ in chan.consumers]}"
            )
    if isinstance(chan.consumer, Tma):
        raise ProtocolError(f"channel {chan.name}: TMA is a producer mechanism")
    tail = chan.consumer is Umma and any(c != 0 for c in empty_ctas)
    return Scope(
        True,
        full_ctas,
        empty_ctas,
        counts(chan.producer, chan.producers, full_ctas),
        counts(chan.consumer, chan.consumers, empty_ctas),
        tx,
        tail=tail,
        full_mask=sum(1 << c for c in full_ctas),
        empty_mask=sum(1 << c for c in empty_ctas),
    )


def apply_overrides(chan, scope: Scope, cluster) -> Scope:
    """``scope`` with the channel's ``override`` values (user values win)."""
    if not chan.override:
        return scope
    changes = {}
    for key, value in chan.override.items():
        if key in ("full_counts", "empty_counts"):
            changes[key] = _per_rank_counts(chan, key, value, cluster)
        elif not isinstance(value, int) or value < 0:
            raise ProtocolError(f"channel {chan.name}: override {key} must be an int >= 0")
        else:
            changes[key] = value
    if "tx" in changes and scope.tx is None:
        raise ProtocolError(f"channel {chan.name}: override tx on a channel without TMA bytes")
    return dataclasses.replace(scope, **changes)


def _per_rank_counts(chan, key: str, value, cluster) -> dict:
    ranks = range(cluster.size)
    if isinstance(value, dict):
        if set(value) != set(ranks):
            raise ProtocolError(f"channel {chan.name}: override {key} needs ranks {list(ranks)}")
        per_rank = value
    else:
        per_rank = dict.fromkeys(ranks, value)
    out = {}
    for rank, counts in per_rank.items():
        counts = [counts] * chan.n if isinstance(counts, int) else list(counts)
        if len(counts) != chan.n or any(not isinstance(c, int) or c <= 0 for c in counts):
            raise ProtocolError(
                f"channel {chan.name}: override {key} needs {chan.n} positive counts, got {counts}"
            )
        out[rank] = counts
    return out


def check_atoms(ctx_values, cluster):
    """Trace-time check that MMA atoms in the role context match the cluster, and that a
    single-CTA kernel has no 2-CTA TMA loads. A pair kernel may hold both kinds of TMA load:
    ``CtaGroup.TWO`` for pair-scoped channels (bytes complete on the leader's barrier) and
    ``CtaGroup.ONE`` for CTA-local ones."""
    for name, v in ctx_values.items():
        if isinstance(v, cute.TiledMma):
            size = cute.size(v.thr_id.shape)
            if isinstance(size, int) and size != cluster.size:
                raise ProtocolError(
                    f"context {name}: tiled MMA spans {size} CTA(s) (CtaGroup."
                    f"{'TWO' if size == 2 else 'ONE'}) but the kernel declares "
                    f"{type(cluster).__name__} ({cluster.size} CTA(s)); build it with "
                    "self.cluster.cta_group"
                )
        op = getattr(v, "op", None) if isinstance(v, cute.CopyAtom) else None
        group = getattr(op, "cta_group", None)
        two = tcgen05.CtaGroup.TWO
        if group == two and "G2S" in type(op).__name__ and cluster.size == 1:
            raise ProtocolError(
                f"context {name}: 2-CTA TMA load atom (CtaGroup.TWO) in a "
                f"{type(cluster).__name__} kernel; its bytes would complete on a peer's barrier"
            )
