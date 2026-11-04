"""Host-side scheduler and derived-count properties, checked against independent models.

* Raster decoding equals an explicit enumeration of the documented visiting order (not only a
  bijection), over random extents, orders and the 1-D fallback boundary.
* Launch grids (``TileProblem.grid``, ``tile_grid_dims``) equal the R12/R13 formulas.
* Every channel scope of a generated kernel (per-rank, per-slot full/empty counts, target CTAs,
  commit masks, armed TMA bytes, tails), the scheduler's warp and work-ring counts, the
  warpgroup register budgets and padding, and the barrier initialization counts equal a
  recomputation from the declared graph per R4, R9, R11 and R12. The recomputation reads only
  the spec, never ``slot_counts``/``derive_scope``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from test.warpdrive_fuzz import gen, oracle

# --------------------------------------------------------------------------- #
# Raster and launch grids
# --------------------------------------------------------------------------- #


def random_stream(rng: random.Random) -> gen.StreamSpec:
    em, en, el = rng.randint(1, 40), rng.randint(1, 40), rng.randint(1, 4)
    if rng.random() < 0.1:
        em, en, el = rng.choice([(1, 1, 1), (1, 9, 1), (9, 1, 1), (1, 1, 5)])
    return gen.StreamSpec(
        policy=rng.choice(["oneshot", "static", "dynamic", "clc"]),
        stages=rng.randint(1, 4),
        along=rng.choice("mn"),
        swizzle=rng.choice([1, 2, 3, 4, 5, 8, 13, 64]),
        reverse=tuple(a for a in "mnl" if rng.random() < 0.3),
        skip=None,
        ready=False,
        extents=(em, en, el),
        max_ctas=rng.choice([1, 2, 3, 7, 148, 1000]),
        scheduler_warp=None,
        rounds=(1, 1, 0, 0, 0),
    )


def raster_of(s: gen.StreamSpec):
    from transformer_nuggets.cute.warpdrive import Raster

    return Raster(along=s.along, swizzle=s.swizzle, reverse=s.reverse)


def check_raster(s: gen.StreamSpec) -> list[str]:
    order = raster_of(s)
    want = oracle.raster_order(s)
    got = [tuple(order.decode(i, s.extents)) for i in range(s.total)]
    if got != want:
        bad = next(i for i, (a, b) in enumerate(zip(got, want, strict=True)) if a != b)
        return [
            f"{order.describe()} over {s.extents}: index {bad} -> {got[bad]}, want {want[bad]}"
        ]
    return []


def check_grid(s: gen.StreamSpec, cs: int) -> list[str]:
    from transformer_nuggets.cute.warpdrive import (
        Clc,
        CtaPair,
        DynamicPersistent,
        OneShot,
        SingleCta,
        StaticPersistent,
        TileStream,
    )
    from transformer_nuggets.cute.warpdrive.scheduling import TileProblem

    policy = {
        "oneshot": OneShot(),
        "static": StaticPersistent(),
        "dynamic": DynamicPersistent(s.stages),
        "clc": Clc(s.stages),
    }[s.policy]
    stream = TileStream(order=raster_of(s), policy=policy)
    stream.owner = SimpleNamespace(cluster=CtaPair() if cs == 2 else SingleCta())
    got = tuple(TileProblem(stream, s.extents, None, s.max_ctas).grid())
    want = oracle.expected_grid(SimpleNamespace(cs=cs, stream=s, units=1))
    return (
        []
        if got == want
        else [f"{s.policy} {s.extents} cs={cs} cap={s.max_ctas}: grid {got}, want {want}"]
    )


def check_tile_grid_dims(space: tuple[int, int, int], cs: int) -> list[str]:
    from transformer_nuggets.cute.warpdrive.scheduling import tile_grid_dims

    em, en, el = space
    if min(space) <= 0:  # R17: an empty space launches one work unit
        want = (cs, 1, 1)
        got = tuple(tile_grid_dims(space, cs))
        return [] if got == want else [f"tile_grid_dims({space}, {cs}) = {got}, want {want}"]
    fits = en <= oracle.GRID_YZ_LIMIT and el <= oracle.GRID_YZ_LIMIT
    want = (max(em, 1) * cs, max(en, 1), max(el, 1)) if fits else (max(em * en * el, 1) * cs, 1, 1)
    got = tuple(tile_grid_dims(space, cs))
    return [] if got == want else [f"tile_grid_dims({space}, {cs}) = {got}, want {want}"]


def grid_boundary_spaces() -> list[tuple[int, int, int]]:
    lim = oracle.GRID_YZ_LIMIT
    return [
        (1, lim, 1),
        (1, lim + 1, 1),
        (3, 1, lim),
        (3, 1, lim + 1),
        (2, lim, lim),
        (7, lim + 1, lim + 1),
        (1, 1, 1),
        (0, 0, 1),
        (5, 0, 3),
    ]


# --------------------------------------------------------------------------- #
# Derived scopes of generated kernels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Inst:
    """One bound role instance: a channel side's contributor."""

    name: str
    threads: int
    cta: str
    slot: int | None


def _ranks(inst: Inst, cs: int) -> tuple[int, ...]:
    return (0,) if inst.cta == "leader" else tuple(range(cs))


@dataclass
class WantScope:
    pair: bool
    full_ctas: tuple
    empty_ctas: tuple
    full_counts: dict
    empty_counts: dict
    tx: int | None
    tail: bool
    full_mask: int
    empty_mask: int


def want_scope(
    n: int,
    prod_mech: str,
    cons_mech: str,
    producers: list[Inst],
    consumers: list[Inst],
    cs: int,
    tma_bytes: int | None,
) -> WantScope:
    """R4 (CTA-local) and R11 (pair) counts, from the participants alone."""

    def local_counts(mech, insts):
        return [
            sum(i.threads if mech == "threads" else 1 for i in insts if i.slot in (None, j))
            for j in range(n)
        ]

    placements = {i.cta for i in (*producers, *consumers)}
    if cs == 1 or len(placements) == 1:
        full, empty = local_counts(prod_mech, producers), local_counts(cons_mech, consumers)
        return WantScope(
            False,
            (),
            (),
            {c: full for c in range(cs)},
            {c: empty for c in range(cs)},
            tma_bytes,
            False,
            1,
            1,
        )
    full_ctas = tuple(sorted({c for i in consumers for c in _ranks(i, cs)}))
    empty_ctas = tuple(sorted({c for i in producers for c in _ranks(i, cs)}))

    def pair_counts(mech, insts, targets):
        out = {}
        for c in range(cs):
            if c not in targets:
                out[c] = [1] * n
                continue
            # Threads: every thread of every instance in each of its CTAs; a TMA arm or a
            # multicast tcgen05.commit: one arrival per target barrier.
            out[c] = [
                sum(i.threads * len(_ranks(i, cs)) for i in insts if i.slot in (None, j))
                if mech == "threads"
                else int(any(i.slot in (None, j) for i in insts))
                for j in range(n)
            ]
        return out

    tx = None
    if tma_bytes is not None:
        if empty_ctas == tuple(range(cs)) and full_ctas == (0,):
            tx = tma_bytes * cs
        else:
            tx = tma_bytes  # leader-only producer: the scheduler ring (user TMA is local here)
    return WantScope(
        True,
        full_ctas,
        empty_ctas,
        pair_counts(prod_mech, producers, full_ctas),
        pair_counts(cons_mech, consumers, empty_ctas),
        tx,
        cons_mech == "umma" and any(c != 0 for c in empty_ctas),
        sum(1 << c for c in full_ctas),
        sum(1 << c for c in empty_ctas),
    )


def _insts(spec: gen.KernelSpec, side: list[tuple[str, bool]]) -> list[Inst]:
    out = []
    for rname, bound in side:
        r = spec.role(rname)
        for i in range(r.instances):
            out.append(Inst(f"{rname}[{i}]", r.threads, r.cta, i if bound else None))
    return out


def scheduler_warp(spec: gen.KernelSpec) -> int:
    s = spec.stream
    if s.scheduler_warp is not None:
        return s.scheduler_warp
    owned = {w for r in spec.roles for rng in r.warp_ranges() for w in rng}
    return next(w for w in range(max(owned) + 2) if w not in owned)


def expected_scopes(spec: gen.KernelSpec) -> dict[str, WantScope]:
    cs = spec.cs
    out = {}
    for ch in spec.channels:
        out[ch.name] = want_scope(
            ch.n,
            ch.producer,
            ch.consumer,
            _insts(spec, list(ch.producers)),
            _insts(spec, list(ch.consumers)),
            cs,
            ch.tma_bytes if ch.producer == "tma" else None,
        )
    s = spec.stream
    if s is not None and s.policy in ("dynamic", "clc"):
        sched = Inst("tiles.scheduler", 32, "leader" if cs > 1 else "all", None)
        users = [
            Inst(f"{r.name}[{i}]", r.threads, r.cta, None)
            for r in spec.roles
            for i in range(r.instances)
        ]
        out["tiles_work"] = want_scope(
            s.stages,
            "tma" if cs > 1 else "threads",
            "threads",
            [sched],
            [*users, sched],
            cs,
            16 if cs > 1 else None,
        )
        if s.policy == "clc":
            out["tiles_clc"] = want_scope(1, "tma", "threads", [sched], [sched], cs, 16)
    return out


def expected_regs(spec: gen.KernelSpec) -> tuple[int, list, dict] | None:
    """(padded warp count, [(hole range, budget)], {role instance: budget}) per R9, or None when
    no role sets a budget."""
    budget = {}
    names = {}
    for r in spec.roles:
        for i, rng in enumerate(r.warp_ranges()):
            for w in rng:
                budget[w] = r.regs
            names[f"{r.name}[{i}]" if r.instances > 1 else r.name] = r.regs
    s = spec.stream
    if not any(r.regs is not None for r in spec.roles):
        return None
    if s is not None and s.policy in ("dynamic", "clc"):
        w = scheduler_warp(spec)
        group = {budget[x] for x in range(w // 4 * 4, w // 4 * 4 + 4) if x in budget}
        numeric = [r.regs for r in spec.roles if r.regs is not None]
        regs = group.pop() if len(group) == 1 else min(numeric)
        budget[w] = regs
        names["tiles.scheduler"] = regs
    nw = -(-(max(budget) + 1) // 4) * 4
    holes = []
    for g in range(nw // 4):
        vals = {budget[w] for w in range(4 * g, 4 * g + 4) if w in budget}
        regs = vals.pop() if vals else gen.MIN_REGS
        run = []
        for w in range(4 * g, 4 * g + 4):
            if w in budget:
                if run:
                    holes.append((range(run[0], run[-1] + 1), regs))
                run = []
            else:
                run.append(w)
        if run:
            holes.append((range(run[0], run[-1] + 1), regs))
    return nw, holes, names


_ROW = re.compile(r"^  (\d+)(?:-(\d+))?\s+(\S+)\s+(\d+)?")


def described_roles(text: str) -> list[tuple[range, str, int | None]]:
    rows = []
    body = text.split("\nRoles\n", 1)[1].split("\n\n", 1)[0].splitlines()[1:]
    for line in body:
        m = _ROW.match(line)
        if m:
            a, b, name, regs = m.groups()
            rows.append((range(int(a), int(b or a) + 1), name, int(regs) if regs else None))
    return rows


@dataclass
class ScopeReport:
    failures: list[str] = field(default_factory=list)
    channels: int = 0


def check_declared(spec: gen.KernelSpec, workdir: Path) -> ScopeReport:
    """Compare a generated kernel's derived scopes, scheduler, budgets and barrier init counts
    with the independent recomputation (host only: ``describe()``)."""
    from test.warpdrive_fuzz import runtime

    rep = ScopeReport()
    op = runtime.load_kernel(spec, workdir)
    text = op.describe()
    chans = op._ws_chans
    for name, want in expected_scopes(spec).items():
        rep.channels += 1
        got = chans[name].scope
        for f in (
            "pair",
            "full_ctas",
            "empty_ctas",
            "full_counts",
            "empty_counts",
            "tx",
            "tail",
            "full_mask",
            "empty_mask",
        ):
            if getattr(got, f) != getattr(want, f):
                rep.failures.append(
                    f"{name}.{f}: layer {getattr(got, f)!r}, model {getattr(want, f)!r}"
                )
    s = spec.stream
    if s is not None and s.policy in ("dynamic", "clc"):
        warp = scheduler_warp(spec)
        if f"scheduler warp {warp} broadcasts" not in text:
            rep.failures.append(f"scheduler not at warp {warp}:\n{text}")
    want = expected_regs(spec)
    if want is not None:
        nw, holes, regs = want
        rows = described_roles(text)
        if op.num_warps != nw:
            rep.failures.append(f"num_warps {op.num_warps}, model {nw}")
        got_holes = [(rng, b) for rng, name, b in rows if name == "(padding)"]
        if got_holes != holes:
            rep.failures.append(f"padding {got_holes}, model {holes}")
        got_regs = {name: b for _, name, b in rows if name != "(padding)"}
        for name, b in regs.items():
            default = gen._default_regs(nw)
            if got_regs.get(name) not in (b, default if b is None else b):
                rep.failures.append(f"{name} budget {got_regs.get(name)}, model {b}")
    # R1: barriers are initialized with the effective counts (full then empty, per stage).
    plan, _ = op._barrier_plan()
    for name, _, counts in plan:
        if name in chans:
            sc = chans[name].scope
            for c in range(spec.cs):
                if counts[c] != sc.full_counts[c] + sc.empty_counts[c]:
                    rep.failures.append(f"{name} rank {c} init counts {counts[c]}")
    return rep
