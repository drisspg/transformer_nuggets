"""Random valid warpdrive kernels, emitted as Python source.

Why source files rather than one constexpr-driven template class: a role's channel ends,
cursor, seat and TMEM view are bound by *parameter name and annotation* and its warps, register
budget and placement are ``@role(...)`` decorator arguments, all fixed when the class body
runs. The number of roles, their signatures, and which channels each takes are exactly what the
fuzzer varies, and CuTeDSL's preprocessor needs real source for every ``@cute.jit`` body. So
each ``KernelSpec`` is rendered to a module (deterministic name from the spec hash), imported
from a file, and compiled like a hand-written kernel.

Validity (the contract's "valid managed program", section 1) is by construction:

* Channels go forward in a random topological order of the roles (a DAG), every participant of
  a channel performs the same per-iteration handle program on it, and every handle a role opens
  in an iteration is closed in that iteration. A ring's per-iteration multiplicity never exceeds
  its depth. Induction on (iteration, topological position) then shows the abstract
  bounded-buffer program is live under any intra-iteration open/close interleaving: iteration
  ``i`` of a role waits only on upstream publications of iteration ``i`` and on downstream
  releases of iterations ``<= i - 1``.
* ``Threads`` sides are executed by every thread of every bound instance; ``Tma``/``Umma`` sides
  belong to one single-warp, single-instance role (leader-only in a pair).
* Payloads only ride on CTA-local channels with ``Threads`` or ``Tma`` producers and ``Threads``
  consumers; ``Umma`` and pair-scoped channels are tokens (counts only). TMA stages read by
  threads declare ``release_fence="smem_async"`` (the U4 user obligation).
* With a ``TileStream`` every role drains it (``while tiles.valid: ...; tiles.advance()``) and
  runs ``rounds(item)`` iterations per item, computed identically by every role.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

MAX_WARPS = 32
WARPGROUP = 4
REGS_PER_SM = 65536
MIN_REGS = 24
TMA_COLS = 32  # Int32 columns per TMA tile row: 128-byte rows
SMS = 148  # nominal SM count for generated caps; the runner never assumes it


Key = tuple[str, str, int]  # (channel, "p" | "c", ring handle k / slot j / BOUND)
BOUND = -1


@dataclass(frozen=True)
class RoleSpec:
    name: str
    warps: tuple[int, ...]  # first warp of each instance
    width: int  # warps per instance
    regs: int | None
    cta: str  # "all" | "leader"
    jitter: int  # spin scale in cycles (0: none)
    tmem: bool
    # Per-iteration handle program: ("o" | "c", channel, side, key) in execution order.
    events: tuple[tuple[str, str, str, int], ...] = ()
    with_handles: tuple[Key, ...] = ()  # handles emitted as nested ``with`` blocks

    @property
    def instances(self) -> int:
        return len(self.warps)

    @property
    def threads(self) -> int:
        return 32 * self.width

    def warp_ranges(self) -> list[range]:
        return [range(w, w + self.width) for w in self.warps]


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    kind: str  # "ring" | "slots"
    n: int
    producer: str  # "threads" | "umma" | "tma"
    consumer: str  # "threads" | "umma"
    producers: tuple[tuple[str, bool], ...]  # (role, bound to Instance)
    consumers: tuple[tuple[str, bool], ...]
    mult: int = 1  # ring: handles per iteration per participant
    extra_mult: int = 0  # ring: handles in the final extra round
    extra_slots: tuple[int, ...] = ()  # slots: slots used in the final extra round
    payload: bool = False
    tma_rows: int = 0
    tma_pair: bool = False  # pair-scoped: both CTAs copy, CtaGroup.TWO, leader-only consumers
    commit_fence: str | None = None
    release_fence: str | None = None
    remote_scope: str | None = None
    salt: int = 0

    @property
    def tma_tiles(self) -> int:
        return self.n + 1  # coprime with the stage count: stale stages read the wrong tile

    @property
    def tiles_checked(self) -> bool:
        """Threads consumers read and verify TMA tiles (an Umma consumer only releases)."""
        return self.producer == "tma" and self.consumer == "threads"

    @property
    def tma_bytes(self) -> int:
        return self.tma_rows * TMA_COLS * 4

    def handle_keys(self, side: str, bound: bool) -> list[int]:
        if self.kind == "ring":
            return list(range(self.mult))
        return [BOUND] if bound else list(range(self.n))


@dataclass(frozen=True)
class StreamSpec:
    policy: str  # "oneshot" | "static" | "dynamic" | "clc"
    stages: int
    along: str
    swizzle: int
    reverse: tuple[str, ...]
    skip: tuple[int, int, int, int] | None  # skip item when (a m + b n + c l) % k == 0
    ready: bool
    extents: tuple[int, int, int]
    max_ctas: int
    scheduler_warp: int | None
    rounds: tuple[int, int, int, int, int]  # (base, spread, a, b, c)
    launch_1d: bool = False  # OneShot: a hand-built (total * cs, 1, 1) grid (R13)
    record: int = 0  # TileStream(record=k): the timeline must match the item logs (R16)

    @property
    def total(self) -> int:
        em, en, el = self.extents
        return em * en * el if min(self.extents) > 0 else 0  # R17: any extent <= 0 is empty

    @property
    def plain_raster(self) -> bool:
        return self.along == "m" and self.swizzle == 1 and not self.reverse and self.skip is None

    @property
    def tile_grid(self) -> bool:
        """The layer launches the tile space itself (R13: OneShot, plain Raster, no record)."""
        return self.policy == "oneshot" and self.plain_raster and not self.record

    def item_rounds(self, m: int, n: int, li: int) -> int:
        base, spread, a, b, c = self.rounds
        return (m * a + n * b + li * c) % spread + base


@dataclass(frozen=True)
class KernelSpec:
    seed: int
    pair: bool
    roles: tuple[RoleSpec, ...]
    channels: tuple[ChannelSpec, ...]
    turn_role: str | None = None
    turn_enabled: bool = True
    stream: StreamSpec | None = None
    iters: int = 1  # iterations without a stream
    units: int = 1  # work units launched without a stream
    tmem_regions: tuple[tuple[str, int, int], ...] = ()
    smem_pad: int = 0  # Int32 words of extra SMEM (CLC: one CTA per SM)
    log_cap: int = 32
    item_cap: int = 64
    turn_cap: int = 64
    launches: int = 2
    trace: bool = False  # trace_protocol: must not change results (R16)
    # Further launches of the same compiled kernel with other runtime values:
    # ((m_tiles, n_tiles, l), max_ctas, iterations) each. Extents, caps and counts are runtime
    # arguments, so one binary must serve them all.
    variants: tuple[tuple[tuple[int, int, int], int, int], ...] = ()
    graph: bool = False  # also capture one launch in a CUDA graph and replay it twice

    @property
    def cs(self) -> int:
        return 2 if self.pair else 1

    def role(self, name: str) -> RoleSpec:
        return next(r for r in self.roles if r.name == name)

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(text: str) -> KernelSpec:
        return spec_from_dict(json.loads(text))


def _tuples(x):
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def spec_from_dict(d: dict) -> KernelSpec:
    roles = tuple(RoleSpec(**{k: _tuples(v) for k, v in r.items()}) for r in d["roles"])
    chans = tuple(ChannelSpec(**{k: _tuples(v) for k, v in c.items()}) for c in d["channels"])
    stream = d.get("stream")
    if stream is not None:
        stream = StreamSpec(**{k: _tuples(v) for k, v in stream.items()})
    rest = {k: _tuples(v) for k, v in d.items() if k not in ("roles", "channels", "stream")}
    return KernelSpec(roles=roles, channels=chans, stream=stream, **rest)


# --------------------------------------------------------------------------- #
# Derived structure shared by the emitter and the oracle
# --------------------------------------------------------------------------- #


def is_local(spec: KernelSpec, ch: ChannelSpec) -> bool:
    """CTA-local per the contract (R11): single CTA, or every participant has one placement."""
    if not spec.pair:
        return True
    ctas = {spec.role(r).cta for r, _ in (*ch.producers, *ch.consumers)}
    return len(ctas) == 1


def payload_offsets(spec: KernelSpec, ch: ChannelSpec) -> tuple[dict, int]:
    """``{role: (offset of each instance)}`` of the words each producer writes, and the width."""
    offsets, width = {}, 0
    for rname, bound in ch.producers:
        r = spec.role(rname)
        if bound:  # one instance per slot: they share one region
            offsets[rname] = (width,) * r.instances
            width += r.threads
        else:
            offsets[rname] = tuple(width + i * r.threads for i in range(r.instances))
            width += r.threads * r.instances
    return offsets, width


@dataclass
class Layout:
    """Per-CTA word offsets of every log the generated kernel writes (``clog[cta, off]``)."""

    per_cta: int = 0
    presence: int = 0
    role_it: dict = field(default_factory=dict)  # (role, instance) -> offset
    role_items: dict = field(default_factory=dict)  # (role, instance) -> [count, items, hashes]
    cons: dict = field(default_factory=dict)  # (chan, role, instance) -> [count, errs, log...]
    prod: dict = field(default_factory=dict)  # (chan, role, instance) -> count
    turn: int | None = None
    accumulators: list = field(default_factory=list)  # offsets initialized to 0 (else -1)
    instance_ids: dict = field(default_factory=dict)  # (role, instance) -> global ordinal

    @staticmethod
    @functools.cache
    def of(spec: KernelSpec) -> Layout:
        lay = Layout()
        off = MAX_WARPS
        lay.accumulators += list(range(MAX_WARPS))
        ordinal = 0
        for r in spec.roles:
            for i in range(r.instances):
                lay.instance_ids[(r.name, i)] = ordinal
                ordinal += 1
                lay.role_it[(r.name, i)] = off
                off += 1
                if spec.stream is not None:
                    lay.role_items[(r.name, i)] = off
                    off += 1 + 2 * spec.item_cap
        for ch in spec.channels:
            for rname, _ in ch.consumers:
                for i in range(spec.role(rname).instances):
                    lay.cons[(ch.name, rname, i)] = off
                    lay.accumulators.append(off + 1)
                    off += 2 + spec.log_cap
            for rname, _ in ch.producers:
                for i in range(spec.role(rname).instances):
                    lay.prod[(ch.name, rname, i)] = off
                    off += 1
        if spec.turn_role is not None:
            lay.turn = off
            lay.accumulators.append(off)
            off += 1 + spec.turn_cap
        lay.per_cta = off
        return lay


def presence_code(spec: KernelSpec, rname: str, inst: int) -> int:
    return 1 + Layout.of(spec).instance_ids[(rname, inst)]


# --------------------------------------------------------------------------- #
# Random generation
# --------------------------------------------------------------------------- #


def _place(rng: random.Random, roles: list[dict]) -> None:
    """Assign each role's instances to warps, in random order with random holes."""
    order = list(range(len(roles)))
    rng.shuffle(order)
    w = rng.choice([0, 0, 0, 1, 2])
    for i in order:
        r = roles[i]
        firsts = []
        for _ in range(r["instances"]):
            firsts.append(w)
            w += r["width"]
            if rng.random() < 0.15:
                w += 1
        r["warps"] = tuple(firsts)
        if rng.random() < 0.3:
            w += rng.randint(1, 3)


def _default_regs(num_warps: int) -> int:
    return min(REGS_PER_SM // (num_warps * 32) // 8 * 8, 256)


def _scheduler_warp(roles: list[dict], explicit: int | None) -> int:
    owned = {w for r in roles for f in r["warps"] for w in range(f, f + r["width"])}
    if explicit is not None:
        return explicit
    return next(w for w in range(max(owned) + 2) if w not in owned)


def register_total(roles: list[dict], sched: int | None, sched_regs) -> int | None:
    """Total accounted registers per R9 (holes take their warpgroup's budget, idle groups 24,
    ``None`` the default), or None when the warpgroup rule is violated."""
    budget = {}
    for r in roles:
        for f in r["warps"]:
            for w in range(f, f + r["width"]):
                budget[w] = r["regs"]
    if sched is not None:
        budget[sched] = sched_regs
    nw = -(-(max(budget) + 1) // WARPGROUP) * WARPGROUP
    default = _default_regs(nw)
    total = 0
    for g in range(nw // WARPGROUP):
        vals = {budget[w] for w in range(g * 4, g * 4 + 4) if w in budget}
        if len(vals) > 1:
            return None
        regs = vals.pop() if vals else MIN_REGS
        total += 4 * 32 * (default if regs is None else regs)
    return total


def _assign_regs(rng: random.Random, roles: list[dict]) -> None:
    """Warpgroup-uniform budgets: roles sharing a warpgroup form one component."""
    parent = list(range(len(roles)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    groups = {}
    for i, r in enumerate(roles):
        for f in r["warps"]:
            for w in range(f, f + r["width"]):
                groups.setdefault(w // WARPGROUP, []).append(i)
    for members in groups.values():
        for j in members[1:]:
            parent[find(j)] = find(members[0])
    choice = {}
    for i in range(len(roles)):
        c = find(i)
        if c not in choice:
            choice[c] = None if rng.random() < 0.25 else rng.choice(range(24, 257, 8))
        roles[i]["regs"] = choice[c]


def _make_events(rng: random.Random, handles: list[Key], kinds: dict) -> list:
    """A random open/close interleaving: each handle opens before it closes; ring handles of one
    channel open in key order (acquire order assigns the epochs)."""
    opened, closed, events = set(), set(), []
    while len(closed) < len(handles):
        avail = []
        for h in handles:
            ch, side, k = h
            if h not in opened:
                prev = (ch, side, k - 1)
                if kinds[ch] != "ring" or k == 0 or prev in opened:
                    avail.append(("o", h))
            elif h not in closed:
                avail.append(("c", h))
        op, h = rng.choice(avail)
        (opened if op == "o" else closed).add(h)
        events.append((op, *h))
    return events


def _laminar(rng: random.Random, events: list) -> list[Key]:
    """A random subset of handles whose [open, close] intervals nest or are disjoint."""
    span = {}
    for pos, (op, *h) in enumerate(events):
        span.setdefault(tuple(h), [pos, pos])[1 if op == "c" else 0] = pos
    chosen = []
    keys = list(span)
    rng.shuffle(keys)
    for h in keys:
        a0, a1 = span[h]
        if rng.random() < 0.6 and all(
            a1 < b0 or b1 < a0 or (a0 < b0 and b1 < a1) or (b0 < a0 and a1 < b1)
            for b0, b1 in (span[c] for c in chosen)
        ):
            chosen.append(h)
    return sorted(chosen)


def generate(
    seed: int, *, pair: bool | None = None, stream: bool | None = None, empty: bool = False
) -> KernelSpec:
    """A random valid kernel. ``pair``/``stream`` force those knobs; otherwise random.
    ``empty``: the zero-work boundary corpus (a TileStream whose space has a zero extent; the
    contract treats it separately from the nonempty corpus, G6)."""
    if empty:
        stream = True
    rng = random.Random(seed)
    pair = rng.random() < 0.3 if pair is None else pair
    use_stream = rng.random() < 0.6 if stream is None else stream
    nroles = rng.randint(2, 6)
    roles = []
    for i in range(nroles):
        width = rng.choice([1, 1, 1, 2, 4])
        roles.append(
            {
                "name": f"r{i}",
                "width": width,
                "instances": rng.choice([1, 1, 1, 2, 3]) if width <= 2 else 1,
                "cta": "leader" if pair and rng.random() < 0.35 else "all",
                "jitter": rng.choice([0, 0, 0, 40, 400, 3000]),
                "tmem": False,
                "regs": None,
            }
        )
    single = [i for i, r in enumerate(roles) if r["width"] == 1 and r["instances"] == 1]

    channels = []
    nch = rng.randint(max(1, nroles - 2), nroles + 2)
    for c in range(nch):
        p, q = sorted(rng.sample(range(nroles), 2))  # forward edge in topological order
        P, Q = roles[p], roles[q]
        kind = "slots" if rng.random() < 0.4 else "ring"
        prod_mech = "threads"
        cons_mech = "threads"
        if p in single and rng.random() < 0.35:
            prod_mech = rng.choice(["umma", "tma", "tma"])
        if q in single and rng.random() < 0.2:
            cons_mech = "umma"
        producers = [(P["name"], False)]
        consumers = [(Q["name"], False)]
        # Fan-in (Threads producers only) and fan-out joins (Threads consumers only).
        # Every producer precedes every consumer in the topological order.
        last_prod = p
        if prod_mech == "threads" and rng.random() < 0.25:
            others = [i for i in range(q) if i != p]
            if others:
                extra = rng.choice(others)
                producers.append((roles[extra]["name"], False))
                last_prod = max(p, extra)
        if cons_mech == "threads" and rng.random() < 0.35:
            others = [i for i in range(last_prod + 1, nroles) if i != q]
            if others:
                consumers.append((roles[rng.choice(others)]["name"], False))
        n = rng.randint(1, 8) if kind == "ring" else rng.randint(1, 4)
        if kind == "slots" and rng.random() < 0.08:
            n = rng.randint(5, 31)  # phase bits up to bit 30 of the Int32 mask
        if kind == "slots":
            # Bind an instanced role to the slots (Producer[Instance] / Consumer[Instance]).
            if prod_mech == "threads" and P["instances"] > 1 and rng.random() < 0.6:
                n = P["instances"]
                producers[0] = (P["name"], True)
            if (
                cons_mech == "threads"
                and Q["instances"] > 1
                and rng.random() < 0.6
                and (not producers[0][1] or Q["instances"] == n)
            ):
                n = Q["instances"]
                consumers[0] = (Q["name"], True)
                if producers[0][1] and P["instances"] != n:
                    producers[0] = (P["name"], False)
        mult = rng.randint(1, min(n, 3)) if kind == "ring" else 1
        channels.append(
            {
                "name": f"c{c}",
                "kind": kind,
                "n": n,
                "producer": prod_mech,
                "consumer": cons_mech,
                "producers": tuple(producers),
                "consumers": tuple(consumers),
                "mult": mult,
                "extra_mult": rng.randint(0, mult) if kind == "ring" else 0,
                "extra_slots": tuple(sorted(rng.sample(range(n), rng.randint(0, n))))
                if kind == "slots"
                else (),
                "salt": rng.randint(0, 4095),
            }
        )

    # Pair placement rules: tcgen05.commit sides are leader-only; TMA only CTA-local.
    if pair:
        for ch in channels:
            for side in ("producers", "consumers"):
                mech = ch["producer" if side == "producers" else "consumer"]
                if mech == "umma":
                    for rname, _ in ch[side]:
                        roles[int(rname[1:])]["cta"] = "leader"
        for ch in channels:  # promote some TMA consumers so the channel becomes pair-scoped
            prod = {roles[int(r[1:])]["cta"] for r, _ in ch["producers"]}
            if ch["producer"] == "tma" and prod == {"all"} and rng.random() < 0.4:
                for rname, _ in ch["consumers"]:
                    roles[int(rname[1:])]["cta"] = "leader"
        for ch in channels:
            if ch["producer"] == "tma":
                prod = {roles[int(r[1:])]["cta"] for r, _ in ch["producers"]}
                cons = {roles[int(r[1:])]["cta"] for r, _ in ch["consumers"]}
                if prod == {"all"} and cons == {"leader"} and rng.random() < 0.7:
                    ch["tma_pair"] = True  # both CTAs' bytes complete on the leader (R11)
                elif len(prod | cons) > 1:
                    ch["producer"] = "threads"
    spec_roles = {r["name"]: r for r in roles}
    for ch in channels:
        local = not pair or (
            len({spec_roles[r]["cta"] for r, _ in (*ch["producers"], *ch["consumers"])}) == 1
        )
        if ch["producer"] == "tma":
            ch["tma_rows"] = rng.choice([1, 2, 4, 8])
            ch["release_fence"] = "smem_async"
            ch["payload"] = False
        else:
            ch["payload"] = local and ch["consumer"] == "threads" and ch["producer"] == "threads"
            ch["commit_fence"] = rng.choice([None, None, "smem_async"])
            ch["release_fence"] = rng.choice([None, None, "smem_async"])
        if not local and ch["producer"] == "threads" or not local and ch["consumer"] == "threads":
            ch["remote_scope"] = rng.choice([None, "cta", "cluster"])
        if ch["payload"]:
            width = sum(
                spec_roles[r]["width"] * 32 * (1 if b else spec_roles[r]["instances"])
                for r, b in ch["producers"]
            )
            if width > 512:
                ch["payload"] = False

    # TMEM: a plan and a few roles that take the view (alloc/free/teardown barrier counts).
    tmem_regions = ()
    if rng.random() < 0.3:
        regions, used = [], 0
        for k in range(rng.randint(1, 3)):
            cols, slots = rng.choice([32, 64, 128]), rng.randint(1, 2)
            if used + cols * slots <= 512:
                regions.append((f"t{k}", cols, slots))
                used += cols * slots
        tmem_regions = tuple(regions)
        users = rng.sample(range(nroles), rng.randint(1, nroles))
        for i in users:
            roles[i]["tmem"] = True
        # E39: a pair allocates TMEM from a role that runs in both CTAs.
        if pair and all(roles[i]["cta"] == "leader" for i in users):
            everywhere = [r for r in roles if r["cta"] == "all"]
            if everywhere:
                rng.choice(everywhere)["tmem"] = True
            else:
                tmem_regions = ()
                for r in roles:
                    r["tmem"] = False

    turn_candidates = [r["name"] for r in roles if r["instances"] >= 2]
    turn_role = rng.choice(turn_candidates) if turn_candidates and rng.random() < 0.5 else None

    _place(rng, roles)
    if max(f + r["width"] for r in roles for f in r["warps"]) > 24:
        return generate(seed + 1_000_003, pair=pair, stream=stream, empty=empty)

    stream_spec = None
    if use_stream:
        policy = rng.choice(["oneshot", "static", "dynamic", "clc"])
        stages = rng.randint(1, 4)
        along = rng.choice(["m", "m", "n"])
        swizzle = rng.choice([1, 1, 2, 3, 8])
        reverse = tuple(a for a in ("m", "n", "l") if rng.random() < 0.15)
        skip = None
        if rng.random() < 0.2:
            skip = (rng.randint(0, 3), rng.randint(0, 3), rng.randint(0, 3), rng.randint(1, 4))
        if policy == "oneshot" and rng.random() < 0.5:
            along, swizzle, reverse, skip = "m", 1, (), None  # the tile-grid launch
        big = policy == "oneshot" and along == "m" and swizzle == 1 and not reverse and not skip
        if big and rng.random() < 0.08:
            extents = (rng.randint(1, 2), 65535 + rng.randint(0, 2), 1)  # y-limit fallback edge
        else:
            extents = (rng.randint(1, 12), rng.randint(1, 12), rng.randint(1, 3))
            if rng.random() < 0.15:
                extents = (1, 1, 1)
        if empty:  # at least one extent is zero or negative (R17: both mean an empty space)
            zero_axis = rng.randrange(3)
            extents = tuple(
                (0 if rng.random() < 0.75 else -rng.randint(1, 3))
                if i == zero_axis or rng.random() < 0.2
                else e
                for i, e in enumerate(extents)
            )
        stream_spec = StreamSpec(
            policy=policy,
            stages=stages,
            along=along,
            swizzle=swizzle,
            reverse=reverse,
            skip=skip,
            ready=rng.random() < 0.2,
            extents=extents,
            max_ctas=rng.choice([1, 2, 3, 5, 8, 37, SMS]),
            scheduler_warp=None,
            rounds=(
                rng.randint(0, 2),
                rng.randint(1, 3),
                rng.randint(0, 3),
                rng.randint(0, 3),
                rng.randint(0, 3),
            ),
            launch_1d=policy == "oneshot" and rng.random() < 0.3,
            record=rng.randint(1, 4) if extents[1] < 65535 and rng.random() < 0.15 else 0,
        )

    # Registers: random warpgroup-uniform budgets that fit, including the scheduler warp.
    sched = None
    if stream_spec is not None and stream_spec.policy in ("dynamic", "clc"):
        owned = sorted({w for r in roles for f in r["warps"] for w in range(f, f + r["width"])})
        holes = [w for w in range(max(owned) + 2) if w not in owned]
        explicit = rng.choice(holes) if rng.random() < 0.3 else None
        stream_spec = StreamSpec(**{**asdict(stream_spec), "scheduler_warp": explicit})
        sched = _scheduler_warp(roles, explicit)
    if rng.random() < 0.5:
        for _ in range(20):
            _assign_regs(rng, roles)
            if sched is not None:
                # R9: a warp the scheduler takes in an owned warpgroup has that group's budget
                # (None included); alone in a warpgroup it takes the smallest numeric budget.
                grp = {
                    r["regs"]
                    for r in roles
                    for f in r["warps"]
                    for w in range(f, f + r["width"])
                    if w // 4 == sched // 4
                }
                nums = [r["regs"] for r in roles if r["regs"] is not None]
                sched_regs = grp.pop() if len(grp) == 1 else (min(nums) if nums else None)
            else:
                sched_regs = None
            total = register_total(roles, sched, sched_regs)
            if total is not None and total <= REGS_PER_SM:
                break
        else:
            for r in roles:
                r["regs"] = None

    # Per-role handle programs.
    kinds = {ch["name"]: ch["kind"] for ch in channels}
    for r in roles:
        handles = []
        for ch in channels:
            for side, lst in (("p", ch["producers"]), ("c", ch["consumers"])):
                for rname, bound in lst:
                    if rname != r["name"]:
                        continue
                    if ch["kind"] == "ring":
                        keys = range(ch["mult"])
                    else:
                        keys = [BOUND] if bound else rng.sample(range(ch["n"]), ch["n"])
                    handles += [(ch["name"], side, k) for k in keys]
        events = _make_events(rng, handles, kinds)
        r["events"] = tuple(tuple(e) for e in events)
        r["with_handles"] = tuple(_laminar(rng, events))

    grid_rows = 1
    if stream_spec is None:
        units = rng.randint(1, 3)
        grid_rows = units
    else:
        units = 1
        s = stream_spec
        if s.policy in ("static", "dynamic"):
            grid_rows = max(1, min(s.total, max(s.max_ctas // (2 if pair else 1), 1)))
        else:
            grid_rows = max(s.total, 1)
    big_grid = grid_rows > 4096
    smem_pad = 0
    if stream_spec is not None and stream_spec.policy == "clc" and rng.random() < 0.5:
        smem_pad = 40000  # ~160 KB: one CTA per SM, so try_cancel has CTAs to cancel
    used = 0
    for ch in channels:  # barriers, layer payloads and TMA stages, 16-byte aligned
        used += 16 * ch["n"]
        if ch["payload"]:
            width = sum(
                spec_roles[r]["width"] * 32 * (1 if b else spec_roles[r]["instances"])
                for r, b in ch["producers"]
            )
            used += 4 * width * ch["n"] + 16
        if ch["producer"] == "tma":
            used += ch["tma_rows"] * TMA_COLS * 4 * ch["n"] + 128
    for ch in sorted(channels, key=lambda c: -c["n"]):
        if used <= 160_000:
            break
        if ch["payload"]:
            ch["payload"] = False
            used -= 4 * 512 * ch["n"]
    smem_pad = min(smem_pad, max(0, (200_000 - used) // 4))
    return KernelSpec(
        seed=seed,
        pair=pair,
        roles=tuple(
            RoleSpec(
                name=r["name"],
                warps=r["warps"],
                width=r["width"],
                regs=r["regs"],
                cta=r["cta"],
                jitter=0 if big_grid else r["jitter"],
                tmem=bool(r["tmem"]),
                events=r["events"],
                with_handles=r["with_handles"],
            )
            for r in roles
        ),
        channels=tuple(ChannelSpec(**ch) for ch in channels),
        turn_role=turn_role,
        turn_enabled=rng.random() < 0.85,
        stream=stream_spec,
        iters=rng.choice([0, 1, 2, 3, 5, 8, 17, 33]),
        units=units,
        tmem_regions=tmem_regions,
        smem_pad=smem_pad,
        log_cap=2 if big_grid else 32,
        item_cap=1 if big_grid else 64,
        turn_cap=2 if big_grid else 64,
        trace=rng.random() < 0.15,
        variants=_variants(rng, stream_spec, big_grid or any(r["jitter"] for r in roles)),
        graph=rng.random() < 0.2 and not (stream_spec is not None and stream_spec.record),
    )


def _variants(rng: random.Random, s: StreamSpec | None, small_only: bool) -> tuple:
    """Runtime-shape variants: other extents (both sides of the tile-grid y/z limit when the
    base allows a large grid), caps and iteration counts."""
    out = []
    for _ in range(rng.choice([0, 0, 1, 2])):
        extents = (rng.randint(1, 16), rng.randint(1, 16), rng.randint(1, 3))
        if rng.random() < 0.1:
            extents = (1, 1, 1)
        if s is not None and s.tile_grid and not small_only and rng.random() < 0.15:
            extents = (1, 65535 + rng.randint(0, 1), 1)
        out.append((extents, rng.choice([1, 2, 3, 5, 8, 37, SMS]), rng.choice([0, 1, 2, 7, 33])))
    return tuple(out)


def variant_specs(spec: KernelSpec) -> list[KernelSpec]:
    """The spec itself, then one spec per runtime variant (same compiled kernel)."""
    out = [spec]
    for extents, max_ctas, iters in spec.variants:
        stream = spec.stream
        if stream is not None:
            stream = dataclasses.replace(stream, extents=extents, max_ctas=max_ctas)
        out.append(dataclasses.replace(spec, stream=stream, iters=iters, variants=()))
    return out


# --------------------------------------------------------------------------- #
# Source emission
# --------------------------------------------------------------------------- #


class _Src:
    def __init__(self):
        self.lines: list[str] = []
        self.depth = 0

    def __call__(self, line: str = ""):
        self.lines.append("    " * self.depth + line if line else "")

    def block(self, header: str):
        self(header)
        self.depth += 1

    def end(self, n: int = 1):
        self.depth -= n

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


_MECH = {"threads": "Threads", "umma": "Umma"}
_POLICY = {
    "oneshot": "OneShot()",
    "static": "StaticPersistent()",
    "dynamic": "DynamicPersistent(stages={})",
    "clc": "Clc(stages={})",
}


def class_name(spec: KernelSpec) -> str:
    return f"FuzzKernel_{spec.key()}"


def kernel_name(spec: KernelSpec) -> str:
    return f"wdfuzz_{spec.key()}"


def _hvar(key) -> str:
    ch, side, k = key
    return f"h_{ch}_{side}_{'b' if k == BOUND else k}"


def _role_params(spec: KernelSpec, r: RoleSpec) -> list[str]:
    params = ["self"]
    if r.instances > 1 or any(
        b for ch in spec.channels for rn, b in (*ch.producers, *ch.consumers) if rn == r.name
    ):
        params.append("stage: Instance")
    for ch in spec.channels:
        for side, lst in (("Producer", ch.producers), ("Consumer", ch.consumers)):
            for rname, bound in lst:
                if rname == r.name:
                    params.append(
                        f"{ch.name}: {side}[Instance]" if bound else f"{ch.name}: {side}"
                    )
    if spec.turn_role == r.name:
        params.append("ts: Turn")
    if spec.stream is not None:
        params.append("tiles: Tiles")
    if r.tmem:
        params.append("tmem: Tmem")
    params.append("ctx")
    return params


def _by_stage(values: tuple, r: RoleSpec) -> str:
    if len(set(values)) == 1:
        return str(values[0])
    return f"{tuple(values)}[stage]"


def _emit_body(src: _Src, spec: KernelSpec, r: RoleSpec, extra: bool):
    """One iteration of the role's handle program (``extra``: the final partial round)."""
    chans = {c.name: c for c in spec.channels}
    lay = Layout.of(spec)

    def present(key) -> str | None:
        """Python condition for the handle being used this round (None: always)."""
        ch, _, k = key
        c = chans[ch]
        if not extra:
            return None
        if c.kind == "ring":
            return "True" if k < c.extra_mult else "False"
        if k == BOUND:
            return f"stage in {tuple(c.extra_slots)!r}"
        return "True" if k in c.extra_slots else "False"

    def epoch(key) -> str:
        c = chans[key[0]]
        return f"(it * {c.mult} + {key[2]})" if c.kind == "ring" else "it"

    def slot(key) -> str:
        c = chans[key[0]]
        if c.kind == "ring":
            return "0"
        return "stage" if key[2] == BOUND else str(key[2])

    def on_open(key):
        ch, side, _ = key
        c = chans[ch]
        h = _hvar(key)
        e = epoch(key)
        counter = f"k_{ch}_{side}"
        if side == "p":
            src(f"D.jitter({e}, tid, {c.salt}, {r.jitter})")
            if c.payload:
                offs, _ = payload_offsets(spec, c)
                src(
                    f"D.write_words({h}.data, {e}, tid, {slot(key)}, {_by_stage(offs[r.name], r)}, "
                    f"{c.salt})"
                )
            elif c.producer == "tma":
                src(
                    f"cute.copy(ctx.tma_{ch}, ctx.tg_{ch}[None, {e} % {c.tma_tiles}], "
                    f"ctx.ts_{ch}[None, {h}.index], tma_bar_ptr={h}.barrier)"
                )
        else:
            offs = tuple(lay.cons[(ch, r.name, i)] for i in range(r.instances))
            log = f"base + {_by_stage(offs, r)} + 2"
            if c.payload:
                _, width = payload_offsets(spec, c)
                src(
                    f"err_{ch} = err_{ch} + D.check_words({h}.data, {e}, tid, {slot(key)}, "
                    f"{width}, {r.threads}, {c.salt})"
                )
                src(
                    f"D.put_capped(ctx.clog, {log}, {counter}, {spec.log_cap}, "
                    f"{h}.data[0] >> 12, lead)"
                )
            elif c.tiles_checked:
                src(
                    f"err_{ch} = err_{ch} + D.check_tile(ctx.s_{ch}, {h}.index, {e} % "
                    f"{c.tma_tiles}, tid, {c.tma_rows}, {TMA_COLS}, {r.threads}, {c.salt})"
                )
                src(
                    f"D.put_capped(ctx.clog, {log}, {counter}, {spec.log_cap}, "
                    f"ctx.s_{ch}[0, 0, {h}.index] >> 12, lead)"
                )
        src(f"{counter} = {counter} + 1")

    def on_close(key):
        ch, side, _ = key
        c = chans[ch]
        h = _hvar(key)
        if side == "c":
            src(f"D.jitter({epoch(key)}, tid, {c.salt + 1}, {r.jitter})")
            if c.payload:
                _, width = payload_offsets(spec, c)
                src(
                    f"err_{ch} = err_{ch} + D.check_words({h}.data, {epoch(key)}, tid, "
                    f"{slot(key)}, {width}, {r.threads}, {c.salt})"
                )
            elif c.tiles_checked:
                src(
                    f"err_{ch} = err_{ch} + D.check_tile(ctx.s_{ch}, {h}.index, {epoch(key)} % "
                    f"{c.tma_tiles}, tid, {c.tma_rows}, {TMA_COLS}, {r.threads}, {c.salt})"
                )

    def open_expr(key) -> str:
        ch, side, k = key
        op = "acquire" if side == "p" else "wait"
        arg = "" if chans[ch].kind == "ring" else ("stage" if k == BOUND else str(k))
        return f"{ch}.{op}({arg})"

    withs = set(r.with_handles) if not extra else set()
    stack = []
    emitted = False
    for op, ch, side, k in r.events:
        key = (ch, side, k)
        cond = present(key)
        if cond == "False":
            continue
        guarded = cond not in (None, "True")
        if guarded:
            src.block(f"if cutlass.const_expr({cond}):")
        if op == "o":
            if key in withs:
                src.block(
                    f"with {open_expr(key).replace('acquire', 'produce').replace('wait', 'consume')}"
                    f" as {_hvar(key)}:"
                )
                stack.append(key)
                on_open(key)
            else:
                src(f"{_hvar(key)} = {open_expr(key)}")
                on_open(key)
        else:
            on_close(key)
            if key in withs:
                assert stack and stack[-1] == key, (key, stack)
                stack.pop()
                src.end()
            else:
                src(f"{_hvar(key)}.{'commit' if side == 'p' else 'release'}()")
        if guarded:
            src.end()
        emitted = True
    if not emitted:
        src("pass")


def _emit_role(src: _Src, spec: KernelSpec, r: RoleSpec):
    lay = Layout.of(spec)
    regs = "" if r.regs is None else f", regs={r.regs}"
    cta = "" if r.cta == "all" else ', cta="leader"'
    warps = f"[{', '.join(f'range({w}, {w + r.width})' for w in r.warps)}]"
    src(f"@role(warps={warps}{regs}{cta})")
    src("@cute.jit")
    src.block(f"def {r.name}({', '.join(_role_params(spec, r))}):")
    first = _by_stage(r.warps, r)
    src(f"tid = Int32(cute.arch.thread_idx()[0]) - {first} * 32")
    src("lead = tid == 0")
    src("rank = self.cluster.rank()")
    src(f"base = D.cta_id() * {lay.per_cta}")
    codes = tuple(presence_code(spec, r.name, i) for i in range(r.instances))
    src(f"D.mark_warp(ctx.clog, base + {lay.presence}, {_by_stage(codes, r)})")
    src("it = Int32(0)")
    counters = []
    for ch in spec.channels:
        for side, lst in (("p", ch.producers), ("c", ch.consumers)):
            if any(rn == r.name for rn, _ in lst):
                counters.append((ch.name, side))
                src(f"k_{ch.name}_{side} = Int32(0)")
                if side == "c":
                    src(f"err_{ch.name} = Int32(0)")
    ordinal = _by_stage(tuple(lay.instance_ids[(r.name, i)] for i in range(r.instances)), r)

    def iteration():
        _emit_body(src, spec, r, extra=False)
        if spec.turn_role == r.name:
            src.block("with ts.turn():")
            if spec.turn_enabled:
                src(f"D.turn_log(ctx.clog, base + {lay.turn}, stage, {spec.turn_cap}, lead)")
            else:
                src("pass")
            src.end()
        src("it = it + 1")

    if spec.stream is None:
        src.block("for _ in cutlass.range(ctx.iters, unroll=1):")
        iteration()
        src.end()
    else:
        s = spec.stream
        items = _by_stage(tuple(lay.role_items[(r.name, i)] for i in range(r.instances)), r)
        base_r, spread, a, b, c = s.rounds
        src("nitems = Int32(0)")
        src.block("while tiles.valid:")
        src("m, n, l = tiles.work")
        src(f"hit = ({ordinal} * {spec.cs} + rank) * ctx.total + tiles.index")
        src(
            f"D.log_item(ctx.clog, ctx.hits, base + {items} + 1, nitems, {spec.item_cap}, "
            "tiles.index, (m, n, l), hit, lead)"
        )
        src("nitems = nitems + 1")
        src(f"rounds = (Int32(m) * {a} + Int32(n) * {b} + Int32(l) * {c}) % {spread} + {base_r}")
        src.block("for _ in cutlass.range(rounds, unroll=1):")
        iteration()
        src.end()
        src("tiles.advance()")
        src.end()
        src(f"D.put(ctx.clog, base + {items}, nitems, lead)")
    _emit_body(src, spec, r, extra=True)
    it_off = _by_stage(tuple(lay.role_it[(r.name, i)] for i in range(r.instances)), r)
    src(f"D.put(ctx.clog, base + {it_off}, it, lead)")
    for ch, side in counters:
        table = lay.cons if side == "c" else lay.prod
        off = _by_stage(tuple(table[(ch, r.name, i)] for i in range(r.instances)), r)
        src(f"D.put(ctx.clog, base + {off}, k_{ch}_{side}, lead)")
        if side == "c":
            src(f"D.add(ctx.clog, base + {off} + 1, err_{ch})")
    returned = [ch for ch, _ in counters]
    if spec.turn_role == r.name:
        returned.append("ts")
    if spec.stream is not None:
        returned.append("tiles")
    if returned:
        src(f"return {', '.join(returned)}")
    src.end()
    src()


def emit(spec: KernelSpec) -> str:
    """Python source of ``spec``'s kernel class."""
    src = _Src()
    src(f'"""Generated by test/warpdrive_fuzz/gen.py: seed {spec.seed}, key {spec.key()}."""')
    src()
    src("from dataclasses import dataclass")
    src("from types import SimpleNamespace")
    src()
    src("import cutlass")
    src("from cuda.bindings import driver as cuda")
    src("from cutlass import Int32, cute")
    src("from cutlass.cute.nvgpu import cpasync, tcgen05")
    src()
    src("from test.warpdrive_fuzz import runtime_device as D")
    src("from transformer_nuggets.cute.warpdrive import (")
    for name in (
        "Clc",
        "Consumer",
        "CtaPair",
        "DynamicPersistent",
        "Instance",
        "OneShot",
        "PdlWait",
        "Producer",
        "Raster",
        "Ring",
        "Slots",
        "StaticPersistent",
        "Threads",
        "Tiles",
        "TileStream",
        "Tma",
        "Tmem",
        "TmemPlan",
        "Turn",
        "Turnstile",
        "Umma",
        "WarpSpecialized",
        "role",
    ):
        src(f"    {name},")
    src(")")
    src()
    s = spec.stream
    if s is not None and s.skip is not None:
        a, b, c, k = s.skip
        src()
        src("@dataclass(frozen=True)")
        src.block("class SkipMod(Raster):")
        src("skips = True")
        src()
        src.block("def skip(self, item, space):")
        src("m, n, l = item")
        src(f"return (m * {a} + n * {b} + l * {c}) % {k} == 0")
        src.end()
        src()
        src.block("def get_key(self):")
        src(f'return "skipmod{a}{b}{c}{k}_" + super().get_key()')
        src.end(2)
        src()
    src()
    src.block(f"class {class_name(spec)}(WarpSpecialized):")
    src.block("def __init__(self):")
    if spec.pair:
        src("self.cluster = CtaPair()")
    for ch in spec.channels:
        cls = "Ring" if ch.kind == "ring" else "Slots"
        prod = f"Tma({ch.tma_bytes})" if ch.producer == "tma" else _MECH[ch.producer]
        args = [str(ch.n), prod, _MECH[ch.consumer]]
        if ch.payload:
            _, width = payload_offsets(spec, ch)
            args.append(f"payload=(Int32, ({width},))")
        if ch.commit_fence:
            args.append(f'commit_fence="{ch.commit_fence}"')
        if ch.release_fence:
            args.append(f'release_fence="{ch.release_fence}"')
        if ch.remote_scope:
            args.append(f'remote_scope="{ch.remote_scope}"')
        src(f"self.{ch.name} = {cls}({', '.join(args)})")
    if spec.turn_role is not None:
        src(f"self.ts = Turnstile(enabled={spec.turn_enabled})")
    if spec.tmem_regions:
        plan = "TmemPlan()" + "".join(f'.region("{n}", {c}, {k})' for n, c, k in spec.tmem_regions)
        src(f"self.tmem_plan = {plan}")
    if s is not None:
        order = "SkipMod" if s.skip is not None else "Raster"
        order += f'(along="{s.along}", swizzle={s.swizzle}, reverse={tuple(s.reverse)!r})'
        policy = _POLICY[s.policy].format(s.stages)
        extra = f", scheduler_warp={s.scheduler_warp}" if s.scheduler_warp is not None else ""
        ready = ", ready=PdlWait()" if s.ready else ""
        record = f", record={s.record}" if s.record else ""
        src(f"self.tiles = TileStream(order={order}, policy={policy}{ready}{extra}{record})")
    src.end()
    src()
    src.block("def get_name(self):")
    src(f'return "{kernel_name(spec)}"')
    src.end()
    src()
    for r in spec.roles:
        _emit_role(src, spec, r)

    tma = [ch for ch in spec.channels if ch.producer == "tma"]
    kargs = ["self", "clog: cute.Tensor", "hits: cute.Tensor", "iters: Int32"]
    for ch in tma:
        kargs += [f"tma_{ch.name}", f"tg_{ch.name}"]
    if s is not None:
        kargs.append("work")
    src("@cute.kernel")
    src.block(f"def kernel({', '.join(kargs)}):")
    src("smem = self.begin_roles(tiles=work)" if s is not None else "smem = self.begin_roles()")
    ctx = ["clog=clog", "hits=hits", "iters=iters"]
    ctx.append("total=work.total" if s is not None else "total=Int32(0)")
    for ch in tma:
        c = ch.name
        src(
            f"s_{c} = smem.allocate_tensor(Int32, cute.make_ordered_layout(({ch.tma_rows}, "
            f"{TMA_COLS}, {ch.n}), order=(1, 0, 2)), 128)"
        )
        src(
            f"ts_{c}, tgp_{c} = cpasync.tma_partition(tma_{c}, 0, cute.make_layout(1), "
            f"cute.group_modes(s_{c}, 0, 2), cute.group_modes(tg_{c}, 0, 2))"
        )
        ctx += [f"tma_{c}=tma_{c}", f"tg_{c}=tgp_{c}", f"ts_{c}=ts_{c}", f"s_{c}=s_{c}"]
    if spec.smem_pad:
        src(f"smem.allocate_array(Int32, {spec.smem_pad})")
    src(f"self.run_roles(SimpleNamespace({', '.join(ctx)}))")
    src.end()
    src()
    cargs = [
        "self",
        "clog: cute.Tensor",
        "hits: cute.Tensor",
        "ws: cute.Tensor",
        "iters: Int32",
        "em: Int32",
        "en: Int32",
        "el: Int32",
        "max_ctas: Int32",
    ]
    cargs += [f"src_{ch.name}: cute.Tensor" for ch in tma]
    cargs.append("stream: cuda.CUstream")
    src("@cute.jit")
    src.block(f"def __call__({', '.join(cargs)}):")
    launch_args = ["clog", "hits", "iters"]
    for ch in tma:
        c = ch.name
        src(
            f"layout_{c} = cute.make_ordered_layout(({ch.tma_rows}, {TMA_COLS}, {ch.n}), "
            "order=(1, 0, 2))"
        )
        group = "TWO" if ch.tma_pair else "ONE"
        src(
            f"tma_{c}, tgt_{c} = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp("
            f"tcgen05.CtaGroup.{group}), src_{c}, cute.select(layout_{c}, mode=[0, 1]), "
            f"({ch.tma_rows}, {TMA_COLS}))"
        )
        src(f"tiles_{c} = cute.local_tile(tgt_{c}, ({ch.tma_rows}, {TMA_COLS}), (None, 0))")
        launch_args += [f"tma_{c}", f"tiles_{c}"]
    if s is not None:
        src("work = self.tiles.problem((em, en, el), ws, max_ctas)")
        launch_args.append("work")
        cfg = "**self.launch_config(work)"
        if s.launch_1d:
            cfg = f"**self.launch_config(grid=(cutlass.max(work.total, 1) * {spec.cs}, 1, 1))"
    else:
        cfg = f"**self.launch_config(grid=({spec.units * spec.cs}, 1, 1))"
    src(f"self.kernel({', '.join(launch_args)}).launch({cfg}, stream=stream)")
    src.end(2)
    return src.text()


def features(spec: KernelSpec) -> set[str]:
    """Knob classes a spec exercises (for campaign coverage summaries)."""
    out = {"pair" if spec.pair else "single", f"roles{len(spec.roles)}"}
    s = spec.stream
    out.add(f"policy_{s.policy}" if s else "no_stream")
    if s is not None:
        if s.skip:
            out.add("skip")
        if s.ready:
            out.add("ready")
        if s.total > 65535 or max(s.extents[1:]) >= 65535:
            out.add("grid_yz_edge")
        if s.scheduler_warp is not None:
            out.add("explicit_scheduler_warp")
        if s.policy in ("static", "dynamic") and s.max_ctas < SMS:
            out.add("capped_grid")
    if spec.smem_pad:
        out.add("one_cta_per_sm")
    if spec.trace:
        out.add("trace_protocol")
    if spec.variants:
        out.add("runtime_shape_variants")
    if spec.graph:
        out.add("cuda_graph_replay")
    if s is not None and s.launch_1d:
        out.add("oneshot_1d_launch")
    if s is not None and s.record:
        out.add("record")
    if spec.turn_role:
        out.add("turnstile" if spec.turn_enabled else "turnstile_disabled")
    if spec.tmem_regions:
        out.add("tmem")
    if any(r.regs is not None for r in spec.roles):
        out.add("regs")
    owned = {w for r in spec.roles for rng in r.warp_ranges() for w in rng}
    if len(owned) != max(owned) + 1:
        out.add("warp_holes")
    if any(r.instances > 1 for r in spec.roles):
        out.add("instances")
    if any(r.cta == "leader" for r in spec.roles):
        out.add("leader_roles")
    if any(r.with_handles for r in spec.roles):
        out.add("with_blocks")
    for ch in spec.channels:
        out.add(f"{ch.kind}_{ch.producer}_{ch.consumer}")
        out.add(f"depth{ch.n}")
        if ch.mult > 1:
            out.add("overlapping_ring_handles")
        if len(ch.producers) > 1:
            out.add("fan_in")
        if len(ch.consumers) > 1:
            out.add("join")
        if any(b for _, b in (*ch.producers, *ch.consumers)):
            out.add("bound_slots")
        if spec.pair and not is_local(spec, ch):
            out.add("pair_scoped")
        if ch.payload:
            out.add("payload")
        if ch.tma_pair:
            out.add("tma_pair")
        if ch.kind == "slots" and ch.n > 4:
            out.add("slots_many")
    return out
