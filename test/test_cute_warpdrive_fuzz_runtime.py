"""Runtime fuzzing of warpdrive's guarantees to user kernels (quick mode).

Generated valid kernels (``warpdrive_fuzz/gen.py``) run in isolated child processes against an
exact host model (``oracle.py``); host properties of the scheduler and of derived counts run
against independent recomputations (``sched.py``); synthetic layer bugs (``runtime_mutants.py``)
check that the oracle notices each failure class. Long campaigns: ``scripts/fuzz_warpdrive.py``.
"""

import random

import pytest
import torch

pytest.importorskip("cutlass.cute")

from cutlass import cute

from test.warpdrive_fuzz import gen, runtime, runtime_mutants, sched
from transformer_nuggets.cute.warpdrive import (
    DynamicPersistent,
    Tiles,
    TileStream,
    WarpSpecialized,
    role,
)

pytestmark = pytest.mark.fuzz

CASE_TIMEOUT = 120.0  # seconds per generated kernel: compile + 2 launches + checks
MUTANT_TIMEOUT = 30.0  # a mutant that hangs is detected; don't wait longer than this


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("warpdrive runtime fuzzing requires SM100-family tcgen05 and CLC")


def _covering_seeds(pool: range, limit: int) -> list[int]:
    """Greedy seeds covering the generator's knob classes, skipping the >65535-CTA grids (their
    host check is slow; the long campaign runs them)."""
    specs = {s: gen.features(gen.generate(s)) for s in pool}
    specs = {s: f for s, f in specs.items() if "grid_yz_edge" not in f}
    chosen, covered = [], set()
    while len(chosen) < limit:
        best = max(specs, key=lambda s: (len(specs[s] - covered), -s))
        if not specs[best] - covered:
            break
        chosen.append(best)
        covered |= specs.pop(best)
    return sorted(chosen)


QUICK_SEEDS = _covering_seeds(range(300), 24)
QUICK_SEEDS += [s for s in range(300, 400) if s not in QUICK_SEEDS][
    : max(0, 16 - len(QUICK_SEEDS))
]


def _explain(spec, res) -> str:
    lines = [
        f"seed {spec.seed} key {res.key}: {res.status}",
        f"source: {res.source}",
        f"rerun: python scripts/fuzz_warpdrive.py runtime --seeds {spec.seed}",
        *res.failures[:20],
        res.detail[-3000:],
    ]
    return "\n".join(lines)


@pytest.mark.parametrize("seed", QUICK_SEEDS)
def test_generated_kernel_matches_model(requires_blackwell, tmp_path_factory, seed):
    """Every payload epoch reaches every consumer once, in channel order and intact; each kept
    work item reaches each placed role once with identical per-CTA sequences; launches are
    bitwise repeatable for deterministic policies; nothing hangs."""
    spec = gen.generate(seed)
    res = runtime.run_isolated(spec, tmp_path_factory.getbasetemp() / "wdfuzz", CASE_TIMEOUT)
    assert res.status == runtime.PASS, _explain(spec, res)


# Mutants the oracle must catch, with the seeds (exercising the mutated feature) to try. Known
# blind spots of a value-based oracle are excluded: the early-release work-ring mutant (its
# read-after-release window is too narrow to hit without delays inside the layer) and dropped
# pair tails (a late remote tcgen05.commit into an exited CTA changes no logged value).
BLIND_SPOTS = ("receive_early_release", "no_pair_tail")
DETECTABLE = [m for m in runtime_mutants.MUTANTS if m not in BLIND_SPOTS]


@pytest.mark.parametrize("mutant", DETECTABLE)
def test_oracle_detects_layer_mutant(requires_blackwell, tmp_path_factory, mutant):
    feature = runtime_mutants.MUTANTS[mutant][1]
    seeds = [s for s in range(300) if feature in gen.features(gen.generate(s))]
    workdir = tmp_path_factory.getbasetemp() / f"wdfuzz_mutant_{mutant}"
    env = {"WDFUZZ_MUTANT": mutant, "CUTE_DSL_DISABLE_FILE_CACHING": "1"}
    seen = []
    for seed in seeds[:4]:
        res = runtime.run_isolated(gen.generate(seed), workdir, MUTANT_TIMEOUT, env=env)
        seen.append((seed, res.status))
        if res.status in (runtime.FAIL, runtime.HANG):
            return
    pytest.fail(f"mutant {mutant} survived seeds {seen}")


# --------------------------------------------------------------------------- #
# Host properties (no GPU)
# --------------------------------------------------------------------------- #


def test_raster_decode_matches_documented_order():
    rng = random.Random(0)
    failures = []
    for _ in range(400):
        failures += sched.check_raster(sched.random_stream(rng))
    assert not failures, failures[:5]


def test_launch_grid_matches_contract():
    rng = random.Random(1)
    failures = []
    for _ in range(400):
        stream = sched.random_stream(rng)
        failures += sched.check_grid(stream, 1) + sched.check_grid(stream, 2)
    for space in sched.grid_boundary_spaces():
        for cs in (1, 2):
            failures += sched.check_tile_grid_dims(space, cs)
    assert not failures, failures[:5]


@pytest.mark.parametrize("chunk", range(4))
def test_declared_scopes_match_independent_model(tmp_path_factory, chunk):
    """Counts, targets, masks, tx, tails, scheduler placement, register padding and barrier
    initialization of generated kernels, against a recomputation from the declared graph."""
    workdir = tmp_path_factory.getbasetemp() / "wdfuzz_decl"
    failures = []
    for seed in range(chunk * 40, chunk * 40 + 40):
        spec = gen.generate(seed)
        failures += [f"seed {seed}: {f}" for f in sched.check_declared(spec, workdir).failures]
    assert not failures, failures[:10]


# --------------------------------------------------------------------------- #
# Minimized regressions of runtime-fuzzer findings (test/warpdrive_fuzz/FINDINGS.md)
# --------------------------------------------------------------------------- #


class _SchedulerInDefaultBudgetGroup(WarpSpecialized):
    """F-A1: warpgroup 0 is declared with the default budget (regs=None) and warp 1 is a hole,
    so the dynamic scheduler lands there; another warpgroup sets regs=160."""

    def __init__(self):
        self.tiles = TileStream(policy=DynamicPersistent())

    @role(warps=0)
    @cute.jit
    def a(self, tiles: Tiles, ctx):
        while tiles.valid:
            tiles.advance()

    @role(warps=range(4, 8), regs=160)
    @cute.jit
    def b(self, tiles: Tiles, ctx):
        while tiles.valid:
            tiles.advance()


def test_scheduler_takes_its_warpgroups_default_budget():
    """F-A1: once raised ``warpgroup 0 has mixed register budgets {'a': None,
    'tiles.scheduler': 160}``; the scheduler now inherits its warpgroup's (default) budget."""
    op = _SchedulerInDefaultBudgetGroup()
    text = op.describe()
    assert "scheduler warp 1 broadcasts" in text
    rows = {name: regs for _, name, regs in sched.described_roles(text)}
    assert rows["tiles.scheduler"] == rows["a"]
