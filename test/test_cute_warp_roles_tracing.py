"""IKET ranges emitted by the warp-role layer and the PipelinePlan exported for them: stable names,
no IKET ops when disabled, and an end-to-end join through the profiler on a synthetic capture
(compile only; no GPU or run-iket needed)."""

import json

import pytest

pytest.importorskip("cutlass.cute")

from transformer_nuggets.cute import cache
from transformer_nuggets.cute.profiler.pipeline import (
    analyze_pipeline,
    load_iket_capture,
    write_pipeline_perfetto,
)
from transformer_nuggets.cute.warp_roles import Clc, CtaPair, StaticPersistent, tracing
from transformer_nuggets.cute.warp_roles.examples.persistent_gemm import (
    PersistentGemm,
    compile_persistent_gemm,
)
from transformer_nuggets.cute.warp_roles.pipeline_plan import plan_from_warp_roles


class _Recorder:
    """Stands in for ``cutlass.cute.experimental.iket``: records pushed names, emits the real
    ops (so the compile is the real instrumented one)."""

    def __init__(self, real):
        self.real, self.pushed, self.pops = real, [], 0

    def range_push(self, name):
        self.pushed.append(name)
        return self.real.range_push(name) if self.real else None

    def range_pop(self):
        self.pops += 1
        return self.real.range_pop() if self.real else None


@pytest.fixture
def iket_recorder(monkeypatch):
    real = tracing.load_iket()
    rec = _Recorder(real)
    monkeypatch.setattr(tracing, "load_iket", lambda: rec)
    cache._kernel_cache.clear()  # force a trace even if another test compiled the same key
    return rec


def _compile_gemm(op):
    compile_persistent_gemm(op)
    return op


def test_disabled_emits_no_iket_ops(monkeypatch):
    def forbidden():
        raise AssertionError("IKET op emitted with IKET disabled")

    monkeypatch.setattr(tracing, "load_iket", forbidden)
    cache._kernel_cache.clear()
    op = _compile_gemm(PersistentGemm(policy=Clc(), cluster=CtaPair()))
    assert op.iket_key() == "" and "iket" not in op.get_name()


def test_gemm_range_names_match_plan(iket_recorder):
    op = _compile_gemm(PersistentGemm(policy=StaticPersistent()).enable_iket())
    plan = plan_from_warp_roles(op)
    names = set(iket_recorder.pushed)
    assert set(plan.roles) == set(plan.regions) == {"epilogue", "load", "mma"}
    assert set(plan.regions) <= names  # every plan region is an emitted range
    assert names == {"epilogue", "load", "mma", "ab.acquire", "ab.wait", "acc.acquire", "acc.wait"}
    assert set(plan.resources) == {"ab", "acc"}
    # Every traced push has a pop: wait ranges, each item's range (closed by advance()), and the
    # last item's range a loop exit other than `valid` would leave open (closed at role exit).
    assert len(iket_recorder.pushed) == iket_recorder.pops
    deps = {(d.source, d.target, d.resource, d.kind, d.distance) for d in plan.dependencies}
    # ab is used inside the K loop (reuse distance 1); acc once per tile (2 stages -> 2).
    assert deps == {
        ("load", "mma", "ab", "data", 0),
        ("mma", "load", "ab", "reuse", 1),
        ("mma", "epilogue", "acc", "data", 0),
        ("epilogue", "mma", "acc", "reuse", 2),
    }
    assert op.get_name().endswith("_iket")


def test_pair_plan_per_cta_and_roles_only(iket_recorder):
    op = _compile_gemm(PersistentGemm(policy=Clc(), cluster=CtaPair()).enable_iket(waits=False))
    names = set(iket_recorder.pushed)
    assert names == {"epilogue", "load", "mma", "tiles.scheduler"}  # waits=False: no waits
    leader, peer = plan_from_warp_roles(op, cta=0), plan_from_warp_roles(op, cta=1)
    assert set(leader.regions) == names
    assert set(peer.regions) == {"epilogue", "load"}  # leader-only roles are not in the peer
    assert (
        leader.roles["mma"].label == "mma (leader)" and peer.roles["load"].label == "load (peer)"
    )
    assert op.get_name().endswith("_iketroles")


def test_fa4_pair_names_and_feedback_edges():
    pytest.importorskip("flash_attn.cute")
    from transformer_nuggets.cute.warp_roles.examples.fa4_fwd import FlashAttentionForward

    plan = plan_from_warp_roles(FlashAttentionForward(policy=Clc(), cluster=CtaPair()))
    assert list(plan.roles) == [
        "softmax[0]",
        "softmax[1]",
        "correction",
        "mma",
        "epilogue",
        "load",
        "tiles.scheduler",
    ]
    feedback = {(d.source, d.target) for d in plan.dependencies if d.kind == "state"}
    assert ("softmax[0]", "mma") in feedback and ("softmax[1]", "mma") in feedback  # p_last
    reuse = {
        (d.source, d.target, d.resource): d.distance
        for d in plan.dependencies
        if d.kind == "reuse"
    }
    # stats: softmax[j] is bound to slot j, so it reuses one slot (distance 1, not 2).
    assert reuse[("correction", "softmax[0]", "stats")] == 1
    # The work ring's join completes with its last consumer in data order (epilogue); slots
    # are released on receipt, one item before the holding region ends (2 stages + 1).
    assert [k for k in reuse if k[2] == "tiles_work"] == [
        ("epilogue", "tiles.scheduler", "tiles_work")
    ]
    assert reuse[("epilogue", "tiles.scheduler", "tiles_work")] == 3
    plan.schedule(iterations=4)  # no same-iteration cycle


def _synthetic_capture(path, plan, iterations, warps_per_role):
    """An IKET JSON with one range per role, work item, and warp, as the layer names them."""
    strings, ranges, locations = [], [], []
    t = 1_000
    for it in range(iterations):
        for role in plan.roles.values():
            for warp in range(role.warp_start, role.warp_end)[:warps_per_role]:
                locations.append({"ctaId": [0, 0, 0], "warpId": warp})
                if role.name not in strings:
                    strings.append(role.name)
                ranges.append(
                    {
                        "rangeNameIdx": strings.index(role.name),
                        "warpLocIdxs": [len(locations) - 1],
                        "startTs": t,
                        "endTs": t + 900,
                    }
                )
            t += 100
        t += 1_000
    payload = {
        "launches": [{"kernelName": "synthetic", "ranges": ranges}],
        "stringTable": strings,
        "locationTable": locations,
    }
    path.write_text(json.dumps(payload))


def test_plan_joins_capture_end_to_end(tmp_path, iket_recorder):
    op = _compile_gemm(PersistentGemm(policy=Clc(), cluster=CtaPair()).enable_iket())
    plan = plan_from_warp_roles(op, cta=0)
    timeline = plan.schedule(iterations=3)
    trace = tmp_path / "capture.trace.json"
    _synthetic_capture(trace, plan, 3, warps_per_role=2)
    measured = load_iket_capture(trace, timeline)
    analysis = analyze_pipeline(timeline, measured)
    write_pipeline_perfetto(tmp_path / "pipeline.pftrace", timeline, measured, analysis=analysis)
    assert {r.role for r in measured.regions} == set(plan.roles)
    assert (tmp_path / "pipeline.pftrace").stat().st_size > 0
