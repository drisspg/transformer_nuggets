"""CPU tests for the tcgen05 benchmark variant table, CLI listing, and SASS hot-loop checks."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("typer")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import tcgen05_throughput
from tcgen05_bench.sass import hot_loop
from tcgen05_bench.variants import (
    MmaVariant,
    all_variants,
    mma_variants,
    select_variants,
)
from typer.testing import CliRunner


def test_variant_names_are_unique():
    names = [variant.name for variant in all_variants()]
    assert len(names) == len(set(names))


def test_mma_work_accounting_matches_envelope():
    # Rates divide this per-CTA work by fitted time/cycles, so it must equal the envelope the
    # kernel actually issues: 128 accumulator rows x n_env columns x k_env per CTA.
    for variant in mma_variants():
        if variant.mode != "throughput":
            continue
        rows = variant.m_panels * variant.rows_per_cta
        envelope = 2 * rows * variant.n_env * variant.kind.k_env
        assert variant.work_per_cta_repeat == envelope, variant.name
        assert variant.m_panels * variant.n_env <= 512, variant.name


def test_arch_selection_respects_kind_availability():
    sm100 = {v.name for v in select_variants(all_variants(), arch="sm_100", families=("mma",))}
    sm107 = {v.name for v in select_variants(all_variants(), arch="sm_107", families=("mma",))}
    assert not any("_k64" in name or "_k128" in name or "_k96" in name for name in sm100)
    assert any(".mxf8_e4m3_k64.cg2." in name for name in sm107)
    # sm_107 still probes the SM100-K forms, which is what exposes the K-extent effect.
    assert any(".mxf8_e4m3.cg1." in name for name in sm107)


def test_selection_filters_family_mode_and_regex():
    latency = select_variants(all_variants(), modes=("latency",), pattern=r"^mma\.bf16_f32\.")
    assert latency and all(v.mode == "latency" and isinstance(v, MmaVariant) for v in latency)
    assert all(v.instructions_per_repeat == 1 for v in latency)
    tmem = select_variants(all_variants(), families=("tmem",), exclude=r"\.(ld|st)\.")
    assert tmem and {v.op for v in tmem} == {"cp", "shift", "commit", "fence", "alloc"}


def test_list_cli_runs_without_gpu_dependencies():
    result = CliRunner().invoke(
        tcgen05_throughput.app,
        ["list", "--arch", "sm_103", "--family", "mma", "--select", "cg2", "--json"],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert rows and all(row["cta_group"] == 2 for row in rows)
    assert "cutlass" not in sys.modules


SASS = """
        /*0100*/                   UMOV UR4, 0x0 ;
.L_x_3:
        /*0110*/                   UTCHMMA gdesc[UR4], gdesc[UR6], tmem[UR8], tmem[UR10], idesc[UR11], UPT ;
        /*0120*/                   UTCHMMA gdesc[UR4], gdesc[UR6], tmem[UR8], tmem[UR10], idesc[UR11], UPT ;
        /*0130*/                   {LOAD_EXTRA}
        /*0140*/                   UIADD3 UR5, UPT, UPT, UR5, 0x1, URZ ;
        /*0150*/                   UISETP.NE.U32.AND UP0, UPT, UR5, UR9, UPT ;
        /*0160*/              @UP0 BRA.U `(.L_x_3) ;
        /*0170*/                   EXIT ;
"""


def test_hot_loop_flags_dropped_or_foreign_instructions():
    clean = hot_loop(SASS.replace("{LOAD_EXTRA}", "NOP ;"), "mma", expected_targets=2)
    assert clean.found and clean.clean and clean.target_opcodes == {"UTCHMMA": 2}
    dropped = hot_loop(SASS.replace("{LOAD_EXTRA}", "NOP ;"), "mma", expected_targets=4)
    assert not dropped.clean
    foreign = hot_loop(SASS.replace("{LOAD_EXTRA}", "LDS R2, [R3] ;"), "mma", expected_targets=2)
    assert not foreign.clean and foreign.foreign_opcodes == {"LDS": 1}
