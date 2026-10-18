"""Measure tcgen05 instruction throughput and latency across instruction variants.

The variant table (``tcgen05_bench/variants.py``) covers ``tcgen05.mma`` kinds (f16, tf32,
f8f6f4, i8, mxf8f6f4, mxf4, mxf4nvf4, plus the sm_103 K=96 and sm_107 K=64/K=128 forms), SS and
TS operand sources, cta_group 1 and 2, M/N shape sweeps, 2:4 sparse A, sm_107 operand-collector
reuse, and the non-MMA instructions ``tcgen05.ld/st/cp/shift/commit/fence/alloc``.

Every MMA variant repeats a fixed per-CTA envelope (128 accumulator rows x 256 TMEM columns x 256
SMEM bytes of K; 384 B for K=96) built from one physical instruction shape, so time per repeat
compares instruction forms at equal work. Collector-reuse variants are the exception: they issue
fill/lastuse pairs over two accumulator panels. All operands are zero-filled once before the
repeated region, so absolute rates are an optimistic low-switching roofline under normal DVFS;
per-clock rates come from in-kernel ``clock64`` and do not depend on the clock. Latency variants
issue one instruction followed by its completion wait per repeat from a single CTA.

Variants that the architecture or CuTeDSL rejects are recorded as unsupported. Results stream to
a JSONL file (row schema in ``tcgen05_bench/README.md``). Reruns resume by default: rows already
in the file are skipped, and a variant that killed the previous process is recorded as crashed,
so a driver can rerun the same command after a fault.

Examples (CuTeDSL 4.8+ on sm_100/sm_103/sm_107; run from the repository root):

    python benchmarks/tcgen05_throughput.py list --arch sm_100
    python benchmarks/tcgen05_throughput.py run --select 'mma\\.bf16_f32' --output agent_space/tc.jsonl
    python benchmarks/tcgen05_throughput.py sass --select 'tcgen05\\.ld' --dump-dir agent_space/sass
    python benchmarks/tcgen05_throughput.py summarize agent_space/tc.jsonl
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from tcgen05_bench.variants import (
    ARCHES,
    FAMILIES,
    MODES,
    MmaVariant,
    Variant,
    all_variants,
    arch_from_capability,
    peak_ops_per_clk_sm,
    select_variants,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)

DEFAULT_OUTPUT = Path("agent_space/tcgen05_variants.jsonl")
DEFAULT_ROUNDS = 15
DEFAULT_WARMUP = 3

ArchOption = Annotated[str | None, typer.Option(help="sm_100, sm_103, or sm_107 (default: GPU)")]
FamilyOption = Annotated[
    list[str] | None, typer.Option("--family", help="mma and/or tmem (default: both)")
]
ModeOption = Annotated[
    list[str] | None, typer.Option("--mode", help="throughput and/or latency (default: both)")
]
SelectOption = Annotated[str | None, typer.Option(help="regex on variant names to include")]
ExcludeOption = Annotated[str | None, typer.Option(help="regex on variant names to exclude")]


def describe(variant: Variant) -> dict:
    """Return the static description of a variant."""
    common = {
        "name": variant.name,
        "family": variant.family,
        "mode": variant.mode,
        "work_per_cta_repeat": variant.work_per_cta_repeat,
        "work_unit": variant.work_unit,
    }
    if isinstance(variant, MmaVariant):
        kind = variant.kind
        return {
            **common,
            "kind_key": kind.key,
            "kind": kind.kind,
            "a": kind.a,
            "b": kind.b,
            "acc": kind.acc,
            "sf": kind.sf,
            "sf_vec": kind.sf_vec,
            "scale_vec": kind.scale_vec,
            "op": kind.op,
            "m": variant.m,
            "n": variant.n,
            "k": variant.k,
            "sparse": variant.sparse,
            "collector": variant.collector,
            "k_env": kind.k_env,
            "cta_group": variant.cta_group,
            "a_src": variant.a_src,
            "instructions_per_repeat": variant.instructions_per_repeat,
            "flops_per_instruction": variant.flops_per_instruction,
        }
    return {
        **common,
        "op": variant.op,
        "shape": variant.shape,
        "num": variant.num,
        "warps": variant.warps,
        "registers": variant.registers if variant.op in ("ld", "st") else None,
        "unroll": variant.unroll,
        "bytes_per_instruction": variant.bytes_per_instruction,
    }


def selected(arch, family, mode, select, exclude) -> list[Variant]:
    """Validate the CLI selectors and apply them to the full table."""
    for flag, values, choices in (
        ("--arch", [arch] if arch else [], ARCHES),
        ("--family", family or [], FAMILIES),
        ("--mode", mode or [], MODES),
    ):
        for value in values:
            if value not in choices:
                raise typer.BadParameter(f"{value!r} is not one of {choices}", param_hint=flag)
    for flag, pattern in (("--select", select), ("--exclude", exclude)):
        try:
            re.compile(pattern or "")
        except re.error as error:
            raise typer.BadParameter(f"invalid regex: {error}", param_hint=flag) from error
    return select_variants(
        all_variants(),
        arch=arch,
        families=family or FAMILIES,
        modes=mode or MODES,
        pattern=select,
        exclude=exclude,
    )


@app.command("list")
def list_variants(
    arch: ArchOption = None,
    family: FamilyOption = None,
    mode: ModeOption = None,
    select: SelectOption = None,
    exclude: ExcludeOption = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Print the variant table without touching a GPU."""
    variants = selected(arch, family, mode, select, exclude)
    if as_json:
        print(json.dumps([describe(variant) for variant in variants], indent=1))
        return
    for variant in variants:
        info = describe(variant)
        print(
            f"{info['name']:60s} work/CTA/repeat={info['work_per_cta_repeat']:.3g} {info['work_unit']}"
        )
    print(f"{len(variants)} variants", file=sys.stderr)


def device_context() -> dict:
    """Return GPU, toolchain, and source provenance."""
    import cutlass
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
        check=False,
    ).stdout.strip()
    driver = None
    if shutil.which("nvidia-smi"):
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit or os.environ.get("TCGEN05_BENCH_COMMIT", "unknown"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutedsl": cutlass.__version__,
        "gpu": torch.cuda.get_device_name(),
        "arch": arch_from_capability(*torch.cuda.get_device_capability()),
        "num_sms": properties.multi_processor_count,
        "driver": driver or None,
    }


def compile_variant(variant: Variant):
    """Compile one probe; returns ``(probe, callable)``."""
    from cutlass import cute
    from cutlass.cute.typing import Int64
    from tcgen05_bench.probes import make_probe

    from transformer_nuggets.cute.utils import compile_tvm_ffi, make_fake_compact_tensor

    probe = make_probe(variant)
    output = make_fake_compact_tensor(
        Int64, (cute.sym_int(),), stride_order=(0,), assumed_align=16
    )
    return probe, compile_tvm_ffi(probe, output, 1, 1)


def check_sass(variant: Variant, dump_dir: Path) -> dict:
    """Disassemble the variant's cubin and classify its hot loop."""
    from tcgen05_bench.probes import kernel_name
    from tcgen05_bench.sass import disassemble, find_cubin, hot_loop

    cubin = find_cubin(dump_dir, kernel_name(variant))
    if cubin is None:
        return {"found": False, "error": "cubin not found"}
    if isinstance(variant, MmaVariant):
        op, expected = "mma", variant.instructions_per_repeat
    else:
        op = variant.op
        expected = variant.unroll if op in ("ld", "st", "cp", "shift") else None
    report = hot_loop(disassemble(cubin), op, expected).to_json()
    report["cubin"] = str(cubin)
    return report


def derived(variant: Variant, measurement, arch: str) -> dict:
    """Return headline rates for one measured variant."""
    one_wave = next(grid for grid in measurement.grids if grid.grid_multiplier == 1)
    best = measurement.best()
    result = {
        "best_grid_multiplier": best.grid_multiplier,
        "work_per_s": best.work_per_s,
        "work_per_clk_sm": one_wave.work_per_clk_cta,
        "one_wave_distinct_sms": one_wave.distinct_sms,
        "one_wave_ctas": one_wave.ctas,
        "effective_mhz": one_wave.effective_mhz,
        "cycles_per_repeat": one_wave.cycles.slope,
        "us_per_repeat": one_wave.time_us.slope,
        "min_r_squared": min(
            min(g.time_us.r_squared, g.cycles.r_squared) for g in measurement.grids
        ),
    }
    if variant.mode == "latency":
        issues = (
            variant.instructions_per_repeat if isinstance(variant, MmaVariant) else variant.unroll
        )
        result["latency_cycles"] = one_wave.cycles.slope / issues
        result["latency_ns"] = one_wave.time_us.slope * 1e3 / issues
    peak = peak_ops_per_clk_sm(variant, arch)
    if peak and variant.mode == "throughput":
        result["peak_per_clk_sm"] = peak
        result["efficiency_vs_peak"] = one_wave.work_per_clk_cta / peak
    return result


def load_done(path: Path) -> set[str]:
    """Return variant names already present in a JSONL result file."""
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if "name" in row:
                names.add(row["name"])
    return names


def strict_json(value):
    """Replace NaN/inf (undefined rates) with None so rows stay valid strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [strict_json(item) for item in value]
    return value


def append(path: Path, row: dict) -> None:
    """Append one JSON row and flush it to disk."""
    with path.open("a") as handle:
        handle.write(json.dumps(strict_json(row), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@app.command()
def run(
    arch: ArchOption = None,
    family: FamilyOption = None,
    mode: ModeOption = None,
    select: SelectOption = None,
    exclude: ExcludeOption = None,
    output: Annotated[Path, typer.Option(help="JSONL results file (one row per variant)")] = (
        DEFAULT_OUTPUT
    ),
    rounds: Annotated[int, typer.Option(min=3, help="timed replays per graph")] = DEFAULT_ROUNDS,
    warmup: Annotated[int, typer.Option(min=1)] = DEFAULT_WARMUP,
    seed: Annotated[
        int, typer.Option(help="shuffle seed for variant order; use one --output per trial")
    ] = 0,
    resume: Annotated[
        bool,
        typer.Option(help="skip variants with any row in OUTPUT; --no-resume overwrites OUTPUT"),
    ] = True,
    sass_dir: Annotated[Path | None, typer.Option(help="keep cubins and check hot loops")] = None,
    variant_timeout: Annotated[
        float, typer.Option(help="seconds per variant before the process exits (hang guard)")
    ] = 300.0,
) -> None:
    """Compile, check, and measure the selected variants, streaming JSONL rows.

    A watchdog thread exits the process with status 124 when one variant exceeds
    ``--variant-timeout`` (a hung kernel blocks the main thread inside CUDA); rerunning the same
    command records that variant as crashed and continues.
    """
    if sass_dir is not None:
        os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
        os.environ["CUTE_DSL_DUMP_DIR"] = str(sass_dir.resolve())
        sass_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from tcgen05_bench.measure import (
        LATENCY_REPEATS,
        MMA_REPEATS,
        TMEM_REPEATS,
        NvmlSampler,
        measure_probe,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("tcgen05 probes require an sm_10x CUDA GPU")
    context = device_context()
    if arch not in (None, context["arch"]):
        raise typer.BadParameter(f"the GPU is {context['arch']}", param_hint="--arch")
    arch = context["arch"]
    variants = selected(arch, family, mode, select, exclude)
    by_name = {variant.name: variant for variant in variants}
    random.Random(seed).shuffle(variants)
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = output.with_suffix(output.suffix + ".inprogress")
    done = load_done(output) if resume else set()
    if not resume and output.exists():
        output.unlink()
    if marker.exists() and resume:
        crashed = marker.read_text().strip()
        if crashed and crashed not in done:
            row = describe(by_name[crashed]) if crashed in by_name else {"name": crashed}
            append(output, {**row, "status": "crashed", "context": context})
            done.add(crashed)
    marker.unlink(missing_ok=True)

    sampler = NvmlSampler(
        f"GPU-{torch.cuda.get_device_properties(torch.cuda.current_device()).uuid}"
    )
    sampler.start()
    heartbeat = [time.monotonic()]
    watchdog = threading.Thread(
        target=_watchdog, args=(heartbeat, variant_timeout, marker), daemon=True
    )
    watchdog.start()
    try:
        for index, variant in enumerate(variants):
            if variant.name in done:
                continue
            heartbeat[0] = time.monotonic()
            marker.write_text(variant.name)
            row = {**describe(variant), "context": context, "seed": seed}
            started = time.monotonic()
            try:
                probe, callable_ = compile_variant(variant)
            except Exception as error:  # noqa: BLE001 - any rejection means unsupported here
                row.update(status="unsupported", error=f"{type(error).__name__}: {error}"[:2000])
                append(output, row)
                print(
                    f"[{index + 1}/{len(variants)}] {variant.name}: unsupported ({type(error).__name__})"
                )
                continue
            row["compile_s"] = time.monotonic() - started
            row["smem_bytes"] = getattr(probe, "smem_bytes", None)
            row["tmem_columns"] = getattr(probe, "tmem_columns", None)
            if sass_dir is not None:
                try:
                    row["sass"] = check_sass(variant, sass_dir)
                except Exception as error:  # noqa: BLE001 - SASS checks must not stop timing
                    row["sass"] = {"found": False, "error": repr(error)}
            is_mma = isinstance(variant, MmaVariant)
            cluster = variant.cta_group if is_mma else 1
            latency = variant.mode == "latency"
            repeats = LATENCY_REPEATS if latency else MMA_REPEATS if is_mma else TMEM_REPEATS
            try:
                measurement = measure_probe(
                    callable_,
                    cluster_size=cluster,
                    work_per_cta_repeat=variant.work_per_cta_repeat,
                    num_sms=context["num_sms"],
                    latency=latency,
                    repeats=repeats,
                    rounds=rounds,
                    warmup=warmup,
                    sampler=sampler,
                )
            except Exception as error:
                row.update(status="failed", error=traceback.format_exc()[-2000:])
                append(output, row)
                if "CUDA error" in str(error):
                    raise  # the CUDA context is unusable; --resume skips this variant
                print(f"[{index + 1}/{len(variants)}] {variant.name}: failed ({error!r})")
                continue
            row.update(
                status="ok",
                measurement=measurement.to_json(),
                **derived(variant, measurement, arch),
            )
            append(output, row)
            print(format_row(row))
    finally:
        sampler.stop()
    marker.unlink(missing_ok=True)


def _watchdog(heartbeat: list[float], timeout: float, marker: Path) -> None:
    """Exit the process when the current variant has run longer than ``timeout`` seconds."""
    while True:
        time.sleep(1.0)
        if time.monotonic() - heartbeat[0] > timeout:
            name = marker.read_text().strip() if marker.exists() else "?"
            print(f"watchdog: {name} exceeded {timeout:.0f} s; exiting", flush=True)
            os._exit(124)


def format_row(row: dict) -> str:
    """Return a one-line summary of a JSONL row."""
    if row.get("status") != "ok":
        return f"{row['name']}: {row.get('status')}"
    unit = row["work_unit"]
    text = f"{row['name']:58s} "
    if row["mode"] == "latency":
        text += f"lat={row['latency_cycles']:8.1f} cyc ({row['latency_ns']:.1f} ns)"
    elif unit == "instr":
        text += f"{1 / row['work_per_clk_sm']:9.1f} cyc/instr/SM (issue interval)"
    else:
        label = "TB/s" if unit == "byte" else f"T{unit}/s"
        text += (
            f"{row['work_per_s'] / 1e12:9.2f} {label}  {row['work_per_clk_sm']:9.1f} {unit}/clk/SM"
        )
        if "efficiency_vs_peak" in row:
            text += f"  {100 * row['efficiency_vs_peak']:5.1f}% peak"
    if row["effective_mhz"] is not None and math.isfinite(row["effective_mhz"]):
        text += f"  {row['effective_mhz']:.0f} MHz"
    sass = row.get("sass")
    if sass is not None:
        text += "  sass=" + ("clean" if sass.get("clean") else "CHECK")
    return text


@app.command()
def sass(
    arch: ArchOption = None,
    family: FamilyOption = None,
    mode: ModeOption = None,
    select: SelectOption = None,
    exclude: ExcludeOption = None,
    dump_dir: Annotated[Path, typer.Option()] = Path("agent_space/tcgen05_sass"),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Compile selected variants (set CUTE_DSL_ARCH to cross-compile) and check their hot loops."""
    os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir.resolve())
    dump_dir.mkdir(parents=True, exist_ok=True)
    arch = arch or os.environ.get("CUTE_DSL_ARCH", "").rstrip("af") or None
    if arch is None:
        import torch

        arch = arch_from_capability(*torch.cuda.get_device_capability())
    rows = []
    for variant in selected(arch, family, mode, select, exclude):
        try:
            compile_variant(variant)
        except Exception as error:  # noqa: BLE001 - any rejection means unsupported here
            rows.append({"name": variant.name, "status": "unsupported", "error": str(error)[:500]})
            print(f"{variant.name}: unsupported ({type(error).__name__}: {str(error)[:120]})")
            continue
        report = check_sass(variant, dump_dir)
        rows.append({"name": variant.name, "status": "ok", "sass": report})
        state = (
            "clean" if report.get("clean") else f"CHECK foreign={report.get('foreign_opcodes')}"
        )
        print(f"{variant.name}: {report.get('target_opcodes')} {state}")
    if output is not None:
        output.write_text(json.dumps(rows, indent=1) + "\n")


@app.command()
def summarize(paths: list[Path]) -> None:
    """Print the measured rows of one or more JSONL files."""
    for path in paths:
        print(f"\n{path}")
        for line in path.read_text().splitlines():
            if line.strip():
                print("  " + format_row(json.loads(line)))


if __name__ == "__main__":
    app()
