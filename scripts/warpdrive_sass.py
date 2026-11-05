"""Dump or compare normalized SASS for warpdrive's seven standard configurations.

    CUTE_DSL_ARCH=sm_100a python scripts/warpdrive_sass.py dump agent_space/sass/before
    python scripts/warpdrive_sass.py compare agent_space/sass/before agent_space/sass/after

Dumping requires CuTeDSL, flash_attn.cute, nvdisasm and an empty output directory. Compare
needs only Python. Persistent GEMM pins tile_n=256 so default-tile changes do not alter the gate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def dump(output: Path, labels: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"{output}: dump needs an empty output directory")
    # Keep compare usable without the compiler or optional example dependencies.
    from cutlass._mlir._mlir_libs._cutlass_ir import _aot_support

    import transformer_nuggets
    from transformer_nuggets.cute.warpdrive import (
        ClusterLaunchControl,
        CtaPair,
        StaticPersistent,
    )
    from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import (
        FlashAttentionForward,
        compile_fa4_fwd,
    )
    from transformer_nuggets.cute.warpdrive.examples.gemm import Gemm, compile_gemm
    from transformer_nuggets.cute.warpdrive.examples.persistent_gemm import (
        PersistentGemm,
        compile_persistent_gemm,
    )

    configs = {
        "gemm_oneshot": (compile_gemm, Gemm()),
        "pgemm_static_single": (
            compile_persistent_gemm,
            PersistentGemm(schedule=StaticPersistent(), tile_n=256),
        ),
        "pgemm_static_pair": (
            compile_persistent_gemm,
            PersistentGemm(schedule=StaticPersistent(), cluster=CtaPair(), tile_n=256),
        ),
        "pgemm_clc_pair": (
            compile_persistent_gemm,
            PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair(), tile_n=256),
        ),
        "fa4_single": (compile_fa4_fwd, FlashAttentionForward()),
        "fa4_clc_pair": (
            compile_fa4_fwd,
            FlashAttentionForward(schedule=ClusterLaunchControl(), cluster=CtaPair()),
        ),
        "fa4_clc_pair_splitp": (
            compile_fa4_fwd,
            FlashAttentionForward(
                schedule=ClusterLaunchControl(), cluster=CtaPair(), split_p=True
            ),
        ),
    }
    unknown = set(labels) - configs.keys()
    if unknown:
        raise ValueError(f"unknown configurations: {sorted(unknown)}; choose {list(configs)}")
    print(transformer_nuggets.__file__, flush=True)
    for label in labels or configs:
        build, op = configs[label]
        compiled = build(op)
        obj = bytes(
            _aot_support.export_module_to_bytes(
                compiled.ir_module, format="o", opt_level=3, enable_pic=True
            )
        )
        # The first ELF is the host object; each embedded ELF is a device cubin.
        starts = [match.start() for match in re.finditer(b"\x7fELF", obj)]
        if len(starts) < 2:
            raise RuntimeError(f"{label}: compiled object contains no device cubin")
        for index, (start, end) in enumerate(zip(starts[1:], [*starts[2:], len(obj)]), 1):
            cubin = output / f"{label}_{index}.cubin"
            cubin.write_bytes(obj[start:end])
            sass = subprocess.check_output(["nvdisasm", str(cubin)], text=True)
            sass = re.sub(r"/\*[0-9a-f]{4,}\*/", "", sass)
            sass = re.sub(r"cutlass_[A-Za-z0-9_]+", "KERNEL", sass)
            cubin.with_suffix(".sass").write_text(sass)
        print(f"{label}: {len(starts) - 1} cubin(s)", flush=True)


def compare(before: Path, after: Path) -> int:
    files_before = {file.name: file for file in before.glob("*.sass")}
    files_after = {file.name: file for file in after.glob("*.sass")}
    if not files_before or not files_after:
        raise ValueError("both directories must contain normalized .sass files")
    different = 0
    for name in sorted(files_before.keys() | files_after.keys()):
        if name not in files_before:
            verdict = "only after"
        elif name not in files_after:
            verdict = "only before"
        elif files_before[name].read_bytes() != files_after[name].read_bytes():
            verdict = "different"
        else:
            verdict = "identical"
        print(f"{name}: {verdict}")
        different += verdict != "identical"
    return int(different != 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dump_parser = commands.add_parser("dump", help="compile and disassemble standard configs")
    dump_parser.add_argument("output", type=Path)
    dump_parser.add_argument("labels", nargs="*", help="default: all seven configurations")
    compare_parser = commands.add_parser("compare", help="compare two normalized SASS dumps")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # CuTeDSL re-parses application arguments at import/compile time.
    if args.command == "compare":
        return compare(args.before, args.after)
    dump(args.output, args.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
