"""Data-driven tcgen05 variant table.

This module is pure Python so the table, its legality filters, and the CLI selectors can be
tested without CuTeDSL or a GPU. Every variant fixes one physical instruction form plus the
fixed-work envelope that each CTA repeats; the probes in ``probes.py`` turn a variant into a
kernel and the measurement core fits time and cycles per repeat.

Legality here is a coarse prefilter from the PTX ISA tables. The CuTeDSL op constructors are
the final authority: a variant that fails to construct or compile is recorded as unsupported.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

ARCHES = ("sm_100", "sm_103", "sm_107")
FAMILIES = ("mma", "tmem")
MODES = ("throughput", "latency")
ALL = ARCHES
SM100_FAMILY = ("sm_100", "sm_103")

# Envelope K extent in SMEM bytes per operand row. 256 B is two 128B swizzle atoms, so every
# SM100-K instruction appears 8 times per panel. K=96 (sm_103 MXF4) needs a multiple of 96.
ENVELOPE_K_BYTES = 256
ENVELOPE_K_BYTES_K96 = 384
# Each CTA owns 128 accumulator rows; the N extent fills 256 TMEM columns.
ENVELOPE_ROWS = 128
ENVELOPE_COLUMNS = 256

# Public dense-MMA peaks in FLOP (or integer OP) per clock per SM, derived from the documented
# B200/B300 tensor-core rates. sm_107 has no public per-clock table, so its rows omit
# ``peak_per_clk_sm`` and ``efficiency_vs_peak``.
PEAK_OPS_PER_CLK_SM: dict[str, dict[str, int]] = {
    "sm_100": {
        "f16": 8192,
        "tf32": 4096,
        "f8f6f4": 16384,
        "i8": 16384,
        "mxf8f6f4": 16384,
        "mxf4": 32768,
        "mxf4nvf4": 32768,
    },
    "sm_103": {
        "f16": 8192,
        "tf32": 4096,
        "f8f6f4": 16384,
        "mxf8f6f4": 16384,
        "mxf4": 49152,
        "mxf4nvf4": 49152,
    },
}


@dataclass(frozen=True)
class MmaKind:
    """One ``tcgen05.mma`` kind and dtype combination plus the CuTeDSL op that emits it.

    ``k`` is the dense instruction K, ``arches`` where the op exists, and ``container_bits`` the
    SMEM bits per element (8 for sub-byte types in ``kind::f8f6f4`` containers, 4 for packed FP4).
    ``tier="full"`` sweeps every legal M/N/cta_group shape plus latency; ``"peak"`` measures
    three representative shapes. ``ts`` adds TMEM-A (TS) variants.
    """

    key: str
    kind: str
    a: str
    b: str
    acc: str
    k: int
    op: str
    arches: tuple[str, ...]
    container_bits: int
    sf: str | None = None
    sf_vec: int | None = None
    tier: str = "peak"
    ts: bool = False

    @property
    def block_scaled(self) -> bool:
        """Return whether the kind consumes TMEM scale factors."""
        return self.sf is not None

    @property
    def k_env(self) -> int:
        """Return the envelope K extent in elements."""
        k_bytes = ENVELOPE_K_BYTES_K96 if self.k == 96 else ENVELOPE_K_BYTES
        return k_bytes * 8 // self.container_bits

    @property
    def scale_vec(self) -> str | None:
        """Return the PTX scale_vec label (1X/2X/4X) for block-scaled kinds."""
        if self.sf_vec is None:
            return None
        return f"{self.k // self.sf_vec}X"


def _dense(key, kind, a, b, acc, k, op, arches, bits, tier="peak", ts=False) -> MmaKind:
    return MmaKind(key, kind, a, b, acc, k, op, arches, bits, tier=tier, ts=ts)


def _scaled(key, kind, a, b, k, op, arches, bits, sf, vec, tier="peak", ts=False) -> MmaKind:
    return MmaKind(key, kind, a, b, "f32", k, op, arches, bits, sf, vec, tier, ts)


# Short aliases keep one kind per table row.
FULL, TS = "full", True
R, B3 = ("sm_107",), ("sm_103",)
F16, TF32, F8, I8 = "MmaF16BF16Op", "MmaTF32Op", "MmaF8F6F4Op", "MmaI8Op"
MX8, MX4, NV4 = "MmaMXF8F6F4Op", "MmaMXF4Op", "MmaMXF4NVF4Op"
MX4_103, NV4_103 = "SM103MmaMXF4Op", "SM103MmaMXF4NVF4Op"
F8_107, MX8_107, NV4_107 = "SM107MmaFP8Op", "SM107BlockScaledMmaMXF8F6F4Op", "SM107MmaMXF4NVF4Op"
E8, U4 = "ue8m0", "ue4m3"

# fmt: off
MMA_KINDS: tuple[MmaKind, ...] = (
    # kind::f16 and kind::tf32
    _dense("bf16_f32", "f16", "bf16", "bf16", "f32", 16, F16, ALL, 16, FULL, TS),
    _dense("f16_f32", "f16", "f16", "f16", "f32", 16, F16, ALL, 16),
    _dense("f16_f16", "f16", "f16", "f16", "f16", 16, F16, ALL, 16),
    _dense("tf32_f32", "tf32", "tf32", "tf32", "f32", 8, TF32, ALL, 32, FULL, TS),
    # kind::f8f6f4, non-scaled, 8-bit SMEM containers, SM100 K=32
    _dense("e4m3_f32", "f8f6f4", "e4m3", "e4m3", "f32", 32, F8, ALL, 8, FULL, TS),
    _dense("e5m2_f32", "f8f6f4", "e5m2", "e5m2", "f32", 32, F8, ALL, 8),
    _dense("e4m3_f16", "f8f6f4", "e4m3", "e4m3", "f16", 32, F8, ALL, 8),
    _dense("e2m3_f32", "f8f6f4", "e2m3", "e2m3", "f32", 32, F8, ALL, 8),
    _dense("e3m2_f32", "f8f6f4", "e3m2", "e3m2", "f32", 32, F8, ALL, 8),
    _dense("e2m1_f32", "f8f6f4", "e2m1", "e2m1", "f32", 32, F8, ALL, 8),
    _dense("e4m3xe2m1_f32", "f8f6f4", "e4m3", "e2m1", "f32", 32, F8, ALL, 8),
    # sm_107 FP8 with the doubled K extent
    _dense("e4m3_f32_k64", "f8f6f4", "e4m3", "e4m3", "f32", 64, F8_107, R, 8, FULL, TS),
    _dense("e5m2_f32_k64", "f8f6f4", "e5m2", "e5m2", "f32", 64, F8_107, R, 8),
    _dense("e4m3_f16_k64", "f8f6f4", "e4m3", "e4m3", "f16", 64, F8_107, R, 8),
    # kind::i8
    _dense("s8_s32", "i8", "s8", "s8", "s32", 32, I8, ALL, 8, FULL),
    _dense("u8_s32", "i8", "u8", "u8", "s32", 32, I8, ALL, 8),
    # kind::mxf8f6f4 (UE8M0, 32-element blocks, scale_vec::1X), SM100 K=32
    _scaled("mxf8_e4m3", "mxf8f6f4", "e4m3", "e4m3", 32, MX8, ALL, 8, E8, 32, FULL, TS),
    _scaled("mxf8_e5m2", "mxf8f6f4", "e5m2", "e5m2", 32, MX8, ALL, 8, E8, 32),
    _scaled("mxf6_e2m3", "mxf8f6f4", "e2m3", "e2m3", 32, MX8, ALL, 8, E8, 32),
    _scaled("mxf4_in_f8", "mxf8f6f4", "e2m1", "e2m1", 32, MX8, ALL, 8, E8, 32),
    _scaled("mxf8xmxf4", "mxf8f6f4", "e4m3", "e2m1", 32, MX8, ALL, 8, E8, 32),
    # sm_107 MXF8F6F4 with K=64 (two scale factors per row per instruction)
    _scaled("mxf8_e4m3_k64", "mxf8f6f4", "e4m3", "e4m3", 64, MX8_107, R, 8, E8, 32, FULL, TS),
    _scaled("mxf6_e2m3_k64", "mxf8f6f4", "e2m3", "e2m3", 64, MX8_107, R, 8, E8, 32),
    _scaled("mxf4_in_f8_k64", "mxf8f6f4", "e2m1", "e2m1", 64, MX8_107, R, 8, E8, 32),
    _scaled("mxf8xmxf4_k64", "mxf8f6f4", "e4m3", "e2m1", 64, MX8_107, R, 8, E8, 32),
    # kind::mxf4 (packed E2M1, UE8M0, scale_vec::2X) and kind::mxf4nvf4 (UE4M3, 4X)
    _scaled("mxf4", "mxf4", "e2m1", "e2m1", 64, MX4, ALL, 4, E8, 32, FULL, TS),
    _scaled("nvf4", "mxf4nvf4", "e2m1", "e2m1", 64, NV4, ALL, 4, U4, 16, FULL, TS),
    # sm_103 K=96 FP4 (circular SMEM descriptors)
    _scaled("mxf4_k96", "mxf4", "e2m1", "e2m1", 96, MX4_103, B3, 4, E8, 32, FULL),
    _scaled("nvf4_k96", "mxf4nvf4", "e2m1", "e2m1", 96, NV4_103, B3, 4, U4, 16, FULL),
    # sm_107 K=128 FP4
    _scaled("nvf4_k128", "mxf4nvf4", "e2m1", "e2m1", 128, NV4_107, R, 4, U4, 16, FULL, TS),
    _scaled("mxf4_k128", "mxf4nvf4", "e2m1", "e2m1", 128, NV4_107, R, 4, E8, 32),
)
# fmt: on

KIND_BY_KEY = {kind.key: kind for kind in MMA_KINDS}
# Kinds measured with structured-sparse A via raw tcgen05.mma.sp (SM100 instruction descriptor).
SPARSE_KINDS = ("bf16_f32", "e4m3_f32")


@dataclass(frozen=True)
class MmaVariant:
    """One physical ``tcgen05.mma`` instruction shape and its fixed-work envelope."""

    kind: MmaKind
    m: int
    n: int
    cta_group: int
    a_src: str = "smem"
    mode: str = "throughput"
    # Operand collector reuse (sm_107 ops): "b" keeps B for a second M panel, "a" keeps A for a
    # second N panel; each K step issues the pair as collector fill then lastuse. Both
    # instructions of a pair read the same SMEM A and B addresses and write separate accumulators.
    collector: str = ""
    # Structured 2:4 sparse A (tcgen05.mma.sp): K per instruction doubles at equal operand bytes.
    sparse: bool = False

    family = "mma"

    @property
    def name(self) -> str:
        """Return the stable variant label used in tables and kernel names."""
        src = "ss" if self.a_src == "smem" else "ts"
        src += ".sp" if self.sparse else ""
        suffix = f".{self.collector}reuse" if self.collector else ""
        suffix += "" if self.mode == "throughput" else f".{self.mode}"
        return f"mma.{self.kind.key}.cg{self.cta_group}.{src}.m{self.m}n{self.n}k{self.k}{suffix}"

    @property
    def k(self) -> int:
        """Return the instruction K (doubled for sparse A)."""
        return self.kind.k * (2 if self.sparse else 1)

    @property
    def rows_per_cta(self) -> int:
        """Return accumulator rows owned by each CTA for one instruction."""
        return self.m // self.cta_group

    @property
    def m_panels(self) -> int:
        """Return accumulator panels stacked to fill the 128-row envelope (2 for B reuse)."""
        if self.collector == "b":
            return 2
        return ENVELOPE_ROWS // self.rows_per_cta

    @property
    def n_env(self) -> int:
        """Return the N extent covered by each panel row of the envelope."""
        if self.collector:
            return self.n * (2 if self.collector == "a" else 1)
        return max(self.n, ENVELOPE_COLUMNS // self.m_panels)

    @property
    def n_panels(self) -> int:
        """Return instruction-N panels per envelope row."""
        return 1 if self.mode == "latency" else self.n_env // self.n

    @property
    def k_instructions(self) -> int:
        """Return K-direction instructions per panel."""
        return 1 if self.mode == "latency" else self.kind.k_env // self.k

    @property
    def instructions_per_repeat(self) -> int:
        """Return instructions the leader CTA issues per repeat."""
        panels = 1 if self.mode == "latency" else self.m_panels * self.n_panels
        return panels * self.k_instructions

    @property
    def flops_per_instruction(self) -> int:
        """Return multiply-add FLOPs (or integer OPs) of one instruction across its CTA group."""
        return 2 * self.m * self.n * self.k

    @property
    def work_per_cta_repeat(self) -> float:
        """Return FLOPs per CTA per repeat, splitting cta_group::2 work across both CTAs."""
        return self.instructions_per_repeat * self.flops_per_instruction / self.cta_group

    @property
    def work_unit(self) -> str:
        """Return the work unit used by derived rates."""
        return "op" if self.kind.kind == "i8" else "flop"


@dataclass(frozen=True)
class TmemVariant:
    """One non-MMA tcgen05 instruction form and its repeated body."""

    op: str
    shape: str = ""
    num: int = 1
    warps: int = 4
    mode: str = "throughput"
    inflight: int = 1

    family = "tmem"

    @property
    def name(self) -> str:
        """Return the stable variant label used in tables and kernel names."""
        parts = [f"tcgen05.{self.op}"]
        if self.shape:
            parts.append(self.shape)
        if self.op in ("ld", "st"):
            parts.append(f"x{self.num}")
            parts.append(f"k{self.inflight}")
        if self.op == "alloc":
            parts.append(f"cols{self.num}")
        parts.append(f"w{self.warps}")
        if self.mode != "throughput":
            parts.append(self.mode)
        return ".".join(parts)

    @property
    def registers(self) -> int:
        """Return 32-bit registers per thread for tcgen05.ld/st."""
        return self.num * LDST_REGISTER_FACTOR.get(self.shape, 1)

    @property
    def unroll(self) -> int:
        """Return instructions issued per repeat by each issuing warp or thread."""
        if self.mode == "latency" or self.op in ("alloc", "fence", "commit"):
            return 1
        return 4 if self.op in ("ld", "st") else 8

    @property
    def issuing_warps(self) -> int:
        """Return warps that issue the measured instruction."""
        return self.warps if self.op in ("ld", "st", "fence") else 1

    @property
    def bytes_per_instruction(self) -> int:
        """Return bytes moved by one warp-level instruction (SMEM source bytes for cp)."""
        if self.op in ("ld", "st"):
            return 32 * self.registers * 4
        if self.op == "cp":
            return CP_SHAPES[self.shape][0]
        return 0

    @property
    def work_per_cta_repeat(self) -> float:
        """Return bytes (ld/st/cp) or instructions (others) per CTA per repeat."""
        count = self.unroll * self.issuing_warps
        return count * (self.bytes_per_instruction or 1)

    @property
    def work_unit(self) -> str:
        """Return the work unit used by derived rates."""
        return "byte" if self.bytes_per_instruction else "instr"


Variant = MmaVariant | TmemVariant

# tcgen05.ld/st shapes: registers per thread per repetition and the legal repetition range.
LDST_REGISTER_FACTOR = {"32x32b": 1, "16x64b": 1, "16x128b": 2, "16x256b": 4}
LDST_MAX_NUM = {"32x32b": 128, "16x64b": 128, "16x128b": 64, "16x256b": 32}
LDST_NUMS = (1, 4, 16, 32, 64, 128)
# tcgen05.cp shapes: (SMEM source bytes, TMEM bytes written incl. multicast).
CP_SHAPES = {
    "128x256b": (4096, 4096),
    "128x128b": (2048, 2048),
    "4x256b": (128, 128),
    "32x128b.warpx4": (512, 2048),
    "64x128b.warpx2::02_13": (1024, 2048),
    "64x128b.warpx2::01_23": (1024, 2048),
}
ALLOC_COLUMNS = (32, 128, 256, 512)
# Registers per thread held by one batch of in-flight tcgen05.ld/st (keeps ptxas from spilling).
LDST_MAX_INFLIGHT_REGISTERS = 128


def max_warps(registers: int) -> int:
    """Return the most warps (4, 8, or 16) whose data registers fit a 48K-register budget."""
    return max(w for w in (4, 8, 16) if registers * w * 32 <= 48 * 1024)


def _legal_n(kind: MmaKind, m: int, cta_group: int) -> Iterable[int]:
    """Yield the sweep's N values that satisfy the PTX N step for this M and CTA group."""
    if cta_group == 2:
        candidates = (32, 64, 128, 256)
    elif kind.block_scaled:
        candidates = (8, 16, 32, 64, 128, 256)
    elif m == 64:
        candidates = (8, 16, 32, 64, 128)
    else:
        candidates = (16, 32, 64, 128, 256)
    return candidates


def _shapes(kind: MmaKind) -> Iterable[tuple[int, int, int]]:
    """Yield (M, N, cta_group) for one kind's tier."""
    if kind.tier == "full":
        cta1_m = (128,) if kind.block_scaled else (64, 128)
        for m in cta1_m:
            for n in _legal_n(kind, m, 1):
                yield m, n, 1
        for n in _legal_n(kind, 256, 2):
            yield 256, n, 2
        if not kind.block_scaled:
            for n in (128, 256):
                yield 128, n, 2
    else:
        yield 128, 64, 1
        yield 128, 256, 1
        yield 256, 256, 2


def mma_variants(kinds: Sequence[MmaKind] = MMA_KINDS) -> list[MmaVariant]:
    """Return SS throughput, TS throughput, and single-issue latency MMA variants."""
    variants = []
    for kind in kinds:
        variants.extend(MmaVariant(kind, m, n, cg) for m, n, cg in _shapes(kind))
        if kind.ts:
            for m, n, cg in (
                (128, 64, 1),
                (128, 128, 1),
                (128, 256, 1),
                (256, 128, 2),
                (256, 256, 2),
            ):
                variants.append(MmaVariant(kind, m, n, cg, a_src="tmem"))
        if kind.tier == "full":
            for m, n, cg in ((128, 64, 1), (128, 256, 1), (256, 256, 2)):
                variants.append(MmaVariant(kind, m, n, cg, mode="latency"))
        if kind.key in SPARSE_KINDS:
            for m, n, cg in ((128, 64, 1), (128, 128, 1), (128, 256, 1), (256, 256, 2)):
                variants.append(MmaVariant(kind, m, n, cg, sparse=True))
        if kind.op.startswith("SM107") and kind.tier == "full":
            # Accumulator pairs need 2N TMEM columns plus scale factors within 512.
            widest = 128 if kind.block_scaled else 256
            for collector in ("b", "a"):
                for m, n, cg in (
                    (128, 64, 1),
                    (128, 128, 1),
                    (128, 256, 1),
                    (256, 128, 2),
                    (256, 256, 2),
                ):
                    if n <= widest and (collector == "b" or n <= 128):
                        variants.append(MmaVariant(kind, m, n, cg, collector=collector))
    return variants


def tmem_variants() -> list[TmemVariant]:
    """Return tcgen05.ld/st/cp/shift/commit/fence/alloc variants."""
    variants = []
    for op in ("ld", "st"):
        for shape, max_num in LDST_MAX_NUM.items():
            for num in (n for n in LDST_NUMS if n <= max_num):
                regs = num * LDST_REGISTER_FACTOR[shape]
                forms = {(1, 4), (1, max_warps(regs))}
                inflight = max(k for k in (1, 2, 4) if k * regs <= LDST_MAX_INFLIGHT_REGISTERS)
                forms.add((inflight, max_warps(inflight * regs)))
                for k, warps in sorted(forms):
                    variants.append(TmemVariant(op, shape, num, warps=warps, inflight=k))
            variants.append(TmemVariant(op, shape, 1, warps=1, mode="latency"))
            variants.append(
                TmemVariant(op, shape, 32 // LDST_REGISTER_FACTOR[shape], 1, "latency")
            )
    for shape in CP_SHAPES:
        variants.append(TmemVariant("cp", shape, warps=4))
        variants.append(TmemVariant("cp", shape, warps=4, mode="latency"))
    variants.append(TmemVariant("shift", "down", warps=4))
    variants.append(TmemVariant("shift", "down", warps=4, mode="latency"))
    variants.append(TmemVariant("commit", warps=4, mode="latency"))
    variants.append(TmemVariant("fence", "before_thread_sync", warps=4, mode="latency"))
    variants.append(TmemVariant("fence", "after_thread_sync", warps=4, mode="latency"))
    for columns in ALLOC_COLUMNS:
        variants.append(TmemVariant("alloc", num=columns, warps=4))
        variants.append(TmemVariant("alloc", num=columns, warps=4, mode="latency"))
    return variants


def all_variants() -> list[Variant]:
    """Return the complete variant table in a stable order."""
    return [*mma_variants(), *tmem_variants()]


def variant_arches(variant: Variant) -> tuple[str, ...]:
    """Return the architectures on which a variant is expected to be expressible."""
    if isinstance(variant, MmaVariant):
        return variant.kind.arches
    return ARCHES


def select_variants(
    variants: Iterable[Variant],
    *,
    arch: str | None = None,
    families: Sequence[str] = FAMILIES,
    modes: Sequence[str] = MODES,
    pattern: str | None = None,
    exclude: str | None = None,
) -> list[Variant]:
    """Filter variants by architecture, family, mode, and name regexes."""
    include = re.compile(pattern) if pattern else None
    skip = re.compile(exclude) if exclude else None
    selected = []
    for variant in variants:
        if arch is not None and arch not in variant_arches(variant):
            continue
        if variant.family not in families or variant.mode not in modes:
            continue
        if include is not None and include.search(variant.name) is None:
            continue
        if skip is not None and skip.search(variant.name) is not None:
            continue
        selected.append(variant)
    return selected


def arch_from_capability(major: int, minor: int) -> str:
    """Map a CUDA compute capability to this table's architecture key."""
    arch = f"sm_{major}{minor}"
    if arch not in ARCHES:
        raise ValueError(f"tcgen05 variants cover {ARCHES}; got {arch}")
    return arch


def peak_ops_per_clk_sm(variant: Variant, arch: str) -> int | None:
    """Return the public dense peak for an MMA variant on ``arch`` when one is documented."""
    if not isinstance(variant, MmaVariant):
        return None
    peak = PEAK_OPS_PER_CLK_SM.get(arch, {}).get(variant.kind.kind)
    if peak is not None and variant.sparse:
        return 2 * peak  # dense-equivalent rate of 2:4 structured sparsity
    return peak
