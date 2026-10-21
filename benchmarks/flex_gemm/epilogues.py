"""Epilogue catalog: one builder per family returning the fused call, its stock spelling,
the bare-GEMM baseline, and a high-precision reference for correctness.

Weight layout: every dense family stores its linear weight as ``[N_phys, K]`` (nn.Linear
layout) and passes ``w.t()`` as B; grouped families use ``[E, N_phys, K]`` and ``w.mT``.
SwiGLU families interleave gate/up rows (``interleaved``) or stack them (``chunked``),
which is the physical contract the FlexGEMM grouped-main lowering keys on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from shapes import seqlens, Shape

import torch
import torch.nn.functional as F
from torch._higher_order_ops import flex_gemm


# Optional pieces of the FlexGEMM API; families that need them raise FamilyUnavailable.
try:
    from torch._higher_order_ops.flex_gemm import to_blocked
    from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
except ImportError as e:  # pragma: no cover - depends on the torch build
    to_blocked = inline_asm_elementwise = None
    OPTIONAL_IMPORT_ERROR = str(e)
else:
    OPTIONAL_IMPORT_ERROR = ""


if TYPE_CHECKING:
    from collections.abc import Callable


MX_GROUP = 32
LSE_GROUP = 256


class FamilyUnavailable(Exception):
    """The family cannot run on this device / shape; the driver records and skips it."""


def flatten(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    return [t for item in value for t in flatten(item)]


@dataclass
class Variant:
    fn: Callable[..., Any]
    args: tuple[Any, ...]


@dataclass
class Case:
    family: str
    shape: Shape
    lanes: int
    flex: Callable[[dict[str, Any]], Variant]
    stock: Variant
    cublas: Variant
    reference: list[torch.Tensor]
    flops: int
    saved_bytes: int
    two_gemm: Variant | None = None
    fast_math: bool = False
    gemm_key: str = "bf16"
    comparable: Callable[[Any], list[torch.Tensor]] = flatten


class Problem:
    """Deterministic operands shared by every family of one shape."""

    def __init__(self, shape: Shape, dtype: torch.dtype, device: str = "cuda") -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(
            1234 + shape.m + 3 * shape.n + 7 * shape.k
        )
        self.a = self._uniform(shape.m, shape.k)
        self.offs: torch.Tensor | None = None
        if shape.grouped:
            rows = seqlens(shape.experts, shape.m, shape.offs_kind)
            self.offs = (
                torch.tensor(rows, device=device, dtype=torch.int32).cumsum(0).to(torch.int32)
            )

    def _uniform(self, *size: int, scale: float = 1.0) -> torch.Tensor:
        x = torch.rand(*size, device=self.device, dtype=torch.float32, generator=self.gen)
        return x.mul_(2 * scale).sub_(scale).to(self.dtype)

    def weight(self, n_rows: int) -> torch.Tensor:
        """``[N, K]`` (dense) or ``[E, N, K]`` (grouped) weight, scaled so the accumulator is O(1)."""
        lead = (self.shape.experts,) if self.shape.grouped else ()
        return self._uniform(*lead, n_rows, self.shape.k, scale=math.sqrt(3 / self.shape.k))

    def row(self, n: int, scale: float = 0.1) -> torch.Tensor:
        return self._uniform(1, n, scale=scale)

    def tile(self, n: int) -> torch.Tensor:
        return self._uniform(self.shape.m, n)

    def gemm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.offs is None:
            return torch.mm(a, b)
        return F.grouped_mm(a, b, offs=self.offs)

    def hp_acc(self, b: torch.Tensor) -> torch.Tensor:
        """fp64 accumulator ``[M, N_phys]`` for the reference epilogue."""
        if self.offs is None:
            return self.a.double() @ b.double()
        bounds = [0, *self.offs.tolist()]
        out = torch.zeros(self.shape.m, b.shape[-1], device=self.device, dtype=torch.float64)
        for g in range(self.shape.experts):
            lo, hi = bounds[g], bounds[g + 1]
            if hi > lo:
                out[lo:hi] = self.a[lo:hi].double() @ b[g].double()
        return out

    def flex(
        self, b: torch.Tensor, epilogue: Callable[[Any], Any], options: dict[str, Any]
    ) -> Variant:
        if self.offs is None:

            def dense(a, b):
                return flex_gemm(torch.mm, (a, b), epilogue, kernel_options=options)

            return Variant(dense, (self.a, b))

        def grouped(a, b, offs):
            return flex_gemm(
                F.grouped_mm,
                (a, b),
                epilogue,
                gemm_kwargs={"offs": offs},
                kernel_options=options,
            )

        return Variant(grouped, (self.a, b, self.offs))

    def cublas(self, b: torch.Tensor) -> Variant:
        if self.offs is None:
            return Variant(torch.mm, (self.a, b))
        return Variant(lambda a, b, offs: F.grouped_mm(a, b, offs=offs), (self.a, b, self.offs))

    def flops(self, n_phys: int) -> int:
        return 2 * self.shape.m * n_phys * self.shape.k

    def saved_bytes(self, n_phys: int) -> int:
        """Stock writes the ``[M, N_phys]`` GEMM result and reads it back in the epilogue kernel."""
        return 2 * self.shape.m * n_phys * torch.tensor([], dtype=self.dtype).element_size()


# --------------------------------------------------------------------------- pointwise


def _pointwise(name, p: Problem, epilogue_fn, stock_math, *, fast_math=False, saved=True) -> Case:
    n = p.shape.n
    w = p.weight(n)
    b = w.mT
    acc = p.hp_acc(b)
    return Case(
        family=name,
        shape=p.shape,
        lanes=1,
        flex=lambda options: p.flex(b, epilogue_fn, options),
        stock=Variant(stock_math, (p.a, b)),
        cublas=p.cublas(b),
        reference=[stock_math(p.a, b, acc=acc)],
        flops=p.flops(n),
        saved_bytes=p.saved_bytes(n) if saved else 0,
        fast_math=fast_math,
    )


def identity(p: Problem) -> Case:
    def stock(a, b, acc=None):
        return (p.gemm(a, b) if acc is None else acc).to(p.dtype)

    return _pointwise("identity", p, lambda acc: acc, stock, saved=False)


def relu(p: Problem) -> Case:
    def stock(a, b, acc=None):
        return torch.relu(p.gemm(a, b) if acc is None else acc).to(p.dtype)

    return _pointwise("relu", p, lambda acc: torch.relu(acc.float()).to(acc.dtype), stock)


def _silu(p: Problem, name: str, fast_math: bool) -> Case:
    def stock(a, b, acc=None):
        return F.silu(p.gemm(a, b) if acc is None else acc).to(p.dtype)

    return _pointwise(
        name,
        p,
        lambda acc: F.silu(acc.float()).to(acc.dtype),
        stock,
        fast_math=fast_math,
    )


def silu(p: Problem) -> Case:
    return _silu(p, "silu", False)


def silu_fast(p: Problem) -> Case:
    return _silu(p, "silu_fast", True)


def bias_gelu(p: Problem) -> Case:
    bias = p.row(p.shape.n)

    def stock(a, b, acc=None):
        return F.gelu((p.gemm(a, b) if acc is None else acc) + bias).to(p.dtype)

    return _pointwise(
        "bias_gelu",
        p,
        lambda acc: F.gelu(acc.float() + bias.float()).to(acc.dtype),
        stock,
    )


def residual(p: Problem) -> Case:
    x = p.tile(p.shape.n)

    def stock(a, b, acc=None):
        return (x + (p.gemm(a, b) if acc is None else acc)).to(p.dtype)

    return _pointwise(
        "residual",
        p,
        lambda acc: (acc.float() + x.float()).to(acc.dtype),
        stock,
        saved=False,
    )


# --------------------------------------------------------------------------- reductions


def _reduction(name, p: Problem, epilogue_fn, stock_math, *, comparable=None) -> Case:
    n = p.shape.n
    w = p.weight(n)
    b = w.mT
    acc = p.hp_acc(b)
    case = Case(
        family=name,
        shape=p.shape,
        lanes=1,
        flex=lambda options: p.flex(b, epilogue_fn, options),
        stock=Variant(stock_math, (p.a, b)),
        cublas=p.cublas(b),
        reference=flatten(stock_math(p.a, b, acc=acc)),
        flops=p.flops(n),
        saved_bytes=p.saved_bytes(n),
    )
    if comparable is not None:
        case.comparable = comparable
    return case


def _require_divisible(p: Problem, group: int) -> None:
    if p.shape.n % group:
        raise FamilyUnavailable(f"N={p.shape.n} is not a multiple of the group {group}")


def _amax(p: Problem, group: int) -> Case:
    _require_divisible(p, group)
    m = p.shape.m

    def epilogue(acc):
        return acc.relu(), acc.float().view(m, -1, group).abs().amax(-1) / 448.0

    def stock(a, b, acc=None):
        acc = p.gemm(a, b) if acc is None else acc
        return acc.relu().to(p.dtype), acc.float().view(m, -1, group).abs().amax(-1) / 448.0

    return _reduction(f"amax_g{group}", p, epilogue, stock)


def amax_g32(p: Problem) -> Case:
    return _amax(p, 32)


def amax_g128(p: Problem) -> Case:
    return _amax(p, 128)


def sum_g32(p: Problem) -> Case:
    _require_divisible(p, 32)
    m = p.shape.m

    def epilogue(acc):
        return acc.relu(), acc.float().view(m, -1, 32).square().sum(-1)

    def stock(a, b, acc=None):
        acc = p.gemm(a, b) if acc is None else acc
        return acc.relu().to(p.dtype), acc.float().view(m, -1, 32).square().sum(-1).sum(-1)

    case = _reduction("sum_g32", p, epilogue, stock)
    b = case.stock.args[1]

    def flex(options):
        def fn(a, b):
            main, partial = flex_gemm(torch.mm, (a, b), epilogue, kernel_options=options)
            return main, partial.sum(-1)

        return Variant(fn, (p.a, b))

    case.flex = flex
    return case


def _lse(p: Problem, gather: bool) -> Case:
    _require_divisible(p, LSE_GROUP)
    if p.shape.grouped:
        raise FamilyUnavailable("logsumexp epilogues are dense torch.mm only")
    m = p.shape.m
    targets = torch.arange(m, device=p.device, dtype=torch.int64).remainder(p.shape.n)

    def epilogue(acc):
        logits = acc.float()
        partial = logits.view(m, -1, LSE_GROUP).logsumexp(-1)
        if gather:
            return acc, partial, acc.gather(1, targets[:, None]).squeeze(1)
        return acc, partial

    def stock(a, b, acc=None):
        logits = p.gemm(a, b) if acc is None else acc
        lse = logits.float().view(m, -1, LSE_GROUP).logsumexp(-1).logsumexp(-1)
        if gather:
            return (
                logits.to(p.dtype),
                lse,
                logits.gather(1, targets[:, None]).squeeze(1).to(p.dtype),
            )
        return logits.to(p.dtype), lse

    name = "lse_g256_gather" if gather else "lse_g256"
    case = _reduction(name, p, epilogue, stock)
    b = case.stock.args[1]

    def flex(options):
        def fn(a, b):
            outputs = flex_gemm(torch.mm, (a, b), epilogue, kernel_options=options)
            if gather:
                logits, partial, selected = outputs
                return logits, partial.logsumexp(-1), selected
            logits, partial = outputs
            return logits, partial.logsumexp(-1)

        return Variant(fn, (p.a, b))

    case.flex = flex
    return case


def lse_g256(p: Problem) -> Case:
    return _lse(p, False)


def lse_g256_gather(p: Problem) -> Case:
    return _lse(p, True)


def mx_scale(amax: torch.Tensor) -> torch.Tensor:
    return inline_asm_elementwise(
        amax.float() / 448.0,
        asm_str="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
        constraints="=h,r",
        dtype=torch.float8_e8m0fnu,
    )


def mx_quantize(acc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = acc.float().view(acc.shape[0], -1, MX_GROUP)
    scale = mx_scale(grouped.abs().amax(-1, keepdim=True))
    quantized = (grouped * scale.float().reciprocal()).view_as(acc)
    return quantized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn), to_blocked(scale.squeeze(-1))


def unpack_blocked(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Invert the 128x4 blocked scale layout."""
    row_blocks = (rows + 127) // 128
    col_blocks = (cols + 3) // 4
    tiled = blocked.view(row_blocks * col_blocks, 32, 4, 4)
    tiled = tiled.transpose(1, 2).reshape(row_blocks, col_blocks, 128, 4).permute(0, 2, 1, 3)
    return tiled.reshape(row_blocks * 128, col_blocks * 4)[:rows, :cols]


def mxfp8_producer(p: Problem) -> Case:
    """GEMM -> MXFP8 data + blocked E8M0 scales; compared after dequantization."""
    if to_blocked is None:
        raise FamilyUnavailable(OPTIONAL_IMPORT_ERROR)
    _require_divisible(p, MX_GROUP)
    if p.shape.grouped:
        raise FamilyUnavailable("blocked-scale producers are dense torch.mm only")
    m, n = p.shape.m, p.shape.n

    def stock(a, b, acc=None):
        return mx_quantize(p.gemm(a, b)) if acc is None else acc

    def dequantize(out):
        quantized, blocked = out
        scale = unpack_blocked(blocked, m, n // MX_GROUP).float()
        return [quantized.float() * scale.repeat_interleave(MX_GROUP, dim=-1)]

    return _reduction("mxfp8_producer", p, mx_quantize, stock, comparable=dequantize)


# --------------------------------------------------------------------------- SwiGLU


def _swiglu_gate(gate, up):
    return F.silu(gate) * up


def _situ_gate(gate, up):
    """Kimi K3 SiTU-GLU: soft-clamped SiLU gate (beta 4) times soft-clamped up (beta 25)."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(up / 25.0))


def _swiglu(p: Problem, name: str, interleaved: bool, glu=_swiglu_gate, fast_math=False) -> Case:
    m, n = p.shape.m, p.shape.n
    w1, w3 = p.weight(n), p.weight(n)
    if interleaved:
        w13 = torch.stack([w1, w3], dim=-2).reshape(*w1.shape[:-2], 2 * n, p.shape.k)
    else:
        w13 = torch.cat([w1, w3], dim=-2)
    b = w13.mT
    acc = p.hp_acc(b)

    def epilogue(acc):
        if interleaved:
            lanes = acc.float().view(m, n, 2)
            gate, up = lanes[..., 0], lanes[..., 1]
        else:
            gate, up = acc.float().chunk(2, dim=-1)
        return glu(gate, up).to(acc.dtype)

    def stock(a, b, acc=None):
        h = (p.gemm(a, b) if acc is None else acc).float()
        if interleaved:
            lanes = h.view(m, n, 2)
            gate, up = lanes[..., 0], lanes[..., 1]
        else:
            gate, up = h.chunk(2, dim=-1)
        return glu(gate, up).to(p.dtype)

    def two_gemm(a, b1, b3):
        return glu(p.gemm(a, b1).float(), p.gemm(a, b3).float()).to(p.dtype)

    return Case(
        family=name,
        shape=p.shape,
        lanes=2,
        flex=lambda options: p.flex(b, epilogue, options),
        stock=Variant(stock, (p.a, b)),
        cublas=p.cublas(b),
        two_gemm=Variant(two_gemm, (p.a, w1.mT, w3.mT)),
        reference=[stock(p.a, b, acc=acc)],
        flops=p.flops(2 * n),
        saved_bytes=p.saved_bytes(2 * n),
        fast_math=fast_math,
    )


def swiglu_interleaved(p: Problem) -> Case:
    return _swiglu(p, "swiglu_interleaved", True)


def swiglu_chunked(p: Problem) -> Case:
    return _swiglu(p, "swiglu_chunked", False)


# --------------------------------------------------------------------------- grouped (MoE)


def _require_grouped(p: Problem, name: str) -> None:
    if not p.shape.grouped:
        raise FamilyUnavailable(
            f"{name} needs a MoE shape (use --shapes moe_16b or --moe E,K,N,M)"
        )


def grouped_silu(p: Problem) -> Case:
    _require_grouped(p, "grouped_silu")
    return _silu(p, "grouped_silu", False)


def grouped_swiglu_interleaved(p: Problem) -> Case:
    _require_grouped(p, "grouped_swiglu_interleaved")
    return _swiglu(p, "grouped_swiglu_interleaved", True)


def grouped_swiglu_chunked(p: Problem) -> Case:
    _require_grouped(p, "grouped_swiglu_chunked")
    return _swiglu(p, "grouped_swiglu_chunked", False)


def situ_glu_interleaved(p: Problem) -> Case:
    return _swiglu(p, "situ_glu_interleaved", True, glu=_situ_gate)


def grouped_situ_glu_interleaved(p: Problem) -> Case:
    _require_grouped(p, "grouped_situ_glu_interleaved")
    return _swiglu(p, "grouped_situ_glu_interleaved", True, glu=_situ_gate)


def grouped_situ_glu_chunked(p: Problem) -> Case:
    _require_grouped(p, "grouped_situ_glu_chunked")
    return _swiglu(p, "grouped_situ_glu_chunked", False, glu=_situ_gate)


def grouped_situ_glu_interleaved_fast(p: Problem) -> Case:
    _require_grouped(p, "grouped_situ_glu_interleaved_fast")
    return _swiglu(p, "grouped_situ_glu_interleaved_fast", True, glu=_situ_gate, fast_math=True)


# --------------------------------------------------------------------------- scaled_mm


def scaled_mm_mxfp8(p: Problem) -> Case:
    """MXFP8 ``F.scaled_mm`` with a ReLU epilogue; reference is eager scaled_mm in fp32."""
    if p.shape.grouped:
        raise FamilyUnavailable("scaled_mm is dense only")
    if not hasattr(F, "scaled_mm") or torch.cuda.get_device_capability() < (10, 0):
        raise FamilyUnavailable("F.scaled_mm with MXFP8 block scales needs SM100+")
    try:
        from torch._vendor.quack.blockscaled.operand import BlockScaledOperand
    except ImportError as e:
        raise FamilyUnavailable(str(e)) from e

    n = p.shape.n
    if n % 128 or p.shape.k % 128:
        raise FamilyUnavailable("MXFP8 scaled_mm needs N and K multiples of 128")
    qa = BlockScaledOperand.quantize(p.a.to(torch.bfloat16), "mxfp8_e4m3")
    qb = BlockScaledOperand.quantize(p.weight(n).mT.to(torch.bfloat16), "mxfp8_e4m3", dim=-2)
    scale_a, scale_b = [qa.scale.flatten()], [qb.scale.flatten()]
    recipes, swizzles = [F.ScalingType.BlockWise1x32], [F.SwizzleType.SWIZZLE_32_4_4]
    gemm_kwargs = {
        "scale_recipe_a": recipes,
        "scale_recipe_b": recipes,
        "swizzle_a": swizzles,
        "swizzle_b": swizzles,
        "output_dtype": torch.bfloat16,
    }

    def scaled(a, b, sa, sb, **overrides):
        return F.scaled_mm(a, b, sa, scale_b=sb, **{**gemm_kwargs, **overrides})

    def stock(a, b, sa, sb):
        return scaled(a, b, sa, sb).relu()

    def flex(options):
        def fn(a, b, sa, sb):
            return flex_gemm(
                F.scaled_mm,
                (a, b, sa, sb),
                lambda acc: acc.relu(),
                gemm_kwargs=gemm_kwargs,
                kernel_options=options,
            )

        return Variant(fn, (qa.qdata, qb.qdata, scale_a, scale_b))

    args = (qa.qdata, qb.qdata, scale_a, scale_b)
    try:
        reference = scaled(*args, output_dtype=torch.float32).relu()
    except RuntimeError:
        reference = scaled(*args).float().relu()
    return Case(
        family="scaled_mm_mxfp8",
        shape=p.shape,
        lanes=1,
        flex=flex,
        stock=Variant(stock, args),
        cublas=Variant(scaled, args),
        reference=[reference],
        flops=p.flops(n),
        saved_bytes=p.saved_bytes(n),
        gemm_key="mxfp8",
    )


FAMILIES: dict[str, Callable[[Problem], Case]] = {
    fn.__name__: fn
    for fn in (
        identity,
        relu,
        silu,
        silu_fast,
        bias_gelu,
        residual,
        amax_g32,
        amax_g128,
        sum_g32,
        lse_g256,
        lse_g256_gather,
        mxfp8_producer,
        swiglu_interleaved,
        swiglu_chunked,
        grouped_silu,
        grouped_swiglu_interleaved,
        grouped_swiglu_chunked,
        situ_glu_interleaved,
        grouped_situ_glu_interleaved,
        grouped_situ_glu_chunked,
        grouped_situ_glu_interleaved_fast,
        scaled_mm_mxfp8,
    )
}
