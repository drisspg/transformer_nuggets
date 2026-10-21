"""Named shape sets for the FlexGEMM benchmark.

``N`` is always the *logical* output width of a family; families with a grouped-main
contraction (SwiGLU) run a GEMM whose physical width is ``lanes * N``.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    experts: int = 0
    offs_kind: str = ""

    @property
    def grouped(self) -> bool:
        return self.experts > 0

    @property
    def label(self) -> str:
        if self.grouped:
            return f"E{self.experts}_K{self.k}_N{self.n}_M{self.m}_{self.offs_kind}"
        return f"M{self.m}_N{self.n}_K{self.k}"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(text: str) -> Shape:
        return Shape(**json.loads(text))


def seqlens(experts: int, total_m: int, kind: str) -> list[int]:
    """Per-expert row counts. ``skewed``: ~20% of the experts hold 80% of the rows
    and two experts are empty (when E is large enough), shuffled deterministically."""
    if kind == "balanced":
        return [total_m // experts] * experts
    if kind == "hot":
        hot_rows = int(total_m * 0.6)
        rest = experts - 2
        light = [(total_m - hot_rows - 1) // rest] * rest if rest else []
        if light:
            light[0] += total_m - hot_rows - 1 - sum(light)
        return [hot_rows, 1, *light]
    if kind != "skewed":
        raise ValueError(f"unknown offs kind {kind!r}; expected balanced, skewed or hot")
    heavy = max(1, experts // 5)
    empty = 2 if experts > heavy + 2 else 0
    light = experts - heavy - empty
    heavy_rows = int(total_m * 0.8)
    light_rows = total_m - heavy_rows
    rows = [heavy_rows // heavy] * heavy
    rows[0] += heavy_rows - sum(rows)
    light_list = [light_rows // light] * light if light else []
    if light_list:
        light_list[0] += light_rows - sum(light_list)
    rows = rows + [0] * empty + light_list
    random.Random(1234).shuffle(rows)
    return rows


def _dense(ms, ns, ks) -> list[Shape]:
    return [Shape(m, n, k) for m, n, k in itertools.product(ms, ns, ks)]


def _moe(experts, kns, tokens, kinds) -> list[Shape]:
    return [
        Shape(m, n, k, experts=e, offs_kind=kind)
        for e, (k, n), m, kind in itertools.product(experts, kns, tokens, kinds)
    ]


# DeepSeek-V3 linear layers as (N, K) with M = tokens: dense FFN w1 / w2, MLA wkv_a / wkv_b / wo,
# lm_head, and wq (16B) or wq_a / wq_b (671B). qk head = 128 nope + 64 rope, v head = 128.
_DSV3_16B = {
    "dim": 2048,
    "dense_hidden": 10944,
    "heads": 16,
    "kv_lora": 512,
    "vocab": 102400,
    "q_lora": 0,
}
_DSV3_671B = {
    "dim": 7168,
    "dense_hidden": 18432,
    "heads": 128,
    "kv_lora": 512,
    "vocab": 129280,
    "q_lora": 1536,
}


def _dsv3(cfg: dict[str, int], tokens) -> list[Shape]:
    dim, heads = cfg["dim"], cfg["heads"]
    layers = [
        (cfg["dense_hidden"], dim),
        (dim, cfg["dense_hidden"]),
        (cfg["kv_lora"] + 64, dim),
        (heads * 256, cfg["kv_lora"]),
        (dim, heads * 128),
        (cfg["vocab"], dim),
    ]
    if cfg["q_lora"]:
        layers += [(cfg["q_lora"], dim), (heads * 192, cfg["q_lora"])]
    else:
        layers.append((heads * 192, dim))
    return [Shape(m, n, k) for m in tokens for n, k in layers]


SHAPE_SETS: dict[str, list[Shape]] = {
    # Frozen dense grid used for cross-run comparisons.
    "grid": _dense((256, 1024, 4096, 16384), (1024, 4096, 8192), (1024, 4096, 8192)),
    # Skinny-N and skinny-M rows from the pointwise audit.
    "skinny": _dense((256, 1024, 4096, 16384), (256,), (4096,))
    + _dense((64,), (256, 1024, 4096, 8192), (4096,)),
    "dsv3_16b": _dsv3(_DSV3_16B, (2048, 8192)),
    "dsv3_671b": _dsv3(_DSV3_671B, (2048, 8192)),
    "moe_16b": _moe(
        (64, 8),
        ((2048, 1408), (1408, 2048)),
        (4096, 8192, 32768),
        ("balanced", "skewed"),
    ),
    "moe_671b": _moe(
        (256, 32, 8),
        ((7168, 2048), (2048, 7168)),
        (4096, 8192, 32768),
        ("balanced", "skewed"),
    ),
}

FAST_SHAPES = [Shape(1024, 4096, 1024), Shape(4096, 4096, 1024)]


def parse_dense(text: str) -> Shape:
    m, n, k = (int(v) for v in text.split(","))
    return Shape(m, n, k)


def parse_moe(text: str) -> Shape:
    parts = text.split(",")
    if len(parts) not in (4, 5):
        raise ValueError(f"--moe expects E,K,N,M[,balanced|skewed|hot], got {text!r}")
    e, k, n, m = (int(v) for v in parts[:4])
    kind = parts[4] if len(parts) == 5 else "balanced"
    return Shape(m, n, k, experts=e, offs_kind=kind)
