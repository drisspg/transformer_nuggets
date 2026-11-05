import math

import pytest
import torch

pytest.importorskip("cutlass.cute")

from transformer_nuggets.cute.warpdrive import (
    AtomicPersistent,
    ClusterLaunchControl,
    CtaPair,
    OneShot,
    RoleConfig,
    StaticPersistent,
)
from transformer_nuggets.cute.warpdrive.examples import grouped_gemm as gg
from transformer_nuggets.cute.warpdrive.examples.grouped_gemm import (
    GroupedGemm,
    compile_grouped_gemm,
    grouped_gemm,
    grouped_gemm_reference,
)


@pytest.fixture
def requires_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("warpdrive examples require SM100-family tcgen05")


def _offsets(sizes: list[int]) -> torch.Tensor:
    return torch.tensor(sizes, device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)


def _inputs(sizes: list[int], n: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn(sum(sizes), k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(len(sizes), n, k, device="cuda", dtype=torch.bfloat16)
    return a, b, _offsets(sizes)


def _bf16_baseline_error(a, b, sizes, ref) -> float:
    """The error of per-group bf16 ``torch.mm`` against the fp32 reference."""
    worst, start = 0.0, 0
    for g, size in enumerate(sizes):
        if size:
            out = (a[start : start + size] @ b[g].T).float()
            worst = max(worst, (out - ref[start : start + size]).abs().max().item())
        start += size
    return worst


def _check(sizes, n, k, rows_after: int = 0, **config):
    """Compare against per-group fp32 matmuls; rows past the last offset must stay untouched."""
    a, b, offsets = _inputs(sizes, n, k)
    if rows_after:
        a = torch.cat([a, torch.randn(rows_after, k, device="cuda", dtype=torch.bfloat16)])
    out = torch.full((a.shape[0], n), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert grouped_gemm(a, b, offsets, out=out, **config) is out
    torch.cuda.synchronize()
    ref = grouped_gemm_reference(a, b, offsets)
    rows = sum(sizes)
    err = (out[:rows].float() - ref[:rows]).abs().max().item() if rows else 0.0
    torch_err = _bf16_baseline_error(a, b, sizes, ref)
    assert err <= 2 * torch_err + 1e-3, (sizes, n, k, config, err, torch_err)
    assert torch.isnan(out[rows:]).all(), "rows past the last offset were written"


SIZE_CASES = {
    "single_group": ([128], 256, 64),
    "single_ragged": ([333], 520, 136),
    "uniform_8": ([256] * 8, 512, 256),
    "ragged_8": ([1000, 24, 0, 129, 128, 1, 256, 300], 520, 136),
    "empty_leading_trailing": ([0, 0, 200, 0, 50, 0, 0, 7], 264, 72),
    "all_but_one_empty": ([0] * 7 + [3], 64, 64),
    "small_groups_64": ([37] * 64, 1032, 200),
    "mixed_64": ([0, 1, 127, 128, 129, 255, 256, 257] * 8, 2056, 264),
}


@pytest.mark.parametrize("sizes, n, k", SIZE_CASES.values(), ids=SIZE_CASES.keys())
def test_grouped_gemm(requires_blackwell, sizes, n, k):
    _check(sizes, n, k)


def test_grouped_gemm_rows_past_last_offset_are_untouched(requires_blackwell):
    """``A`` may hold more rows than the offsets cover (padded routing buffers)."""
    _check([300, 0, 128, 77], 512, 256, rows_after=131)


@pytest.mark.parametrize(
    "schedule", [OneShot(), StaticPersistent(), ClusterLaunchControl()], ids=lambda s: s.get_key()
)
def test_grouped_gemm_schedules(requires_blackwell, schedule):
    _check([1000, 24, 0, 129, 128, 1, 256, 300], 520, 136, schedule=schedule)


@pytest.mark.parametrize(
    "schedule", [AtomicPersistent(), StaticPersistent()], ids=lambda s: s.get_key()
)
def test_grouped_gemm_cta_pair(requires_blackwell, schedule):
    """256-row MMA tiles: the peer CTA's half of a ragged tile can hold no valid rows at all."""
    _check([1000, 24, 0, 129, 128, 1, 256, 300, 5], 520, 136, schedule=schedule, cluster=CtaPair())


CONFIGS = [
    {"tile_n": 128},
    {"tile_n": 224, "epi_tile_n": 16, "c_stages": 3},
    {"ab_stages": 2, "acc_stages": 1},
    {"max_ctas": 7},
    {"roles": {"epilogue": RoleConfig(range(4), regs=152)}},
]


@pytest.mark.parametrize(
    "config",
    CONFIGS,
    ids=["_".join(f"{k}{v}" for k, v in c.items() if k != "roles") or "roles" for c in CONFIGS],
)
def test_grouped_gemm_configs(requires_blackwell, config):
    _check([1000, 24, 0, 129, 128, 1, 256, 300], 520, 136, **config)


def test_grouped_gemm_reuses_one_compile_across_shapes_and_offsets(requires_blackwell):
    """Shapes, G and the routing are runtime values of one compiled kernel."""
    _check([128], 256, 64)  # the default configuration is compiled (or loaded) once
    before = compile_grouped_gemm.cache_info()
    _check([300, 0, 128, 77], 512, 256)
    _check([37] * 64, 1032, 200)
    _check([5, 0, 1000], 2056, 1032)
    _check([0, 0, 0, 5], 8, 8)
    after = compile_grouped_gemm.cache_info()
    assert after.currsize == before.currsize, "a shape or offset change compiled another kernel"
    assert after.misses == before.misses
    assert after.hits == before.hits + 4  # one launch each
    # The wrapper's launch cache also keys on the configuration, not the shapes.
    ops = len(gg._OPS)
    _check([64, 64], 64, 64)
    assert len(gg._OPS) == ops


def _reference_tiles(sizes: list[int], tile_m: int) -> set[tuple[int, int, int]]:
    tiles, start = set(), 0
    for g, size in enumerate(sizes):
        for row0 in range(start, start + size, tile_m):
            tiles.add((g, row0, min(start + size - row0, tile_m)))
        start += size
    return tiles


def _python_locate_tile(mt: int, offsets: list[int], tile_m: int, total_m: int | None = None):
    """Mirror of ``_locate_tile``: the padded row space gives group g the m-tiles from
    ``ceil((offsets[g-1] + g * tile_m) / tile_m)`` on."""
    total_m = offsets[-1] if total_m is None else total_m
    count = len(offsets)
    lo = 0
    step = 1
    while step < count:
        step *= 2
    while step > 1:
        step //= 2
        cand = lo + step
        if cand < count and offsets[cand - 1] <= (mt - cand) * tile_m:
            lo = cand
    start = min(max(offsets[lo - 1] if lo else 0, 0), total_m)
    first_tile = -(-(start + lo * tile_m) // tile_m)
    row0 = start + (mt - first_tile) * tile_m
    end = min(offsets[lo], total_m)
    return row0 < end, lo, row0, min(end - row0, tile_m)


@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize(
    "sizes",
    [
        [128],
        [1000, 24, 0, 129, 128, 1, 256, 300],
        [0, 0, 200, 0, 50, 0, 0, 7],
        [0] * 7 + [3],
        [37] * 64,
        [0, 1, 127, 128, 129, 255, 256, 257] * 16,
    ],
    ids=["one", "ragged", "empty_runs", "last_only", "small_64", "mixed_128"],
)
def test_grouped_tiles_decode_is_a_bijection(sizes, tile_m):
    """Every (group, row0, rows) tile of the work appears exactly once among the
    ``ceil(total_M / tile_m) + G`` m-tile slots the host launches; the others are dropped."""
    offsets = list(torch.tensor(sizes).cumsum(0).tolist())
    slots = math.ceil(sum(sizes) / tile_m) + len(sizes)
    kept = [_python_locate_tile(mt, offsets, tile_m) for mt in range(slots)]
    kept = [(g, row0, rows) for found, g, row0, rows in kept if found]
    assert len(kept) == len(set(kept))
    assert set(kept) == _reference_tiles(sizes, tile_m)


def test_grouped_gemm_zero_work(requires_blackwell):
    """Empty K writes zeros over the covered rows only; an empty output or G=0 launches
    nothing and leaves ``out`` alone."""
    a, b, offsets = _inputs([64, 0, 24], 24, 0)
    a = torch.cat([a, a[:10]])
    out = torch.full((98, 24), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert grouped_gemm(a, b, offsets, out) is out
    torch.cuda.synchronize()
    assert torch.equal(out[:88], torch.zeros_like(out[:88]))
    assert torch.isnan(out[88:]).all()
    a, b, offsets = _inputs([0, 0], 8, 64)
    assert grouped_gemm(a, b, offsets).shape == (0, 8)
    a, b, offsets = _inputs([], 8, 64)
    a = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.full((16, 8), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert grouped_gemm(a, b, offsets, out) is out
    assert torch.isnan(out).all()


def test_grouped_gemm_clamps_offsets_to_a(requires_blackwell):
    """Offsets past the end of ``A`` (a routing bug) must not read or write outside it: the
    rows that exist are computed with the group the offsets name, the rest do not exist."""
    sizes = [64, 300]
    a, b, offsets = _inputs(sizes, 72, 64)
    a = a[:200].clone()  # offsets say [0, 64) and [64, 364) but A has 200 rows
    out = torch.full((200, 72), float("nan"), device="cuda", dtype=torch.bfloat16)
    grouped_gemm(a, b, offsets, out=out)
    torch.cuda.synchronize()
    ref = torch.cat([a[:64].float() @ b[0].float().T, a[64:].float() @ b[1].float().T])
    assert (out.float() - ref).abs().max().item() <= 0.1


@pytest.mark.parametrize(
    "mutate, error, match",
    [
        (lambda a, b, o: (a.float(), b, o), TypeError, "bfloat16"),
        (lambda a, b, o: (a, b, o.long()), TypeError, "int32"),
        (lambda a, b, o: (a, b[:, :, :32], o), ValueError, "incompatible shapes"),
        (lambda a, b, o: (a, b, o[:3]), ValueError, "offsets must have shape"),
        (lambda a, b, o: (a, b[:, :, ::2].contiguous()[:, :, :28], o), ValueError, "incompatible"),
        (lambda a, b, o: (a[:, ::2], b[:, :, ::2], o), ValueError, "contiguous"),
        (
            lambda a, b, o: (a[:, :60].contiguous(), b[:, :, :60].contiguous(), o),
            ValueError,
            "multiples of 8",
        ),
    ],
    ids=[
        "a_dtype",
        "offsets_dtype",
        "k_mismatch",
        "offsets_shape",
        "k_mismatch2",
        "strided",
        "k_mod_8",
    ],
)
def test_grouped_gemm_rejects_bad_operands(requires_blackwell, mutate, error, match):
    a, b, offsets = _inputs([64, 64, 64, 64], 64, 64)
    with pytest.raises(error, match=match):
        grouped_gemm(*mutate(a, b, offsets))


def test_grouped_gemm_rejects_too_many_rows(requires_blackwell):
    """The decode's Int32 arithmetic covers ``total_M + (G + 1) * 256`` rows."""
    _, b, offsets = _inputs([64, 64], 64, 64)
    a = torch.empty(1, 64, device="cuda", dtype=torch.bfloat16).expand(2**31 - 600, 64)
    with pytest.raises(ValueError, match="fit in int32"):
        grouped_gemm(a, b, offsets)


def test_grouped_gemm_rejects_bad_out(requires_blackwell):
    a, b, offsets = _inputs([64, 64], 64, 64)
    with pytest.raises(ValueError, match="must not overlap"):
        grouped_gemm(a, b, offsets, out=a)
    with pytest.raises(ValueError, match="must not overlap"):
        grouped_gemm(a, b, offsets, out=b.view(128, 64))
    with pytest.raises(TypeError, match="out must be bfloat16"):
        grouped_gemm(a, b, offsets, out=torch.empty(128, 64, device="cuda"))
    with pytest.raises(ValueError, match="out must have shape"):
        grouped_gemm(a, b, offsets, out=torch.empty(64, 64, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        grouped_gemm(
            a, b, offsets, out=torch.empty(64, 128, device="cuda", dtype=torch.bfloat16).T
        )


def test_grouped_gemm_rejects_unfit_configurations():
    with pytest.raises(ValueError, match="epi_tile_n must be one of"):
        GroupedGemm(epi_tile_n=128)
    with pytest.raises(ValueError, match="multiple of epi_tile_n=64"):
        GroupedGemm(tile_n=224, epi_tile_n=64)
    with pytest.raises(ValueError, match="c_stages must be in"):
        GroupedGemm(c_stages=0)
    assert GroupedGemm(tile_n=128).get_name() != GroupedGemm().get_name()
    assert GroupedGemm(cluster=CtaPair()).work.mapping.tile_m == 256
    assert "GroupedTiles(tile_m=128)" in GroupedGemm().describe()
