import math
from collections import Counter

import pytest
import torch

pytest.importorskip("cutlass.cute")

from transformer_nuggets.cute.profiler import profile_session
from transformer_nuggets.cute.sgemm import (
    SGEMM_PROFILE_EVENTS,
    SGEMM_PROFILE_TAGS,
    SGEMM_PROFILE_TILES,
    SgemmConfig,
    naive_sgemm,
    sgemm,
)

CONFIGS = [
    SgemmConfig(128, 256, 8, 3, 256),
    SgemmConfig(128, 128, 32, 2, 256),
    SgemmConfig(64, 128, 16, 4, 128),
    SgemmConfig(64, 64, 8, 6, 64),
]


@pytest.fixture
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def requires_blackwell(cuda_device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    if torch.cuda.get_device_capability(cuda_device)[0] < 10:
        pytest.skip("sgemm requires Blackwell or newer")
    monkeypatch.delenv("SGEMM_PROFILE", raising=False)


def make_operand(rows: int, cols: int, layout: str) -> torch.Tensor:
    tensor = torch.randn(rows, cols, device="cuda")
    return tensor if layout == "row" else tensor.t().contiguous().t()


def assert_matches_fp64(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    reference = (a.double() @ b.double()).float()
    torch.testing.assert_close(out, reference, atol=1e-5 * math.sqrt(a.shape[1]), rtol=1e-5)


@pytest.mark.parametrize(
    "field",
    ["tile_m", "tile_n", "tile_k", "num_stages", "num_threads", "threads_n", "k_group"],
)
@pytest.mark.parametrize(
    "value, error, message",
    [
        pytest.param(0, ValueError, "positive", id="zero"),
        pytest.param(-1, ValueError, "positive", id="negative"),
        pytest.param(True, TypeError, "integer", id="bool"),
        pytest.param(16.0, TypeError, "integer", id="float"),
    ],
)
def test_sgemm_config_requires_positive_integers(
    field: str, value: float, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=rf"{field}.*{message}"):
        SgemmConfig(**{field: value})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        pytest.param({"num_threads": 33}, "num_threads", id="partial-warp"),
        pytest.param({"num_threads": 1024, "tile_m": 256}, "num_threads", id="block-limit"),
        pytest.param({"tile_k": 12}, "tile_k", id="unsupported-k-tile"),
        pytest.param({"num_stages": 1}, "num_stages", id="single-stage"),
        pytest.param({"threads_n": 3}, "threads_n", id="thread-partition"),
        pytest.param({"k_group": 3}, "k_group", id="unsupported-k-group"),
        pytest.param({"tile_m": 96}, "tile_m", id="m-granule"),
        pytest.param({"tile_n": 96}, "tile_n", id="n-granule"),
        pytest.param({"tile_m": 512}, "256", id="m-tma-limit"),
        pytest.param({"tile_n": 512}, "256", id="n-tma-limit"),
    ],
)
def test_sgemm_config_rejects_unsupported_tiling(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SgemmConfig(**kwargs)


@pytest.mark.usefixtures("requires_blackwell")
class TestSgemm:
    def test_wide_output_beyond_grid_y_limit(self) -> None:
        a = torch.arange(16, device="cuda", dtype=torch.float32).view(4, 4)
        b = torch.ones(4, 65536 * SgemmConfig().tile_n, device="cuda")
        out = sgemm(a, b)
        torch.testing.assert_close(out, a.sum(1, keepdim=True).expand_as(out), atol=0, rtol=0)

    @pytest.mark.parametrize("k_group", [1, 2])
    @pytest.mark.parametrize("layouts", [("row", "col"), ("col", "row")])
    def test_register_groups_after_ring_wrap(self, k_group: int, layouts: tuple[str, str]) -> None:
        torch.manual_seed(0)
        a = make_operand(132, 132, layouts[0])
        b = make_operand(132, 260, layouts[1])
        config = SgemmConfig(tile_k=32, num_stages=2, k_group=k_group)
        out = sgemm(a, b, config=config)
        assert_matches_fp64(out, a, b)
        a.neg_()
        b.mul_(0.5)
        assert sgemm(a, b, out=out, config=config) is out
        assert_matches_fp64(out, a, b)

    @pytest.mark.parametrize("a_layout", ["row", "col"])
    @pytest.mark.parametrize("b_layout", ["row", "col"])
    @pytest.mark.parametrize(
        "shape",
        [(4, 4, 4), (128, 128, 8), (132, 260, 36), (1000, 1028, 1004), (516, 132, 12)],
    )
    def test_matches_fp64(self, shape: tuple[int, int, int], a_layout: str, b_layout: str) -> None:
        torch.manual_seed(0)
        m, n, k = shape
        a = make_operand(m, k, a_layout)
        b = make_operand(k, n, b_layout)
        a_before, b_before = a.clone(), b.clone()

        out = sgemm(a, b)

        assert out.shape == (m, n)
        assert out.dtype == torch.float32
        assert out.device == a.device
        assert out.is_contiguous()
        assert_matches_fp64(out, a, b)
        torch.testing.assert_close(a, a_before, atol=0, rtol=0)
        torch.testing.assert_close(b, b_before, atol=0, rtol=0)

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.name)
    def test_configs(self, config: SgemmConfig) -> None:
        torch.manual_seed(0)
        a = torch.randn(520, 700, device="cuda")
        b = torch.randn(700, 1036, device="cuda")
        assert_matches_fp64(sgemm(a, b, config=config), a, b)

    @pytest.mark.parametrize("a_layout", ["row", "col"])
    @pytest.mark.parametrize("b_layout", ["row", "col"])
    @pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.name)
    def test_changed_inputs_and_output_reuse_after_ring_wrap(
        self, config: SgemmConfig, a_layout: str, b_layout: str
    ) -> None:
        torch.manual_seed(0)
        m, n = config.tile_m + 4, config.tile_n + 4
        k = config.tile_k * (2 * config.num_stages + 1) + 4
        a = make_operand(m, k, a_layout)
        b = make_operand(k, n, b_layout)
        storage = torch.full((m * n + 8,), -12345.0, device="cuda")
        out = storage[4:-4].view(m, n)

        for _ in range(3):
            a.random_(-2, 3)
            b.random_(-2, 3)
            out.fill_(float("nan"))

            assert sgemm(a, b, out=out, config=config) is out

            # Small integers keep every FP32 product and partial sum exact.
            reference = (a.double() @ b.double()).float()
            torch.testing.assert_close(out, reference, atol=0, rtol=0)
            assert torch.all(storage[:4] == -12345)
            assert torch.all(storage[-4:] == -12345)

    @pytest.mark.parametrize("k", [4, 132, 516], ids=["one-tile", "ring-wrap", "sampling-cap"])
    def test_profiling_preserves_output_and_records_sampled_tiles(
        self, k: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch.manual_seed(0)
        a = make_operand(132, k, "row")
        b = make_operand(k, 132, "col")
        unprofiled = sgemm(a, b)
        sampled_tiles = min(SGEMM_PROFILE_TILES, math.ceil(k / SgemmConfig().tile_k))

        monkeypatch.setenv("SGEMM_PROFILE", "1")
        with profile_session(
            max_events_per_unit=SGEMM_PROFILE_EVENTS,
            num_units=1,
            tag_names=list(SGEMM_PROFILE_TAGS),
            compact=True,
            expected_events_per_unit=4 * sampled_tiles,
        ) as session:
            profiled = sgemm(a, b, profile_buffer=session.prof.tensor)
            torch.testing.assert_close(profiled, unprofiled, atol=0, rtol=0)
            a.add_(1)
            b.neg_()
            assert sgemm(a, b, out=profiled, profile_buffer=session.prof.tensor) is profiled

        assert_matches_fp64(profiled, a, b)
        assert Counter(event.tag_name for event in session.events) == dict.fromkeys(
            SGEMM_PROFILE_TAGS, sampled_tiles
        )
        assert all(event.unit_id == 0 and event.dur_ns >= 0 for event in session.events)

        monkeypatch.setenv("SGEMM_PROFILE", "0")
        torch.testing.assert_close(sgemm(a, b), profiled, atol=0, rtol=0)

    @pytest.mark.parametrize("operand", ["a", "b", "out"])
    def test_rejects_profile_buffer_overlapping_operands(
        self, operand: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = torch.randn(32, 16, device="cuda")
        b = torch.randn(16, 32, device="cuda")
        out = torch.empty(32, 32, device="cuda")
        aliased = {"a": a, "b": b, "out": out}[operand]
        buffer = aliased.flatten().view(torch.int64)[: 1 + SGEMM_PROFILE_EVENTS]
        monkeypatch.setenv("SGEMM_PROFILE", "1")
        with pytest.raises(ValueError, match="overlap"):
            sgemm(a, b, out=out, profile_buffer=buffer)

    @pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "missing-buffer"])
    def test_rejects_profile_buffer_configuration(
        self, enabled: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = torch.randn(16, 16, device="cuda")
        b = torch.randn_like(a)
        buffer = (
            None
            if enabled
            else torch.zeros(1 + SGEMM_PROFILE_EVENTS, device="cuda", dtype=torch.int64)
        )
        monkeypatch.setenv("SGEMM_PROFILE", "1" if enabled else "0")
        with pytest.raises(ValueError, match="profile_buffer"):
            sgemm(a, b, profile_buffer=buffer)

    @pytest.mark.parametrize("case", ["dtype", "device", "shape", "strided"])
    def test_rejects_invalid_profile_buffer(
        self, case: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = torch.randn(16, 16, device="cuda")
        b = torch.randn_like(a)
        size = 1 + SGEMM_PROFILE_EVENTS
        match case:
            case "dtype":
                buffer = torch.zeros(size, device="cuda", dtype=torch.float32)
            case "device":
                buffer = torch.zeros(size, device="cpu", dtype=torch.int64)
            case "shape":
                buffer = torch.zeros(1, size, device="cuda", dtype=torch.int64)
            case "strided":
                buffer = torch.zeros(2 * size, device="cuda", dtype=torch.int64)[::2]
        monkeypatch.setenv("SGEMM_PROFILE", "1")
        with pytest.raises(ValueError, match="profile_buffer"):
            sgemm(a, b, profile_buffer=buffer)

    @pytest.mark.parametrize(
        "case, error, message",
        [
            pytest.param("dtype", TypeError, "out.*float32", id="dtype"),
            pytest.param("device", ValueError, "out.*device", id="cpu-output"),
            pytest.param("shape", ValueError, "out", id="shape"),
            pytest.param("column_major", ValueError, "out", id="column-major"),
            pytest.param("strided", ValueError, "out", id="strided"),
            pytest.param("alignment", ValueError, "out", id="misaligned"),
        ],
    )
    def test_rejects_invalid_output(self, case: str, error: type[Exception], message: str) -> None:
        a = torch.randn(16, 16, device="cuda")
        b = torch.randn_like(a)
        match case:
            case "dtype":
                out = torch.empty_like(a, dtype=torch.float16)
            case "device":
                out = torch.empty_like(a, device="cpu")
            case "shape":
                out = torch.empty(16, 20, device="cuda")
            case "column_major":
                out = torch.empty_like(a).t()
            case "strided":
                out = torch.empty(16, 32, device="cuda")[:, ::2]
            case "alignment":
                out = torch.empty(16 * 16 + 1, device="cuda")[1:].view(16, 16)

        with pytest.raises(error, match=message):
            sgemm(a, b, out=out)

    @pytest.mark.parametrize("operand", ["a", "b"])
    @pytest.mark.parametrize(
        "offset", [0, 4, -4], ids=["exact-alias", "output-after-input", "output-before-input"]
    )
    def test_rejects_output_overlapping_input(self, operand: str, offset: int) -> None:
        storage = torch.randn(16 * 16 + abs(offset), device="cuda")
        input_start, output_start = max(-offset, 0), max(offset, 0)
        aliased = storage[input_start : input_start + 16 * 16].view(16, 16)
        other = torch.randn_like(aliased)
        a, b = (aliased, other) if operand == "a" else (other, aliased)
        out = storage[output_start : output_start + 16 * 16].view(16, 16)
        before = storage.clone()

        with pytest.raises(ValueError, match="overlap"):
            sgemm(a, b, out=out)

        torch.testing.assert_close(storage, before, atol=0, rtol=0)

    @pytest.mark.parametrize("same_inputs", [False, True], ids=["distinct-inputs", "a-is-b"])
    def test_allows_disjoint_output_in_shared_storage(self, same_inputs: bool) -> None:
        torch.manual_seed(0)
        storage = torch.randn(3, 16, 16, device="cuda")
        a = storage[0]
        b = a if same_inputs else storage[1]
        out = storage[2]
        before = storage[:2].clone()

        assert sgemm(a, b, out=out) is out

        assert_matches_fp64(out, a, b)
        torch.testing.assert_close(storage[:2], before, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", [(6, 8, 8), (8, 6, 8), (8, 8, 6)], ids=["m", "n", "k"])
    def test_rejects_unaligned_extent(self, shape: tuple[int, int, int]) -> None:
        m, n, k = shape
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        with pytest.raises(ValueError, match="multiples of 4"):
            sgemm(a, b)

    @pytest.mark.parametrize("shape", [(0, 8, 8), (8, 0, 8), (8, 8, 0)], ids=["m", "n", "k"])
    def test_rejects_empty_extent(self, shape: tuple[int, int, int]) -> None:
        m, n, k = shape
        a = torch.empty(m, k, device="cuda")
        b = torch.empty(k, n, device="cuda")
        with pytest.raises(ValueError, match="positive"):
            sgemm(a, b)


def test_naive_sgemm_tall_output_beyond_grid_y_limit(cuda_device: torch.device) -> None:
    a = torch.ones(65536 * 8, 4, device=cuda_device)
    b = torch.ones(4, 4, device=cuda_device)
    torch.testing.assert_close(naive_sgemm(a, b), torch.full_like(a, 4), atol=0, rtol=0)


def test_naive_sgemm_matches_fp64(cuda_device: torch.device) -> None:
    torch.manual_seed(0)
    a = torch.randn(132, 36, device=cuda_device)
    b = torch.randn(36, 260, device=cuda_device)
    assert_matches_fp64(naive_sgemm(a, b), a, b)
