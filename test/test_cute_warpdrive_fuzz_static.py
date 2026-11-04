"""Small regressions from declaration and compile-only mutation fuzzing."""

import pytest

pytest.importorskip("cutlass.cute")

from transformer_nuggets.cute.warpdrive import Barriers, Ring, Threads, Tma, Umma


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_tma_requires_positive_integer_bytes(value):
    with pytest.raises(ValueError, match="positive int"):
        Tma(value)


@pytest.mark.parametrize("side", ["producer", "consumer"])
@pytest.mark.parametrize("mechanism", [Threads(), Umma(), object(), Tma])
def test_channel_rejects_unknown_mechanisms(side, mechanism):
    with pytest.raises(ValueError, match="mechanism"):
        Ring(1, **{side: mechanism})


def test_local_channel_rejects_tma_consumer():
    with pytest.raises(ValueError, match="producer mechanism"):
        Ring(1, Threads, Tma(16))


@pytest.mark.parametrize("value", [1.5, True])
@pytest.mark.parametrize("declaration", [Ring, Barriers])
def test_stage_count_requires_integer(declaration, value):
    with pytest.raises(ValueError, match="positive int"):
        declaration(value)


@pytest.mark.parametrize("counts", [[1.5], [True], 1.5, "x"])
def test_raw_barrier_counts_require_positive_integers(counts):
    with pytest.raises(ValueError, match="positive ints"):
        Barriers(1, counts=counts)
