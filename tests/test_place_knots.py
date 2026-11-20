"""Test place_knots and evenly_spaced_knots."""

import itertools

import pytest

from bdpinf.inference import evenly_spaced_knots, place_knots


class TestPlaceKnots:
    def test_spacing_enforced(self) -> None:
        got = place_knots([100, 110, 120], 50, 950, 50.0, margin=1.0)
        assert all(b - a >= 50.0 for a, b in itertools.pairwise(got))

    def test_pushed_back_from_upper_bound(self) -> None:
        got = place_knots([940, 945], 50, 950, 50.0, margin=1.0)
        assert got[-1] < 950.0
        assert all(b - a >= 50.0 for a, b in itertools.pairwise(got))

    def test_infeasible_raises(self) -> None:
        with pytest.raises(ValueError, match="do not fit"):
            place_knots([60, 70, 80], 50, 140, 50.0, margin=1.0)

    def test_evenly_spaced(self) -> None:
        assert evenly_spaced_knots(2, 50, 950, 50.0, margin=1.0) == pytest.approx(
            [350.0, 650.0]
        )

    def test_zero_margin_allows_bounds(self) -> None:
        assert place_knots([-5, 105], 0, 100) == [0.0, 100.0]

    def test_empty_and_k_zero(self) -> None:
        assert place_knots([], 0, 100, 10.0) == []
        assert evenly_spaced_knots(0, 0, 100) == []

    def test_unsorted_input_sorted(self) -> None:
        assert place_knots([70, 20, 40], 0, 100) == [20.0, 40.0, 70.0]

    @pytest.mark.parametrize(
        "kwargs",
        [{"margin": -1.0}, {"min_spacing": -1.0}],
    )
    def test_negative_arguments_raise(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            place_knots([1.0], 0, 100, **kwargs)

    def test_hi_not_above_lo_raises(self) -> None:
        with pytest.raises(ValueError, match="exceed"):
            place_knots([1.0], 5, 5)

    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            evenly_spaced_knots(-1, 0, 100)
