"""Test absolute KDE bandwidths and leave-one-out range-likelihood selection."""

import numpy as np
import pytest
from scipy.stats import norm

from bdpinf.analysis import (
    RangeBandwidthSelection,
    boundary_corrected_kde_on_support,
    date_range_density,
    explode_date_ranges,
    range_loo_log_likelihood,
    select_range_bandwidth,
)

CURRENT_YEAR = 1000


def _brute_force(early, late, h, drop_identical=False):
    """Direct numerical LOO score using a dense grid and the reflected KDE."""
    early, late = np.asarray(early), np.asarray(late)
    support = float(CURRENT_YEAR - early.min())
    x = np.linspace(0.0, support, 20001)
    total = 0.0
    for i in range(len(early)):
        same = (early == early[i]) & (late == late[i])
        keep = ~same if drop_identical else np.arange(len(early)) != i
        years, w = explode_date_ranges(early[keep], late[keep])
        ages = CURRENT_YEAR - years
        dens = np.zeros_like(x)
        for centre in (ages, -ages, 2 * support - ages):
            dens += (w[:, None] * norm.pdf(x[None, :], centre[:, None], h)).sum(0)
        norm_const = np.trapezoid(dens, x)
        lo = max(CURRENT_YEAR - late[i] - 0.5, 0.0)
        hi = min(CURRENT_YEAR - early[i] + 0.5, support)
        window = np.linspace(lo, hi, 4001)
        dens_window = np.zeros_like(window)
        for centre in (ages, -ages, 2 * support - ages):
            dens_window += (
                w[:, None] * norm.pdf(window[None, :], centre[:, None], h)
            ).sum(0) / norm_const
        total += np.log(np.trapezoid(dens_window, window))
    return total


def _mixture(seed=0, n=120):
    rng = np.random.default_rng(seed)
    centres = np.where(rng.uniform(size=n) < 0.5, 300.0, 700.0)
    years = np.round(centres + rng.normal(0, 25, size=n)).astype(int)
    half = rng.integers(0, 30, size=n)
    return years - half, years + half


class TestBandwidth:
    """Test the absolute bandwidth option and the LOO selection."""

    def test_default_path_is_unchanged(self) -> None:
        early, late = _mixture(1, 40)
        grid, dens = date_range_density(early, late, CURRENT_YEAR)
        grid2, dens2 = date_range_density(early, late, CURRENT_YEAR, bandwidth=None)
        np.testing.assert_array_equal(dens, dens2)
        np.testing.assert_array_equal(grid, grid2)

    def test_absolute_bandwidth_gives_kernel_sd(self) -> None:
        samples = np.array([10.0, 20.0, 25.0, 40.0])
        weights = np.array([0.5, 1.0, 2.0, 1.0])
        h = 3.0
        x, dens = boundary_corrected_kde_on_support(
            samples, 100.0, n_grid=2001, weights=weights, bandwidth=h
        )
        assert x is not None
        assert dens is not None
        expected = sum(
            w * norm.pdf(x, c, h) for c, w in zip(samples, weights, strict=True)
        )
        expected += sum(
            w * (norm.pdf(x, -c, h) + norm.pdf(x, 200.0 - c, h))
            for c, w in zip(samples, weights, strict=True)
        )
        expected /= np.trapezoid(expected, x)
        np.testing.assert_allclose(dens, expected, atol=1e-10)

    def test_invalid_bandwidth_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            boundary_corrected_kde_on_support(np.array([1.0, 2.0]), 5.0, bandwidth=0.0)

    @pytest.mark.parametrize("leave_out", ["self", "identical"])
    def test_loo_matches_brute_force(self, leave_out) -> None:
        early = np.array([100, 120, 120, 150, 200, 210])
        late = np.array([110, 140, 140, 150, 260, 210])
        h = 12.0
        got = range_loo_log_likelihood(
            early, late, h, CURRENT_YEAR, leave_out=leave_out
        )
        want = _brute_force(early, late, h, drop_identical=leave_out == "identical")
        assert got == pytest.approx(want, abs=1e-4)

    def test_identical_differs_from_self_with_ties(self) -> None:
        early = np.array([100, 100, 100, 300, 320, 500])
        late = np.array([199, 199, 199, 350, 380, 560])
        own = range_loo_log_likelihood(early, late, 20.0, CURRENT_YEAR)
        ident = range_loo_log_likelihood(
            early, late, 20.0, CURRENT_YEAR, leave_out="identical"
        )
        assert ident < own
        no_ties = (
            np.array([100, 150, 300, 320, 500]),
            np.array([120, 170, 350, 380, 560]),
        )
        assert range_loo_log_likelihood(*no_ties, 20.0, CURRENT_YEAR) == pytest.approx(
            range_loo_log_likelihood(
                *no_ties, 20.0, CURRENT_YEAR, leave_out="identical"
            )
        )

    def test_mixture_has_interior_optimum(self) -> None:
        early, late = _mixture()
        bandwidths = np.geomspace(1.0, 400.0, 14)
        result = select_range_bandwidth(
            {"a": (early, late, CURRENT_YEAR), "b": (*_mixture(3), CURRENT_YEAR)},
            bandwidths,
        )
        assert isinstance(result, RangeBandwidthSelection)
        assert bandwidths[0] < result.shared_optimum < bandwidths[-1]
        assert set(result.curves) == {"a", "b"}
        assert all(v >= 0 for v in result.loss_at_shared.values())
        assert result.curves["a"].shape == bandwidths.shape
        assert result.optimal["a"] == bandwidths[np.argmax(result.curves["a"])]

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            range_loo_log_likelihood([1, 2], [2, 3], -1.0, 10)
        with pytest.raises(ValueError, match="leave_out"):
            range_loo_log_likelihood([1, 2], [2, 3], 1.0, 10, leave_out="x")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="required"):
            select_range_bandwidth({}, [1.0])
