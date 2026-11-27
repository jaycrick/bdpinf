"""Test sampling ages from the exact age laws for arbitrary rate functions."""

from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from bdpinf.models import math_general
from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate
from bdpinf.simulation import (
    SampledAges,
    sample_ages_in_ranges,
    sample_ages_in_ranges_with_spine,
    simulate_ages_from_rates,
)

_TIMES = jnp.array([4.0])


def _constant(birth: float, death: float) -> tuple[PiecewiseConstantRate, ...]:
    return (
        PiecewiseConstantRate(jnp.array([birth]), _TIMES),
        PiecewiseConstantRate(jnp.array([death]), _TIMES),
    )


class TestSimulateAgesFromRates:
    """Test the sampler against the densities it inverts."""

    def test_population_layout(self) -> None:
        sample = simulate_ages_from_rates(
            *_constant(1.0, 0.4), np.random.default_rng(0)
        )
        assert isinstance(sample, SampledAges)
        assert sample.spine_age is not None
        assert sample.population == 1 + len(sample.non_spine_ages)
        assert sample.all_ages[0] == sample.spine_age
        assert np.all((sample.all_ages >= 0) & (sample.all_ages <= 4.0))

    def test_extinction_returns_empty_population(self) -> None:
        birth, death = _constant(0.2, 2.0)
        rng = np.random.default_rng(1)
        samples = [
            simulate_ages_from_rates(birth, death, rng, ensure_non_extinct=False)
            for _ in range(200)
        ]
        extinct = [s for s in samples if s.population == 0]
        assert extinct
        assert extinct[0].spine_age is None
        assert extinct[0].all_ages.size == 0

    def test_non_spine_ages_follow_exact_density(self) -> None:
        birth, death = _constant(1.0, 0.4)
        rng = np.random.default_rng(3)
        ages = np.concatenate(
            [
                simulate_ages_from_rates(birth, death, rng).non_spine_ages
                for _ in range(300)
            ]
        )
        terms = math_general.terms_for(birth, death)
        grid = jnp.linspace(0.0, 4.0, 4001)
        pdf = np.exp(np.asarray(math_general.non_spine_log_density(terms, grid)))
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * 4.0 / 4000)]
        )
        cdf /= cdf[-1]
        result = stats.kstest(ages, lambda x: np.interp(x, np.asarray(grid), cdf))
        assert result.pvalue > 0.001

    @pytest.mark.slow
    def test_linear_rates_and_population_size(self) -> None:
        times = jnp.array([4.0])
        birth = PiecewiseLinearRate.continuous(jnp.array([0.4, 1.2]), times)
        death = PiecewiseLinearRate.continuous(jnp.array([0.6, 0.3]), times)
        rng = np.random.default_rng(5)
        sizes = [
            simulate_ages_from_rates(birth, death, rng).population for _ in range(400)
        ]
        terms = math_general.terms_for(birth, death)
        pi = float(np.exp(-math_general.math_nested.log_nested_w_total(terms)))
        assert np.mean(sizes) == pytest.approx(1.0 / pi, rel=0.25)


class TestSampleAgesInRanges:
    """Test range-truncated draws from the non-spine age density."""

    def test_shape_and_ranges(self) -> None:
        birth, death = _constant(1.0, 0.4)
        lower = np.array([0.0, 1.0, 3.5, 2.0])
        upper = np.array([4.0, 2.0, 4.0, 2.0])
        ages = sample_ages_in_ranges(
            birth, death, lower, upper, np.random.default_rng(0), n_draws=50
        )
        assert ages.shape == (50, 4)
        assert np.all(ages >= lower)
        assert np.all(ages <= upper)
        assert np.all(ages[:, 3] == 2.0)

    def test_matches_truncated_density(self) -> None:
        times = jnp.array([2.0, 4.0])
        birth = PiecewiseLinearRate(jnp.array([0.1, 1.0]), jnp.array([0.4, 0.2]), times)
        death = PiecewiseLinearRate(jnp.array([0.3, 0.3]), jnp.array([0.3, 0.3]), times)
        lo, hi = 0.5, 3.0
        n_draws = 4000
        ages = sample_ages_in_ranges(
            birth,
            death,
            np.array([lo]),
            np.array([hi]),
            np.random.default_rng(4),
            n_draws=n_draws,
        )[:, 0]
        terms = math_general.terms_for(birth, death)
        grid = np.linspace(lo, hi, 20001)
        pdf = np.exp(
            np.asarray(math_general.non_spine_log_density(terms, jnp.asarray(grid)))
        )
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))]
        )
        cdf /= cdf[-1]
        result = stats.kstest(ages, lambda x: np.interp(x, grid, cdf))
        assert result.pvalue > 1e-3

    def test_zero_mass_falls_back_to_uniform(self) -> None:
        # No births before time 2, so no individual is older than 2: ages in
        # [3, 4] have zero mass.
        times = jnp.array([2.0, 4.0])
        birth = PiecewiseConstantRate(jnp.array([0.0, 1.0]), times)
        death = PiecewiseConstantRate(jnp.array([0.2, 0.2]), times)
        ages = sample_ages_in_ranges(
            birth,
            death,
            np.array([3.0]),
            np.array([4.0]),
            np.random.default_rng(2),
            n_draws=2000,
        )[:, 0]
        assert np.all((ages >= 3.0) & (ages <= 4.0))
        assert stats.kstest(ages, "uniform", args=(3.0, 1.0)).pvalue > 1e-3

    def test_rejects_bad_input(self) -> None:
        birth, death = _constant(1.0, 0.4)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="same length"):
            sample_ages_in_ranges(birth, death, [0.0], [1.0, 2.0], rng, n_draws=2)
        with pytest.raises(ValueError, match="exceed"):
            sample_ages_in_ranges(birth, death, [2.0], [1.0], rng, n_draws=2)


def _mass(grid_pdf: np.ndarray, grid: np.ndarray, lo: float, hi: float) -> float:
    inside = (grid >= lo) & (grid <= hi)
    return float(np.trapezoid(grid_pdf[inside], grid[inside]))


class TestSampleAgesInRangesWithSpine:
    """Test the spine-aware range sampler."""

    def _densities(self, birth, death, n=40001):
        terms = math_general.terms_for(birth, death)
        grid = np.linspace(0.0, 4.0, n)
        jgrid = jnp.asarray(grid)
        f = np.exp(np.asarray(math_general.non_spine_log_density(terms, jgrid)))
        fs = np.exp(
            np.asarray(
                math_general.spine_log_density(terms, jgrid, atom_at_boundary=False)
            )
        )
        atom = float(np.exp(math_general.spine_log_atom(terms)))
        return grid, f, fs, atom

    def test_full_ranges_spine_marginal_matches_spine_density(self) -> None:
        birth, death = _constant(1.0, 0.4)
        grid, _, fs, atom = self._densities(birth, death)
        n = 5
        ages, spine = sample_ages_in_ranges_with_spine(
            birth, death, np.zeros(n), np.full(n, 4.0),
            np.random.default_rng(1), n_draws=20000,
        )  # fmt: skip
        spine_ages = ages[np.arange(len(spine)), spine]
        edges = np.linspace(0.0, 4.0, 9)
        observed = np.histogram(spine_ages[spine_ages < 4.0], bins=edges)[0] / len(
            spine
        )
        expected = np.array([_mass(fs, grid, a, b) for a, b in pairwise(edges)])
        assert np.allclose(observed, expected, atol=0.012)
        assert np.mean(spine_ages == 4.0) == pytest.approx(atom, abs=0.012)
        assert np.bincount(spine, minlength=n) / len(spine) == pytest.approx(
            1 / n, abs=0.015
        )

    def test_spine_index_frequencies_match_numerical_integration(self) -> None:
        birth, death = _constant(1.0, 0.4)
        grid, f, fs, _ = self._densities(birth, death)
        lower = np.array([0.2, 1.4, 2.6])
        upper = np.array([0.5, 1.7, 2.9])
        weights = np.array(
            [
                _mass(fs, grid, a, b) / _mass(f, grid, a, b)
                for a, b in zip(lower, upper, strict=True)
            ]
        )
        expected = weights / weights.sum()
        ages, spine = sample_ages_in_ranges_with_spine(
            birth, death, lower, upper, np.random.default_rng(2), n_draws=30000
        )
        observed = np.bincount(spine, minlength=3) / len(spine)
        assert np.allclose(observed, expected, atol=0.012)
        assert np.all((ages >= lower) & (ages <= upper))

    def test_range_containing_end_time_uses_atom(self) -> None:
        birth, death = _constant(1.0, 0.4)
        atom = float(
            np.exp(math_general.spine_log_atom(math_general.terms_for(birth, death)))
        )
        ages, spine = sample_ages_in_ranges_with_spine(
            birth, death, [0.0, 3.5], [1.0, 9.0], np.random.default_rng(3), n_draws=8000
        )
        on_boundary = ages[np.arange(len(spine)), spine] == 4.0
        assert on_boundary[spine == 1].mean() > 0.5 * atom
        assert not np.any(on_boundary[spine == 0])
        assert np.all(ages[:, 1] <= 4.0)

    def test_shapes_and_validation(self) -> None:
        birth, death = _constant(1.0, 0.4)
        ages, spine = sample_ages_in_ranges_with_spine(
            birth, death, [0.0, 1.0, 2.0, 5.0], [1.0, 2.0, 3.0, 6.0],
            np.random.default_rng(4), n_draws=7,
        )  # fmt: skip
        assert ages.shape == (7, 4)
        assert spine.shape == (7,)
        assert np.all((spine >= 0) & (spine < 4))
        ages1, spine1 = sample_ages_in_ranges_with_spine(
            birth, death, [0.0], [1.0], np.random.default_rng(5)
        )
        assert ages1.shape == (1, 1)
        assert spine1.tolist() == [0]
        with pytest.raises(ValueError, match="must not exceed"):
            sample_ages_in_ranges_with_spine(
                birth, death, [2.0], [1.0], np.random.default_rng(0)
            )
        with pytest.raises(ValueError, match="1-D"):
            sample_ages_in_ranges_with_spine(
                birth, death, [], [], np.random.default_rng(0)
            )
