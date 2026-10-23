"""Test sampling ages from the exact age laws for arbitrary rate functions."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from bdpinf.models import math_general
from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate
from bdpinf.simulation import SampledAges, simulate_ages_from_rates

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
