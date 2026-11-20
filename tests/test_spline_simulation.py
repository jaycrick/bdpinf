"""Test simulating from spline rates through the bdsurv hazard adapter."""

import bdsurv
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.models.rate_functions import (
    PiecewiseCubicRate,
    PiecewiseLinearRate,
    PiecewiseQuadraticRate,
)
from bdpinf.simulation import (
    simulate_birth_death,
    simulate_contour_given_survival,
    to_bdsurv_rate,
)

needs_polynomials = pytest.mark.skipif(
    not hasattr(bdsurv.PiecewiseHazard, "from_polynomials"),
    reason="needs bdsurv >= 0.8.0 (PiecewiseHazard.from_polynomials)",
)

TIMES = jnp.array([3.0, 6.0])
N_STANDARD_ERRORS = 4.0
N_SIMULATIONS = 400


def _cubic_birth() -> PiecewiseCubicRate:
    return PiecewiseCubicRate.natural_spline(jnp.array([0.3, 0.6, 0.4]), TIMES)


def _quadratic_birth() -> PiecewiseQuadraticRate:
    return PiecewiseQuadraticRate.spline(jnp.array([0.3, 0.5, 0.4]), 0.05, TIMES)


def _expected_mean_population(
    birth: PiecewiseCubicRate | PiecewiseQuadraticRate, death: float
) -> float:
    """``E[Z_T] = exp(int (birth - death))`` for a process started from one."""
    end = float(birth.end_time)
    return float(np.exp(float(birth.integral(0.0, end)) - death * end))


class TestSplineAdapter:
    """Test the conversion of spline rates to bdsurv hazards."""

    def test_other_rates_pass_through(self) -> None:
        assert to_bdsurv_rate(0.5) == 0.5
        linear = PiecewiseLinearRate.continuous(jnp.array([1.0, 2.0, 1.0]), TIMES)
        assert to_bdsurv_rate(linear) is linear

    @needs_polynomials
    @pytest.mark.parametrize("make", [_cubic_birth, _quadratic_birth])
    def test_hazard_matches_the_rate(
        self, make: type[PiecewiseCubicRate] | type[PiecewiseQuadraticRate]
    ) -> None:
        rate = make()  # type: ignore[call-arg]
        hazard = to_bdsurv_rate(rate)
        t = np.linspace(0.05, 5.95, 41)
        np.testing.assert_allclose(
            np.asarray(hazard.hazard(t)),  # type: ignore[union-attr]
            np.asarray(rate(jnp.asarray(t))),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(hazard.cumulative_hazard(t)),  # type: ignore[union-attr]
            np.asarray(rate.integral(0.0, jnp.asarray(t))),
            rtol=1e-10,
        )

    @needs_polynomials
    @pytest.mark.parametrize("make", [_cubic_birth, _quadratic_birth])
    def test_forward_simulation_mean_population(
        self, make: type[PiecewiseCubicRate] | type[PiecewiseQuadraticRate]
    ) -> None:
        birth = make()  # type: ignore[call-arg]
        death = 0.2
        sizes = np.array(
            [
                simulate_birth_death(birth, death, random_seed=seed)
                .records["survived"]
                .sum()
                for seed in range(N_SIMULATIONS)
            ],
            dtype=float,
        )
        expected = _expected_mean_population(birth, death)
        standard_error = sizes.std(ddof=1) / np.sqrt(N_SIMULATIONS)
        assert abs(sizes.mean() - expected) < N_STANDARD_ERRORS * standard_error

    @needs_polynomials
    def test_conditioned_simulation_runs_with_a_cubic_birth_rate(self) -> None:
        process = simulate_contour_given_survival(_cubic_birth(), 0.2, random_seed=3)
        assert process.records is not None
        assert bool(process.records["survived"].any())

    @needs_polynomials
    def test_cubic_death_rate_is_converted_too(self) -> None:
        process = simulate_birth_death(0.4, _cubic_birth(), random_seed=1)
        assert process.records is not None
