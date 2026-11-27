"""Test the logistic rate, its floor and the bdsurv adapter."""

import bdsurv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.models import LogisticRate, PiecewiseConstantRate
from bdpinf.simulation import (
    simulate_birth_death,
    simulate_contour_given_survival,
    to_bdsurv_rate,
)

_T = 6.0
N_STANDARD_ERRORS = 4.0
N_SIMULATIONS = 400

_CASES = {
    "rising": LogisticRate(1.2, 1.5, 2.0, end_time=_T),
    "falling-defective": LogisticRate(1.5, -1.2, 3.0, end_time=_T),
    "falling-negative-midpoint": LogisticRate(0.8, -0.7, -1.0, end_time=_T),
    "with-floor": LogisticRate.with_floor(0.15, 1.0, 3.0, 2.5, end_time=_T),
    "steep": LogisticRate(1.0, 40.0, 3.0, end_time=_T),
}


def _quad(rate: LogisticRate, s: float, e: float) -> float:
    points = [float(rate.c)] if s < float(rate.c) < e else None
    return quad(
        lambda t: float(rate(jnp.asarray(t))), s, e, points=points, epsabs=1e-13
    )[0]


class TestLogisticRate:
    """Test values, integrals and validation."""

    @pytest.mark.parametrize("rate", list(_CASES.values()), ids=list(_CASES))
    def test_integral_matches_quadrature(self, rate: LogisticRate) -> None:
        for s, e in [(0.0, _T), (0.5, 4.0), (2.9, 3.1)]:
            got = float(rate.integral(jnp.asarray(s), jnp.asarray(e)))
            assert got == pytest.approx(_quad(rate, s, e), rel=1e-9)

    def test_value_is_logistic(self) -> None:
        rate = LogisticRate(2.0, 1.5, 1.0, end_time=_T)
        assert float(rate(jnp.asarray(1.0))) == pytest.approx(1.0)
        assert float(rate(jnp.asarray(5.0))) == pytest.approx(
            2.0 / (1.0 + np.exp(-6.0))
        )

    def test_integral_is_stable_far_from_the_midpoint(self) -> None:
        rate = LogisticRate(1.0, 50.0, -20.0, end_time=_T)  # ~constant L
        assert float(rate.integral(jnp.asarray(1.0), jnp.asarray(2.0))) == (
            pytest.approx(1.0, rel=1e-12)
        )
        late = LogisticRate(1.0, 50.0, 200.0, end_time=_T)  # ~0 on [0, T]
        tiny = float(late.integral(jnp.asarray(1.0), jnp.asarray(2.0)))
        assert 0.0 <= tiny < 1e-300

    def test_zero_steepness_is_half_the_asymptote(self) -> None:
        rate = LogisticRate(1.6, 0.0, 2.0, end_time=_T)
        assert float(rate(jnp.asarray(3.0))) == pytest.approx(0.8)
        assert float(rate.integral(jnp.asarray(1.0), jnp.asarray(4.0))) == (
            pytest.approx(0.8 * 3.0)
        )
        assert bool(
            jnp.isfinite(
                jax.grad(
                    lambda k: LogisticRate(1.6, k, 2.0, end_time=_T).integral(0.0, _T)
                )(0.0)
            )
        )

    def test_floor_adds_a_constant(self) -> None:
        base = LogisticRate(1.0, 3.0, 2.5, end_time=_T)
        floored = LogisticRate.with_floor(0.15, 1.0, 3.0, 2.5, end_time=_T)
        assert float(floored(jnp.asarray(1.0))) == pytest.approx(
            0.15 + float(base(jnp.asarray(1.0)))
        )
        assert float(floored.integral(jnp.asarray(1.0), jnp.asarray(4.0))) == (
            pytest.approx(0.15 * 3.0 + float(base.integral(1.0, 4.0)))
        )

    def test_breakpoints(self) -> None:
        assert _CASES["rising"].breakpoints.tolist() == [2.0]
        outside = LogisticRate(1.0, 1.0, -3.0, end_time=_T)
        assert outside.breakpoints.shape == (0,)

    def test_jit_and_grad_are_finite(self) -> None:
        def total(params: jax.Array) -> jax.Array:
            rate = LogisticRate(params[0], params[1], params[2], end_time=_T)
            return rate.integral(0.0, _T)

        value, grad = jax.jit(jax.value_and_grad(total))(jnp.array([1.2, -1.5, 3.0]))
        assert bool(jnp.isfinite(value))
        assert bool(jnp.all(jnp.isfinite(grad)))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"L": 0.0, "k": 1.0, "c": 1.0}, "positive"),
            ({"L": 1.0, "k": np.inf, "c": 1.0}, "finite"),
            ({"L": 1.0, "k": 1.0, "c": np.nan}, "finite"),
            ({"L": 1.0, "k": 1.0, "c": 1.0, "floor": -0.1}, "floor"),
        ],
    )
    def test_invalid_parameters_raise(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            LogisticRate(end_time=_T, **kwargs)

    def test_zero_steepness_likelihood_equals_constant_rate(self) -> None:
        ages = np.array([0.4, 1.1, 2.3, 3.0, 4.8, 5.5])
        data = AgeData.from_groups([ages], _T)
        death = PiecewiseConstantRate(jnp.array([0.3]), jnp.array([_T]))
        logistic = LogisticRate(1.6, 0.0, 2.0, end_time=_T)
        constant = PiecewiseConstantRate(jnp.array([0.8]), jnp.array([_T]))
        assert float(log_likelihood_rates(data, logistic, death)) == pytest.approx(
            float(log_likelihood_rates(data, constant, death)), rel=1e-10
        )

    def test_steep_early_step_likelihood_approaches_constant_rate(self) -> None:
        ages = np.array([0.4, 1.1, 2.3, 3.0, 4.8, 5.5])
        data = AgeData.from_groups([ages], _T)
        death = PiecewiseConstantRate(jnp.array([0.3]), jnp.array([_T]))
        logistic = LogisticRate(0.9, 400.0, -1.0, end_time=_T)  # ~0.9 on [0, T]
        constant = PiecewiseConstantRate(jnp.array([0.9]), jnp.array([_T]))
        assert float(log_likelihood_rates(data, logistic, death)) == pytest.approx(
            float(log_likelihood_rates(data, constant, death)), rel=1e-8
        )


class TestLogisticAdapter:
    """Test the conversion to bdsurv hazards and forward simulation."""

    @pytest.mark.parametrize("rate", list(_CASES.values()), ids=list(_CASES))
    def test_hazard_and_cumulative_hazard_match(self, rate: LogisticRate) -> None:
        hazard = to_bdsurv_rate(rate)
        assert isinstance(hazard, bdsurv.Hazard)
        t = np.linspace(0.05, _T - 0.05, 41)
        np.testing.assert_allclose(
            np.asarray(hazard.hazard(t)),  # pyright: ignore[reportAttributeAccessIssue]
            np.asarray(rate(jnp.asarray(t))),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(hazard.cumulative_hazard(t)),  # pyright: ignore[reportAttributeAccessIssue]
            np.asarray(rate.integral(0.0, jnp.asarray(t))),
            rtol=1e-10,
        )

    def test_types(self) -> None:
        assert isinstance(to_bdsurv_rate(_CASES["rising"]), bdsurv.LogisticHazard)
        assert isinstance(to_bdsurv_rate(_CASES["with-floor"]), bdsurv.SumHazard)

    @pytest.mark.parametrize("name", ["rising", "falling-defective", "with-floor"])
    def test_forward_simulation_mean_population(self, name: str) -> None:
        birth = _CASES[name]
        death = 0.2
        sizes = np.array(
            [
                simulate_birth_death(birth, death, _T, random_seed=seed)
                .records["survived"]
                .sum()
                for seed in range(N_SIMULATIONS)
            ],
            dtype=float,
        )
        expected = float(np.exp(float(birth.integral(0.0, _T)) - death * _T))
        standard_error = sizes.std(ddof=1) / np.sqrt(N_SIMULATIONS)
        assert abs(sizes.mean() - expected) < N_STANDARD_ERRORS * standard_error

    def test_conditioned_simulation_runs(self) -> None:
        process = simulate_contour_given_survival(
            _CASES["with-floor"], 0.2, _T, random_seed=3
        )
        assert process.records is not None
        assert bool(process.records["survived"].any())
