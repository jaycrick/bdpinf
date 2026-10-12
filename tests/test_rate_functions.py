"""Test piecewise-constant and piecewise-linear rate functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.models.rate_functions import (
    PiecewiseConstantRate,
    PiecewiseLinearRate,
    RateFunction,
)

_TIMES = jnp.array([1.0, 2.5])


class TestPiecewiseConstantRate:
    def test_value_and_integral(self) -> None:
        rate = PiecewiseConstantRate(jnp.array([2.0, 1.0]), _TIMES)
        assert float(rate(jnp.array(0.5))) == 2.0
        assert float(rate(jnp.array(2.0))) == 1.0
        assert float(rate.integral(jnp.array(0.5), jnp.array(2.0))) == pytest.approx(
            0.5 * 2.0 + 1.0 * 1.0
        )

    def test_constant_constructor_and_protocol(self) -> None:
        rate = PiecewiseConstantRate.constant(0.3, 4.0)
        assert isinstance(rate, RateFunction)
        assert float(rate.end_time) == 4.0
        assert rate.breakpoints.shape == (0,)

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            PiecewiseConstantRate(jnp.array([-1.0, 1.0]), _TIMES)

    def test_rejects_non_increasing_times(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            PiecewiseConstantRate(jnp.array([1.0, 1.0]), jnp.array([2.0, 1.0]))


class TestPiecewiseLinearRate:
    def test_linear_value_and_integral(self) -> None:
        rate = PiecewiseLinearRate.continuous(jnp.array([1.0, 2.0, 0.5]), _TIMES)
        assert float(rate(jnp.array(0.5))) == pytest.approx(1.5)
        assert float(rate(jnp.array(1.0))) == pytest.approx(2.0)
        assert float(rate.integral(jnp.array(0.0), jnp.array(1.0))) == pytest.approx(
            1.5
        )
        total = 1.5 + 0.5 * (2.0 + 0.5) * 1.5
        assert float(rate.integral(jnp.array(0.0), jnp.array(2.5))) == pytest.approx(
            total
        )

    def test_jump_between_epochs(self) -> None:
        rate = PiecewiseLinearRate(jnp.array([1.0, 3.0]), jnp.array([1.0, 3.0]), _TIMES)
        assert float(rate(jnp.array(0.9))) == pytest.approx(1.0)
        assert float(rate(jnp.array(1.1))) == pytest.approx(3.0)

    def test_integral_matches_numerical_quadrature(self) -> None:
        rate = PiecewiseLinearRate(jnp.array([0.4, 1.5]), jnp.array([1.2, 0.2]), _TIMES)
        grid = np.linspace(0.3, 2.2, 200001)
        numeric = np.trapezoid(np.asarray(rate(jnp.asarray(grid))), grid)
        exact = float(rate.integral(jnp.array(0.3), jnp.array(2.2)))
        assert exact == pytest.approx(numeric, rel=1e-5)  # trapezoid straddles the kink

    def test_zero_slope_equals_piecewise_constant(self) -> None:
        values = jnp.array([0.7, 1.3])
        pl = PiecewiseLinearRate.from_constant(values, _TIMES)
        pc = PiecewiseConstantRate(values, _TIMES)
        s, e = jnp.array(0.2), jnp.array(2.0)
        assert float(pl.integral(s, e)) == pytest.approx(float(pc.integral(s, e)))
        assert float(pl(jnp.array(1.7))) == pytest.approx(float(pc(jnp.array(1.7))))

    def test_is_pytree_and_differentiable(self) -> None:
        def total(knots: jax.Array) -> jax.Array:
            rate = PiecewiseLinearRate.continuous(knots, _TIMES)
            return rate.integral(jnp.array(0.0), jnp.array(2.5))

        grad = jax.jit(jax.grad(total))(jnp.array([1.0, 2.0, 0.5]))
        assert np.all(np.isfinite(np.asarray(grad)))
        assert float(grad.sum()) == pytest.approx(2.5)
