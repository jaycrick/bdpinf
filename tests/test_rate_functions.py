"""Test piecewise-constant and piecewise-linear rate functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from bdpinf.models.rate_functions import (
    PiecewiseConstantRate,
    PiecewiseExponentialRate,
    PiecewiseLinearRate,
    PiecewiseQuadraticRate,
    RateFunction,
    SinusoidalRate,
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


class TestPiecewiseExponentialRate:
    def test_log_is_linear_within_epochs(self) -> None:
        rate = PiecewiseExponentialRate(
            jnp.array([0.5, 2.0]), jnp.array([1.5, 0.3]), _TIMES
        )
        assert float(rate(jnp.array(0.5))) == pytest.approx(np.sqrt(0.5 * 1.5))
        assert float(rate(jnp.array(1.0))) == pytest.approx(1.5)
        assert float(rate(jnp.array(1.0 + 1.5 / 2))) == pytest.approx(
            np.sqrt(2.0 * 0.3)
        )
        assert float(rate(jnp.array(0.0))) == pytest.approx(0.5)

    def test_closed_form_integral(self) -> None:
        rate = PiecewiseExponentialRate.continuous(
            jnp.array([0.4, 1.6]), jnp.array([2.0])
        )
        c = np.log(1.6 / 0.4) / 2.0
        exact = 0.4 * np.expm1(c * 1.3) / c
        assert float(rate.integral(jnp.array(0.0), jnp.array(1.3))) == pytest.approx(
            exact, rel=1e-13
        )

    def test_continuous_has_no_jump(self) -> None:
        rate = PiecewiseExponentialRate.continuous(jnp.array([1.0, 2.0, 0.5]), _TIMES)
        left, right = rate(jnp.array(1.0 - 1e-12)), rate(jnp.array(1.0 + 1e-12))
        assert float(left) == pytest.approx(float(right), rel=1e-9)
        assert float(rate(jnp.array(2.5))) == pytest.approx(0.5)

    def test_from_constant_equals_piecewise_constant(self) -> None:
        values = jnp.array([0.7, 1.3])
        pe = PiecewiseExponentialRate.from_constant(values, _TIMES)
        pc = PiecewiseConstantRate(values, _TIMES)
        s, e = jnp.array(0.2), jnp.array(2.0)
        assert float(pe.integral(s, e)) == pytest.approx(
            float(pc.integral(s, e)), rel=1e-14
        )
        assert float(pe(jnp.array(1.7))) == pytest.approx(1.3, rel=1e-14)

    @pytest.mark.parametrize("rel_change", [0.0, 1e-15, 1e-12, -1e-9, 1e-6, 1e-4])
    def test_near_constant_epoch_matches_scipy(self, rel_change: float) -> None:
        rate = PiecewiseExponentialRate(
            jnp.array([0.8]), jnp.array([0.8 * (1.0 + rel_change)]), jnp.array([1.7])
        )
        numeric, _ = quad(
            lambda x: float(rate(jnp.float64(x))), 0.2, 1.5, epsabs=1e-14, epsrel=1e-13
        )
        exact = float(rate.integral(jnp.array(0.2), jnp.array(1.5)))
        assert exact == pytest.approx(numeric, rel=1e-12)

    def test_gradient_at_exactly_constant_epoch(self) -> None:
        def total(end: jax.Array) -> jax.Array:
            rate = PiecewiseExponentialRate(jnp.array([0.8]), end, jnp.array([2.0]))
            return rate.integral(jnp.array(0.0), jnp.array(2.0))

        grad = jax.grad(total)(jnp.array([0.8]))
        # d/d(end) of a d phi(log(end/a)) at end = a is d / 2.
        np.testing.assert_allclose(np.asarray(grad), [1.0], rtol=1e-10)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_or_non_finite(self, bad: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            PiecewiseExponentialRate(
                jnp.array([1.0, bad]), jnp.array([1.0, 1.0]), _TIMES
            )

    def test_breakpoints_are_interior_epoch_ends(self) -> None:
        rate = PiecewiseExponentialRate.continuous(jnp.array([1.0, 2.0, 0.5]), _TIMES)
        np.testing.assert_array_equal(np.asarray(rate.breakpoints), [1.0])


class TestPiecewiseQuadraticRate:
    def test_interpolates_the_three_values(self) -> None:
        rate = PiecewiseQuadraticRate(
            jnp.array([0.4, 1.0]),
            jnp.array([1.6, 0.2]),
            jnp.array([0.8, 1.5]),
            _TIMES,
        )
        assert float(rate(jnp.array(0.0))) == pytest.approx(0.4)
        assert float(rate(jnp.array(0.5))) == pytest.approx(1.6)
        assert float(rate(jnp.array(1.0))) == pytest.approx(0.8)
        assert float(rate(jnp.array(1.0 + 0.75))) == pytest.approx(0.2)
        assert float(rate(jnp.array(2.5))) == pytest.approx(1.5)

    def test_full_epoch_integral_is_simpson(self) -> None:
        rate = PiecewiseQuadraticRate(
            jnp.array([0.4]), jnp.array([1.6]), jnp.array([0.8]), jnp.array([2.0])
        )
        simpson = 2.0 * (0.4 + 4 * 1.6 + 0.8) / 6.0
        assert float(rate.integral(jnp.array(0.0), jnp.array(2.0))) == pytest.approx(
            simpson, rel=1e-14
        )

    def test_partial_epoch_uses_the_antiderivative(self) -> None:
        rate = PiecewiseQuadraticRate(
            jnp.array([0.4]), jnp.array([1.6]), jnp.array([0.8]), jnp.array([2.0])
        )
        numeric, _ = quad(lambda x: float(rate(jnp.float64(x))), 0.3, 1.1)
        assert float(rate.integral(jnp.array(0.3), jnp.array(1.1))) == pytest.approx(
            numeric, rel=1e-12
        )

    def test_continuous_has_no_jump(self) -> None:
        rate = PiecewiseQuadraticRate.continuous(
            jnp.array([0.6, 1.4, 0.3]), jnp.array([1.3, 0.2]), _TIMES
        )
        assert float(rate(jnp.array(1.0 - 1e-12))) == pytest.approx(1.4, abs=1e-9)
        assert float(rate(jnp.array(1.0 + 1e-12))) == pytest.approx(1.4, abs=1e-9)

    def test_spline_is_c1_at_the_knots(self) -> None:
        times = jnp.array([0.8, 1.9, 3.0])
        rate = PiecewiseQuadraticRate.spline(
            jnp.array([1.0, 1.6, 1.2, 0.8]), 0.5, times
        )
        slope = jax.grad(lambda t: rate(t))
        assert float(slope(jnp.array(0.0))) == pytest.approx(0.5, abs=1e-12)
        for knot, value in zip([0.8, 1.9], [1.6, 1.2], strict=True):
            eps = 1e-6
            left = float(rate(jnp.array(knot - eps)))
            right = float(rate(jnp.array(knot + eps)))
            assert left == pytest.approx(value, abs=1e-5)
            assert right == pytest.approx(value, abs=1e-5)
            left_slope = float(slope(jnp.array(knot - eps)))
            right_slope = float(slope(jnp.array(knot + eps)))
            assert left_slope == pytest.approx(right_slope, abs=1e-5), (
                f"slope jumps at {knot}: {left_slope} vs {right_slope}"
            )
        assert float(rate(jnp.array(3.0))) == pytest.approx(0.8)

    def test_spline_slope_recurrence(self) -> None:
        rate = PiecewiseQuadraticRate.spline(
            jnp.array([1.0, 1.6, 1.2]), 0.5, jnp.array([0.8, 1.9])
        )
        end_slope = 2 * (1.6 - 1.0) / 0.8 - 0.5
        slope = jax.grad(lambda t: rate(t))
        assert float(slope(jnp.array(0.8 - 1e-9))) == pytest.approx(end_slope, abs=1e-6)

    def test_spline_dipping_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="epoch 0"):
            PiecewiseQuadraticRate.spline(
                jnp.array([1.0, 0.1, 1.0]), -3.0, jnp.array([1.0, 2.0])
            )

    def test_interior_minimum_below_zero_raises(self) -> None:
        # Stored values are all non-negative; the vertex of epoch 1 dips below zero.
        with pytest.raises(ValueError, match="epoch 1"):
            PiecewiseQuadraticRate(
                jnp.array([1.0, 1.0]),
                jnp.array([1.0, 0.05]),
                jnp.array([1.0, 0.0]),
                _TIMES,
            )

    def test_vertex_outside_the_epoch_is_allowed(self) -> None:
        # Increasing on the epoch; the parabola dips below zero only before it.
        rate = PiecewiseQuadraticRate.from_linear(
            jnp.array([0.0]), jnp.array([2.0]), jnp.array([1.0])
        )
        assert float(rate(jnp.array(0.5))) == pytest.approx(1.0)

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            PiecewiseQuadraticRate(
                jnp.array([1.0]),
                jnp.array([jnp.nan]),
                jnp.array([1.0]),
                jnp.array([1.0]),
            )

    def test_from_linear_matches_piecewise_linear(self) -> None:
        start, end = jnp.array([0.4, 1.5]), jnp.array([1.2, 0.2])
        quadratic = PiecewiseQuadraticRate.from_linear(start, end, _TIMES)
        linear = PiecewiseLinearRate(start, end, _TIMES)
        grid = jnp.linspace(0.0, 2.5, 101)
        np.testing.assert_allclose(
            np.asarray(quadratic(grid)), np.asarray(linear(grid)), atol=1e-14
        )
        np.testing.assert_allclose(
            np.asarray(quadratic.integral(jnp.zeros_like(grid), grid)),
            np.asarray(linear.integral(jnp.zeros_like(grid), grid)),
            atol=1e-14,
        )


class TestSinusoidalRate:
    def test_value_and_closed_form_integral(self) -> None:
        rate = SinusoidalRate(1.0, 0.4, 2.0, 0.3, end_time=5.0)
        assert float(rate(jnp.array(0.0))) == pytest.approx(1.0 + 0.4 * np.sin(0.3))
        s, e = 0.7, 3.9
        omega = 2 * np.pi / 2.0
        exact = 1.0 * (e - s) - 0.4 / omega * (
            np.cos(omega * e + 0.3) - np.cos(omega * s + 0.3)
        )
        assert float(rate.integral(jnp.array(s), jnp.array(e))) == pytest.approx(
            exact, rel=1e-13
        )

    def test_breakpoints_are_interior_half_periods(self) -> None:
        rate = SinusoidalRate(1.0, 0.5, 1.0, end_time=3.2)
        np.testing.assert_allclose(
            np.asarray(rate.breakpoints), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        )

    def test_breakpoints_exclude_the_end_time(self) -> None:
        rate = SinusoidalRate(1.0, 0.5, 1.0, end_time=3.0)
        np.testing.assert_allclose(
            np.asarray(rate.breakpoints), [0.5, 1.0, 1.5, 2.0, 2.5]
        )

    def test_no_breakpoints_below_half_a_period(self) -> None:
        rate = SinusoidalRate(1.0, 0.5, 10.0, end_time=2.0)
        assert rate.breakpoints.shape == (0,)

    def test_period_and_end_time_are_static(self) -> None:
        rate = SinusoidalRate(1.0, 0.5, 1.0, end_time=3.0)
        leaves = jax.tree_util.tree_leaves(rate)
        assert len(leaves) == 3

    def test_gradient_wrt_mean_amplitude_phase(self) -> None:
        def total(params: jax.Array) -> jax.Array:
            rate = SinusoidalRate(params[0], params[1], 1.3, params[2], end_time=2.5)
            return rate.integral(jnp.array(0.0), jnp.array(2.5))

        grad = jax.jit(jax.grad(total))(jnp.array([1.0, 0.4, 0.2]))
        assert grad[0] == pytest.approx(2.5)
        assert np.all(np.isfinite(np.asarray(grad)))

    def test_default_phase_is_zero(self) -> None:
        rate = SinusoidalRate(1.0, 0.4, 2.0, end_time=5.0)
        assert float(rate(jnp.array(0.5))) == pytest.approx(1.0 + 0.4)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"mean": 0.3, "amplitude": 0.5}, "mean must be at least"),
            ({"mean": 0.3, "amplitude": -0.5}, "mean must be at least"),
            ({"period": 0.0}, "period"),
            ({"period": -1.0}, "period"),
            ({"end_time": 0.0}, "end_time"),
            ({"mean": float("nan")}, "finite"),
            ({"amplitude": float("inf")}, "finite"),
            ({"phase": float("nan")}, "finite"),
        ],
    )
    def test_validation(self, kwargs: dict[str, float], match: str) -> None:
        params = {"mean": 1.0, "amplitude": 0.5, "period": 1.0, "end_time": 2.0}
        with pytest.raises(ValueError, match=match):
            SinusoidalRate(**{**params, **kwargs})
