"""Test the piecewise cubic rate and the quadratic spline's minimum."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, PchipInterpolator

from bdpinf.models.rate_functions import (
    PiecewiseCubicRate,
    PiecewiseLinearRate,
    PiecewiseQuadraticRate,
)

TIMES = jnp.array([300.0, 700.0, 1000.0, 1200.0])
KNOTS_T = np.array([0.0, 300.0, 700.0, 1000.0, 1200.0])
VALUES = np.array([1.0, 2.0, 0.5, 1.5, 0.2])
GRID = np.linspace(0.0, 1200.0, 2001)


def _dense_min(rate: PiecewiseCubicRate | PiecewiseQuadraticRate) -> np.ndarray:
    """Per-epoch minimum from a dense evaluation grid."""
    ends = np.asarray(rate.epoch_end_times)
    starts = np.concatenate([[0.0], ends[:-1]])
    out = []
    for lo, hi in zip(starts, ends, strict=True):
        t = np.linspace(lo, hi, 20001)
        out.append(float(np.asarray(rate(t)).min()))
    return np.array(out)


class TestPiecewiseCubicRate:
    """Test the cubic Hermite rate and its constructors."""

    @pytest.mark.parametrize("kind", ["natural", "clamped", "pchip"])
    def test_constructors_match_scipy(self, kind: str) -> None:
        if kind == "natural":
            rate = PiecewiseCubicRate.natural_spline(VALUES, TIMES)
            ref = CubicSpline(KNOTS_T, VALUES, bc_type="natural")
        elif kind == "clamped":
            rate = PiecewiseCubicRate.clamped_spline(VALUES, 0.001, -0.002, TIMES)
            ref = CubicSpline(KNOTS_T, VALUES, bc_type=((1, 0.001), (1, -0.002)))
        else:
            rate = PiecewiseCubicRate.pchip(VALUES, TIMES)
            ref = PchipInterpolator(KNOTS_T, VALUES)
        np.testing.assert_allclose(rate(GRID), ref(GRID), atol=1e-12)
        np.testing.assert_allclose(
            rate.integral(0.0, GRID), ref.antiderivative()(GRID), atol=1e-10
        )

    def test_splines_join_with_continuous_slope(self) -> None:
        rate = PiecewiseCubicRate.natural_spline(VALUES, TIMES)
        np.testing.assert_allclose(rate.end_slopes[:-1], rate.start_slopes[1:])

    def test_partial_integral_and_derivative(self) -> None:
        rate = PiecewiseCubicRate.hermite(
            VALUES, np.array([0.0, 0.01, -0.01, 0.0, 0.002]), TIMES
        )
        s, e = 123.0, 987.0
        t = np.linspace(s, e, 200001)
        expected = np.trapezoid(np.asarray(rate(t)), t)
        assert float(rate.integral(s, e)) == pytest.approx(expected, rel=1e-9)
        h = 1e-4
        fd = (rate(jnp.array([400.0 + h])) - rate(jnp.array([400.0 - h]))) / (2 * h)
        assert float(rate.derivative(jnp.array([400.0]))[0]) == pytest.approx(
            float(fd[0]), rel=1e-6
        )

    def test_full_epoch_integral_formula(self) -> None:
        rate = PiecewiseCubicRate(
            jnp.array([1.0]),
            jnp.array([3.0]),
            jnp.array([0.5]),
            jnp.array([-0.25]),
            jnp.array([4.0]),
        )
        expected = 4 * (1 + 3) / 2 + 16 * (0.5 + 0.25) / 12
        assert float(rate.integral(0.0, 4.0)) == pytest.approx(expected)

    def test_min_values_matches_dense_search(self) -> None:
        rng = np.random.default_rng(5)
        for _ in range(20):
            v = rng.uniform(0.0, 2.0, 5)
            m = rng.normal(0.0, 0.01, 5)
            rate = PiecewiseCubicRate.hermite(v, m, TIMES)
            np.testing.assert_allclose(
                rate.min_values(), _dense_min(rate), atol=1e-6, rtol=0
            )

    def test_min_values_handles_degenerate_epochs(self) -> None:
        constant = PiecewiseCubicRate.hermite(np.ones(3), np.zeros(3), [1.0, 2.0])
        np.testing.assert_allclose(constant.min_values(), [1.0, 1.0])
        line = PiecewiseCubicRate.hermite(
            np.array([1.0, 0.5, 0.0]), np.full(3, -0.5), [1.0, 2.0]
        )
        np.testing.assert_allclose(line.min_values(), [0.5, 0.0], atol=1e-12)

    def test_min_values_jittable(self) -> None:
        @jax.jit
        def lowest(v: jax.Array) -> jax.Array:
            return PiecewiseCubicRate.natural_spline(v, TIMES).min_values()

        eager = PiecewiseCubicRate.natural_spline(VALUES, TIMES).min_values()
        np.testing.assert_allclose(lowest(jnp.asarray(VALUES)), eager)

    def test_pchip_never_overshoots(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(20):
            v = rng.uniform(0.0, 1.0, 5)
            rate = PiecewiseCubicRate.pchip(v, TIMES)
            assert float(rate.min_values().min()) >= v.min() - 1e-12

    def test_pchip_single_epoch_is_linear(self) -> None:
        rate = PiecewiseCubicRate.pchip(np.array([1.0, 3.0]), [10.0])
        np.testing.assert_allclose(rate(jnp.array([0.0, 5.0, 10.0])), [1.0, 2.0, 3.0])

    def test_differentiable_in_values_and_knot_times(self) -> None:
        def total(v: jax.Array, t: jax.Array) -> jax.Array:
            return PiecewiseCubicRate.natural_spline(v, t).integral(0.0, 900.0)

        v = jnp.asarray(VALUES)
        grad_v, grad_t = jax.grad(total, argnums=(0, 1))(v, TIMES)
        assert np.all(np.isfinite(grad_v))
        assert np.all(np.isfinite(grad_t))
        h = 1e-5
        step = jnp.zeros_like(v).at[2].set(h)
        fd = (total(v + step, TIMES) - total(v - step, TIMES)) / (2 * h)
        assert float(grad_v[2]) == pytest.approx(float(fd), rel=1e-6)
        shift = jnp.zeros_like(TIMES).at[1].set(h)
        fd_t = (total(v, TIMES + shift) - total(v, TIMES - shift)) / (2 * h)
        assert float(grad_t[1]) == pytest.approx(float(fd_t), rel=1e-5)

    def test_local_coefficients_reproduce_the_rate(self) -> None:
        rate = PiecewiseCubicRate.clamped_spline(VALUES, 0.001, -0.002, TIMES)
        coef = rate.local_coefficients()
        starts = np.concatenate([[0.0], np.asarray(TIMES)[:-1]])
        t = np.array([10.0, 299.0, 450.0, 800.0, 1100.0, 1199.0])
        k = np.searchsorted(np.asarray(TIMES), t)
        local = t - starts[k]
        poly = sum(coef[k, j] * local**j for j in range(4))
        np.testing.assert_allclose(poly, rate(t), rtol=1e-12)

    def test_validation_on_concrete_inputs(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            PiecewiseCubicRate.hermite(np.array([1.0, np.nan]), np.zeros(2), [1.0])
        with pytest.raises(ValueError, match="increasing"):
            PiecewiseCubicRate.hermite(np.ones(3), np.zeros(3), [2.0, 1.0])
        with pytest.raises(ValueError, match="shape"):
            PiecewiseCubicRate(
                jnp.ones(2), jnp.ones(1), jnp.zeros(1), jnp.zeros(1), jnp.array([1.0])
            )

    def test_negative_dip_is_reported_not_raised(self) -> None:
        rate = PiecewiseCubicRate.hermite(
            np.array([0.1, 0.1]), np.array([-1.0, 1.0]), [1.0]
        )
        assert float(rate.min_values()[0]) < 0.0

    def test_dips_are_allowed_under_tracing(self) -> None:
        @jax.jit
        def lowest(m: jax.Array) -> jax.Array:
            return PiecewiseCubicRate.hermite(
                jnp.array([0.1, 0.1]), jnp.array([m, m]), jnp.array([1.0])
            ).min_values()

        assert float(lowest(jnp.asarray(-3.0))[0]) < 0.0


class TestQuadraticMinValues:
    """Test the exact per-epoch minimum of the quadratic rate."""

    def test_matches_dense_search(self) -> None:
        rate = PiecewiseQuadraticRate.spline(
            jnp.array([1.0, 2.0, 1.5, 1.0, 2.0]), 0.001, TIMES
        )
        np.testing.assert_allclose(rate.min_values(), _dense_min(rate), atol=1e-6)

    def test_interior_vertex_below_endpoints(self) -> None:
        rate = PiecewiseQuadraticRate(
            jnp.array([1.0]), jnp.array([0.2]), jnp.array([1.0]), jnp.array([1.0])
        )
        assert float(rate.min_values()[0]) < 0.2

    def test_jittable(self) -> None:
        rate = PiecewiseQuadraticRate.from_linear(
            jnp.array([1.0, 2.0]), jnp.array([2.0, 0.5]), jnp.array([1.0, 2.0])
        )
        np.testing.assert_allclose(jax.jit(rate.min_values)(), [1.0, 0.5])

    def test_derivative_matches_finite_difference(self) -> None:
        rate = PiecewiseQuadraticRate.spline(
            jnp.array([1.0, 2.0, 1.5, 1.0, 2.0]), 0.001, TIMES
        )
        t, h = jnp.array([450.0, 800.0]), 1e-4
        fd = (rate(t + h) - rate(t - h)) / (2 * h)
        np.testing.assert_allclose(rate.derivative(t), fd, rtol=1e-6)

    def test_local_coefficients_reproduce_the_rate(self) -> None:
        rate = PiecewiseQuadraticRate.spline(
            jnp.array([1.0, 2.0, 1.5, 1.0, 2.0]), 0.001, TIMES
        )
        coef = rate.local_coefficients()
        starts = np.concatenate([[0.0], np.asarray(TIMES)[:-1]])
        t = np.array([10.0, 450.0, 800.0, 1100.0])
        k = np.searchsorted(np.asarray(TIMES), t)
        poly = sum(coef[k, j] * (t - starts[k]) ** j for j in range(3))
        np.testing.assert_allclose(poly, rate(t), rtol=1e-12)


class TestFlatUnless:
    """Test the harmless replacement used by the likelihood guard."""

    def test_cubic_and_quadratic_become_constant_when_not_ok(self) -> None:
        cubic = PiecewiseCubicRate.natural_spline(VALUES, TIMES)
        quad = PiecewiseQuadraticRate.spline(
            jnp.array([1.0, 2.0, 1.5, 1.0, 2.0]), 0.001, TIMES
        )
        for rate in (cubic, quad):
            kept = rate.flat_unless(jnp.asarray(True))
            np.testing.assert_allclose(kept(jnp.asarray(GRID)), rate(jnp.asarray(GRID)))
            flat = rate.flat_unless(jnp.asarray(False))
            np.testing.assert_allclose(flat(jnp.asarray(GRID)), 1.0)


class TestLinearDerivative:
    """Test the slope of the linear rate used by the spline warm starts."""

    def test_slope(self) -> None:
        rate = PiecewiseLinearRate.continuous(
            jnp.array([1.0, 2.0, 0.5]), jnp.array([10.0, 20.0])
        )
        np.testing.assert_allclose(
            rate.derivative(jnp.array([5.0, 15.0])), [0.1, -0.15]
        )
