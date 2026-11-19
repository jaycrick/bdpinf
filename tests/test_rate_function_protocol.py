"""Test the RateFunction protocol properties for every rate family."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad

from bdpinf.models.rate_functions import (
    LogLogisticRate,
    LomaxRate,
    PiecewiseConstantRate,
    PiecewiseExponentialRate,
    PiecewiseLinearRate,
    PiecewiseQuadraticRate,
    PiecewiseRate,
    PowerRate,
    RateFunction,
    SinusoidalRate,
)

_J = jnp.array
_TIMES3 = _J([0.8, 1.9, 3.0])

# Add an example here when adding a family; every test below is parametrised over it.
_EXAMPLES: dict[str, RateFunction] = {
    "constant-single": PiecewiseConstantRate.constant(0.7, 2.0),
    "constant-three-epochs": PiecewiseConstantRate(_J([1.2, 0.4, 2.0]), _TIMES3),
    "constant-with-zero": PiecewiseConstantRate(_J([0.0, 1.5, 0.0]), _TIMES3),
    "linear-single": PiecewiseLinearRate.continuous(_J([0.5, 1.8]), _J([2.5])),
    "linear-continuous-three": PiecewiseLinearRate.continuous(
        _J([1.0, 2.0, 0.5, 1.5]), _TIMES3
    ),
    "linear-jumps-and-zeros": PiecewiseLinearRate(
        _J([0.0, 2.0, 0.3]), _J([1.0, 0.0, 0.0]), _TIMES3
    ),
    "exponential-single": PiecewiseExponentialRate.continuous(
        _J([0.3, 1.7]), _J([2.2])
    ),
    "exponential-continuous-three": PiecewiseExponentialRate.continuous(
        _J([1.0, 2.5, 0.4, 1.1]), _TIMES3
    ),
    "exponential-jumps": PiecewiseExponentialRate(
        _J([0.5, 2.0, 0.2]), _J([1.5, 0.3, 0.9]), _TIMES3
    ),
    "exponential-constant-epoch": PiecewiseExponentialRate(
        _J([0.9, 0.6]), _J([0.9, 1.4]), _J([1.0, 2.5])
    ),
    "exponential-nearly-constant": PiecewiseExponentialRate(
        _J([0.9]), _J([0.9 * (1 + 1e-9)]), _J([2.0])
    ),
    "quadratic-single": PiecewiseQuadraticRate(
        _J([0.4]), _J([1.6]), _J([0.8]), _J([2.0])
    ),
    "quadratic-convex": PiecewiseQuadraticRate(
        _J([1.0]), _J([0.2]), _J([1.0]), _J([2.0])
    ),
    "quadratic-continuous-three": PiecewiseQuadraticRate.continuous(
        _J([0.6, 1.4, 0.3, 0.9]), _J([1.3, 0.2, 0.8]), _TIMES3
    ),
    "quadratic-spline-three": PiecewiseQuadraticRate.spline(
        _J([1.0, 1.6, 1.2, 0.8]), 0.5, _TIMES3
    ),
    "quadratic-from-linear-jumps": PiecewiseQuadraticRate.from_linear(
        _J([0.0, 2.0, 0.3]), _J([1.0, 0.0, 0.5]), _TIMES3
    ),
    "sinusoidal-short": SinusoidalRate(1.0, 0.6, 0.7, end_time=2.5),
    "sinusoidal-phase": SinusoidalRate(0.9, 0.5, 1.3, 0.8, end_time=3.0),
    "sinusoidal-touches-zero": SinusoidalRate(0.5, 0.5, 2.0, 1.1, end_time=3.0),
    "sinusoidal-under-half-period": SinusoidalRate(1.2, -0.4, 10.0, end_time=2.0),
    "sinusoidal-period-divides-end": SinusoidalRate(1.0, 0.3, 1.0, end_time=3.0),
    "gompertz-growing": PiecewiseExponentialRate.gompertz(0.2, 0.4, 2.5),
    "gompertz-decaying": PiecewiseExponentialRate.gompertz(0.5, -0.6, 2.5),
    "power-increasing": PowerRate(0.7, 0.5, end_time=2.5),
    "power-rayleigh": PowerRate.rayleigh(0.3, end_time=2.5),
    "power-weibull-wearout": PowerRate.weibull(2.5, 3.0, end_time=2.5),
    "loglogistic-unimodal": LogLogisticRate(2.0, 3.0, end_time=2.5),
    "lomax": LomaxRate(1.5, 2.0, end_time=2.5),
    "piecewise-mixed-families": PiecewiseRate(
        (
            PiecewiseExponentialRate.gompertz(0.3, 0.5, 0.8),
            PowerRate.weibull(0.7, 2.0, end_time=1.1),
            LomaxRate(1.2, 1.0, end_time=1.1),
        ),
        _TIMES3,
    ),
}
_IDS = list(_EXAMPLES)
_RATES = list(_EXAMPLES.values())


_CALL = jax.jit(lambda rate, x: rate(x))


def _end(rate: RateFunction) -> float:
    return float(rate.end_time)


def _pairs(rate: RateFunction) -> list[tuple[float, float]]:
    """Pairs ``(s, e)``: the whole window, inside one epoch, spanning breakpoints."""
    t = _end(rate)
    pairs = [(0.0, t), (0.13 * t, 0.29 * t), (0.31 * t, 0.97 * t), (0.0, 0.5 * t)]
    pairs.extend(
        (float(b) - 0.03 * t, float(b) + 0.05 * t)
        for b in np.asarray(rate.breakpoints)[:4]
    )
    return [(s, e) for s, e in pairs if 0.0 <= s < e <= t]


def _quad_integral(rate: RateFunction, s: float, e: float) -> float:
    pts = [float(b) for b in np.asarray(rate.breakpoints) if s < float(b) < e]
    # Rates singular at an epoch start (Weibull shape < 1) are integrable but
    # make quad warn at this tolerance; the caller's comparison with the exact
    # integral is the real check.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        value, _ = quad(
            lambda x: float(_CALL(rate, jnp.float64(x))),
            s,
            e,
            points=pts or None,
            epsabs=1e-13,
            epsrel=1e-13,
            limit=1000,
        )
    return value


def _replace_leaf(rate: RateFunction, index: int, leaf: jax.Array) -> RateFunction:
    leaves, treedef = jax.tree_util.tree_flatten(rate)
    leaves[index] = leaf
    return jax.tree_util.tree_unflatten(treedef, leaves)


class TestRateFunctionProtocol:
    @pytest.mark.parametrize("rate", _RATES, ids=_IDS)
    def test_structure(self, rate: RateFunction) -> None:
        name = type(rate).__name__
        assert isinstance(rate, RateFunction), f"{name} is not a RateFunction"
        t = rate.end_time
        assert np.ndim(t) == 0, f"{name}: end_time not scalar, shape {np.shape(t)}"
        assert float(t) > 0, f"{name}: end_time={float(t)} not positive"
        bp = np.asarray(rate.breakpoints)
        assert bp.ndim == 1, f"{name}: breakpoints shape {bp.shape} not 1-D"
        assert np.all(np.diff(bp) > 0), f"{name}: breakpoints {bp} not increasing"
        assert np.all((bp > 0) & (bp < float(t))), (
            f"{name}: breakpoints {bp} not strictly inside (0, {float(t)})"
        )

    @pytest.mark.parametrize("rate", _RATES, ids=_IDS)
    def test_values_finite_and_non_negative(self, rate: RateFunction) -> None:
        grid = jnp.linspace(0.0, _end(rate), 2001)
        values = np.asarray(rate(grid))
        assert np.all(np.isfinite(values)), f"{type(rate).__name__}: non-finite values"
        assert values.min() >= -1e-12, (
            f"{type(rate).__name__}: min value {values.min()} is negative"
        )

    @pytest.mark.parametrize("rate", _RATES, ids=_IDS)
    def test_integral_matches_scipy_quad(self, rate: RateFunction) -> None:
        for s, e in _pairs(rate):
            exact = float(rate.integral(jnp.float64(s), jnp.float64(e)))
            numeric = _quad_integral(rate, s, e)
            assert exact == pytest.approx(numeric, rel=1e-10, abs=1e-12), (
                f"{type(rate).__name__} on [{s}, {e}]: integral {exact!r} vs quad {numeric!r}"
            )

    @pytest.mark.parametrize("rate", _RATES, ids=_IDS)
    def test_integral_additive_zero_and_monotone(self, rate: RateFunction) -> None:
        t = _end(rate)
        name = type(rate).__name__
        for s, e in _pairs(rate):
            m = 0.5 * (s + e)
            total = float(rate.integral(jnp.float64(s), jnp.float64(e)))
            parts = float(rate.integral(jnp.float64(s), jnp.float64(m))) + float(
                rate.integral(jnp.float64(m), jnp.float64(e))
            )
            assert parts == pytest.approx(total, rel=1e-12, abs=1e-13), (
                f"{name}: additivity on [{s}, {m}, {e}]: {parts!r} vs {total!r}"
            )
            assert float(rate.integral(jnp.float64(s), jnp.float64(s))) == 0.0, (
                f"{name}: integral({s}, {s}) is not zero"
            )
        grid = jnp.linspace(0.0, t, 500)
        cumulative = np.asarray(rate.integral(jnp.zeros_like(grid), grid))
        assert np.all(np.diff(cumulative) >= -1e-13), (
            f"{name}: integral(0, e) decreases, worst step {np.diff(cumulative).min()}"
        )

    @pytest.mark.parametrize("shape", [(), (7,), (2, 3), (2, 1, 4)])
    @pytest.mark.parametrize("rate", _RATES, ids=_IDS)
    def test_arrays_of_any_shape(
        self, rate: RateFunction, shape: tuple[int, ...]
    ) -> None:
        t = _end(rate)
        n = int(np.prod(shape))
        e = jnp.linspace(0.2 * t, t, n).reshape(shape)
        s = 0.5 * e
        name = type(rate).__name__
        assert rate(e).shape == shape, f"{name}: call shape {rate(e).shape} != {shape}"
        assert rate.integral(s, e).shape == shape, (
            f"{name}: integral shape {rate.integral(s, e).shape} != {shape}"
        )
        flat = np.asarray(rate.integral(s.ravel(), e.ravel())).reshape(shape)
        np.testing.assert_allclose(np.asarray(rate.integral(s, e)), flat, rtol=1e-13)

    @pytest.mark.parametrize("rate", _RATES, ids=_IDS)
    def test_pytree_round_trip(self, rate: RateFunction) -> None:
        leaves, treedef = jax.tree_util.tree_flatten(rate)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        grid = jnp.linspace(0.0, _end(rate), 101)
        np.testing.assert_array_equal(
            np.asarray(rebuilt(grid)),
            np.asarray(rate(grid)),
            err_msg=f"{type(rate).__name__} pytree round trip changed values",
        )
        assert type(rebuilt) is type(rate)

    @pytest.mark.parametrize("rate", _RATES, ids=_IDS)
    def test_jit_matches_eager(self, rate: RateFunction) -> None:
        t = _end(rate)

        def total(r: RateFunction) -> jax.Array:
            return r.integral(jnp.float64(0.0), jnp.float64(t))

        eager, jitted = float(total(rate)), float(jax.jit(total)(rate))
        assert jitted == pytest.approx(eager, rel=1e-13), (
            f"{type(rate).__name__}: jit {jitted!r} vs eager {eager!r}"
        )

    @pytest.mark.parametrize("rate", _RATES, ids=_IDS)
    def test_gradient_is_finite(self, rate: RateFunction) -> None:
        t = _end(rate)
        leaves = jax.tree_util.tree_leaves(rate)
        assert leaves, f"{type(rate).__name__} has no array parameters"
        for i, leaf in enumerate(leaves):

            def total(x: jax.Array, i: int = i) -> jax.Array:
                return _replace_leaf(rate, i, x).integral(
                    jnp.float64(0.0), jnp.float64(t)
                )

            grad = np.asarray(jax.grad(total)(jnp.asarray(leaf, dtype=jnp.float64)))
            assert np.all(np.isfinite(grad)), (
                f"{type(rate).__name__}: gradient wrt leaf {i} is {grad}"
            )
