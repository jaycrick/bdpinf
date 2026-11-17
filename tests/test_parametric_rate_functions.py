"""Test the parametric (power, log-logistic, Lomax) and piecewise-composed rates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from bdpinf.models.math_age_densities import (
    non_spine_age_cdf,
    spine_age_cdf,
    spine_age_log_atom,
)
from bdpinf.models.rate_functions import (
    LogLogisticRate,
    LomaxRate,
    PiecewiseExponentialRate,
    PiecewiseLinearRate,
    PiecewiseRate,
    PowerRate,
    RateFunction,
    SinusoidalRate,
)

_T = 5.0


def _scalar(x: jax.Array) -> float:
    return float(np.asarray(x))


def _rates() -> list[RateFunction]:
    return [
        PowerRate(0.7, 0.5, end_time=_T),
        PowerRate.weibull(2.5, 3.0, end_time=_T),
        PowerRate.weibull(0.6, 2.0, end_time=_T),
        PowerRate.rayleigh(0.3, end_time=_T),
        LogLogisticRate(2.0, 3.0, end_time=_T),
        LogLogisticRate(0.7, 3.0, end_time=_T),
        LomaxRate(1.5, 2.0, end_time=_T),
        PiecewiseExponentialRate.gompertz(0.2, 0.4, _T),
        PiecewiseExponentialRate.gompertz(0.5, -0.6, _T),
    ]


def _oracle(rate: RateFunction, s: float, e: float) -> float:
    pts = [float(b) for b in np.asarray(rate.breakpoints) if s < b < e]
    return quad(lambda t: _scalar(rate(jnp.array(t))), s, e, points=pts or None)[0]


class TestPowerRate:
    def test_values(self) -> None:
        rate = PowerRate(2.0, 3.0, end_time=_T)
        assert _scalar(rate(jnp.array(1.5))) == pytest.approx(2.0 * 1.5**3)

    def test_weibull_matches_hazard_formula(self) -> None:
        k, eta, t = 1.7, 2.5, 1.3
        rate = PowerRate.weibull(k, eta, end_time=_T)
        expected = (k / eta) * (t / eta) ** (k - 1)
        assert _scalar(rate(jnp.array(t))) == pytest.approx(expected, rel=1e-12)
        cum = (_T / eta) ** k - (t / eta) ** k
        assert _scalar(rate.integral(jnp.array(t), jnp.array(_T))) == pytest.approx(
            cum, rel=1e-12
        )

    def test_rayleigh(self) -> None:
        rate = PowerRate.rayleigh(0.8, end_time=_T)
        assert _scalar(rate(jnp.array(2.0))) == pytest.approx(1.6)
        assert _scalar(rate.integral(jnp.array(0.0), jnp.array(2.0))) == pytest.approx(
            0.8 * 2.0
        )

    def test_breakpoints(self) -> None:
        assert PowerRate(1.0, 0.5, end_time=_T).breakpoints.shape == (0,)
        bp = np.asarray(PowerRate(1.0, -0.4, end_time=_T).breakpoints)
        assert bp.shape == (30,)
        assert np.all(np.diff(bp) > 0)
        assert bp[-1] == pytest.approx(_T / 2)
        assert bp[0] == pytest.approx(_T * 2.0**-30)

    def test_integral_stable_for_close_endpoints(self) -> None:
        rate = PowerRate(1.0, 0.5, end_time=_T)
        s = 2.0
        e = s + 1e-12
        assert _scalar(rate.integral(jnp.array(s), jnp.array(e))) == pytest.approx(
            np.sqrt(s) * 1e-12, rel=1e-6
        )

    def test_integral_from_zero_gradient_finite(self) -> None:
        def f(p: jax.Array) -> jax.Array:
            return PowerRate(1.0, p, end_time=_T).integral(
                jnp.array(0.0), jnp.array(2.0)
            )

        assert np.isfinite(_scalar(jax.grad(f)(jnp.array(-0.4))))

    @pytest.mark.parametrize(
        ("c", "p", "match"),
        [
            (0.0, 1.0, "c must be positive"),
            (-1.0, 1.0, "c must be positive"),
            (1.0, -1.0, "p must be greater than -1"),
            (1.0, float("nan"), "p must be finite"),
        ],
    )
    def test_validation(self, c: float, p: float, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            PowerRate(c, p, end_time=_T)

    def test_rejects_bad_end_time(self) -> None:
        with pytest.raises(ValueError, match="end_time"):
            PowerRate(1.0, 1.0, end_time=0.0)
        with pytest.raises(ValueError, match="end_time"):
            PowerRate(1.0, 1.0)


class TestLogLogisticRate:
    def test_value(self) -> None:
        k, eta, t = 2.0, 3.0, 1.5
        x = (t / eta) ** k
        expected = (k / eta) * (t / eta) ** (k - 1) / (1 + x)
        rate = LogLogisticRate(k, eta, end_time=_T)
        assert _scalar(rate(jnp.array(t))) == pytest.approx(expected, rel=1e-12)

    def test_integral_is_cumulative_hazard(self) -> None:
        rate = LogLogisticRate(1.8, 2.0, end_time=_T)
        got = _scalar(rate.integral(jnp.array(0.0), jnp.array(3.0)))
        assert got == pytest.approx(np.log1p((3.0 / 2.0) ** 1.8), rel=1e-12)

    @pytest.mark.parametrize("shape", [0.4, 0.7, 1.0, 1.5])
    def test_integral_from_zero_gradient_matches_closed_form(
        self, shape: float
    ) -> None:
        # The cumulative hazard is log1p((e / scale)**shape); for shape < 1 the
        # power has an infinite slope at 0, which once gave a NaN gradient.
        e, scale = 2.5, 3.0

        def cumulative(params: jax.Array) -> jax.Array:
            return LogLogisticRate(params[0], params[1], end_time=e).integral(
                jnp.array(0.0), jnp.array(e)
            )

        got = np.asarray(jax.grad(cumulative)(jnp.array([shape, scale])))
        x = (e / scale) ** shape
        expected = np.array(
            [x * np.log(e / scale) / (1 + x), -shape * x / (scale * (1 + x))]
        )
        assert np.all(np.isfinite(got)), f"gradient {got} is not finite"
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_breakpoints_only_when_singular(self) -> None:
        assert LogLogisticRate(1.0, 2.0, end_time=_T).breakpoints.shape == (0,)
        assert LogLogisticRate(0.5, 2.0, end_time=_T).breakpoints.shape == (30,)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="shape must be positive"):
            LogLogisticRate(0.0, 1.0, end_time=_T)
        with pytest.raises(ValueError, match="scale must be positive"):
            LogLogisticRate(1.0, -1.0, end_time=_T)


class TestLomaxRate:
    def test_value_and_integral(self) -> None:
        rate = LomaxRate(1.5, 2.0, end_time=_T)
        assert _scalar(rate(jnp.array(1.0))) == pytest.approx(0.5)
        assert _scalar(rate.integral(jnp.array(1.0), jnp.array(4.0))) == pytest.approx(
            1.5 * np.log(6.0 / 3.0), rel=1e-12
        )
        assert rate.breakpoints.shape == (0,)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="shape must be positive"):
            LomaxRate(-1.0, 1.0, end_time=_T)
        with pytest.raises(ValueError, match="scale must be positive"):
            LomaxRate(1.0, float("inf"), end_time=_T)


class TestGompertz:
    def test_single_epoch_values(self) -> None:
        a, b = 0.2, 0.4
        rate = PiecewiseExponentialRate.gompertz(a, b, _T)
        assert float(rate.end_time) == _T
        assert _scalar(rate(jnp.array(2.0))) == pytest.approx(a * np.exp(b * 2.0))
        assert _scalar(rate.integral(jnp.array(0.0), jnp.array(_T))) == pytest.approx(
            a * np.expm1(b * _T) / b, rel=1e-12
        )

    def test_b_zero_is_constant(self) -> None:
        rate = PiecewiseExponentialRate.gompertz(0.3, 0.0, _T)
        assert _scalar(rate.integral(jnp.array(1.0), jnp.array(3.0))) == pytest.approx(
            0.6
        )


class TestAgainstQuad:
    @pytest.mark.parametrize("rate", _rates())
    def test_protocol(self, rate: RateFunction) -> None:
        assert isinstance(rate, RateFunction)

    @pytest.mark.parametrize("rate", _rates())
    def test_integral_matches_quad(self, rate: RateFunction) -> None:
        for s, e in [(0.5, 3.0), (1.0, _T), (0.01, 0.2)]:
            got = _scalar(rate.integral(jnp.array(s), jnp.array(e)))
            assert got == pytest.approx(_oracle(rate, s, e), rel=1e-9)

    @pytest.mark.parametrize("rate", _rates())
    def test_integral_from_zero_is_additive(self, rate: RateFunction) -> None:
        whole = _scalar(rate.integral(jnp.array(0.0), jnp.array(_T)))
        parts = _scalar(rate.integral(jnp.array(0.0), jnp.array(1.3))) + _scalar(
            rate.integral(jnp.array(1.3), jnp.array(_T))
        )
        assert whole == pytest.approx(parts, rel=1e-12)

    def test_singular_integral_from_zero_matches_quad(self) -> None:
        rate = PowerRate.weibull(0.6, 2.0, end_time=_T)
        expected = quad(lambda t: _scalar(rate(jnp.array(t))), 0.0, 2.0)[0]
        got = _scalar(rate.integral(jnp.array(0.0), jnp.array(2.0)))
        assert got == pytest.approx(expected, rel=1e-6)
        assert got == pytest.approx((2.0 / 2.0) ** 0.6, rel=1e-12)


class TestJitAndGrad:
    def test_jit_runs_for_each_family(self) -> None:
        @jax.jit
        def total(rate: RateFunction) -> jax.Array:
            return rate.integral(jnp.array(0.5), jnp.array(3.0))

        for rate in _rates():
            assert _scalar(total(rate)) == pytest.approx(
                _scalar(rate.integral(jnp.array(0.5), jnp.array(3.0))), rel=1e-12
            )

    @pytest.mark.parametrize(
        "build",
        [
            lambda x: PowerRate(x, 0.5, end_time=_T),
            lambda x: PowerRate(0.7, x, end_time=_T),
            lambda x: PowerRate.weibull(x, 2.0, end_time=_T),
            lambda x: LogLogisticRate(x, 2.0, end_time=_T),
            lambda x: LogLogisticRate(1.5, x, end_time=_T),
            lambda x: LomaxRate(x, 2.0, end_time=_T),
            lambda x: LomaxRate(1.5, x, end_time=_T),
            lambda x: PiecewiseExponentialRate.gompertz(x, 0.3, _T),
            lambda x: PiecewiseExponentialRate.gompertz(0.3, x, _T),
        ],
    )
    def test_grad_matches_finite_differences(self, build) -> None:
        def f(x: jax.Array) -> jax.Array:
            return build(x).integral(jnp.array(0.5), jnp.array(3.0))

        x0 = jnp.array(0.9)
        h = 1e-6
        fd = (_scalar(f(x0 + h)) - _scalar(f(x0 - h))) / (2 * h)
        g = _scalar(jax.grad(f)(x0))
        assert np.isfinite(g)
        assert g == pytest.approx(fd, rel=1e-6)

    def test_grad_through_pytree_argument(self) -> None:
        rate = PowerRate(0.7, 0.5, end_time=_T)
        grads = jax.grad(lambda r: r.integral(jnp.array(0.5), jnp.array(3.0)))(rate)
        assert np.isfinite(_scalar(grads.c))
        assert np.isfinite(_scalar(grads.p))


class TestPiecewiseRate:
    _TIMES = jnp.array([1.0, 2.5, 4.0])

    def _linear_pair(self) -> tuple[PiecewiseRate, PiecewiseLinearRate]:
        starts = jnp.array([0.5, 1.5, 0.2])
        ends = jnp.array([1.0, 0.4, 0.9])
        glued = PiecewiseRate(
            tuple(
                PiecewiseLinearRate(starts[k : k + 1], ends[k : k + 1], jnp.array([d]))
                for k, d in enumerate([1.0, 1.5, 1.5])
            ),
            self._TIMES,
        )
        return glued, PiecewiseLinearRate(starts, ends, self._TIMES)

    def test_matches_piecewise_linear(self) -> None:
        glued, linear = self._linear_pair()
        ts = jnp.array([0.0, 0.3, 1.0, 1.2, 2.5, 3.1, 4.0])
        np.testing.assert_allclose(glued(ts), linear(ts), rtol=1e-12)
        s, e = jnp.array([0.0, 0.5, 1.0, 2.0]), jnp.array([4.0, 3.9, 1.0, 2.7])
        np.testing.assert_allclose(
            glued.integral(s, e), linear.integral(s, e), rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(glued.breakpoints, linear.breakpoints)
        assert float(glued.end_time) == 4.0
        assert isinstance(glued, RateFunction)

    def test_local_time(self) -> None:
        pieces = (
            PowerRate.rayleigh(1.0, end_time=1.0),
            PowerRate.rayleigh(1.0, end_time=2.0),
        )
        rate = PiecewiseRate(pieces, jnp.array([1.0, 3.0]))
        assert _scalar(rate(jnp.array(2.0))) == pytest.approx(1.0)
        assert _scalar(rate(jnp.array(0.5))) == pytest.approx(0.5)
        assert _scalar(rate.integral(jnp.array(0.0), jnp.array(3.0))) == pytest.approx(
            0.5 + 2.0
        )

    def test_breakpoints_merge_and_sort(self) -> None:
        pieces = (
            LomaxRate(1.0, 1.0, end_time=1.0),
            PowerRate.weibull(0.5, 1.0, end_time=2.0),
        )
        bp = np.asarray(PiecewiseRate(pieces, jnp.array([1.0, 3.0])).breakpoints)
        assert bp.shape == (31,)
        assert np.all(np.diff(bp) > 0)
        assert 1.0 in bp
        assert bp[1] == pytest.approx(1.0 + 2.0 * 2.0**-30)

    def test_heterogeneous_pieces_match_quad(self) -> None:
        pieces = (
            PiecewiseExponentialRate.gompertz(0.3, 0.5, 1.5),
            PowerRate.weibull(0.7, 2.0, end_time=2.0),
            SinusoidalRate(1.0, 0.5, 1.0, end_time=1.0),
        )
        rate = PiecewiseRate(pieces, jnp.array([1.5, 3.5, 4.5]))
        for s, e in [(0.4, 4.2), (1.6, 3.4)]:
            pts = [b for b in (1.5, 3.5) if s < b < e]
            expected = quad(
                lambda t: _scalar(rate(jnp.array(t))), s, e, points=pts, limit=200
            )[0]
            got = _scalar(rate.integral(jnp.array(s), jnp.array(e)))
            assert got == pytest.approx(expected, rel=1e-8)

    def test_jit_and_grad(self) -> None:
        def f(x: jax.Array) -> jax.Array:
            pieces = (
                PowerRate.weibull(x, 1.0, end_time=1.0),
                LomaxRate(1.5, x, end_time=2.0),
            )
            rate = PiecewiseRate(pieces, jnp.array([1.0, 3.0]))
            return rate.integral(jnp.array(0.2), jnp.array(2.6))

        x0 = jnp.array(1.4)
        h = 1e-6
        fd = (_scalar(f(x0 + h)) - _scalar(f(x0 - h))) / (2 * h)
        g = _scalar(jax.jit(jax.grad(f))(x0))
        assert np.isfinite(g)
        assert g == pytest.approx(fd, rel=1e-6)

    def test_rejects_end_time_mismatch(self) -> None:
        pieces = (LomaxRate(1.0, 1.0, end_time=1.0), LomaxRate(1.0, 1.0, end_time=1.0))
        with pytest.raises(ValueError, match="piece 1 has end_time"):
            PiecewiseRate(pieces, jnp.array([1.0, 3.0]))

    def test_rejects_wrong_piece_count(self) -> None:
        pieces = (LomaxRate(1.0, 1.0, end_time=1.0),)
        with pytest.raises(ValueError, match="1 pieces for 2 epochs"):
            PiecewiseRate(pieces, jnp.array([1.0, 3.0]))

    def test_rejects_bad_epochs(self) -> None:
        pieces = (LomaxRate(1.0, 1.0, end_time=2.0),)
        with pytest.raises(ValueError, match="strictly increasing"):
            PiecewiseRate(pieces, jnp.array([-1.0]))


class TestAgeDensityNormalisation:
    def test_spine_and_non_spine_integrate_to_one(self) -> None:
        end = 6.0
        birth = PowerRate.weibull(0.6, 2.0, end_time=end)
        death = LomaxRate(1.5, 4.0, end_time=end)
        t = jnp.array(end)
        spine_total = _scalar(spine_age_cdf(t, birth, death))
        non_spine_total = _scalar(non_spine_age_cdf(t, birth, death))
        atom = float(np.exp(_scalar(spine_age_log_atom(birth, death))))
        before_atom = _scalar(spine_age_cdf(t - 1e-9, birth, death))
        print(
            f"spine err={abs(spine_total - 1):.3e} "
            f"non-spine err={abs(non_spine_total - 1):.3e} "
            f"continuous+atom err={abs(before_atom + atom - 1):.3e}"
        )
        assert 0.0 < atom < 1.0
        assert spine_total == pytest.approx(1.0, abs=1e-6)
        assert non_spine_total == pytest.approx(1.0, abs=1e-6)
        assert before_atom + atom == pytest.approx(1.0, abs=1e-6)
