"""Test general-rate age densities against closed forms, the piecewise-constant path and Wolfram."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from bdpinf.models.math_age_densities import (
    log_nested_w_general,
    non_spine_age_cdf,
    non_spine_age_log_density,
    random_individual_cdf,
    spine_age_cdf,
    spine_age_log_atom,
    spine_age_log_density,
)
from bdpinf.models.math_nested import (
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.models.rate_functions import (
    PiecewiseConstantRate,
    PiecewiseExponentialRate,
    PiecewiseLinearRate,
    PiecewiseQuadraticRate,
    RateFunction,
    SinusoidalRate,
)

_T = 2.5
_AGES = jnp.linspace(0.05, _T - 0.05, 41)


def _linear_both(
    lam_a: float, lam_b: float, mu_a: float, mu_b: float
) -> tuple[PiecewiseLinearRate, PiecewiseLinearRate]:
    times = jnp.array([_T])
    return (
        PiecewiseLinearRate.continuous(jnp.array([lam_a, lam_a + lam_b * _T]), times),
        PiecewiseLinearRate.continuous(jnp.array([mu_a, mu_a + mu_b * _T]), times),
    )


def _piecewise_linear_both() -> tuple[PiecewiseLinearRate, PiecewiseLinearRate]:
    tau = 0.9
    times = jnp.array([tau, _T])
    lam_v, s1, s2, mu_v, r1, r2 = 1.2, 0.3, -0.2, 0.7, 0.15, -0.1
    lam = PiecewiseLinearRate(
        jnp.array([lam_v - s1 * tau, lam_v]),
        jnp.array([lam_v, lam_v + s2 * (_T - tau)]),
        times,
    )
    mu = PiecewiseLinearRate(
        jnp.array([mu_v - r1 * tau, mu_v]),
        jnp.array([mu_v, mu_v + r2 * (_T - tau)]),
        times,
    )
    return lam, mu


def _jump_rates() -> tuple[PiecewiseLinearRate, PiecewiseLinearRate]:
    times = jnp.array([1.0, _T])
    lam = PiecewiseLinearRate(jnp.array([0.5, 2.0]), jnp.array([1.5, 0.4]), times)
    mu = PiecewiseLinearRate(jnp.array([1.0, 0.3]), jnp.array([0.2, 0.9]), times)
    return lam, mu


class TestConstantRates:
    @pytest.mark.parametrize(("lam", "mu"), [(1.3, 0.7), (0.6, 1.1)])
    def test_matches_corollaries(self, lam: float, mu: float) -> None:
        b = PiecewiseConstantRate.constant(lam, _T)
        d = PiecewiseConstantRate.constant(mu, _T)
        a = np.asarray(_AGES)
        non_spine = (lam * np.exp(lam * (_T - a)) - mu * np.exp(mu * (_T - a))) / (
            np.exp(lam * _T) - np.exp(mu * _T)
        )
        spine = lam * mu * (np.exp(-mu * a) - np.exp(-lam * a)) / (lam - mu)
        atom = (lam * np.exp(-mu * _T) - mu * np.exp(-lam * _T)) / (lam - mu)
        np.testing.assert_allclose(
            np.exp(non_spine_age_log_density(_AGES, b, d)), non_spine, rtol=1e-11
        )
        np.testing.assert_allclose(
            np.exp(spine_age_log_density(_AGES, b, d)), spine, rtol=1e-11
        )
        assert float(jnp.exp(spine_age_log_atom(b, d))) == pytest.approx(
            atom, rel=1e-11
        )

    def test_exact_critical_case(self) -> None:
        lam = 1.0
        b = PiecewiseConstantRate.constant(lam, _T)
        a = np.asarray(_AGES)
        np.testing.assert_allclose(
            np.exp(non_spine_age_log_density(_AGES, b, b)),
            np.exp(-lam * a) * (1 + lam * (_T - a)) / _T,
            rtol=1e-11,
        )
        np.testing.assert_allclose(
            np.exp(spine_age_log_density(_AGES, b, b)),
            lam**2 * a * np.exp(-lam * a),
            rtol=1e-11,
        )
        assert float(jnp.exp(spine_age_log_atom(b, b))) == pytest.approx(
            (1 + lam * _T) * np.exp(-lam * _T), rel=1e-11
        )


class TestArrayWrappersAgainstQuadrature:
    """The array-in ``calculate_*`` wrappers share the exact backend; check it independently."""

    def test_multi_epoch_wrappers_match_quadrature_oracle(self) -> None:
        times = jnp.array([0.8, 1.7, _T])
        b_vals = jnp.array([0.9, 1.6, 0.7])
        d_vals = jnp.array([1.2, 0.5, 1.0])
        b = PiecewiseConstantRate(b_vals, times)
        d = PiecewiseConstantRate(d_vals, times)
        np.testing.assert_allclose(
            non_spine_log_density(nested_terms(b_vals, d_vals, times), _AGES),
            non_spine_age_log_density(_AGES, b, d, method="quadrature", n_nodes=48),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            spine_log_density(nested_terms(b_vals, d_vals, times), _AGES),
            spine_age_log_density(_AGES, b, d, method="quadrature", n_nodes=48),
            atol=1e-10,
        )

    def test_atom_normalises_multi_epoch(self) -> None:
        # The atom plus the continuous part must sum to one in every epoch layout.
        times = jnp.array([0.8, 1.7, _T])
        b = PiecewiseConstantRate(jnp.array([0.9, 1.6, 0.7]), times)
        d = PiecewiseConstantRate(jnp.array([1.2, 0.5, 1.0]), times)
        grid = jnp.linspace(0.0, _T, 200001)[1:-1]
        interior = np.trapezoid(
            np.exp(np.asarray(spine_age_log_density(grid, b, d))), np.asarray(grid)
        )
        atom = float(jnp.exp(spine_age_log_atom(b, d)))
        assert interior + atom == pytest.approx(1.0, abs=1e-5)


@pytest.mark.regression
class TestWolframGoldenValues:
    """Values from Paper01_AgeBDPVarEnv/Wolfram/AgeDist_{LinearBoth,PiecewiseLinearBoth}.nb."""

    @pytest.mark.parametrize(
        ("params", "age", "non_spine", "spine"),
        [
            ((1.2, 0.2, 0.7, 0.1), 0.9, 0.35126706726992174, 0.40656350086630205),
            ((1.2, -0.2, 0.7, 0.1), 0.9, 0.4751603899370419, 0.32802710517641637),
            ((1.2, 0.2, 0.7, -0.15), 0.9, 0.36397983001209344, 0.24167170327933266),
        ],
    )
    def test_linear_both(
        self,
        params: tuple[float, float, float, float],
        age: float,
        non_spine: float,
        spine: float,
    ) -> None:
        b, d = _linear_both(*params)
        a = jnp.array([age])
        assert float(jnp.exp(non_spine_age_log_density(a, b, d))[0]) == pytest.approx(
            non_spine, abs=1e-9
        )
        assert float(jnp.exp(spine_age_log_density(a, b, d))[0]) == pytest.approx(
            spine, abs=1e-9
        )

    def test_linear_both_nested_w(self) -> None:
        b, d = _linear_both(1.2, 0.2, 0.7, 0.1)
        w = float(jnp.exp(log_nested_w_general(b, d, jnp.array(2.0))))
        assert w == pytest.approx(6.439292686758441, rel=1e-10)

    def test_piecewise_linear_both(self) -> None:
        b, d = _piecewise_linear_both()
        a = jnp.array([1.2])
        assert float(jnp.exp(non_spine_age_log_density(a, b, d))[0]) == pytest.approx(
            0.3430933066148787, abs=1e-9
        )
        assert float(jnp.exp(spine_age_log_density(a, b, d))[0]) == pytest.approx(
            0.31050420949098767, abs=1e-9
        )
        w = float(jnp.exp(log_nested_w_general(b, d, jnp.array(_T))))
        assert w == pytest.approx(5.661968424654191, rel=1e-10)


class TestNormalisationAndShape:
    @pytest.mark.parametrize(
        "rates",
        [_piecewise_linear_both(), _jump_rates(), _linear_both(1.2, -0.2, 0.7, 0.1)],
        ids=["kinked", "jumps", "linear"],
    )
    def test_cdfs_reach_one(
        self, rates: tuple[PiecewiseLinearRate, PiecewiseLinearRate]
    ) -> None:
        b, d = rates
        end = jnp.array([_T])
        assert float(non_spine_age_cdf(end, b, d)[0]) == pytest.approx(1.0, abs=1e-10)
        assert float(spine_age_cdf(end, b, d)[0]) == pytest.approx(1.0, abs=1e-10)
        assert float(random_individual_cdf(end, b, d)[0]) == pytest.approx(
            1.0, abs=1e-10
        )

    def test_cdfs_monotone_and_start_at_zero(self) -> None:
        b, d = _jump_rates()
        grid = jnp.linspace(0.0, _T, 60)
        for cdf in (non_spine_age_cdf, spine_age_cdf, random_individual_cdf):
            values = np.asarray(cdf(grid, b, d))
            assert values[0] == pytest.approx(0.0, abs=1e-12)
            assert np.all(np.diff(values) >= -1e-12)

    def test_zero_slope_linear_equals_constant(self) -> None:
        times = jnp.array([1.0, _T])
        vals_b, vals_d = jnp.array([0.9, 1.4]), jnp.array([1.1, 0.6])
        pc = (
            PiecewiseConstantRate(vals_b, times),
            PiecewiseConstantRate(vals_d, times),
        )
        pl = (
            PiecewiseLinearRate.from_constant(vals_b, times),
            PiecewiseLinearRate.from_constant(vals_d, times),
        )
        np.testing.assert_allclose(
            spine_age_log_density(_AGES, *pl), spine_age_log_density(_AGES, *pc)
        )

    def test_quadrature_converged(self) -> None:
        b, d = _jump_rates()
        coarse = spine_age_log_density(_AGES, b, d, n_nodes=16)
        fine = spine_age_log_density(_AGES, b, d, n_nodes=64)
        np.testing.assert_allclose(coarse, fine, atol=1e-10)

    def test_pure_birth_has_no_nonspine_death_term(self) -> None:
        b = PiecewiseLinearRate.continuous(jnp.array([1.0, 2.0]), jnp.array([_T]))
        d = PiecewiseConstantRate.constant(0.0, _T)
        values = np.asarray(spine_age_log_density(_AGES, b, d))
        assert np.all(values == -np.inf)

    def test_gradient_wrt_knots_is_finite(self) -> None:
        d = PiecewiseConstantRate.constant(0.8, _T)

        def loglik(knots: jax.Array) -> jax.Array:
            b = PiecewiseLinearRate.continuous(knots, jnp.array([_T]))
            return spine_age_log_density(_AGES, b, d).sum()

        grad = jax.grad(loglik)(jnp.array([1.0, 1.5]))
        assert np.all(np.isfinite(np.asarray(grad)))


_TWO_EPOCHS = jnp.array([1.1, _T])
_CONST_DEATH = PiecewiseConstantRate.constant(0.8, _T)

# (birth, death) pairs for the new rate families; the last has more than six
# periods of the faster rate inside [0, T].
_NEW_SCENARIOS: dict[str, tuple[RateFunction, RateFunction]] = {
    "exponential-birth-one-epoch": (
        PiecewiseExponentialRate.continuous(jnp.array([0.6, 1.9]), jnp.array([_T])),
        _CONST_DEATH,
    ),
    "exponential-birth-two-epochs": (
        PiecewiseExponentialRate.continuous(jnp.array([0.6, 1.9, 0.9]), _TWO_EPOCHS),
        _CONST_DEATH,
    ),
    "quadratic-birth-one-epoch": (
        PiecewiseQuadraticRate(
            jnp.array([0.5]), jnp.array([1.9]), jnp.array([0.7]), jnp.array([_T])
        ),
        _CONST_DEATH,
    ),
    "quadratic-spline-birth-two-epochs": (
        PiecewiseQuadraticRate.spline(jnp.array([0.7, 1.7, 1.0]), 0.8, _TWO_EPOCHS),
        _CONST_DEATH,
    ),
    "sinusoidal-birth": (SinusoidalRate(1.3, 0.9, 1.4, 0.4, end_time=_T), _CONST_DEATH),
    "sinusoidal-death": (
        PiecewiseConstantRate.constant(1.1, _T),
        SinusoidalRate(0.9, 0.7, 1.2, 1.0, end_time=_T),
    ),
    "sinusoidal-both-different-periods": (
        SinusoidalRate(1.4, 1.0, 0.9, end_time=_T),
        SinusoidalRate(1.0, 0.6, 1.7, 2.0, end_time=_T),
    ),
    "sinusoidal-both-seven-periods": (
        SinusoidalRate(1.3, 0.8, 0.35, 0.5, end_time=_T),
        SinusoidalRate(1.0, 0.5, 1.7, 1.0, end_time=_T),
    ),
}
_NEW_IDS = list(_NEW_SCENARIOS)
_NEW_PAIRS = list(_NEW_SCENARIOS.values())
_ORACLE_AGES = np.array([0.15, 0.7, 1.45, 2.3])


def _quad(
    f: Callable[[float], float], lo: float, hi: float, points: list[float]
) -> float:
    value, _ = quad(
        f,
        lo,
        hi,
        points=[p for p in points if lo < p < hi] or None,
        epsabs=1e-14,
        epsrel=1e-13,
        limit=2000,
    )
    return value


def _oracle(birth: RateFunction, death: RateFunction) -> tuple[object, float]:
    """Return ``(density, atom)`` from the spine-age formulas by independent ``quad``."""
    end = float(birth.end_time)
    points = sorted(
        float(b) for b in np.concatenate([birth.breakpoints, death.breakpoints])
    )

    @jax.jit
    def integrand(t: jax.Array) -> jax.Array:
        gamma = birth.integral(t, end) - death.integral(t, end)
        return death(t) * jnp.exp(gamma)

    @jax.jit
    def prefactor(s: jax.Array) -> jax.Array:
        return birth(s) * jnp.exp(-birth.integral(s, end))

    def l_mu(s: float) -> float:
        return _quad(lambda t: float(integrand(jnp.float64(t))), s, end, points)

    def density(age: float) -> float:
        s = end - age
        return float(prefactor(jnp.float64(s))) * l_mu(s)

    zero = jnp.float64(0.0)
    gamma_total = float(birth.integral(zero, end) - death.integral(zero, end))
    w_total = np.exp(gamma_total) + l_mu(0.0)
    atom = w_total * np.exp(-float(birth.integral(zero, end)))
    return density, atom


class TestNewRateFamilies:
    @pytest.mark.parametrize(("birth", "death"), _NEW_PAIRS, ids=_NEW_IDS)
    def test_total_mass_is_one(self, birth: RateFunction, death: RateFunction) -> None:
        end = jnp.array([_T])
        assert float(spine_age_cdf(end, birth, death)[0]) == pytest.approx(
            1.0, abs=1e-8
        )
        assert float(non_spine_age_cdf(end, birth, death)[0]) == pytest.approx(
            1.0, abs=1e-8
        )

    @pytest.mark.parametrize(("birth", "death"), _NEW_PAIRS, ids=_NEW_IDS)
    def test_spine_density_and_atom_match_quadrature_oracle(
        self, birth: RateFunction, death: RateFunction
    ) -> None:
        density, atom = _oracle(birth, death)
        got = np.exp(
            np.asarray(spine_age_log_density(jnp.asarray(_ORACLE_AGES), birth, death))
        )
        expected = np.array([density(float(a)) for a in _ORACLE_AGES])
        np.testing.assert_allclose(got, expected, rtol=1e-8)
        assert float(jnp.exp(spine_age_log_atom(birth, death))) == pytest.approx(
            atom, rel=1e-8
        )

    @pytest.mark.parametrize(("birth", "death"), _NEW_PAIRS, ids=_NEW_IDS)
    def test_default_nodes_agree_with_64_nodes(
        self, birth: RateFunction, death: RateFunction
    ) -> None:
        for log_density in (spine_age_log_density, non_spine_age_log_density):
            default = np.exp(np.asarray(log_density(_AGES, birth, death)))
            fine = np.exp(np.asarray(log_density(_AGES, birth, death, n_nodes=64)))
            np.testing.assert_allclose(default, fine, rtol=1e-9)
        atom = float(jnp.exp(spine_age_log_atom(birth, death)))
        atom_fine = float(jnp.exp(spine_age_log_atom(birth, death, n_nodes=64)))
        assert atom == pytest.approx(atom_fine, rel=1e-9)
