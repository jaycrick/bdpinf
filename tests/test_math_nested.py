"""Test the exact epoch-wise nested integral W and the age densities built on it."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.models import math_nested as mn
from bdpinf.models.math_age_densities import (
    age_log_densities_general,
    non_spine_age_log_density,
    spine_age_log_atom,
    spine_age_log_density,
)
from bdpinf.models.math_extinction import calculate_extinction_pr
from bdpinf.models.math_populations import (
    calculate_populations_log_lik,
    calculate_populations_log_pmf,
)
from bdpinf.models.rate_functions import PiecewiseConstantRate

_TIMES = [0.8, 1.7, 2.5]
_BIRTH = [0.9, 1.6, 0.7]
_DEATH = [1.2, 0.5, 1.0]

_PARAM_SETS = [
    ([2.5], [1.3], [0.7]),
    ([2.5], [0.6], [1.1]),
    ([3.0], [1.0], [1.0]),
    (_TIMES, _BIRTH, _DEATH),
    ([1.0, 2.0, 4.0], [0.4, 1.9, 0.3], [0.4, 0.2, 1.5]),  # critical first epoch
    ([0.5, 1.0, 1.5, 2.0, 5.0], [2.0, 0.1, 1.2, 0.9, 0.05], [0.3, 1.4, 1.2, 0.2, 0.9]),
]


def _arrays(
    times: list[float], birth: list[float], death: list[float]
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return jnp.array(birth), jnp.array(death), jnp.array(times)


def _terms(
    times: list[float], birth: list[float], death: list[float]
) -> mn.NestedTerms:
    return mn.nested_terms(*_arrays(times, birth, death))


def _w_restart(
    times: list[float], birth: list[float], death: list[float], b: float
) -> float:
    """Oracle: the Wolfram restart recurrence W_k = e^{gd} W_{k-1} + mu expm1(gd)/g."""
    w, start = 1.0, 0.0
    for tau, lam, mu in zip(times, birth, death, strict=True):
        d = min(b, tau) - start
        if d <= 0:
            break
        g = lam - mu
        kernel = d if abs(g * d) < 1e-12 else math.expm1(g * d) / g
        w = math.exp(g * d) * w + mu * kernel
        start = tau
    return w


def _lambda_int(times: list[float], rates: list[float], s: float, e: float) -> float:
    total, start = 0.0, 0.0
    for tau, r in zip(times, rates, strict=True):
        lo, hi = max(s, start), min(e, tau)
        total += r * max(hi - lo, 0.0)
        start = tau
    return total


class TestNestedW:
    @pytest.mark.parametrize(("times", "birth", "death"), _PARAM_SETS)
    def test_matches_restart_recurrence(
        self, times: list[float], birth: list[float], death: list[float]
    ) -> None:
        terms = _terms(times, birth, death)
        grid = np.linspace(0.0, times[-1], 23)
        got = np.exp(np.asarray(mn.log_nested_w(terms, jnp.asarray(grid))))
        want = [_w_restart(times, birth, death, float(b)) for b in grid]
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_constant_rate_closed_forms(self) -> None:
        lam, mu, b = 1.3, 0.7, 2.0
        terms = _terms([2.5], [lam], [mu])
        w = float(jnp.exp(mn.log_nested_w(terms, jnp.array(b))))
        assert w == pytest.approx((lam * math.exp((lam - mu) * b) - mu) / (lam - mu))

    def test_critical_case_is_exact(self) -> None:
        lam, b = 1.0, 2.0
        terms = _terms([3.0], [lam], [lam])
        w = float(jnp.exp(mn.log_nested_w(terms, jnp.array(b))))
        assert w == pytest.approx(1.0 + lam * b, rel=1e-13)

    @pytest.mark.parametrize(("times", "birth", "death"), _PARAM_SETS)
    def test_relations_between_w_n_and_l(
        self, times: list[float], birth: list[float], death: list[float]
    ) -> None:
        terms = _terms(times, birth, death)
        gamma = float(terms.gamma_total)
        w_t = float(jnp.exp(mn.log_nested_w_total(terms)))
        n_t = float(jnp.exp(mn.log_extinction_numerator(terms)))
        assert w_t == pytest.approx(math.exp(gamma) * (1.0 + n_t), rel=1e-12)
        assert w_t - 1.0 == pytest.approx(
            float(jnp.exp(mn.log_nested_w_minus_one(terms))), rel=1e-10
        )

    def test_pi_equals_one_over_one_plus_n_only_when_gamma_is_zero(self) -> None:
        terms = _terms([2.0, 4.0], [1.5, 0.5], [0.5, 1.5])  # Gamma(0,T] = 0
        assert float(terms.gamma_total) == pytest.approx(0.0, abs=1e-14)
        pi_t = float(jnp.exp(-mn.log_nested_w_total(terms)))
        n_t = float(jnp.exp(mn.log_extinction_numerator(terms)))
        assert pi_t == pytest.approx(1.0 / (1.0 + n_t), rel=1e-12)
        skewed = _terms([2.0, 4.0], [1.5, 0.9], [0.5, 1.5])
        pi_s = float(jnp.exp(-mn.log_nested_w_total(skewed)))
        n_s = float(jnp.exp(mn.log_extinction_numerator(skewed)))
        assert pi_s != pytest.approx(1.0 / (1.0 + n_s), rel=1e-3)

    def test_small_horizon_keeps_relative_accuracy(self) -> None:
        lam, mu, t = 0.9, 0.4, 1e-9
        terms = _terms([t], [lam], [mu])
        log_w_minus_one = float(mn.log_nested_w_minus_one(terms))
        assert math.exp(log_w_minus_one) == pytest.approx(lam * t, rel=1e-6)


class TestBoundaryFixes:
    """Regression tests for the multi-epoch ``age == T`` bugs of the time-reversal code."""

    def test_spine_atom_equals_w_times_exp_minus_lambda(self) -> None:
        terms = _terms(_TIMES, _BIRTH, _DEATH)
        w_t = _w_restart(_TIMES, _BIRTH, _DEATH, _TIMES[-1])
        lam_total = _lambda_int(_TIMES, _BIRTH, 0.0, _TIMES[-1])
        atom = float(jnp.exp(mn.spine_log_atom(terms)))
        assert atom == pytest.approx(w_t * math.exp(-lam_total), rel=1e-12)
        assert atom == pytest.approx(0.315414, abs=1e-6)  # legacy returned 0.2057

    @pytest.mark.parametrize(("times", "birth", "death"), _PARAM_SETS)
    def test_density_plus_atom_is_one(
        self, times: list[float], birth: list[float], death: list[float]
    ) -> None:
        terms = _terms(times, birth, death)
        grid = jnp.linspace(0.0, times[-1], 400001)[1:-1]
        interior = np.trapezoid(
            np.exp(np.asarray(mn.spine_log_density(terms, grid))), np.asarray(grid)
        )
        atom = float(jnp.exp(mn.spine_log_atom(terms)))
        assert interior + atom == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize(("times", "birth", "death"), _PARAM_SETS)
    def test_non_spine_density_integrates_to_one(
        self, times: list[float], birth: list[float], death: list[float]
    ) -> None:
        terms = _terms(times, birth, death)
        grid = jnp.linspace(0.0, times[-1], 400001)
        mass = np.trapezoid(
            np.exp(np.asarray(mn.non_spine_log_density(terms, grid))), np.asarray(grid)
        )
        assert mass == pytest.approx(1.0, abs=2e-5)  # density jumps at epoch edges

    def test_atom_is_invariant_to_splitting_an_epoch(self) -> None:
        one = _terms([2.5], [0.9], [1.2])
        two = _terms([1.0, 2.5], [0.9, 0.9], [1.2, 1.2])
        assert float(mn.spine_log_atom(one)) == pytest.approx(
            float(mn.spine_log_atom(two)), rel=1e-13
        )
        ages = jnp.array([0.3, 1.1, 2.0, 2.5])
        np.testing.assert_allclose(
            mn.spine_log_density(one, ages), mn.spine_log_density(two, ages), rtol=1e-12
        )
        np.testing.assert_allclose(
            mn.non_spine_log_density(one, ages),
            mn.non_spine_log_density(two, ages),
            rtol=1e-12,
        )

    def test_non_spine_density_is_continuous_at_t(self) -> None:
        terms = _terms(_TIMES, _BIRTH, _DEATH)
        at_t = float(mn.non_spine_log_density(terms, jnp.array(_TIMES[-1])))
        just_below = float(
            mn.non_spine_log_density(terms, jnp.array(_TIMES[-1] - 1e-9))
        )
        assert at_t == pytest.approx(just_below, abs=1e-7)


class TestAgreementWithQuadrature:
    @pytest.mark.parametrize(("times", "birth", "death"), _PARAM_SETS)
    def test_exact_equals_quadrature(
        self, times: list[float], birth: list[float], death: list[float]
    ) -> None:
        b = PiecewiseConstantRate(jnp.array(birth), jnp.array(times))
        d = PiecewiseConstantRate(jnp.array(death), jnp.array(times))
        ages = jnp.linspace(0.01, times[-1] - 0.01, 37)
        for fn in (non_spine_age_log_density, spine_age_log_density):
            np.testing.assert_allclose(
                fn(ages, b, d),
                fn(ages, b, d, method="quadrature", n_nodes=48),
                atol=1e-11,
            )
        assert float(spine_age_log_atom(b, d)) == pytest.approx(
            float(spine_age_log_atom(b, d, method="quadrature", n_nodes=48)), abs=1e-11
        )

    def test_pair_matches_individual_densities(self) -> None:
        b = PiecewiseConstantRate(jnp.array(_BIRTH), jnp.array(_TIMES))
        d = PiecewiseConstantRate(jnp.array(_DEATH), jnp.array(_TIMES))
        ages = jnp.linspace(0.05, 2.4, 11)
        non_spine, spine = age_log_densities_general(ages, b, d)
        np.testing.assert_allclose(non_spine, non_spine_age_log_density(ages, b, d))
        np.testing.assert_allclose(spine, spine_age_log_density(ages, b, d))


class TestPopulation:
    @pytest.mark.parametrize(("times", "birth", "death"), _PARAM_SETS)
    def test_conditional_pmf_from_w(
        self, times: list[float], birth: list[float], death: list[float]
    ) -> None:
        terms = _terms(times, birth, death)
        log_w = float(mn.log_nested_w_total(terms))
        log_w1 = float(mn.log_nested_w_minus_one(terms))
        counts = jnp.array([1.0, 4.0, 9.0])
        got = float(
            calculate_populations_log_lik(
                *_arrays(times, birth, death)[:2],
                counts,
                jnp.array(times),
                unbias=True,
            )
        )
        want = sum(-z * log_w + (z - 1) * log_w1 for z in (1, 4, 9))
        assert got == pytest.approx(want, rel=1e-12)

    def test_pmf_sums_to_one(self) -> None:
        b, d, t = _arrays(_TIMES, _BIRTH, _DEATH)
        counts = jnp.arange(0, 20001, dtype=jnp.float64)
        pmf = np.exp(np.asarray(calculate_populations_log_pmf(b, d, counts, t)))
        assert pmf.sum() == pytest.approx(1.0, abs=1e-10)
        assert pmf[0] == pytest.approx(
            float(calculate_extinction_pr(b, d, t)), rel=1e-12
        )


class TestNumericalStability:
    def test_deeply_subcritical_is_finite_and_normalised(self) -> None:
        terms = _terms([60.0], [0.1], [1.5])  # Gamma(0,T] = -84
        grid = jnp.linspace(0.0, 60.0, 200001)
        mass = np.trapezoid(
            np.exp(np.asarray(mn.non_spine_log_density(terms, grid))), np.asarray(grid)
        )
        assert mass == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.isfinite(np.asarray(mn.spine_log_density(terms, grid[1:-1]))))

    @pytest.mark.parametrize("value", [0.0])
    def test_zero_rate_epochs_are_handled(self, value: float) -> None:
        # No deaths in the first epoch and no births in the last.
        terms = _terms([1.0, 2.0], [0.8, value], [value, 0.9])
        ages = jnp.array([0.3, 0.9, 1.5])
        non_spine = np.asarray(mn.non_spine_log_density(terms, ages))
        spine = np.asarray(mn.spine_log_density(terms, ages))
        # Births at 1.7 and 1.1 fall in the zero-birth epoch; 0.5 does not.
        assert np.all(np.isneginf(non_spine[:2]))
        assert np.all(np.isneginf(spine[:2]))
        assert np.isfinite(non_spine[2])
        assert np.isfinite(spine[2])

    def test_gradients_are_finite_with_critical_and_zero_epochs(self) -> None:
        ages = jnp.array([0.2, 0.7, 1.4, 2.2])
        times = jnp.array([1.0, 2.5])

        def nll(params: jax.Array) -> jax.Array:
            birth = jnp.exp(params[:2])
            death = jnp.array([birth[0], jnp.exp(params[2])])  # epoch 1 critical
            terms = mn.nested_terms(birth, death, times)
            non_spine, spine = mn.age_log_densities(terms, ages)
            return -(non_spine.sum() + jnp.max(spine - non_spine))

        value, grad = jax.jit(jax.value_and_grad(nll))(
            jnp.log(jnp.array([0.9, 0.4, 0.8]))
        )
        assert np.isfinite(float(value))
        assert np.all(np.isfinite(np.asarray(grad)))

    def test_gradients_finite_with_fixed_zero_death_epoch(self) -> None:
        ages = jnp.array([0.2, 0.7, 1.4])
        times = jnp.array([1.0, 2.5])

        def ll(params: jax.Array) -> jax.Array:
            birth = jnp.exp(params)
            death = jnp.array([0.0, 0.5])
            terms = mn.nested_terms(birth, death, times)
            return mn.non_spine_log_density(terms, ages).sum()

        grad = jax.grad(ll)(jnp.log(jnp.array([0.9, 0.7])))
        assert np.all(np.isfinite(np.asarray(grad)))


class TestLogKappa:
    @pytest.mark.parametrize("growth", [-3.0, -1e-3, 0.0, 1e-9, 1e-3, 2.5])
    @pytest.mark.parametrize("duration", [1e-8, 0.3, 4.0])
    def test_matches_high_precision(self, growth: float, duration: float) -> None:
        from decimal import Decimal, getcontext

        getcontext().prec = 50
        g, d = Decimal(growth), Decimal(duration)
        exact = d if g == 0 else (1 - (-g * d).exp()) / g
        got = float(
            jnp.exp(mn._log_kappa_minus(jnp.array(growth), jnp.array(duration)))
        )  # pyright: ignore[reportPrivateUsage]
        assert got == pytest.approx(float(exact), rel=1e-9)

    def test_zero_duration_is_log_zero(self) -> None:
        out = float(mn._log_kappa_minus(jnp.array(0.5), jnp.array(0.0)))  # pyright: ignore[reportPrivateUsage]
        assert out == mn.LOG_ZERO
