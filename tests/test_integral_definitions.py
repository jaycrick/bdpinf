"""Test the closed forms against direct numerical integration of their definitions."""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from bdpinf.models.math_extinction import calculate_extinction_pr
from bdpinf.models.math_nested import (
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.models.math_populations import calculate_populations_log_lik


class _Reference(NamedTuple):
    """Reference log-densities built only from the paper's integral definitions."""

    spine: Callable[[float], float]
    non_spine: Callable[[float], float]
    log_pop: Callable[[int], float]
    extinction: float


def _reference(birth: list[float], death: list[float], ends: list[float]) -> _Reference:
    """Evaluate Gamma, N, W and L_mu with scipy quadrature (no closed forms).

    With Gamma(s, e] = int (lambda - mu), N(b) = int_0^b mu e^{-Gamma(0, t]} dt,
    W(b) = e^{Gamma(0, b]} (1 + N(b)) and L_mu(s, T) = int_s^T mu e^{Gamma(u, T]} du:
    spine f*(a) = lambda(s) e^{-Lambda(s, T]} L_mu(s, T) and non-spine
    f(a) = lambda(s) W(s) e^{-M(s, T]} / (W(T) - 1), with s = T - a.
    """
    t_end = ends[-1]
    kinks = ends[:-1]

    def rate(values: list[float]) -> Callable[[float], float]:
        return lambda t: values[min(int(np.searchsorted(ends, t)), len(values) - 1)]

    lam, mu = rate(birth), rate(death)

    def integral(f: Callable[[float], float], a: float, b: float) -> float:
        if b <= a:
            return 0.0
        points = [k for k in kinks if a < k < b] or None
        return quad(f, a, b, points=points, epsabs=1e-12, epsrel=1e-11, limit=500)[0]

    def gamma(a: float, b: float) -> float:
        return integral(lambda t: lam(t) - mu(t), a, b)

    def n_of(b: float) -> float:
        return integral(lambda t: mu(t) * np.exp(-gamma(0, t)), 0, b)

    def w_of(b: float) -> float:
        return float(np.exp(gamma(0, b)) * (1 + n_of(b)))

    def spine(age: float) -> float:
        s = t_end - age
        l_mu = integral(lambda u: mu(u) * np.exp(gamma(u, t_end)), s, t_end)
        return float(np.log(lam(s)) - integral(lam, s, t_end) + np.log(l_mu))

    def non_spine(age: float) -> float:
        s = t_end - age
        return float(
            np.log(lam(s))
            + np.log(w_of(s))
            - integral(mu, s, t_end)
            - np.log(w_of(t_end) - 1)
        )

    n_total = n_of(t_end)
    extinction_pr = n_total / (1 + n_total)
    pi = 1 / w_of(t_end)

    def log_pop(n: int) -> float:
        # Pr(Z_T = n) = (1 - extinction_pr) pi (1 - pi)^(n-1) for n >= 1
        return float(np.log((1 - extinction_pr) * pi * (1 - pi) ** (n - 1)))

    return _Reference(spine, non_spine, log_pop, float(extinction_pr))


_CASES = {
    "supercritical": ([0.05], [0.03], [1500.0]),
    "subcritical": ([0.03], [0.05], [200.0]),
    "critical_epoch": ([0.05, 0.04, 0.02], [0.03, 0.04, 0.05], [100.0, 250.0, 400.0]),
    "two_epoch": ([0.05, 0.01], [0.03, 0.02], [750.0, 1500.0]),
}


@pytest.fixture(params=sorted(_CASES))
def case(request: pytest.FixtureRequest):
    """Parameters ``(birth, death, ends)`` plus the quadrature reference."""
    birth, death, ends = _CASES[request.param]
    return birth, death, ends, _reference(birth, death, ends)


class TestAgainstIntegralDefinitions:
    """Test the epoch-wise closed forms against quadrature of the definitions."""

    def test_extinction_probability(self, case) -> None:
        birth, death, ends, ref = case
        got = calculate_extinction_pr(
            jnp.array(birth), jnp.array(death), jnp.array(ends)
        )
        assert float(got) == pytest.approx(ref.extinction, rel=1e-9)

    @pytest.mark.parametrize("fraction", [0.1, 0.4, 0.75, 0.95])
    def test_age_densities(self, case, fraction: float) -> None:
        birth, death, ends, ref = case
        age = fraction * ends[-1]
        terms = nested_terms(jnp.array(birth), jnp.array(death), jnp.array(ends))
        assert float(spine_log_density(terms, jnp.array(age))) == pytest.approx(
            ref.spine(age), rel=1e-9
        )
        assert float(non_spine_log_density(terms, jnp.array(age))) == pytest.approx(
            ref.non_spine(age), rel=1e-9
        )

    @pytest.mark.parametrize("n", [1, 5, 40])
    def test_population_pmf(self, case, n: int) -> None:
        birth, death, ends, ref = case
        got = calculate_populations_log_lik(
            jnp.array(birth),
            jnp.array(death),
            jnp.array([n]),
            jnp.array(ends),
            condition_on_survival=False,
        )
        assert float(got) == pytest.approx(ref.log_pop(n), rel=1e-9)
