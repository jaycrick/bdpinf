"""Test autodiff gradients against explicit scores derived symbolically in Mathematica.

The closed forms for constant rates (``AgeDistCommon.wl``,
``Consistency_LBDP_JointMLE.nb``) were differentiated with ``D[...]`` in the
Wolfram Language; they are an independent oracle for the JAX derivatives.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.models.rate_functions import PiecewiseConstantRate
from bdpinf.simulation import simulate_ages_from_rates

_T = 7.0


def _explicit(
    lam: float, mu: float, spine: float, ages: np.ndarray, n: int
) -> tuple[float, np.ndarray]:
    """Return log-likelihood and its (lam, mu) score for constant rates.

    Known spine below ``T``; the population term is conditioned on survival.
    """
    big_t, e = _T, np.exp((lam - mu) * _T)
    pi = (lam - mu) / (lam * e - mu)
    value = np.log(pi) + (n - 1) * np.log1p(-pi)
    a = ages
    log_fa = np.log(
        (lam * np.exp(lam * (big_t - a)) - mu * np.exp(mu * (big_t - a)))
        / (np.exp(lam * big_t) - np.exp(mu * big_t))
    )
    log_fs = np.log(
        lam * mu * (np.exp(-mu * spine) - np.exp(-lam * spine)) / (lam - mu)
    )
    value += log_fa.sum() + log_fs

    # Scores from Mathematica: D[Log[...], {{l, m}}] for each factor.
    s_pop = np.array(
        [
            (e * lam + mu * (n - 1) - lam * n)
            * (-(e * lam**2 * big_t) + mu * (-1 + e * (1 + lam * big_t)))
            / ((e - 1) * lam * (lam - mu) * (e * lam - mu)),
            (e * lam + mu * (n - 1) - lam * n)
            * (1 + e * (-1 + lam * big_t - mu * big_t))
            / ((e - 1) * (lam - mu) * (e * lam - mu)),
        ]
    )
    el, em = np.exp(lam * big_t), np.exp(mu * big_t)
    den = (el - em) * (
        np.exp(a * mu + lam * big_t) * lam - np.exp(a * lam + mu * big_t) * mu
    )
    s_a = np.stack(
        [
            el
            * (
                np.exp(a * mu + lam * big_t) * (1 - a * lam)
                + np.exp(a * lam + mu * big_t) * mu * big_t
                + np.exp(mu * (a + big_t)) * (-1 + a * lam - lam * big_t)
            )
            / den,
            -(
                em
                * (
                    np.exp(a * lam + mu * big_t) * (-1 + a * mu)
                    - np.exp(a * mu + lam * big_t) * lam * big_t
                    + np.exp(lam * (a + big_t)) * (1 - a * mu + mu * big_t)
                )
            )
            / den,
        ]
    ).sum(axis=1)
    ea_l, ea_m = np.exp(spine * lam), np.exp(spine * mu)
    s_s = np.array(
        [
            ((-ea_l + ea_m) * mu - spine * ea_m * lam * (-lam + mu))
            / ((ea_l - ea_m) * lam * (lam - mu)),
            (-(ea_m * lam) + ea_l * (lam - spine * lam * mu + spine * mu**2))
            / ((ea_l - ea_m) * (lam - mu) * mu),
        ]
    )
    return float(value), s_pop + s_a + s_s


@pytest.fixture(scope="module")
def sample() -> tuple[float, np.ndarray, np.ndarray]:
    """Return a seeded population: its spine age and non-spine ages."""
    times = jnp.array([_T])
    birth = PiecewiseConstantRate(jnp.array([1.0]), times)
    death = PiecewiseConstantRate(jnp.array([0.6]), times)
    drawn = simulate_ages_from_rates(birth, death, np.random.default_rng(4))
    assert drawn.spine_age is not None
    assert drawn.spine_age < _T
    return drawn.spine_age, drawn.non_spine_ages, drawn.all_ages


class TestExplicitScores:
    """Test JAX value and gradient against the symbolic scores."""

    @pytest.mark.parametrize(("lam", "mu"), [(1.0, 0.6), (0.7, 0.9), (0.5, 0.2)])
    def test_value_and_gradient_match(
        self, sample: tuple[float, np.ndarray, np.ndarray], lam: float, mu: float
    ) -> None:
        spine, ages, everything = sample
        n = len(everything)
        data = AgeData.from_groups([everything], _T, is_ordered=True)
        times = jnp.array([_T])

        def log_lik(theta: jax.Array) -> jax.Array:
            return log_likelihood_rates(
                data,
                PiecewiseConstantRate(theta[:1], times),
                PiecewiseConstantRate(theta[1:], times),
            )

        value, grad = jax.value_and_grad(log_lik)(jnp.array([lam, mu]))
        want_value, want_grad = _explicit(lam, mu, spine, ages, n)
        np.testing.assert_allclose(value, want_value, rtol=1e-10)
        np.testing.assert_allclose(grad, want_grad, rtol=1e-8)
