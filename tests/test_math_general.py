"""Test the quadrature terms against the exact epoch-wise backend and Wolfram oracles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.models import math_general as mg
from bdpinf.models import math_nested as mn
from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate

_TIMES = jnp.array([0.9, 2.5])
_AGES = jnp.linspace(0.02, 2.48, 40)


def _pc(
    birth: list[float], death: list[float], times: jax.Array = _TIMES
) -> tuple[PiecewiseConstantRate, PiecewiseConstantRate]:
    return (
        PiecewiseConstantRate(jnp.array(birth), times),
        PiecewiseConstantRate(jnp.array(death), times),
    )


def _wolfram_piecewise_linear() -> tuple[PiecewiseLinearRate, PiecewiseLinearRate]:
    """Return the kinked linear pair of ``AgeDist_PiecewiseLinearBoth.nb``."""
    lam_v, s1, s2, mu_v, r1, r2, tau, end = 1.2, 0.3, -0.2, 0.7, 0.15, -0.1, 0.9, 2.5
    times = jnp.array([tau, end])
    birth = PiecewiseLinearRate(
        jnp.array([lam_v - s1 * tau, lam_v]),
        jnp.array([lam_v, lam_v + s2 * (end - tau)]),
        times,
    )
    death = PiecewiseLinearRate(
        jnp.array([mu_v - r1 * tau, mu_v]),
        jnp.array([mu_v, mu_v + r2 * (end - tau)]),
        times,
    )
    return birth, death


class TestQuadratureMatchesExact:
    """Test that quadrature reproduces ``nested_terms`` for piecewise-constant rates."""

    @pytest.mark.parametrize(
        ("birth", "death"),
        [
            ([1.2, 0.8], [0.7, 1.1]),
            ([1.0, 1.0], [1.0, 1.0]),  # critical in both epochs
            ([0.5, 2.0], [1.5, 0.3]),
            ([1.2, 0.0], [0.0, 0.9]),  # zero rates
        ],
    )
    def test_terms_and_densities(self, birth: list[float], death: list[float]) -> None:
        b, d = _pc(birth, death)
        exact = mg.terms_for(b, d)
        quad = mg.terms_for(b, d, method="quadrature")
        assert isinstance(exact, mn.NestedTerms)
        assert isinstance(quad, mg.GeneralTerms)
        for name in ("gamma_total", "lam_total"):
            np.testing.assert_allclose(
                getattr(exact, name), getattr(quad, name), rtol=1e-12, atol=1e-12
            )
        if birth[0] > 0 and death[0] > 0:
            for name in ("log_i_total", "log_l_total"):
                np.testing.assert_allclose(
                    getattr(exact, name), getattr(quad, name), rtol=1e-10
                )
        for density in (mg.non_spine_log_density, mg.spine_log_density):
            want = np.asarray(density(exact, _AGES))
            got = np.asarray(density(quad, _AGES))
            finite = np.isfinite(want)
            np.testing.assert_array_equal(finite, np.isfinite(got))
            np.testing.assert_allclose(got[finite], want[finite], atol=1e-9)

    def test_atom(self) -> None:
        b, d = _pc([1.2, 0.8], [0.7, 1.1])
        np.testing.assert_allclose(
            mg.spine_log_atom(mg.terms_for(b, d, method="quadrature")),
            mg.spine_log_atom(mg.terms_for(b, d)),
            rtol=1e-11,
        )


class TestWolframOracles:
    """Test against values computed in Mathematica (``AgeDist_PiecewiseLinearBoth.nb``)."""

    def test_nested_w_total(self) -> None:
        terms = mg.terms_for(*_wolfram_piecewise_linear())
        np.testing.assert_allclose(
            np.exp(mn.log_nested_w_total(terms)), 5.661968424654191, rtol=1e-12
        )

    def test_densities(self) -> None:
        terms = mg.terms_for(*_wolfram_piecewise_linear())
        age = jnp.array([1.2])
        np.testing.assert_allclose(
            np.exp(mg.non_spine_log_density(terms, age)),
            0.3430933066148787,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.exp(mg.spine_log_density(terms, age)), 0.31050420949098767, rtol=1e-8
        )

    def test_densities_integrate_to_one(self) -> None:
        terms = mg.terms_for(*_wolfram_piecewise_linear())
        grid = jnp.linspace(0.0, 2.5, 20001)[1:-1]
        step = 2.5 / 20000
        non_spine = np.exp(np.asarray(mg.non_spine_log_density(terms, grid)))
        spine = np.exp(np.asarray(mg.spine_log_density(terms, grid)))
        atom = float(np.exp(mg.spine_log_atom(terms)))
        np.testing.assert_allclose(np.trapezoid(non_spine, dx=step), 1.0, atol=2e-4)
        np.testing.assert_allclose(np.trapezoid(spine, dx=step) + atom, 1.0, atol=2e-4)


class TestDifferentiability:
    """Test that values and derivatives stay finite and agree across backends."""

    def test_gradient_matches_exact_backend(self) -> None:
        def total(values: jax.Array, method: str) -> jax.Array:
            b = PiecewiseConstantRate(values[:2], _TIMES)
            d = PiecewiseConstantRate(values[2:], _TIMES)
            terms = mg.terms_for(b, d, method=method)
            non_spine, spine = mg.age_log_densities(terms, _AGES)
            return non_spine.sum() + spine.sum() + mn.log_nested_w_total(terms)

        x = jnp.array([1.2, 0.8, 0.7, 1.1])
        exact = jax.jit(jax.grad(lambda v: total(v, "auto")))(x)
        quad = jax.jit(jax.grad(lambda v: total(v, "quadrature")))(x)
        assert bool(jnp.all(jnp.isfinite(quad)))
        np.testing.assert_allclose(quad, exact, rtol=1e-7)

    def test_hessian_finite_for_linear_rates(self) -> None:
        def nll(knots: jax.Array) -> jax.Array:
            times = jnp.array([1.0, 2.5])
            b = PiecewiseLinearRate.continuous(jnp.exp(knots[:3]), times)
            d = PiecewiseLinearRate.continuous(jnp.exp(knots[3:]), times)
            non_spine, _ = mg.age_log_densities(mg.terms_for(b, d), _AGES)
            return -non_spine.sum()

        hess = jax.jit(jax.hessian(nll))(
            jnp.log(jnp.array([1.2, 1.0, 0.8, 0.7, 0.9, 1.1]))
        )
        assert bool(jnp.all(jnp.isfinite(hess)))
