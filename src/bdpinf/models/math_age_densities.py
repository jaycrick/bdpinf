r"""Age densities of a birth-death process with time-varying rates.

Implements the general-rate results of the paper (Theorems on the spine,
non-spine and uniformly chosen individual's age) for any pair of
:class:`~bdpinf.models.rate_functions.RateFunction` objects.

Notation: $\gamma = \lambda - \mu$,
$\Lambda(s,e] = \int_s^e \lambda$, $M(s,e] = \int_s^e \mu$,
$\Gamma(s,e] = \Lambda(s,e] - M(s,e]$, and

$$
\mathcal W(b) = e^{\Gamma(0,b]} + \int_0^b \mu(t) e^{\Gamma(t,b]}\,dt
= 1 + \int_0^b \lambda(t) e^{\Gamma(t,b]}\,dt,
\qquad \pi(b) = 1/\mathcal W(b).
$$

Using $\mathcal W' = \gamma\mathcal W + \mu$ the exponent in the
theorems satisfies
$\int_{T-a}^T (\mu\pi + \lambda) = \ln[\pi(T-a)/\pi(T)] + M(T-a,T]$.
Writing $L_w(s,e) = \int_s^e w(t) e^{\Gamma(t,e]}\,dt$, the densities
become subtraction-free and need no division by $\gamma$ (so
$\lambda \equiv \mu$ is handled exactly):

- spine, $0 < a < T$:
  $f^*(a) = \lambda(T-a)\, e^{-\Lambda(T-a,T]}\, L_\mu(T-a, T)$;
- spine atom: $\Pr(A^* = T) = \mathcal W(T)\, e^{-\Lambda(0,T]}$;
- non-spine:
  $f(a) = \lambda(T-a)\,\mathcal W(T-a)\, e^{-M(T-a,T]} / L_\lambda(0,T)$,
  since $\mathcal W(T) - 1 = L_\lambda(0,T)$.

$L_w$ is evaluated by Gauss-Legendre quadrature on every segment
between rate breakpoints; its integrand is smooth there, so convergence is
spectral for piecewise-polynomial (and any analytic) rates.
All computation is in log space.
"""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from bdpinf.models import math_nested
from bdpinf.models.rate_functions import PiecewiseConstantRate, RateFunction

__all__ = [
    "age_log_densities_general",
    "log_nested_w_general",
    "non_spine_age_cdf",
    "non_spine_age_log_density",
    "random_individual_cdf",
    "spine_age_cdf",
    "spine_age_log_atom",
    "spine_age_log_density",
]

_DEFAULT_NODES = 32

Method = Literal["auto", "quadrature"]
"""``"auto"`` uses the exact epoch-wise backend for piecewise-constant rates
(:mod:`bdpinf.models.math_nested`) and Gauss-Legendre quadrature otherwise;
``"quadrature"`` forces quadrature (an independent oracle for the exact path)."""


def _exact_terms(
    birth: RateFunction, death: RateFunction, method: Method
) -> math_nested.NestedTerms | None:
    """Epoch terms when both rates are piecewise constant on the same epochs."""
    if method != "auto":
        return None
    if not (
        isinstance(birth, PiecewiseConstantRate)
        and isinstance(death, PiecewiseConstantRate)
    ):
        return None
    if birth.epoch_end_times.shape != death.epoch_end_times.shape:
        return None
    both = (birth.epoch_end_times, death.epoch_end_times)
    if not any(isinstance(x, jax.core.Tracer) for x in both) and not bool(
        jnp.array_equal(*both)
    ):
        return None
    return math_nested.nested_terms(birth.values, death.values, birth.epoch_end_times)


# ---------------------------------------------------------------------------
# Quadrature primitives
# ---------------------------------------------------------------------------


def _safe_log(x: jax.Array) -> jax.Array:
    positive = x > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), -jnp.inf)


def _sorted_breakpoints(*rates: RateFunction) -> jax.Array:
    return jnp.sort(jnp.concatenate([r.breakpoints for r in rates]))


def _segments(
    lo: jax.Array, hi: jax.Array, kinks: jax.Array, n_nodes: int
) -> tuple[jax.Array, jax.Array]:
    """Gauss-Legendre nodes and weights on [lo, hi] split at ``kinks``.

    Returns arrays of shape ``(n_kinks + 1, n_nodes)``; segments outside
    ``[lo, hi]`` collapse to zero length (zero weight).
    """
    xi, wi = np.polynomial.legendre.leggauss(n_nodes)
    edges = jnp.concatenate([lo[None], jnp.clip(kinks, lo, hi), hi[None]])
    a, b = edges[:-1], edges[1:]
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return mid[:, None] + half[:, None] * xi[None, :], half[:, None] * wi[None, :]


def _log_l(
    weight: RateFunction,
    birth: RateFunction,
    death: RateFunction,
    s: jax.Array,
    e: jax.Array,
    n_nodes: int,
) -> jax.Array:
    r"""Scalar $\ln L_w(s, e)$ with $L_w = \int_s^e w\,e^{\Gamma(t,e]}$."""
    x, w = _segments(s, e, _sorted_breakpoints(birth, death), n_nodes)
    gamma_te = birth.integral(x, e) - death.integral(x, e)
    return jax.nn.logsumexp(_safe_log(weight(x)) + gamma_te, b=w)


def log_nested_w_general(
    birth: RateFunction,
    death: RateFunction,
    b: jax.Array,
    *,
    n_nodes: int = _DEFAULT_NODES,
    method: Method = "auto",
) -> jax.Array:
    r"""Return $\ln \mathcal W(b)$, where $\pi(b) = 1/\mathcal W(b)$.

    Parameters
    ----------
    birth, death : RateFunction
        Birth rate $\lambda$ and death rate $\mu$, same end time.
    b : jax.Array
        Times in ``[0, T]``, any shape.
    n_nodes : int
        Gauss-Legendre nodes per segment (quadrature path only).
    method : {"auto", "quadrature"}
        See :data:`Method`.

    """
    exact = _exact_terms(birth, death, method)
    if exact is not None:
        return math_nested.log_nested_w(exact, b)

    def scalar(b_: jax.Array) -> jax.Array:
        zero = jnp.zeros_like(b_)
        gamma_0b = birth.integral(zero, b_) - death.integral(zero, b_)
        return jnp.logaddexp(gamma_0b, _log_l(death, birth, death, zero, b_, n_nodes))

    return jnp.vectorize(scalar, signature="()->()")(jnp.asarray(b, dtype=jnp.float64))


# ---------------------------------------------------------------------------
# Densities
# ---------------------------------------------------------------------------


def spine_age_log_density(
    ages: jax.Array,
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = _DEFAULT_NODES,
    method: Method = "auto",
) -> jax.Array:
    r"""Log density of the continuous part of the spine age $A^*$, ``0 < a < T``.

    The atom at $a = T$ is :func:`spine_age_log_atom`.
    """
    exact = _exact_terms(birth, death, method)
    if exact is not None:
        return math_nested.spine_log_density(exact, ages, atom_at_boundary=False)
    end = birth.end_time

    def scalar(a: jax.Array) -> jax.Array:
        s = end - a
        return (
            _safe_log(birth(s))
            - birth.integral(s, end)
            + _log_l(death, birth, death, s, end, n_nodes)
        )

    return jnp.vectorize(scalar, signature="()->()")(
        jnp.asarray(ages, dtype=jnp.float64)
    )


def spine_age_log_atom(
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = _DEFAULT_NODES,
    method: Method = "auto",
) -> jax.Array:
    r"""Log of $\Pr(A^* = T) = \mathcal W(T) e^{-\Lambda(0,T]}$."""
    exact = _exact_terms(birth, death, method)
    if exact is not None:
        return math_nested.spine_log_atom(exact)
    end = birth.end_time
    return log_nested_w_general(
        birth, death, end, n_nodes=n_nodes, method="quadrature"
    ) - birth.integral(jnp.zeros_like(end), end)


def non_spine_age_log_density(
    ages: jax.Array,
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = _DEFAULT_NODES,
    method: Method = "auto",
) -> jax.Array:
    r"""Log density of the non-spine age $A$, ``0 < a < T``."""
    exact = _exact_terms(birth, death, method)
    if exact is not None:
        return math_nested.non_spine_log_density(exact, ages)
    end = birth.end_time
    log_norm = _log_l(birth, birth, death, jnp.zeros_like(end), end, n_nodes)

    def scalar(a: jax.Array) -> jax.Array:
        s = end - a
        return (
            _safe_log(birth(s))
            + log_nested_w_general(
                birth, death, s, n_nodes=n_nodes, method="quadrature"
            )
            - death.integral(s, end)
            - log_norm
        )

    return jnp.vectorize(scalar, signature="()->()")(
        jnp.asarray(ages, dtype=jnp.float64)
    )


def age_log_densities_general(
    ages: jax.Array,
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = _DEFAULT_NODES,
    method: Method = "auto",
) -> tuple[jax.Array, jax.Array]:
    """Return the continuous ``(non_spine, spine)`` log densities in one call.

    For piecewise-constant rates both come from one shared pass.
    """
    exact = _exact_terms(birth, death, method)
    if exact is not None:
        return math_nested.age_log_densities(exact, ages, atom_at_boundary=False)
    return (
        non_spine_age_log_density(ages, birth, death, n_nodes=n_nodes, method=method),
        spine_age_log_density(ages, birth, death, n_nodes=n_nodes, method=method),
    )


# ---------------------------------------------------------------------------
# CDFs
# ---------------------------------------------------------------------------


def _integrate_age_density(
    log_density: Callable[[jax.Array], jax.Array],
    upper: jax.Array,
    kinks: jax.Array,
    n_nodes: int,
) -> jax.Array:
    """Integral of ``exp(log_density)`` over ``[0, upper]``, split at age kinks."""

    def scalar(u: jax.Array) -> jax.Array:
        x, w = _segments(jnp.zeros_like(u), u, kinks, n_nodes)
        return jnp.sum(w * jnp.exp(log_density(x)))

    return jnp.vectorize(scalar, signature="()->()")(upper)


def _age_kinks(birth: RateFunction, death: RateFunction) -> jax.Array:
    return jnp.sort(birth.end_time - _sorted_breakpoints(birth, death))


def non_spine_age_cdf(
    ages: jax.Array,
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = _DEFAULT_NODES,
) -> jax.Array:
    """CDF $F_A(a)$ of the non-spine age."""
    end = birth.end_time
    a = jnp.minimum(jnp.asarray(ages, dtype=jnp.float64), end)
    return _integrate_age_density(
        lambda x: non_spine_age_log_density(x, birth, death, n_nodes=n_nodes),
        a,
        _age_kinks(birth, death),
        n_nodes,
    )


def spine_age_cdf(
    ages: jax.Array,
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = _DEFAULT_NODES,
) -> jax.Array:
    """CDF $F_{A^*}(a)$ of the spine age, including the atom at ``T``."""
    end = birth.end_time
    ages = jnp.asarray(ages, dtype=jnp.float64)
    cont = _integrate_age_density(
        lambda x: spine_age_log_density(x, birth, death, n_nodes=n_nodes),
        jnp.minimum(ages, end),
        _age_kinks(birth, death),
        n_nodes,
    )
    atom = jnp.exp(spine_age_log_atom(birth, death, n_nodes=n_nodes))
    return cont + jnp.where(ages >= end, atom, 0.0)


def random_individual_cdf(
    ages: jax.Array,
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = _DEFAULT_NODES,
) -> jax.Array:
    r"""CDF $F_{A^\dagger}$ of the age of a uniformly chosen individual.

    The mixture $w_A F_A + w_{A^*} F_{A^*}$ with
    $w_{A^*} = -\pi(T)\ln\pi(T) / (1 - \pi(T))$ and
    $w_A = 1 - w_{A^*}$.
    """
    log_w_t = log_nested_w_general(birth, death, birth.end_time, n_nodes=n_nodes)
    pi_t = jnp.exp(-log_w_t)
    w_spine = pi_t * log_w_t / -jnp.expm1(-log_w_t)
    return (1.0 - w_spine) * non_spine_age_cdf(
        ages, birth, death, n_nodes=n_nodes
    ) + w_spine * spine_age_cdf(ages, birth, death, n_nodes=n_nodes)
