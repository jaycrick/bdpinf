r"""Age densities of a birth-death process with time-varying rates.

Implements the general-rate results of the paper (the spine, non-spine and
uniformly chosen individual's age theorems, [paper: Theorems ?]) for any pair
of :class:`~bdpinf.models.rate_functions.RateFunction` objects.

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

from bdpinf.models import math_general
from bdpinf.models.rate_functions import RateFunction

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


def _terms(
    birth: RateFunction, death: RateFunction, n_nodes: int, method: Method
) -> math_general.AgeTerms:
    return math_general.terms_for(birth, death, n_nodes=n_nodes, method=method)


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
    return math_general.log_nested_w(_terms(birth, death, n_nodes, method), b)


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
    return math_general.spine_log_density(
        _terms(birth, death, n_nodes, method), ages, atom_at_boundary=False
    )


def spine_age_log_atom(
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = _DEFAULT_NODES,
    method: Method = "auto",
) -> jax.Array:
    r"""Log of $\Pr(A^* = T) = \mathcal W(T) e^{-\Lambda(0,T]}$."""
    return math_general.spine_log_atom(_terms(birth, death, n_nodes, method))


def non_spine_age_log_density(
    ages: jax.Array,
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = _DEFAULT_NODES,
    method: Method = "auto",
) -> jax.Array:
    r"""Log density of the non-spine age $A$, ``0 < a < T``."""
    return math_general.non_spine_log_density(
        _terms(birth, death, n_nodes, method), ages
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
    return math_general.age_log_densities(
        _terms(birth, death, n_nodes, method), ages, atom_at_boundary=False
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
