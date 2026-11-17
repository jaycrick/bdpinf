r"""Probabilities that an age lies in a range, for interval-censored ages.

An age known only to lie in $[lo, hi]$ contributes the age density averaged
uniformly over the range, $(1/w)\int_{lo}^{hi} f(a)\,da$ with $w = hi - lo$.
For piecewise-constant rates the masses have closed forms (no quadrature),
using $\mathcal W' = \gamma\mathcal W + \mu$:

- non-spine: $G(s) = (\mathcal W(s) - 1)\,e^{-M(s,T]}$ has
  $G' = \lambda\mathcal W e^{-M}$,
  so $\Pr(lo \le A \le hi) = [G(T-lo) - G(T-hi)] / (\mathcal W(T) - 1)$;
- spine: $H(s) = e^{-\Lambda(s,T]}(L_\mu(s,T) + e^{\Gamma(s,T]})$ has
  $H' = \lambda e^{-\Lambda} L_\mu$, $H(T) = 1$ and $H(0) = $ the atom at
  $A^* = T$, so the range containing $T$ takes the founder's atom.

Other rates integrate each density by Gauss-Legendre quadrature, cutting each
range at the ages $T - \tau$ where the rates change, so the masses stay smooth
as a knot moves through a range.
"""

import jax
import jax.numpy as jnp

from bdpinf.models import math_nested
from bdpinf.models.math_general import (
    AgeTerms,
    GeneralTerms,
    gauss_legendre,
    log_non_spine_cdf_kernel,
    log_spine_cdf_kernel,
    non_spine_log_density,
    spine_log_atom,
    spine_log_density,
)
from bdpinf.models.math_nested import (
    LOG_ZERO,
    NestedTerms,
    epoch_row,
    log_kappa_minus,
    log_nested_w_minus_one,
    log_or_floor,
)
from bdpinf.models.rate_functions import epoch_index

__all__ = [
    "cdf_log_masses",
    "closed_form_log_masses",
    "quadrature_log_masses",
    "range_log_masses",
]

_RANGE_NODES = 16
"""Gauss-Legendre nodes per piece for range masses.

Fewer than ``DEFAULT_NODES`` (32): each range is already cut at every rate
breakpoint, so a panel covers a single smooth piece of a (usually short) range.
The commit that introduced it (39a81c5) records no other rationale.
"""


def _log_l_before(terms: NestedTerms) -> jax.Array:
    r"""$\ln L(\tau_{k-1})$ per epoch (``LOG_ZERO`` for the first epoch).

    Formula and symbols: see the module docstring.

    """
    durations = terms.times - terms.starts
    l_terms = (
        log_or_floor(terms.birth_rates)
        - terms.gamma_before
        + log_kappa_minus(terms.growth, durations)
    )
    cumulative = jax.lax.cumlogsumexp(l_terms)
    return jnp.concatenate([jnp.full((1,), LOG_ZERO), cumulative[:-1]])


def _log_g(terms: NestedTerms, log_l_before: jax.Array, s: jax.Array) -> jax.Array:
    r"""$\ln G(s)$ with $G = (\mathcal W - 1)e^{-M(s,T]}$.

    Formula and symbols: see the module docstring.

    """
    row = epoch_row(terms, s)
    u = s - row.start
    rem = row.time - s
    k = epoch_index(terms.times, s)
    partial = (
        log_or_floor(row.birth) - row.gamma_before + log_kappa_minus(row.growth, u)
    )
    log_l = jnp.logaddexp(jnp.take(log_l_before, k), partial)
    gamma_s = row.gamma_before + row.growth * u
    return gamma_s + log_l - (row.death * rem + row.mu_after)


def _log_h(terms: NestedTerms, s: jax.Array) -> jax.Array:
    r"""$\ln H(s)$, the CDF of the spine's birth time (atom at ``s = 0``).

    Formula and symbols: see the module docstring.

    """
    row = epoch_row(terms, s)
    u = s - row.start
    rem = row.time - s
    partial = (
        row.log_death
        + (terms.gamma_total - row.gamma_end)
        + row.growth * rem
        + log_kappa_minus(row.growth, rem)
    )
    log_l_mu = jnp.logaddexp(partial, row.log_spine_after)
    gamma_to_end = terms.gamma_total - (row.gamma_before + row.growth * u)
    return -(row.birth * rem + row.lam_after) + jnp.logaddexp(log_l_mu, gamma_to_end)


def _log_diff(
    upper: jax.Array, lower: jax.Array, *, lower_is_zero: jax.Array
) -> jax.Array:
    """``log(exp(upper) - exp(lower))``, with ``lower`` treated as 0 where flagged."""
    delta = jnp.where(lower_is_zero, -jnp.inf, lower - upper)
    return upper + jnp.log1p(-jnp.exp(delta))


def closed_form_log_masses(
    terms: NestedTerms, lo: jax.Array, hi: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Log masses of ``[lo, hi]`` under the non-spine and spine age laws (exact).

    Formula and symbols: see the module docstring.

    Valid for piecewise-constant rates only.
    """
    end = terms.times[-1]
    s_big = jnp.clip(end - lo, 0.0, end)
    s_small = jnp.clip(end - hi, 0.0, end)
    log_l_before = _log_l_before(terms)
    never = jnp.zeros_like(lo, dtype=bool)
    non_spine = _log_diff(
        _log_g(terms, log_l_before, s_big),
        _log_g(terms, log_l_before, s_small),
        lower_is_zero=never,
    ) - log_nested_w_minus_one(terms)
    spine = _log_diff(
        _log_h(terms, s_big), _log_h(terms, s_small), lower_is_zero=hi >= end
    )
    return non_spine, spine


def _end_time(terms: AgeTerms) -> jax.Array:
    """Observation time of either kind of terms."""
    return terms.times[-1] if isinstance(terms, NestedTerms) else terms.edges[-1]


def _interior_breakpoints(terms: AgeTerms) -> jax.Array:
    """Interior epoch boundaries (birth times where the rates are not smooth)."""
    if isinstance(terms, NestedTerms):
        return terms.times[:-1]
    return terms.edges[1:-1]


def quadrature_log_masses(
    terms: AgeTerms, lo: jax.Array, hi: jax.Array, n_nodes: int = _RANGE_NODES
) -> tuple[jax.Array, jax.Array]:
    """Log masses of ``[lo, hi]`` by Gauss-Legendre quadrature (any rates).

    Formula and symbols: see the module docstring.

    """
    end = _end_time(terms)
    cuts = jnp.clip(
        end - _interior_breakpoints(terms)[None, :], lo[:, None], hi[:, None]
    )
    points = jnp.sort(jnp.concatenate([lo[:, None], cuts, hi[:, None]], axis=1), axis=1)
    left, right = points[:, :-1], points[:, 1:]
    x, weights = gauss_legendre(left, right, n_nodes)
    flat = x.reshape(-1)

    def integrate(log_density: jax.Array) -> jax.Array:
        dens = jnp.exp(log_density.reshape(x.shape))
        return jnp.sum(jnp.sum(weights * dens, axis=-1), axis=-1)

    non_spine = integrate(non_spine_log_density(terms, flat))
    continuous = integrate(spine_log_density(terms, flat, atom_at_boundary=False))
    atom = jnp.where(hi >= end, jnp.exp(spine_log_atom(terms)), 0.0)
    return jnp.log(non_spine), jnp.log(continuous + atom)


def cdf_log_masses(
    terms: GeneralTerms, lo: jax.Array, hi: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Log masses of ``[lo, hi]`` from the cumulative functions at the endpoints.

    Formula and symbols: see the module docstring.

    Equal to :func:`quadrature_log_masses` but needs one short partial-segment
    integral per endpoint instead of a density quadrature over every piece of
    every range, which makes the likelihood and its derivatives far cheaper.
    """
    end = terms.edges[-1]
    s_big = jnp.clip(end - lo, 0.0, end)
    s_small = jnp.clip(end - hi, 0.0, end)
    never = jnp.zeros_like(lo, dtype=bool)
    non_spine = _log_diff(
        log_non_spine_cdf_kernel(terms, s_big),
        log_non_spine_cdf_kernel(terms, s_small),
        lower_is_zero=never,
    ) - math_nested.log_nested_w_minus_one(terms)
    spine = _log_diff(
        log_spine_cdf_kernel(terms, s_big),
        log_spine_cdf_kernel(terms, s_small),
        lower_is_zero=hi >= end,
    )
    return non_spine, spine


def range_log_masses(
    terms: AgeTerms, lo: jax.Array, hi: jax.Array, *, quadrature: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Log masses of ``[lo, hi]``: closed form for piecewise-constant rates.

    Formula and symbols: see the module docstring.

    Other rates use the cumulative-function form (:func:`cdf_log_masses`).
    Set ``quadrature`` to integrate the densities numerically instead, which is
    slower but independent (used to cross-check the other two).
    """
    if quadrature:
        return quadrature_log_masses(terms, lo, hi)
    if isinstance(terms, NestedTerms):
        return closed_form_log_masses(terms, lo, hi)
    return cdf_log_masses(terms, lo, hi)
