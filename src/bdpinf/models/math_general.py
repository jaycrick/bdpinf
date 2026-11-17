r"""Epoch-wise terms of the nested integral $\mathcal W$ for any pair of rate functions.

This is the general-rate counterpart of :mod:`bdpinf.models.math_nested`.
Between the breakpoints of the birth and death rates the integrands are smooth,
so the segment integrals

$$
J_\mu = \int \mu(t)\,e^{-\Gamma(0,t]}\,dt, \qquad
J_\lambda = \int \lambda(t)\,e^{-\Gamma(0,t]}\,dt
$$

are evaluated by Gauss-Legendre quadrature once per segment
(:func:`general_terms`), accumulated with ``cumlogsumexp``
and then reused for every age, which needs only two short quadratures:
one on the part of its segment before $s = T - a$ (for $\mathcal W(s)$)
and one on the part after (for the spine tail $L_\mu(s, T)$).
Everything is in log space and subtraction-free,
so the same relations as in :mod:`bdpinf.models.math_nested` hold:

- $\mathcal W(b) = e^{\Gamma(0,b]}\,(1 + I(b))$, $I(b) = \int_0^b \mu e^{-\Gamma(0,t]}$;
- $\mathcal W(T) - 1 = e^{\Gamma(0,T]} L(T)$,
  $L(T) = \int_0^T \lambda e^{-\Gamma(0,t]}$;
- non-spine age:
  $f(a) = \lambda(T-a)\,\mathcal W(T-a)\,e^{-M(T-a,T]} / (\mathcal W(T) - 1)$;
- spine age: $f^*(a) = \lambda(T-a)\,e^{-\Lambda(T-a,T]}\,L_\mu(T-a,T)$,
  with atom $\Pr(A^*=T) = \mathcal W(T)\,e^{-\Lambda(0,T]}$.

:func:`terms_for` picks the exact epoch-wise backend for piecewise-constant
rates on shared epochs and quadrature otherwise,
and the density functions here accept either kind of terms.

The quadrature is spectrally accurate while the exponent
$\Gamma$ varies by at most a few tens within a segment.
For extreme rates (a rate times its segment length in the hundreds or more)
a single panel of Gauss-Legendre nodes cannot resolve the integrand;
the exact backend has no such limit, which is one reason piecewise-constant
rates keep their own path.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bdpinf.models import math_nested
from bdpinf.models.math_nested import (
    LOG_ZERO,
    NestedTerms,
    log_or_floor,
    restore_neg_inf,
)
from bdpinf.models.rate_functions import PiecewiseConstantRate, RateFunction

__all__ = [
    "DEFAULT_NODES",
    "AgeTerms",
    "GeneralTerms",
    "age_log_densities",
    "general_terms",
    "log_nested_w",
    "non_spine_log_density",
    "spine_log_atom",
    "spine_log_density",
    "terms_for",
]

DEFAULT_NODES = 32


class GeneralTerms(NamedTuple):
    """Segment-wise precomputation for arbitrary rate functions.

    The fields ``gamma_total``, ``log_i_total``, ``log_l_total`` and
    ``lam_total`` mean the same as in :class:`NestedTerms`, so the
    population-level functions of :mod:`bdpinf.models.math_nested`
    (``log_nested_w_total``, ``log_extinction_and_survival_pr``, ...)
    accept either type.
    """

    birth: RateFunction
    death: RateFunction
    n_nodes: int
    edges: jax.Array  # (S + 1,) segment boundaries 0 = e_0 <= ... <= e_S = T
    gamma_total: jax.Array
    log_i_total: jax.Array
    log_l_total: jax.Array
    lam_total: jax.Array
    log_i_before: jax.Array  # (S,) log of the mu-integral over earlier segments
    log_i_after: jax.Array  # (S,) log of the mu-integral over later segments
    log_l_before: jax.Array  # (S,) log of the lambda-integral over earlier segments


AgeTerms = NestedTerms | GeneralTerms
"""Either kind of precomputed terms; the functions below dispatch on the type."""


# ---------------------------------------------------------------------------
# Quadrature primitives
# ---------------------------------------------------------------------------


@functools.cache
def _unit_rule(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on ``[-1, 1]``, computed once per size."""
    return np.polynomial.legendre.leggauss(n_nodes)


def gauss_legendre(
    lo: jax.Array, hi: jax.Array, n_nodes: int
) -> tuple[jax.Array, jax.Array]:
    """Gauss-Legendre nodes and weights on each interval ``[lo, hi]``.

    ``lo`` and ``hi`` may have any (equal) shape ``S``;
    both results have shape ``S + (n_nodes,)``, and empty intervals get zero weight.
    The unit rule is cached per ``n_nodes``, so this is safe under ``jit``.
    """
    xi, wi = _unit_rule(n_nodes)
    half = 0.5 * (hi - lo)
    return (0.5 * (lo + hi))[..., None] + half[..., None] * xi, half[..., None] * wi


def _log_gamma_neg(birth: RateFunction, death: RateFunction, x: jax.Array) -> jax.Array:
    r"""$-\Gamma(0, x]$ for each ``x``."""
    zero = jnp.zeros_like(x)
    return death.integral(zero, x) - birth.integral(zero, x)


def _log_integral(
    weight: RateFunction,
    birth: RateFunction,
    death: RateFunction,
    lo: jax.Array,
    hi: jax.Array,
    *,
    n_nodes: int,
) -> jax.Array:
    r"""$\ln\int_{lo}^{hi} w(t)\,e^{-\Gamma(0,t]}\,dt$ by Gauss-Legendre.

    ``lo`` and ``hi`` may have any (equal) shape; the result has that shape.
    ``LOG_ZERO`` is returned for an empty interval or an identically zero weight.
    """
    x, w = gauss_legendre(lo, hi, n_nodes)
    log_f = log_or_floor(weight(x)) + _log_gamma_neg(birth, death, x)
    peak = jax.lax.stop_gradient(jnp.max(log_f, axis=-1, keepdims=True))
    total = jnp.sum(w * jnp.exp(log_f - peak), axis=-1)
    positive = total > 0
    return jnp.where(
        positive,
        jnp.log(jnp.where(positive, total, 1.0)) + peak[..., 0],
        LOG_ZERO,
    )


def _segment_edges(birth: RateFunction, death: RateFunction) -> jax.Array:
    inner = jnp.sort(jnp.concatenate([birth.breakpoints, death.breakpoints]))
    end = birth.end_time
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), inner, end[None]])


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------


def general_terms(
    birth: RateFunction, death: RateFunction, *, n_nodes: int = DEFAULT_NODES
) -> GeneralTerms:
    r"""Precompute the segment integrals behind $\mathcal W$.

    Parameters
    ----------
    birth, death : RateFunction
        Birth and death rates on the same observation window.
    n_nodes : int
        Gauss-Legendre nodes per segment.

    """
    edges = _segment_edges(birth, death)
    lo, hi = edges[:-1], edges[1:]
    log_i_seg = _log_integral(death, birth, death, lo, hi, n_nodes=n_nodes)
    log_l_seg = _log_integral(birth, birth, death, lo, hi, n_nodes=n_nodes)
    if log_i_seg.shape[0] == 1:  # static: nothing to accumulate
        log_i_total = log_i_seg[0]
        log_l_total = log_l_seg[0]
        log_i_before = jnp.full_like(log_i_seg, LOG_ZERO)
        log_i_after = jnp.full_like(log_i_seg, LOG_ZERO)
        log_l_before = jnp.full_like(log_l_seg, LOG_ZERO)
    else:
        cum_i = jax.lax.cumlogsumexp(log_i_seg)
        log_i_total = cum_i[-1]
        log_l_total = jax.lax.cumlogsumexp(log_l_seg)[-1]
        log_i_before = jnp.concatenate([jnp.full((1,), LOG_ZERO), cum_i[:-1]])
        log_i_after = jnp.concatenate(
            [
                jax.lax.cumlogsumexp(log_i_seg, reverse=True)[1:],
                jnp.full((1,), LOG_ZERO),
            ]
        )
        cum_l = jax.lax.cumlogsumexp(log_l_seg)
        log_l_before = jnp.concatenate([jnp.full((1,), LOG_ZERO), cum_l[:-1]])
    zero, end = jnp.zeros_like(edges[-1]), edges[-1]
    return GeneralTerms(
        birth=birth,
        death=death,
        n_nodes=n_nodes,
        edges=edges,
        gamma_total=birth.integral(zero, end) - death.integral(zero, end),
        log_i_total=log_i_total,
        log_l_total=log_l_total,
        lam_total=birth.integral(zero, end),
        log_i_before=log_i_before,
        log_i_after=log_i_after,
        log_l_before=log_l_before,
    )


def _same_epochs(birth: PiecewiseConstantRate, death: PiecewiseConstantRate) -> bool:
    """Whether two piecewise-constant rates share epochs (assumed if traced)."""
    a, b = birth.epoch_end_times, death.epoch_end_times
    if a.shape != b.shape:
        return False
    if isinstance(a, jax.core.Tracer) or isinstance(b, jax.core.Tracer):
        return True
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def terms_for(
    birth: RateFunction,
    death: RateFunction,
    *,
    n_nodes: int = DEFAULT_NODES,
    method: str = "auto",
) -> AgeTerms:
    """Choose the backend for a pair of rates.

    ``method="auto"`` uses the exact epoch-wise terms of
    :mod:`bdpinf.models.math_nested` when both rates are piecewise constant
    on the same epochs and quadrature otherwise;
    ``method="quadrature"`` forces quadrature (an independent oracle).
    """
    if (
        method == "auto"
        and isinstance(birth, PiecewiseConstantRate)
        and isinstance(death, PiecewiseConstantRate)
        and _same_epochs(birth, death)
    ):
        return math_nested.nested_terms(
            birth.values, death.values, birth.epoch_end_times
        )
    return general_terms(birth, death, n_nodes=n_nodes)


# ---------------------------------------------------------------------------
# Age densities
# ---------------------------------------------------------------------------


def spine_log_atom(terms: AgeTerms) -> jax.Array:
    r"""Log of $\Pr(A^* = T) = \mathcal W(T)\,e^{-\Lambda(0,T]}$."""
    return math_nested.spine_log_atom(terms)


def _segment_of(terms: GeneralTerms, s: jax.Array) -> jax.Array:
    """Index of the segment ``(e_k, e_{k+1}]`` containing each ``s``."""
    return jnp.clip(
        jnp.searchsorted(terms.edges[1:], s, side="left"), 0, terms.edges.shape[0] - 2
    )


def _log_w_at(terms: GeneralTerms, s: jax.Array, k: jax.Array) -> jax.Array:
    r"""$\ln\mathcal W(s)$ for ``s`` inside segment ``k``."""
    birth, death = terms.birth, terms.death
    part = _log_integral(death, birth, death, terms.edges[k], s, n_nodes=terms.n_nodes)
    log_i_s = jnp.logaddexp(terms.log_i_before[k], part)
    zero = jnp.zeros_like(s)
    gamma_s = birth.integral(zero, s) - death.integral(zero, s)
    return gamma_s + jnp.logaddexp(0.0, log_i_s)


def log_non_spine_cdf_kernel(terms: GeneralTerms, s: jax.Array) -> jax.Array:
    r"""$\ln G(s)$ with $G = (\mathcal W(s) - 1)\,e^{-M(s,T]}$, one partial integral.

    $G' = \lambda\mathcal W e^{-M}$ and $G(0) = 0$, so the non-spine mass of the
    ages $[a_1, a_2]$ is $[G(T-a_1) - G(T-a_2)] / (\mathcal W(T) - 1)$.
    Uses $\mathcal W(s) - 1 = e^{\Gamma(0,s]} L(s)$ with $L$ the
    $\lambda$-integral, which is subtraction-free in log space.
    """
    birth, death = terms.birth, terms.death
    end = terms.edges[-1]
    k = _segment_of(terms, s)
    part = _log_integral(birth, birth, death, terms.edges[k], s, n_nodes=terms.n_nodes)
    log_l_s = jnp.logaddexp(terms.log_l_before[k], part)
    zero = jnp.zeros_like(s)
    gamma_s = birth.integral(zero, s) - death.integral(zero, s)
    return gamma_s + log_l_s - death.integral(s, end)


def log_spine_cdf_kernel(terms: GeneralTerms, s: jax.Array) -> jax.Array:
    r"""$\ln H(s)$, the CDF of the spine's birth time $s$ (atom at $s = 0$).

    $H(s) = e^{-\Lambda(s,T]}(L_\mu(s,T) + e^{\Gamma(s,T]})$ with
    $H(T) = 1$, so the spine mass of the ages $[a_1, a_2]$ is
    $H(T-a_1) - H(T-a_2)$, and the range that contains $T$ takes the atom.
    """
    birth, death = terms.birth, terms.death
    end = terms.edges[-1]
    k = _segment_of(terms, s)
    part = _log_integral(
        death, birth, death, s, terms.edges[k + 1], n_nodes=terms.n_nodes
    )
    log_l_mu = restore_neg_inf(
        terms.gamma_total + jnp.logaddexp(part, terms.log_i_after[k])
    )
    zero = jnp.zeros_like(s)
    gamma_s = birth.integral(zero, s) - death.integral(zero, s)
    return -birth.integral(s, end) + jnp.logaddexp(
        log_l_mu, terms.gamma_total - gamma_s
    )


def log_nested_w(terms: AgeTerms, b: jax.Array) -> jax.Array:
    r"""Return $\ln\mathcal W(b)$ at times ``b`` (clipped to ``[0, T]``)."""
    if isinstance(terms, NestedTerms):
        return math_nested.log_nested_w(terms, b)
    b = jnp.clip(jnp.asarray(b, dtype=jnp.float64), 0.0, terms.edges[-1])
    return _log_w_at(terms, b, _segment_of(terms, b))


class _Pass(NamedTuple):
    """Per-age quantities shared by the non-spine and spine densities."""

    ages: jax.Array
    s: jax.Array  # birth time ``T - age`` of each individual
    k: jax.Array  # segment containing ``s``
    log_birth: jax.Array  # -inf where the rate is zero


def _prepare(terms: GeneralTerms, ages: jax.Array) -> _Pass:
    ages = jnp.asarray(ages, dtype=jnp.float64)
    end = terms.edges[-1]
    s = jnp.clip(end - ages, 0.0, end)
    birth_s = terms.birth(s)
    log_birth = jnp.where(birth_s > 0, log_or_floor(birth_s), -jnp.inf)
    return _Pass(ages, s, _segment_of(terms, s), log_birth)


def _non_spine(terms: GeneralTerms, p: _Pass) -> jax.Array:
    log_norm = math_nested.log_nested_w_minus_one(terms)
    value = (
        p.log_birth
        + _log_w_at(terms, p.s, p.k)
        - terms.death.integral(p.s, terms.edges[-1])
        - log_norm
    )
    return jnp.where(log_norm < 0.5 * LOG_ZERO, -jnp.inf, value)


def _spine(terms: GeneralTerms, p: _Pass, *, atom_at_boundary: bool) -> jax.Array:
    birth, death, end = terms.birth, terms.death, terms.edges[-1]
    part = _log_integral(
        death, birth, death, p.s, terms.edges[p.k + 1], n_nodes=terms.n_nodes
    )
    log_l_mu = restore_neg_inf(
        terms.gamma_total + jnp.logaddexp(part, terms.log_i_after[p.k])
    )
    continuous = p.log_birth - birth.integral(p.s, end) + log_l_mu
    if atom_at_boundary:
        return jnp.where(p.ages >= end, spine_log_atom(terms), continuous)
    return continuous


def non_spine_log_density(terms: AgeTerms, ages: jax.Array) -> jax.Array:
    r"""Log density of the non-spine age $A$ at ``ages`` (no atom)."""
    if isinstance(terms, NestedTerms):
        return math_nested.non_spine_log_density(terms, ages)
    return _non_spine(terms, _prepare(terms, ages))


def spine_log_density(
    terms: AgeTerms, ages: jax.Array, *, atom_at_boundary: bool = True
) -> jax.Array:
    r"""Log density of $A^*$; ages ``>= T`` score the atom if ``atom_at_boundary``."""
    if isinstance(terms, NestedTerms):
        return math_nested.spine_log_density(
            terms, ages, atom_at_boundary=atom_at_boundary
        )
    return _spine(terms, _prepare(terms, ages), atom_at_boundary=atom_at_boundary)


def age_log_densities(
    terms: AgeTerms, ages: jax.Array, *, atom_at_boundary: bool = True
) -> tuple[jax.Array, jax.Array]:
    """Return ``(non_spine, spine)`` log densities sharing one pass."""
    if isinstance(terms, NestedTerms):
        return math_nested.age_log_densities(
            terms, ages, atom_at_boundary=atom_at_boundary
        )
    p = _prepare(terms, ages)
    return _non_spine(terms, p), _spine(terms, p, atom_at_boundary=atom_at_boundary)
