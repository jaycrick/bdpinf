r"""Exact epoch-wise nested integral :math:`\mathcal W` and the age densities it gives.

This is the piecewise-constant backend for the general-rate results of the paper
(``sections/2_MainResults.tex``).
For rates that are constant on each epoch :math:`(\tau_{k-1}, \tau_k]` every
quantity below has a closed form, evaluated here in O(K) once and then in
O(log K) per age, entirely in log space and without subtraction.

Notation (paper symbol, code name)
----------------------------------
=========================================  ========================================
:math:`T = \tau_K` (observation time)      ``times[-1]``
:math:`\lambda_k, \mu_k`                   ``birth_rates``, ``death_rates``
:math:`\gamma_k = \lambda_k - \mu_k`       ``growth``
:math:`d_k = \tau_k - \tau_{k-1}`          ``durations``
:math:`\Gamma(0, \tau_k]`                  ``gamma_end`` (``gamma_before``: k-1)
:math:`\Lambda(s, T], M(s, T]`             ``lam_after``/``mu_after`` plus a partial
:math:`N(b)=\int_0^b \mu e^{-\Gamma(0,t]}` ``log_n_*``
:math:`L(b)=\int_0^b \lambda e^{-\Gamma(0,t]}`  ``log_l_*``
:math:`\mathcal W(b)`                      :func:`log_nested_w`
:math:`\pi(b) = 1/\mathcal W(b)`           ``exp(-log_nested_w(...))``
:math:`L_\mu(s,T)`                         ``log_spine_after`` plus a partial epoch
:math:`A` (non-spine age)                  :func:`non_spine_log_density`
:math:`A^*` (spine age)                    :func:`spine_log_density`, atom at ``T``
=========================================  ========================================

Key relations (all exact):

- :math:`\mathcal W(b) = e^{\Gamma(0,b]}\,(1 + N(b))`
  and :math:`\mathcal W(b) - 1 = e^{\Gamma(0,b]} L(b)`.
  Hence :math:`\pi(T) = 1/(1+\texttt{ext\_num})` only when :math:`\Gamma(0,T]=0`,
  where ``ext_num`` is :math:`N(T)`.
- :math:`\Pr(Z_T = 0) = N/(1+N)` and :math:`\Pr(Z_T = z \mid Z_T \ge 1)
  = \pi(1-\pi)^{z-1}`.
- Per epoch, with :math:`\kappa^-(g, d) = (1 - e^{-gd})/g` (equal to ``d`` at
  :math:`g = 0`, so the critical case needs no special treatment),
  :math:`N` and :math:`L` accumulate :math:`w_k e^{-\Gamma(0,\tau_{k-1}]}
  \kappa^-(\gamma_k, d_k)` with :math:`w = \mu` and :math:`w = \lambda`.
- Non-spine age: :math:`f(a) = \lambda(T-a)\,\mathcal W(T-a)\,e^{-M(T-a,T]}
  / (\mathcal W(T) - 1)`.
- Spine age: :math:`f^*(a) = \lambda(T-a)\,e^{-\Lambda(T-a,T]}\,L_\mu(T-a,T)`,
  with atom :math:`\Pr(A^*=T) = \mathcal W(T)\,e^{-\Lambda(0,T]}`.

Zero rates are represented internally by the finite log-zero ``LOG_ZERO`` so
that cumulative log-sums and their gradients stay finite;
public results are mapped back to ``-inf``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bdpinf.models.validation import normalize_and_validate_inputs

__all__ = [
    "LOG_ZERO",
    "NestedTerms",
    "age_log_densities",
    "log_extinction_numerator",
    "log_nested_w",
    "log_nested_w_minus_one",
    "log_nested_w_total",
    "nested_terms",
    "non_spine_log_density",
    "restore_neg_inf",
    "spine_log_atom",
    "spine_log_density",
]

LOG_ZERO: float = -1e30
_SERIES_THRESHOLD: float = 1e-6


class NestedTerms(NamedTuple):
    """Epoch-wise precomputation shared by all quantities in this module.

    ``table`` stacks the per-epoch columns listed in ``_COLUMNS`` so that one
    gather per age retrieves everything (and none at all for a single epoch).
    """

    birth_rates: jax.Array
    death_rates: jax.Array
    times: jax.Array
    starts: jax.Array
    growth: jax.Array
    gamma_before: jax.Array
    gamma_end: jax.Array
    gamma_total: jax.Array
    log_n_before: jax.Array
    log_n_total: jax.Array
    log_l_total: jax.Array
    log_spine_after: jax.Array
    lam_after: jax.Array
    mu_after: jax.Array
    lam_total: jax.Array
    table: jax.Array


_COLUMNS = (
    "birth",
    "death",
    "time",
    "start",
    "growth",
    "gamma_before",
    "gamma_end",
    "log_n_before",
    "log_spine_after",
    "lam_after",
    "mu_after",
    "log_birth",
    "log_death",
)


class _Row(NamedTuple):
    """Per-age view of the epoch table (arrays shaped like the ages)."""

    birth: jax.Array
    death: jax.Array
    time: jax.Array
    start: jax.Array
    growth: jax.Array
    gamma_before: jax.Array
    gamma_end: jax.Array
    log_n_before: jax.Array
    log_spine_after: jax.Array
    lam_after: jax.Array
    mu_after: jax.Array
    log_birth: jax.Array  # -inf where the rate is zero
    log_death: jax.Array  # LOG_ZERO where the rate is zero


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def _log_or_floor(x: jax.Array) -> jax.Array:
    """``log x`` with ``LOG_ZERO`` at ``x <= 0`` (gradient-safe)."""
    positive = x > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), LOG_ZERO)


def _log_or_neg_inf(x: jax.Array) -> jax.Array:
    """``log x`` with ``-inf`` at ``x <= 0`` (gradient-safe)."""
    positive = x > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), -jnp.inf)


def restore_neg_inf(x: jax.Array) -> jax.Array:
    """Map the internal ``LOG_ZERO`` floor back to ``-inf``."""
    return jnp.where(x < 0.5 * LOG_ZERO, -jnp.inf, x)


def _log_kappa_minus(growth: jax.Array, duration: jax.Array) -> jax.Array:
    r"""Return :math:`\ln\kappa^-(g, d)` with :math:`\kappa^- = (1-e^{-gd})/g`.

    Writing :math:`\kappa^- = d\,(1-e^{-x})/x` with :math:`x = gd`:
    for :math:`|x|` below ``_SERIES_THRESHOLD`` (including exactly critical
    epochs) a short polynomial for :math:`(1-e^{-x})/x` is used, otherwise
    :math:`\ln\bigl[(1-e^{-|x|})/|x|\bigr] + \max(-x, 0)`, which is stable for
    either sign of ``g``.
    Returns ``LOG_ZERO`` when ``d = 0``.
    """
    x = growth * duration
    small = jnp.abs(x) < _SERIES_THRESHOLD
    x_safe = jnp.where(small, 1.0, x)
    ax = jnp.abs(x_safe)
    phi = jnp.where(small, 1.0 - x / 2.0 + x * x / 6.0, -jnp.expm1(-ax) / ax)
    boost = jnp.where(small, 0.0, jnp.maximum(-x_safe, 0.0))
    positive = duration > 0
    log_d = jnp.log(jnp.where(positive, duration, 1.0))
    return jnp.where(positive, log_d + jnp.log(phi) + boost, LOG_ZERO)


def _epoch_index(times: jax.Array, x: jax.Array) -> jax.Array:
    return jnp.clip(jnp.searchsorted(times, x, side="left"), 0, times.shape[0] - 1)


def _shift_right(x: jax.Array, fill: float) -> jax.Array:
    return jnp.concatenate([jnp.full((1,), fill, dtype=x.dtype), x[:-1]])


def _shift_left(x: jax.Array, fill: float) -> jax.Array:
    return jnp.concatenate([x[1:], jnp.full((1,), fill, dtype=x.dtype)])


# ---------------------------------------------------------------------------
# Epoch precomputation
# ---------------------------------------------------------------------------


def _is_concrete(*arrays: object) -> bool:
    """Return whether host-side validation of ``arrays`` is possible.

    False if any input is a tracer (``grad``/``vmap``) or if we are inside a
    ``jit``/``while_loop`` trace, where even closed-over constants turn into
    tracers as soon as they meet a ``jnp`` op.
    """
    traced = isinstance(jnp.zeros(()), jax.core.Tracer)
    return not traced and not any(isinstance(a, jax.core.Tracer) for a in arrays)


def nested_terms(
    birth_rates: jax.Array | float,
    death_rates: jax.Array | float,
    times: jax.Array | float,
) -> NestedTerms:
    r"""Precompute the epoch-wise quantities behind :math:`\mathcal W`.

    Concrete inputs are validated and coerced to 1-D float64 arrays;
    traced inputs (inside ``jax.jit``/``grad``) are assumed already valid.

    Parameters
    ----------
    birth_rates, death_rates : jax.Array | float
        Non-negative epoch rates, shape ``(n_epochs,)`` float64.
    times : jax.Array | float
        Strictly increasing epoch end times, shape ``(n_epochs,)``.

    Raises
    ------
    ValueError
        If the inputs have mismatched lengths, negative rates, or times that
        are not strictly increasing and positive.

    """
    if _is_concrete(birth_rates, death_rates, times):
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )
    starts = _shift_right(times, 0.0)
    durations = times - starts
    growth = birth_rates - death_rates
    x = growth * durations
    gamma_end = jnp.cumsum(x)
    gamma_before = gamma_end - x
    gamma_total = gamma_end[-1]
    log_kappa = _log_kappa_minus(growth, durations)
    log_mu = _log_or_floor(death_rates)
    log_lam = _log_or_floor(birth_rates)

    n_terms = log_mu - gamma_before + log_kappa
    l_terms = log_lam - gamma_before + log_kappa
    # kappa^+ = e^{gd} kappa^-: contribution of epoch j to L_mu(tau_k, T) for k < j.
    spine_terms = log_mu + (gamma_total - gamma_end) + x + log_kappa
    lam_dur = birth_rates * durations
    mu_dur = death_rates * durations
    if times.shape[0] == 1:  # static: nothing to accumulate over a single epoch
        log_n_cum, log_l_cum = n_terms, l_terms
        log_n_before = jnp.full_like(n_terms, LOG_ZERO)
        log_spine_after = jnp.full_like(n_terms, LOG_ZERO)
        lam_after = jnp.zeros_like(n_terms)
        mu_after = jnp.zeros_like(n_terms)
    else:
        log_n_cum = jax.lax.cumlogsumexp(n_terms)
        log_l_cum = jax.lax.cumlogsumexp(l_terms)
        log_n_before = _shift_right(log_n_cum, LOG_ZERO)
        log_spine_after = _shift_left(
            jax.lax.cumlogsumexp(spine_terms, reverse=True), LOG_ZERO
        )
        lam_after = _shift_left(jnp.cumsum(lam_dur[::-1])[::-1], 0.0)
        mu_after = _shift_left(jnp.cumsum(mu_dur[::-1])[::-1], 0.0)
    table = jnp.stack(
        [
            birth_rates,
            death_rates,
            times,
            starts,
            growth,
            gamma_before,
            gamma_end,
            log_n_before,
            log_spine_after,
            lam_after,
            mu_after,
            _log_or_neg_inf(birth_rates),
            log_mu,
        ],
        axis=1,
    )
    return NestedTerms(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        starts=starts,
        growth=growth,
        gamma_before=gamma_before,
        gamma_end=gamma_end,
        gamma_total=gamma_total,
        log_n_before=log_n_before,
        log_n_total=log_n_cum[-1],
        log_l_total=log_l_cum[-1],
        log_spine_after=log_spine_after,
        lam_after=lam_after,
        mu_after=mu_after,
        lam_total=jnp.sum(lam_dur),
        table=table,
    )


def _row(terms: NestedTerms, s: jax.Array) -> _Row:
    """Epoch quantities for each time ``s`` (no gather at all for one epoch)."""
    if terms.table.shape[0] == 1:
        row = terms.table[0]
        return _Row(*(row[i] for i in range(len(_COLUMNS))))
    k = _epoch_index(terms.times, s)
    rows = jnp.take(terms.table, k, axis=0)
    return _Row(*(rows[..., i] for i in range(len(_COLUMNS))))


# ---------------------------------------------------------------------------
# W, N and related scalars
# ---------------------------------------------------------------------------


def _log_w_from_row(row: _Row, u: jax.Array, *, single_epoch: bool) -> jax.Array:
    """``log W`` at ``start + u`` inside the epoch described by ``row``."""
    partial = row.log_death - row.gamma_before + _log_kappa_minus(row.growth, u)
    log_n_b = partial if single_epoch else jnp.logaddexp(row.log_n_before, partial)
    return row.gamma_before + row.growth * u + jnp.logaddexp(0.0, log_n_b)


def log_nested_w(terms: NestedTerms, b: jax.Array) -> jax.Array:
    r"""Return :math:`\ln\mathcal W(b)` at times ``b`` (clipped to ``[0, T]``)."""
    b = jnp.clip(jnp.asarray(b, dtype=jnp.float64), 0.0, terms.times[-1])
    row = _row(terms, b)
    return _log_w_from_row(row, b - row.start, single_epoch=terms.table.shape[0] == 1)


def log_nested_w_total(terms: NestedTerms) -> jax.Array:
    r"""Return :math:`\ln\mathcal W(T)`, i.e. :math:`-\ln\pi(T)`."""
    return terms.gamma_total + jnp.logaddexp(0.0, terms.log_n_total)


def log_extinction_numerator(terms: NestedTerms) -> jax.Array:
    r"""Return :math:`\ln N(T)` with :math:`N(T) = \int_0^T \mu e^{-\Gamma(0,t]}dt`.

    ``LOG_ZERO`` (not ``-inf``) when :math:`N(T) = 0`.
    """
    return terms.log_n_total


def log_nested_w_minus_one(terms: NestedTerms) -> jax.Array:
    r"""Return :math:`\ln(\mathcal W(T) - 1) = \Gamma(0,T] + \ln L(T)`.

    ``LOG_ZERO`` when there are no births.
    """
    return terms.gamma_total + terms.log_l_total


# ---------------------------------------------------------------------------
# Age densities
# ---------------------------------------------------------------------------


def spine_log_atom(terms: NestedTerms) -> jax.Array:
    r"""Log of :math:`\Pr(A^* = T) = \mathcal W(T)\,e^{-\Lambda(0,T]}`."""
    return log_nested_w_total(terms) - terms.lam_total


def _densities(
    terms: NestedTerms,
    ages: jax.Array,
    *,
    want_non_spine: bool,
    want_spine: bool,
    atom_at_boundary: bool,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Shared single pass behind the public density functions."""
    if _is_concrete(ages, terms.times):
        normalize_and_validate_inputs(times=terms.times, ages=ages)  # checks only
    ages = jnp.asarray(ages, dtype=jnp.float64)
    end = terms.times[-1]
    s = jnp.clip(end - ages, 0.0, end)
    row = _row(terms, s)
    single = terms.table.shape[0] == 1
    u = s - row.start
    rem = row.time - s

    non_spine = None
    if want_non_spine:
        log_norm = log_nested_w_minus_one(terms)
        value = (
            row.log_birth
            + _log_w_from_row(row, u, single_epoch=single)
            - (row.death * rem + row.mu_after)
            - log_norm
        )
        non_spine = jnp.where(log_norm < 0.5 * LOG_ZERO, -jnp.inf, value)

    spine = None
    if want_spine:
        partial = (
            row.log_death
            + (terms.gamma_total - row.gamma_end)
            + row.growth * rem
            + _log_kappa_minus(row.growth, rem)
        )
        log_l_mu = restore_neg_inf(
            partial if single else jnp.logaddexp(partial, row.log_spine_after)
        )
        continuous = row.log_birth - (row.birth * rem + row.lam_after) + log_l_mu
        spine = (
            jnp.where(ages >= end, spine_log_atom(terms), continuous)
            if atom_at_boundary
            else continuous
        )
    return non_spine, spine


def non_spine_log_density(terms: NestedTerms, ages: jax.Array) -> jax.Array:
    r"""Log density of the non-spine age :math:`A` at ``ages`` (no atom)."""
    out, _ = _densities(
        terms, ages, want_non_spine=True, want_spine=False, atom_at_boundary=False
    )
    assert out is not None
    return out


def spine_log_density(
    terms: NestedTerms, ages: jax.Array, *, atom_at_boundary: bool = True
) -> jax.Array:
    r"""Log density of :math:`A^*`.

    With ``atom_at_boundary`` (default) ages ``>= T`` return the log atom
    :math:`\Pr(A^*=T)`; otherwise only the continuous density is evaluated.
    """
    _, out = _densities(
        terms,
        ages,
        want_non_spine=False,
        want_spine=True,
        atom_at_boundary=atom_at_boundary,
    )
    assert out is not None
    return out


def age_log_densities(
    terms: NestedTerms, ages: jax.Array, *, atom_at_boundary: bool = True
) -> tuple[jax.Array, jax.Array]:
    """Return ``(non_spine, spine)`` log densities sharing one pass."""
    non_spine, spine = _densities(
        terms,
        ages,
        want_non_spine=True,
        want_spine=True,
        atom_at_boundary=atom_at_boundary,
    )
    assert non_spine is not None
    assert spine is not None
    return non_spine, spine
