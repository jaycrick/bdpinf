r"""Exact epoch-wise nested integral $\mathcal W$ and the age densities it gives.

This is the piecewise-constant backend for the general-rate results of the paper
[paper: Theorems ?].
For rates that are constant on each epoch $(\tau_{k-1}, \tau_k]$ every
quantity below has a closed form, evaluated here in O(K) once and then in
O(log K) per age, entirely in log space and without subtraction.

Notation (paper symbol, code name)
----------------------------------
| Paper symbol | Code name |
| --- | --- |
| $T = \tau_K$ (observation time) | ``times[-1]`` |
| $\lambda_k, \mu_k$ | ``birth_rates``, ``death_rates`` |
| $\gamma_k = \lambda_k - \mu_k$ | ``growth`` |
| $d_k = \tau_k - \tau_{k-1}$ | ``durations`` |
| $\Gamma(0, \tau_k]$ | ``gamma_end`` (``gamma_before``: k-1) |
| $\Lambda(s, T], M(s, T]$ | ``lam_after``/``mu_after`` plus a partial |
| $I(b)=\int_0^b \mu e^{-\Gamma(0,t]}$ (extinction odds at $b = T$) | ``log_i_*`` |
| $L(b)=\int_0^b \lambda e^{-\Gamma(0,t]}$ | ``log_l_*`` |
| $\mathcal W(b)$ | `log_nested_w` |
| $\pi(b) = 1/\mathcal W(b)$ | ``exp(-log_nested_w(...))`` |
| $L_\mu(s,T)$ | ``log_spine_after`` plus a partial epoch |
| $A$ (non-spine age) | `non_spine_log_density` |
| $A^*$ (spine age) | `spine_log_density`, atom at ``T`` |

Key relations (all exact):

- $\mathcal W(b) = e^{\Gamma(0,b]}\,(1 + I(b))$
  and $\mathcal W(b) - 1 = e^{\Gamma(0,b]} L(b)$.
- $I(T) = P_0(T)/(1 - P_0(T))$ is the odds of extinction by $T$,
  so $\Pr(Z_T = 0) = I/(1+I)$, $1 + I(T) = 1/\Pr(Z_T \ge 1)$ and
  $\mathcal W(T) = \mathbb E[Z_T]/\Pr(Z_T \ge 1) = \mathbb E[Z_T \mid Z_T \ge 1]$.
  Given survival, $\Pr(Z_T = z \mid Z_T \ge 1) = \pi(1-\pi)^{z-1}$.
- Per epoch, with $\kappa^-(g, d) = (1 - e^{-gd})/g$ (equal to ``d`` at
  $g = 0$, so the critical case needs no special treatment),
  $I$ and $L$ accumulate $w_k e^{-\Gamma(0,\tau_{k-1}]}
  \kappa^-(\gamma_k, d_k)$ with $w = \mu$ and $w = \lambda$.
- Non-spine age: $f(a) = \lambda(T-a)\,\mathcal W(T-a)\,e^{-M(T-a,T]}
  / (\mathcal W(T) - 1)$.
- Spine age: $f^*(a) = \lambda(T-a)\,e^{-\Lambda(T-a,T]}\,L_\mu(T-a,T)$,
  with atom $\Pr(A^*=T) = \mathcal W(T)\,e^{-\Lambda(0,T]}$.

Zero rates are represented internally by the finite log-zero ``LOG_ZERO`` so
that cumulative log-sums and their gradients stay finite;
public results are mapped back to ``-inf``.
"""

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from bdpinf.models.rate_functions import epoch_index
from bdpinf.models.validation import is_concrete, normalize_and_validate_inputs

__all__ = [
    "LOG_ZERO",
    "NestedTerms",
    "Totals",
    "age_log_densities",
    "log_extinction_and_survival_pr",
    "log_extinction_odds",
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


class Totals(Protocol):
    """The whole-window scalars shared by exact and quadrature terms."""

    @property
    def gamma_total(self) -> jax.Array:
        r"""$\Gamma(0, T]$."""
        ...

    @property
    def log_i_total(self) -> jax.Array:
        r"""$\ln I(T)$, with ``LOG_ZERO`` for zero."""
        ...

    @property
    def log_l_total(self) -> jax.Array:
        r"""$\ln L(T)$, with ``LOG_ZERO`` for zero."""
        ...

    @property
    def lam_total(self) -> jax.Array:
        r"""$\Lambda(0, T]$."""
        ...


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
    log_i_before: jax.Array
    log_i_total: jax.Array
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
    "log_i_before",
    "log_spine_after",
    "lam_after",
    "mu_after",
    "log_birth",
    "log_death",
)


class EpochRow(NamedTuple):
    """Per-age view of the epoch table (arrays shaped like the ages)."""

    birth: jax.Array
    death: jax.Array
    time: jax.Array
    start: jax.Array
    growth: jax.Array
    gamma_before: jax.Array
    gamma_end: jax.Array
    log_i_before: jax.Array
    log_spine_after: jax.Array
    lam_after: jax.Array
    mu_after: jax.Array
    log_birth: jax.Array  # -inf where the rate is zero
    log_death: jax.Array  # LOG_ZERO where the rate is zero


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def log_or_floor(x: jax.Array) -> jax.Array:
    """``log x`` with the finite floor ``LOG_ZERO`` at ``x <= 0`` (gradient-safe)."""
    positive = x > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), LOG_ZERO)


def _log_or_neg_inf(x: jax.Array) -> jax.Array:
    """``log x`` with ``-inf`` at ``x <= 0`` (gradient-safe)."""
    positive = x > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), -jnp.inf)


def restore_neg_inf(x: jax.Array) -> jax.Array:
    """Map the internal ``LOG_ZERO`` floor back to ``-inf``."""
    return jnp.where(x < 0.5 * LOG_ZERO, -jnp.inf, x)


def log_kappa_minus(growth: jax.Array, duration: jax.Array) -> jax.Array:
    r"""Return $\ln\kappa^-(g, d)$ with $\kappa^- = (1-e^{-gd})/g$.

    Writing $\kappa^- = d\,(1-e^{-x})/x$ with $x = gd$:
    for $|x|$ below ``_SERIES_THRESHOLD`` (including exactly critical
    epochs) a short polynomial for $(1-e^{-x})/x$ is used, otherwise
    $\ln\bigl[(1-e^{-|x|})/|x|\bigr] + \max(-x, 0)$, which is stable for
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


def _shift_right(x: jax.Array, fill: float) -> jax.Array:
    return jnp.concatenate([jnp.full((1,), fill, dtype=x.dtype), x[:-1]])


def _shift_left(x: jax.Array, fill: float) -> jax.Array:
    return jnp.concatenate([x[1:], jnp.full((1,), fill, dtype=x.dtype)])


# ---------------------------------------------------------------------------
# Epoch precomputation
# ---------------------------------------------------------------------------


def nested_terms(
    birth_rates: jax.Array | float,
    death_rates: jax.Array | float,
    times: jax.Array | float,
) -> NestedTerms:
    r"""Precompute the epoch-wise quantities behind $\mathcal W$.

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
    if is_concrete(birth_rates, death_rates, times):
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
    log_kappa = log_kappa_minus(growth, durations)
    log_mu = log_or_floor(death_rates)
    log_lam = log_or_floor(birth_rates)

    i_terms = log_mu - gamma_before + log_kappa
    l_terms = log_lam - gamma_before + log_kappa
    # kappa^+ = e^{gd} kappa^-: contribution of epoch j to L_mu(tau_k, T) for k < j.
    spine_terms = log_mu + (gamma_total - gamma_end) + x + log_kappa
    lam_dur = birth_rates * durations
    mu_dur = death_rates * durations
    if times.shape[0] == 1:  # static: nothing to accumulate over a single epoch
        log_i_cum, log_l_cum = i_terms, l_terms
        log_i_before = jnp.full_like(i_terms, LOG_ZERO)
        log_spine_after = jnp.full_like(i_terms, LOG_ZERO)
        lam_after = jnp.zeros_like(i_terms)
        mu_after = jnp.zeros_like(i_terms)
    else:
        log_i_cum = jax.lax.cumlogsumexp(i_terms)
        log_l_cum = jax.lax.cumlogsumexp(l_terms)
        log_i_before = _shift_right(log_i_cum, LOG_ZERO)
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
            log_i_before,
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
        log_i_before=log_i_before,
        log_i_total=log_i_cum[-1],
        log_l_total=log_l_cum[-1],
        log_spine_after=log_spine_after,
        lam_after=lam_after,
        mu_after=mu_after,
        lam_total=jnp.sum(lam_dur),
        table=table,
    )


def epoch_row(terms: NestedTerms, s: jax.Array) -> EpochRow:
    """Epoch quantities for each time ``s`` (no gather at all for one epoch)."""
    if terms.table.shape[0] == 1:
        row = terms.table[0]
        return EpochRow(*(row[i] for i in range(len(_COLUMNS))))
    k = epoch_index(terms.times, s)
    rows = jnp.take(terms.table, k, axis=0)
    return EpochRow(*(rows[..., i] for i in range(len(_COLUMNS))))


# ---------------------------------------------------------------------------
# W, I and related scalars
# ---------------------------------------------------------------------------


def _log_w_from_row(row: EpochRow, u: jax.Array, *, single_epoch: bool) -> jax.Array:
    """``log W`` at ``start + u`` inside the epoch described by ``row``."""
    partial = row.log_death - row.gamma_before + log_kappa_minus(row.growth, u)
    log_i_b = partial if single_epoch else jnp.logaddexp(row.log_i_before, partial)
    return row.gamma_before + row.growth * u + jnp.logaddexp(0.0, log_i_b)


def log_nested_w(terms: NestedTerms, b: jax.Array) -> jax.Array:
    r"""Return $\ln\mathcal W(b)$ at times ``b`` (clipped to ``[0, T]``)."""
    b = jnp.clip(jnp.asarray(b, dtype=jnp.float64), 0.0, terms.times[-1])
    row = epoch_row(terms, b)
    return _log_w_from_row(row, b - row.start, single_epoch=terms.table.shape[0] == 1)


def log_nested_w_total(terms: Totals) -> jax.Array:
    r"""Return $\ln\mathcal W(T)$, i.e. $-\ln\pi(T)$."""
    return terms.gamma_total + jnp.logaddexp(0.0, terms.log_i_total)


def log_extinction_odds(terms: Totals) -> jax.Array:
    r"""Return $\ln I(T)$ with $I(T) = \int_0^T \mu e^{-\Gamma(0,t]}dt$.

    $I(T) = P_0(T)/(1 - P_0(T))$ is the odds of extinction by $T$,
    so $\ln I = \operatorname{logit} P_0$.
    ``LOG_ZERO`` (not ``-inf``) when $I(T) = 0$.
    """
    return terms.log_i_total


def log_extinction_and_survival_pr(terms: Totals) -> tuple[jax.Array, jax.Array]:
    r"""Return $(\ln P_0(T), \ln(1 - P_0(T)))$ from the extinction odds.

    Evaluated as $\ln I - \operatorname{softplus}(\ln I)$ and
    $-\operatorname{softplus}(\ln I)$, which stay accurate when $I$ is tiny
    or huge.
    """
    log_i = log_extinction_odds(terms)
    softplus_i = jnp.logaddexp(0.0, log_i)
    return restore_neg_inf(log_i - softplus_i), -softplus_i


def log_nested_w_minus_one(terms: Totals) -> jax.Array:
    r"""Return $\ln(\mathcal W(T) - 1) = \Gamma(0,T] + \ln L(T)$.

    ``LOG_ZERO`` when there are no births.
    """
    return terms.gamma_total + terms.log_l_total


# ---------------------------------------------------------------------------
# Age densities
# ---------------------------------------------------------------------------


def spine_log_atom(terms: Totals) -> jax.Array:
    r"""Log of $\Pr(A^* = T) = \mathcal W(T)\,e^{-\Lambda(0,T]}$."""
    return log_nested_w_total(terms) - terms.lam_total


class _Pass(NamedTuple):
    """Per-age quantities shared by the non-spine and spine densities."""

    ages: jax.Array
    end: jax.Array
    row: EpochRow
    single: bool
    u: jax.Array  # time since the start of the epoch containing T - age
    rem: jax.Array  # time from T - age to the end of that epoch


def _prepare(terms: NestedTerms, ages: jax.Array) -> _Pass:
    """Validate concrete ``ages`` and gather the epoch row of each ``T - age``."""
    if is_concrete(ages, terms.times):
        normalize_and_validate_inputs(times=terms.times, ages=ages)  # checks only
    ages = jnp.asarray(ages, dtype=jnp.float64)
    end = terms.times[-1]
    s = jnp.clip(end - ages, 0.0, end)
    row = epoch_row(terms, s)
    return _Pass(ages, end, row, terms.table.shape[0] == 1, s - row.start, row.time - s)


def _non_spine(terms: NestedTerms, p: _Pass) -> jax.Array:
    row = p.row
    log_norm = log_nested_w_minus_one(terms)
    value = (
        row.log_birth
        + _log_w_from_row(row, p.u, single_epoch=p.single)
        - (row.death * p.rem + row.mu_after)
        - log_norm
    )
    return jnp.where(log_norm < 0.5 * LOG_ZERO, -jnp.inf, value)


def _spine(terms: NestedTerms, p: _Pass, *, atom_at_boundary: bool) -> jax.Array:
    row, rem = p.row, p.rem
    partial = (
        row.log_death
        + (terms.gamma_total - row.gamma_end)
        + row.growth * rem
        + log_kappa_minus(row.growth, rem)
    )
    log_l_mu = restore_neg_inf(
        partial if p.single else jnp.logaddexp(partial, row.log_spine_after)
    )
    continuous = row.log_birth - (row.birth * rem + row.lam_after) + log_l_mu
    if atom_at_boundary:
        return jnp.where(p.ages >= p.end, spine_log_atom(terms), continuous)
    return continuous


def non_spine_log_density(terms: NestedTerms, ages: jax.Array) -> jax.Array:
    r"""Log density of the non-spine age $A$ at ``ages`` (no atom)."""
    return _non_spine(terms, _prepare(terms, ages))


def spine_log_density(
    terms: NestedTerms, ages: jax.Array, *, atom_at_boundary: bool = True
) -> jax.Array:
    r"""Log density of $A^*$.

    With ``atom_at_boundary`` (default) ages ``>= T`` return the log atom
    $\Pr(A^*=T)$; otherwise only the continuous density is evaluated.
    """
    return _spine(terms, _prepare(terms, ages), atom_at_boundary=atom_at_boundary)


def age_log_densities(
    terms: NestedTerms, ages: jax.Array, *, atom_at_boundary: bool = True
) -> tuple[jax.Array, jax.Array]:
    """Return ``(non_spine, spine)`` log densities sharing one pass."""
    p = _prepare(terms, ages)
    return _non_spine(terms, p), _spine(terms, p, atom_at_boundary=atom_at_boundary)
