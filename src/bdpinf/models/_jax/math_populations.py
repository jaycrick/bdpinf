"""JAX port of models/math_populations.py — Phase 2a spike.

Control-flow notes vs torch original:

- Scalar-truthiness guards on ``alpha_t`` / ``beta_t`` (``if not (0 <= alpha_t <= 1)``)
  are kept as eager Python checks here — they act as host-side validation, not as
  traced conditions. In the full Phase 2b port they become ``jnp.where``-selected
  ``-inf`` results so the function is ``jit``-safe.
- The single-epoch critical branch
  (``torch.isclose(birth_rate_val, death_rate_val)``) stays as an eager Python
  ``bool`` comparison so the two algebraically-different formulas can be selected
  at trace time (acceptable because the branch depends on static rate values for
  the equivalence benchmark, not on traced arrays).
- In-place ``log_pr[zero_mask] = ...`` → ``jnp.where``.
- ``_compute_populations_log_lik_multi_start``: the ragged ``arange(j_min, j_max+1)``
  per-element loop is re-expressed as a fixed-width ``n0``-bounded loop with
  boolean masking.  For the equivalence gate the initial_population=1 path is the
  primary concern; the multi-start path is also implemented correctly.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import scipy as jsp

from inf_bp_texts.models._jax.math_extinction import (
    calculate_extinction_pr,
    pr_extinction_numerator,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_populations_log_lik_multi_start(
    ns: jax.Array,
    nonzero_mask: jax.Array,
    n0: int,
    alpha_t: jax.Array,
    beta_t: jax.Array,
) -> jax.Array:
    r"""Log probabilities for populations when initial_population > 1.

    Replaces the ragged ``torch.arange(j_min, j_max+1)`` per-element loop with a
    fixed-width ``n0``-length loop masked by validity conditions.

    Parameters
    ----------
    ns : jax.Array
        Shape (m,) float64 population counts.
    nonzero_mask : jax.Array
        Shape (m,) bool, True where ns > 0.
    n0 : int
        Initial population (static).
    alpha_t, beta_t : jax.Array
        Scalar float64 process parameters.

    Returns
    -------
    jax.Array
        Shape (m,) float64 log probabilities; 0.0 at zero-count positions,
        ``-inf`` for invalid entries.

    """
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    log_alpha = jnp.where(alpha_t > 0, jnp.log(alpha_t), _ninf)
    log_beta = jnp.where(beta_t > 0, jnp.log(beta_t), _ninf)
    log_one_minus_alpha = jnp.where(alpha_t < 1, jnp.log1p(-alpha_t), _ninf)
    log_one_minus_beta = jnp.where(beta_t < 1, jnp.log1p(-beta_t), _ninf)
    log_one_minus = log_one_minus_alpha + log_one_minus_beta

    def _log_pr_one(n_val: Any) -> jax.Array:
        """Log P(Z_T = n_val | Z_0 = n0)."""
        n_f = jnp.float64(n_val)
        n0_f = jnp.float64(n0)
        # j ranges from max(0, n0-n_val) to n0-1 — fixed width n0
        j_all = jnp.arange(n0, dtype=jnp.float64)
        j_min = jnp.maximum(0.0, n0_f - n_f)
        j_max = n0_f - 1.0
        valid_j = (j_all >= j_min) & (j_all <= j_max)

        log_binom_n0 = (
            jsp.special.gammaln(n0_f + 1.0)
            - jsp.special.gammaln(j_all + 1.0)
            - jsp.special.gammaln(n0_f - j_all + 1.0)
        )
        log_binom_second = (
            jsp.special.gammaln(n_f)
            - jsp.special.gammaln(n0_f - j_all)
            - jsp.special.gammaln(n_f - n0_f + j_all + 1.0)
        )
        log_alpha_term = j_all * log_alpha
        log_beta_term = (n_f - n0_f + j_all) * log_beta
        log_one_minus_term = (n0_f - j_all) * log_one_minus

        log_terms = (
            log_binom_n0
            + log_binom_second
            + log_alpha_term
            + log_beta_term
            + log_one_minus_term
        )
        # Mask invalid j positions to -inf before logsumexp
        log_terms = jnp.where(valid_j, log_terms, _ninf)
        return jsp.special.logsumexp(log_terms)

    # Apply over each element; nonzero_mask selects where to compute
    log_pr = jnp.zeros_like(ns, dtype=jnp.float64)
    for i in range(len(ns)):
        n_int = int(float(ns[i]))
        val = _log_pr_one(n_int) if bool(nonzero_mask[i]) else jnp.float64(0.0)
        log_pr = log_pr.at[i].set(val)
    return log_pr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def population_geometric_parameter(
    birth_rates: jax.Array,
    death_rates: jax.Array,
    times: jax.Array,
    *,
    _skip_normalization: bool = False,
) -> jax.Array:
    """Geometric parameter 1 − β_T for the offspring survival distribution.

    Parameters
    ----------
    birth_rates, death_rates, times : jax.Array
        Shape (n_epochs,) float64.

    Returns
    -------
    jax.Array
        Scalar float64 in (0, 1).

    """
    epoch_durations = jnp.concatenate([times[:1], jnp.diff(times)])
    growth_rates = birth_rates - death_rates
    cum_growth = jnp.cumsum(growth_rates * epoch_durations)
    geom_sum = pr_extinction_numerator(birth_rates, death_rates, times)
    log_cond = -cum_growth[-1] - jnp.log1p(geom_sum)
    return jnp.exp(log_cond)


def calculate_populations_log_lik(
    birth_rates: jax.Array,
    death_rates: jax.Array,
    ns: jax.Array,
    times: jax.Array,
    *,
    initial_population: int = 1,
    unbias: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    r"""Log P(Z_T = n) summed over provided counts.

    Tavaré (2018) formula. Mirrors ``calculate_populations_log_lik`` from the
    torch module exactly, including the ``initial_population > 1`` multi-start
    path.

    Parameters
    ----------
    birth_rates, death_rates, times : jax.Array
        Shape (n_epochs,) float64.
    ns : jax.Array
        Shape (m,) float64 (or int64-valued float64) population counts.
    initial_population : int
        Z_0. Default 1.
    unbias : bool
        If True and initial_population == 1, condition on non-extinction for
        positive counts (geometric on positive integers with parameter 1-β).

    Returns
    -------
    jax.Array
        Scalar float64.

    """
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)
    _zero = jnp.float64(0.0)

    n_epochs = len(times)

    if n_epochs > 1:
        alpha_t = calculate_extinction_pr(birth_rates, death_rates, times)
        beta_t = 1.0 - population_geometric_parameter(birth_rates, death_rates, times)
    else:
        # Single-epoch: choose formula based on whether rates are (near-)equal
        b = float(birth_rates[0])
        d = float(death_rates[0])
        T = float(times[0])
        if abs(b - d) < 1e-12 * max(abs(b), 1.0):
            avg = (b + d) / 2.0
            rt = avg * T
            alpha_t = beta_t = jnp.float64(rt / (1.0 + rt))
        else:
            g = b - d
            exp_term = jnp.exp(jnp.float64(g * T))
            denom = jnp.float64(b) * exp_term - jnp.float64(d)
            if d == 0.0:
                alpha_t = _zero
                beta_t = jnp.float64(b) * (exp_term - 1.0) / denom
            else:
                alpha_t = jnp.float64(d) * (exp_term - 1.0) / denom
                beta_t = (jnp.float64(b) / jnp.float64(d)) * alpha_t

    # Host-side range check (matching torch behaviour)
    a_f = float(alpha_t)
    b_f = float(beta_t)
    if not (0.0 <= a_f <= 1.0):
        return _ninf
    if not (0.0 <= b_f <= 1.0):
        return _ninf
    if b_f == 0.0 and jnp.any(ns > 1):
        return _ninf

    zero_mask = ns == 0
    nonzero_mask = ~zero_mask

    # Log probability array — filled per case below
    log_pr = jnp.zeros_like(ns, dtype=jnp.float64)

    # n == 0 case
    log_alpha_n0 = jnp.log(alpha_t) * jnp.float64(initial_population)
    log_pr = jnp.where(zero_mask, log_alpha_n0, log_pr)

    # n > 0 case
    if unbias and initial_population == 1:
        log_one_minus_beta = jnp.log1p(-beta_t)
        log_beta = jnp.where(beta_t > 0, jnp.log(beta_t), _ninf)
        log_term3 = jnp.where(ns == 1, _zero, (ns - 1.0) * log_beta)
        nonzero_val = log_one_minus_beta + log_term3
        if not (
            jnp.isfinite(log_one_minus_beta)
            and jnp.all(jnp.isfinite(log_term3[nonzero_mask]))
        ):
            return _ninf
        log_pr = jnp.where(nonzero_mask, nonzero_val, log_pr)

    elif initial_population == 1:
        log_term1 = jnp.log1p(-alpha_t)
        log_term2 = jnp.log1p(-beta_t)
        log_beta = jnp.where(beta_t > 0, jnp.log(beta_t), _ninf)
        log_term3 = jnp.where(ns == 1, _zero, (ns - 1.0) * log_beta)
        if not (jnp.isfinite(log_term1) and jnp.isfinite(log_term2)):
            return _ninf
        nonzero_val = log_term1 + log_term2 + log_term3
        log_pr = jnp.where(nonzero_mask, nonzero_val, log_pr)

    else:
        # Multi-start
        log_pr = _compute_populations_log_lik_multi_start(
            ns=ns,
            nonzero_mask=nonzero_mask,
            n0=initial_population,
            alpha_t=alpha_t,
            beta_t=beta_t,
        )

    if not jnp.all(jnp.isfinite(log_pr)):
        return _ninf

    return log_pr.sum()
