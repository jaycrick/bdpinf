"""Population log-likelihood calculations for birth-death processes.

Tavaré (2018) formula for P(Z_T = n | Z_0 = n0).

Control-flow notes vs the Phase 2a spike:

- Single-epoch critical/non-critical branch → merged via ``jnp.where`` so the
  function is ``jit``- and ``grad``-safe.  Both branches are always computed and
  the result is selected element-wise.
- Host-side range guards on ``alpha_t`` / ``beta_t`` → replaced by
  ``jnp.where``-selected ``-inf``.
- Finiteness guards → ``jnp.where``-selected ``-inf``.
- The multi-start path (``initial_population > 1``) still uses a Python loop over
  the fixed static bound ``n0``; this is JAX-compatible because ``n0`` is a
  compile-time constant.

All inputs are assumed float64 (``jax_enable_x64`` is set in ``__init__.py``).
"""

import jax
import jax.numpy as jnp
from jax import scipy as jsp
from jaxtyping import Array, Float

from inf_bp_texts.models.math_extinction import (
    calculate_extinction_pr,
    pr_extinction_numerator,
)
from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import ensure_1d_tensor

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

    Replaces the ragged ``torch.arange(j_min, j_max+1)`` per-element loop with
    a fixed-width ``n0``-length loop masked by validity conditions.

    Parameters
    ----------
    ns : jax.Array
        Shape (m,) float64 population counts.
    nonzero_mask : jax.Array
        Shape (m,) bool, True where ns > 0.
    n0 : int
        Initial population (static compile-time constant).
    alpha_t, beta_t : jax.Array
        Scalar float64 process parameters.

    Returns
    -------
    jax.Array
        Shape (m,) float64 log probabilities.

    """
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    log_alpha = jnp.where(alpha_t > 0, jnp.log(alpha_t), _ninf)
    log_beta = jnp.where(beta_t > 0, jnp.log(beta_t), _ninf)
    log_one_minus_alpha = jnp.where(alpha_t < 1, jnp.log1p(-alpha_t), _ninf)
    log_one_minus_beta = jnp.where(beta_t < 1, jnp.log1p(-beta_t), _ninf)
    log_one_minus = log_one_minus_alpha + log_one_minus_beta

    def _log_pr_one(n_val: int) -> jax.Array:
        """Log P(Z_T = n_val | Z_0 = n0)."""
        n_f = jnp.float64(n_val)
        n0_f = jnp.float64(n0)
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
        log_terms = jnp.where(valid_j, log_terms, _ninf)
        return jsp.special.logsumexp(log_terms)

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
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    _skip_normalization: bool = False,
) -> Float[Array, ""]:
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
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )

    epoch_durations = jnp.concatenate([times[:1], jnp.diff(times)])
    growth_rates = birth_rates - death_rates
    cum_growth = jnp.cumsum(growth_rates * epoch_durations)
    geom_sum = pr_extinction_numerator(
        birth_rates, death_rates, times, _skip_normalization=True
    )
    log_cond = -cum_growth[-1] - jnp.log1p(geom_sum)
    return jnp.exp(log_cond)


def calculate_populations_log_lik(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    ns: Float[Array, " n"],
    times: Float[Array, " epochs"],
    *,
    initial_population: int = 1,
    unbias: bool = False,
    _skip_normalization: bool = False,
) -> Float[Array, ""]:
    r"""Log P(Z_T = n) summed over provided counts.

    Tavaré (2018) formula. Supports single- and multi-epoch models.

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

    ns = ensure_1d_tensor(ns)

    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )

    # Unified JAX-native formulas (grad-safe, works for single and multi-epoch)
    alpha_t = calculate_extinction_pr(
        birth_rates, death_rates, times, _skip_normalization=True
    )
    beta_t = 1.0 - population_geometric_parameter(
        birth_rates, death_rates, times, _skip_normalization=True
    )

    # Range validation — functional path, returns -inf
    valid_params = (
        (alpha_t >= 0.0) & (alpha_t <= 1.0) & (beta_t >= 0.0) & (beta_t <= 1.0)
    )
    beta_zero_invalid = (beta_t <= 0) & jnp.any(ns > 1)

    zero_mask = ns == 0
    nonzero_mask = ~zero_mask

    log_pr = jnp.zeros_like(ns, dtype=jnp.float64)

    # n == 0: log alpha^n0
    safe_alpha = jnp.where(alpha_t > 0, alpha_t, jnp.ones_like(alpha_t))
    log_alpha_n0 = jnp.log(safe_alpha) * jnp.float64(initial_population)
    log_pr = jnp.where(
        zero_mask,
        jnp.where(alpha_t > 0, log_alpha_n0, _ninf),
        log_pr,
    )

    if unbias and initial_population == 1:
        safe_beta_lt1 = jnp.where(beta_t < 1, beta_t, jnp.zeros_like(beta_t))
        log_one_minus_beta = jnp.log1p(-safe_beta_lt1)
        log_beta = jnp.where(beta_t > 0, jnp.log(beta_t), _ninf)
        log_term3 = jnp.where(ns == 1, _zero, (ns - 1.0) * log_beta)
        nonzero_val = log_one_minus_beta + log_term3
        # beta_t < 1 required: safe_beta_lt1 zeroes out log1p(-beta) when beta=1
        finite_nonzero = (
            (beta_t < 1)
            & jnp.isfinite(log_one_minus_beta)
            & jnp.all(jnp.where(nonzero_mask, jnp.isfinite(log_term3), True))
        )
        log_pr = jnp.where(
            nonzero_mask,
            jnp.where(finite_nonzero, nonzero_val, _ninf),
            log_pr,
        )

    elif initial_population == 1:
        safe_alpha_lt1 = jnp.where(alpha_t < 1, alpha_t, jnp.zeros_like(alpha_t))
        safe_beta_lt1 = jnp.where(beta_t < 1, beta_t, jnp.zeros_like(beta_t))
        log_term1 = jnp.log1p(-safe_alpha_lt1)
        log_term2 = jnp.log1p(-safe_beta_lt1)
        log_beta = jnp.where(beta_t > 0, jnp.log(beta_t), _ninf)
        log_term3 = jnp.where(ns == 1, _zero, (ns - 1.0) * log_beta)
        # alpha_t < 1 and beta_t < 1 required: the safe_*_lt1 trick zeroes out
        # the argument when the rate hits its boundary, making log1p return 0
        # instead of -inf; the explicit bounds guard enforces the correct -inf.
        finite_12 = (
            (alpha_t < 1)
            & (beta_t < 1)
            & jnp.isfinite(log_term1)
            & jnp.isfinite(log_term2)
        )
        nonzero_val = log_term1 + log_term2 + log_term3
        log_pr = jnp.where(
            nonzero_mask,
            jnp.where(finite_12, nonzero_val, _ninf),
            log_pr,
        )

    else:
        log_pr = _compute_populations_log_lik_multi_start(
            ns=ns,
            nonzero_mask=nonzero_mask,
            n0=initial_population,
            alpha_t=alpha_t,
            beta_t=beta_t,
        )

    all_finite = jnp.all(jnp.isfinite(log_pr))
    return jnp.where(
        all_finite & valid_params & ~beta_zero_invalid, log_pr.sum(), _ninf
    )
