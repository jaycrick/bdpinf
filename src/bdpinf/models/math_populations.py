r"""Population log-likelihood calculations for birth-death processes.

Tavaré (2018) formula for P(Z_T = n | Z_0 = n0), written in the paper's
notation (``sections/1_Intro.tex``, ``eq:pop-dist``): from one ancestor,
$\Pr(Z_T = 0) = \alpha = N/(1+N)$ and, given survival,
$Z_T$ is geometric with success probability $\pi(T) = 1/\mathcal W(T)$,
i.e. $\Pr(Z_T = z \mid Z_T \ge 1) = \pi(1-\pi)^{z-1}$.
With $\beta = 1 - \pi$ the code uses
$\ln\alpha = \ln N - \mathrm{softplus}(\ln N)$,
$\ln(1-\alpha) = -\mathrm{softplus}(\ln N)$,
$\ln(1-\beta) = \ln\pi = -\ln\mathcal W(T)$ and
$\ln\beta = \ln(\mathcal W(T) - 1) - \ln\mathcal W(T)$,
all from :mod:`bdpinf.models.math_nested`, so nothing is recovered through
``log1p(-beta)`` round trips.

Control-flow notes vs the Phase 2a spike:

- Single-epoch critical/non-critical branch → merged via ``jnp.where`` so the
  function is ``jit``- and ``grad``-safe.  Both branches are always computed and
  the result is selected element-wise.
- Host-side range guards on ``alpha_t`` / ``beta_t`` → replaced by
  ``jnp.where``-selected ``-inf``.
- Finiteness guards → ``jnp.where``-selected ``-inf``.
- The multi-start path (``initial_population > 1``) vectorizes over ``ns`` with
  ``jax.vmap`` (previously a Python loop with per-element ``int()``/``bool()``
  casts, which is not ``jit``-traceable); the inner sum over the fixed static
  bound ``n0`` stays a ``jnp.arange(n0)`` vector op, JAX-compatible because
  ``n0`` is a compile-time constant.

All inputs are assumed float64 (``jax_enable_x64`` is set in ``__init__.py``).
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import scipy as jsp
from jaxtyping import Array, Bool, Float, jaxtyped

from bdpinf.models.math_nested import (
    log_extinction_numerator,
    log_nested_w_minus_one,
    log_nested_w_total,
    nested_terms,
    restore_neg_inf,
)
from bdpinf.models.validation import normalize_and_validate_inputs
from bdpinf.utils import ensure_1d_tensor

# ---------------------------------------------------------------------------
# Internal helpers
#
# _compute_populations_log_lik_multi_start is only ever called with already-
# normalised jax.Array values (never the public API's loose scalar/list
# convenience inputs), so its jaxtyping shape is runtime-enforced via
# beartype -- see the equivalent note in math_extinction.py.
# ---------------------------------------------------------------------------


def _safe_mul(count: jax.Array, log_value: jax.Array) -> jax.Array:
    """``count * log_value``, with ``0 * -inf = 0`` for a zero exponent."""
    return jnp.where(count == 0, 0.0, count * log_value)


def _population_logs(
    birth_rates: jax.Array, death_rates: jax.Array, times: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Return $(\ln\alpha, \ln(1-\alpha), \ln\beta, \ln\pi)$ at ``T``."""
    terms = nested_terms(birth_rates, death_rates, times)
    log_n = log_extinction_numerator(terms)
    softplus_n = jnp.logaddexp(0.0, log_n)
    log_w_total = log_nested_w_total(terms)
    log_alpha = restore_neg_inf(log_n - softplus_n)
    log_one_minus_alpha = -softplus_n
    log_beta = restore_neg_inf(log_nested_w_minus_one(terms) - log_w_total)
    return log_alpha, log_one_minus_alpha, log_beta, -log_w_total


@jaxtyped(typechecker=beartype)
def _compute_populations_log_lik_multi_start(
    ns: Float[Array, " n"],
    nonzero_mask: Bool[Array, " n"],
    n0: int,
    log_alpha: Float[Array, ""],
    log_beta: Float[Array, ""],
    log_one_minus: Float[Array, ""],
) -> Float[Array, " n"]:
    r"""Log probabilities for populations when initial_population > 1.

    Replaces the ragged ``torch.arange(j_min, j_max+1)`` per-element loop with
    a fixed-width ``n0``-length sum masked by validity conditions, vectorized
    over ``ns`` via ``jax.vmap`` so the whole function is ``jit``-traceable.

    Parameters
    ----------
    ns : jax.Array
        Shape (m,) float64 population counts.
    nonzero_mask : jax.Array
        Shape (m,) bool, True where ns > 0.
    n0 : int
        Initial population (static compile-time constant).
    log_alpha, log_beta : jax.Array
        Scalar float64 $\ln\alpha$ and $\ln\beta$.
    log_one_minus : jax.Array
        Scalar float64 $\ln(1-\alpha) + \ln(1-\beta)$.

    Returns
    -------
    jax.Array
        Shape (m,) float64 log probabilities.

    """
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    def _log_pr_one(n_val: jax.Array) -> jax.Array:
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
        log_alpha_term = _safe_mul(j_all, log_alpha)
        log_beta_term = _safe_mul(n_f - n0_f + j_all, log_beta)
        log_one_minus_term = _safe_mul(n0_f - j_all, log_one_minus)

        log_terms = (
            log_binom_n0
            + log_binom_second
            + log_alpha_term
            + log_beta_term
            + log_one_minus_term
        )
        log_terms = jnp.where(valid_j, log_terms, _ninf)
        return jsp.special.logsumexp(log_terms)

    vals = jax.vmap(_log_pr_one)(ns)
    return jnp.where(nonzero_mask, vals, jnp.float64(0.0))


# ---------------------------------------------------------------------------
# Private vectorized PMF helper
# ---------------------------------------------------------------------------


def calculate_populations_log_pmf(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    ns: Float[Array, " n"],
    times: Float[Array, " epochs"],
    *,
    initial_population: int = 1,
    _skip_normalization: bool = False,
) -> Float[Array, " n"]:
    r"""Per-element log P(Z_T = ns[i]) for each i.

    Same math as ``calculate_populations_log_lik`` but returns the shape-(m,)
    per-element log-probability array rather than their sum.  JIT- and vmap-safe
    with ``_skip_normalization=True``.

    Parameters
    ----------
    birth_rates, death_rates, times : jax.Array
        Shape (n_epochs,) float64.
    ns : jax.Array
        Shape (m,) float64 population counts to evaluate.
    initial_population : int
        Z_0. Default 1.

    Returns
    -------
    jax.Array
        Shape (m,) float64 log probabilities.

    """
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)
    _zero = jnp.float64(0.0)

    ns = ensure_1d_tensor(ns)

    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )

    log_alpha, log_one_minus_alpha, log_beta, log_pi = _population_logs(
        birth_rates, death_rates, times
    )

    zero_mask = ns == 0
    nonzero_mask = ~zero_mask

    if initial_population == 1:
        log_term3 = jnp.where(ns == 1, _zero, _safe_mul(ns - 1.0, log_beta))
        nonzero_val = log_one_minus_alpha + log_pi + log_term3
        log_pr = jnp.where(zero_mask, log_alpha, nonzero_val)
    else:
        log_pr = _compute_populations_log_lik_multi_start(
            ns=ns,
            nonzero_mask=nonzero_mask,
            n0=initial_population,
            log_alpha=log_alpha,
            log_beta=log_beta,
            log_one_minus=log_one_minus_alpha + log_pi,
        )
        log_pr = jnp.where(
            zero_mask,
            jnp.float64(initial_population) * log_alpha,
            log_pr,
        )

    return jnp.where(jnp.isnan(log_pr), _ninf, log_pr)


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
    r"""Geometric parameter $\pi(T) = 1/\mathcal W(T)$ of $Z_T$.

    Given survival, $Z_T$ is geometric on $\{1, 2, \dots\}$.

    Parameters
    ----------
    birth_rates, death_rates, times : jax.Array
        Shape (n_epochs,) float64.

    Returns
    -------
    jax.Array
        Scalar float64 in (0, 1].

    """
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )
    return jnp.exp(-log_nested_w_total(nested_terms(birth_rates, death_rates, times)))


def calculate_populations_log_lik(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    ns: Float[Array, " n"],
    times: Float[Array, " epochs"],
    *,
    initial_population: int = 1,
    condition_on_survival: bool = False,
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
    condition_on_survival : bool
        If True and initial_population == 1, condition on non-extinction for
        positive counts: $\Pr(Z_T = z \mid Z_T \ge 1) = \pi(1-\pi)^{z-1}$.

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

    if not condition_on_survival:
        log_pr = calculate_populations_log_pmf(
            birth_rates,
            death_rates,
            ns,
            times,
            initial_population=initial_population,
            _skip_normalization=True,
        )
        return jnp.where(jnp.all(jnp.isfinite(log_pr)), log_pr.sum(), _ninf)

    # condition_on_survival=True path (initial_population==1 only):
    # condition on non-extinction.
    log_alpha, _, log_beta, log_pi = _population_logs(birth_rates, death_rates, times)
    zero_mask = ns == 0
    log_term3 = jnp.where(ns == 1, _zero, _safe_mul(ns - 1.0, log_beta))
    log_pr = jnp.where(
        zero_mask,
        jnp.float64(initial_population) * log_alpha,
        log_pi + log_term3,
    )
    return jnp.where(jnp.all(jnp.isfinite(log_pr)), log_pr.sum(), _ninf)
