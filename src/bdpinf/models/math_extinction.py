"""Extinction probability calculations for birth-death processes.

All functions are pure JAX: no Python branching on tensor values, no in-place
mutation, no .item() calls.  Control-flow hazards from the original torch
implementation are resolved as follows:

- Python-level ``if ... .item()`` guards → ``jnp.where`` masks.
- In-place masked writes → ``jnp.where`` / ``.at[].set()``.
- Reverse recurrence for tail → ``jax.lax.scan`` (reverse).
- ``torch.diff(prepend=...)`` → ``jnp.concatenate([jnp.zeros((1,)), arr])``.

All inputs are assumed float64 (``jax_enable_x64`` is set in __init__.py).
The ``_skip_normalization`` kwarg is accepted for API compatibility but ignored
— JAX functions always expect pre-normalised float64 arrays.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import ensure_1d_tensor

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ATOL_CRITICAL: float = 1e-10


def _epoch_starts(times: jax.Array) -> jax.Array:
    """Return epoch start times given epoch end times."""
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), times[:-1]])


def _safe_div_kernel(g: jax.Array, dur: jax.Array) -> jax.Array:
    """(1 − e^{−g·Δ}) / g with the critical (g≈0) limit → Δ."""
    critical = jnp.abs(g) < _ATOL_CRITICAL
    safe_g = jnp.where(critical, jnp.ones_like(g), g)
    exp_neg = jnp.where(critical, jnp.ones_like(g), jnp.exp(-g * dur))
    return jnp.where(critical, dur, (1.0 - exp_neg) / safe_g)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pr_extinction_numerator(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    _skip_normalization: bool = False,
) -> Float[Array, ""]:
    """Numerator of the extinction probability integral.

    Parameters
    ----------
    birth_rates : jax.Array
        1-D float64 epoch birth rates λ, shape (n_epochs,).
    death_rates : jax.Array
        1-D float64 epoch death rates μ, same shape.
    times : jax.Array
        1-D strictly-increasing float64 epoch end times, same length.

    Returns
    -------
    jax.Array
        Scalar float64.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )

    epoch_durations = jnp.concatenate([times[:1], jnp.diff(times)])
    growth_rates = birth_rates - death_rates
    x = growth_rates * epoch_durations
    critical_mask = jnp.abs(x) < _ATOL_CRITICAL
    cum_growth = jnp.cumsum(x)
    trunc_cum_growth = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float64), cum_growth[:-1]]
    )

    safe_g = jnp.where(critical_mask, jnp.ones_like(growth_rates), growth_rates)
    exp_neg_x = jnp.where(critical_mask, jnp.ones_like(x), jnp.exp(-x))
    kernel = jnp.where(critical_mask, epoch_durations, (1.0 - exp_neg_x) / safe_g)

    summands = death_rates * jnp.exp(-trunc_cum_growth) * kernel
    return summands.sum()


def calculate_extinction_pr(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    return_log_prob: bool = False,
    _skip_normalization: bool = False,
) -> Float[Array, ""]:
    r"""Extinction probability P(Z_T = 0), or its log.

    Parameters
    ----------
    birth_rates, death_rates, times
        As for ``pr_extinction_numerator``.
    return_log_prob : bool
        If True return log P(Z_T = 0). Default False.

    Returns
    -------
    jax.Array
        Scalar float64.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )

    ext_num = pr_extinction_numerator(
        birth_rates, death_rates, times, _skip_normalization=True
    )
    if return_log_prob:
        return jnp.log(ext_num) - jnp.log1p(ext_num)
    return 1.0 - 1.0 / (1.0 + ext_num)


def calculate_log_lik_lifetimes(
    lifetimes: Float[Array, " n"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    as_array: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    r"""Log P(lifetime) = log μ_k − ∫_0^s μ(t) dt, vectorised over lifetimes.

    Uses ``jnp.searchsorted``-based vectorisation instead of a Python loop.

    Parameters
    ----------
    lifetimes : jax.Array
        1-D float64, shape (n,). Each value ≤ times[-1].
    death_rates : jax.Array
        1-D float64 epoch death rates, shape (n_epochs,).
    times : jax.Array
        1-D strictly-increasing float64 epoch end times.
    as_array : bool
        Return per-lifetime contributions instead of summing.

    Returns
    -------
    jax.Array
        Scalar or (n,) float64.

    """
    lifetimes = ensure_1d_tensor(lifetimes)
    if not _skip_normalization:
        death_rates, times = normalize_and_validate_inputs(
            death_rates=death_rates, times=times
        )

    epoch_starts = _epoch_starts(times)
    epoch_durations = times - epoch_starts
    cum_mu_dur = jnp.cumsum(death_rates * epoch_durations)

    epoch_of = jnp.searchsorted(times, lifetimes, side="left")
    epoch_of = jnp.clip(epoch_of, 0, len(times) - 1)

    cum_before = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), cum_mu_dur[:-1]])
    cum_integral = (
        cum_before[epoch_of]
        + (lifetimes - epoch_starts[epoch_of]) * death_rates[epoch_of]
    )

    mu_k = death_rates[epoch_of]

    at_boundary = lifetimes >= times[-1]
    log_liks = jnp.where(
        at_boundary,
        -cum_mu_dur[-1],
        jnp.log(jnp.where(mu_k > 0, mu_k, jnp.ones_like(mu_k))) - cum_integral,
    )

    return log_liks if as_array else log_liks.sum()


def _analytic_log_pr_extinction_per_lifetime(
    lifetimes: jax.Array,
    birth_rates: jax.Array,
    death_rates: jax.Array,
    times: jax.Array,
) -> jax.Array:
    r"""Closed-form log P(offspring extinct | ancestor lifetime s), vectorised.

    The backward recurrence for ``tail`` is expressed as a ``lax.scan``
    (reverse pass) instead of a Python for-loop.

    Parameters
    ----------
    lifetimes : jax.Array
        Shape (n,) float64.
    birth_rates, death_rates, times : jax.Array
        Shape (n_epochs,) float64.

    Returns
    -------
    jax.Array
        Shape (n,) float64, per-lifetime log probabilities.

    """
    n_epochs = len(birth_rates)
    zero = jnp.float64(0.0)

    epoch_starts = _epoch_starts(times)
    dur = times - epoch_starts
    g = birth_rates - death_rates

    critical = jnp.abs(g) < _ATOL_CRITICAL
    safe_g = jnp.where(critical, jnp.ones_like(g), g)
    exp_neg_g_dur = jnp.where(critical, jnp.ones_like(g), jnp.exp(-g * dur))
    kernel = jnp.where(critical, dur, (1.0 - exp_neg_g_dur) / safe_g)

    # Backward recurrence: tail[k] = mu[k]*kernel[k] + exp(-g[k]*dur[k])*tail[k+1]
    # tail[K-1] = 0.  Scan reversed arrays from k=K-2 down to k=0.
    mu_rev = jnp.flip(death_rates)
    kernel_rev = jnp.flip(kernel)
    exp_rev = jnp.flip(exp_neg_g_dur)

    def _backward_step(
        carry: jax.Array,
        xs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        mu_k, ker_k, exp_k = xs
        tail_k = mu_k * ker_k + exp_k * carry
        return tail_k, tail_k

    _, tail_rev = lax.scan(
        _backward_step,
        zero,
        (mu_rev[:-1], kernel_rev[:-1], exp_rev[:-1]),
    )
    tail = jnp.concatenate([jnp.flip(tail_rev), jnp.zeros(1, dtype=jnp.float64)])

    ext_from = death_rates * kernel + exp_neg_g_dur * tail

    per_epoch = jnp.log1p(tail) - jnp.log1p(ext_from) - g * dur
    prefix = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(per_epoch)])

    epoch_of = jnp.searchsorted(times, lifetimes, side="left")
    epoch_of = jnp.clip(epoch_of, 0, n_epochs - 1)

    remaining = times[epoch_of] - lifetimes
    g_lt = g[epoch_of]
    mu_lt = death_rates[epoch_of]
    tail_lt = tail[epoch_of]
    ext_from_lt = ext_from[epoch_of]
    critical_lt = critical[epoch_of]

    safe_g_lt = jnp.where(critical_lt, jnp.ones_like(g_lt), g_lt)
    exp_rem = jnp.where(critical_lt, jnp.ones_like(g_lt), jnp.exp(-g_lt * remaining))
    kernel_rem = jnp.where(critical_lt, remaining, (1.0 - exp_rem) / safe_g_lt)
    ext_num_at_si = mu_lt * kernel_rem + exp_rem * tail_lt

    dur_partial = lifetimes - epoch_starts[epoch_of]
    partial = jnp.log1p(ext_num_at_si) - jnp.log1p(ext_from_lt) - g_lt * dur_partial

    return prefix[epoch_of] + partial


def calculate_log_pr_extinction_given_initial_lifetimes(
    lifetimes: Float[Array, " n"],
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    as_array: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    """Log P(offspring extinct | ancestor lifetimes), summed or per-element.

    Parameters
    ----------
    lifetimes : jax.Array
        Shape (n,) float64, each ≤ times[-1].
    birth_rates, death_rates, times : jax.Array
        Shape (n_epochs,) float64.
    as_array : bool
        If True return per-lifetime values; otherwise sum.

    Returns
    -------
    jax.Array
        Scalar or (n,) float64.

    """
    lifetimes = ensure_1d_tensor(lifetimes)
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )

    log_probs = _analytic_log_pr_extinction_per_lifetime(
        lifetimes, birth_rates, death_rates, times
    )
    return log_probs if as_array else log_probs.sum()


def calculate_log_lik_initial_lifetimes_given_extinction(
    lifetimes: Float[Array, " n"],
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    as_array: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    """Log P(lifetime | extinction), summed or per-element.

    Parameters
    ----------
    lifetimes : jax.Array
        Shape (n,) float64.
    birth_rates, death_rates, times : jax.Array
        Shape (n_epochs,) float64.
    as_array : bool
        Per-element if True, summed if False.

    Returns
    -------
    jax.Array
        Scalar or (n,) float64.

    """
    lifetimes = ensure_1d_tensor(lifetimes)
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )

    log_pr_ext_given_lt = calculate_log_pr_extinction_given_initial_lifetimes(
        lifetimes,
        birth_rates,
        death_rates,
        times,
        as_array=True,
        _skip_normalization=True,
    )
    log_lik_lt = calculate_log_lik_lifetimes(
        lifetimes, death_rates, times, as_array=True, _skip_normalization=True
    )
    log_ext_pr = calculate_extinction_pr(
        birth_rates, death_rates, times, return_log_prob=True, _skip_normalization=True
    )

    per_element = log_pr_ext_given_lt + log_lik_lt - log_ext_pr
    return per_element if as_array else per_element.sum()


def calculate_log_lik_initial_lifetimes_given_non_extinction(
    lifetimes: Float[Array, " n"],
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    as_array: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    r"""Log P(lifetime | non-extinction), summed or per-element.

    Uses ``log(-expm1(log_p_ext))`` = log(1 - P_ext) for numerical stability.

    Parameters
    ----------
    lifetimes : jax.Array
        Shape (n,) float64.
    birth_rates, death_rates, times : jax.Array
        Shape (n_epochs,) float64.
    as_array : bool
        Per-element if True, summed if False.

    Returns
    -------
    jax.Array
        Scalar or (n,) float64.

    """
    lifetimes = ensure_1d_tensor(lifetimes)
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )

    log_pr_ext_given_lt = calculate_log_pr_extinction_given_initial_lifetimes(
        lifetimes,
        birth_rates,
        death_rates,
        times,
        as_array=True,
        _skip_normalization=True,
    )
    log_lik_lt = calculate_log_lik_lifetimes(
        lifetimes, death_rates, times, as_array=True, _skip_normalization=True
    )
    log_ext_pr = calculate_extinction_pr(
        birth_rates, death_rates, times, return_log_prob=True, _skip_normalization=True
    )

    log_pr_non_ext_given_lt = jnp.log(-jnp.expm1(log_pr_ext_given_lt))
    log_pr_non_ext = jnp.log(-jnp.expm1(log_ext_pr))

    per_element = log_pr_non_ext_given_lt + log_lik_lt - log_pr_non_ext
    return per_element if as_array else per_element.sum()


def _integrand_surv_from_weighted_birth_time(
    birth_time: float,
    birth_rates: jax.Array,
    death_rates: jax.Array,
    times: jax.Array,
) -> float:
    """Integrand λ(t)·P_ext(t, T) at a single birth time t.

    Parameters
    ----------
    birth_time : float
        Time at which birth occurred (scalar).
    birth_rates : jax.Array
        Birth rates λ for each epoch.
    death_rates : jax.Array
        Death rates μ for each epoch.
    times : jax.Array
        Epoch end times.

    Returns
    -------
    float
        Integrand value for this birth time.

    """
    birth_rates = ensure_1d_tensor(birth_rates)
    death_rates = ensure_1d_tensor(death_rates)
    times = ensure_1d_tensor(times)
    shifted_times = times - birth_time
    valid_mask = shifted_times > 0
    valid_indices = jnp.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return 0.0
    trunc_times = shifted_times[valid_indices]
    trunc_birth_rates = birth_rates[valid_indices]
    trunc_death_rates = death_rates[valid_indices]
    ext_prob = calculate_extinction_pr(
        birth_rates=trunc_birth_rates,
        death_rates=trunc_death_rates,
        times=trunc_times,
        _skip_normalization=True,
    )
    return float(ext_prob * trunc_birth_rates[0])
