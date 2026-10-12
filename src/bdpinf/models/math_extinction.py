r"""Extinction probability calculations for birth-death processes.

Notation follows the paper (``sections/1_Intro.tex``):
:math:`\Gamma(0,t] = \int_0^t (\lambda - \mu)`,
:math:`N(T) = \int_0^T \mu(t)\, e^{-\Gamma(0,t]}\,dt`,
:math:`\mathcal W(T) = e^{\Gamma(0,T]}(1 + N(T))` and :math:`\pi(T) = 1/\mathcal W(T)`.
The extinction probability is :math:`\Pr(Z_T = 0) = N(T) / (1 + N(T))`
(``eq:pr-ext-gbdp``).
The repo's historical name for :math:`N(T)` is the *extinction numerator*
(``ext_num``); it equals :math:`\mathcal W(T) - 1` only after the rescaling
above, so :math:`\pi(T) = 1/(1 + N(T))` holds only when :math:`\Gamma(0,T] = 0`.

The epoch-wise closed forms live in :mod:`bdpinf.models.math_nested`; this
module exposes the extinction quantities built on them.

All inputs are assumed float64 (``jax_enable_x64`` is set in __init__.py).
The ``_skip_normalization`` kwarg lets callers that have already validated
and normalised their inputs (e.g. the JIT-compiled MLE hot path, the NumPyro
model) skip the redundant ``normalize_and_validate_inputs`` call, which uses
Python-level ``float()``/``bool()`` calls that are unsafe under ``jax.jit``.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bdpinf.models.math_nested import (
    log_extinction_numerator,
    nested_terms,
    restore_neg_inf,
)
from bdpinf.models.validation import normalize_and_validate_inputs
from bdpinf.utils import ensure_1d_tensor

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_pr_extinction_numerator(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    _skip_normalization: bool = False,
) -> Float[Array, ""]:
    r"""Return :math:`\ln N(T)`, ``-inf`` when :math:`N(T) = 0` (no deaths).

    Parameters
    ----------
    birth_rates, death_rates, times : jax.Array
        Shape (n_epochs,) float64; ``times`` strictly increasing.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )
    return restore_neg_inf(
        log_extinction_numerator(nested_terms(birth_rates, death_rates, times))
    )


def pr_extinction_numerator(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    _skip_normalization: bool = False,
) -> Float[Array, ""]:
    r"""Extinction numerator :math:`N(T) = \int_0^T \mu(t) e^{-\Gamma(0,t]}\,dt`.

    Parameters
    ----------
    birth_rates : jax.Array
        1-D float64 epoch birth rates :math:`\lambda`, shape (n_epochs,).
    death_rates : jax.Array
        1-D float64 epoch death rates :math:`\mu`, same shape.
    times : jax.Array
        1-D strictly-increasing float64 epoch end times, same length.

    Returns
    -------
    jax.Array
        Scalar float64.

    """
    return jnp.exp(
        log_pr_extinction_numerator(
            birth_rates, death_rates, times, _skip_normalization=_skip_normalization
        )
    )


def calculate_extinction_pr(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    return_log_prob: bool = False,
    _skip_normalization: bool = False,
) -> Float[Array, ""]:
    r"""Extinction probability :math:`\Pr(Z_T = 0) = N/(1+N)`, or its log.

    Evaluated as :math:`\ln N - \operatorname{softplus}(\ln N)`, which stays
    accurate when :math:`N` is tiny.

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
    log_n = log_pr_extinction_numerator(
        birth_rates, death_rates, times, _skip_normalization=_skip_normalization
    )
    log_pr = restore_neg_inf(log_n - jnp.logaddexp(0.0, log_n))
    return log_pr if return_log_prob else jnp.exp(log_pr)


def _integrand_surv_from_weighted_birth_time(  # pyright: ignore[reportUnusedFunction]
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
