"""JAX port of inference/likelihood.py — Phase 2a spike.

The host-side control flow (how many ages equal times[-1], spine extraction,
ns construction) is kept as eager Python since these decisions depend on *data
values* not on the traced rate parameters.  The numerically-sensitive
differentiable core (populations log-lik + spine/non-spine age log-liks +
the logsumexp unknown-spine marginal) is expressed purely in JAX.

Note: ``log_scale_params=True`` is supported via ``jnp.exp`` before entering
the math layer — this keeps the differentiable path clean.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from inf_bp_texts.models._jax.math_ages import (
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)
from inf_bp_texts.models._jax.math_populations import calculate_populations_log_lik


def _to_f64(x: object) -> jnp.ndarray:
    """Convert scalar / list / numpy array to 1-D float64 JAX array."""
    return jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))


def log_likelihood(
    birth_rates: object,
    death_rates: object,
    ages: object,
    times: object,
    *,
    spine_ages: object | None = None,
    ns: object | None = None,
    initial_population: int = 1,
    unbias: bool = False,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> jnp.ndarray:
    """Compute log-likelihood of observed manuscript ages — JAX implementation.

    Drop-in equivalent to ``inference.likelihood.log_likelihood``; all
    semantics (unknown-spine marginal, ordered flag, ns handling) are
    preserved.

    Parameters
    ----------
    birth_rates, death_rates : array-like
        Epoch birth/death rates (or their logs if ``log_scale_params=True``).
    ages : array-like
        Observed manuscript ages, all in [0, times[-1]].
    times : array-like
        Epoch end times, strictly increasing.
    spine_ages : array-like or None
        Explicitly known spine ages. Mutually exclusive with ``is_ordered``.
    ns : array-like or None
        Population counts. Inferred as ``[len(ages)]`` if None.
    initial_population : int
        Z_0, the initial population size.
    unbias : bool
        Condition on non-extinction for positive counts when
        ``initial_population == 1``.
    is_ordered : bool
        If True, first age in ``ages`` is the spine.
    log_scale_params : bool
        If True, exponentiate rates before use.

    Returns
    -------
    jnp.ndarray
        Scalar float64 log-likelihood, or ``-inf`` for numerically invalid
        inputs.

    Raises
    ------
    ValueError
        For programming errors (conflicting flags, invalid array shapes).

    """
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    birth_rates = _to_f64(birth_rates)
    death_rates = _to_f64(death_rates)
    ages = _to_f64(ages)
    times = _to_f64(times)

    if not (birth_rates.shape == death_rates.shape == times.shape):
        msg = (
            "birth_rates, death_rates, and times must have the same shape, "
            f"got {birth_rates.shape}, {death_rates.shape}, {times.shape}."
        )
        raise ValueError(msg)

    if log_scale_params:
        birth_rates = jnp.exp(birth_rates)
        death_rates = jnp.exp(death_rates)
    elif bool(jnp.any(birth_rates <= 0)) or bool(jnp.any(death_rates <= 0)):
        return _ninf

    if spine_ages is not None:
        spine_ages_arr = _to_f64(spine_ages)
        if len(spine_ages_arr) == 0:
            spine_ages_arr = None
        else:
            if is_ordered:
                msg = "Cannot have both is_ordered=True and spine_ages provided."
                raise ValueError(msg)
            spine_ages = spine_ages_arr
    else:
        spine_ages_arr = None

    if float(times[0]) <= 0:
        msg = "Observation times must be positive."
        raise ValueError(msg)

    if bool(jnp.any(jnp.diff(times) < 0)):
        msg = "Observation times must be non-decreasing."
        raise ValueError(msg)

    if bool(jnp.any(ages < 0)) or bool(jnp.any(ages > times[-1])):
        msg = f"Ages must be in the range [0, {float(times[-1])}]."
        raise ValueError(msg)

    # ------------ ns handling ---------------
    n_explicit_spine = len(spine_ages_arr) if spine_ages_arr is not None else 0
    num_ages = len(ages) + n_explicit_spine

    if ns is not None:
        ns_arr = _to_f64(ns)
        total_pop_size = int(float(ns_arr.sum()))
        if total_pop_size < num_ages:
            msg = "Sum of the populations must equal the number of ages provided."
            raise ValueError(msg)
        if total_pop_size > num_ages:
            warnings.warn(
                "Sum of the populations exceeds the number of ages provided; "
                "not all ages have been provided (age information is partial).",
                UserWarning,
                stacklevel=2,
            )
    else:
        ns_arr = jnp.array([num_ages], dtype=jnp.float64)
        total_pop_size = num_ages

    if not (
        bool(jnp.all(jnp.isfinite(birth_rates)))
        and bool(jnp.all(jnp.isfinite(death_rates)))
    ):
        return _ninf

    # ------------ population log-likelihood ---------------
    log_P_n_T = calculate_populations_log_lik(
        birth_rates=birth_rates,
        death_rates=death_rates,
        ns=ns_arr,
        times=times,
        initial_population=initial_population,
        unbias=unbias,
    )

    if not bool(jnp.isfinite(log_P_n_T)):
        return _ninf
    if total_pop_size == 0:
        return log_P_n_T

    ages_log_lik = jnp.float64(0.0)

    # ------------ spine detection (host-side, data-dependent) ---------------
    if spine_ages_arr is None and not is_ordered and len(ages) > 0:
        at_final = np.isclose(np.array(ages, dtype=np.float64), float(times[-1]))
        num_terminal = int(at_final.sum())
        if num_terminal > 1:
            msg = (
                "More than one age equals the final observation time. "
                "Exactly one spine is assumed; provide explicit spine_ages."
            )
            raise ValueError(msg)
        if num_terminal == 1:
            spine_ages_arr = ages[jnp.array(at_final)]
            ages = ages[jnp.array(~at_final)]
        elif len(ages) == 1:
            spine_ages_arr = ages
            ages = ages[:0]

    use_unknown_spine_marginal = (not is_ordered) and (spine_ages_arr is None)

    # ------------ age log-likelihood ---------------
    if use_unknown_spine_marginal and len(ages) > 0:
        n_ages = len(ages)
        non_spine_lls = calculate_non_spine_ages_log_lik(
            birth_rates, death_rates, ages, times, return_array=True
        )
        spine_lls = calculate_spine_ages_log_lik(
            birth_rates, death_rates, ages, times, return_array=True
        )
        sum_log_f = non_spine_lls.sum()
        a = spine_lls - non_spine_lls
        log_sum_exp = jnp.log(jnp.sum(jnp.exp(a - a.max()))) + a.max()  # stable
        ages_log_lik = sum_log_f - jnp.log(jnp.float64(n_ages)) + log_sum_exp
    elif is_ordered or (spine_ages_arr is not None):
        if is_ordered:
            spine_ages_arr = ages[:1]
            ages = ages[1:]
        if spine_ages_arr is not None:
            ages_log_lik = ages_log_lik + calculate_spine_ages_log_lik(
                birth_rates, death_rates, spine_ages_arr, times
            )

    if len(ages) > 0 and not use_unknown_spine_marginal:
        ages_log_lik = ages_log_lik + calculate_non_spine_ages_log_lik(
            birth_rates, death_rates, ages, times
        )

    if not bool(jnp.isfinite(ages_log_lik)):
        return _ninf

    total = log_P_n_T + ages_log_lik
    if not bool(jnp.isfinite(total)):
        return _ninf
    return total
