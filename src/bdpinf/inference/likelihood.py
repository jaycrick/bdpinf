"""Likelihood calculation functions for birth-death process parameter estimation.

This module contains functions for calculating log-likelihood values for
birth-death process models, used in parameter fitting and model evaluation.
"""

import warnings

import jax.numpy as jnp
import numpy as np

from inf_bp_texts.models.math_ages import (
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)
from inf_bp_texts.models.math_populations import calculate_populations_log_lik
from inf_bp_texts.utils import ensure_1d_tensor, ensure_1d_tensors


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
    """Compute the log-likelihood of observed manuscript ages in a birth--death process.

    This function evaluates the joint log-likelihood
    for a dataset of observed manuscript ages
    given a birth (λ) and death (μ) rate under a birth--death model.
    The likelihood is computed
    as the sum of two contributions:

        1. The log-probability of observing the total population configuration
            at observation time T (log P_n(T)).
        2. The sum of log age-densities for the observed manuscripts (Σ log f(t_i)),
            with an option to treat one or more observations as "spines"
            (special conditional ages).

    Behavior and notes
    - If log_scale_params is True, birth_rates and death_rates are interpreted as log(λ)
        and log(μ); otherwise they are interpreted on the original (positive) scale.
    - If ns is None, the function assumes each provided age corresponds to
        one individual (i.e., a count of 1 per age) and constructs ns accordingly.
    - If spine_ages is None and is_ordered is False, the function treats the spine
        as unknown among the provided ages and marginalizes over all possible
        spine positions (uniformly) using a numerically stable log-sum-exp term.
    - Exception: if spine_ages is None and any observed age equals the final
        observation time, that age is treated as the spine automatically.
        If more than one age equals the final observation time, a ValueError is
        raised because the model assumes a unique spine.
    - Exception: if spine_ages is None and there is exactly one observed age,
        that individual is treated as the spine automatically.
    - If is_ordered is True, the first provided age is treated as a spine
        (unique conditional distribution)
        and the remaining ages are treated as ordinary observations.
        is_ordered and spine_ages cannot both be provided.
    - If spine_ages is provided, those ages are treated as spines;
        ordinary ages are treated as non-spine observations.
    - The function validates input ranges: rates must be finite
        (and positive if not in log scale), observation_time must be > 0,
        and all ages/spine_ages must lie in [0, T].
    - Returns -inf for numerical issues
        (non-finite rates, invalid parameter domains, overflow).
    - Raises exceptions for programming errors
        (type mismatches, array shape errors, etc.).

    Parameters
    ----------
    birth_rates : float | jax.Array
        Birth rate parameter λ > 0, or log(λ) if log_scale_params is True.
    death_rates : float | jax.Array
        Death rate parameter μ > 0, or log(μ) if log_scale_params is True.
    ages : jax.Array | float
        Observed ages of manuscripts. A scalar is treated as a single-element array.
        All ages must satisfy 0 <= age <= observation_time.
    times : jax.Array | float
        Observation times (> 0) at which the population/ages are observed.
    spine_ages : jax.Array | None, optional
        Ages to be treated as spines (special conditional observations). If provided,
        these ages must also satisfy 0 <= age <= observation_time and are used
        explicitly. If not provided and is_ordered=False, the spine is treated as
        unknown among the provided ages and marginalized out. Mutually exclusive
        with is_ordered=True.
    ns : jax.Array | None, optional
        Counts per population/class that sum to the total number of ages. If None,
        a single population with one count per provided age is assumed.
    initial_population: int
        Initial population size.
    is_ordered : bool, optional
        If True, the first element of ages is treated as a spine and the rest as
        ordinary observations. Cannot be used together with spine_ages. When False
        and spine_ages is None, the unknown-spine marginal likelihood is used.
    log_scale_params : bool, optional
        If True, treat birth_rates and death_rates as log-scale parameters.

    Returns
    -------
    jnp.ndarray
        The computed total log-likelihood (log P_n(T) + Σ log f(t_i)).
        Returns -inf for numerical issues
        (non-positive rates, overflow, invalid domains).

    Raises
    ------
    ValueError
        If both is_ordered=True and spine_ages is provided, or if the sum of ns does
        not equal the number of provided ages.

    See Also
    --------
        calculate_populations_log_likelihood, calculate_ages_log_likelihood

    """
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    birth_rates, death_rates, ages, times = ensure_1d_tensors(
        birth_rates,
        death_rates,
        ages,
        times,
    )

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
        spine_ages = ensure_1d_tensor(spine_ages)
        if len(spine_ages) == 0:
            spine_ages = None
        else:
            if is_ordered:
                msg = "Cannot have both is_ordered=True and spine_ages provided."
                raise ValueError(msg)
            if ns is None:
                if len(spine_ages) > 1:
                    msg = (
                        "When multiple spine ages are provided, "
                        "population sizes (ns) must also be provided."
                    )
                    raise ValueError(msg)
                ns = jnp.array([len(ages) + len(spine_ages)], dtype=jnp.float64)
            if bool(jnp.any(spine_ages < 0)) or bool(jnp.any(spine_ages > times[-1])):
                msg = f"Spine ages must be in the range [0, {times[-1]}]."
                raise ValueError(msg)

    if float(times[0]) <= 0:
        msg = "Observation times must be positive."
        raise ValueError(msg)

    if bool(jnp.any(jnp.diff(times) < 0)):
        msg = "Observation times must be non-decreasing."
        raise ValueError(msg)

    if bool(jnp.any(ages < 0)) or bool(jnp.any(ages > times[-1])):
        msg = f"Ages must be in the range [0, {float(times[-1])}]."
        raise ValueError(msg)

    num_ages = len(ages) + (len(spine_ages) if spine_ages is not None else 0)
    if ns is not None:
        ns_arr = ensure_1d_tensor(ns)
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

    log_P_n_T = calculate_populations_log_lik(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        ns=ns_arr,
        initial_population=initial_population,
        unbias=unbias,
    )

    if not bool(jnp.isfinite(log_P_n_T)):
        return _ninf
    if total_pop_size == 0:
        return log_P_n_T

    ages_log_lik = jnp.float64(0.0)

    if spine_ages is None and not is_ordered and len(ages) > 0:
        # Host-side boundary detection (data-dependent, not traced)
        at_final_time = np.isclose(np.array(ages, dtype=np.float64), float(times[-1]))
        num_terminal_ages = int(at_final_time.sum())
        if num_terminal_ages > 1:
            msg = (
                "More than one age equals the final observation time. "
                "Exactly one spine is assumed; provide explicit spine_ages."
            )
            raise ValueError(msg)
        if num_terminal_ages == 1:
            spine_ages = ages[jnp.array(at_final_time)]
            ages = ages[jnp.array(~at_final_time)]
        elif len(ages) == 1:
            spine_ages = ages
            ages = ages[:0]

    use_unknown_spine_marginal = (not is_ordered) and (spine_ages is None)

    if use_unknown_spine_marginal and len(ages) > 0:
        n_ages = len(ages)
        non_spine_log_liks = calculate_non_spine_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times=times,
            return_array=True,
        )
        spine_log_liks = calculate_spine_ages_log_lik(
            birth_rates,
            death_rates,
            ages=ages,
            times=times,
            return_array=True,
        )

        sum_log_f = non_spine_log_liks.sum()
        a = spine_log_liks - non_spine_log_liks
        log_sum_exp = jnp.log(jnp.sum(jnp.exp(a - a.max()))) + a.max()
        ages_log_lik = sum_log_f - jnp.log(jnp.float64(n_ages)) + log_sum_exp
    elif is_ordered or (spine_ages is not None):
        if is_ordered:
            spine_ages = ages[:1]
            ages = ages[1:]
        if spine_ages is not None:
            ages_log_lik = ages_log_lik + calculate_spine_ages_log_lik(
                birth_rates,
                death_rates,
                ages=spine_ages,
                times=times,
            )

    if len(ages) > 0 and not use_unknown_spine_marginal:
        ages_log_lik = ages_log_lik + calculate_non_spine_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times=times,
        )

    if not bool(jnp.isfinite(ages_log_lik)):
        return _ninf

    total_log_likelihood = log_P_n_T + ages_log_lik

    if not bool(jnp.isfinite(total_log_likelihood)):
        return _ninf

    return total_log_likelihood


def calculate_unknown_spine_mixture_age_density(
    birth_rates: object,
    death_rates: object,
    age_grid: object,
    times: object,
    num_manuscripts: int,
) -> jnp.ndarray:
    """Compute age density for a random manuscript when the spine is unknown.

    Assumes exactly one spine among ``num_manuscripts`` observations and uses
    the mixture

        (1 / n) * ((n - 1) * f_non_spine(a) + f_spine(a)).

    This helper is useful for plotting theoretical age distributions under the
    same unknown-spine assumption used by the likelihood.

    Parameters
    ----------
    birth_rates : jax.Array | float
        Birth rate(s) λ.
    death_rates : jax.Array | float
        Death rate(s) μ.
    age_grid : jax.Array | float
        Age grid values where the density is evaluated.
    times : jax.Array | float
        Observation time(s). Scalar for single epoch or 1D epoch end times.
    num_manuscripts : int
        Number of manuscripts ``n`` in the mixture formula. Must be positive.

    Returns
    -------
    jnp.ndarray
        Density evaluated at each point in ``age_grid`` (shape matches grid).

    Raises
    ------
    ValueError
        If ``num_manuscripts`` is not positive.

    """
    if num_manuscripts <= 0:
        msg = "num_manuscripts must be a positive integer."
        raise ValueError(msg)

    birth_rates, death_rates, age_grid, times = ensure_1d_tensors(
        birth_rates,
        death_rates,
        age_grid,
        times,
    )

    mixture_log_density_vals: list[jnp.ndarray] = []
    log_n = jnp.log(jnp.float64(num_manuscripts))
    if num_manuscripts > 1:
        log_n_minus_one = jnp.log(jnp.float64(num_manuscripts - 1))
    else:
        log_n_minus_one = jnp.array(-jnp.inf, dtype=jnp.float64)

    for age in age_grid:
        age_arr = jnp.atleast_1d(jnp.asarray([age], dtype=jnp.float64))
        log_non_spine = calculate_non_spine_ages_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=age_arr,
            times=times,
        )
        log_spine = calculate_spine_ages_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=age_arr,
            times=times,
        )

        ab = jnp.stack([log_n_minus_one + log_non_spine, log_spine])
        log_mixture_numerator = jnp.log(jnp.sum(jnp.exp(ab - ab.max()))) + ab.max()
        mixture_log_density_vals.append(log_mixture_numerator - log_n)

    mixture_log_density = jnp.stack(mixture_log_density_vals)
    return jnp.exp(mixture_log_density)


def calculate_unknown_spine_mixture_birth_density(
    birth_rates: object,
    death_rates: object,
    birth_time_grid: object,
    times: object,
    num_manuscripts: int,
) -> jnp.ndarray:
    """Compute birth-time density for a random manuscript when the spine is unknown.

    This transforms the age-density mixture to a density over birth times by the
    change of variables birth_time = T - age, where T = times[-1]. The Jacobian
    is 1 in absolute value, so the birth-time density at b is the age density at
    age = T - b evaluated with the same mixture formula.

    Parameters
    ----------
    birth_rates, death_rates, birth_time_grid, times, num_manuscripts
        Same semantics as :func:`calculate_unknown_spine_mixture_age_density`,
        except that ``birth_time_grid`` is interpreted as times since the
        model `start_time` (i.e., in the same coordinate system as ``times``).

    Returns
    -------
    jnp.ndarray
        Density evaluated at each point in ``birth_time_grid`` (shape matches grid).

    """
    if num_manuscripts <= 0:
        msg = "num_manuscripts must be a positive integer."
        raise ValueError(msg)

    birth_rates, death_rates, birth_time_grid, times = ensure_1d_tensors(
        birth_rates,
        death_rates,
        birth_time_grid,
        times,
    )

    T = times[-1]

    if bool(jnp.any(birth_time_grid < 0)) or bool(jnp.any(birth_time_grid > T)):
        msg = f"Birth times must be in the range [0, {T}]. Got: {birth_time_grid}"
        raise ValueError(msg)

    age_grid = T - birth_time_grid

    return calculate_unknown_spine_mixture_age_density(
        birth_rates=birth_rates,
        death_rates=death_rates,
        age_grid=age_grid,
        times=times,
        num_manuscripts=num_manuscripts,
    )


def negative_log_likelihood(
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
    """Compute the negative log-likelihood for use with numerical optimizers.

    This function is a thin wrapper around log_likelihood that negates its
    value so it can be minimized by optimizers.
    It expects the model's birth and death rates
    (or their logarithms, see log_scale_params).

    Parameters
    ----------
    birth_rates : float | jax.Array
        Birth rate parameter(s) λ > 0, or log(λ) if log_scale_params is True.
    death_rates : float | jax.Array
        Death rate parameter(s) μ > 0, or log(μ) if log_scale_params is True.
    ages : jax.Array | float
        Ages of observed manuscripts (or a single age).
    times : jax.Array | float
        Observation times (> 0) at which the population/ages are observed.
    spine_ages : jax.Array | None, optional
        Optional spine ages required by some likelihood formulations.
    ns : jax.Array | None, optional
        Optional counts/weights associated with each age.
    initial_population: int
        Initial population size.
    is_ordered : bool, default False
        If True, interpret ``ages`` as ordered-event data.
    log_scale_params : bool, default False
        If True, parameters are provided in log-space.

    Returns
    -------
    jnp.ndarray
        The negative log-likelihood (i.e. -log_likelihood).

    """
    return -log_likelihood(
        birth_rates=birth_rates,
        death_rates=death_rates,
        ages=ages,
        times=times,
        spine_ages=spine_ages,
        ns=ns,
        initial_population=initial_population,
        unbias=unbias,
        is_ordered=is_ordered,
        log_scale_params=log_scale_params,
    )
