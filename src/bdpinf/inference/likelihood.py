"""Likelihood calculation functions for birth-death process parameter estimation.

This module contains functions for calculating log-likelihood values for
birth-death process models, used in parameter fitting and model evaluation.
"""

import warnings

import torch

from inf_bp_texts.models.mathematical_functions import (
    calculate_ages_log_lik,
    calculate_populations_log_lik,
)
from inf_bp_texts.utils import ensure_1d_tensor, ensure_1d_tensors


def log_likelihood(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    spine_ages: torch.Tensor | None = None,
    ns: torch.Tensor | None = None,
    starting_population: int = 1,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> torch.Tensor:
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
    - If log_scale_params is True, birth_rate and death_rate are interpreted as log(λ)
        and log(μ); otherwise they are interpreted on the original (positive) scale.
    - If ns is None, the function assumes each provided age corresponds to
        one individual (i.e., a count of 1 per age) and constructs ns accordingly.
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
    birth_rate : float | torch.Tensor
        Birth rate parameter λ > 0, or log(λ) if log_scale_params is True.
    death_rate : float | torch.Tensor
        Death rate parameter μ > 0, or log(μ) if log_scale_params is True.
    ages : torch.Tensor | float
        Observed ages of manuscripts. A scalar is treated as a single-element tensor.
        All ages must satisfy 0 <= age <= observation_time.
    times : torch.Tensor | float
        Observation times (> 0) at which the population/ages are observed.
    spine_ages : torch.Tensor | None, optional
        Ages to be treated as spines (special conditional observations). If provided,
        these ages must also satisfy 0 <= age <= observation_time. Mutually exclusive
        with is_ordered=True.
    ns : torch.Tensor | None, optional
        Counts per population/class that sum to the total number of ages. If None,
        a single population with one count per provided age is assumed.
    starting_population: int
        Starting population size.
    is_ordered : bool, optional
        If True, the first element of ages is treated as a spine and the rest as
        ordinary observations. Cannot be used together with spine_ages.
    log_scale_params : bool, optional
        If True, treat birth_rate and death_rate as log-scale parameters.

    Returns
    -------
    torch.Tensor
        The computed total log-likelihood (log P_n(T) + Σ log f(t_i)).
        Returns -inf for numerical issues
        (non-positive rates, overflow, invalid domains).

    Raises
    ------
    ValueError
        If both is_ordered=True and spine_ages is provided, or if the sum of ns does
        not equal the number of provided ages.
    TypeError, IndexError, etc.
        Programming errors (type mismatches, array indexing errors) raise exceptions
        instead of returning -inf for easier debugging.

    See Also
    --------
        calculate_populations_log_likelihood, calculate_ages_log_likelihood

    """
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
        raise ValueError(
            msg,
        )

    # Handle parameter conversion if in log scale
    if log_scale_params:
        birth_rates = torch.exp(birth_rates)
        death_rates = torch.exp(death_rates)
    elif torch.any(birth_rates <= 0) or torch.any(death_rates <= 0):
        return torch.tensor(-torch.inf, dtype=torch.float64)

    if spine_ages is not None:
        spine_ages = ensure_1d_tensor(spine_ages)
        if len(spine_ages) == 0:
            spine_ages = None
        else:
            if is_ordered:
                msg = "Cannot have both is_ordered=True and spine_ages provided."
                raise ValueError(
                    msg,
                )
            if ns is None:
                if len(spine_ages) > 1:
                    msg = (
                        "When multiple spine ages are provided, "
                        "population sizes (ns) must also be provided."
                    )
                    raise ValueError(
                        msg,
                    )
                ns = torch.tensor([len(ages) + len(spine_ages)], dtype=torch.int64)
            if torch.any(spine_ages < 0) or torch.any(spine_ages > times[-1]):
                msg = f"Spine ages must be in the range [0, {times[-1]}]."
                raise ValueError(msg)

    if times[0] <= 0:
        msg = "Observation times must be positive."
        raise ValueError(msg)

    if torch.any(torch.diff(times) < 0):
        msg = "Observation times must be non-decreasing."
        raise ValueError(msg)

    if torch.any(ages < 0) or torch.any(ages > times[-1]):
        msg = f"Ages must be in the range [0, {times[-1]}]."
        raise ValueError(msg)

    # Total number of observed manuscripts
    num_ages = len(ages) + (len(spine_ages) if spine_ages is not None else 0)
    if ns is not None:
        total_pop_size = sum(ns)
        if total_pop_size < num_ages:
            msg = "Sum of the populations must equal the number of ages provided."
            raise ValueError(
                msg,
            )
        if total_pop_size > num_ages:
            warnings.warn(
                "Sum of the populations exceeds the number of ages provided; "
                "not all ages have been provided (age information is partial).",
                UserWarning,
            )
    else:
        ns = torch.tensor([num_ages], dtype=torch.int64)
        total_pop_size = num_ages

    # Input validation
    if not (
        torch.all(torch.isfinite(birth_rates))
        and torch.all(torch.isfinite(death_rates))
    ):
        return torch.tensor(-torch.inf, dtype=torch.float64)
    # 1. Calculate log(P_n(T)) - log probability of n individuals at time T
    log_P_n_T = calculate_populations_log_lik(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        ns=ns,
        starting_population=starting_population,
    )

    # Check if population probability calculation succeeded
    if not torch.isfinite(log_P_n_T):
        return torch.tensor(-torch.inf, dtype=torch.float64)
    if total_pop_size == 0:
        return log_P_n_T

    ages_log_lik = torch.tensor(0.0, dtype=torch.float64)

    if is_ordered or (spine_ages is not None):
        if is_ordered:
            ages = torch.as_tensor(ages, dtype=torch.float64)
            spine_ages = torch.tensor([ages[0]], dtype=torch.float64)
            ages = ages[1:]
        # Calculate log age density
        assert spine_ages is not None
        ages_log_lik += calculate_ages_log_lik(
            birth_rates,
            death_rates,
            ages=spine_ages,
            times=times,
            are_spines=True,
        )
    if len(ages) > 0:
        ages_log_lik += calculate_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times=times,
            are_spines=False,
        )

    # Check if age density calculation succeeded
    if not torch.isfinite(torch.as_tensor(ages_log_lik)):
        return torch.tensor(-torch.inf, dtype=torch.float64)

    total_log_likelihood = log_P_n_T + ages_log_lik

    # Final numerical validation
    if not torch.isfinite(total_log_likelihood):
        return torch.tensor(-torch.inf, dtype=torch.float64)

    return total_log_likelihood


def negative_log_likelihood(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    spine_ages: torch.Tensor | None = None,
    ns: torch.Tensor | None = None,
    starting_population: int = 1,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> torch.Tensor:
    """Compute the negative log-likelihood for use with numerical optimizers.

    This function is a thin wrapper around log_likelihood that negates its
    value so it can be minimized by optimizers.
    It expects the model's birth and death rates
    (or their logarithms, see log_scale_params).

    Parameters
    ----------
    birth_rates : torch.Tensor | float
        Birth rate parameter(s) λ > 0, or log(λ) if log_scale_params is True.
    death_rates : torch.Tensor | float
        Death rate parameter(s) μ > 0, or log(μ) if log_scale_params is True.
    ages : torch.Tensor | float
        Ages of observed manuscripts (or a single age). May be a 1-D tensor
        of observation ages used by the likelihood computation.
    times : torch.Tensor | float
        Observation times (> 0) at which the population/ages are observed.
    spine_ages : torch.Tensor | None, optional
        Optional spine ages required by some likelihood formulations. If not
        provided, a default or simplified likelihood is computed (depending on
        the underlying log_likelihood implementation).
    ns : torch.Tensor | None, optional
        Optional counts/weights associated with each age. If omitted, each
        observation is treated as a single count.
    starting_population: int
        Starting population size.
    is_ordered : bool, default False
        If True, interpret `ages` (and `ns` when provided) as ordered-event data
        and use the ordered-data likelihood branch in the underlying function.
    log_scale_params : bool, default False
        If True, indicates that parameters are provided in log-space and the
        underlying log_likelihood should convert them back to the original scale
        before evaluating the likelihood.

    Returns
    -------
    torch.Tensor
        The negative log-likelihood (i.e. -log_likelihood). This is returned
        as a torch.Tensor scalar suitable for use with optimizers.

    Notes
    -----
    - This wrapper does not validate parameter bounds beyond checking for validity;
      any additional constraints (e.g. non-negativity) should be enforced by the
      optimizer or by inspecting the returned value.
    - Returns inf for numerical issues in the underlying log_likelihood computation.
    - Programming errors in underlying functions will raise exceptions.
    - Designed to be cheap to call repeatedly by optimization routines.

    """
    return -log_likelihood(
        birth_rates=birth_rates,
        death_rates=death_rates,
        ages=ages,
        times=times,
        spine_ages=spine_ages,
        ns=ns,
        starting_population=starting_population,
        is_ordered=is_ordered,
        log_scale_params=log_scale_params,
    )
