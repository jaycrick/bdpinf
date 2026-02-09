"""
Likelihood calculation functions for birth-death process parameter estimation.

This module contains functions for calculating log-likelihood values for
birth-death process models, used in parameter fitting and model evaluation.
"""

import warnings

import torch

from ..models.mathematical_derivatives import (
    calculate_ages_log_lik_derivatives,
    calculate_populations_log_lik_derivatives,
)
from ..models.mathematical_functions import (
    calculate_ages_log_lik,
    calculate_populations_log_lik,
)
from ..utils import ensure_1d_tensor, ensure_1d_tensors


def log_likelihood(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    spine_ages: torch.Tensor | None = None,
    ns: torch.Tensor | None = None,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> torch.Tensor:
    """
    Compute the log-likelihood of observed manuscript ages under a birth–death process.

    This function evaluates the joint log-likelihood for a dataset of observed manuscript ages
    given a birth (λ) and death (μ) rate under a birth–death model. The likelihood is computed
    as the sum of two contributions:

        1. The log-probability of observing the total population configuration at observation
             time T (log P_n(T)).
        2. The sum of log age-densities for the observed manuscripts (Σ log f(t_i)), with an
             option to treat one or more observations as "spines" (special conditional ages).

    Behavior and notes
    - If log_scale_params is True, birth_rate and death_rate are interpreted as log(λ) and
        log(μ); otherwise they are interpreted on the original (positive) scale.
    - If ns is None, the function assumes each provided age corresponds to one individual
        (i.e., a count of 1 per age) and constructs ns accordingly.
    - If is_ordered is True, the first provided age is treated as a spine (unique conditional
        distribution) and the remaining ages are treated as ordinary observations. is_ordered
        and spine_ages cannot both be provided.
    - If spine_ages is provided, those ages are treated as spines; ordinary ages are treated
        as non-spine observations.
    - The function validates input ranges: rates must be finite (and positive if not in log
        scale), observation_time must be > 0, and all ages/spine_ages must lie in [0, T].
    - Returns -inf for numerical issues (non-finite rates, invalid parameter domains, overflow).
    - Raises exceptions for programming errors (type mismatches, array shape errors, etc.).

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
    is_ordered : bool, optional
        If True, the first element of ages is treated as a spine and the rest as
        ordinary observations. Cannot be used together with spine_ages.
    log_scale_params : bool, optional
        If True, treat birth_rate and death_rate as log-scale parameters.

    Returns
    -------
    torch.Tensor
        The computed total log-likelihood (log P_n(T) + Σ log f(t_i)).
        Returns -inf for numerical issues (non-positive rates, overflow, invalid domains).

    Raises
    ------
    ValueError
        If both is_ordered=True and spine_ages is provided, or if the sum of ns does
        not equal the number of provided ages.
    TypeError, IndexError, etc.
        Programming errors (type mismatches, array indexing errors) raise exceptions
        instead of returning -inf for easier debugging.

    See also:
        calculate_populations_log_likelihood, calculate_ages_log_likelihood
    """
    birth_rates, death_rates, ages, times = ensure_1d_tensors(
        birth_rates, death_rates, ages, times
    )

    if not (birth_rates.shape == death_rates.shape == times.shape):
        raise ValueError(
            "birth_rates, death_rates, and times must have the same shape, "
            f"got {birth_rates.shape}, {death_rates.shape}, {times.shape}."
        )

    # Handle parameter conversion if in log scale
    if log_scale_params:
        birth_rates = torch.exp(birth_rates)
        death_rates = torch.exp(death_rates)
    else:
        if torch.any(birth_rates <= 0) or torch.any(death_rates <= 0):
            return torch.tensor(-torch.inf, dtype=torch.float64)

    if spine_ages is not None:
        spine_ages = ensure_1d_tensor(spine_ages)
        if len(spine_ages) == 0:
            spine_ages = None
        else:
            if is_ordered:
                raise ValueError(
                    "Cannot have both is_ordered=True and spine_ages provided."
                )
            if ns is None:
                if len(spine_ages) > 1:
                    raise ValueError(
                        "When multiple spine ages are provided, "
                        "population sizes (ns) must also be provided."
                    )
                else:
                    ns = torch.tensor([len(ages) + len(spine_ages)], dtype=torch.int64)
            if torch.any(spine_ages < 0) or torch.any(spine_ages > times[-1]):
                raise ValueError(f"Spine ages must be in the range [0, {times[-1]}].")

    if times[0] <= 0:
        raise ValueError("Observation times must be positive.")

    if torch.any(torch.diff(times) < 0):
        raise ValueError("Observation times must be non-decreasing.")

    if torch.any(ages < 0) or torch.any(ages > times[-1]):
        raise ValueError(f"Ages must be in the range [0, {times[-1]}].")

    # Total number of observed manuscripts
    num_ages = len(ages) + (len(spine_ages) if spine_ages is not None else 0)
    if ns is not None:
        total_pop_size = sum(ns)
        if total_pop_size < num_ages:
            raise ValueError(
                "Sum of the populations must equal the number of ages provided."
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
        lambda_params=birth_rates,
        mu_params=death_rates,
        times=times,
        ns=ns,
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
            ages_t=spine_ages,
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


def log_likelihood_gradient(
    params: torch.Tensor | float,
    ages: torch.Tensor | float,
    observation_time: float,
    spine_ages: torch.Tensor | float | None = None,
    ns: torch.Tensor | None = None,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> torch.Tensor:
    """
    Compute the gradient of the model log-likelihood with respect to the birth and
    death parameters.

    This function validates inputs and then combines two contributions to the
    gradient:
        1. The derivative of the log-probability of the observed population sizes
             (P(n | T)) with respect to the parameters.
        2. The derivative of the log-density of the observed ages (and optional
             spine ages) with respect to the parameters.

    The contribution from ages can be computed in a sequential/ordered fashion:
    if `is_ordered` is True the first age in `ages` is treated as a spine age;
    alternatively a separate `spine_ages` array may be supplied.

    Parameters
    ----------
    params : torch.Tensor | float
        Length-2 tensor containing the birth and death parameters in the order
        [birth_rate, death_rate]. If `log_scale_params` is True these are the
        log-parameters (log(birth), log(death)); otherwise they are in the
        natural scale.
    ages : torch.Tensor | float
        Observed manuscript ages (can be a scalar or a 1-D tensor). All ages
        must lie in [0, observation_time].
    observation_time : float
        Positive scalar giving the observation time T. Ages must be in
        the interval [0, T].
    spine_ages : torch.Tensor | float | None, optional
        Optional ages that are treated as spine observations. Cannot be
        provided if `is_ordered` is True. If provided, must lie in [0, T].
    ns : torch.Tensor | None, optional
        Optional tensor of population counts per grouping. If omitted the total
        population is assumed to be len(ensure_1d_tensor(ages)). When provided, the sum(ns) must
        equal len(ensure_1d_tensor(ages)).
    is_ordered : bool, optional
        If True, the input `ages` array is interpreted as ordered and the first
        entry is used as a spine age; `spine_ages` must then be None.
    log_scale_params : bool, optional
        If True, `params` are interpreted as log-scale parameters and the
        returned gradient is with respect to those log-parameters. If False,
        `params` are on the original scale and derivatives are computed
        accordingly.

    Returns
    -------
    torch.Tensor
        A length-2 tensor of dtype float64 giving the gradient of the
        log-likelihood with respect to the birth and death parameters in the
        same ordering as `params`. If `log_scale_params` is True the gradient
        is with respect to the log-parameters.

        When `log_scale_params` is False and either `birth_rate` <= 0 or
        `death_rate` <= 0 the function returns torch.tensor([nan, nan])
        to indicate an invalid original-scale input.

    Raises
    ------
    ValueError
        - if `params` does not have length 2,
        - if any parameter is not finite,
        - if `is_ordered` is True and `spine_ages` is also provided,
        - if `observation_time` is not positive,
        - if any value in `ages` or `spine_ages` lies outside [0, observation_time],
        - if `ns` is provided and sum(ns) != len(ensure_1d_tensor(ages)).
    TypeError, IndexError, etc.
        Programming errors raise exceptions for easier debugging.

    Example:
        >>> # params as log-parameters
        >>> grad = log_likelihood_gradient(
                torch.log(torch.tensor([1.2, 0.3])), ages=[0.1, 0.4], observation_time=1.0, log_scale_params=True
            )
    """
    params_tensor, ages = ensure_1d_tensors(params, ages)
    if len(params_tensor) != 2:
        raise ValueError("params must have length 2")
    if is_ordered and (spine_ages is not None):
        raise ValueError("Cannot have both is_ordered=True and spine_ages provided.")

    birth_rate, death_rate = params_tensor
    # Input validation
    if not (torch.isfinite(birth_rate) and torch.isfinite(death_rate)):
        raise ValueError("Parameters must be finite.")

    if not log_scale_params:
        # Parameters are in original scale, validate directly
        if birth_rate <= 0 or death_rate <= 0:
            return torch.tensor([torch.nan, torch.nan], dtype=torch.float64)

    # Check age range
    if observation_time <= 0:
        raise ValueError("Observation time must be positive.")

    if spine_ages is not None:
        spine_ages = ensure_1d_tensor(spine_ages)
    if torch.any(ages < 0) or torch.any(ages > observation_time):
        raise ValueError("Ages must be in the range [0, observation_time].")

    if spine_ages is not None and (
        torch.any(spine_ages < 0) or torch.any(spine_ages > observation_time)
    ):
        raise ValueError("Spine ages must be in the range [0, observation_time].")

    # Total number of observed manuscripts
    if ns is None:
        total_pop_size = len(ages)
        ns = torch.tensor([total_pop_size], dtype=torch.int64)
    else:
        total_pop_size = int(torch.as_tensor(ns, dtype=torch.int64).sum().item())
        ns = torch.as_tensor(ns, dtype=torch.int64)

    if not total_pop_size == len(ages) + (
        len(spine_ages) if spine_ages is not None else 0
    ):
        raise ValueError(
            "Sum of the populations must equal the number of ages provided."
        )

    # Gradient components
    grad_log_P_n_T = calculate_populations_log_lik_derivatives(
        birth_rate,
        death_rate,
        observation_time,
        ns,
    )

    if total_pop_size == 0:
        return grad_log_P_n_T

    # Sequential computation for small datasets
    grad_ages = torch.zeros(2, dtype=torch.float64)
    if is_ordered or (spine_ages is not None):
        if is_ordered:
            ages = torch.as_tensor(ages, dtype=torch.float64)
            spine_ages = torch.tensor([ages[0]], dtype=torch.float64)
            ages = ages[1:]
        assert spine_ages is not None
        grad_ages += calculate_ages_log_lik_derivatives(
            birth_rate,
            death_rate,
            spine_ages,
            observation_time,
            are_spines=True,
        )

    if len(ages) > 0:
        grad_ages += calculate_ages_log_lik_derivatives(
            birth_rate,
            death_rate,
            ages,
            observation_time,
            are_spines=False,
        )

    return grad_log_P_n_T + grad_ages


def negative_log_likelihood(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    spine_ages: torch.Tensor | None = None,
    ns: torch.Tensor | None = None,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> torch.Tensor:
    """
    Compute the negative log-likelihood for use with numerical optimizers.

    This function is a thin wrapper around log_likelihood that negates its
    value so it can be minimized by optimizers.
    It expects the model's birth and death rates (or their logarithms, see log_scale_params).

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

    result = -log_likelihood(
        birth_rates=birth_rates,
        death_rates=death_rates,
        ages=ages,
        times=times,
        spine_ages=spine_ages,
        ns=ns,
        is_ordered=is_ordered,
        log_scale_params=log_scale_params,
    )

    return result

    return result
