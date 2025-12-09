"""
Likelihood calculation functions for birth-death process parameter estimation.

This module contains functions for calculating log-likelihood values for
birth-death process models, used in parameter fitting and model evaluation.
"""

import numpy as np
from numpy.typing import NDArray

from ..models.mathematical_functions import (
    calculate_ages_log_likelihood,
    calculate_ages_log_likelihood_derivatives,
    calculate_populations_log_likelihood,
    calculate_populations_log_likelihood_derivatives,
)


def log_likelihood(
    birth_rate: float,
    death_rate: float,
    ages: NDArray[np.float64] | float,
    observation_time: float,
    spine_ages: NDArray[np.float64] | None = None,
    ns: NDArray[np.int64] | None = None,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> np.float64:
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
    birth_rate : float
        Birth rate parameter λ > 0, or log(λ) if log_scale_params is True.
    death_rate : float
        Death rate parameter μ > 0, or log(μ) if log_scale_params is True.
    ages : ndarray[np.float64] | float
        Observed ages of manuscripts. A scalar is treated as a single-element array.
        All ages must satisfy 0 <= age <= observation_time.
    observation_time : float
        Observation time T (> 0) at which the population/ages are observed.
    spine_ages : ndarray[np.float64] | None, optional
        Ages to be treated as spines (special conditional observations). If provided,
        these ages must also satisfy 0 <= age <= observation_time. Mutually exclusive
        with is_ordered=True.
    ns : ndarray[np.int64] | None, optional
        Counts per population/class that sum to the total number of ages. If None,
        a single population with one count per provided age is assumed.
    is_ordered : bool, optional
        If True, the first element of ages is treated as a spine and the rest as
        ordinary observations. Cannot be used together with spine_ages.
    log_scale_params : bool, optional
        If True, treat birth_rate and death_rate as log-scale parameters.

    Returns
    -------
    np.float64
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
    # Normalize ages and spine_ages to numpy arrays so len() and vectorized ops are valid
    ages = np.atleast_1d(np.array(ages, dtype=np.float64))
    if spine_ages is not None:
        spine_ages = np.atleast_1d(np.array(spine_ages, dtype=np.float64))

    if is_ordered and (spine_ages is not None):
        raise ValueError("Cannot have both is_ordered=True and spine_ages provided.")
    # Input validation
    if not (np.isfinite(birth_rate) and np.isfinite(death_rate)):
        return np.float64(-np.inf)

    if not log_scale_params and (birth_rate <= 0 or death_rate <= 0):
        return np.float64(-np.inf)

    # Check age range
    if observation_time <= 0:
        return np.float64(-np.inf)

    if np.any(ages < 0) or np.any(ages > observation_time):
        return np.float64(-np.inf)

    if spine_ages is not None and (
        np.any(spine_ages < 0) or np.any(spine_ages > observation_time)
    ):
        return np.float64(-np.inf)

    # Total number of observed manuscripts
    if ns is None:
        total_pop_size = len(np.atleast_1d(ages))
        ns = np.array([total_pop_size], dtype=np.int64)
    else:
        total_pop_size = sum(ns)

    if not total_pop_size == (
        len(np.atleast_1d(ages))
        + (len(np.atleast_1d(spine_ages)) if spine_ages is not None else 0)
    ):
        raise ValueError(
            "Sum of the populations must equal the number of ages provided."
        )

    # 1. Calculate log(P_n(T)) - log probability of n individuals at time T
    log_P_n_T = calculate_populations_log_likelihood(
        lambda_params=birth_rate,
        mu_params=death_rate,
        times=observation_time,
        ns=ns,
        log_scale_params=log_scale_params,
    )

    # Check if population probability calculation succeeded
    if not np.isfinite(log_P_n_T):
        return np.float64(-np.inf)
    if total_pop_size == 0:
        return np.float64(log_P_n_T)

    ages_log_lik = 0.0

    if is_ordered or (spine_ages is not None):
        if is_ordered:
            ages = np.array(ages, dtype=np.float64)
            spine_ages = np.array([ages[0]], dtype=np.float64)
            ages = ages[1:]
        # Calculate log age density
        assert spine_ages is not None
        ages_log_lik += calculate_ages_log_likelihood(
            birth_rate,
            death_rate,
            ages_t=spine_ages,
            T=observation_time,
            log_scale_params=log_scale_params,
            are_spines=True,
        )
    if len(ages) > 0:
        ages_log_lik += calculate_ages_log_likelihood(
            birth_rate,
            death_rate,
            ages,
            observation_time,
            log_scale_params=log_scale_params,
            are_spines=False,
        )

    # Check if age density calculation succeeded
    if not np.isfinite(ages_log_lik):
        return np.float64(-np.inf)

    total_log_likelihood = log_P_n_T + ages_log_lik

    # Final numerical validation
    if not np.isfinite(total_log_likelihood):
        return np.float64(-np.inf)

    return np.float64(total_log_likelihood)


def log_likelihood_gradient(
    params: NDArray[np.float64],
    ages: NDArray[np.float64] | float,
    observation_time: float,
    spine_ages: NDArray[np.float64] | float | None = None,
    ns: NDArray[np.int64] | None = None,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
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
    params : NDArray[np.float64]
        Length-2 array containing the birth and death parameters in the order
        [birth_rate, death_rate]. If `log_scale_params` is True these are the
        log-parameters (log(birth), log(death)); otherwise they are in the
        natural scale.
    ages : NDArray[np.float64] | float
        Observed manuscript ages (can be a scalar or a 1-D array). All ages
        must lie in [0, observation_time].
    observation_time : float
        Positive scalar giving the observation time T. Ages must be in
        the interval [0, T].
    spine_ages : NDArray[np.float64] | float | None, optional
        Optional ages that are treated as spine observations. Cannot be
        provided if `is_ordered` is True. If provided, must lie in [0, T].
    ns : NDArray[np.int64] | None, optional
        Optional array of population counts per grouping. If omitted the total
        population is assumed to be len(np.atleast_1d(ages)). When provided, the sum(ns) must
        equal len(np.atleast_1d(ages)).
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
    NDArray[np.float64]
        A length-2 numpy array of dtype float64 giving the gradient of the
        log-likelihood with respect to the birth and death parameters in the
        same ordering as `params`. If `log_scale_params` is True the gradient
        is with respect to the log-parameters.

        When `log_scale_params` is False and either `birth_rate` <= 0 or
        `death_rate` <= 0 the function returns np.array([np.nan, np.nan])
        to indicate an invalid original-scale input.

    Raises
    ------
    ValueError
        - if `params` does not have length 2,
        - if any parameter is not finite,
        - if `is_ordered` is True and `spine_ages` is also provided,
        - if `observation_time` is not positive,
        - if any value in `ages` or `spine_ages` lies outside [0, observation_time],
        - if `ns` is provided and sum(ns) != len(np.atleast_1d(ages)).
    TypeError, IndexError, etc.
        Programming errors raise exceptions for easier debugging.

    Example:
        >>> # params as log-parameters
        >>> grad = log_likelihood_gradient(
                np.log([1.2, 0.3]), ages=[0.1, 0.4], observation_time=1.0, log_scale_params=True
            )
    """
    if len(params) != 2:
        raise ValueError("params must have length 2")
    if is_ordered and (spine_ages is not None):
        raise ValueError("Cannot have both is_ordered=True and spine_ages provided.")

    birth_rate, death_rate = params
    # Input validation
    if not (np.isfinite(birth_rate) and np.isfinite(death_rate)):
        raise ValueError("Parameters must be finite.")

    if not log_scale_params:
        # Parameters are in original scale, validate directly
        if birth_rate <= 0 or death_rate <= 0:
            return np.array([np.nan, np.nan], dtype=np.float64)

    # Check age range
    if observation_time <= 0:
        raise ValueError("Observation time must be positive.")

    # Normalize ages and spine_ages to numpy arrays so len() and vectorized ops are valid
    ages = np.atleast_1d(np.array(ages, dtype=np.float64))
    if spine_ages is not None:
        spine_ages = np.atleast_1d(np.array(spine_ages, dtype=np.float64))

    if np.any(ages < 0) or np.any(ages > observation_time):
        raise ValueError("Ages must be in the range [0, observation_time].")

    if spine_ages is not None and (
        np.any(spine_ages < 0) or np.any(spine_ages > observation_time)
    ):
        raise ValueError("Spine ages must be in the range [0, observation_time].")

    # Total number of observed manuscripts
    if ns is None:
        total_pop_size = len(np.atleast_1d(ages))
        ns = np.array([total_pop_size], dtype=np.int64)
    else:
        total_pop_size = sum(ns)

    if not total_pop_size == len(np.atleast_1d(ages)) + (
        len(np.atleast_1d(spine_ages)) if spine_ages is not None else 0
    ):
        raise ValueError(
            "Sum of the populations must equal the number of ages provided."
        )

    # Gradient components
    grad_log_P_n_T = calculate_populations_log_likelihood_derivatives(
        birth_rate,
        death_rate,
        observation_time,
        ns,
        log_scale_params=log_scale_params,
    )

    if total_pop_size == 0:
        return grad_log_P_n_T

    # Sequential computation for small datasets
    grad_ages = np.zeros(2, dtype=np.float64)
    if is_ordered or (spine_ages is not None):
        if is_ordered:
            ages = np.array(ages, dtype=np.float64)
            spine_ages = np.array([ages[0]], dtype=np.float64)
            ages = ages[1:]
        assert spine_ages is not None
        grad_ages += calculate_ages_log_likelihood_derivatives(
            birth_rate,
            death_rate,
            spine_ages,
            observation_time,
            log_scale_params=log_scale_params,
            are_spines=True,
        )

    if len(np.atleast_1d(ages)) > 0:
        grad_ages += calculate_ages_log_likelihood_derivatives(
            birth_rate,
            death_rate,
            ages,
            observation_time,
            log_scale_params=log_scale_params,
            are_spines=False,
        )

    return grad_log_P_n_T + grad_ages


def negative_log_likelihood(
    params: NDArray[np.float64],
    ages: NDArray[np.float64] | float,
    observation_time: float,
    spine_ages: NDArray[np.float64] | None = None,
    ns: NDArray[np.int64] | None = None,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Compute the negative log-likelihood for use with numerical optimizers.

    This function is a thin wrapper around log_likelihood that negates its
    value so it can be minimized by optimizers such as scipy.optimize.
    It expects a two-element parameter vector representing the model's
    birth and death rates (or their logarithms, see log_scale_params).

    Parameters
    ----------
    params : numpy.ndarray[np.float64]
        Parameter vector of length 2 containing [birth_rate, death_rate].
        If `log_scale_params` is True then these are the logarithms of the
        original parameters; otherwise they are interpreted on their natural scale.
        If `params` has length != 2 the function returns np.inf.
    ages : numpy.ndarray[np.float64] | float
        Ages of observed manuscripts (or a single age). May be a 1-D array
        of observation ages used by the likelihood computation.
    observation_time : float
        The observation time T (end of the observation window) used by the model.
    spine_ages : numpy.ndarray[np.float64] | None, optional
        Optional spine ages required by some likelihood formulations. If not
        provided, a default or simplified likelihood is computed (depending on
        the underlying log_likelihood implementation).
    ns : numpy.ndarray[np.int64] | None, optional
        Optional counts/weights associated with each age. If omitted, each
        observation is treated as a single count.
    is_ordered : bool, default False
        If True, interpret `ages` (and `ns` when provided) as ordered-event data
        and use the ordered-data likelihood branch in the underlying function.
    log_scale_params : bool, default False
        If True, indicates that `params` are provided in log-space and the
        underlying log_likelihood should convert them back to the original scale
        before evaluating the likelihood.

    Returns
    -------
    numpy.float64
        The negative log-likelihood (i.e. -log_likelihood). This is returned
        as a numpy.float64 scalar suitable for use with scipy optimizers.
        If `params` does not have length 2, returns numpy.inf.

    Notes
    -----
    - This wrapper does not validate parameter bounds beyond checking length;
      any additional constraints (e.g. non-negativity) should be enforced by the
      optimizer or by inspecting the returned value.
    - Returns -inf (as np.inf when negated) for numerical issues in the underlying
      log_likelihood computation.
    - Programming errors in underlying functions will raise exceptions.
    - Designed to be cheap to call repeatedly by optimization routines.
    """
    if len(params) != 2:
        return np.float64(np.inf)

    # Extract log-scale parameters and convert to original scale
    birth_rate, death_rate = params

    result = -log_likelihood(
        birth_rate=birth_rate,
        death_rate=death_rate,
        ages=ages,
        observation_time=observation_time,
        spine_ages=spine_ages,
        ns=ns,
        is_ordered=is_ordered,
        log_scale_params=log_scale_params,
    )

    return result
