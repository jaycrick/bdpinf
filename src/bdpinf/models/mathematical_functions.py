"""
Mathematical functions for birth-and-death process calculations.

This module contains pure mathematical functions for calculating birth-and-death
process probabilities and densities. These functions are isolated from the
main BirthDeathProcess class for better testability and modularity.
"""

import numpy as np
import scipy
from numpy.typing import NDArray


def calculate_populations_log_likelihood(
    lambda_params: float | NDArray[np.float64],
    mu_params: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    ns: NDArray[np.integer] | int,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate the log-probability (log-likelihood) of observing given population counts
    under a birth-and-death process, with support for piecewise-constant rate epochs.

    This function computes log p_n(T) for each count n in `ns` using closed-form
    expressions for the birth-and-death process probabilities. The computed log-probabilities
    are summed and returned as a single np.float64 value. Invalid or numerically
    unstable inputs return -inf (np.float64(-np.inf)).

    Parameters
    ----------
    lambda_params : float or NDArray[np.float64]
        Birth rate(s) λ. If `log_scale_params` is True, interpreted as log(λ).

        - **Scalar**: Constant birth rate throughout the time period.
        - **Array**: Piecewise-constant birth rates for time epochs. The i-th element
          is the rate during epoch i, which ends at time[i]. Array length must match
          `time` and `mu_param` if those are also arrays.

    mu_params : float or NDArray[np.float64]
        Death rate(s) μ. If `log_scale_params` is True, interpreted as log(μ).

        - **Scalar**: Constant death rate throughout the time period.
        - **Array**: Piecewise-constant death rates for time epochs. The i-th element
          is the rate during epoch i, which ends at time[i]. Array length must match
          `time` and `lambda_param` if those are also arrays.

    time : float or NDArray[np.float64]
        Time(s) T at which the distribution is evaluated. Must be ≥ 0.

        - **Scalar**: Single observation time point.
        - **Array**: Epoch end times for piecewise-constant rate model. The i-th element
          is the end time of epoch i, during which rates lambda_param[i] and mu_param[i]
          apply. Times must be monotonically increasing. Array length must match
          `lambda_param` and `mu_param` if those are also arrays.

    ns : numpy.ndarray[int] or int
        Non-negative integer or array of non-negative integers representing the
        population counts n for which to evaluate the log-probabilities.

    log_scale_params : bool, optional
        If True, `lambda_param` and `mu_param` are treated as natural logarithms
        of the rates and are exponentiated internally. Default is False.

    Returns
    -------
    numpy.float64
        Sum of log p_n(T) across the provided `ns`. Returns np.float64(-np.inf) for
        numerically invalid inputs (negative time, negative counts, non-positive rates
        when not in log-space, or α_t/β_t not in (0,1)). Programming errors (type errors,
        indexing errors) will raise exceptions.

    Behavior and formulas
    ---------------------
    For constant rates (scalar inputs):

    Let α_t and β_t denote auxiliary probabilities defined by the birth-and-death process:
        α_t = μ (e^{(λ-μ) T} - 1) / (λ e^{(λ-μ) T} - μ)
        β_t = (λ / μ) * α_t

    The log-probabilities used are:
        log p_n(T) = log(α_t)                                           for n = 0
        log p_n(T) = log(1 - α_t) + log(1 - β_t) + (n - 1) * log(β_t)  for n ≥ 1

    For piecewise-constant rates (array inputs):

    The process evolves through sequential epochs with different rates. The calculation
    compounds the effects across epochs to compute the final distribution at the last
    epoch's end time.

    The implementation handles the limiting case λ ≈ μ by using the limiting form
    α_t = β_t = (λ T) / (1 + λ T) to avoid catastrophic cancellation.

    Edge cases
    ----------
    - Negative time or negative counts produce -inf.
    - When `log_scale_params` is False, non-positive λ or μ produce -inf.
    - If α_t or β_t are not strictly between 0 and 1, the function returns -inf.
    - Non-finite intermediate values (overflow/underflow) produce -inf.
    - Array inputs must have matching lengths (lambda_param, mu_params, time).
    - Epoch times must be monotonically increasing when using array inputs.
    - The function accepts scalar or array-like `ns`; intermediate computations
      preserve array shape, but the returned value is the scalar sum of log-probabilities.
    - Type errors, shape mismatches, and other programming errors raise exceptions.

    Examples
    --------
    # Constant rates (scalar inputs)
    >>> calculate_populations_log_likelihood(0.5, 0.3, 10.0, 2)

    # Vectorized population counts
    >>> calculate_populations_log_likelihood(0.5, 0.3, 10.0, np.array([0, 1, 2]))

    # Piecewise-constant rates (array inputs)
    >>> # Two epochs: [0, 5) with λ=0.5, μ=0.3; [5, 10) with λ=0.4, μ=0.2
    >>> calculate_populations_log_likelihood(
    ...     lambda_param=np.array([0.5, 0.4]),
    ...     mu_param=np.array([0.3, 0.2]),
    ...     time=np.array([5.0, 10.0]),
    ...     ns=2
    ... )
    """
    # Input validation
    if np.any(ns < 0) or np.any(times < 0):
        raise ValueError("ns must be non-negative and time must be non-negative")

    # Ensure consistent input types: either all arrays or all scalars
    is_array_lambda = isinstance(lambda_params, np.ndarray)
    is_array_mu = isinstance(mu_params, np.ndarray)
    is_array_time = isinstance(times, np.ndarray)

    if not (is_array_lambda == is_array_mu == is_array_time):
        raise ValueError(
            "lambda_param, mu_params, and time must all be arrays or all scalars. "
            f"Got types: lambda={type(lambda_params).__name__}, mu={type(mu_params).__name__}, time={type(times).__name__}"
        )

    if (
        isinstance(times, np.ndarray)
        and isinstance(lambda_params, np.ndarray)
        and isinstance(mu_params, np.ndarray)
    ):
        if not (len(lambda_params) == len(mu_params) == len(times)):
            raise ValueError(
                f"Array inputs must have matching lengths. "
                f"Got: lambda_params length={len(lambda_params)}, "
                f"mu_params length={len(mu_params)}, time length={len(times)}"
            )
        # Handle parameter scaling
        if log_scale_params:
            lambda_val_array = np.exp(lambda_params)
            mu_val_array = np.exp(mu_params)
        else:
            # Validate non-negative parameters
            if np.any(lambda_params < 0) or np.any(mu_params < 0):
                return np.float64(-np.inf)
            lambda_val_array = lambda_params
            mu_val_array = mu_params
        if not np.all(np.diff(times) > 0):
            raise ValueError(
                f"Epoch times must be strictly monotonically increasing. "
                f"Got time array: {times}"
            )

        alpha_t = calculate_extinction_probability(
            lambda_val_array, mu_val_array, times
        )

        beta_t = 1 - _calculate_geometric_parameter(
            lambda_val_array, mu_val_array, times
        )
    else:
        if (
            not isinstance(lambda_params, float)
            or not isinstance(mu_params, float)
            or not isinstance(times, float)
        ):
            raise ValueError(
                f"When using scalar inputs, lambda_params, mu_params, and times must all be floats. "
                f"Got types: lambda_params={type(lambda_params).__name__}, "
                f"mu_params={type(mu_params).__name__}, times={type(times).__name__}. "
                f"Note: numpy scalar types (np.float64, np.int32, etc.) are not accepted - use Python float type."
            )
        # Handle parameter scaling
        if log_scale_params:
            log_lambda = lambda_params
            log_mu = mu_params
            lambda_val = np.exp(lambda_params)
            mu_val = np.exp(mu_params)
        else:
            # Validate non-negative parameters
            if np.any(lambda_params < 0) or np.any(mu_params < 0):
                return np.float64(-np.inf)
            log_lambda = np.log(lambda_params)
            log_mu = np.log(mu_params)
            lambda_val = lambda_params
            mu_val = mu_params
        if abs(log_lambda - log_mu) < 1e-12:
            # Limiting case: λ ≈ μ
            rate_time_product = lambda_val * times
            alpha_t = beta_t = rate_time_product / (1 + rate_time_product)
        else:
            # General case: λ ≠ μ
            growth_rate = lambda_val - mu_val
            exponent = growth_rate * times
            exp_term = np.exp(exponent)
            if not np.isfinite(exp_term):
                return np.float64(-np.inf)
            denominator = lambda_val * exp_term - mu_val
            alpha_t = mu_val * (exp_term - 1) / denominator

            # Calculate β_t = (λ/μ) * α_t
            beta_t = (lambda_val / mu_val) * alpha_t

    # Validate α_t and β_t are in valid range (0, 1)
    if not ((0 < alpha_t < 1) and (0 < beta_t < 1)):
        return np.float64(-np.inf)

    # Initialize log probability array
    log_probability = np.zeros_like(ns, dtype=np.float64)

    # Handle n == 0 case
    zero_mask = ns == 0
    if np.any(zero_mask):
        log_probability[zero_mask] = np.log(alpha_t)

    # Handle n > 0 case
    nonzero_mask = np.invert(zero_mask)
    if np.any(nonzero_mask):
        log_term1 = np.log(1 - alpha_t)
        log_term2 = np.log(1 - beta_t)
        log_term3 = (ns - 1) * np.log(beta_t)

        # Check all terms are finite
        if not (
            np.isfinite(log_term1)
            and np.isfinite(log_term2)
            and np.all(np.isfinite(log_term3))
        ):
            return np.float64(-np.inf)

        log_probability[nonzero_mask] = log_term1 + log_term2 + log_term3[nonzero_mask]

    # Final numerical check
    if not np.all(np.isfinite(log_probability)):
        return np.float64(-np.inf)

    return np.float64(np.sum(log_probability))


def calculate_populations_log_likelihood_derivatives(
    lambda_param: float,
    mu_param: float,
    time: float,
    ns: NDArray[np.int64] | int,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of the log age-density f(t) with respect to the birth rate (λ)
    and death rate (μ).
    Parameters
    ----------
    lambda_param : float
        Birth rate λ > 0. If `log_scale_params` is True, this argument is interpreted as
        log(λ).
    mu_param : float
        Death rate μ > 0. If `log_scale_params` is True, this argument is interpreted as
        log(μ).
    time : float
        Age t at which the density is evaluated (t >= 0).
    ns : int or array_like of int
        Observed lineage counts (n). The function is vectorized over `ns`. Entries equal
        to 0 are treated with the n=0 formula; entries > 0 use the n>0 formula.
    log_scale_params : bool, optional
        If True, `lambda_param` and `mu_param` are provided in log-scale and will be
        exponentiated internally to obtain λ and μ. Default is False.
    Returns
    -------
    numpy.ndarray
        A 2-by-... ndarray of dtype np.float64 where the first row contains ∂ log f(t)/∂λ
        and the second row contains ∂ log f(t)/∂μ. The trailing shape matches that of `ns`
        (scalar input yields shape (2,), array input yields shape (2, n_elements)).
    Behavior and numerical notes
    ----------------------------
    - The function assumes positive rates (λ > 0 and μ > 0). If `log_scale_params` is False,
      passing non-positive values will raise errors when taking logarithms or produce
      invalid results.
    - To avoid catastrophic cancellation when λ and μ are nearly equal, the implementation
      detects the limiting case by comparing log(λ) and log(μ) with a tolerance of 1e-12
      and substitutes the analytic limit formulas for that case.
    - The computation is vectorized across `ns`: boolean masks are used to select and apply
      the appropriate closed-form expressions for n = 0 and n > 0.
    - Large `time` and/or large differences between λ and μ can produce very large
      exponentials; care should be taken to avoid overflow/underflow in such regimes.
    Examples
    --------
    Scalar n:
    >>> calculate_populations_log_likelihood_derivatives(0.2, 0.1, time=5.0, ns=0)
    array([d_lambda_scalar, d_mu_scalar])

    Vectorized n:
    >>> calculate_populations_log_likelihood_derivatives(0.2, 0.1, time=5.0, ns=np.array([0,1,2]))
    array([sum_d_lambda, sum_d_mu])
    """

    # Handle parameter scaling
    if log_scale_params:
        log_lambda = lambda_param
        log_mu = mu_param
        # Convert log parameters back to original scale
        lambda_val = np.exp(lambda_param)
        mu_val = np.exp(mu_param)
    else:
        lambda_val = lambda_param
        mu_val = mu_param
        log_lambda = np.log(lambda_param)
        log_mu = np.log(mu_param)

    d_lambda = np.zeros_like(ns, dtype=np.float64)
    d_mu = np.zeros_like(ns, dtype=np.float64)

    zero_mask = ns == 0
    non_zero_mask = np.invert(zero_mask)

    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ

        if np.any(zero_mask):
            d_lambda[zero_mask] = 1 / (lambda_val + lambda_val**2 * time)
        if np.any(non_zero_mask):
            numerator = ns - 1 - 2 * time * lambda_val
            denominator = lambda_val * (1 + time * lambda_val)
            d_lambda[non_zero_mask] = numerator / denominator

        return np.array([np.sum(d_lambda), np.sum(d_lambda)], dtype=np.float64)

    # General case: λ ≠ μ
    growth_rate = lambda_val - mu_val
    if np.any(zero_mask):
        lambda_numerator = np.exp(time * (growth_rate)) * (
            np.exp(time * (growth_rate)) - lambda_val * time + mu_val * time - 1
        )
        lambda_denominator = (np.exp(time * (growth_rate)) - 1) * (
            lambda_val * np.exp(time * (growth_rate)) - mu_val
        )
        mu_numerator = np.exp(time * (growth_rate)) * (
            lambda_val * (np.exp(time * (growth_rate)) - mu_val * time - 1)
            + time * mu_val**2
        )
        mu_denominator = (
            mu_val
            * (np.exp(time * (growth_rate)) - 1)
            * (lambda_val * np.exp(time * (growth_rate)) - mu_val)
        )
        d_lambda[zero_mask] = lambda_numerator / lambda_denominator
        d_mu[zero_mask] = mu_numerator / mu_denominator
    if np.any(non_zero_mask):
        lambda_numerator_1 = mu_val * (
            mu_val
            - mu_val * ns
            + lambda_val * (ns - mu_val * time + 1)
            + lambda_val**2 * time
        )
        lambda_numerator_2 = np.exp(time * (growth_rate)) * (
            ns * (growth_rate) * (lambda_val * time * (growth_rate) - mu_val)
            - mu_val * (3 * lambda_val + mu_val)
        )
        lambda_numerator_3 = (
            lambda_val
            * np.exp(2 * time * (growth_rate))
            * (2 * mu_val + lambda_val * time * (mu_val - lambda_val))
        )
        lambda_numerator = lambda_numerator_1 + lambda_numerator_2 + lambda_numerator_3
        lambda_denominator = (
            lambda_val
            * (growth_rate)
            * (np.exp(time * (growth_rate)) - 1)
            * (lambda_val * np.exp(time * (growth_rate)) - mu_val)
        )

        mu_numerator_1 = (
            lambda_val**2
            * time
            * np.exp(time * (growth_rate))
            * (np.exp(time * (growth_rate)) - ns)
        )
        mu_numerator_2 = -lambda_val * (
            -(2 * mu_val * ns * time + ns + 3) * np.exp(time * (growth_rate))
            + ns
            + (mu_val * time + 2) * np.exp(2 * time * (growth_rate))
            + mu_val * time
            + 1
        )
        mu_numerator_3 = -mu_val * (
            (mu_val * ns * time + ns - 1) * np.exp(time * (growth_rate))
            - ns
            - mu_val * time
            + 1
        )
        mu_numerator = mu_numerator_1 + mu_numerator_2 + mu_numerator_3
        mu_denominator = (
            (growth_rate)
            * (np.exp(time * (growth_rate)) - 1)
            * (lambda_val * np.exp(time * (growth_rate)) - mu_val)
        )
        d_lambda[non_zero_mask] = lambda_numerator[non_zero_mask] / lambda_denominator
        d_mu[non_zero_mask] = mu_numerator[non_zero_mask] / mu_denominator

    return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)


def calculate_spine_ages_log_likelihood(
    lambda_param: float,
    mu_param: float,
    ages_t: NDArray[np.float64] | float,
    T: float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate the log-likelihood of manuscript ages for the special-case "spine"
    individual under a birth-and-death process with birth rate λ and death rate μ.
    This function supports scalar or array-like ages and returns the sum of log-likelihoods
    across all provided ages (i.e. the joint log-likelihood for independent age observations).
    The age distribution for the spine is a mixture of a continuous density on (0, T) and a
    point mass at t = T. The implementation handles the general case λ != μ and the limiting
    case λ = μ. Parameters may be supplied either on the
    original scale or as log-parameters controlled by `log_scale_params`.
    Parameters
    ----------
    lambda_param : float
        Birth rate (λ). If `log_scale_params` is True this is log(λ).
    mu_param : float
        Death rate (μ). If `log_scale_params` is True this is log(μ).
    ages_t : NDArray[np.float64] | float
        Age or array of ages of manuscripts (each t must satisfy 0 <= t <= T).
    T : float
        Total observation time (must satisfy T >= max(ages_t)).
    log_scale_params : bool, optional
        If True, `lambda_param` and `mu_param` are interpreted as log(λ) and log(μ)
        and are exponentiated inside the function. Default is False.
    Returns
    -------
    np.float64
        The sum of the log age densities for all entries of `ages_t` (a scalar).
        Returns -inf for numerically invalid inputs (negative ages, ages > T, non-positive
        rates, or non-finite intermediate values). Programming errors raise exceptions.
    Notes
    -----
    - Input validation: the function returns -inf if any age is negative or greater than T,
      or if λ or μ are non-positive when provided on the original scale.
    - Boundary behavior: ages exactly equal to T are treated as the discrete mass at T;
      ages strictly less than T use the continuous density component.
    - Limiting case: when λ is numerically very close to μ the function uses the analytic
      limit form to avoid catastrophic cancellation.
    - Numerical robustness: the implementation checks signs of intermediate expressions and
      finiteness of results; if any intermediate quantity is invalid (non-finite or a sign
      mismatch that would lead to a non-real log) the function returns -inf.
    - Programming errors (type mismatches, array indexing errors) will raise exceptions
      rather than returning -inf.
    - The returned value is a scalar np.float64 equal to the summed log-density across
      the provided ages (suitable for use inside likelihood computations or MCMC).
    """
    # Input validation
    if np.any(ages_t < 0) or np.any(ages_t > T):
        return np.float64(-np.inf)

    # Handle parameter scaling
    if log_scale_params:
        log_lambda = lambda_param
        log_mu = mu_param
        # Convert log parameters back to original scale
        lambda_val = np.exp(lambda_param)
        mu_val = np.exp(mu_param)
    else:
        # Validate positive parameters
        if lambda_param <= 0 or mu_param <= 0:
            return np.float64(-np.inf)
        lambda_val = lambda_param
        mu_val = mu_param
        log_lambda = np.log(lambda_param)
        log_mu = np.log(mu_param)

    at_boundary = ages_t == T
    not_at_boundary = np.invert(at_boundary)

    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ
        boundary_val = np.log(1 + lambda_val * T) - lambda_val * T
        log_density = np.where(
            at_boundary,
            boundary_val,
            2 * np.log(lambda_val) + np.log(ages_t) - lambda_val * ages_t,
        )
        return np.float64(np.sum(log_density))

    # General case: λ ≠ μ

    # Initialize arrays for numerator and denominator
    log_numerator = np.zeros_like(ages_t, dtype=np.float64)
    denominator = lambda_val - mu_val
    sign_denominator = np.sign(denominator)
    log_denominator = np.log(np.abs(denominator))

    if np.any(at_boundary):
        numerator_boundary = lambda_val * np.exp(-mu_val * T) - mu_val * np.exp(
            -lambda_val * T
        )

        if not np.all(np.sign(numerator_boundary) == sign_denominator):
            return np.float64(-np.inf)

        log_numerator[at_boundary] = np.log(np.abs(numerator_boundary))

    if np.any(not_at_boundary):
        numerator_1 = np.exp(-mu_val * T) - np.exp(-lambda_val * T)

        if not np.all(np.sign(numerator_1) == sign_denominator):
            return np.float64(-np.inf)

        log_numerator[not_at_boundary] = (
            np.log(lambda_val) + np.log(mu_val) + np.log(np.abs(numerator_1))
        )

    # Check for numerical validity
    if not (np.all(np.isfinite(log_numerator)) and np.isfinite(log_denominator)):
        return np.float64(-np.inf)

    # Calculate final log density (sum across all ages)
    log_density = np.sum(log_numerator - log_denominator)

    # Final numerical check
    if not np.isfinite(log_density):
        return np.float64(-np.inf)

    return np.float64(log_density)


def calculate_non_spine_ages_log_likelihood(
    lambda_params: NDArray[np.float64] | float,
    mu_params: NDArray[np.float64] | float,
    ages_t: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate the log-likelihood (log density) of observing one or more manuscript ages
    under a birth-and-death process conditional on survival at observation time T.

    The age density for a non-spine individual is
        f(t) = [λ e^{-λ t} - μ e^{-(λ-μ) T} e^{-μ t}] / [1 - e^{-(λ-μ) T}],
    for λ != μ, and in the limiting case λ == μ it reduces to
        f(t) = [1 + λ (T - t)] / T * e^{-λ t}.

    This function returns the sum of log f(t) across the elements of ages_t (i.e. a
    log-likelihood for independent observations). It is implemented to accept either
    scalar or array-like ages_t and is numerically stabilized around the λ ≈ μ case.

    Parameters
    ----------
    lambda_param : float
        Birth rate (λ > 0). If log_scale_params is True this argument is interpreted
        as log(λ).
    mu_param : float
        Death rate (μ > 0). If log_scale_params is True this argument is interpreted
        as log(μ).
    ages_t : float or numpy.ndarray of float
        Age(s) of manuscript(s) t (t >= 0). If an array is provided the returned
        value is the sum of log-densities over its elements.
    T : float
        Observation time (time from process start to present). Must satisfy T >= t
        for each provided t.
    log_scale_params : bool, optional
        If True, lambda_param and mu_param are provided on the log scale and are
        exponentiated inside the function. Default is False.

    Returns
    -------
    numpy.float64
        The summed log-likelihood (log density) across ages_t. Returns -inf for
        numerically invalid inputs (negative ages, T < t, non-positive rates when not
        using log-scale, or non-finite intermediate values). Programming errors (type
        errors, shape mismatches) raise exceptions.

    Notes
    -----
    - The function is vectorized over ages_t and will broadcast computations
      accordingly.
    - A numerically stable branch is used when λ and μ are nearly equal to avoid
      cancellation errors (uses the analytic λ = μ limit).
    - Numerical issues (non-finite values, invalid domains) return -inf.
    - Programming errors raise exceptions for easier debugging.

    Examples
    --------
    >>> # scalar inputs
    >>> calculate_non_spine_ages_log_likelihood(1.0, 0.5, 2.0, 5.0)
    -2.03...

    >>> # log-scale parameters and multiple ages
    >>> calculate_non_spine_ages_log_likelihood(np.log(1.0), np.log(0.5),
    ...                                         np.array([1.0, 2.0]), 5.0,
    ...                                         log_scale_params=True)
    -3.01...
    """
    times = np.atleast_1d(times)
    end_time = times[-1]

    if np.any(np.diff(np.insert(times, 0, 0.0)) <= 0):
        raise ValueError(f"times must be strictly increasing. Got times: {times}")
    # Input validation
    if np.any(ages_t < 0) or np.any(ages_t > end_time):
        return np.float64(-np.inf)

    if np.shape(lambda_params) != np.shape(mu_params):
        raise ValueError(
            f"lambda_param and mu_param must have the same shape. "
            f"Got shapes: lambda_param={np.shape(lambda_params)}, mu_param={np.shape(mu_params)}"
        )

    if np.shape(lambda_params) != np.shape(times):
        raise ValueError(
            f"lambda_param and times must have the same shape. "
            f"Got shapes: lambda_param={np.shape(lambda_params)}, times={np.shape(times)}"
        )

    if len(times) > 1:
        inverted_times = np.flip(end_time - times)
        reflected_birth_rates = np.flip(mu_params)
        reflected_death_rates = np.flip(lambda_params)
        return _log_likelihood_age_given_extinction(
            ages_t,
            reflected_birth_rates,
            reflected_death_rates,
            inverted_times,
            log_scale_params,
        )

    # Handle parameter scaling
    # Extract scalar values from single-element arrays
    lambda_scalar = float(
        lambda_params[0] if isinstance(lambda_params, np.ndarray) else lambda_params
    )
    mu_scalar = float(mu_params[0] if isinstance(mu_params, np.ndarray) else mu_params)

    if log_scale_params:
        log_lambda = lambda_scalar
        log_mu = mu_scalar
        # Convert log parameters back to original scale
        lambda_val = np.exp(lambda_scalar)
        mu_val = np.exp(mu_scalar)
    else:
        # Validate positive parameters
        if lambda_scalar <= 0 or mu_scalar <= 0:
            return np.float64(-np.inf)
        lambda_val = lambda_scalar
        mu_val = mu_scalar
        log_lambda = np.log(lambda_scalar)
        log_mu = np.log(mu_scalar)

    # Check for numerical stability in parameter space
    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ

        log_numerator = -lambda_val * ages_t + np.log(
            1 + lambda_val * (end_time - ages_t)
        )
        log_denominator = np.log(end_time)

        # Check for numerical validity
        if not (np.isfinite(log_numerator) and np.isfinite(log_denominator)):
            return np.float64(-np.inf)

    else:
        # General case: λ ≠ μ
        growth_rate = lambda_val - mu_val
        growth_rate_T = growth_rate * end_time

        # Calculate numerator terms
        numerator_term1 = lambda_val * np.exp(-lambda_val * ages_t)
        numerator_term2 = mu_val * np.exp(-growth_rate_T) * np.exp(-mu_val * ages_t)

        numerator = numerator_term1 - numerator_term2
        denominator = 1 - np.exp(-growth_rate_T)

        if not np.any(np.sign(numerator) == np.sign(denominator)):
            return np.float64(-np.inf)

        log_numerator = np.log(abs(numerator))
        log_denominator = np.log(abs(denominator))

        # Check for numerical validity
        if not (np.isfinite(log_numerator).all() and np.isfinite(log_denominator)):
            return np.float64(-np.inf)

    # Calculate final log density
    log_density = np.sum(log_numerator - log_denominator)

    # Final numerical check
    if not np.isfinite(log_density):
        return np.float64(-np.inf)

    return np.float64(log_density)


def calculate_ages_log_likelihood(
    lambda_param: float,
    mu_param: float,
    ages_t: NDArray[np.float64] | float,
    T: float,
    log_scale_params: bool = False,
    are_spines: bool = False,
) -> np.float64:
    """
     Compute the log-density (log-likelihood) of manuscript ages under a birth-and-death model.

    The age density for a manuscript of age t observed within a total time window T is
    given by:
        f(t) = (λ e^{-λ t} - μ e^{-(λ-μ)T} e^{-μ t}) / (1 - e^{-(λ-μ)T})

    For numerical stability the computation is performed in log-space:
        log f(t) = log(λ e^{-λ t} - μ e^{-(λ-μ)T} e^{-μ t}) - log(1 - e^{-(λ-μ)T})

    This function delegates to a spine-aware implementation when are_spines is True.

        lambda_param (float):
            Birth rate λ. Expected to be positive. If log_scale_params is True,
            this is interpreted as log(λ).
        mu_param (float):
            Death rate μ. Expected to be positive. If log_scale_params is True,
            this is interpreted as log(μ).
        ages_t (float or array-like):
            Age(s) t of manuscript(s). Each t should satisfy 0 <= t <= T.
        T (float):
            Total observation time. Must satisfy T >= max(ages_t) and be non-negative.
        log_scale_params (bool, optional):
            If True, lambda_param and mu_param are provided on the log scale and will
            be exponentiated before use. Default: False.
        are_spines (bool, optional):
            If True, computation is routed to calculate_spine_ages_log_likelihood;
            otherwise it uses the non-spine implementation. Default: False.

        numpy.float64 or numpy.ndarray
            The log-density log f(t) for the provided age(s). If ages_t is scalar,
            a scalar numpy.float64 is returned; if array-like, an array of log-densities
            of corresponding shape is returned.

    Returns -inf (not an exception):
        - Non-positive rates (when not using log_scale_params)
        - Negative T or ages outside [0, T]
        - Non-finite intermediate values

    Raises:
        - Type errors, shape mismatches, and other programming errors will raise exceptions

    Notes:
        - The implementation is careful to compute the numerator in log-space to
          avoid catastrophic cancellation when the two exponential terms are similar.
        - For extremely small or large rate differences (λ - μ) the normalization
          term 1 - exp(-(λ-μ)T) may underflow; callers may wish to check parameter
          regimes or use higher-precision numerics if necessary.
        - This function is intended for use in likelihood calculations and can be
          vectorized over ages_t when supported by the delegated implementation.
    """
    if are_spines:
        return calculate_spine_ages_log_likelihood(
            lambda_param, mu_param, ages_t, T, log_scale_params
        )
    else:
        # calculate_non_spine_ages_log_likelihood requires array inputs
        # Wrap scalar parameters in single-element arrays for single-epoch case
        return calculate_non_spine_ages_log_likelihood(
            np.array([lambda_param]),
            np.array([mu_param]),
            ages_t,
            np.array([T]),
            log_scale_params,
        )


def calculate_non_spine_ages_log_likelihood_derivatives(
    lambda_param: float,
    mu_param: float,
    ages_t: NDArray[np.float64] | float,
    T: float,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of the log age-density f(t) with respect to the birth rate (λ)
    and death rate (μ) for non-spine (observed) manuscript ages.

    This function computes d/dλ log f(t) and d/dμ log f(t) for one or more observed ages
    under an exponential birth-death process observed up to time T. The implementation
    handles two regimes for numerical stability: the general case λ ≠ μ and a limiting
    case used when λ ≈ μ (checked in log-space). When multiple ages are provided,
    derivatives are computed for each age and then summed to produce scalar derivatives
    for the whole dataset.

    Parameters
    ----------
    lambda_param : float
        Birth rate parameter. If log_scale_params is False, this is λ > 0. If
        log_scale_params is True, this is log(λ).
    mu_param : float
        Death rate parameter. If log_scale_params is False, this is μ > 0. If
        log_scale_params is True, this is log(μ).
    ages_t : float or array-like
        Age(s) of the manuscript(s). Can be a scalar or a 1-D array of ages. Each
        entry should satisfy 0 <= age <= T.
    T : float
        Total observation time (a scalar). Must be greater than or equal to each
        entry in ages_t.
    log_scale_params : bool, optional
        If True, lambda_param and mu_param are interpreted as log(λ) and log(μ)
        respectively and are exponentiated internally. Default is False.

    Returns
    -------
    numpy.ndarray (dtype=np.float64)
        1-D array of length 2: [d_logf_d_lambda, d_logf_d_mu]. Each component is a
        scalar obtained by summing the per-age derivatives across the provided
        ages_t. The returned dtype is np.float64.

    Notes
    -----
    - The function expects strictly positive rates (λ > 0 and μ > 0) either directly
      or after exponentiating log-scale inputs.
    - For numerical stability when λ ≈ μ the function uses a limiting analytic
      expression evaluated when |log(λ) - log(μ)| < 1e-12.
    - Invalid parameter values may produce NaN or inf results; callers should validate
      inputs if needed.
    - The derivative values correspond to derivatives of the log of the age density,
      not the raw density.

    Examples
    --------
    # Single age, parameters on natural scale
    d_vals = calculate_non_spine_ages_log_likelihood_derivatives(0.5, 0.3, 2.0, 5.0)

    # Multiple ages, parameters given in log-scale
    log_lambda = np.log(0.5); log_mu = np.log(0.3)
    d_vals = calculate_non_spine_ages_log_likelihood_derivatives(
        log_lambda, log_mu, np.array([1.0, 2.5]), 5.0, log_scale_params=True
    )
    """

    # Handle parameter scaling
    if log_scale_params:
        log_lambda = lambda_param
        log_mu = mu_param
        # Convert log parameters back to original scale
        lambda_val = np.exp(lambda_param)
        mu_val = np.exp(mu_param)
    else:
        lambda_val = lambda_param
        mu_val = mu_param
        log_lambda = np.log(lambda_param)
        log_mu = np.log(mu_param)

    # Check for numerical stability in parameter space
    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ
        numerator = T + ages_t**2 * lambda_val - ages_t * (2 + lambda_val * T)
        denominator = 1 - ages_t * lambda_val + lambda_val * T
        d_mu = d_lambda = numerator / denominator
    else:
        # General case: λ ≠ μ
        # Pre-compute common exponentials and products to avoid redundant calculations
        lambda_ages_t = lambda_val * ages_t
        mu_ages_t = mu_val * ages_t
        growth_rate = lambda_val - mu_val
        T_growth_rate = T * growth_rate

        # Cache exponentials - these are expensive operations
        exp_lambda_t = np.exp(lambda_ages_t)
        exp_mu_t = np.exp(mu_ages_t)
        exp_T_growth = np.exp(T_growth_rate)
        exp_2T_growth = exp_T_growth * exp_T_growth  # Faster than np.exp(2*...)

        # Pre-compute combined exponentials
        exp_mu_t_T_growth = exp_mu_t * exp_T_growth
        exp_mu_t_2T_growth = exp_mu_t * exp_2T_growth
        lambda_numerator_1 = (lambda_ages_t - 1) * exp_mu_t_T_growth
        lambda_numerator_2 = -(lambda_ages_t + lambda_val * T - 1) * exp_mu_t_2T_growth
        lambda_numerator_3 = -mu_val * T * exp_lambda_t
        lambda_numerator_4 = 2 * mu_val * T * exp_lambda_t * exp_T_growth
        lambda_numerator = (
            lambda_numerator_1
            + lambda_numerator_2
            + lambda_numerator_3
            + lambda_numerator_4
        )
        denominator = (np.exp(T_growth_rate) - 1) * (
            lambda_val * exp_mu_t_T_growth - mu_val * exp_lambda_t
        )
        mu_numerator_1 = lambda_val * T * exp_mu_t_2T_growth
        mu_numerator_2 = exp_lambda_t * (-mu_ages_t + mu_val * T + 1)
        mu_numerator_3 = (mu_ages_t - 2 * mu_val * T - 1) * np.exp(
            lambda_ages_t + T_growth_rate
        )
        mu_numerator = mu_numerator_1 + mu_numerator_2 + mu_numerator_3

        d_lambda = lambda_numerator / denominator
        d_mu = mu_numerator / denominator

    return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)


def calculate_spine_ages_log_likelihood_derivatives(
    lambda_param: float,
    mu_param: float,
    ages_t: NDArray[np.float64] | float,
    T: float,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of the log age-density f(t) for a "spine" individual
    with respect to the birth rate (λ)
    and death rate (μ).

    Parameters
    ----------
    lambda_param : float
        Birth rate λ (> 0) if log_scale_params is False; otherwise log(λ) when
        log_scale_params is True.
    mu_param : float
        Death rate μ (> 0) if log_scale_params is False; otherwise log(μ) when
        log_scale_params is True.
    ages_t : float or array_like
        Age or array of ages a of the manuscript observation. Values equal to T
        are treated as boundary/censored observations.
    T : float
        Total observation time (T >= ages_t is expected).
    log_scale_params : bool, optional
        If True, lambda_param and mu_param are provided in log-space. They are
        exponentiated internally for evaluation. Returned derivatives correspond to
        partial derivatives with respect to the original (unlogged) parameters λ and μ.

    Returns
    -------
    numpy.ndarray
        1-D array of shape (2,) and dtype np.float64:
        [ sum_i ∂ log f(a_i) / ∂λ,  sum_i ∂ log f(a_i) / ∂μ ],
        i.e. the partial derivatives summed across all provided ages_t entries.

    Notes
    -----
    - The function is vectorized over ages_t. Entries with ages_t == T use a boundary
      closed-form expression; entries with ages_t < T use the interior expression.
    - A numerically stable limiting formula is used when λ ≈ μ; the equality test is
      performed in log-space with a tolerance of 1e-12.
    - Invalid inputs (non-positive rates, inconsistent T/ages_t) may produce NaN or inf;
      callers should validate inputs if needed.
    - Returned values are sums over the provided ages; if per-age derivatives
      are desired, inspect or modify the implementation to return elementwise arrays
      instead of sums.
    """
    # Handle parameter scaling
    if log_scale_params:
        log_lambda = lambda_param
        log_mu = mu_param
        # Convert log parameters back to original scale
        lambda_val = np.exp(lambda_param)
        mu_val = np.exp(mu_param)
    else:
        lambda_val = lambda_param
        mu_val = mu_param
        log_lambda = np.log(lambda_param)
        log_mu = np.log(mu_param)

    at_boundary = ages_t == T
    not_at_boundary = np.invert(at_boundary)
    d_lambda = np.zeros_like(ages_t, dtype=np.float64)
    d_mu = np.zeros_like(ages_t, dtype=np.float64)

    # Check for numerical stability in parameter space
    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ
        if np.any(at_boundary):
            # d/dλ = -(λ T^2)/(λ T + 1)
            d_mu[at_boundary] = d_lambda[at_boundary] = -(lambda_val * T**2) / (
                lambda_val * T + 1
            )
        if np.any(not_at_boundary):
            # d/dλ = 2/λ - a
            d_mu[not_at_boundary] = d_lambda[not_at_boundary] = 2 / lambda_val - ages_t
    else:
        # General case: λ ≠ μ
        if np.any(at_boundary):
            # d/dλ = (μ (e^(μ T) (T (λ - μ) + 1) - e^(λ T)))/((λ - μ) (λ e^(λ T) - μ e^(μ T)))
            lambda_numerator = mu_val * (
                np.exp(mu_val * T) * (T * (lambda_val - mu_val) + 1)
                - np.exp(lambda_val * T)
            )
            denominator = (lambda_val - mu_val) * (
                lambda_val * np.exp(lambda_val * T) - mu_val * np.exp(mu_val * T)
            )
            d_lambda[at_boundary] = lambda_numerator / denominator
            # d/dμ = -(λ (e^(λ T) (T (λ - μ) - 1) + e^(μ T)))/((λ - μ) (λ e^(λ T) - μ e^(μ T)))
            mu_numerator = -lambda_val * (
                np.exp(lambda_val * T) * (T * (lambda_val - mu_val) - 1)
                + np.exp(mu_val * T)
            )
            d_mu[at_boundary] = mu_numerator / denominator
        if np.any(not_at_boundary):
            # d/dλ = (e^(μ a) (μ + λ a (λ - μ)) - μ e^(λ a))/(λ (λ - μ) (e^(λ a) - e^(μ a)))
            lambda_numerator = np.exp(mu_val * ages_t) * (
                mu_val + lambda_val * ages_t * (lambda_val - mu_val)
            ) - mu_val * np.exp(lambda_val * ages_t)
            lambda_denominator = (
                lambda_val
                * (lambda_val - mu_val)
                * (np.exp(lambda_val * ages_t) - np.exp(mu_val * ages_t))
            )
            d_lambda[not_at_boundary] = (
                lambda_numerator[not_at_boundary] / lambda_denominator[not_at_boundary]
            )
            # d/dμ = (e^(λ a) (λ - λ μ a + μ^2 a) - λ e^(μ a))/(μ (λ - μ) (e^(λ a) - e^(μ a)))
            mu_numerator = np.exp(lambda_val * ages_t) * (
                lambda_val - lambda_val * mu_val * ages_t + mu_val**2 * ages_t
            ) - lambda_val * np.exp(mu_val * ages_t)
            mu_denominator = (
                mu_val
                * (lambda_val - mu_val)
                * (np.exp(lambda_val * ages_t) - np.exp(mu_val * ages_t))
            )
            d_mu[not_at_boundary] = (
                mu_numerator[not_at_boundary] / mu_denominator[not_at_boundary]
            )

    return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)


def calculate_ages_log_likelihood_derivatives(
    lambda_param: float,
    mu_param: float,
    ages_t: NDArray[np.float64] | float,
    T: float,
    log_scale_params: bool = False,
    are_spines: bool = False,
) -> NDArray[np.float64]:
    """
    Compute the partial derivatives of the log age-density f(t) with respect to
    the birth rate (λ) and death rate (μ).

    This function handles both "spine" and "non-spine" individuals by delegating to
    the appropriate implementation and supports interpreting the rate parameters
    in log-scale.

    Parameters
    ----------
    lambda_param : float
        Birth rate (λ). If `log_scale_params` is True, this is interpreted as
        log(λ). Must be finite; when interpreted as λ, must be strictly positive.
    mu_param : float
        Death rate (μ). If `log_scale_params` is True, this is interpreted as
        log(μ). Must be finite; when interpreted as μ, must be strictly positive.
    ages_t : float or array-like
        Age(s) t at which to evaluate the derivatives. Each age must satisfy
        0 <= t <= T. Can be a scalar or a 1-D array; the return shape will match.
    T : float
        Total observation time. Must satisfy T >= max(ages_t) and be finite.
    log_scale_params : bool, optional
        If True, `lambda_param` and `mu_param` are provided as their natural
        logarithms. The returned derivatives are those of the log-likelihood with
        respect to the parameters as provided (i.e., with chain-rule applied when
        parameters are given in log-scale). Default is False.
    are_spines : bool, optional
        If True, compute derivatives for a spine individual; otherwise compute for
        a non-spine individual. Default is False.

    Returns
    -------
    numpy.ndarray
        Array of derivatives of the log age-density with respect to the two
        parameters, in the order (d/dλ, d/dμ). If `ages_t` is a scalar the array
        has shape (2,); if `ages_t` is an array of length n the array has shape
        (2, n). The dtype is float64.

    Notes on error handling
    -----------------------
    This function delegates to spine or non-spine implementations which return -inf
    for numerical issues but raise exceptions for programming errors (type mismatches,
    array shape errors, etc.).

    Notes
    -----
    - This function returns derivatives of the log of the age density f(t), not
      derivatives of f(t) itself.
    - Implementation delegates to specialized routines for spine vs non-spine
      individuals; behavior and formulae may differ between these cases.
    - When `log_scale_params` is True, derivatives are computed using the chain
      rule so that they correspond to derivatives with respect to the supplied
      log-parameters.

    Examples
    --------
    # Scalar ages
    >>> calculate_ages_log_likelihood_derivatives(1.0, 0.5, 2.0, 3.0)
    array([..., ...])

    # Vectorized ages
    >>> calculate_ages_log_likelihood_derivatives(1.0, 0.5, [0.5, 1.5, 2.5], 3.0)
    array([[..., ..., ...],
           [..., ..., ...]])
    """
    if are_spines:
        return calculate_spine_ages_log_likelihood_derivatives(
            lambda_param, mu_param, ages_t, T, log_scale_params
        )
    else:
        return calculate_non_spine_ages_log_likelihood_derivatives(
            lambda_param, mu_param, ages_t, T, log_scale_params
        )


def calculate_extinction_probability(
    birth_rates: NDArray[np.float64],
    death_rates: NDArray[np.float64],
    times: NDArray[np.float64],
    log_scale_params: bool = False,
    return_log_prob: bool = False,
) -> np.float64:
    """
    Calculate the extinction probability (α_t) for a birth-death process with piecewise-constant rates.

    This function computes the probability that a lineage starting with a single individual
    at time 0 will have gone extinct by the final observation time, given piecewise-constant
    birth and death rates across multiple time epochs.

    The extinction probability is computed as:
        α_t = probability_extinction_numerator / (1 + probability_extinction_numerator)

    where probability_extinction_numerator is the integral of the extinction probability density across all epochs,
    accounting for the cumulative growth rate effects.

    Parameters
    ----------
    birth_rates : NDArray[np.float64]
        Array of birth rates λ for each epoch. If `log_scale_params` is True,
        these are interpreted as log(λ).
    death_rates : NDArray[np.float64]
        Array of death rates μ for each epoch. If `log_scale_params` is True,
        these are interpreted as log(μ).
    times : NDArray[np.float64]
        Array of epoch end times. The i-th element is the end time of epoch i.
        Must be monotonically increasing. Epochs are defined as [0, times[0]),
        [times[0], times[1]), etc.
    log_scale_params : bool, optional
        If True, birth_rates and death_rates are provided in log-scale and will
        be exponentiated internally. Default is False.

    Returns
    -------
    np.float64
        The extinction probability α_t ∈ (0, 1), representing the probability
        that a lineage starting with one individual will be extinct by times[-1].

    Notes
    -----
    - The function handles the limiting case λ ≈ μ using numerically stable
      formulas when |log(λ) - log(μ)| < 1e-12.
    - For the critical case (λ ≈ μ), uses: μ * exp(-cum_growth) * duration
    - For the non-critical case (λ ≠ μ), uses:
      (μ * exp(-cum_growth) / (λ - μ)) * (1 - exp(-(λ - μ) * duration))
    - The cumulative growth rate accounts for the compounding effects across epochs.

    See Also
    --------
    _calculate_geometric_parameter : Calculates the geometric parameter β_t
    calculate_populations_log_likelihood : Uses both α_t and β_t for likelihood calculations
    """
    probability_extinction_numerator = _probability_extinction_numerator(
        birth_rates,
        death_rates,
        times,
        log_scale_params,
        start_time=0.0,
    )
    if return_log_prob:
        log_alpha_t = np.log(probability_extinction_numerator) - np.log1p(
            probability_extinction_numerator
        )
        return np.float64(log_alpha_t)

    alpha_t = 1 - 1 / (1 + probability_extinction_numerator)

    return alpha_t


def _calculate_geometric_parameter(
    birth_rates: NDArray[np.float64] | float,
    death_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Conditional on survival at the present, this calculated the geometric parameter (1 - β_t)
    for the offspring distribution of a birth-death process with piecewise-constant rates.

    This is a private helper function that computes the parameter 1 - β_t used in the
    geometric offspring distribution of a birth-death process. The parameter represents
    the probability related to the survival and reproduction dynamics across multiple
    time epochs with varying rates.

    The geometric parameter is computed as:
        β_t = geom_num / (exp(-cum_growth[-1]) + geom_num)

    where geom_num is the integral accounting for birth events across all epochs,
    weighted by the cumulative growth rate at each point in time.

    Parameters
    ----------
    birth_rates : NDArray[np.float64]
        Array of birth rates λ for each epoch. If `log_scale_params` is True,
        these are interpreted as log(λ). Length must match death_rates and times.
    death_rates : NDArray[np.float64]
        Array of death rates μ for each epoch. If `log_scale_params` is True,
        these are interpreted as log(μ). Length must match birth_rates and times.
    times : NDArray[np.float64]
        Array of epoch end times. The i-th element is the end time of epoch i.
        Must be monotonically increasing. Epochs span [0, times[0]),
        [times[0], times[1]), etc.
    log_scale_params : bool, optional
        If True, birth_rates and death_rates are provided in log-scale and will
        be exponentiated internally. Default is False.

    Returns
    -------
    float
        The geometric parameter 1 - β_t ∈ (0, 1), representing the success probability
        in the geometric offspring distribution. This parameter, combined with the
        extinction probability α_t, fully characterizes the population distribution.

    Notes
    -----
    - This is a private function (prefixed with _) used internally by
      calculate_populations_log_likelihood for array (piecewise-constant rate) inputs.
    - Handles the limiting case λ ≈ μ when |log(λ) - log(μ)| < 1e-12 using
      numerically stable formulas to avoid catastrophic cancellation.
    - For critical case (λ ≈ μ): λ * exp(-cum_growth) * duration
    - For non-critical case (λ ≠ μ):
      (λ * exp(-cum_growth) / (λ - μ)) * (1 - exp(-(λ - μ) * duration))
    - The final division by (exp(-cum_growth[-1]) + geom_num) normalizes
      the result to ensure β_t ∈ (0, 1).

    See Also
    --------
    calculate_extinction_probability : Calculates the extinction probability α_t
    calculate_populations_log_likelihood : Uses β_t for likelihood calculations
    """
    birth_rates = np.atleast_1d(birth_rates)
    death_rates = np.atleast_1d(death_rates)
    times = np.atleast_1d(times)
    epoch_durations = np.diff(np.insert(times, 0, 0.0))
    if np.any(epoch_durations < 0):
        raise ValueError("times array must be strictly increasing.")
    if log_scale_params:
        lambda_val_array = np.exp(birth_rates)
        mu_val_array = np.exp(death_rates)
    else:
        lambda_val_array = birth_rates
        mu_val_array = death_rates

    growth_rates = lambda_val_array - mu_val_array
    cum_growth = np.cumsum(growth_rates * epoch_durations)

    geom_sum = _probability_extinction_numerator(
        birth_rates, death_rates, times, log_scale_params
    )
    cond_geom_prob = np.exp(-cum_growth[-1]) / (1 + geom_sum)
    return np.float64(cond_geom_prob)


def _probability_extinction_numerator(
    birth_rates: NDArray[np.float64] | float,
    death_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool,
    start_time: float = 0.0,
) -> np.float64:
    """
    Calculate the numerator of the extinction probability integral.
    This function computes the integral that forms the numerator in the extinction
    probability calculation for a birth-death process with piecewise-constant rates.
    birth_rates : NDArray[np.float64]
        Array of birth rates λ for each epoch. If `log_scale_params` is True,
        these are interpreted as log(λ).
    death_rates : NDArray[np.float64]
        Array of death rates μ for each epoch. If `log_scale_params` is True,
        these are interpreted as log(μ).
    times : NDArray[np.float64]
        Array of epoch end times. The i-th element is the end time of epoch i.
        Must be monotonically increasing. Epochs are defined as [0, times[0]),
        [times[0], times[1]), etc.
    log_scale_params : bool
        If True, birth_rates and death_rates are provided in log-scale and will
        be exponentiated internally.
    start_time : float, optional
        The time from which to start the extinction probability calculation. Default is 0.0.
    """
    birth_rates = np.atleast_1d(birth_rates)
    death_rates = np.atleast_1d(death_rates)
    times = np.atleast_1d(times)

    if log_scale_params:
        log_lambda_array = birth_rates
        log_mu_array = death_rates
        lambda_val_array = np.exp(birth_rates)
        mu_val_array = np.exp(death_rates)
    else:
        log_lambda_array = np.log(birth_rates)
        log_mu_array = np.log(death_rates)
        lambda_val_array = birth_rates
        mu_val_array = death_rates

    times -= start_time
    after_start_mask = times > 0
    times = times[after_start_mask]
    birth_rates = birth_rates[after_start_mask]
    death_rates = death_rates[after_start_mask]
    # Also filter the derived arrays
    log_lambda_array = log_lambda_array[after_start_mask]
    log_mu_array = log_mu_array[after_start_mask]
    lambda_val_array = lambda_val_array[after_start_mask]
    mu_val_array = mu_val_array[after_start_mask]

    epoch_durations = np.diff(np.insert(times, 0, 0.0))
    critical_mask = np.abs(log_lambda_array - log_mu_array) < 1e-12
    growth_rates = lambda_val_array - mu_val_array
    cum_growth = np.cumsum(growth_rates * epoch_durations)
    trunc_cum_growth = np.insert(cum_growth, 0, 0.0)[:-1]

    growth_rates = lambda_val_array - mu_val_array
    cum_growth = np.cumsum(growth_rates * epoch_durations)
    trunc_cum_growth = np.insert(cum_growth, 0, 0.0)[:-1]

    critical_mask = np.abs(growth_rates) < 1e-12
    probability_extinction_numerator = (
        mu_val_array
        * np.exp(-trunc_cum_growth)
        * np.where(
            critical_mask,
            epoch_durations,
            (1 - np.exp(-growth_rates * epoch_durations)) / growth_rates,
        )
    )

    return np.sum(probability_extinction_numerator, dtype=np.float64)


def _integrand_survival(
    start_time: float,
    birth_rates: NDArray[np.float64],
    death_rates: NDArray[np.float64],
    times: NDArray[np.float64],
    log_scale_params: bool = False,
) -> np.float64:
    """
    Integrand function for manuscript survival probability calculation.

    This function computes the integrand value for a given age based on the
    birth and death rates.

    Args:
        birth_rate: Birth rate parameter
        death_rate: Death rate parameter
        age: Age at which to evaluate the integrand
    Returns:
        The computed integrand value
    """
    # Find the index of the last epoch that starts before or at start_time
    valid_indices = np.where(times <= start_time)[0]
    if len(valid_indices) > 0:
        last_valid_idx = valid_indices[-1]
        current_birth_rate = birth_rates[last_valid_idx]
    else:
        current_birth_rate = birth_rates[0]

    ext_prob = 1 - 1 / (
        1
        + _probability_extinction_numerator(
            start_time=start_time,
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            log_scale_params=log_scale_params,
        )
    )

    return ext_prob * float(current_birth_rate)


def _log_probability_extinction_given_ages(
    ages: NDArray[np.float64] | float,
    birth_rates: NDArray[np.float64],
    death_rates: NDArray[np.float64],
    times: NDArray[np.float64],
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate the log probability of extinction given age(s).

    This function computes the log probability that a manuscript of a given age
    has gone extinct based on the birth and death rates. Efficiently handles
    both scalar and array inputs for ages.

    Args:
        ages: Age(s) of the manuscript(s) - scalar or array
        birth_rates: Array of birth rates for each epoch
        death_rates: Array of death rates for each epoch
        times: Array of epoch end times
        log_scale_params: Whether the rates are in log scale
    Returns:
        The sum of the log extinction probabilities given age(s)
    """
    # Handle scalar input
    is_scalar = isinstance(ages, (int, float))
    if is_scalar:
        ages = np.array([ages], dtype=np.float64)
    else:
        ages = np.asarray(ages, dtype=np.float64)

    if log_scale_params:
        lambda_val_array = np.exp(birth_rates)
        mu_val_array = np.exp(death_rates)
    else:
        lambda_val_array = birth_rates
        mu_val_array = death_rates

    epoch_durations = np.diff(np.insert(times, 0, 0.0))
    growth_rates = lambda_val_array - mu_val_array
    cum_growth = np.cumsum(growth_rates * epoch_durations)

    # Vectorized computation for all ages
    log_probs = np.zeros_like(ages, dtype=np.float64)

    # For each age, find which epoch it falls into and compute
    for i, age in enumerate(ages):
        # Find the last epoch that ends at or before this age
        valid_indices = np.where(times <= age)[0]
        if len(valid_indices) > 0:
            last_valid_idx = valid_indices[-1]
            current_cum_growth = cum_growth[last_valid_idx]
        else:
            current_cum_growth = 0.0

        # Compute integral using numerical integration
        integral = scipy.integrate.quad(
            _integrand_survival,
            0,
            age,
            args=(birth_rates, death_rates, times, log_scale_params),
        )[0]

        log_probs[i] = -current_cum_growth - integral

    return np.sum(log_probs)


def _log_likelihood_ages(
    ages: NDArray[np.float64] | float,
    birth_rates: NDArray[np.float64],
    times: NDArray[np.float64],
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate the log likelihood of manuscript age(s).

    This function computes the log likelihood of observing a manuscript of a given
    age based on the birth rates. Efficiently handles both scalar and array inputs.

    Args:
        ages: Age(s) of the manuscript(s) - scalar or array
        birth_rates: Array of birth rates for each epoch
        times: Array of epoch end times
        log_scale_params: Whether the rates are in log scale
    Returns:
        The sum of computed log likelihood values
    """
    # Handle scalar input
    is_scalar = isinstance(ages, (int, float))
    if is_scalar:
        ages = np.array([ages], dtype=np.float64)
    else:
        ages = np.asarray(ages, dtype=np.float64)

    if log_scale_params:
        lambda_val_array = np.exp(birth_rates)
    else:
        lambda_val_array = birth_rates

    epoch_starts = np.insert(times, 0, 0.0)
    epoch_durations = np.diff(epoch_starts)

    log_likelihoods = np.zeros_like(ages, dtype=np.float64)

    for i, age in enumerate(ages):
        valid_indices = np.where(epoch_starts < age)[0]
        completed_epochs = valid_indices[:-1]

        # The current epoch index is valid_indices[-1], but we need to cap it
        # at len(lambda_val_array)-1 since epoch_starts has one extra element
        current_epoch_idx = min(valid_indices[-1], len(lambda_val_array) - 1)

        cum_growth = (
            np.sum(
                lambda_val_array[completed_epochs] * epoch_durations[completed_epochs]
            )
            + (age - epoch_starts[valid_indices[-1]])
            * lambda_val_array[current_epoch_idx]
        )
        current_lambda = lambda_val_array[current_epoch_idx]

        log_likelihoods[i] = np.log(current_lambda) - cum_growth

    return np.sum(log_likelihoods)


def _log_likelihood_age_given_extinction(
    ages: NDArray[np.float64] | float,
    birth_rates: NDArray[np.float64],
    death_rates: NDArray[np.float64],
    times: NDArray[np.float64],
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate the log likelihood of manuscript age(s) given extinction.

    This function computes the log likelihood of observing manuscript(s) of given
    age(s) under the assumption that it/they have gone extinct. Handles both
    scalar and array inputs efficiently.

    Args:
        ages: Age(s) of the manuscript(s) - scalar or array
        birth_rates: Array of birth rates for each epoch
        death_rates: Array of death rates for each epoch
        times: Array of epoch end times
        log_scale_params: Whether the rates are in log scale
    Returns:
        The summed log likelihood values
    """
    # Get log extinction probability (scalar)
    log_extinction_prob = calculate_extinction_probability(
        birth_rates, death_rates, times, log_scale_params, return_log_prob=True
    )

    # Get log probabilities given age (array or scalar)
    log_probability_extinction_given_ages = _log_probability_extinction_given_ages(
        ages, birth_rates, death_rates, times, log_scale_params
    )

    # Get log likelihood of ages (array or scalar)
    log_likelihood_ages = _log_likelihood_ages(
        ages, birth_rates, times, log_scale_params
    )

    # Compute conditional log likelihood
    # For arrays, we sum over all ages; for scalar, just return the value
    ages_array = np.atleast_1d(ages)
    n_ages = len(ages_array)
    result = (
        np.sum(log_probability_extinction_given_ages + log_likelihood_ages)
        - log_extinction_prob * n_ages
    )
    return result
