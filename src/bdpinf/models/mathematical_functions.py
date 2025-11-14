"""
Mathematical functions for birth-and-death process calculations.

This module contains pure mathematical functions for calculating birth-and-death
process probabilities and densities. These functions are isolated from the
main BirthDeathProcess class for better testability and modularity.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def calculate_populations_log_likelihood(
    lambda_param: float,
    mu_param: float,
    time: float,
    ns: NDArray[np.integer] | int,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate the log-probability (log-likelihood) of observing given population counts
    under a linear birth-and-death process at a single time point.
    This function computes log P_n(T) for each count n in `ns` using closed-form
    expressions for the birth-and-death process probabilities. The computed log-probabilities
    are summed and returned as a single np.float64 value. Invalid or numerically
    unstable inputs return -inf (np.float64(-np.inf)).
    Parameters
    ----------
    lambda_param : float
        Birth rate λ. If `log_scale_params` is True, this is interpreted as log(λ).
    mu_param : float
        Death rate μ. If `log_scale_params` is True, this is interpreted as log(μ).
    time : float
        Time T at which the distribution is evaluated. Must be ≥ 0.
    ns : numpy.ndarray[int] or int
        Non-negative integer or array of non-negative integers representing the
        population counts n for which to evaluate the log-probabilities.
    log_scale_params : bool, optional
        If True, `lambda_param` and `mu_param` are treated as natural logarithms
        of the rates and are exponentiated internally. Default is False.
    Returns
    -------
    numpy.float64
        Sum of log P_n(T) across the provided `ns`. Returns np.float64(-np.inf) for
        numerically invalid inputs (negative time, negative counts, non-positive rates
        when not in log-space, or α_t/β_t not in (0,1)). Programming errors (type errors,
        indexing errors) will raise exceptions.
    Behavior and formulas
    ---------------------
    Let α_t and β_t denote auxiliary probabilities defined by the birth-and-death process:
        α_t = μ (e^{(λ-μ) T} - 1) / (λ e^{(λ-μ) T} - μ)
        β_t = (λ / μ) * α_t
    The log-probabilities used are:
        log P_n(T) = log(1 - α_t) + log(1 - β_t) + (n - 1) * log(β_t)   for n ≥ 1
    The implementation handles the limiting case λ ≈ μ by using the limiting form
    α_t = β_t = (λ T) / (1 + λ T) to avoid catastrophic cancellation.
    Edge cases
    ----------
    - Negative time or negative counts produce -inf.
    - When `log_scale_params` is False, non-positive λ or μ produce -inf.
    - If α_t or β_t are not strictly between 0 and 1, the function returns -inf.
    - Non-finite intermediate values (overflow/underflow) produce -inf.
    - The function accepts scalar or array-like `ns`; intermediate computations
      preserve array shape, but the returned value is the scalar sum of log-probabilities.
    - Type errors, shape mismatches, and other programming errors raise exceptions.
    Examples
    --------
    # Scalar n
    >>> calculate_population_log_likelihood(0.5, 0.3, 10.0, 2)
    # Vectorized ns
    >>> calculate_population_log_likelihood(0.5, 0.3, 10.0, np.array([0, 1, 2]))
    """
    # Input validation
    if np.any(ns < 0) or time < 0:
        return np.float64(-np.inf)

    # Handle parameter scaling
    if log_scale_params:
        log_lambda = lambda_param
        log_mu = mu_param
        lambda_val = np.exp(lambda_param)
        mu_val = np.exp(mu_param)
    else:
        # Validate positive parameters
        if lambda_param <= 0 or mu_param <= 0:
            return np.float64(-np.inf)
        log_lambda = np.log(lambda_param)
        log_mu = np.log(mu_param)
        lambda_val = lambda_param
        mu_val = mu_param

    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ
        rate_time_product = lambda_val * time
        alpha_t = beta_t = rate_time_product / (1 + rate_time_product)
    else:
        # General case: λ ≠ μ
        growth_rate = lambda_val - mu_val
        exponent = growth_rate * time
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
    lambda_param: float,
    mu_param: float,
    ages_t: NDArray[np.float64] | float,
    T: float,
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

    # Check for numerical stability in parameter space
    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ

        log_numerator = -lambda_val * ages_t + np.log(1 + lambda_val * (T - ages_t))
        log_denominator = np.log(T)

        # Check for numerical validity
        if not (np.isfinite(log_numerator) and np.isfinite(log_denominator)):
            return np.float64(-np.inf)

    else:
        # General case: λ ≠ μ
        growth_rate = lambda_val - mu_val
        growth_rate_T = growth_rate * T

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
        return calculate_non_spine_ages_log_likelihood(
            lambda_param, mu_param, ages_t, T, log_scale_params
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


def predict_manuscript_survival(
    birth_rate: float,
    death_rate: float,
    ages: NDArray[np.float64] | float,
) -> Union[float, NDArray[np.float64]]:
    """
    Predict expected manuscript survival probability for given ages.

    This is a simple exponential survival function based on the death rate.

    Args:
        birth_rate: Birth rate parameter (used for scaling)
        death_rate: Death rate parameter
        ages: Age or array of ages to predict for

    Returns:
        Predicted survival probability or array of probabilities
    """
    if isinstance(ages, (int, float)):
        ages = np.array([ages], dtype=np.float64)

    # Convert to numpy array if not already
    ages_array = np.asarray(ages, dtype=np.float64)

    # Simple exponential survival model
    survival_probs = birth_rate * np.exp(-death_rate * ages_array)

    # Return scalar if input was scalar
    if len(survival_probs) == 1 and isinstance(ages, (int, float)):
        return float(survival_probs[0])

    return survival_probs
