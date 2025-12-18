"""
Mathematical functions for birth-and-death process calculations.

This module contains pure mathematical functions for calculating birth-and-death
process probabilities and densities. These functions are isolated from the
main BirthDeathProcess class for better testability and modularity.
"""

import numpy as np
import scipy
from numpy.typing import NDArray


def calculate_populations_log_lik(
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
        Birth rate(s) λ. Scalar for constant rate, array for piecewise-constant rates.
        If `log_scale_params` is True, interpreted as log(λ).
    mu_params : float or NDArray[np.float64]
        Death rate(s) μ. Scalar for constant rate, array for piecewise-constant rates.
        If `log_scale_params` is True, interpreted as log(μ).
    times : float or NDArray[np.float64]
        Time(s) T at which to evaluate the distribution. Scalar for single time point,
        array for epoch end times in piecewise-constant model. Must be ≥ 0 and increasing.
    ns : int or NDArray[np.integer]
        Non-negative population count(s) for which to evaluate log-probabilities.
    log_scale_params : bool, optional
        If True, parameters are treated as logarithms and exponentiated. Default False.

    Returns
    -------
    np.float64
        Sum of log p_n(T) across provided `ns`. Returns -inf for invalid inputs.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Array inputs must have
    matching lengths. Returns -inf for numerical issues, raises exceptions for
    programming errors.
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
            f"Got types: lambda={type(lambda_params).__name__}, "
            f"mu={type(mu_params).__name__}, time={type(times).__name__}"
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

        alpha_t = calculate_extinction_pr(lambda_val_array, mu_val_array, times)

        beta_t = 1 - _population_geometric_parameter(
            lambda_val_array, mu_val_array, times
        )
    else:
        if (
            not isinstance(lambda_params, float)
            or not isinstance(mu_params, float)
            or not isinstance(times, float)
        ):
            raise ValueError(
                "When using scalar inputs, lambda_params, mu_params, "
                "and times must all be floats. "
                f"Got types: lambda_params={type(lambda_params).__name__}, "
                f"mu_params={type(mu_params).__name__}, times={type(times).__name__}. "
                f"Note: numpy scalar types (np.float64, np.int32, etc.) "
                f"are not accepted - use Python float type."
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
    log_pr = np.zeros_like(ns, dtype=np.float64)

    # Handle n == 0 case
    zero_mask = ns == 0
    if np.any(zero_mask):
        log_pr[zero_mask] = np.log(alpha_t)

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

        log_pr[nonzero_mask] = log_term1 + log_term2 + log_term3[nonzero_mask]

    # Final numerical check
    if not np.all(np.isfinite(log_pr)):
        return np.float64(-np.inf)

    return np.float64(np.sum(log_pr))


def calculate_populations_log_lik_derivatives(
    lambda_param: float,
    mu_param: float,
    time: float,
    ns: NDArray[np.int64] | int,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of log population probabilities with respect to λ and μ.

    Parameters
    ----------
    lambda_param : float
        Birth rate λ. If `log_scale_params` is True, interpreted as log(λ).
    mu_param : float
        Death rate μ. If `log_scale_params` is True, interpreted as log(μ).
    time : float
        Time t at which to evaluate (t ≥ 0).
    ns : int or NDArray[np.int64]
        Population count(s) to evaluate.
    log_scale_params : bool, optional
        If True, parameters are in log-scale. Default False.

    Returns
    -------
    NDArray[np.float64]
        Array [∂/∂λ, ∂/∂μ] containing summed derivatives across all `ns`.

    Notes
    -----
    Vectorized over `ns`. Handles λ ≈ μ limiting case for numerical stability.
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

    # Ensure ns is an array for consistent masking
    ns = np.atleast_1d(ns)

    d_lambda = np.zeros_like(ns, dtype=np.float64)
    d_mu = np.zeros_like(ns, dtype=np.float64)

    zero_mask = ns == 0
    non_zero_mask = np.invert(zero_mask)

    if abs(log_lambda - log_mu) < 1e-14:  # Tighter tolerance for derivatives
        # Limiting case: λ ≈ μ
        # α_t = β_t = λT/(1 + λT)

        if np.any(zero_mask):
            # For n=0: log p_0 = log(α_t) = log(λT) - log(1 + λT)
            # ∂/∂λ = 1/λ - T/(1 + λT) = 1/(λ(1 + λT))
            d_lambda[zero_mask] = 1 / (lambda_val * (1 + lambda_val * time))
            # When λ = μ, both derivatives are the same
            d_mu[zero_mask] = d_lambda[zero_mask]

        if np.any(non_zero_mask):
            # Extract only non-zero n values
            ns_nonzero = ns[non_zero_mask]
            # For n>0: log p_n = (n-1)log(λT) - (n+1)log(1+λT)
            # ∂/∂λ = (n-1)/λ - (n+1)T/(1+λT) = [n - 1 - 2λT]/(λ(1+λT))
            numerator = ns_nonzero - 1 - 2 * lambda_val * time
            denominator = lambda_val * (1 + lambda_val * time)
            d_lambda[non_zero_mask] = numerator / denominator
            # When λ = μ, both derivatives are the same
            d_mu[non_zero_mask] = d_lambda[non_zero_mask]

        return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)

    # General case: λ ≠ μ
    growth_rate = lambda_val - mu_val
    exp_rT = np.exp(growth_rate * time)

    if np.any(zero_mask):
        # For n=0: d log(α_t) / d param = (1/α_t) * dα_t/d param

        # α_t = μ(e^{rT} - 1) / (λe^{rT} - μ)
        alpha_t = mu_val * (exp_rT - 1) / (lambda_val * exp_rT - mu_val)

        # ∂α_t/∂λ = μe^{rT} [(λ - μ)T - e^{rT} + 1] / (λe^{rT} - μ)²
        d_alpha_d_lambda = (
            mu_val
            * exp_rT
            * (growth_rate * time - exp_rT + 1)
            / (lambda_val * exp_rT - mu_val) ** 2
        )

        # ∂α_t/∂μ = e^{rT}[λ(e^{rT} - 1) - μT(λ - μ)] / (λe^{rT} - μ)²
        d_alpha_d_mu = (
            exp_rT
            * (lambda_val * (exp_rT - 1) - mu_val * time * growth_rate)
            / (lambda_val * exp_rT - mu_val) ** 2
        )

        d_lambda[zero_mask] = d_alpha_d_lambda / alpha_t
        d_mu[zero_mask] = d_alpha_d_mu / alpha_t
    if np.any(non_zero_mask):
        # Extract only non-zero n values for computation
        ns_nonzero = ns[non_zero_mask]

        # For n>0: log p_n = log(1-α_t) + log(1-β_t) + (n-1)log(β_t)
        # where β_t = (λ/μ) * α_t

        alpha_t = mu_val * (exp_rT - 1) / (lambda_val * exp_rT - mu_val)
        beta_t = (lambda_val / mu_val) * alpha_t

        # Derivatives of α_t (from above)
        d_alpha_d_lambda = (
            mu_val
            * exp_rT
            * (growth_rate * time - exp_rT + 1)
            / (lambda_val * exp_rT - mu_val) ** 2
        )
        d_alpha_d_mu = (
            exp_rT
            * (lambda_val * (exp_rT - 1) - mu_val * time * growth_rate)
            / (lambda_val * exp_rT - mu_val) ** 2
        )

        # Derivatives of β_t = (λ/μ) * α_t
        # ∂β_t/∂λ = (1/μ)·α_t + (λ/μ)·∂α_t/∂λ
        d_beta_d_lambda = (alpha_t / mu_val) + (lambda_val / mu_val) * d_alpha_d_lambda

        # ∂β_t/∂μ = -(λ/μ²)·α_t + (λ/μ)·∂α_t/∂μ
        d_beta_d_mu = (
            -(lambda_val / mu_val**2) * alpha_t + (lambda_val / mu_val) * d_alpha_d_mu
        )

        # ∂/∂λ [log(1-α_t) + log(1-β_t) + (n-1)log(β_t)]
        #     = -1/(1-α_t)·∂α_t/∂λ - 1/(1-β_t)·∂β_t/∂λ + (n-1)/β_t·∂β_t/∂λ
        d_lambda_nonzero = (
            -d_alpha_d_lambda / (1 - alpha_t)
            - d_beta_d_lambda / (1 - beta_t)
            + (ns_nonzero - 1) * d_beta_d_lambda / beta_t
        )

        # ∂/∂μ [log(1-α_t) + log(1-β_t) + (n-1)log(β_t)]
        #     = -1/(1-α_t)·∂α_t/∂μ - 1/(1-β_t)·∂β_t/∂μ + (n-1)/β_t·∂β_t/∂μ
        d_mu_nonzero = (
            -d_alpha_d_mu / (1 - alpha_t)
            - d_beta_d_mu / (1 - beta_t)
            + (ns_nonzero - 1) * d_beta_d_mu / beta_t
        )

        d_lambda[non_zero_mask] = d_lambda_nonzero
        d_mu[non_zero_mask] = d_mu_nonzero

    return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)


def calculate_non_spine_ages_log_lik(
    lambda_params: NDArray[np.float64] | float,
    mu_params: NDArray[np.float64] | float,
    ages_t: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate log-likelihood of manuscript ages for non-spine individuals.

    Parameters
    ----------
    lambda_params : float or NDArray[np.float64]
        Birth rate(s) λ. If `log_scale_params` is True, interpreted as log(λ).
    mu_params : float or NDArray[np.float64]
        Death rate(s) μ. If `log_scale_params` is True, interpreted as log(μ).
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (t ≥ 0).
    times : float or NDArray[np.float64]
        Observation time(s) T. Must satisfy T ≥ t for all ages.
    log_scale_params : bool, optional
        If True, parameters are in log-scale. Default False.

    Returns
    -------
    np.float64
        Summed log-likelihood across ages. Returns -inf for invalid inputs.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Supports piecewise-constant
    rates via array inputs.
    """
    times = np.atleast_1d(times)
    end_time = times[-1]

    if np.any(np.diff(np.insert(times, 0, 0.0)) <= 0):
        raise ValueError(f"times must be strictly increasing. Got times: {times}")
    # Input validation
    if np.any(ages_t < 0) or np.any(ages_t > end_time):
        return np.float64(-np.inf)

    # Convert scalar parameters to arrays for single-epoch case
    lambda_params = np.atleast_1d(lambda_params)
    mu_params = np.atleast_1d(mu_params)

    if len(lambda_params) != len(mu_params):
        raise ValueError(
            f"lambda_params and mu_params must have the same length. "
            f"Got lengths: lambda_param={len(lambda_params)}, "
            f"mu_param={len(mu_params)}"
        )

    if len(lambda_params) != len(times):
        raise ValueError(
            f"lambda_params and times must have the same length. "
            f"Got lengths: lambda_param={len(lambda_params)}, times={len(times)}"
        )

    if len(times) > 1:
        inverted_times = np.flip(end_time - np.insert(times, 0, 0.0)[:-1])
        reflected_birth_rates = np.flip(mu_params)
        reflected_death_rates = np.flip(lambda_params)
        return calculate_log_lik_initial_lifetimes_given_extinction(
            ages_t,
            reflected_birth_rates,
            reflected_death_rates,
            inverted_times,
            log_scale_params,
        )

    # Handle parameter scaling
    # Extract scalar values from single-element arrays (atleast_1d guarantees arrays)
    lambda_scalar = float(lambda_params[0])
    mu_scalar = float(mu_params[0])

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
        if not np.all(np.isfinite([log_numerator, log_denominator])):
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


def calculate_spine_ages_log_lik(
    lambda_params: NDArray[np.float64] | float,
    mu_params: NDArray[np.float64] | float,
    ages_t: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate log-likelihood of manuscript ages for the "spine" individual.

    Parameters
    ----------
    lambda_params : float or NDArray[np.float64]
        Birth rate(s) λ. If `log_scale_params` is True, interpreted as log(λ).
    mu_params : float or NDArray[np.float64]
        Death rate(s) μ. If `log_scale_params` is True, interpreted as log(μ).
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T).
    times : float or NDArray[np.float64]
        Observation time(s) T.
    log_scale_params : bool, optional
        If True, parameters are in log-scale. Default False.

    Returns
    -------
    np.float64
        Summed log-likelihood across ages. Returns -inf for invalid inputs.

    Notes
    -----
    Ages at T use discrete mass; ages < T use continuous density. Handles λ ≈ μ
    limiting case. Supports piecewise-constant rates via array inputs.
    """
    times = np.atleast_1d(times)
    end_time = times[-1]

    if np.any(np.diff(np.insert(times, 0, 0.0)) <= 0):
        raise ValueError(f"times must be strictly increasing. Got times: {times}")
    # Input validation
    if np.any(ages_t < 0) or np.any(ages_t > end_time):
        return np.float64(-np.inf)

    # Convert scalar parameters to arrays for single-epoch case
    lambda_params = np.atleast_1d(lambda_params)
    mu_params = np.atleast_1d(mu_params)

    if len(lambda_params) != len(mu_params):
        raise ValueError(
            f"lambda_params and mu_params must have the same length. "
            f"Got lengths: lambda_param={len(lambda_params)}, "
            f"mu_param={len(mu_params)}"
        )

    if len(lambda_params) != len(times):
        raise ValueError(
            f"lambda_params and times must have the same length. "
            f"Got lengths: lambda_param={len(lambda_params)}, times={len(times)}"
        )

    if len(times) > 1:
        inverted_times = np.flip(end_time - times)
        reflected_birth_rates = np.flip(mu_params)
        reflected_death_rates = np.flip(lambda_params)
        return calculate_log_lik_initial_lifetimes_given_non_extinction(
            lifetimes=ages_t,
            birth_rates=reflected_birth_rates,
            death_rates=reflected_death_rates,
            times=inverted_times,
            log_scale_params=log_scale_params,
        )

    # Handle parameter scaling
    # Extract scalar values from single-element arrays (atleast_1d guarantees arrays)
    lambda_scalar = float(lambda_params[0])
    mu_scalar = float(mu_params[0])

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

    at_boundary = ages_t == end_time
    not_at_boundary = np.invert(at_boundary)

    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ
        boundary_val = np.log(1 + lambda_val * end_time) - lambda_val * end_time
        log_density = np.where(
            at_boundary,
            boundary_val,
            2 * np.log(lambda_val) + np.log(ages_t) - lambda_val * ages_t,
        )
        return np.sum(log_density, dtype=np.float64)

    # General case: λ ≠ μ

    # Initialize arrays for numerator and denominator
    log_numerator = np.zeros_like(ages_t, dtype=np.float64)
    denominator = lambda_val - mu_val
    sign_denominator = np.sign(denominator)
    log_denominator = np.log(np.abs(denominator))

    if np.any(at_boundary):
        numerator_boundary = lambda_val * np.exp(-mu_val * end_time) - mu_val * np.exp(
            -lambda_val * end_time
        )

        if not np.all(np.sign(numerator_boundary) == sign_denominator):
            return np.float64(-np.inf)

        log_numerator[at_boundary] = np.log(np.abs(numerator_boundary))

    if np.any(not_at_boundary):
        numerator_1 = np.exp(-mu_val * end_time) - np.exp(-lambda_val * end_time)

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


def calculate_ages_log_lik(
    lambda_params: NDArray[np.float64] | float,
    mu_params: NDArray[np.float64] | float,
    ages_t: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
    are_spines: bool = False,
) -> np.float64:
    """
    Compute log-likelihood of manuscript ages under a birth-death model.

    Parameters
    ----------
    lambda_params : float or NDArray[np.float64]
        Birth rate(s) λ. If `log_scale_params` is True, interpreted as log(λ).
    mu_params : float or NDArray[np.float64]
        Death rate(s) μ. If `log_scale_params` is True, interpreted as log(μ).
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T).
    times : float or NDArray[np.float64]
        Observation time(s) T.
    log_scale_params : bool, optional
        If True, parameters are in log-scale. Default False.
    are_spines : bool, optional
        If True, use spine calculation; otherwise non-spine. Default False.

    Returns
    -------
    np.float64
        Summed log-likelihood across ages. Returns -inf for invalid inputs.
    """
    if are_spines:
        return calculate_spine_ages_log_lik(
            lambda_params, mu_params, ages_t, times, log_scale_params
        )
    else:
        # calculate_non_spine_ages_log_lik requires array inputs
        # Wrap scalar parameters in single-element arrays for single-epoch case
        return calculate_non_spine_ages_log_lik(
            lambda_params,
            mu_params,
            ages_t,
            times,
            log_scale_params,
        )


def calculate_non_spine_ages_log_lik_derivatives(
    lambda_param: float,
    mu_param: float,
    ages_t: NDArray[np.float64] | float,
    T: float,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of log age-density for non-spine manuscripts.

    Parameters
    ----------
    lambda_param : float
        Birth rate λ. If `log_scale_params` is True, interpreted as log(λ).
    mu_param : float
        Death rate μ. If `log_scale_params` is True, interpreted as log(μ).
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T).
    T : float
        Observation time T.
    log_scale_params : bool, optional
        If True, parameters are in log-scale. Default False.

    Returns
    -------
    NDArray[np.float64]
        Array [∂/∂λ, ∂/∂μ] containing summed derivatives across all ages.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability.
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
    if abs(log_lambda - log_mu) < 1e-14:  # Tighter tolerance for derivatives
        # Limiting case: λ ≈ μ
        # f(t) = [1 + λ(T-t)] / T * e^{-λt}
        # log f(t) = log(1 + λ(T-t)) - log(T) - λt
        # ∂ log f(t)/∂λ = (T-t)/(1 + λ(T-t)) - t
        d_mu = d_lambda = (T - ages_t) / (1 + lambda_val * (T - ages_t)) - ages_t
    else:
        # General case: λ ≠ μ
        # log f(t) = log(λe^{-λt} - μe^{-(λ-μ)T}e^{-μt}) - log(1 - e^{-(λ-μ)T})
        # Let N = λe^{-λt} - μe^{-(λ-μ)T}e^{-μt}
        # Let D = 1 - e^{-(λ-μ)T}

        growth_rate = lambda_val - mu_val

        # Calculate N and D
        exp_lambda_t = np.exp(-lambda_val * ages_t)
        exp_mu_t = np.exp(-mu_val * ages_t)
        exp_growth_T = np.exp(-growth_rate * T)

        N = lambda_val * exp_lambda_t - mu_val * exp_growth_T * exp_mu_t
        D = 1 - exp_growth_T

        # ∂N/∂λ = e^{-λt}(1 - λt) + μT·e^{-(λ-μ)T}e^{-μt}
        dN_dlambda = (
            exp_lambda_t * (1 - lambda_val * ages_t)
            + mu_val * T * exp_growth_T * exp_mu_t
        )

        # ∂D/∂λ = T·e^{-(λ-μ)T}
        dD_dlambda = T * exp_growth_T

        # ∂ log f(t)/∂λ = (1/N)·∂N/∂λ - (1/D)·∂D/∂λ
        d_lambda = dN_dlambda / N - dD_dlambda / D

        # ∂N/∂μ = -e^{-(λ-μ)T}e^{-μt}[1 + μ(T - t)]
        dN_dmu = -exp_growth_T * exp_mu_t * (1 + mu_val * (T - ages_t))

        # ∂D/∂μ = -T·e^{-(λ-μ)T}
        dD_dmu = -T * exp_growth_T

        # ∂ log f(t)/∂μ = (1/N)·∂N/∂μ - (1/D)·∂D/∂μ
        d_mu = dN_dmu / N - dD_dmu / D

    return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)


def calculate_spine_ages_log_lik_derivatives(
    lambda_param: float,
    mu_param: float,
    ages_t: NDArray[np.float64] | float,
    T: float,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of log age-density for spine manuscripts.

    Parameters
    ----------
    lambda_param : float
        Birth rate λ. If `log_scale_params` is True, interpreted as log(λ).
    mu_param : float
        Death rate μ. If `log_scale_params` is True, interpreted as log(μ).
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T). Values at T use boundary formula.
    T : float
        Observation time T.
    log_scale_params : bool, optional
        If True, parameters are in log-scale. Default False.

    Returns
    -------
    NDArray[np.float64]
        Array [∂/∂λ, ∂/∂μ] containing summed derivatives across all ages.

    Notes
    -----
    Vectorized over ages. Handles λ ≈ μ limiting case for numerical stability.
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
    if abs(log_lambda - log_mu) < 1e-14:  # Tighter tolerance for derivatives
        # Limiting case: λ ≈ μ
        # At boundary: log f(T) = 2log(λ) + log(T) - λT, so ∂/∂λ = 2/λ - T
        # Interior: log f(t) = 2log(λ) + log(t) - λt, so ∂/∂λ = 2/λ - t

        if np.any(at_boundary):
            # d/dλ = 2/λ - T
            d_mu[at_boundary] = d_lambda[at_boundary] = 2 / lambda_val - T

        if np.any(not_at_boundary):
            # d/dλ = 2/λ - t
            ages_interior = (
                ages_t[not_at_boundary] if isinstance(ages_t, np.ndarray) else ages_t
            )
            d_mu[not_at_boundary] = d_lambda[not_at_boundary] = (
                2 / lambda_val - ages_interior
            )
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
            # Extract ages not at boundary for computation
            ages_interior = (
                ages_t[not_at_boundary] if isinstance(ages_t, np.ndarray) else ages_t
            )

            # Interior: log f(t) = log(λ) + log(μ) + log(e^{-μT} - e^{-λT}) - log(λ - μ)
            # Let B = e^{-μT} - e^{-λT}

            B = np.exp(-mu_val * T) - np.exp(-lambda_val * T)

            # ∂ log f(t)/∂λ = 1/λ + T·e^{-λT}/B - 1/(λ-μ)
            d_lambda[not_at_boundary] = (
                1 / lambda_val
                + T * np.exp(-lambda_val * T) / B
                - 1 / (lambda_val - mu_val)
            )

            # ∂ log f(t)/∂μ = 1/μ - T·e^{-μT}/B + 1/(λ-μ)
            d_mu[not_at_boundary] = (
                1 / mu_val - T * np.exp(-mu_val * T) / B + 1 / (lambda_val - mu_val)
            )

    return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)


def calculate_ages_log_lik_derivatives(
    lambda_param: float,
    mu_param: float,
    ages_t: NDArray[np.float64] | float,
    T: float,
    log_scale_params: bool = False,
    are_spines: bool = False,
) -> NDArray[np.float64]:
    """
    Compute derivatives of log age-density with respect to λ and μ.

    Parameters
    ----------
    lambda_param : float
        Birth rate λ. If `log_scale_params` is True, interpreted as log(λ).
    mu_param : float
        Death rate μ. If `log_scale_params` is True, interpreted as log(μ).
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T).
    T : float
        Observation time T.
    log_scale_params : bool, optional
        If True, parameters are in log-scale. Default False.
    are_spines : bool, optional
        If True, use spine calculation; otherwise non-spine. Default False.

    Returns
    -------
    NDArray[np.float64]
        Array [∂/∂λ, ∂/∂μ] containing summed derivatives across all ages.
    """
    if are_spines:
        return calculate_spine_ages_log_lik_derivatives(
            lambda_param, mu_param, ages_t, T, log_scale_params
        )
    else:
        return calculate_non_spine_ages_log_lik_derivatives(
            lambda_param, mu_param, ages_t, T, log_scale_params
        )


def calculate_extinction_pr(
    birth_rates: NDArray[np.float64] | float,
    death_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
    return_log_prob: bool = False,
) -> np.float64:
    """
    Calculate extinction probability for a birth-death process.

    Parameters
    ----------
    birth_rates : float or NDArray[np.float64]
        Birth rate(s) λ. If `log_scale_params` is True, interpreted as log(λ).
    death_rates : float or NDArray[np.float64]
        Death rate(s) μ. If `log_scale_params` is True, interpreted as log(μ).
    times : float or NDArray[np.float64]
        Observation time(s) T. For arrays, must be monotonically increasing.
    log_scale_params : bool, optional
        If True, parameters are in log-scale. Default False.
    return_log_prob : bool, optional
        If True, return log(α_t) instead of α_t. Default False.

    Returns
    -------
    np.float64
        Extinction probability α_t (or log(α_t) if return_log_prob=True).

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability.
    """
    pr_extinction_numerator = _pr_extinction_numerator(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        log_scale_params=log_scale_params,
    )
    if return_log_prob:
        log_alpha_t = np.log(pr_extinction_numerator) - np.log1p(
            pr_extinction_numerator
        )
        return np.float64(log_alpha_t)

    alpha_t = 1 - 1 / (1 + pr_extinction_numerator)

    return alpha_t


def calculate_log_lik_initial_lifetimes_given_extinction(
    lifetimes: NDArray[np.float64] | float,
    birth_rates: NDArray[np.float64] | float,
    death_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate log-likelihood of initial lifetime(s) given extinction.

    Parameters
    ----------
    lifetimes : float or NDArray[np.float64]
        Lifetime(s) of manuscript(s).
    birth_rates : float or NDArray[np.float64]
        Birth rate(s) λ for each epoch.
    death_rates : float or NDArray[np.float64]
        Death rate(s) μ for each epoch.
    times : float or NDArray[np.float64]
        Epoch end time(s).
    log_scale_params : bool, optional
        If True, rates are in log-scale. Default False.

    Returns
    -------
    np.float64
        Summed log-likelihood values.
    """
    lifetimes = np.atleast_1d(lifetimes)
    n = len(lifetimes)

    log_prs_extinction_given_lifetimes = (
        calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes=lifetimes,
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            log_scale_params=log_scale_params,
        )
    )
    log_liks_lifetimes = calculate_log_lik_lifetimes(
        lifetimes=lifetimes,
        birth_rates=birth_rates,
        times=times,
        log_scale_params=log_scale_params,
    )
    log_extinction_prob = calculate_extinction_pr(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        log_scale_params=log_scale_params,
        return_log_prob=True,
    )

    # Compute conditional log likelihood
    result = (
        np.sum(log_prs_extinction_given_lifetimes)
        + np.sum(log_liks_lifetimes)
        - n * log_extinction_prob
    )
    return result


def calculate_log_lik_initial_lifetimes_given_non_extinction(
    lifetimes: NDArray[np.float64] | float,
    birth_rates: NDArray[np.float64] | float,
    death_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate log-likelihood of initial lifetime(s) given non-extinction.

    Parameters
    ----------
    lifetimes : float or NDArray[np.float64]
        Lifetime(s) of manuscript(s).
    birth_rates : float or NDArray[np.float64]
        Birth rate(s) λ for each epoch.
    death_rates : float or NDArray[np.float64]
        Death rate(s) μ for each epoch.
    times : float or NDArray[np.float64]
        Epoch end time(s).
    log_scale_params : bool, optional
        If True, rates are in log-scale. Default False.

    Returns
    -------
    np.float64
        Summed log-likelihood values.
    """
    lifetimes = np.atleast_1d(lifetimes)
    n = len(lifetimes)

    log_prs_extinction_given_lifetimes = (
        calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes=lifetimes,
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            log_scale_params=log_scale_params,
        )
    )
    log_liks_lifetimes = calculate_log_lik_lifetimes(
        lifetimes=lifetimes,
        birth_rates=birth_rates,
        times=times,
        log_scale_params=log_scale_params,
    )
    log_extinction_prob = calculate_extinction_pr(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        log_scale_params=log_scale_params,
        return_log_prob=True,
    )

    # Compute conditional log likelihood
    result = (
        np.log(-np.expm1(log_prs_extinction_given_lifetimes))
        + log_liks_lifetimes
        - n * np.log(-np.expm1(log_extinction_prob))
    )
    return np.float64(result)


def calculate_log_pr_extinction_given_initial_lifetimes(
    lifetimes: NDArray[np.float64] | float,
    birth_rates: NDArray[np.float64] | float,
    death_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate log probability of extinction given initial lifetime(s).

    Parameters
    ----------
    lifetimes : float or NDArray[np.float64]
        Lifetime(s) of manuscript(s).
    birth_rates : float or NDArray[np.float64]
        Birth rate(s) λ for each epoch.
    death_rates : float or NDArray[np.float64]
        Death rate(s) μ for each epoch.
    times : float or NDArray[np.float64]
        Epoch end time(s).
    log_scale_params : bool, optional
        If True, rates are in log-scale. Default False.

    Returns
    -------
    np.float64
        Sum of log extinction probabilities.
    """
    lifetimes = np.atleast_1d(lifetimes)
    birth_rates = np.atleast_1d(birth_rates)
    death_rates = np.atleast_1d(death_rates)
    times = np.atleast_1d(times)

    if len(birth_rates) != len(death_rates) or len(birth_rates) != len(times):
        raise ValueError("Length of birth_rates, death_rates, and times must be equal.")

    if log_scale_params:
        birth_rates_array = np.exp(birth_rates)
    else:
        birth_rates_array = birth_rates

    # Sort lifetimes to enable cumulative integral caching
    sorted_lifetimes = np.sort(lifetimes)

    log_prob_sum = 0.0
    cumulative_integral = 0.0
    prev_lifetime = 0.0

    # Process sorted lifetimes, computing integrals only over new segments
    for lifetime in sorted_lifetimes:
        # Compute integral from prev_lifetime to lifetime (only the new segment)
        if lifetime > prev_lifetime:
            integral_segment = scipy.integrate.quad(
                _integrand_surv_from_weighted_birth_time,
                prev_lifetime,
                lifetime,
                args=(birth_rates, death_rates, times, log_scale_params),
            )[0]
            cumulative_integral += integral_segment

        # Compute cumulative birth rate for this lifetime
        full_epochs = times[np.where(times < lifetime)[0]]
        num_full_epochs = len(full_epochs)
        lifetime_epochs = np.append(full_epochs, lifetime)
        if len(birth_rates_array[0 : num_full_epochs + 1]) != len(
            np.diff(np.insert(lifetime_epochs, 0, 0.0))
        ):
            raise ValueError(
                "Mismatch in lengths when computing cumulative birth rates."
                + " birth_rates_array length: "
                + f"{len(birth_rates_array[0 : num_full_epochs + 1])}, "
                + f" lifetime: {lifetime}, times: {times}.\n"
                + f" lifetime_epochs: {lifetime_epochs}"
            )
        cum_birth_rate = np.sum(
            birth_rates_array[0 : num_full_epochs + 1]
            * np.diff(np.insert(lifetime_epochs, 0, 0.0))
        )

        log_prob_sum += cumulative_integral - cum_birth_rate
        prev_lifetime = lifetime

    return np.float64(log_prob_sum)


def calculate_log_lik_lifetimes(
    lifetimes: NDArray[np.float64] | float,
    birth_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate log-likelihood of manuscript lifetime(s).

    Parameters
    ----------
    lifetimes : float or NDArray[np.float64]
        Lifetime(s) of manuscript(s).
    birth_rates : float or NDArray[np.float64]
        Birth rate(s) λ for each epoch.
    times : float or NDArray[np.float64]
        Epoch end time(s).
    log_scale_params : bool, optional
        If True, rates are in log-scale. Default False.

    Returns
    -------
    np.float64
        Sum of log-likelihood values.
    """
    # Handle scalar input
    lifetimes = np.atleast_1d(lifetimes)
    birth_rates = np.atleast_1d(birth_rates)
    times = np.atleast_1d(times)

    if log_scale_params:
        lambda_val_array = np.exp(birth_rates)
    else:
        lambda_val_array = birth_rates

    epoch_starts = np.insert(times, 0, 0.0)[:-1]
    epoch_durations = times - epoch_starts

    cum_growth_at_epochs = np.cumsum(lambda_val_array * epoch_durations)

    log_liks = np.zeros_like(lifetimes, dtype=np.float64)

    for i, lifetime in enumerate(lifetimes):
        if lifetime == times[-1]:
            # Boundary case
            log_liks[i] = -cum_growth_at_epochs[-1]
            continue

        valid_indices = np.where(epoch_starts < lifetime)[0]

        # # The current epoch index is valid_indices[-1], but we need to cap it
        # # at len(lambda_val_array)-1 since epoch_starts has one extra element
        current_epoch_idx = valid_indices[-1]
        current_lambda = lambda_val_array[current_epoch_idx]

        cum_growth = (
            cum_growth_at_epochs[current_epoch_idx] if current_epoch_idx > 0 else 0.0
        ) + (lifetime - epoch_starts[current_epoch_idx]) * current_lambda

        log_liks[i] = np.log(current_lambda) - cum_growth

    return np.sum(log_liks)


def _population_geometric_parameter(
    birth_rates: NDArray[np.float64] | float,
    death_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate geometric parameter (1 - β_t) for offspring distribution.

    Private helper function for computing the conditional geometric parameter
    of the population distribution given survival.

    Parameters
    ----------
    birth_rates : float or NDArray[np.float64]
        Birth rate(s) λ. If `log_scale_params` is True, interpreted as log(λ).
    death_rates : float or NDArray[np.float64]
        Death rate(s) μ. If `log_scale_params` is True, interpreted as log(μ).
    times : float or NDArray[np.float64]
        Epoch end time(s). Must be monotonically increasing.
    log_scale_params : bool, optional
        If True, rates are in log-scale. Default False.

    Returns
    -------
    np.float64
        Geometric parameter 1 - β_t ∈ (0, 1).

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability.
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

    geom_sum = _pr_extinction_numerator(
        birth_rates, death_rates, times, log_scale_params
    )
    # Compute in log-domain to avoid overflow: cond = exp(-C) / (1 + geom_sum)
    # => log(cond) = -C - log1p(geom_sum)
    log_cond = -cum_growth[-1] - np.log1p(geom_sum)
    cond_geom_prob = np.exp(log_cond)
    return np.float64(cond_geom_prob)


def _integrand_surv_from_weighted_birth_time(
    birth_time: float,
    birth_rates: NDArray[np.float64] | float,
    death_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Integrand for survival probability calculation.

    Parameters
    ----------
    birth_time : float
        Time at which birth occurred.
    birth_rates : float or NDArray[np.float64]
        Birth rate(s) λ.
    death_rates : float or NDArray[np.float64]
        Death rate(s) μ.
    times : float or NDArray[np.float64]
        Epoch end time(s).
    log_scale_params : bool, optional
        If True, rates are in log-scale. Default False.

    Returns
    -------
    np.float64
        Integrand value.
    """
    # Find the index of the last epoch that starts before or at start_time
    if log_scale_params:
        birth_rates = np.exp(birth_rates)
        death_rates = np.exp(death_rates)
    birth_rates = np.atleast_1d(birth_rates)
    death_rates = np.atleast_1d(death_rates)
    times = np.atleast_1d(times) - birth_time
    valid_indices = np.where(times > 0)[0]

    if len(valid_indices) == 0:
        raise ValueError("birth_time exceeds all epoch end times.")

    trunc_times = times[valid_indices]
    trunc_birth_rates = birth_rates[valid_indices]
    trunc_death_rates = death_rates[valid_indices]

    ext_prob = calculate_extinction_pr(
        birth_rates=trunc_birth_rates,
        death_rates=trunc_death_rates,
        times=trunc_times,
        log_scale_params=False,
    )

    return np.float64(ext_prob * trunc_birth_rates[0])


def _pr_extinction_numerator(
    birth_rates: NDArray[np.float64] | float,
    death_rates: NDArray[np.float64] | float,
    times: NDArray[np.float64] | float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate numerator of extinction probability integral.

    Private helper function for extinction probability calculation.

    Parameters
    ----------
    birth_rates : float or NDArray[np.float64]
        Birth rate(s) λ. If `log_scale_params` is True, interpreted as log(λ).
    death_rates : float or NDArray[np.float64]
        Death rate(s) μ. If `log_scale_params` is True, interpreted as log(μ).
    times : float or NDArray[np.float64]
        Epoch end time(s). Must be monotonically increasing.
    log_scale_params : bool, optional
        If True, rates are in log-scale. Default False.

    Returns
    -------
    np.float64
        Numerator value for extinction probability.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability.
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

    epoch_durations = np.diff(np.insert(times, 0, 0.0))
    critical_mask = np.abs(log_lambda_array - log_mu_array) < 1e-12
    growth_rates = lambda_val_array - mu_val_array
    cum_growth = np.cumsum(growth_rates * epoch_durations)
    trunc_cum_growth = np.insert(cum_growth, 0, 0.0)[:-1]

    # Stable evaluation of (1 - exp(-x)) / x via expm1 to avoid overflow/underflow
    x = growth_rates * epoch_durations
    stable_frac = -np.expm1(-x) / np.where(growth_rates == 0.0, 1.0, growth_rates)
    kernel = np.where(critical_mask, epoch_durations, stable_frac)
    pr_extinction_numerator_summand = mu_val_array * np.exp(-trunc_cum_growth) * kernel

    return np.sum(pr_extinction_numerator_summand, dtype=np.float64)
