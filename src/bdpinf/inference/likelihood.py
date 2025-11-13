"""
Likelihood calculation functions for birth-death process parameter estimation.

This module contains functions for calculating log-likelihood values for
birth-death process models, used in parameter fitting and model evaluation.
"""

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from ..models.mathematical_functions import (
    calculate_ages_log_likelihood,
    calculate_ages_log_likelihood_derivatives,
    calculate_populations_log_likelihood,
    calculate_populations_log_likelihood_derivatives,
)


def _calculate_single_age_density(
    i: int,
    age_t: float,
    birth_rate: float,
    death_rate: float,
    observation_time: float,
    is_ordered: bool,
    log_scale_params: bool,
) -> tuple[int, np.float64]:
    """
    Helper function to calculate log density for a single age.

    Returns tuple of (index, log_density) for parallel processing.
    """
    if age_t > observation_time:
        return (i, np.float64(-np.inf))

    are_spines = is_ordered and (i == 0)

    log_density = calculate_ages_log_likelihood(
        birth_rate,
        death_rate,
        age_t,
        observation_time,
        log_scale_params=log_scale_params,
        are_spines=are_spines,
    )

    return (i, np.float64(log_density))


def _calculate_single_gradient(
    i: int,
    age_t: float,
    birth_rate: float,
    death_rate: float,
    observation_time: float,
    is_ordered: bool,
    log_scale_params: bool,
) -> tuple[int, NDArray[np.float64]]:
    """
    Helper function to calculate gradient for a single age.

    Returns tuple of (index, gradient) for parallel processing.
    """
    are_spines = is_ordered and (i == 0)

    grad_density = calculate_ages_log_likelihood_derivatives(
        birth_rate,
        death_rate,
        age_t,
        observation_time,
        log_scale_params=log_scale_params,
        are_spines=are_spines,
    )

    return (i, grad_density)


def log_likelihood(
    birth_rate: float,
    death_rate: float,
    ages: NDArray[np.float64],
    observation_time: float,
    is_ordered: bool = False,
    log_scale_params: bool = False,
    parallel: bool = False,
) -> np.float64:
    """
    Calculate log-likelihood for observed manuscript data using birth-death process.

    This implements the proper birth-death process likelihood with:
    1. Probability of n individuals at time T: P_n(T)
    2. Age density for each manuscript: f(t)
    3. Total log-likelihood = log(P_n(T)) + Σlog(f(t_i))

    Args:
        birth_rate: Birth rate (λ > 0) or log(λ) if log_scale_params=True
        death_rate: Death rate (μ > 0) or log(μ) if log_scale_params=True
        ages: Array of manuscript ages
        observation_time: Observation time T
        is_ordered: If True, the first age is treated as having a unique distribution
        counts: Array of manuscript counts for each age (defaults to 1 for each)
        log_scale_params: If True, birth_rate and death_rate are in log scale

    Returns:
        Log-likelihood value
    """
    try:
        # Input validation
        if not (np.isfinite(birth_rate) and np.isfinite(death_rate)):
            return np.float64(-np.inf)

        if not log_scale_params:
            # Parameters are in original scale, validate directly
            if birth_rate <= 0 or death_rate <= 0:
                return np.float64(-np.inf)

        # Check for negative ages
        if np.any(ages < 0):
            return np.float64(-np.inf)

        # Additional validation: observation_time should be positive
        if observation_time <= 0:
            return np.float64(-np.inf)

        # Total number of observed manuscripts
        n_total: int = len(ages)

        # 1. Calculate log(P_n(T)) - log probability of n individuals at time T
        log_P_n_T = calculate_populations_log_likelihood(
            birth_rate,
            death_rate,
            observation_time,
            n_total,
            log_scale_params=log_scale_params,
        )

        # Check if population probability calculation succeeded
        if not np.isfinite(log_P_n_T):
            return np.float64(-np.inf)

        if n_total == 0:
            return np.float64(log_P_n_T)

        # If no ages provided, return just the population probability
        if len(ages) == 0:
            return np.float64(-np.inf)

        # 2. Calculate age density for each manuscript
        # Use parallel processing for large datasets (>100 ages)
        if len(ages) > 100 and parallel:
            # Parallel computation
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(_calculate_single_age_density)(
                    i,
                    age_t,
                    birth_rate,
                    death_rate,
                    observation_time,
                    is_ordered,
                    log_scale_params,
                )
                for i, age_t in enumerate(ages)
            )

            # Extract log densities in correct order
            log_densities = np.zeros_like(ages, dtype=np.float64)
            for i, log_density in results:
                if not np.isfinite(log_density):
                    return np.float64(-np.inf)
                log_densities[i] = log_density
        else:
            # Sequential computation for small datasets
            log_densities = np.zeros_like(ages, dtype=np.float64)

            for i, age_t in enumerate(ages):
                # Validate individual age vs. total time
                if age_t > observation_time:
                    return np.float64(-np.inf)
                if is_ordered and i == 0:
                    # Calculate log age density
                    log_density = calculate_ages_log_likelihood(
                        birth_rate,
                        death_rate,
                        age_t,
                        observation_time,
                        log_scale_params=log_scale_params,
                        are_spines=True,
                    )
                else:
                    log_density = calculate_ages_log_likelihood(
                        birth_rate,
                        death_rate,
                        age_t,
                        observation_time,
                        log_scale_params=log_scale_params,
                        are_spines=False,
                    )

                # Check if age density calculation succeeded
                if not np.isfinite(log_density):
                    return np.float64(-np.inf)

                # Add log density weighted by count
                log_densities[i] = log_density

        # 3. Total log-likelihood = log(P_n(T)) + sum of log densities
        log_density_sum = np.sum(log_densities)

        if not np.isfinite(log_density_sum):
            return np.float64(-np.inf)

        total_log_likelihood = log_P_n_T + log_density_sum

        # Final numerical validation
        if not np.isfinite(total_log_likelihood):
            return np.float64(-np.inf)

        return np.float64(total_log_likelihood)

    except (
        ValueError,
        FloatingPointError,
        OverflowError,
        ZeroDivisionError,
        IndexError,
    ):
        return np.float64(-np.inf)


def log_likelihood_gradient(
    params: NDArray[np.float64],
    ages: NDArray[np.float64],
    observation_time: float,
    is_ordered: bool = False,
    log_scale_params: bool = False,
    parallel: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate gradient of log-likelihood w.r.t. log-scale parameters.

    Args:
        params: parameters [birth_rate, death_rate] (possibly log-scale)
        ages: Array of manuscript ages
        observation_time: Observation time T

    Returns:
        Gradient array [dL/dlog(birth_rate), dL/dlog(death_rate)]
    """
    if len(params) != 2:
        return np.array([0.0, 0.0], dtype=np.float64)

    birth_rate, death_rate = params

    # Input validation
    if not (np.isfinite(birth_rate) and np.isfinite(death_rate)):
        raise ValueError("params must be finite numbers")

    # Check for negative ages
    if np.any(ages < 0):
        return np.array([0.0, 0.0], dtype=np.float64)

    # Additional validation: observation_time should be positive
    if observation_time <= 0:
        raise ValueError("observation_time must be positive")

    # Total number of observed manuscripts
    n_total: int = len(ages)

    # Gradient components
    grad_log_P_n_T = calculate_populations_log_likelihood_derivatives(
        birth_rate,
        death_rate,
        observation_time,
        n_total,
        log_scale_params=log_scale_params,
    )

    if n_total == 0:
        return grad_log_P_n_T

    # Use parallel processing for large datasets (>100 ages)
    if len(ages) > 100 and parallel:
        # Parallel computation
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_calculate_single_gradient)(
                i,
                age_t,
                birth_rate,
                death_rate,
                observation_time,
                is_ordered,
                log_scale_params,
            )
            for i, age_t in enumerate(ages)
        )

        # Sum gradients
        grad_log_densities = np.zeros(2, dtype=np.float64)
        for i, grad_density in results:
            grad_log_densities += grad_density
    else:
        # Sequential computation for small datasets
        grad_log_densities = np.zeros(2, dtype=np.float64)

        for i, age_t in enumerate(ages):
            # Calculate derivatives of log age density
            if is_ordered and i == 0:
                grad_density = calculate_ages_log_likelihood_derivatives(
                    birth_rate,
                    death_rate,
                    age_t,
                    observation_time,
                    log_scale_params=log_scale_params,
                    are_spines=True,
                )
            else:
                grad_density = calculate_ages_log_likelihood_derivatives(
                    birth_rate,
                    death_rate,
                    age_t,
                    observation_time,
                    log_scale_params=log_scale_params,
                    are_spines=False,
                )
            grad_log_densities += grad_density

    return grad_log_P_n_T + grad_log_densities


def negative_log_likelihood(
    params: NDArray[np.float64],
    ages: NDArray[np.float64],
    observation_time: float,
    is_ordered: bool = False,
    log_scale_params: bool = False,
    parallel: bool = False,
) -> np.float64:
    """
    Calculate negative log-likelihood for optimization.

    This method is designed for use with scipy.optimize functions. It takes
    log-scale parameters for better numerical stability during optimization.

    Args:
        log_params: Log-scale parameters [log(birth_rate), log(death_rate)]
        ages: Array of manuscript ages
        observation_time: Observation time T
        counts: Array of manuscript counts for each age (defaults to 1 for each)

    Returns:
        Negative log-likelihood value (float for scipy optimization)
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
        is_ordered=is_ordered,
        log_scale_params=log_scale_params,
        parallel=parallel,
    )

    return result
