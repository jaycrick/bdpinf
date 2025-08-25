"""
Likelihood calculation functions for birth-death process parameter estimation.

This module contains functions for calculating log-likelihood values for
birth-death process models, used in parameter fitting and model evaluation.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..models.mathematical_functions import (
    calculate_age_density,
    calculate_log_population_probability,
)


def log_likelihood(
    birth_rate: float,
    death_rate: float,
    ages: NDArray[np.float64],
    observation_time: float,
    counts: Optional[NDArray[np.float64]] = None,
    log_scale_params: bool = False,
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

        # Set default counts if not provided
        if counts is None:
            counts = np.ones_like(ages, dtype=np.float64)
        elif len(counts) != len(ages):
            raise ValueError("counts and ages must have the same length")

        # Additional validation: observation_time should be positive
        if observation_time <= 0:
            return np.float64(-np.inf)

        # Total number of observed manuscripts
        n_total: int = int(np.sum(counts))

        # 1. Calculate log(P_n(T)) - log probability of n individuals at time T
        log_P_n_T = calculate_log_population_probability(
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
        log_densities = np.zeros_like(ages, dtype=np.float64)

        for i, age_t in enumerate(ages):
            # Skip zero-count entries
            if counts[i] == 0:
                continue

            # Validate individual age vs. total time
            if age_t > observation_time:
                return np.float64(-np.inf)

            # Calculate log age density
            log_density = calculate_age_density(
                birth_rate,
                death_rate,
                age_t,
                observation_time,
                log_scale_params=log_scale_params,
            )

            # Check if age density calculation succeeded
            if not np.isfinite(log_density):
                return np.float64(-np.inf)

            # Add log density weighted by count
            log_densities[i] = counts[i] * log_density

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


def negative_log_likelihood(
    log_params: NDArray[np.float64],
    ages: NDArray[np.float64],
    observation_time: float,
    counts: Optional[NDArray[np.float64]] = None,
) -> float:
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
    if len(log_params) != 2:
        return float(np.inf)

    # Extract log-scale parameters and convert to original scale
    log_birth_rate, log_death_rate = log_params
    birth_rate = float(np.exp(log_birth_rate))
    death_rate = float(np.exp(log_death_rate))

    result = -log_likelihood(
        birth_rate=birth_rate,
        death_rate=death_rate,
        ages=ages,
        observation_time=observation_time,
        counts=counts,
        log_scale_params=False,
    )

    return float(result)
