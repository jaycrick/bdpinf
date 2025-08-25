"""
Parameter fitting and estimation functions for birth-death process models.

This module contains functions for fitting birth-death process parameters
to manuscript data using maximum likelihood estimation and various optimization
strategies.

Examples:
    Basic parameter fitting:

    >>> import numpy as np
    >>> from inf_bp_texts.inference import fit_birth_death_process
    >>>
    >>> # Simulate some manuscript ages
    >>> ages = np.array([50, 100, 150, 200, 250])
    >>> observation_time = 300
    >>>
    >>> # Fit the model
    >>> result = fit_birth_death_process(ages, observation_time)
    >>> if result['success']:
    ...     print(f"Birth rate: {result['birth_rate']:.3f}")
    ...     print(f"Death rate: {result['death_rate']:.3f}")

    Advanced usage with custom options:

    >>> # Fit with initial parameter guess and bounds
    >>> result = fit_birth_death_process(
    ...     ages, observation_time,
    ...     initial_params=(0.5, 0.3),
    ...     bounds=[(0.1, 2.0), (0.1, 1.0)]
    ... )
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize

from .likelihood import (
    log_likelihood,
    negative_log_likelihood,
)


def get_crude_parameter_estimates(
    ages: NDArray[np.float64],
    observation_time: float,
) -> Tuple[float, float]:
    """
    Get crude initial parameter estimates based on manuscript age data.

    Uses simple heuristics to estimate birth and death rates:
    - Death rate inversely related to max survival age
    - Birth rate related to total observed manuscripts

    Args:
        ages: Array of manuscript ages
        observation_time: Total observation time

    Returns:
        Tuple of (estimated_birth_rate, estimated_death_rate) in log scale
    """
    if len(ages) == 0:
        raise ValueError("Cannot estimate parameters from empty age data")

    mean_age = float(ages.mean())
    n_manuscripts = len(ages)

    # Simple heuristic: death rate inversely related to max survival age
    # birth rate related to total observed manuscripts
    est_birth_rate = 1.0 / mean_age
    est_death_rate = np.max(
        [est_birth_rate - np.log(n_manuscripts) / observation_time, 1e-8]
    )

    # Convert to log scale for optimization
    return (np.log(est_birth_rate), np.log(est_death_rate))


def fit_birth_death_process(
    ages: NDArray[np.float64],
    observation_time: float,
    counts: Optional[NDArray[np.float64]] = None,
    method: str = "L-BFGS-B",
    initial_params: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Fit birth-death process parameters to manuscript data.

    Optimization is performed over log-scale parameters for better numerical stability.

    Args:
        ages: Array of manuscript ages
        observation_time: Total observation time
        counts: Array of manuscript counts for each age (defaults to 1 for each)
        method: Optimization method for scipy.optimize.minimize
        initial_params: Initial parameter values (birth_rate, death_rate) in original scale

    Returns:
        Dictionary with fit results including parameters and statistics
    """
    # Input validation
    if len(ages) == 0:
        raise ValueError("No valid data points for fitting")

    if observation_time <= 0:
        raise ValueError("observation_time must be positive")

    # Set default counts if not provided
    if counts is None:
        counts = np.ones_like(ages, dtype=np.float64)
    elif len(counts) != len(ages):
        raise ValueError("counts and ages must have the same length")

    # Remove zero-count entries
    non_zero_mask = counts > 0
    ages_fit = ages[non_zero_mask]
    counts_fit = counts[non_zero_mask]

    if len(ages_fit) == 0:
        raise ValueError("No valid data points after removing zero counts")

    # Set initial parameters with better estimates based on data
    if initial_params is None:
        initial_params = get_crude_parameter_estimates(ages_fit, observation_time)
    else:
        # Convert provided initial parameters to log scale
        initial_params = (np.log(initial_params[0]), np.log(initial_params[1]))

    # Parameter bounds in log space: log(birth_rate), log(death_rate)
    # birth_rate: 1e-6 to 10.0 -> log bounds: log(1e-6) to log(10.0)
    # death_rate: 1e-8 to 1.0 -> log bounds: log(1e-8) to log(1.0)
    bounds: List[Tuple[Optional[float], Optional[float]]] = [
        (np.log(1e-6), np.log(10.0)),  # log(birth_rate) bounds
        (np.log(1e-8), np.log(1.0)),  # log(death_rate) bounds
    ]

    # Create objective function wrapper
    def objective_func(params: NDArray[np.float64]) -> float:
        return negative_log_likelihood(
            log_params=params,
            ages=ages_fit,
            observation_time=observation_time,
            counts=counts_fit,
        )

    # Optimize with multiple strategies for robustness
    optimization_methods = [method, "Nelder-Mead", "Powell"]
    best_result: Any = None
    best_likelihood = -np.inf

    for opt_method in optimization_methods:
        try:
            # Ensure parameters are numpy arrays for scipy compatibility
            x0 = np.array(initial_params, dtype=np.float64)

            result: Any = optimize.minimize(
                objective_func,
                x0,
                method=opt_method,  # type: ignore
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

            if result.success:
                # Check if this is the best result so far
                likelihood = -result.fun
                if likelihood > best_likelihood:
                    best_result = result
                    best_likelihood = likelihood

            elif best_result is None:
                # Keep even failed results if we have nothing better
                best_result = result

        except Exception as e:
            # Continue trying other methods
            if best_result is None:
                # Store the error for the first method only
                return {
                    "success": False,
                    "error": f"All optimization methods failed. First error: {str(e)}",
                }
            continue

    # Process the best result found
    if best_result is not None:
        result = best_result

        if result.success:
            # Convert log-scale parameters back to original scale
            log_birth_rate, log_death_rate = result.x
            birth_rate = np.float64(np.exp(log_birth_rate))
            death_rate = np.float64(np.exp(log_death_rate))

            # Calculate additional statistics using fitted parameters
            log_like = log_likelihood(
                birth_rate=float(birth_rate),
                death_rate=float(death_rate),
                ages=ages_fit,
                observation_time=observation_time,
                counts=counts_fit,
            )

            n_data = len(ages_fit)

            return {
                "success": True,
                "birth_rate": birth_rate,
                "death_rate": death_rate,
                "log_likelihood": log_like,
                "n_data_points": n_data,
                "optimization_result": result,
                "ages_used": ages_fit,
                "counts_used": counts_fit,
            }

        else:
            return {
                "success": False,
                "message": result.message,
                "optimization_result": result,
            }
    else:
        # No optimization succeeded
        return {
            "success": False,
            "error": "All optimization methods failed",
        }


def fit_from_dataframe(
    manuscript_data: pd.DataFrame,
    age_column: str = "age",
    count_column: Optional[str] = None,
    observation_time: Optional[float] = None,
    method: str = "L-BFGS-B",
    initial_params: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Fit birth-death process parameters from a pandas DataFrame.

    Args:
        manuscript_data: DataFrame with manuscript information
        age_column: Column name containing manuscript ages
        count_column: Column name containing manuscript counts (optional)
        observation_time: Total observation time (if None, uses max age + buffer)
        method: Optimization method for scipy.optimize.minimize
        initial_params: Initial parameter values (birth_rate, death_rate) in original scale

    Returns:
        Dictionary with fit results including parameters and statistics
    """
    # Extract ages from DataFrame
    if age_column not in manuscript_data.columns:
        raise ValueError(f"Column '{age_column}' not found in manuscript_data")

    ages = manuscript_data[age_column].values
    ages_array = np.asarray(ages, dtype=np.float64)

    # Extract counts if provided
    counts_array = None
    if count_column is not None:
        if count_column not in manuscript_data.columns:
            raise ValueError(f"Column '{count_column}' not found in manuscript_data")
        counts = manuscript_data[count_column].values
        counts_array = np.asarray(counts, dtype=np.float64)

    # Set observation_time if not provided
    if observation_time is None:
        max_age = float(ages_array.max()) if len(ages_array) > 0 else 0.0
        observation_time = max_age + 100.0  # Add buffer beyond maximum observed age

    return fit_birth_death_process(
        ages=ages_array,
        observation_time=observation_time,
        counts=counts_array,
        method=method,
        initial_params=initial_params,
    )


def bootstrap_confidence_intervals(
    ages: NDArray[np.float64],
    observation_time: float,
    counts: Optional[NDArray[np.float64]] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = "L-BFGS-B",
) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence intervals for birth-death process parameters.

    Args:
        ages: Array of manuscript ages
        observation_time: Total observation time
        counts: Array of manuscript counts for each age
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: Optimization method

    Returns:
        Dictionary with confidence intervals and bootstrap results
    """
    if len(ages) == 0:
        raise ValueError("No data for bootstrap analysis")

    # Set default counts if not provided
    if counts is None:
        counts = np.ones_like(ages, dtype=np.float64)

    # Original fit
    original_fit = fit_birth_death_process(
        ages=ages,
        observation_time=observation_time,
        counts=counts,
        method=method,
    )

    if not original_fit["success"]:
        return {"success": False, "error": "Original fit failed"}

    # Bootstrap samples
    bootstrap_birth_rates: List[float] = []
    bootstrap_death_rates: List[float] = []
    successful_fits = 0

    np.random.seed(42)  # For reproducibility

    for _ in range(n_bootstrap):
        # Resample with replacement
        n_samples = len(ages)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

        bootstrap_ages = ages[bootstrap_indices]
        bootstrap_counts = counts[bootstrap_indices]

        try:
            # Fit to bootstrap sample
            bootstrap_fit = fit_birth_death_process(
                ages=bootstrap_ages,
                observation_time=observation_time,
                counts=bootstrap_counts,
                method=method,
            )

            if bootstrap_fit["success"]:
                bootstrap_birth_rates.append(float(bootstrap_fit["birth_rate"]))
                bootstrap_death_rates.append(float(bootstrap_fit["death_rate"]))
                successful_fits += 1

        except Exception:
            # Skip failed fits
            continue

    if successful_fits == 0:
        return {"success": False, "error": "No successful bootstrap fits"}

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    birth_rate_ci = np.percentile(
        bootstrap_birth_rates, [lower_percentile, upper_percentile]
    )
    death_rate_ci = np.percentile(
        bootstrap_death_rates, [lower_percentile, upper_percentile]
    )

    return {
        "success": True,
        "original_fit": original_fit,
        "n_bootstrap": n_bootstrap,
        "successful_fits": successful_fits,
        "confidence_level": confidence_level,
        "birth_rate_ci": birth_rate_ci,
        "death_rate_ci": death_rate_ci,
        "birth_rate_std": np.std(bootstrap_birth_rates),
        "death_rate_std": np.std(bootstrap_death_rates),
        "bootstrap_birth_rates": bootstrap_birth_rates,
        "bootstrap_death_rates": bootstrap_death_rates,
    }
