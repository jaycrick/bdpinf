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

from .likelihood import log_likelihood_gradient, negative_log_likelihood


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
        Tuple of (estimated_birth_rate, estimated_death_rate)
    """
    n_manuscripts = len(ages)
    if n_manuscripts == 0:
        raise ValueError("Cannot estimate parameters from empty age data")

    mean_age = float(ages.mean())

    # Simple heuristic: death rate inversely related to max survival age
    # birth rate related to total observed manuscripts
    est_birth_rate = 1.0 / mean_age
    est_death_rate = np.max(
        [est_birth_rate - np.log(n_manuscripts) / observation_time, 1e-8]
    )

    # Convert to log scale for optimization
    return (est_birth_rate, est_death_rate)


def fit_birth_death_processes(
    list_of_ages: List[NDArray[np.float64]],
    observation_time: float,
    is_ordered: bool = False,
    method: str = "L-BFGS-B",
    initial_params: Tuple[float, float] | None = None,
    spine_ages: NDArray[np.float64] | None = None,
    ns: NDArray[np.int64] | None = None,
) -> Dict[str, Any]:
    """
    Fit birth-death process parameters to manuscript data.

    Optimization is performed over log-scale parameters for better numerical stability.

    Args:
        ages: Array of manuscript ages
        observation_time: Total observation time
        method: Optimization method for scipy.optimize.minimize
        initial_params: Initial parameter values (birth_rate, death_rate)

    Returns:
        Dictionary with fit results including parameters and statistics
    """
    # Input validation
    non_empty_age = None
    for ages in list_of_ages:
        if len(ages) > 0:
            non_empty_age = ages
            continue

    if non_empty_age is None:
        raise ValueError("No valid data points for fitting")

    if observation_time <= 0:
        raise ValueError("observation_time must be positive")

    # Set initial parameters with better estimates based on data
    if initial_params is None:
        initial_params = get_crude_parameter_estimates(non_empty_age, observation_time)
    # Convert initial parameters to log scale
    log_initial_params = np.log(initial_params)

    # Parameter bounds in log space: log(birth_rate), log(death_rate)
    # birth_rate: 1e-8 to 100.0 -> log bounds: log(1e-8) to log(100.0)
    # death_rate: 1e-8 to 100.0 -> log bounds: log(1e-8) to log(100.0)
    bounds: List[Tuple[Optional[float], Optional[float]]] = [
        (np.log(1e-8), np.log(100.0)),  # log(birth_rate) bounds
        (np.log(1e-8), np.log(100.0)),  # log(death_rate) bounds
    ]

    if ns is not None:
        if sum(ns) != len(list_of_ages):
            raise ValueError("Length of ns must match length of list_of_ages")
    else:
        ns = np.array(
            [len(ages) + int(spine_ages is not None) for ages in list_of_ages],
            dtype=np.int64,
        )

    if is_ordered and (spine_ages is None):
        spine_list: List[float] = []
        remaining: List[NDArray[np.float64]] = []
        for arr in list_of_ages:
            if arr.size == 0:
                continue
            # collect first element as spine
            spine_list.append(float(arr[0]))
            # keep the rest for the main ages list
            if arr.size > 1:
                remaining.append(arr[1:])

        spine_ages = np.array(spine_list, dtype=np.float64)

        # If no remaining non-spine ages, ensure we have an empty array instead of failing on concatenate
        if remaining:
            non_spine_ages = np.atleast_1d(np.concatenate(remaining))
        else:
            non_spine_ages = np.array([], dtype=np.float64)
    else:
        # For unordered data, combine all non-empty age arrays into one array
        non_spine_ages = np.concatenate([arr for arr in list_of_ages if arr.size > 0])

    # Create objective function wrapper
    def objective_func(log_params: NDArray[np.float64]) -> np.float64:
        # Convert log-scale parameters back to original scale
        params = np.exp(log_params)
        # Single age array - no parallelization needed
        nll = negative_log_likelihood(
            params=params,
            ages=non_spine_ages,
            observation_time=observation_time,
            spine_ages=spine_ages,
            ns=ns,
            is_ordered=False,
            log_scale_params=False,
        )
        return nll

    def gradient_func(log_params: NDArray[np.float64]) -> NDArray[np.float64]:
        # Convert log-scale parameters back to original scale
        params = np.exp(log_params)
        nll_grad = np.multiply(
            params,
            -log_likelihood_gradient(
                params=params,
                ages=non_spine_ages,
                observation_time=observation_time,
                spine_ages=spine_ages,
                is_ordered=is_ordered,
                log_scale_params=False,
            ),
        )
        return np.array(nll_grad)

    # Optimize with multiple strategies for robustness
    optimization_methods = [method, "Nelder-Mead", "Powell"]
    best_result: Any = None
    best_likelihood = -np.inf

    for opt_method in optimization_methods:
        try:
            # Ensure parameters are numpy arrays for scipy compatibility
            x0 = np.array(log_initial_params, dtype=np.float64)

            result: Any = optimize.minimize(
                objective_func,
                x0,
                jac=gradient_func,
                method=opt_method,  # type: ignore
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

            if result.success:
                # First successful optimization - use it and stop trying others
                best_result = result
                best_likelihood = -result.fun
                break

            elif best_result is None:
                # Keep even failed results if we have nothing better
                best_result = result

        except Exception as e:
            # Continue trying other methods
            if best_result is None:
                # Store the error for the first method only
                first_error = str(e)
            continue

    # Process the best result found
    if best_result is not None:
        result = best_result

        if result.success:
            # Convert log-scale parameters back to original scale
            log_birth_rate, log_death_rate = result.x
            birth_rate = np.exp(log_birth_rate)
            death_rate = np.exp(log_death_rate)

            return {
                "success": True,
                "birth_rate": birth_rate,
                "death_rate": death_rate,
                "log_likelihood": best_likelihood,
                "optimization_result": result,
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
            "error": f"All optimization methods failed. First error: {first_error if 'first_error' in locals() else 'Unknown error'}",
        }


def fit_from_dataframe(
    df: pd.DataFrame,
    age_column: str,
    observation_time: float,
    is_ordered: bool = False,
    method: str = "L-BFGS-B",
    initial_params: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """
    Fit birth-death process parameters from a pandas DataFrame.

    Args:
        df: DataFrame containing manuscript data
        age_column: Name of the column with manuscript ages
        observation_time: Total observation time
        is_ordered: Whether the ages are ordered
        method: Optimization method for scipy.optimize.minimize
        initial_params: Initial parameter values (birth_rate, death_rate) in original scale
    Returns:
        Dictionary with fit results including parameters and statistics
    """

    if age_column not in df.columns:
        raise ValueError(f"Column '{age_column}' not found in DataFrame")

    ages = np.asarray(df[age_column], dtype=np.float64)

    return fit_birth_death_processes(
        list_of_ages=[ages],
        observation_time=observation_time,
        is_ordered=is_ordered,
        method=method,
        initial_params=initial_params,
    )
