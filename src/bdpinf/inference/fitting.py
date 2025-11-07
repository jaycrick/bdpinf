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
    return (est_birth_rate, est_death_rate)


def fit_birth_death_processes(
    list_of_ages: List[NDArray[np.float64]],
    observation_time: float,
    is_ordered: bool = False,
    method: str = "L-BFGS-B",
    initial_params: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Fit birth-death process parameters to manuscript data.

    Optimization is performed over log-scale parameters for better numerical stability.

    Args:
        ages: Array of manuscript ages
        observation_time: Total observation time
        method: Optimization method for scipy.optimize.minimize
        initial_params: Initial parameter values (birth_rate, death_rate) in original scale

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

    # Parameter bounds in log space: log(birth_rate), log(death_rate)
    # birth_rate: 1e-8 to 100.0 -> log bounds: log(1e-8) to log(100.0)
    # death_rate: 1e-8 to 100.0 -> log bounds: log(1e-8) to log(100.0)
    bounds: List[Tuple[Optional[float], Optional[float]]] = [
        (np.log(1e-8), np.log(100.0)),  # log(birth_rate) bounds
        (np.log(1e-8), np.log(100.0)),  # log(death_rate) bounds
    ]

    # Create objective function wrapper
    def objective_func(params: NDArray[np.float64]) -> np.float64:
        nll = np.float64(0)
        for ages in list_of_ages:
            nll += negative_log_likelihood(
                params=params,
                ages=ages,
                observation_time=observation_time,
                is_ordered=is_ordered,
                log_scale_params=False,
            )
        return nll

    def gradient_func(params: NDArray[np.float64]) -> NDArray[np.float64]:
        nll_grad = np.zeros(2, dtype=np.float64)
        for ages in list_of_ages:
            nll_grad -= log_likelihood_gradient(
                params=params,
                ages=ages,
                observation_time=observation_time,
                is_ordered=is_ordered,
                log_scale_params=False,
            )
        return nll_grad

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
                jac=gradient_func,
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
            birth_rate, death_rate = result.x

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
            "error": "All optimization methods failed",
        }


def fit_from_dataframe(
    df: pd.DataFrame,
    age_column: str,
    observation_time: float,
    is_ordered: bool = False,
    method: str = "L-BFGS-B",
    initial_params: Optional[Tuple[float, float]] = None,
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
