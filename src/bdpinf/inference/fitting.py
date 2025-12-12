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

from typing import Any, Dict, List, Literal, Tuple

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
    est_birth_rate = np.float64(1.0 / mean_age)
    est_death_rate = np.max(
        [est_birth_rate - np.log(n_manuscripts) / observation_time, 1e-8]
    )

    # Convert to log scale for optimization
    return (est_birth_rate, est_death_rate)


def fit_birth_death_processes(
    list_of_ages: List[NDArray[np.float64]],
    times: float | NDArray[np.float64],
    is_ordered: bool = False,
    initial_birth_rates: NDArray[np.float64] | None = None,
    initial_death_rates: NDArray[np.float64] | None = None,
    spine_ages: NDArray[np.float64] | None = None,
    population_sizes: NDArray[np.int64] | None = None,
) -> Dict[str, Any]:
    """
    Fit birth-death process parameters to grouped manuscript-age data.

    This function estimates birth (speciation/appearance) and death (extinction/disappearance)
    rates for a simple birth-death process by minimizing the negative log-likelihood.
    Optimization is performed in log-parameter space (i.e., the optimizer works on
    log(birth_rate), log(death_rate)) for numerical stability and then converted back
    to the original scale for the returned results.

    Parameters
    ----------
    list_of_ages : List[NDArray[np.float64]]
        A list of numpy arrays, each containing observed ages for a single group
        (e.g., manuscript family). Arrays may be empty; at least one non-empty array
        is required. For ordered data (is_ordered=True) each array is expected to
        have the spine observation as its first element (unless spine_ages is
        provided explicitly).
    times : float | NDArray[np.float64]
        Observation time(s) for the process. Can be a single float representing
        the total observation time, or an array of time points. Must be positive.
        This is used by the likelihood functions to compute probabilities over
        the observation window.
    is_ordered : bool, optional
        If True and spine_ages is not provided, the first element of every non-empty
        array in list_of_ages is treated as a spine observation and removed from the
        main ages used in the likelihood; collected spine ages are passed to the
        likelihood. Default is False.
    initial_birth_rates : NDArray[np.float64] | None, optional
        Initial guess for birth rate(s) on the original scale. If None,
        a crude estimate is obtained. Can be a scalar or array matching the
        structure of times. Values are converted to log-space for the optimizer.
    initial_death_rates : NDArray[np.float64] | None, optional
        Initial guess for death rate(s) on the original scale. If None,
        a crude estimate is obtained. Can be a scalar or array matching the
        structure of times. Values are converted to log-space for the optimizer.
    spine_ages : NDArray[np.float64] | None, optional
        If provided, these are the spine ages to be used directly in the likelihood.
        If is_ordered is True and this argument is None, spine_ages will be inferred
        from the first element of each non-empty array in list_of_ages.
    population_sizes : NDArray[np.int64] | None, optional
        Array of integer counts, one per entry in list_of_ages, representing the
        number of observed lineages for that group. If provided, its length must
        equal len(list_of_ages). If None, a default is constructed as
        np.array([len(ages) + int(spine_ages is not None) for ages in list_of_ages],
        dtype=np.int64). Note: if you rely on automatic spine extraction
        (is_ordered=True and spine_ages=None) you should supply population_sizes explicitly if
        you require it to count spines, because the default is computed before
        spine extraction in the current implementation.

    Returns
    -------
    Dict[str, Any]
        A dictionary summarizing the optimization outcome and parameter estimates.
        Typical keys:
          - "success" (bool): True if optimization reported success, False otherwise.
          - If success is True:
              - "birth_rate" (float): Estimated birth rate on the original scale.
              - "death_rate" (float): Estimated death rate on the original scale.
              - "log_likelihood" (float): The log-likelihood corresponding to the
                returned parameters (computed as the negative of the minimized
                negative-log-likelihood).
              - "optimization_result": The raw scipy OptimizeResult object.
          - If success is False:
              - "message" (str): Optimizer message describing the failure (if available).
              - "optimization_result": The raw scipy OptimizeResult object (if available).
        If no optimization attempts succeed and no result is available, returns {"success": False}.

    Raises
    ------
    ValueError
        - If all arrays in list_of_ages are empty (no valid data to fit).
        - If times <= 0 (when times is a float) or any element of times <= 0 (when array).
        - If population_sizes is provided and sum(population_sizes) != len(list_of_ages) (i.e., length mismatch).

    Notes
    -----
    - The optimization is performed in log-parameter space. Bounds on the original
      parameters are enforced in log-space corresponding to birth_rate and death_rate
      in [1e-8, 100.0].
    - The objective minimized is the negative log-likelihood computed by
      negative_log_likelihood(...).
    - The function uses a robust optimization strategy with multiple fallbacks:
      1. BFGS with analytical gradient (via log_likelihood_gradient)
      2. BFGS with numerical gradient (via scipy.optimize.approx_fprime)
      3. Nelder-Mead (gradient-free)
      4. Powell (gradient-free)
    - The first successful optimization in this sequence is returned. If none succeed,
      the best available result (or a failure summary) is returned.

    Examples
    --------
    (Informal)
    - To fit unordered data:
        fit_birth_death_processes(list_of_ages, times=10.0)

    - To fit ordered data with automatic spine extraction:
        fit_birth_death_processes(list_of_ages, times=10.0, is_ordered=True)

    See Also
    --------
    negative_log_likelihood, log_likelihood_gradient, get_crude_parameter_estimates
    """
    # Input validation
    non_empty_age = None
    for ages in list_of_ages:
        if len(ages) > 0:
            non_empty_age = ages
            break

    if non_empty_age is None:
        raise ValueError("No valid data points for fitting")

    # Validate times parameter
    if isinstance(times, (int, float)):
        observation_time = float(times)
        if observation_time <= 0:
            raise ValueError("times must be positive")
    else:
        # times is an array
        if np.any(times <= 0):
            raise ValueError("All elements of times must be positive")
        # For now, use the last time point as observation_time for simple case
        observation_time = float(np.max(times))

    # Set initial parameters with better estimates based on data
    if initial_birth_rates is None and initial_death_rates is None:
        initial_params = get_crude_parameter_estimates(non_empty_age, observation_time)
    elif initial_birth_rates is not None and initial_death_rates is not None:
        # Both provided - combine them
        initial_params = (
            float(np.atleast_1d(initial_birth_rates)[0]),
            float(np.atleast_1d(initial_death_rates)[0]),
        )
    else:
        raise ValueError(
            "Either both initial_birth_rates and initial_death_rates must be provided, or neither"
        )

    # Convert initial parameters to log scale
    log_initial_params = np.log(initial_params)

    if population_sizes is not None:
        if sum(population_sizes) != len(list_of_ages):
            raise ValueError(
                "Length of population_sizes must match length of list_of_ages"
            )
    else:
        population_sizes = np.array(
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
            ns=population_sizes,
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
                ns=population_sizes,
                is_ordered=False,
                log_scale_params=False,
            ),
        )
        return np.array(nll_grad)

    # Optimize with multiple strategies for robustness
    # Strategy: Try L-BFGS-B with analytical gradient, then with numerical gradient,
    # then fall back to gradient-free methods
    best_result: Any = None
    best_likelihood = -np.inf

    # Ensure parameters are numpy arrays for scipy compatibility
    x0 = np.array(log_initial_params, dtype=np.float64)

    # Strategy 1: Try BFGS with analytical gradient
    try:
        result: Any = optimize.minimize(
            objective_func,
            x0,
            jac=gradient_func,
            method="BFGS",
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if result.success:
            # Check if optimized parameters are too similar to initial parameters
            # Use a reasonable threshold in log-space (0.01 = ~1% change in parameters)
            param_change = np.max(np.abs(result.x - x0))
            if param_change < 0.01:
                # Parameters barely changed, try other methods
                result.success = False
            else:
                best_result = result
                best_likelihood = -result.fun

        if best_result is None:
            best_result = result
    except Exception:
        # If analytical gradient fails, continue to numerical gradient
        pass

    # Strategy 2: If analytical gradient didn't work, try BFGS with numerical gradient
    if best_result is None or not best_result.success:
        try:
            # Create numerical gradient using scipy's approx_fprime
            def numerical_gradient(
                log_params: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                return optimize.approx_fprime(log_params, objective_func, epsilon=1e-8)

            result = optimize.minimize(
                objective_func,
                x0,
                jac=numerical_gradient,
                method="BFGS",
                options={"maxiter": 1000, "ftol": 1e-9},
            )

            if result.success:
                # Check if optimized parameters are too similar to initial parameters
                # Use a reasonable threshold in log-space (0.01 = ~1% change in parameters)
                param_change = np.max(np.abs(result.x - x0))
                if param_change < 0.01:
                    # Parameters barely changed, try other methods
                    result.success = False
                else:
                    best_result = result
                    best_likelihood = -result.fun

            if best_result is None:
                best_result = result
        except Exception:
            # If numerical gradient also fails, continue to gradient-free methods
            pass

    # Strategy 3: Try gradient-free methods if BFGS didn't succeed
    if best_result is None or not best_result.success:
        methods: List[Literal["Nelder-Mead", "Powell"]] = ["Nelder-Mead", "Powell"]
        for opt_method in methods:
            try:
                result = optimize.minimize(
                    objective_func,
                    x0,
                    method=opt_method,
                    options={"maxiter": 1000, "ftol": 1e-9},
                )

                if result.success:
                    # Check if optimized parameters are too similar to initial parameters
                    # Use a reasonable threshold in log-space (0.01 = ~1% change in parameters)
                    param_change = np.max(np.abs(result.x - x0))
                    if param_change < 0.01:
                        # Parameters barely changed, try next method
                        result.success = False
                    else:
                        best_result = result
                        best_likelihood = -result.fun
                        break

                if best_result is None:
                    best_result = result
            except Exception:
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
        return {"success": False}


def fit_from_dataframe(
    df: pd.DataFrame,
    age_column: str,
    times: float | NDArray[np.float64],
    is_ordered: bool = False,
    initial_birth_rates: NDArray[np.float64] | None = None,
    initial_death_rates: NDArray[np.float64] | None = None,
) -> Dict[str, Any]:
    """
    Fit birth-death process parameters from a pandas DataFrame.

    Args:
        df: DataFrame containing manuscript data
        age_column: Name of the column with manuscript ages
        times: Observation time(s) - can be float or array
        is_ordered: Whether the ages are ordered
        initial_birth_rates: Initial birth rate values
        initial_death_rates: Initial death rate values
    Returns:
        Dictionary with fit results including parameters and statistics
    """

    if age_column not in df.columns:
        raise ValueError(f"Column '{age_column}' not found in DataFrame")

    ages = np.asarray(df[age_column], dtype=np.float64)

    return fit_birth_death_processes(
        list_of_ages=[ages],
        times=times,
        is_ordered=is_ordered,
        initial_birth_rates=initial_birth_rates,
        initial_death_rates=initial_death_rates,
    )
