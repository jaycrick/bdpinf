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

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from ..utils import ensure_1d_tensor
from .likelihood import negative_log_likelihood


def get_crude_parameter_estimates(
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    birth_rates: torch.Tensor | float | None = None,
    death_rates: torch.Tensor | float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    if birth_rates is not None and death_rates is not None:
        raise ValueError("Cannot provide both birth_rates and death_rates")

    times = ensure_1d_tensor(times)
    ages = ensure_1d_tensor(ages)

    observation_time = times[-1]
    n_manuscripts = len(ages)
    if n_manuscripts == 0:
        raise ValueError("Cannot estimate parameters from empty age data")

    mean_age = float(ages.mean())

    # Simple heuristic: death rate inversely related to max survival age
    # birth rate related to total observed manuscripts
    if birth_rates is None:
        est_birth_rates = torch.ones_like(times, dtype=torch.float64) / mean_age
    else:
        est_birth_rates = ensure_1d_tensor(birth_rates)

    if death_rates is None:
        est_death_rates = torch.maximum(
            est_birth_rates
            - torch.log(torch.tensor(float(n_manuscripts), dtype=torch.float64))
            / observation_time,
            torch.full_like(times, 1e-8, dtype=torch.float64),
        )
    else:
        est_death_rates = ensure_1d_tensor(death_rates)

    # Validate that all arrays have consistent sizes
    if len(times) != len(est_birth_rates) or len(times) != len(est_death_rates):
        raise ValueError(
            f"Size mismatch: times ({len(times)}), "
            f"est_birth_rates ({len(est_birth_rates)}), "
            f"est_death_rates ({len(est_death_rates)}) must all be equal"
        )

    return (est_birth_rates, est_death_rates)


def fit_birth_death_processes(
    list_of_ages: List[torch.Tensor],
    times: torch.Tensor | float,
    is_ordered: bool = False,
    initial_birth_rates: torch.Tensor | None = None,
    initial_death_rates: torch.Tensor | None = None,
    spine_ages: torch.Tensor | None = None,
    population_sizes: torch.Tensor | None = None,
) -> Dict[str, Any]:
    """
    Fit birth–death process parameters to grouped manuscript-age data using PyTorch.

    This version performs optimization fully in PyTorch (LBFGS) on log-parameters to
    maintain positivity. Gradients flow through the torch-native `negative_log_likelihood`
    (with `log_scale_params=True`).

    Parameters
    ----------
    list_of_ages : List[torch.Tensor]
        A list of torch tensors, each containing observed ages for a single group
        (e.g., manuscript family). Arrays may be empty; at least one non-empty array
        is required. For ordered data (is_ordered=True) each array is expected to
        have the spine observation as its first element (unless spine_ages is
        provided explicitly).
    times : torch.Tensor
        Observation time(s) for the process. Must be positive and strictly increasing
        for multi-epoch models. This is used by the likelihood functions to compute
        probabilities over the observation window.
    is_ordered : bool, optional
        If True and spine_ages is not provided, the first element of every non-empty
        array in list_of_ages is treated as a spine observation and removed from the
        main ages used in the likelihood; collected spine ages are passed to the
        likelihood. Default is False.
    initial_birth_rates : torch.Tensor | None, optional
        Initial guess for birth rate(s) on the original scale. If None,
        a crude estimate is obtained. Can be a scalar or array matching the
        structure of times. Values are converted to log-space for the optimizer.
    initial_death_rates : torch.Tensor | None, optional
        Initial guess for death rate(s) on the original scale. If None,
        a crude estimate is obtained. Can be a scalar or array matching the
        structure of times. Values are converted to log-space for the optimizer.
    spine_ages : torch.Tensor | None, optional
        If provided, these are the spine ages to be used directly in the likelihood.
        If is_ordered is True and this argument is None, spine_ages will be inferred
        from the first element of each non-empty array in list_of_ages.
    population_sizes : torch.Tensor | None, optional
        1D tensor of integer counts, one per entry in list_of_ages, representing the
        number of observed lineages for that group. If provided, its length must
        equal len(list_of_ages). If None, a default is constructed as
        torch.tensor([len(ages) + int(spine_ages is not None) for ages in list_of_ages],
        dtype=torch.int64). Note: if you rely on automatic spine extraction
        (is_ordered=True and spine_ages=None) you should supply population_sizes explicitly if
        you require it to count spines, because the default is computed before
        spine extraction in the current implementation.

    Returns
    -------
    Dict[str, Any]
        A dictionary summarizing the optimization outcome and parameter estimates.
                Typical keys:
                    - "success" (bool)
                    - "birth_rates" (np.ndarray): Estimated birth rates (original scale)
                    - "death_rates" (np.ndarray): Estimated death rates (original scale)
                    - "log_likelihood" (float)
                    - "optimization_result": minimal torch optimizer metadata
                Returns {"success": False} on failure.

    Raises
    ------
    ValueError
        - If all arrays in list_of_ages are empty (no valid data to fit).
        - If times <= 0 (when times is a float) or any element of times <= 0 (when array).
                - If population_sizes is provided and sum(population_sizes) != len(list_of_ages) (i.e., length mismatch).

        Notes
        -----
        - Optimization is LBFGS in torch on log-parameters for stability and positivity.
        - Objective is the torch-based negative_log_likelihood with log_scale_params=True.

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
    # Input validation and tensor preparation
    non_empty_age = None
    for ages in list_of_ages:
        if len(ages) > 0:
            non_empty_age = ages
            break

    if non_empty_age is None:
        raise ValueError("No valid data points for fitting")

    times_tensor = ensure_1d_tensor(times)
    if torch.any(times_tensor <= 0):
        raise ValueError("All elements of times must be positive")

    # Prepare population sizes
    if population_sizes is not None:
        population_sizes_tensor = ensure_1d_tensor(population_sizes).to(
            dtype=torch.int64
        )
        if population_sizes_tensor.numel() != len(list_of_ages):
            raise ValueError(
                "Length of population_sizes must match length of list_of_ages"
            )
    else:
        population_sizes_tensor = torch.as_tensor(
            [len(ages) + int(spine_ages is not None) for ages in list_of_ages],
            dtype=torch.int64,
        )

    # Handle spine extraction for ordered data
    spine_ages_tensor: torch.Tensor | None = None
    non_spine_ages: torch.Tensor = torch.tensor([], dtype=torch.float64)

    if is_ordered and (spine_ages is None):
        spine_list: List[float] = []
        remaining: List[torch.Tensor] = []
        for arr in list_of_ages:
            if len(arr) == 0:
                continue
            arr_tensor = ensure_1d_tensor(arr)
            spine_list.append(float(arr_tensor[0]))
            if len(arr_tensor) > 1:
                remaining.append(arr_tensor[1:])

        spine_ages_tensor = ensure_1d_tensor(
            torch.tensor(spine_list, dtype=torch.float64)
        )
        non_spine_ages = (
            torch.cat(remaining) if remaining else torch.tensor([], dtype=torch.float64)
        )
    else:
        spine_ages_tensor = (
            ensure_1d_tensor(spine_ages) if spine_ages is not None else None
        )
        non_spine_ages = torch.cat(
            [ensure_1d_tensor(arr) for arr in list_of_ages if len(arr) > 0]
        )

    # Initial parameters in torch
    if initial_birth_rates is None or initial_death_rates is None:
        crude_birth, crude_death = get_crude_parameter_estimates(
            non_empty_age, times_tensor, initial_birth_rates, initial_death_rates
        )
        initial_params = torch.cat((crude_birth, crude_death))
    else:
        initial_params = torch.cat(
            (
                ensure_1d_tensor(initial_birth_rates),
                ensure_1d_tensor(initial_death_rates),
            )
        )

    if torch.any(initial_params <= 0):
        raise ValueError("Initial parameters must be positive before log transform")

    n_epochs = len(times_tensor)
    expected_len = 2 * n_epochs
    if len(initial_params) != expected_len:
        raise ValueError(
            "Initial parameters size mismatch with times: "
            f"expected {expected_len}, got {len(initial_params)}"
        )

    log_params = torch.nn.Parameter(torch.log(initial_params.clone().detach()))
    optimizer = torch.optim.LBFGS(
        [log_params], lr=0.8, max_iter=400, tolerance_grad=1e-6, tolerance_change=1e-9
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        nll = negative_log_likelihood(
            birth_rates=log_params[:n_epochs],
            death_rates=log_params[n_epochs:],
            ages=non_spine_ages,
            times=times_tensor,
            spine_ages=spine_ages_tensor,
            ns=population_sizes_tensor,
            is_ordered=False,
            log_scale_params=True,
        )
        # Ensure scalar and backprop
        nll.backward()  # type: ignore[no-untyped-call]
        return nll

    try:
        optimizer.step(closure)  # type: ignore[no-untyped-call]
    except Exception:
        return {"success": False}

    with torch.no_grad():
        final_nll = negative_log_likelihood(
            birth_rates=log_params[:n_epochs],
            death_rates=log_params[n_epochs:],
            ages=non_spine_ages,
            times=times_tensor,
            spine_ages=spine_ages_tensor,
            ns=population_sizes_tensor,
            is_ordered=False,
            log_scale_params=True,
        )

        if not torch.isfinite(final_nll):
            return {"success": False}

        birth_rates = torch.exp(log_params[:n_epochs]).detach().cpu().numpy()
        death_rates = torch.exp(log_params[n_epochs:]).detach().cpu().numpy()
        final_log_likelihood = -float(final_nll.item())

    return {
        "success": True,
        "birth_rates": birth_rates,
        "death_rates": death_rates,
        "log_likelihood": final_log_likelihood,
        "optimization_result": {
            "optimizer": "torch.optim.LBFGS",
            "iterations": optimizer.state_dict().get("state", {}),
        },
    }


def fit_from_dataframe(
    df: pd.DataFrame,
    age_column: str,
    times: float | NDArray[np.float64] | torch.Tensor,
    is_ordered: bool = False,
    initial_birth_rates: NDArray[np.float64] | torch.Tensor | None = None,
    initial_death_rates: NDArray[np.float64] | torch.Tensor | None = None,
) -> Dict[str, Any]:
    """
    Fit birth-death process parameters from a pandas DataFrame (torch-friendly).

    Expects `age_column` to contain numeric ages; converts to torch via
    `ensure_1d_tensor` before fitting. `times` may be float, numpy array, or
    torch tensor; initial rates may be numpy arrays or torch tensors.
    """

    if age_column not in df.columns:
        raise ValueError(f"Column '{age_column}' not found in DataFrame")

    ages_tensor = ensure_1d_tensor(
        torch.as_tensor(df[age_column].to_numpy(dtype=np.float64), dtype=torch.float64)
    )
    times_tensor = ensure_1d_tensor(torch.as_tensor(times, dtype=torch.float64))

    init_birth_tensor = (
        ensure_1d_tensor(torch.as_tensor(initial_birth_rates, dtype=torch.float64))
        if initial_birth_rates is not None
        else None
    )
    init_death_tensor = (
        ensure_1d_tensor(torch.as_tensor(initial_death_rates, dtype=torch.float64))
        if initial_death_rates is not None
        else None
    )

    return fit_birth_death_processes(
        list_of_ages=[ages_tensor],
        times=times_tensor,
        is_ordered=is_ordered,
        initial_birth_rates=init_birth_tensor,
        initial_death_rates=init_death_tensor,
    )
