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

from typing import Any, Dict, List, Sequence, Tuple

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


def _find_non_empty_age(list_of_ages: List[torch.Tensor]) -> torch.Tensor:
    for ages in list_of_ages:
        if len(ages) > 0:
            return ages
    raise ValueError("No valid data points for fitting")


def _validate_times(times: torch.Tensor | float) -> torch.Tensor:
    times_tensor = ensure_1d_tensor(times)
    if torch.any(times_tensor <= 0):
        raise ValueError("All elements of times must be positive")
    return times_tensor


def _prepare_population_sizes(
    list_of_ages: List[torch.Tensor],
    population_sizes: torch.Tensor | None,
    spine_ages: torch.Tensor | None,
) -> torch.Tensor:
    if population_sizes is not None:
        population_sizes_tensor = ensure_1d_tensor(population_sizes).to(
            dtype=torch.int64
        )
        if population_sizes_tensor.numel() != len(list_of_ages):
            raise ValueError(
                "Length of population_sizes must match length of list_of_ages"
            )
        return population_sizes_tensor

    return torch.as_tensor(
        [len(ages) + int(spine_ages is not None) for ages in list_of_ages],
        dtype=torch.int64,
    )


def _extract_spine_and_ages(
    list_of_ages: List[torch.Tensor],
    is_ordered: bool,
    spine_ages: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    spine_ages_tensor: torch.Tensor | None
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
        return spine_ages_tensor, non_spine_ages

    if spine_ages is None:
        spine_ages_tensor = None
    else:
        spine_ages_tensor = ensure_1d_tensor(spine_ages)
    non_spine_ages = torch.cat(
        [ensure_1d_tensor(arr) for arr in list_of_ages if len(arr) > 0]
    )
    return spine_ages_tensor, non_spine_ages


def _build_initial_params(
    initial_birth_rates: torch.Tensor | None,
    initial_death_rates: torch.Tensor | None,
    non_empty_age: torch.Tensor,
    times_tensor: torch.Tensor,
) -> torch.Tensor:
    if initial_birth_rates is None or initial_death_rates is None:
        crude_birth, crude_death = get_crude_parameter_estimates(
            non_empty_age, times_tensor, initial_birth_rates, initial_death_rates
        )
        return torch.cat((crude_birth, crude_death))

    return torch.cat(
        (
            ensure_1d_tensor(initial_birth_rates),
            ensure_1d_tensor(initial_death_rates),
        )
    )


def _apply_param_perturbation(
    initial_params: torch.Tensor,
    perturb_initial_params_std: float,
    random_seed: int | None,
) -> torch.Tensor:
    if perturb_initial_params_std <= 0:
        return initial_params
    if random_seed is not None:
        torch.manual_seed(random_seed)
    noise = torch.normal(
        mean=0.0,
        std=perturb_initial_params_std * initial_params,
    )
    return torch.clamp(initial_params + noise, min=1e-6)


def _build_optimizer_strategies(
    optimizer_kwargs: Dict[str, Dict[str, Any]] | None,
) -> dict[str, tuple[str, type, int, dict[str, Any]]]:
    strategies_map: dict[str, tuple[str, type, int, dict[str, Any]]] = {
        "adam": ("Adam", torch.optim.Adam, 500, {"lr": 0.001}),
        "lbfgs": (
            "LBFGS",
            torch.optim.LBFGS,
            1,
            {"lr": 1.0, "max_iter": 200, "line_search_fn": "strong_wolfe"},
        ),
        "sgd": ("SGD", torch.optim.SGD, 1000, {"lr": 0.0001}),
    }

    if optimizer_kwargs is None:
        return strategies_map

    for key, custom_kwargs in optimizer_kwargs.items():
        key_lower = key.lower()
        if key_lower in strategies_map:
            opt_name, OptimClass, max_iters, default_kwargs = strategies_map[key_lower]
            merged_kwargs = {**default_kwargs, **custom_kwargs}
            strategies_map[key_lower] = (
                opt_name,
                OptimClass,
                max_iters,
                merged_kwargs,
            )
    return strategies_map


def _normalize_requested_optimizers(
    optimizers: Sequence[str] | str,
    strategies_map: dict[str, tuple[str, type, int, dict[str, Any]]],
) -> List[str]:
    requested_opts = [optimizers] if isinstance(optimizers, str) else list(optimizers)
    requested_keys = [opt.strip().lower() for opt in requested_opts]

    for key in requested_keys:
        if key not in strategies_map:
            raise ValueError(
                f"Unsupported optimizer '{key}'. Choose from {list(strategies_map.keys())}."
            )
    return requested_keys


def _run_optimizer_strategies(
    strategies_map: dict[str, tuple[str, type, int, dict[str, Any]]],
    requested_keys: List[str],
    log_params: torch.nn.Parameter,
    initial_params: torch.Tensor,
    n_epochs: int,
    non_spine_ages: torch.Tensor,
    times_tensor: torch.Tensor,
    spine_ages_tensor: torch.Tensor | None,
    population_sizes_tensor: torch.Tensor,
    convergence_tolerance: float,
    absolute_tolerance: float,
    patience: int,
    gradient_threshold: float,
) -> tuple[torch.nn.Parameter, List[float], str, int, bool]:
    converged = False
    iterations = 0
    optimizer_used = "Unknown"
    nll_history: List[float] = []

    for key in requested_keys:
        non_finite_objective = False
        opt_name, OptimClass, max_iters, opt_kwargs = strategies_map[key]
        try:
            with torch.no_grad():
                log_params.copy_(torch.log(initial_params.clone().detach()))
            optimizer = OptimClass([log_params], **opt_kwargs)
            iterations = 0
            optimizer_used = opt_name
            best_log_params: torch.Tensor | None = None

            if opt_name == "LBFGS":

                def closure() -> torch.Tensor:
                    nonlocal iterations
                    nonlocal non_finite_objective
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
                    if torch.isfinite(nll):
                        nll.backward()  # type: ignore[no-untyped-call]
                        nll_history.append(float(nll.item()))
                        iterations += 1
                    else:
                        non_finite_objective = True
                        return torch.tensor(
                            1e10, dtype=torch.float64, requires_grad=False
                        )
                    return nll

                optimizer.step(closure)
                if non_finite_objective:
                    continue
            else:
                best_nll = float("inf")
                iterations_without_improvement = 0

                for _ in range(max_iters):
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
                    if torch.isfinite(nll):
                        nll.backward()  # type: ignore[no-untyped-call]

                        grad_norm = 0.0
                        for param in [log_params]:
                            if param.grad is not None:
                                grad_norm = max(
                                    grad_norm, float(torch.norm(param.grad).item())
                                )

                        optimizer.step()
                        nll_value = float(nll.item())
                        nll_history.append(nll_value)
                        iterations += 1

                        if nll_value < best_nll:
                            improvement = best_nll - nll_value
                            relative_improvement = abs(improvement / (best_nll + 1e-10))

                            if (
                                improvement < absolute_tolerance
                                and relative_improvement < convergence_tolerance
                            ):
                                iterations_without_improvement += 1
                            else:
                                iterations_without_improvement = 0

                            best_nll = nll_value
                            best_log_params = log_params.detach().clone()
                        else:
                            iterations_without_improvement += 1

                        if grad_norm < gradient_threshold:
                            break

                        if iterations_without_improvement >= patience:
                            break
                    else:
                        non_finite_objective = True
                        break

                if non_finite_objective:
                    continue

            if best_log_params is not None:
                with torch.no_grad():
                    log_params.copy_(best_log_params)

            converged = True
            break
        except Exception:
            continue

    return log_params, nll_history, optimizer_used, iterations, converged


def _finalize_fit(
    log_params: torch.nn.Parameter,
    n_epochs: int,
    non_spine_ages: torch.Tensor,
    times_tensor: torch.Tensor,
    spine_ages_tensor: torch.Tensor | None,
    population_sizes_tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
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
            return None

        birth_rates = torch.exp(log_params[:n_epochs]).detach().cpu().numpy()
        death_rates = torch.exp(log_params[n_epochs:]).detach().cpu().numpy()
        final_log_likelihood = -float(final_nll.item())

    return birth_rates, death_rates, final_log_likelihood, float(final_nll.item())


def fit_birth_death_processes(
    list_of_ages: List[torch.Tensor],
    times: torch.Tensor | float,
    is_ordered: bool = False,
    initial_birth_rates: torch.Tensor | None = None,
    initial_death_rates: torch.Tensor | None = None,
    spine_ages: torch.Tensor | None = None,
    population_sizes: torch.Tensor | None = None,
    optimizers: Sequence[str] | str = ("Adam", "SGD", "LBFGS"),
    optimizer_kwargs: Dict[str, Dict[str, Any]] | None = None,
    perturb_initial_params_std: float = 0.0,
    random_seed: int | None = None,
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
    optimizers : Sequence[str] | str, optional
        Ordered list (or single name) of optimizers to try. Supported values:
        "Adam" (default first), "LBFGS", "SGD". Defaults to ("Adam", "LBFGS", "SGD").
    optimizer_kwargs : Dict[str, Dict[str, Any]] | None, optional
        Custom keyword arguments for optimizers. Keys should be optimizer names
        (lowercase: 'lbfgs', 'adam', 'sgd') and values are dicts of kwargs to pass
        to the optimizer constructor. These will override the default kwargs.
        Example: {"lbfgs": {"lr": 0.5, "max_iter": 200}, "adam": {"lr": 0.001}}
    perturb_initial_params_std : float, optional
        If > 0, adds normally distributed noise to initial parameters with standard
        deviation equal to this fraction of each parameter value. For example, 0.1
        adds noise with std = 10% of the parameter value. The perturbed parameters
        are clamped to be positive (min=1e-6). Default is 0.0 (no perturbation).
    random_seed : int | None, optional
        Random seed for reproducible parameter perturbation. Only used when
        perturb_initial_params_std > 0. If None, noise will not be reproducible.
        Default is None.

    Returns
    -------
    Dict[str, Any]
        A dictionary summarizing the optimization outcome and parameter estimates.
                Typical keys:
                    - "success" (bool)
                    - "birth_rates" (np.ndarray): Estimated birth rates (original scale)
                    - "death_rates" (np.ndarray): Estimated death rates (original scale)
                    - "log_likelihood" (float)
                    - "nll_history" (List[float]):
                        History of negative log likelihood values during optimization
                    - "optimization_result": minimal torch optimizer metadata
                Returns {"success": False} on failure.

    Raises
    ------
    ValueError
        - If all arrays in list_of_ages are empty (no valid data to fit).
        - If times <= 0 (when times is a float) or any element of times <= 0 (when array).
        - If population_sizes is provided and
            sum(population_sizes) != len(list_of_ages) (i.e., length mismatch).

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
    non_empty_age = _find_non_empty_age(list_of_ages)
    times_tensor = _validate_times(times)
    population_sizes_tensor = _prepare_population_sizes(
        list_of_ages, population_sizes, spine_ages
    )
    spine_ages_tensor, non_spine_ages = _extract_spine_and_ages(
        list_of_ages, is_ordered, spine_ages
    )

    initial_params = _build_initial_params(
        initial_birth_rates, initial_death_rates, non_empty_age, times_tensor
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

    initial_params = _apply_param_perturbation(
        initial_params, perturb_initial_params_std, random_seed
    )
    log_params = torch.nn.Parameter(torch.log(initial_params.clone().detach()))

    strategies_map = _build_optimizer_strategies(optimizer_kwargs)
    requested_keys = _normalize_requested_optimizers(optimizers, strategies_map)

    log_params, nll_history, optimizer_used, iterations, converged = (
        _run_optimizer_strategies(
            strategies_map,
            requested_keys,
            log_params,
            initial_params,
            n_epochs,
            non_spine_ages,
            times_tensor,
            spine_ages_tensor,
            population_sizes_tensor,
            convergence_tolerance=1e-7,
            absolute_tolerance=1e-6,
            patience=20,
            gradient_threshold=1e-5,
        )
    )

    if not converged:
        return {"success": False}

    finalized = _finalize_fit(
        log_params,
        n_epochs,
        non_spine_ages,
        times_tensor,
        spine_ages_tensor,
        population_sizes_tensor,
    )
    if finalized is None:
        return {"success": False}

    birth_rates, death_rates, final_log_likelihood, final_loss = finalized
    return {
        "success": converged,
        "birth_rates": birth_rates,
        "death_rates": death_rates,
        "log_likelihood": final_log_likelihood,
        "nll_history": nll_history,
        "optimization_result": {
            "optimizer": optimizer_used,
            "iterations": iterations,
            "final_loss": final_loss,
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
