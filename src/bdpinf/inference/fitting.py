"""Maximum likelihood estimation for birth-death process parameters using PyTorch."""

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
    *,
    birth_rates: torch.Tensor | float | None = None,
    death_rates: torch.Tensor | float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get crude initial parameter estimates from age data."""
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
    """Return first non-empty age array; raise ValueError if all empty." """
    for ages in list_of_ages:
        if len(ages) > 0:
            return ages
    raise ValueError("No valid data points for fitting")


def _validate_times(times: torch.Tensor | float) -> torch.Tensor:
    """Convert and validate times; raise ValueError if not positive."""
    times_tensor = ensure_1d_tensor(times)
    if torch.any(times_tensor <= 0):
        raise ValueError("All elements of times must be positive")
    return times_tensor


def _prepare_population_sizes(
    list_of_ages: List[torch.Tensor],
    population_sizes: torch.Tensor | None,
    spine_ages: torch.Tensor | None,
) -> torch.Tensor:
    """Return population_sizes or auto-compute from list_of_ages + spine_ages."""
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
    """Extract/auto-extract spine ages and return (spine_ages, non_spine_ages)."""
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
    """Build initial param vector [birth_rates, death_rates]; auto-estimate if needed."""
    if initial_birth_rates is None or initial_death_rates is None:
        crude_birth, crude_death = get_crude_parameter_estimates(
            non_empty_age,
            times_tensor,
            birth_rates=initial_birth_rates,
            death_rates=initial_death_rates,
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
    """Add Gaussian noise to params"""
    if perturb_initial_params_std <= 0:
        return initial_params
    if random_seed is not None:
        torch.manual_seed(random_seed)
    noise = torch.clamp(
        torch.normal(
            mean=1.0,
            std=perturb_initial_params_std,
            size=initial_params.shape,
            dtype=initial_params.dtype,
        ),
        min=0.1,
    )
    return initial_params * noise


def _build_optimizer_strategies(
    optimizer_kwargs: Dict[str, Dict[str, Any]] | None,
) -> dict[str, tuple[str, type, int, dict[str, Any]]]:
    """Build optimizer config dict"""
    strategies_map: dict[str, tuple[str, type, int, dict[str, Any]]] = {
        "adam": ("Adam", torch.optim.Adam, 1000, {"lr": 0.5}),
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
            opt_name, optimizer_cls, max_iters, default_opt_kwargs = strategies_map[
                key_lower
            ]
            merged_opt_kwargs = {**default_opt_kwargs, **custom_kwargs}
            strategies_map[key_lower] = (
                opt_name,
                optimizer_cls,
                max_iters,
                merged_opt_kwargs,
            )
    return strategies_map


def _normalize_requested_optimizers(
    optimizers: Sequence[str] | str,
    strategies_map: dict[str, tuple[str, type, int, dict[str, Any]]],
) -> List[str]:
    """Normalize optimizer names to lowercase list; validate against strategies_map."""
    requested_opts = [optimizers] if isinstance(optimizers, str) else list(optimizers)
    requested_keys = [opt.strip().lower() for opt in requested_opts]

    for key in requested_keys:
        if key not in strategies_map:
            raise ValueError(
                f"Unsupported optimizer '{key}'. "
                f"Choose from {list(strategies_map.keys())}."
            )
    return requested_keys


def _run_optimizer_strategies(
    strategies_map: dict[str, tuple[str, type, int, dict[str, Any]]],
    requested_keys: List[str],
    log_params: torch.nn.Parameter,
    initial_params: torch.Tensor,
    n_epochs: int,
    non_spine_ages: torch.Tensor,
    starting_population: int,
    times_tensor: torch.Tensor,
    spine_ages_tensor: torch.Tensor | None,
    population_sizes_tensor: torch.Tensor,
    convergence_tolerance: float,
    absolute_tolerance: float,
    patience: int,
    gradient_threshold: float = 1e-5,
) -> tuple[torch.nn.Parameter, List[float], str, int, bool, Exception | None]:
    """Run optimizers sequentially until convergence; return params, history, metadata.

    Args:
        strategies_map: Optimizer configs.
        requested_keys: Optimizer names to try sequentially.
        log_params: Parameter tensor (log-scale).
        initial_params: Initial params on original scale.
        n_epochs: Number of time epochs.
        non_spine_ages, times_tensor, spine_ages_tensor, population_sizes_tensor: Data.
        starting_population: Starting population size.
        convergence_tolerance, absolute_tolerance, patience, gradient_threshold: Stopping criteria.

    Returns:
        (optimized_log_params, nll_history, optimizer_name, iterations, converged, exception).
    """
    converged = False
    iterations: int = 0
    optimizer_used = "Unknown"
    nll_history: List[float] = []
    last_exception: Exception | None = None

    def _compute_nll() -> torch.Tensor:
        """Compute negative log likelihood for current log_params."""
        return negative_log_likelihood(
            birth_rates=log_params[:n_epochs],
            death_rates=log_params[n_epochs:],
            ages=non_spine_ages,
            times=times_tensor,
            spine_ages=spine_ages_tensor,
            ns=population_sizes_tensor,
            starting_population=starting_population,
            is_ordered=False,
            log_scale_params=True,
        )

    for key in requested_keys:
        non_finite_objective = False
        did_converge = False
        opt_name, OptimClass, max_iters, opt_kwargs = strategies_map[key]
        try:
            with torch.no_grad():
                log_params.copy_(torch.log(initial_params.clone().detach()))
            optimizer = OptimClass([log_params], **opt_kwargs)
            iterations = 0
            optimizer_used = opt_name
            best_log_params: torch.Tensor | None = None
            first_nll: float | None = None
            improved_once = False

            if opt_name == "LBFGS":
                last_nll_value: float | None = None

                def closure() -> torch.Tensor:
                    nonlocal non_finite_objective
                    nonlocal last_nll_value
                    nonlocal iterations
                    nonlocal nll_history
                    nonlocal first_nll
                    nonlocal improved_once
                    optimizer.zero_grad()
                    nll = _compute_nll()
                    iterations += 1
                    if not torch.isfinite(nll):
                        non_finite_objective = True
                        inf_with_grad = (log_params * 0).sum() + torch.tensor(
                            float("inf"),
                            device=log_params.device,
                            dtype=log_params.dtype,
                        )
                        return inf_with_grad
                    nll.backward()  # type: ignore[no-untyped-call]
                    nll_value = float(nll.item())
                    nll_history.append(nll_value)
                    last_nll_value = nll_value
                    if first_nll is None:
                        first_nll = nll_value
                    elif nll_value < first_nll - absolute_tolerance:
                        improved_once = True
                    return nll

                optimizer.step(closure)
                if non_finite_objective:
                    last_exception = RuntimeError(
                        f"Non-finite objective encountered "
                        f"during optimization with {opt_name}."
                    )
                    continue

                if last_nll_value is None:
                    last_exception = RuntimeError(
                        f"Optimizer {opt_name} did not produce a valid objective value."
                    )
                    continue

                # Check convergence: gradient norm below threshold
                grad_norm = 0.0
                if log_params.grad is not None:
                    grad_norm = float(torch.norm(log_params.grad).item())

                if not improved_once:
                    last_exception = RuntimeError(
                        f"Objective did not improve from initial value with {opt_name}."
                    )
                    continue

                best_log_params = log_params.detach().clone()
                did_converge = grad_norm < gradient_threshold
            else:
                best_nll = float("inf")
                iterations_without_improvement = 0
                iterations = 0

                for _ in range(max_iters):
                    optimizer.zero_grad()
                    nll = _compute_nll()
                    if torch.isfinite(nll):
                        nll.backward()  # type: ignore[no-untyped-call]

                        grad_norm = 0.0
                        if log_params.grad is not None:
                            grad_norm = float(torch.norm(log_params.grad).item())
                        optimizer.step()
                        iterations += 1
                        nll_value = float(nll.item())
                        nll_history.append(nll_value)
                        if first_nll is None:
                            first_nll = nll_value
                        elif nll_value < first_nll - absolute_tolerance:
                            improved_once = True

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

                        if grad_norm < gradient_threshold and improved_once:
                            did_converge = True
                            break

                        if iterations_without_improvement >= patience and improved_once:
                            did_converge = True
                            break
                    else:
                        non_finite_objective = True
                        break

                if non_finite_objective:
                    last_exception = RuntimeError(
                        f"Non-finite objective encountered during optimization with {opt_name}."
                    )
                    continue

                if not improved_once:
                    last_exception = RuntimeError(
                        f"Objective did not improve from initial value with {opt_name}."
                    )
                    continue

            if best_log_params is not None:
                with torch.no_grad():
                    log_params.copy_(best_log_params)

            if did_converge:
                converged = True
                break
        except (RuntimeError, ValueError) as exc:
            last_exception = exc
            continue

    return (
        log_params,
        nll_history,
        optimizer_used,
        iterations,
        converged,
        last_exception,
    )


def _finalize_fit(
    log_params: torch.nn.Parameter,
    n_epochs: int,
    non_spine_ages: torch.Tensor,
    times_tensor: torch.Tensor,
    *,
    spine_ages_tensor: torch.Tensor | None,
    population_sizes_tensor: torch.Tensor,
    starting_population: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Convert log params to original scale; compute final NLL; return rates + likelihoods."""
    with torch.no_grad():
        final_nll = negative_log_likelihood(
            birth_rates=log_params[:n_epochs],
            death_rates=log_params[n_epochs:],
            ages=non_spine_ages,
            times=times_tensor,
            spine_ages=spine_ages_tensor,
            ns=population_sizes_tensor,
            starting_population=starting_population,
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
    *,
    initial_birth_rates: torch.Tensor | None = None,
    initial_death_rates: torch.Tensor | None = None,
    is_ordered: bool = False,
    spine_ages: torch.Tensor | None = None,
    population_sizes: torch.Tensor | None = None,
    starting_population: int = 1,
    optimizers: Sequence[str] | str = ("Adam", "SGD", "LBFGS"),
    optimizer_kwargs: Dict[str, Dict[str, Any]] | None = None,
    perturb_initial_params_std: float = 0.0,
    random_seed: int | None = None,
) -> Dict[str, Any]:
    """Fit birth-death parameters to grouped age data via PyTorch.

    Optimizes log-parameters (LBFGS, Adam, SGD) until convergence.

    Parameters
    ----------
    list_of_ages : List[torch.Tensor]
        Age tensors per group; ≥1 non-empty array required.
    times : torch.Tensor | float
        Observation time(s), must be positive.
    is_ordered : bool
        If True, extract spine from first element per array.
    initial_birth_rates, initial_death_rates : torch.Tensor | None
        Initial guesses; auto-estimated if None.
    spine_ages : torch.Tensor | None
        Explicit spine ages (if not auto-extracted).
    population_sizes : torch.Tensor | None
        Counts per group; auto-computed if None.
    starting_population : int
        The number of starting individuals at time 0. Default is 1.
    optimizers : Sequence[str] | str
        Optimizer names ('LBFGS', 'Adam', 'SGD') to try sequentially.
    optimizer_kwargs : Dict[str, Dict[str, Any]] | None
        Custom per-optimizer hyperparams (e.g. {'adam': {'lr': 0.01}}).
    perturb_initial_params_std : float
        Noise std as fraction of initial params; 0 = no noise.
    random_seed : int | None
        Seed for reproducible perturbation.

    Returns
    -------
    dict
        {'success': bool, 'birth_rates': ndarray, 'death_rates': ndarray,
         'log_likelihood': float, 'nll_history': list, 'optimization_result': dict}
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

    (
        log_params,
        nll_history,
        optimizer_used,
        iterations,
        converged,
        last_exception,
    ) = _run_optimizer_strategies(
        strategies_map=strategies_map,
        requested_keys=requested_keys,
        log_params=log_params,
        initial_params=initial_params,
        n_epochs=n_epochs,
        non_spine_ages=non_spine_ages,
        times_tensor=times_tensor,
        spine_ages_tensor=spine_ages_tensor,
        population_sizes_tensor=population_sizes_tensor,
        starting_population=starting_population,
        convergence_tolerance=1e-7,
        absolute_tolerance=1e-6,
        patience=20,
        gradient_threshold=1e-5,
    )

    if not converged:
        error_message = str(last_exception) if last_exception is not None else None
        error_type = (
            type(last_exception).__name__ if last_exception is not None else None
        )
        return {"success": False, "error": error_message, "error_type": error_type}

    finalized = _finalize_fit(
        log_params,
        n_epochs,
        non_spine_ages,
        times_tensor,
        spine_ages_tensor=spine_ages_tensor,
        population_sizes_tensor=population_sizes_tensor,
        starting_population=starting_population,
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
    """Fit parameters from DataFrame age column."""

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
