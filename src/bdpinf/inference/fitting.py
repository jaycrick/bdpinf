"""Maximum likelihood estimation for birth-death process parameters using PyTorch."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, Unpack, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import ensure_1d_tensor

from .likelihood import negative_log_likelihood

DEFAULT_GRADIENT_THRESHOLD = 1e-5
DEFAULT_ABSOLUTE_TOLERANCE = 1e-9
DEFAULT_PATIENCE = 5
DEFAULT_ITERATIONS = 1000


@dataclass(frozen=True)
class OptimizerStrategy:
    """Configuration for a single optimizer strategy."""

    name: str
    optimizer_class: type[torch.optim.Optimizer]
    max_iters: int
    optimizer_kwargs: dict[str, Any]


@dataclass
class OptimizerRunResult:
    """Result payload from running optimizer strategies."""

    log_params: torch.nn.Parameter
    nll_history: list[float]
    optimizer_used: str
    iterations: int
    converged: bool
    all_exceptions: list[Exception]
    failed_optimizer_histories: list[dict[str, Any]]


class FitKwargs(TypedDict, total=False):
    """TypedDict for fit_birth_death_processes kwargs."""

    initial_birth_rates: torch.Tensor | None
    initial_death_rates: torch.Tensor | None
    is_ordered: bool
    spine_ages: torch.Tensor | None
    population_sizes: torch.Tensor | None
    starting_population: int
    optimizers: Sequence[str] | str
    optimizer_kwargs: dict[str, dict[str, Any]] | None
    perturb_initial_params_std: float
    random_seed: int | None


@dataclass(frozen=True)
class ConvergenceConfig:
    """Configuration for convergence criteria evaluation."""

    gradient_threshold: float
    absolute_tolerance: float
    patience: int

    @classmethod
    def default(cls) -> "ConvergenceConfig":
        """Create configuration with default values."""
        return cls(
            gradient_threshold=DEFAULT_GRADIENT_THRESHOLD,
            absolute_tolerance=DEFAULT_ABSOLUTE_TOLERANCE,
            patience=DEFAULT_PATIENCE,
        )


def fit_birth_death_processes(
    list_of_ages: list[torch.Tensor],
    times: torch.Tensor | float,
    *,
    initial_birth_rates: torch.Tensor | None = None,
    initial_death_rates: torch.Tensor | None = None,
    is_ordered: bool = False,
    spine_ages: torch.Tensor | None = None,
    population_sizes: torch.Tensor | None = None,
    starting_population: int = 1,
    optimizers: Sequence[str] | str = ("LBFGS", "Adam", "SGD"),
    optimizer_kwargs: dict[str, dict[str, Any]] | None = None,
    perturb_initial_params_std: float = 0.0,
    random_seed: int | None = None,
    convergence_config: ConvergenceConfig | None = None,
) -> dict[str, Any]:
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
    convergence_config : ConvergenceConfig | None
        Custom convergence criteria; defaults used if None.

    Returns
    -------
    dict
        {'success': bool, 'birth_rates': ndarray, 'death_rates': ndarray,
         'log_likelihood': float, 'nll_history': list, 'optimization_result': dict}

    """
    non_empty_age = _find_non_empty_age(list_of_ages)
    (times_tensor,) = normalize_and_validate_inputs(times=times)
    population_sizes_tensor = _prepare_population_sizes(
        list_of_ages,
        population_sizes,
        spine_ages,
    )
    spine_ages_tensor, non_spine_ages = _extract_spine_and_ages(
        list_of_ages,
        is_ordered=is_ordered,
        spine_ages=spine_ages,
    )
    initial_params = _build_initial_params(
        initial_birth_rates,
        initial_death_rates,
        non_empty_age,
        times_tensor,
        starting_population=starting_population,
    )
    if torch.any(initial_params <= 0):
        msg = "Initial parameters must be positive before log transform"
        raise ValueError(msg)

    n_epochs = len(times_tensor)
    expected_len = 2 * n_epochs
    if len(initial_params) != expected_len:
        msg = (
            "Initial parameters size mismatch with times: "
            f"expected {expected_len}, got {len(initial_params)}"
        )
        raise ValueError(
            msg,
        )
    initial_params = _apply_param_perturbation(
        initial_params=initial_params,
        perturb_initial_params_std=perturb_initial_params_std,
        random_seed=random_seed,
    )
    log_params = torch.nn.Parameter(torch.log(initial_params.clone().detach()))
    convergence_config = convergence_config or ConvergenceConfig.default()
    strategies_map = _build_optimizer_strategies(
        optimizer_kwargs=optimizer_kwargs,
        convergence_config=convergence_config,
    )
    requested_keys = _normalize_requested_optimizers(optimizers, strategies_map)

    optimizer_run_result = _run_optimizer_strategies(
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
        convergence_config=convergence_config,
    )

    log_params = optimizer_run_result.log_params
    nll_history = optimizer_run_result.nll_history
    optimizer_used = optimizer_run_result.optimizer_used
    iterations = optimizer_run_result.iterations
    converged = optimizer_run_result.converged
    all_exceptions = optimizer_run_result.all_exceptions
    failed_optimizer_histories = optimizer_run_result.failed_optimizer_histories

    if not converged:
        last_exception = all_exceptions[-1] if all_exceptions else None
        error_message = str(last_exception) if last_exception is not None else None
        error_type = (
            type(last_exception).__name__ if last_exception is not None else None
        )
        errors = [
            {"error": str(exc), "error_type": type(exc).__name__}
            for exc in all_exceptions
        ]
        final_nll = nll_history[-1] if nll_history else None
        return {
            "success": False,
            "error": error_message,
            "error_type": error_type,
            "errors": errors,
            "nll_history": nll_history,
            "failed_optimizer_histories": failed_optimizer_histories,
            "optimization_result": {
                "optimizer": optimizer_used,
                "iterations": iterations,
                "converged": converged,
                "nll_history_length": len(nll_history),
                "final_nll": final_nll,
                "exceptions_count": len(all_exceptions),
            },
        }

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
    times: float | NDArray[np.float64] | torch.Tensor,
    *,
    age_column: str | None = None,
    **fit_kwargs: Unpack[FitKwargs],
) -> dict[str, Any]:
    """Fit parameters from DataFrame age column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing age data.
    times : float | NDArray | torch.Tensor
        Observation time(s).
    age_column : str | None
        Name of age column; auto-detected if None.
    **fit_kwargs : Any
        Additional keyword arguments forwarded to ``fit_birth_death_processes``.

    Returns
    -------
    dict
        Fitting results with success, rates, and optimization metadata.

    """
    if age_column is None or age_column not in df.columns:
        exact_matches = [col for col in df.columns if col in ("age", "ages")]
        prefix_matches = [col for col in df.columns if col.startswith("age_")]
        candidates = exact_matches or prefix_matches
        if not candidates:
            msg = (
                f"Column '{age_column if age_column is not None else 'of ages'}' "
                "not found, and no 'age', 'ages', or 'age_*' "
                "columns were found."
            )
            raise ValueError(msg)
        if len(candidates) > 1:
            msg = f"Multiple age columns found: {candidates}"
            raise ValueError(msg)
        age_column = candidates[0]

    ages_tensor = torch.as_tensor(
        df[age_column].to_numpy(dtype=np.float64, copy=True),
        dtype=torch.float64,
    )
    times_tensor = torch.as_tensor(times, dtype=torch.float64)
    times_tensor, ages_tensor = normalize_and_validate_inputs(
        times=times_tensor,
        ages=ages_tensor,
    )

    return fit_birth_death_processes(
        list_of_ages=[ages_tensor],
        times=times_tensor,
        **fit_kwargs,
    )


def _run_optimizer_strategies(
    strategies_map: dict[str, OptimizerStrategy],
    requested_keys: list[str],
    log_params: torch.nn.Parameter,
    initial_params: torch.Tensor,
    n_epochs: int,
    non_spine_ages: torch.Tensor,
    starting_population: int,
    times_tensor: torch.Tensor,
    spine_ages_tensor: torch.Tensor | None,
    population_sizes_tensor: torch.Tensor,
    convergence_config: ConvergenceConfig,
) -> OptimizerRunResult:
    """Run optimizers sequentially until convergence; return params, history, metadata.

    Args:
        strategies_map: Optimizer configs.
        requested_keys: Optimizer names to try sequentially.
        log_params: Parameter tensor (log-scale).
        initial_params: Initial params on original scale.
        n_epochs: Number of time epochs.
        non_spine_ages: Non-spine age data.
        starting_population: Starting population size.
        times_tensor: Time points.
        spine_ages_tensor: Spine ages if provided.
        population_sizes_tensor: Population sizes per group.
        convergence_config: Configuration for convergence criteria.

    Returns:
        OptimizerRunResult with optimized params, history, and metadata.

    """
    converged = False
    iterations: int = 0
    optimizer_used = "Unknown"
    nll_history: list[float] = []
    all_exceptions: list[Exception] = []
    failed_optimizer_histories: list[dict[str, Any]] = []

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
        strategy = strategies_map[key]
        optimizer_used = strategy.name

        success, best_log_params, opt_iterations = _run_single_optimizer(
            strategy=strategy,
            log_params=log_params,
            initial_params=initial_params,
            nll_history=nll_history,
            all_exceptions=all_exceptions,
            failed_optimizer_histories=failed_optimizer_histories,
            convergence_config=convergence_config,
            compute_nll_fn=_compute_nll,
        )

        iterations = opt_iterations

        if best_log_params is not None:
            with torch.no_grad():
                log_params.copy_(best_log_params)

        if success:
            converged = True
            break

    return OptimizerRunResult(
        log_params=log_params,
        nll_history=nll_history,
        optimizer_used=optimizer_used,
        iterations=iterations,
        converged=converged,
        all_exceptions=all_exceptions,
        failed_optimizer_histories=failed_optimizer_histories,
    )


def _run_single_optimizer(
    strategy: OptimizerStrategy,
    log_params: torch.nn.Parameter,
    initial_params: torch.Tensor,
    nll_history: list[float],
    all_exceptions: list[Exception],
    failed_optimizer_histories: list[dict[str, Any]],
    convergence_config: ConvergenceConfig,
    compute_nll_fn: Callable[[], torch.Tensor],
) -> tuple[bool, torch.Tensor | None, int]:
    """Run a single optimizer strategy and handle results.

    Returns:
        (success, best_log_params, iterations)

    """
    history_start_idx = len(nll_history)
    iterations = 0

    try:
        with torch.no_grad():
            log_params.copy_(torch.log(initial_params.clone().detach()))
        optimizer = strategy.optimizer_class([log_params], **strategy.optimizer_kwargs)

        if strategy.name == "LBFGS":
            lbfgs_optimizer = cast("torch.optim.LBFGS", optimizer)
            did_converge, exc, best_log_params = _run_lbfgs_optimizer(
                strategy=strategy,
                log_params=log_params,
                optimizer=lbfgs_optimizer,
                max_iters=strategy.max_iters,
                nll_history=nll_history,
                convergence_config=convergence_config,
                compute_nll_fn=compute_nll_fn,
            )
        else:
            did_converge, exc, best_log_params, iterations = _run_first_order_optimizer(
                strategy=strategy,
                log_params=log_params,
                optimizer=optimizer,
                max_iters=strategy.max_iters,
                nll_history=nll_history,
                convergence_config=convergence_config,
                compute_nll_fn=compute_nll_fn,
            )
    except (RuntimeError, ValueError) as exc:
        all_exceptions.append(exc)
        return False, None, iterations
    else:
        if exc is not None:
            all_exceptions.append(exc)
            optimizer_history = nll_history[history_start_idx:]
            failed_optimizer_histories.append(
                {
                    "optimizer": strategy.name,
                    "nll_history": optimizer_history,
                },
            )
            return False, None, iterations

        return did_converge, best_log_params, iterations


def _run_lbfgs_optimizer(
    strategy: OptimizerStrategy,
    log_params: torch.nn.Parameter,
    optimizer: torch.optim.LBFGS,
    max_iters: int,
    nll_history: list[float],
    convergence_config: ConvergenceConfig,
    compute_nll_fn: Callable[[], torch.Tensor],
) -> tuple[bool, Exception | None, torch.Tensor | None]:
    """Run LBFGS optimization loop.

    Returns:
        (did_converge, current_exception, best_log_params)

    """
    last_nll_value: float | None = None
    last_grad_norm: float | None = None
    best_nll = float("inf")
    best_log_params: torch.Tensor | None = None
    first_nll: float | None = None
    improved_once = False
    current_exception: Exception | None = None

    for _ in range(max_iters):

        def closure() -> torch.Tensor:
            nonlocal last_nll_value
            nonlocal nll_history
            nonlocal best_nll
            nonlocal best_log_params
            optimizer.zero_grad()
            nll = compute_nll_fn()
            nll.backward()  # type: ignore[no-untyped-call]
            last_nll_value = float(nll.item())
            if last_nll_value < best_nll:
                best_nll = last_nll_value
                best_log_params = log_params.detach().clone()
            nll_history.append(last_nll_value)
            return nll

        optimizer.step(closure)  # type: ignore[no-untyped-call]

        if last_nll_value is None:
            current_exception = RuntimeError(
                f"Optimizer {strategy.name} did not produce a valid objective value.",
            )
            break

        if first_nll is None:
            first_nll = last_nll_value
        elif last_nll_value < first_nll - convergence_config.absolute_tolerance:
            improved_once = True

        if improved_once and _check_convergence_criteria(
            optimizer_name=strategy.name,
            log_params=log_params,
            nll_history=nll_history,
            convergence_config=convergence_config,
        ):
            return True, None, best_log_params

        last_grad_norm = (
            float(torch.norm(log_params.grad).item())
            if log_params.grad is not None
            else float("inf")
        )

    if not improved_once:
        current_exception = RuntimeError(
            f"Objective did not improve from initial value with {strategy.name}.",
        )

    if current_exception is not None:
        return False, current_exception, best_log_params

    first_nll_value = first_nll if first_nll is not None else float("nan")
    final_nll_value = last_nll_value if last_nll_value is not None else float("nan")
    improvement = first_nll_value - final_nll_value
    current_exception = RuntimeError(
        "Convergence not reached with "
        f"{strategy.name}: max_iters={max_iters}, "
        f"history_len={len(nll_history)}, "
        f"first_nll={first_nll_value:.6f}, "
        f"final_nll={final_nll_value:.6f}, "
        f"improvement={improvement:.6f}, "
        f"last_grad_norm={last_grad_norm}, "
        f"gradient_threshold={convergence_config.gradient_threshold}",
    )
    return False, current_exception, best_log_params


def _run_first_order_optimizer(
    strategy: OptimizerStrategy,
    log_params: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    max_iters: int,
    nll_history: list[float],
    convergence_config: ConvergenceConfig,
    compute_nll_fn: Callable[[], torch.Tensor],
) -> tuple[bool, Exception | None, torch.Tensor | None, int]:
    """Run first-order optimizer (Adam/SGD) loop.

    Returns:
        (did_converge, current_exception, best_log_params, iterations)

    """
    best_nll = float("inf")
    iterations_without_improvement = 0
    last_grad_norm = None
    best_log_params: torch.Tensor | None = None
    first_nll: float | None = None
    improved_once = False
    current_exception: Exception | None = None
    absolute_tolerance = convergence_config.absolute_tolerance

    for iteration in range(max_iters):
        nll = compute_nll_fn()
        optimizer.zero_grad()

        if not torch.isfinite(nll):
            current_exception = RuntimeError(
                f"Non-finite objective encountered during optimization with "
                f"{strategy.name}.",
            )
            return False, current_exception, None, iteration

        nll.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        nll_value = float(nll.item())
        nll_history.append(nll_value)

        if first_nll is None:
            first_nll = nll_value
        elif nll_value < first_nll - absolute_tolerance:
            improved_once = True

        if nll_value >= best_nll:
            iterations_without_improvement += 1
        else:
            improvement = best_nll - nll_value
            if improvement < absolute_tolerance:
                iterations_without_improvement += 1
            else:
                iterations_without_improvement = 0

            best_nll = nll_value
            best_log_params = log_params.detach().clone()

        if improved_once and _check_convergence_criteria(
            optimizer_name=strategy.name,
            log_params=log_params,
            nll_history=nll_history,
            convergence_config=convergence_config,
            iterations_without_improvement=iterations_without_improvement,
        ):
            return True, None, best_log_params, iteration

        last_grad_norm = (
            float(torch.norm(log_params.grad).item())
            if log_params.grad is not None
            else None
        )

    if not improved_once:
        current_exception = RuntimeError(
            f"Objective did not improve from initial value with {strategy.name}.",
        )
        return False, current_exception, None, iteration

    first_nll_value = first_nll if first_nll is not None else float("nan")
    final_nll_value = nll_history[-1] if nll_history else float("nan")
    improvement = first_nll_value - final_nll_value
    current_exception = RuntimeError(
        "Convergence not reached with "
        f"{strategy.name}: max_iters={max_iters}, "
        f"iterations={iteration}, "
        f"history_len={len(nll_history)}, "
        f"first_nll={first_nll_value:.6f}, "
        f"best_nll={best_nll:.6f}, "
        f"final_nll={final_nll_value:.6f}, "
        f"improvement={improvement:.6f}, "
        f"last_grad_norm={last_grad_norm}, "
        f"iterations_without_improvement={iterations_without_improvement}, "
        f"patience={convergence_config.patience}, "
        f"gradient_threshold={convergence_config.gradient_threshold}",
    )
    return False, current_exception, best_log_params, iteration


def _get_crude_parameter_estimates(
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    birth_rates: torch.Tensor | float | None = None,
    death_rates: torch.Tensor | float | None = None,
    starting_population: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get crude initial parameter estimates from age data."""
    if birth_rates is not None and death_rates is not None:
        msg = "Cannot provide both birth_rates and death_rates"
        raise ValueError(msg)

    times, ages = normalize_and_validate_inputs(times=times, ages=ages)

    observation_time = times[-1]
    n_manuscripts = len(ages)
    if n_manuscripts == 0:
        msg = "Cannot estimate parameters from empty age data"
        raise ValueError(msg)

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
            - torch.log(
                torch.tensor(
                    float(n_manuscripts / starting_population),
                    dtype=torch.float64,
                ),
            )
            / observation_time,
            torch.full_like(times, 1e-8, dtype=torch.float64),
        )
    else:
        est_death_rates = ensure_1d_tensor(death_rates)

    # Validate that all arrays have consistent sizes
    if len(times) != len(est_birth_rates) or len(times) != len(est_death_rates):
        msg = (
            f"Size mismatch: times ({len(times)}), "
            f"est_birth_rates ({len(est_birth_rates)}), "
            f"est_death_rates ({len(est_death_rates)}) must all be equal"
        )
        raise ValueError(
            msg,
        )

    return (est_birth_rates, est_death_rates)


def _find_non_empty_age(list_of_ages: list[torch.Tensor]) -> torch.Tensor:
    """Return first non-empty age array; raise ValueError if all empty."""
    for ages in list_of_ages:
        if len(ages) > 0:
            return ages
    msg = "No valid data points for fitting"
    raise ValueError(msg)


def _prepare_population_sizes(
    list_of_ages: list[torch.Tensor],
    population_sizes: torch.Tensor | None,
    spine_ages: torch.Tensor | None,
) -> torch.Tensor:
    """Return population_sizes or auto-compute from list_of_ages + spine_ages."""
    if population_sizes is not None:
        population_sizes_tensor = ensure_1d_tensor(population_sizes).to(
            dtype=torch.int64,
        )
        if population_sizes_tensor.numel() != len(list_of_ages):
            msg = "Length of population_sizes must match length of list_of_ages"
            raise ValueError(
                msg,
            )
        return population_sizes_tensor

    return torch.as_tensor(
        [len(ages) + int(spine_ages is not None) for ages in list_of_ages],
        dtype=torch.int64,
    )


def _extract_spine_and_ages(
    list_of_ages: list[torch.Tensor],
    *,
    is_ordered: bool,
    spine_ages: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Extract/auto-extract spine ages and return (spine_ages, non_spine_ages)."""
    spine_ages_tensor: torch.Tensor | None
    if is_ordered and (spine_ages is None):
        spine_list: list[float] = []
        remaining: list[torch.Tensor] = []
        for arr in list_of_ages:
            if len(arr) == 0:
                continue
            arr_tensor = ensure_1d_tensor(arr)
            spine_list.append(float(arr_tensor[0]))
            if len(arr_tensor) > 1:
                remaining.append(arr_tensor[1:])

        spine_ages_tensor = ensure_1d_tensor(
            torch.tensor(spine_list, dtype=torch.float64),
        )
        non_spine_ages = (
            torch.cat(remaining) if remaining else torch.tensor([], dtype=torch.float64)
        )
        return spine_ages_tensor, non_spine_ages

    spine_ages_tensor = None if spine_ages is None else ensure_1d_tensor(spine_ages)
    non_spine_ages = torch.cat(
        [ensure_1d_tensor(arr) for arr in list_of_ages if len(arr) > 0],
    )
    return spine_ages_tensor, non_spine_ages


def _build_initial_params(
    initial_birth_rates: torch.Tensor | None,
    initial_death_rates: torch.Tensor | None,
    non_empty_age: torch.Tensor,
    times_tensor: torch.Tensor,
    starting_population: int = 1,
) -> torch.Tensor:
    """Build initial param vector; auto-estimate if needed."""
    if initial_birth_rates is None or initial_death_rates is None:
        crude_birth, crude_death = _get_crude_parameter_estimates(
            non_empty_age,
            times_tensor,
            birth_rates=initial_birth_rates,
            death_rates=initial_death_rates,
            starting_population=starting_population,
        )
        return torch.cat((crude_birth, crude_death))

    return torch.cat(
        (
            ensure_1d_tensor(initial_birth_rates),
            ensure_1d_tensor(initial_death_rates),
        ),
    )


def _apply_param_perturbation(
    initial_params: torch.Tensor,
    perturb_initial_params_std: float,
    random_seed: int | None,
) -> torch.Tensor:
    """Add Gaussian noise to params."""
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
    optimizer_kwargs: dict[str, dict[str, Any]] | None,
    convergence_config: ConvergenceConfig,
) -> dict[str, OptimizerStrategy]:
    """Build optimizer config dict."""
    strategies_map: dict[str, OptimizerStrategy] = {
        "adam": OptimizerStrategy(
            "Adam",
            torch.optim.Adam,
            DEFAULT_ITERATIONS,
            {"lr": 0.001},
        ),
        "lbfgs": OptimizerStrategy(
            "LBFGS",
            torch.optim.LBFGS,
            DEFAULT_ITERATIONS // 20,
            {
                "lr": 1.0,
                "tolerance_grad": convergence_config.gradient_threshold,
                "tolerance_change": convergence_config.absolute_tolerance,
            },
        ),
        "sgd": OptimizerStrategy(
            "SGD",
            torch.optim.SGD,
            DEFAULT_ITERATIONS,
            {"lr": 0.001},
        ),
    }

    if optimizer_kwargs is None:
        return strategies_map

    for key, custom_kwargs in optimizer_kwargs.items():
        key_lower = key.lower()
        if key_lower in strategies_map:
            strategy = strategies_map[key_lower]
            merged_opt_kwargs = {**strategy.optimizer_kwargs, **custom_kwargs}
            strategies_map[key_lower] = OptimizerStrategy(
                name=strategy.name,
                optimizer_class=strategy.optimizer_class,
                max_iters=strategy.max_iters,
                optimizer_kwargs=merged_opt_kwargs,
            )
    return strategies_map


def _normalize_requested_optimizers(
    optimizers: Sequence[str] | str,
    strategies_map: dict[str, OptimizerStrategy],
) -> list[str]:
    """Normalize optimizer names to lowercase list; validate against strategies_map."""
    requested_opts = [optimizers] if isinstance(optimizers, str) else list(optimizers)
    requested_keys = [opt.strip().lower() for opt in requested_opts]

    for key in requested_keys:
        if key not in strategies_map:
            msg = (
                f"Unsupported optimizer '{key}'. "
                f"Choose from {list(strategies_map.keys())}."
            )
            raise ValueError(
                msg,
            )
    return requested_keys


def _check_convergence_criteria(
    optimizer_name: str,
    *,
    log_params: torch.nn.Parameter | None = None,
    nll_history: list[float] | None = None,
    convergence_config: ConvergenceConfig | None = None,
    iterations_without_improvement: int | None = None,
) -> bool:
    """Check if optimizer meets convergence criteria.

    Evaluates convergence based on optimizer type and stopping criteria.

    Args:
        optimizer_name: Name of optimizer type ("LBFGS", "Adam", "SGD").
        log_params: Parameter tensor to compute gradient norm from.
        nll_history: History of negative log likelihood values.
        convergence_config: Convergence configuration with thresholds.
        iterations_without_improvement: For non-LBFGS, iterations without improvement.

    Returns:
        True if convergence criteria are met, False otherwise.

    """
    # Compute gradient norm if log_params provided
    grad_norm = float("inf")
    if log_params is not None and log_params.grad is not None:
        grad_norm = float(torch.norm(log_params.grad).item())

    if convergence_config is None:
        convergence_config = ConvergenceConfig.default()

    if grad_norm <= convergence_config.gradient_threshold:
        return True

    min_nll_history_len = 2

    if optimizer_name.lower() == "lbfgs":
        if nll_history is None or len(nll_history) < min_nll_history_len:
            return False
        return nll_history[-1] - nll_history[-2] < convergence_config.absolute_tolerance

    if iterations_without_improvement is None:
        return False

    return iterations_without_improvement >= convergence_config.patience


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
    """Convert log params to original scale; return rates + final likelihoods."""
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
