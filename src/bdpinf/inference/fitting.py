"""Maximum likelihood estimation for birth-death process parameters using PyTorch."""

import dataclasses
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, Unpack, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import ensure_1d_tensor

from ._fit_types import (
    DEFAULT_ITERATIONS,
    MIN_EQUALITY_GROUP_SIZE,
    ConvergenceConfig,
    OptimizerAttemptResult,
    OptimizerRunResult,
    OptimizerStrategy,
)
from ._optimizer_runners import (
    _run_first_order_optimizer,
    _run_lbfgs_optimizer,
)
from .likelihood import negative_log_likelihood
from .uncertainty import estimate_mle_covariance

logger = logging.getLogger(__name__)


class FitKwargs(TypedDict, total=False):
    """TypedDict for fit_birth_death_processes kwargs."""

    initial_birth_rates: torch.Tensor | None
    initial_death_rates: torch.Tensor | None
    is_ordered: bool
    spine_ages: torch.Tensor | None
    population_sizes: torch.Tensor | None
    initial_population: int
    optimizers: Sequence[str] | str
    unbias: bool
    optimizer_kwargs: dict[str, dict[str, Any]] | None
    perturb_initial_params_std: float
    random_seed: int | None
    param_constraints: "ParameterConstraints | None"
    estimate_uncertainty: bool


@dataclass(frozen=True)
class ParameterConstraints:
    """Constraints for fitting: fixed parameters and equality groups.

    Attributes
    ----------
    fixed_params : dict[int, float]
        Mapping from parameter index to fixed value.
        Index i where i < n_epochs refers to birth rates,
        i >= n_epochs refers to death rates at index (i - n_epochs).
    equality_groups : list[list[int]]
        List of groups where each group contains parameter indices
        that must be equal during fitting. Parameters in the same
        group will share a single fitted value.

    Examples
    --------
    Make first two birth rates equal (indices 0, 1) and fix death rate
    at epoch 0 (index n_epochs) to 0.1:

        >>> constraints = ParameterConstraints(
        ...     fixed_params={5: 0.1},  # assuming n_epochs >= 2
        ...     equality_groups=[[0, 1]]
        ... )

    """

    fixed_params: dict[int, float] = dataclasses.field(default_factory=dict)
    equality_groups: list[list[int]] = dataclasses.field(default_factory=list)

    def validate(self, n_total_params: int) -> None:
        """Validate constraints are consistent with parameter count.

        Parameters
        ----------
        n_total_params : int
            Total number of parameters (2 * n_epochs).

        Raises
        ------
        ValueError
            If indices are out of bounds or constraints are invalid.

        """
        all_constrained_indices = set(self.fixed_params.keys())
        for idx, value in self.fixed_params.items():
            if idx not in range(n_total_params):
                msg = f"Parameter index {idx} out of range [0, {n_total_params})"
                raise ValueError(msg)
            if not np.isfinite(value) or value < 0:
                msg = f"Fixed parameter at index {idx} must be non-negative and finite"
                raise ValueError(msg)

        for group in self.equality_groups:
            if len(group) < MIN_EQUALITY_GROUP_SIZE:
                msg = "Equality groups must have at least 2 elements"
                raise ValueError(msg)
            for idx in group:
                if idx not in range(n_total_params):
                    msg = f"Parameter index {idx} out of range [0, {n_total_params})"
                    raise ValueError(msg)
                if idx in all_constrained_indices:
                    msg = (
                        f"Index {idx} appears in both fixed_params and equality_groups"
                    )
                    raise ValueError(msg)
            all_constrained_indices.update(group)

    def get_free_param_indices(self, n_total_params: int) -> list[int]:
        """Get indices of parameters that will be optimized.

        Parameters
        ----------
        n_total_params : int
            Total number of parameters (2 * n_epochs).

        Returns
        -------
        list[int]
            Indices of free parameters in the original parameter space.

        """
        self.validate(n_total_params)
        fixed_set = set(self.fixed_params.keys())
        equality_set = set()
        for group in self.equality_groups:
            equality_set.update(group)

        free_indices = [
            i
            for i in range(n_total_params)
            if i not in fixed_set and i not in equality_set
        ]
        # Add one representative from each equality group
        free_indices.extend([min(group) for group in self.equality_groups])

        return sorted(free_indices)


def _apply_constraints_to_full_params(
    free_params: torch.Tensor,
    constraints: ParameterConstraints,
    n_total_params: int,
) -> torch.Tensor:
    """Expand free parameters to full parameter space applying all constraints.

    Parameters
    ----------
    free_params : torch.Tensor
        Parameters corresponding to free indices only.
    constraints : ParameterConstraints
        The constraint specification.
    n_total_params : int
        Total number of parameters (2 * n_epochs).

    Returns
    -------
    torch.Tensor
        Full parameter vector with constraints applied.

    """
    free_indices = constraints.get_free_param_indices(n_total_params)
    free_position_by_index = {idx: pos for pos, idx in enumerate(free_indices)}

    representative_by_index: dict[int, int] = {}
    for group in constraints.equality_groups:
        representative_idx = min(group)
        for idx in group:
            representative_by_index[idx] = representative_idx

    parts: list[torch.Tensor] = []
    for idx in range(n_total_params):
        if idx in constraints.fixed_params:
            parts.append(
                torch.tensor(
                    constraints.fixed_params[idx],
                    dtype=free_params.dtype,
                    device=free_params.device,
                ),
            )
            continue

        source_idx = representative_by_index.get(idx, idx)
        parts.append(free_params[free_position_by_index[source_idx]])

    return torch.stack(parts)


def _reduce_to_free_params(
    full_params: torch.Tensor,
    constraints: ParameterConstraints,
    n_total_params: int,
) -> torch.Tensor:
    """Extract only free parameters from full parameter space.

    Parameters
    ----------
    full_params : torch.Tensor
        Full parameter vector.
    constraints : ParameterConstraints
        The constraint specification.
    n_total_params : int
        Total number of parameters (2 * n_epochs).

    Returns
    -------
    torch.Tensor
        Reduced parameter vector containing only free parameters.

    """
    free_indices = constraints.get_free_param_indices(n_total_params)
    return full_params[free_indices]


def fit_birth_death_processes(
    list_of_ages: list[torch.Tensor],
    times: torch.Tensor | float,
    *,
    initial_birth_rates: torch.Tensor | None = None,
    initial_death_rates: torch.Tensor | None = None,
    is_ordered: bool = False,
    spine_ages: torch.Tensor | None = None,
    population_sizes: torch.Tensor | None = None,
    initial_population: int = 1,
    optimizers: Sequence[str] | str = ("LBFGS", "Adam"),
    optimizer_kwargs: dict[str, dict[str, Any]] | None = None,
    perturb_initial_params_std: float = 0.0,
    random_seed: int | None = None,
    unbias: bool = True,
    convergence_config: ConvergenceConfig | None = None,
    param_constraints: ParameterConstraints | None = None,
    estimate_uncertainty: bool = True,
) -> dict[str, Any]:
    """Fit birth-death parameters to grouped age data via PyTorch.

    Optimizes log-parameters with requested optimizers (defaults: LBFGS, Adam).

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
    initial_population : int
        The number of initial individuals at time 0. Default is 1.
    optimizers : Sequence[str] | str
        Optimizer names ('LBFGS', 'Adam', 'SGD') to try sequentially.
        Defaults to ('LBFGS', 'Adam').
    optimizer_kwargs : Dict[str, Dict[str, Any]] | None
        Custom per-optimizer hyperparameters (e.g. {'adam': {'lr': 0.01}}).
    perturb_initial_params_std : float
        Noise std as fraction of initial params; 0 = no noise.
    random_seed : int | None
        Seed for reproducible perturbation.
    convergence_config : ConvergenceConfig | None
        Custom convergence criteria; defaults used if None.
    param_constraints : ParameterConstraints | None
        Constraints for fixed/equal parameters; None = no constraints.
    estimate_uncertainty : bool
        If True (default), estimate uncertainty via the observed Fisher
        information (Hessian of NLL in free log-parameter space). The returned
        covariance is in the full log-parameter space.
    unbias : bool
        If True (default), use the conditional non-extinction population mass
        for positive counts when evaluating the likelihood.

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
        initial_population=initial_population,
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

    # Initialize and validate constraints
    if param_constraints is None:
        param_constraints = ParameterConstraints()
    param_constraints.validate(len(initial_params))

    initial_params = _apply_param_perturbation(
        initial_params=initial_params,
        perturb_initial_params_std=perturb_initial_params_std,
        random_seed=random_seed,
    )

    # For constrained params, reduce to free parameter space
    free_indices = param_constraints.get_free_param_indices(len(initial_params))
    free_initial_params = initial_params[free_indices]
    log_params = torch.nn.Parameter(torch.log(free_initial_params.clone().detach()))

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
        initial_population=initial_population,
        unbias=unbias,
        convergence_config=convergence_config,
        param_constraints=param_constraints,
    )

    log_params = optimizer_run_result.log_params
    converged = optimizer_run_result.converged

    optimization_result = _build_optimization_result_payload(optimizer_run_result)

    # Expand log_params from reduced to full space for finalization
    full_params = _apply_constraints_to_full_params(
        torch.exp(log_params),
        param_constraints,
        2 * n_epochs,
    )
    full_log_params = torch.log(full_params).detach()

    finalized = _finalize_fit(
        full_log_params,
        n_epochs,
        non_spine_ages,
        times_tensor,
        spine_ages_tensor=spine_ages_tensor,
        population_sizes_tensor=population_sizes_tensor,
        initial_population=initial_population,
        unbias=unbias,
    )
    if finalized is None:
        return {
            "success": False,
            "optimization_result": optimization_result,
        }

    birth_rates, death_rates, final_log_likelihood, final_loss = finalized
    result = {
        "success": converged,
        "birth_rates": birth_rates,
        "death_rates": death_rates,
        "log_likelihood": final_log_likelihood,
        "optimization_result": {
            **optimization_result,
            "final_loss": final_loss,
        },
    }

    if estimate_uncertainty:

        def full_log_params_fn(theta_free: torch.Tensor) -> torch.Tensor:
            return torch.log(
                _apply_constraints_to_full_params(
                    torch.exp(theta_free),
                    param_constraints,
                    2 * n_epochs,
                ),
            )

        covariance = estimate_mle_covariance(
            free_log_params=log_params.detach(),
            n_epochs=n_epochs,
            full_log_params_fn=full_log_params_fn,
            non_spine_ages=non_spine_ages,
            times_tensor=times_tensor,
            spine_ages_tensor=spine_ages_tensor,
            population_sizes_tensor=population_sizes_tensor,
            initial_population=initial_population,
            unbias=unbias,
        )
        result["uncertainty"] = covariance.detach().cpu().numpy()

    return result


def _build_optimization_result_payload(
    optimizer_run_result: OptimizerRunResult,
) -> dict[str, Any]:
    """Create attempt-centric optimization payload for fit results."""
    attempts_payload: list[dict[str, Any]] = []
    for attempt in optimizer_run_result.attempts:
        first_nll = attempt.nll_history[0] if attempt.nll_history else None
        best_nll = min(attempt.nll_history) if attempt.nll_history else None
        final_nll = attempt.nll_history[-1] if attempt.nll_history else None
        attempts_payload.append(
            {
                "optimizer": attempt.optimizer,
                "attempt_index": attempt.attempt_index,
                "converged": attempt.converged,
                "iterations": attempt.iterations,
                "nll_history": attempt.nll_history,
                "grad_norms": attempt.grad_norms,
                "param_history": attempt.param_history,
                "timestamps": attempt.timestamps,
                "first_nll": first_nll,
                "best_nll": best_nll,
                "final_nll": final_nll,
                "exception_type": attempt.exception_type,
                "exception": attempt.exception_message,
            },
        )

    best_attempt_index: int | None = None
    best_attempt_value = float("inf")
    for attempt_payload in attempts_payload:
        if attempt_payload["best_nll"] is None:
            continue
        if attempt_payload["best_nll"] < best_attempt_value:
            best_attempt_value = attempt_payload["best_nll"]
            best_attempt_index = attempt_payload["attempt_index"]

    first_attempt_index = 0 if attempts_payload else None
    last_attempt_index = len(attempts_payload) - 1 if attempts_payload else None

    return {
        "converged": optimizer_run_result.converged,
        "selected_attempt_index": optimizer_run_result.selected_attempt_index,
        "first_attempt_index": first_attempt_index,
        "last_attempt_index": last_attempt_index,
        "best_attempt_index": best_attempt_index,
        "attempts": attempts_payload,
    }


def fit_from_data_frame(
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
        Input data_frame containing age data.
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
    times_tensor: torch.Tensor,
    spine_ages_tensor: torch.Tensor | None,
    population_sizes_tensor: torch.Tensor,
    initial_population: int,
    convergence_config: ConvergenceConfig,
    param_constraints: ParameterConstraints | None = None,
    unbias: bool = True,
) -> OptimizerRunResult:
    """Run optimizers sequentially until convergence; return params, history, metadata.

    Args:
        strategies_map: Optimizer configs.
        requested_keys: Optimizer names to try sequentially.
        log_params: Parameter tensor (log-scale, may be reduced if constraints apply).
        initial_params: Initial params on original scale (full space).
        n_epochs: Number of time epochs.
        non_spine_ages: Non-spine age data.
        initial_population: Initial population size.
        times_tensor: Time points.
        spine_ages_tensor: Spine ages if provided.
        population_sizes_tensor: Population sizes per group.
        convergence_config: Configuration for convergence criteria.
        param_constraints: Parameter constraints (fixed/equal params).

    Returns
    -------
        OptimizerRunResult with optimized params, history, and metadata.

    """
    if param_constraints is None:
        param_constraints = ParameterConstraints()

    n_total_params = 2 * n_epochs

    converged = False
    selected_attempt_index: int | None = None
    attempts: list[OptimizerAttemptResult] = []

    def _compute_nll() -> torch.Tensor:
        """Compute negative log likelihood for current log_params."""
        # Expand from reduced to full parameter space with constraints applied
        full_log_params = _apply_constraints_to_full_params(
            torch.exp(log_params),
            param_constraints,
            n_total_params,
        )
        full_log_params_log = torch.log(full_log_params)
        return negative_log_likelihood(
            birth_rates=full_log_params_log[:n_epochs],
            death_rates=full_log_params_log[n_epochs:],
            ages=non_spine_ages,
            times=times_tensor,
            spine_ages=spine_ages_tensor,
            ns=population_sizes_tensor,
            initial_population=initial_population,
            unbias=unbias,
            is_ordered=False,
            log_scale_params=True,
        )

    for idx, key in enumerate(requested_keys):
        strategy = strategies_map[key]

        attempt, best_log_params = _run_single_optimizer(
            strategy=strategy,
            attempt_index=idx,
            log_params=log_params,
            initial_params=initial_params,
            convergence_config=convergence_config,
            compute_nll_fn=_compute_nll,
            param_constraints=param_constraints,
            n_total_params=n_total_params,
        )
        attempts.append(attempt)

        if best_log_params is not None:
            with torch.no_grad():
                log_params.copy_(best_log_params)

        if attempt.converged:
            converged = True
            selected_attempt_index = idx
            break

    if not converged and attempts:
        selected_attempt_index = 0

    return OptimizerRunResult(
        log_params=log_params,
        converged=converged,
        selected_attempt_index=selected_attempt_index,
        attempts=attempts,
    )


def _run_single_optimizer(
    strategy: OptimizerStrategy,
    attempt_index: int,
    log_params: torch.nn.Parameter,
    initial_params: torch.Tensor,
    convergence_config: ConvergenceConfig,
    compute_nll_fn: Callable[[], torch.Tensor],
    param_constraints: ParameterConstraints | None = None,
    n_total_params: int | None = None,
) -> tuple[OptimizerAttemptResult, torch.Tensor | None]:
    """Run a single optimizer strategy and handle results.

    Returns
    -------
        (attempt_result, best_log_params)

    """
    if param_constraints is None:
        param_constraints = ParameterConstraints()
    if n_total_params is None:
        n_total_params = len(initial_params)

    attempt_nll_history: list[float] = []
    attempt_grad_norms: list[float] = []
    attempt_param_history: list[list[float]] = []
    attempt_timestamps: list[float] = []
    iterations = 0

    try:
        with torch.no_grad():
            # Reduce initial_params to free space with constraints applied
            free_initial_params = _reduce_to_free_params(
                initial_params,
                param_constraints,
                n_total_params,
            )
            log_params.copy_(torch.log(free_initial_params.clone().detach()))
        # remove internal-only kwargs before constructing torch optimizer
        opt_kwargs = {k: v for k, v in strategy.optimizer_kwargs.items()}
        opt_kwargs.pop("record_every", None)
        opt_kwargs.pop("min_iterations", None)
        optimizer = strategy.optimizer_class([log_params], **opt_kwargs)

        if strategy.name == "LBFGS":
            lbfgs_optimizer = cast("torch.optim.LBFGS", optimizer)
            did_converge, exc, best_log_params, iterations = _run_lbfgs_optimizer(
                strategy=strategy,
                log_params=log_params,
                optimizer=lbfgs_optimizer,
                max_iters=strategy.max_iters,
                nll_history=attempt_nll_history,
                grad_norms=attempt_grad_norms,
                param_history=attempt_param_history,
                timestamps=attempt_timestamps,
                convergence_config=convergence_config,
                compute_nll_fn=compute_nll_fn,
            )
        else:
            did_converge, exc, best_log_params, iterations = _run_first_order_optimizer(
                strategy=strategy,
                log_params=log_params,
                optimizer=optimizer,
                max_iters=strategy.max_iters,
                nll_history=attempt_nll_history,
                grad_norms=attempt_grad_norms,
                param_history=attempt_param_history,
                timestamps=attempt_timestamps,
                convergence_config=convergence_config,
                compute_nll_fn=compute_nll_fn,
            )
    except (RuntimeError, ValueError) as exc:
        return (
            OptimizerAttemptResult(
                optimizer=strategy.name,
                attempt_index=attempt_index,
                converged=False,
                iterations=iterations,
                nll_history=attempt_nll_history,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                grad_norms=attempt_grad_norms or None,
                param_history=attempt_param_history or None,
                timestamps=attempt_timestamps or None,
            ),
            None,
        )
    else:
        if exc is not None:
            return (
                OptimizerAttemptResult(
                    optimizer=strategy.name,
                    attempt_index=attempt_index,
                    converged=False,
                    iterations=iterations,
                    nll_history=attempt_nll_history,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                    grad_norms=attempt_grad_norms or None,
                    param_history=attempt_param_history or None,
                    timestamps=attempt_timestamps or None,
                ),
                None,
            )

        return (
            OptimizerAttemptResult(
                optimizer=strategy.name,
                attempt_index=attempt_index,
                converged=did_converge,
                iterations=iterations,
                nll_history=attempt_nll_history,
                grad_norms=attempt_grad_norms or None,
                param_history=attempt_param_history or None,
                timestamps=attempt_timestamps or None,
            ),
            best_log_params,
        )


def get_crude_parameter_estimates(
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    birth_rates: torch.Tensor | float | None = None,
    death_rates: torch.Tensor | float | None = None,
    initial_population: int = 1,
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
                    float(n_manuscripts / initial_population),
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

    est_birth_rates = est_birth_rates + (est_death_rates - est_birth_rates) / 4
    est_death_rates = est_death_rates - (est_death_rates - est_birth_rates) / 4

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
    initial_population: int = 1,
) -> torch.Tensor:
    """Build initial param vector; auto-estimate if needed."""
    if initial_birth_rates is None or initial_death_rates is None:
        crude_birth, crude_death = get_crude_parameter_estimates(
            non_empty_age,
            times_tensor,
            birth_rates=initial_birth_rates,
            death_rates=initial_death_rates,
            initial_population=initial_population,
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
            {"lr": 0.1},
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


def _finalize_fit(
    log_params: torch.Tensor,
    n_epochs: int,
    non_spine_ages: torch.Tensor,
    times_tensor: torch.Tensor,
    *,
    spine_ages_tensor: torch.Tensor | None,
    population_sizes_tensor: torch.Tensor,
    initial_population: int = 1,
    unbias: bool = True,
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
            initial_population=initial_population,
            unbias=unbias,
            is_ordered=False,
            log_scale_params=True,
        )

        if not torch.isfinite(final_nll):
            return None

        birth_rates = torch.exp(log_params[:n_epochs]).detach().cpu().numpy()
        death_rates = torch.exp(log_params[n_epochs:]).detach().cpu().numpy()
        final_log_likelihood = -float(final_nll.item())

    return birth_rates, death_rates, final_log_likelihood, float(final_nll.item())
