"""Maximum likelihood estimation for birth-death process parameters using JAX."""

import dataclasses
import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, Unpack

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray

from bdpinf.models.validation import (
    normalize_and_validate_inputs,
)
from bdpinf.utils import ensure_1d_tensor

from ._fit_types import (
    DEFAULT_ITERATIONS,
    MIN_EQUALITY_GROUP_SIZE,
    ConvergenceConfig,
    OptimizerAttemptResult,
    OptimizerRunResult,
    OptimizerStrategy,
)
from ._optimizer_runners import (
    _scaled_config,  # pyright: ignore[reportPrivateUsage]
    run_first_order_optimizer,
    run_lbfgs_optimizer,
)
from .likelihood import (
    _log_likelihood_core,  # pyright: ignore[reportPrivateUsage]
    log_likelihood,
    resolve_spine_ages,
)
from .uncertainty import estimate_mle_covariance

logger = logging.getLogger(__name__)


class FitKwargs(TypedDict, total=False):
    """TypedDict for fit_birth_death_processes kwargs."""

    initial_birth_rates: jax.Array | NDArray[np.float64] | None
    initial_death_rates: jax.Array | NDArray[np.float64] | None
    is_ordered: bool
    spine_ages: jax.Array | NDArray[np.float64] | None
    population_sizes: jax.Array | NDArray[np.float64] | None
    initial_population: int
    optimizers: Sequence[str] | str
    condition_on_survival: bool
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
    free_params: jax.Array,
    constraints: ParameterConstraints,
    n_total_params: int,
) -> jax.Array:
    """Expand free parameters to full parameter space applying all constraints.

    Parameters
    ----------
    free_params : jax.Array
        Parameters corresponding to free indices only.
    constraints : ParameterConstraints
        The constraint specification.
    n_total_params : int
        Total number of parameters (2 * n_epochs).

    Returns
    -------
    jax.Array
        Full parameter vector with constraints applied.

    """
    free_indices = constraints.get_free_param_indices(n_total_params)
    free_position_by_index = {idx: pos for pos, idx in enumerate(free_indices)}

    representative_by_index: dict[int, int] = {}
    for group in constraints.equality_groups:
        representative_idx = min(group)
        for idx in group:
            representative_by_index[idx] = representative_idx

    parts: list[jax.Array] = []
    for idx in range(n_total_params):
        if idx in constraints.fixed_params:
            parts.append(
                jnp.array(constraints.fixed_params[idx], dtype=free_params.dtype),
            )
            continue

        source_idx = representative_by_index.get(idx, idx)
        parts.append(free_params[free_position_by_index[source_idx]])

    return jnp.stack(parts)


def _reduce_to_free_params(
    full_params: jax.Array,
    constraints: ParameterConstraints,
    n_total_params: int,
) -> jax.Array:
    """Extract only free parameters from full parameter space.

    Parameters
    ----------
    full_params : jax.Array
        Full parameter vector.
    constraints : ParameterConstraints
        The constraint specification.
    n_total_params : int
        Total number of parameters (2 * n_epochs).

    Returns
    -------
    jax.Array
        Reduced parameter vector containing only free parameters.

    """
    free_indices = constraints.get_free_param_indices(n_total_params)
    return full_params[jnp.array(free_indices)]


def fit_birth_death_processes(
    list_of_ages: list[jax.Array | NDArray[np.float64]],
    times: jax.Array | NDArray[np.float64] | float,
    *,
    initial_birth_rates: jax.Array | NDArray[np.float64] | None = None,
    initial_death_rates: jax.Array | NDArray[np.float64] | None = None,
    is_ordered: bool = False,
    spine_ages: jax.Array | NDArray[np.float64] | None = None,
    population_sizes: jax.Array | NDArray[np.float64] | None = None,
    initial_population: int = 1,
    optimizers: Sequence[str] | str = ("LBFGS", "Adam"),
    optimizer_kwargs: dict[str, dict[str, Any]] | None = None,
    perturb_initial_params_std: float = 0.0,
    random_seed: int | None = None,
    condition_on_survival: bool = True,
    convergence_config: ConvergenceConfig | None = None,
    param_constraints: ParameterConstraints | None = None,
    estimate_uncertainty: bool = True,
) -> dict[str, Any]:
    """Fit birth-death parameters to grouped age data via JAX.

    Optimizes log-parameters with requested optimizers (defaults: LBFGS, Adam).

    Parameters
    ----------
    list_of_ages : list[jax.Array | NDArray]
        Age arrays per group; ≥1 non-empty array required.
    times : jax.Array | NDArray | float
        Observation time(s), must be positive.
    is_ordered : bool
        If True, extract spine from first element per array.
    initial_birth_rates, initial_death_rates : jax.Array | NDArray | None
        Initial guesses; auto-estimated if None.
    spine_ages : jax.Array | NDArray | None
        Explicit spine ages (if not auto-extracted).
    population_sizes : jax.Array | NDArray | None
        Counts per group; auto-computed if None.
    initial_population : int
        The number of initial individuals at time 0. Default is 1.
    optimizers : Sequence[str] | str
        Optimizer names ('LBFGS', 'Adam', 'SGD') to try sequentially.
        Defaults to ('LBFGS', 'Adam').
    optimizer_kwargs : dict[str, dict[str, Any]] | None
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
    condition_on_survival : bool
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
    if bool(jnp.any(initial_params <= 0)):
        msg = "Initial parameters must be positive before log transform"
        raise ValueError(msg)

    n_epochs = len(times_tensor)
    expected_len = 2 * n_epochs
    if len(initial_params) != expected_len:
        msg = (
            "Initial parameters size mismatch with times: "
            f"expected {expected_len}, got {len(initial_params)}"
        )
        raise ValueError(msg)

    if param_constraints is None:
        param_constraints = ParameterConstraints()
    param_constraints.validate(len(initial_params))

    initial_params = _apply_param_perturbation(
        initial_params=initial_params,
        perturb_initial_params_std=perturb_initial_params_std,
        random_seed=random_seed,
    )

    convergence_config = convergence_config or ConvergenceConfig.default()
    strategies_map = _build_optimizer_strategies(
        optimizer_kwargs=optimizer_kwargs,
        convergence_config=convergence_config,
    )
    requested_keys = _normalize_requested_optimizers(optimizers, strategies_map)

    optimizer_run_result = _run_optimizer_strategies(
        strategies_map=strategies_map,
        requested_keys=requested_keys,
        initial_params=initial_params,
        n_epochs=n_epochs,
        non_spine_ages=non_spine_ages,
        times_tensor=times_tensor,
        spine_ages_tensor=spine_ages_tensor,
        population_sizes_tensor=population_sizes_tensor,
        initial_population=initial_population,
        condition_on_survival=condition_on_survival,
        convergence_config=convergence_config,
        param_constraints=param_constraints,
    )

    log_params = optimizer_run_result.log_params
    converged = optimizer_run_result.converged

    optimization_result = _build_optimization_result_payload(optimizer_run_result)

    full_params = _apply_constraints_to_full_params(
        jnp.exp(log_params),
        param_constraints,
        2 * n_epochs,
    )
    full_log_params = jnp.log(full_params)

    finalized = _finalize_fit(
        full_log_params,
        n_epochs,
        non_spine_ages,
        times_tensor,
        spine_ages_tensor=spine_ages_tensor,
        population_sizes_tensor=population_sizes_tensor,
        initial_population=initial_population,
        condition_on_survival=condition_on_survival,
    )
    if finalized is None:
        return {
            "success": False,
            "optimization_result": optimization_result,
        }

    birth_rates, death_rates, final_log_likelihood, final_loss = finalized
    result: dict[str, Any] = {
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
        if not converged:
            warnings.warn(
                "Estimating uncertainty from a fit that did not converge; the "
                "Hessian is evaluated at a possibly non-stationary point, so "
                "the resulting covariance/standard errors may not be valid.",
                UserWarning,
                stacklevel=2,
            )

        def full_log_params_fn(theta_free: jax.Array) -> jax.Array:
            return jnp.log(
                _apply_constraints_to_full_params(
                    jnp.exp(theta_free),
                    param_constraints,
                    2 * n_epochs,
                ),
            )

        covariance = estimate_mle_covariance(
            free_log_params=log_params,
            n_epochs=n_epochs,
            full_log_params_fn=full_log_params_fn,
            non_spine_ages=non_spine_ages,
            times_tensor=times_tensor,
            spine_ages_tensor=spine_ages_tensor,
            population_sizes_tensor=population_sizes_tensor,
            initial_population=initial_population,
            condition_on_survival=condition_on_survival,
        )
        result["uncertainty"] = np.array(covariance)

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
    df: "Any",
    times: float | NDArray[np.float64] | jax.Array,
    *,
    age_column: str | None = None,
    **fit_kwargs: Unpack[FitKwargs],
) -> dict[str, Any]:
    """Fit parameters from DataFrame age column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data_frame containing age data.
    times : float | NDArray | jax.Array
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

    ages_arr = jnp.array(
        df[age_column].to_numpy(dtype=np.float64, copy=True),
        dtype=jnp.float64,
    )
    times_arr = jnp.array(times, dtype=jnp.float64)
    times_arr, ages_arr = normalize_and_validate_inputs(
        times=times_arr,
        ages=ages_arr,
    )

    return fit_birth_death_processes(
        list_of_ages=[ages_arr],
        times=times_arr,
        **fit_kwargs,
    )


def _run_optimizer_strategies(
    strategies_map: dict[str, OptimizerStrategy],
    requested_keys: list[str],
    initial_params: jax.Array,
    n_epochs: int,
    non_spine_ages: jax.Array,
    times_tensor: jax.Array,
    spine_ages_tensor: jax.Array | None,
    population_sizes_tensor: jax.Array,
    initial_population: int,
    convergence_config: ConvergenceConfig,
    param_constraints: ParameterConstraints | None = None,
    condition_on_survival: bool = True,
) -> OptimizerRunResult:
    """Run optimizers sequentially until convergence; return params and metadata."""
    if param_constraints is None:
        param_constraints = ParameterConstraints()

    n_total_params = 2 * n_epochs

    free_initial_params = _reduce_to_free_params(
        initial_params,
        param_constraints,
        n_total_params,
    )
    log_params_initial = jnp.log(free_initial_params)
    log_params = log_params_initial

    # ------------------------------------------------------------------
    # Host-side spine pre-resolution (same resolution as ``log_likelihood``).
    # We resolve all data-dependent and host-side branches here once so the
    # jitted NLL core sees only static-shape, concrete closure constants.
    # ------------------------------------------------------------------
    _jit_ns = population_sizes_tensor
    _jit_times = times_tensor

    if spine_ages_tensor is None and len(non_spine_ages) > 0:
        resolved = resolve_spine_ages(
            non_spine_ages,
            float(times_tensor[-1]),
            multiple_terminal="marginalize",
        )
        _jit_spine: jax.Array | None = (
            None
            if resolved.spine_ages is None
            else jnp.asarray(resolved.spine_ages, dtype=jnp.float64)
        )
        _jit_non_spine = jnp.asarray(resolved.non_spine_ages, dtype=jnp.float64)
        _use_unknown_spine = resolved.use_unknown_spine
    else:
        _jit_spine = spine_ages_tensor
        _jit_non_spine = non_spine_ages
        _use_unknown_spine = False

    # ------------------------------------------------------------------
    # Build the jitted value-and-gradient callable.
    # The NLL closure calls the math kernels directly with
    # _skip_normalization=True, avoiding all host-side bool() hazards.
    # All static info (shapes, flags, data arrays) are closure constants;
    # only log_params_free is the traced argument.
    # ------------------------------------------------------------------
    def _jittable_nll(log_params_free: jax.Array) -> jax.Array:
        full_params = _apply_constraints_to_full_params(
            jnp.exp(log_params_free),
            param_constraints,
            n_total_params,
        )
        birth_rates = full_params[:n_epochs]
        death_rates = full_params[n_epochs:]
        return -_log_likelihood_core(
            birth_rates=birth_rates,
            death_rates=death_rates,
            non_spine_ages=_jit_non_spine,
            spine_ages=_jit_spine,
            times=_jit_times,
            ns=_jit_ns,
            initial_population=initial_population,
            condition_on_survival=condition_on_survival,
            use_unknown_spine_marginal=_use_unknown_spine,
            skip_normalization=True,
        )

    val_and_grad_fn = jax.jit(jax.value_and_grad(_jittable_nll))

    # The objective sums a term per observed age, so its gradient norm scales
    # with the number of observations; scale the gradient-threshold portion of
    # the convergence criteria accordingly (see _optimizer_runners._scaled_config
    # for why) so the stopping rule is sample-size invariant.
    n_observations = len(_jit_non_spine) + (
        len(_jit_spine) if _jit_spine is not None else 0
    )
    scaled_convergence_config = _scaled_config(convergence_config, n_observations)

    converged = False
    selected_attempt_index: int | None = None
    attempts: list[OptimizerAttemptResult] = []

    # Warm-start each subsequent optimizer from the running best parameters
    # (not the original initial guess) so a sequence like ("LBFGS", "Adam")
    # actually refines LBFGS's result with Adam, matching the documented
    # LBFGS-then-Adam workflow instead of running two independent fits.
    log_params = log_params_initial
    best_overall_nll = float("inf")
    best_overall_log_params: jax.Array | None = None
    best_overall_attempt_index: int | None = None

    for idx, key in enumerate(requested_keys):
        strategy = strategies_map[key]

        attempt, best_log_params = _run_single_optimizer(
            strategy=strategy,
            attempt_index=idx,
            log_params=log_params,
            convergence_config=scaled_convergence_config,
            val_and_grad_fn=val_and_grad_fn,
        )
        attempts.append(attempt)

        if best_log_params is not None:
            log_params = best_log_params

        attempt_best_nll = min(attempt.nll_history) if attempt.nll_history else None
        if (
            attempt_best_nll is not None
            and math.isfinite(attempt_best_nll)
            and best_log_params is not None
            and attempt_best_nll < best_overall_nll
        ):
            best_overall_nll = attempt_best_nll
            best_overall_log_params = best_log_params
            best_overall_attempt_index = idx

        if attempt.converged:
            converged = True
            break

    # Return parameters from the globally-best attempt (lowest NLL across all
    # attempts actually run), not merely the last attempt run, and report that
    # attempt's index so the returned rates, `selected_attempt_index`, and
    # `best_attempt_index` (computed from the same nll_history data in
    # _build_optimization_result_payload) all agree.  When an attempt converges
    # we stop immediately (as before), and that attempt is necessarily the
    # best one seen so far, so this also covers the converged path.
    if best_overall_log_params is not None:
        log_params = best_overall_log_params
        selected_attempt_index = best_overall_attempt_index
    elif attempts:
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
    log_params: jax.Array,
    convergence_config: ConvergenceConfig,
    val_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
) -> tuple[OptimizerAttemptResult, jax.Array | None]:
    """Run a single optimizer strategy and handle results.

    Returns
    -------
        (attempt_result, best_log_params)

    """
    attempt_nll_history: list[float] = []
    attempt_grad_norms: list[float] = []
    attempt_param_history: list[list[float]] = []
    attempt_timestamps: list[float] = []
    iterations = 0

    try:
        runner = (
            run_lbfgs_optimizer
            if strategy.name == "LBFGS"
            else run_first_order_optimizer
        )
        did_converge, run_exc, best_log_params, iterations = runner(
            strategy=strategy,
            log_params=log_params,
            max_iters=strategy.max_iters,
            nll_history=attempt_nll_history,
            grad_norms=attempt_grad_norms,
            param_history=attempt_param_history,
            timestamps=attempt_timestamps,
            convergence_config=convergence_config,
            val_and_grad_fn=val_and_grad_fn,
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

    if run_exc is not None:
        return (
            OptimizerAttemptResult(
                optimizer=strategy.name,
                attempt_index=attempt_index,
                converged=False,
                iterations=iterations,
                nll_history=attempt_nll_history,
                exception_type=type(run_exc).__name__,
                exception_message=str(run_exc),
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
    ages: jax.Array | NDArray[np.float64] | float,
    times: jax.Array | NDArray[np.float64] | float,
    *,
    birth_rates: jax.Array | NDArray[np.float64] | float | None = None,
    death_rates: jax.Array | NDArray[np.float64] | float | None = None,
    initial_population: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Get crude initial parameter estimates from age data."""
    if birth_rates is not None and death_rates is not None:
        msg = "Cannot provide both birth_rates and death_rates"
        raise ValueError(msg)

    times_arr, ages_arr = normalize_and_validate_inputs(times=times, ages=ages)

    observation_time = times_arr[-1]
    n_individuals = len(ages_arr)
    if n_individuals == 0:
        msg = "Cannot estimate parameters from empty age data"
        raise ValueError(msg)

    mean_age = float(ages_arr.mean())

    if birth_rates is None:
        est_birth_rates = jnp.ones_like(times_arr, dtype=jnp.float64) / mean_age
    else:
        est_birth_rates = ensure_1d_tensor(birth_rates)

    if death_rates is None:
        est_death_rates = jnp.maximum(
            est_birth_rates
            - jnp.log(
                jnp.array(float(n_individuals / initial_population), dtype=jnp.float64),
            )
            / observation_time,
            jnp.full_like(times_arr, 1e-8, dtype=jnp.float64),
        )
    else:
        est_death_rates = ensure_1d_tensor(death_rates)

    if len(times_arr) != len(est_birth_rates) or len(times_arr) != len(est_death_rates):
        msg = (
            f"Size mismatch: times ({len(times_arr)}), "
            f"est_birth_rates ({len(est_birth_rates)}), "
            f"est_death_rates ({len(est_death_rates)}) must all be equal"
        )
        raise ValueError(msg)

    # Pull birth and death rates a quarter of the way toward each other.
    # Both updates must use the *original* (pre-shrink) values: computing
    # est_death_rates from the already-updated est_birth_rates would make the
    # pull asymmetric (birth moves a quarter of the original gap, death moves
    # a quarter of the now-smaller gap).
    rate_gap = est_death_rates - est_birth_rates
    est_birth_rates = est_birth_rates + rate_gap / 4
    est_death_rates = est_death_rates - rate_gap / 4

    return (est_birth_rates, est_death_rates)


def _find_non_empty_age(
    list_of_ages: list[jax.Array | NDArray[np.float64]],
) -> jax.Array:
    """Return first non-empty age array; raise ValueError if all empty."""
    for ages in list_of_ages:
        if len(ages) > 0:
            return ensure_1d_tensor(ages)
    msg = "No valid data points for fitting"
    raise ValueError(msg)


def _prepare_population_sizes(
    list_of_ages: list[jax.Array | NDArray[np.float64]],
    population_sizes: jax.Array | NDArray[np.float64] | None,
    spine_ages: jax.Array | NDArray[np.float64] | None,
) -> jax.Array:
    """Return population_sizes or auto-compute from list_of_ages + spine_ages."""
    if population_sizes is not None:
        ps = ensure_1d_tensor(population_sizes)
        ps_int = jnp.array(ps, dtype=jnp.int32)
        if ps_int.size != len(list_of_ages):
            msg = "Length of population_sizes must match length of list_of_ages"
            raise ValueError(msg)
        return ps_int

    return jnp.array(
        [len(ages) + int(spine_ages is not None) for ages in list_of_ages],
        dtype=jnp.int32,
    )


def _extract_spine_and_ages(
    list_of_ages: list[jax.Array | NDArray[np.float64]],
    *,
    is_ordered: bool,
    spine_ages: jax.Array | NDArray[np.float64] | None,
) -> tuple[jax.Array | None, jax.Array]:
    """Extract/auto-extract spine ages and return (spine_ages, non_spine_ages)."""
    spine_ages_tensor: jax.Array | None
    if is_ordered and (spine_ages is None):
        spine_list: list[float] = []
        remaining: list[jax.Array] = []
        for arr in list_of_ages:
            if len(arr) == 0:
                continue
            arr_jax = ensure_1d_tensor(arr)
            spine_list.append(float(arr_jax[0]))
            if len(arr_jax) > 1:
                remaining.append(arr_jax[1:])

        spine_ages_tensor = jnp.array(spine_list, dtype=jnp.float64)
        non_spine_ages = (
            jnp.concatenate(remaining)
            if remaining
            else jnp.array([], dtype=jnp.float64)
        )
        return spine_ages_tensor, non_spine_ages

    spine_ages_tensor = None if spine_ages is None else ensure_1d_tensor(spine_ages)
    non_spine_tensors = [ensure_1d_tensor(arr) for arr in list_of_ages if len(arr) > 0]
    non_spine_ages = (
        jnp.concatenate(non_spine_tensors)
        if non_spine_tensors
        else jnp.array([], dtype=jnp.float64)
    )
    return spine_ages_tensor, non_spine_ages


def _build_initial_params(
    initial_birth_rates: jax.Array | NDArray[np.float64] | None,
    initial_death_rates: jax.Array | NDArray[np.float64] | None,
    non_empty_age: jax.Array,
    times_tensor: jax.Array,
    initial_population: int = 1,
) -> jax.Array:
    """Build initial param vector; auto-estimate if needed."""
    if initial_birth_rates is None or initial_death_rates is None:
        crude_birth, crude_death = get_crude_parameter_estimates(
            non_empty_age,
            times_tensor,
            birth_rates=initial_birth_rates,
            death_rates=initial_death_rates,
            initial_population=initial_population,
        )
        return jnp.concatenate([crude_birth, crude_death])

    return jnp.concatenate(
        [
            ensure_1d_tensor(initial_birth_rates),
            ensure_1d_tensor(initial_death_rates),
        ],
    )


def _apply_param_perturbation(
    initial_params: jax.Array,
    perturb_initial_params_std: float,
    random_seed: int | None,
) -> jax.Array:
    """Add multiplicative log-normal noise to params."""
    if perturb_initial_params_std <= 0:
        return initial_params
    seed = random_seed if random_seed is not None else 0
    key = jax.random.PRNGKey(seed)
    raw_noise = jax.random.normal(key, shape=initial_params.shape, dtype=jnp.float64)
    noise = jnp.maximum(raw_noise * perturb_initial_params_std + 1.0, 0.1)
    return initial_params * noise


def _lbfgs_solver_factory(**_: object) -> None:
    """Return None; LBFGS runs via scipy and has no optax solver object."""
    return


def _build_optimizer_strategies(
    optimizer_kwargs: dict[str, dict[str, Any]] | None,
    convergence_config: ConvergenceConfig,
) -> dict[str, OptimizerStrategy]:
    """Build optimizer config dict."""
    strategies_map: dict[str, OptimizerStrategy] = {
        "adam": OptimizerStrategy(
            name="Adam",
            solver_factory=optax.adam,
            max_iters=DEFAULT_ITERATIONS,
            solver_kwargs={"lr": 0.1},
        ),
        "lbfgs": OptimizerStrategy(
            name="LBFGS",
            solver_factory=_lbfgs_solver_factory,
            max_iters=DEFAULT_ITERATIONS // 20,
            solver_kwargs={
                "tolerance_grad": convergence_config.gradient_threshold,
                "tolerance_change": convergence_config.absolute_tolerance,
            },
        ),
        "sgd": OptimizerStrategy(
            name="SGD",
            solver_factory=optax.sgd,
            max_iters=DEFAULT_ITERATIONS,
            solver_kwargs={"lr": 0.001},
        ),
    }

    if optimizer_kwargs is None:
        return strategies_map

    for key, custom_kwargs in optimizer_kwargs.items():
        key_lower = key.lower()
        if key_lower in strategies_map:
            strategy = strategies_map[key_lower]
            merged_solver_kwargs = {**strategy.solver_kwargs, **custom_kwargs}
            strategies_map[key_lower] = OptimizerStrategy(
                name=strategy.name,
                solver_factory=strategy.solver_factory,
                max_iters=strategy.max_iters,
                solver_kwargs=merged_solver_kwargs,
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
            raise ValueError(msg)
    return requested_keys


def _finalize_fit(
    log_params: jax.Array,
    n_epochs: int,
    non_spine_ages: jax.Array,
    times_tensor: jax.Array,
    *,
    spine_ages_tensor: jax.Array | None,
    population_sizes_tensor: jax.Array,
    initial_population: int = 1,
    condition_on_survival: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float] | None:
    """Convert log params to original scale; return rates + final likelihoods."""
    final_nll = -log_likelihood(
        birth_rates=log_params[:n_epochs],
        death_rates=log_params[n_epochs:],
        ages=non_spine_ages,
        times=times_tensor,
        spine_ages=spine_ages_tensor,
        ns=population_sizes_tensor,
        initial_population=initial_population,
        condition_on_survival=condition_on_survival,
        is_ordered=False,
        log_scale_params=True,
    )

    if not math.isfinite(float(final_nll)):
        return None

    birth_rates = np.array(jnp.exp(log_params[:n_epochs]))
    death_rates = np.array(jnp.exp(log_params[n_epochs:]))
    final_log_likelihood = -float(final_nll)

    return birth_rates, death_rates, final_log_likelihood, float(final_nll)
