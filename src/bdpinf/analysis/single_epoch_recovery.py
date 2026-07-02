"""Simulate-then-fit replicate logic for single-epoch bulk-inference recovery studies.

Moved out of ``qmd_scripts/validation/single_epoch/06_bulk_inference.qmd`` (and
the equivalent sampling step in ``04_bulk_inference_single_rate.qmd``) so the
heavy per-replicate work can be run once by
``scripts/generate_recovery_studies.py`` and persisted via
:func:`~inf_bp_texts.inference.save_recovery_results`, instead of re-running on
every notebook render.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inf_bp_texts.analysis.validation_helpers import get_selected_attempt
from inf_bp_texts.inference import ConvergenceConfig, fit_birth_death_processes
from inf_bp_texts.simulation import simulate_bdp_ages_from_distributions

_DEFAULT_OPTIMIZER_KWARGS: dict[str, dict[str, Any]] = {
    "lbfgs": {"lr": 1.0, "line_search_fn": "strong_wolfe", "min_iterations": 10},
    "adam": {"lr": 1e-3},
}
_DEFAULT_CONVERGENCE_CONFIG = ConvergenceConfig(
    gradient_threshold=1e-6, absolute_tolerance=1e-10, patience=10
)


def sample_single_epoch_ages(
    seed: int,
    birth_rate: float,
    death_rate: float,
    time_span: float,
) -> NDArray[np.float64]:
    """Sample one replicate's age vector from the birth-death age distribution."""
    rng = np.random.default_rng(seed)
    sampled = simulate_bdp_ages_from_distributions(
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
        max_time=float(time_span),
        rng=rng,
    )
    assert sampled.spine_ages is not None and sampled.non_spine_ages is not None
    return np.asarray(
        sampled.spine_ages.tolist() + sampled.non_spine_ages.tolist(),
        dtype=np.float64,
    )


def fit_single_epoch_replicate(
    ages: NDArray[np.float64],
    birth_rate: float,
    death_rate: float,
    time_span: float,
    *,
    optimizers: Sequence[str] = ("lbfgs", "adam"),
    optimizer_kwargs: dict[str, dict[str, Any]] | None = None,
    convergence_config: ConvergenceConfig | None = None,
) -> dict[str, Any] | None:
    """Fit a single-epoch model to one age vector; ``None`` on failure or empty ages."""
    if len(ages) == 0:
        return None

    try:
        return fit_birth_death_processes(
            list_of_ages=[ages],
            times=time_span,
            is_ordered=True,
            perturb_initial_params_std=0.0,
            initial_birth_rates=np.array([birth_rate], dtype=float),
            initial_death_rates=np.array([death_rate], dtype=float),
            optimizers=list(optimizers),
            optimizer_kwargs=optimizer_kwargs or _DEFAULT_OPTIMIZER_KWARGS,
            convergence_config=convergence_config or _DEFAULT_CONVERGENCE_CONFIG,
        )
    except Exception:
        return None


def run_single_epoch_recovery_replicate(
    simulation_index: int,
    seed: int,
    birth_rate: float,
    death_rate: float,
    time_span: float,
) -> dict[str, Any]:
    """Sample and fit one replicate; return a tidy result row.

    Used by the ``single_epoch_bulk_inference`` recovery study
    (``06_bulk_inference.qmd``).
    """
    ages = sample_single_epoch_ages(seed, birth_rate, death_rate, time_span)
    return _row_from_fit(
        simulation_index,
        ages,
        time_span,
        fit_single_epoch_replicate(ages, birth_rate, death_rate, time_span),
    )


def _row_from_fit(
    simulation_index: int,
    ages: NDArray[np.float64],
    time_span: float,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    if len(ages) == 0:
        return {
            "simulation_index": simulation_index,
            "success": False,
            "population": 0,
            "mean_age": np.nan,
            "has_boundary_age": False,
            "birth_rate": np.nan,
            "death_rate": np.nan,
            "optimizer": "none",
        }

    population = len(ages)
    mean_age = float(np.mean(ages))

    if result is None or not result.get("success", False):
        return {
            "simulation_index": simulation_index,
            "success": False,
            "population": population,
            "mean_age": mean_age,
            "has_boundary_age": False,
            "birth_rate": np.nan,
            "death_rate": np.nan,
            "optimizer": "none",
        }

    has_boundary_age = bool(np.max(ages) == time_span)
    selected_attempt = get_selected_attempt(result)
    optimizer = (
        str(selected_attempt.get("optimizer", "unknown"))
        if selected_attempt is not None
        else "unknown"
    )

    return {
        "simulation_index": simulation_index,
        "success": True,
        "population": population,
        "mean_age": mean_age,
        "has_boundary_age": has_boundary_age,
        "birth_rate": float(np.atleast_1d(result["birth_rates"])[0]),
        "death_rate": float(np.atleast_1d(result["death_rates"])[0]),
        "optimizer": optimizer,
    }
