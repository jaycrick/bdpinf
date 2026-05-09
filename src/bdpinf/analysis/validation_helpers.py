"""Shared helpers for validation notebooks and scripts."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

from inf_bp_texts.inference import fit_birth_death_processes


@dataclass(frozen=True)
class FitModelConfig:
    """Configuration for single-model fitting helpers."""

    times: float | NDArray[np.float64]
    is_ordered: bool = True
    perturb_initial_params_std: float | None = None
    random_seed: int | None = None
    initial_birth_rates: NDArray[np.float64] | None = None
    initial_death_rates: NDArray[np.float64] | None = None
    unbias: bool = True


def fit_single_model(
    ages: NDArray[np.float64],
    config: FitModelConfig,
) -> dict[str, Any] | None:
    """Fit one model from a single age vector; return None on failure."""
    if len(ages) == 0:
        return None

    fit_kwargs: dict[str, Any] = {
        "list_of_ages": [ages],
        "times": config.times,
        "is_ordered": config.is_ordered,
    }
    if config.perturb_initial_params_std is not None:
        fit_kwargs["perturb_initial_params_std"] = config.perturb_initial_params_std
    if config.random_seed is not None:
        fit_kwargs["random_seed"] = config.random_seed
    if config.initial_birth_rates is not None:
        fit_kwargs["initial_birth_rates"] = config.initial_birth_rates
    if config.initial_death_rates is not None:
        fit_kwargs["initial_death_rates"] = config.initial_death_rates
    fit_kwargs["unbias"] = config.unbias

    try:
        return fit_birth_death_processes(**fit_kwargs)
    except Exception:
        return None


def first_failed_fits(
    fitted_models: list[dict[str, Any] | None],
    ages_by_fit: list[NDArray[np.float64]],
    start_index: int = 0,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    """Return failed-fit metadata sorted by population size descending."""

    def _failure_reason(model: dict[str, Any] | None) -> str:
        if model is None:
            return "fit returned None (empty ages or fitting exception)"

        optimization_result = model.get("optimization_result")
        if isinstance(optimization_result, dict):
            attempts = optimization_result.get("attempts")
            if isinstance(attempts, list) and len(attempts) > 0:
                selected_attempt_index = optimization_result.get(
                    "selected_attempt_index",
                )
                selected_attempt: dict[str, Any] | None = None

                if isinstance(selected_attempt_index, int) and (
                    0 <= selected_attempt_index < len(attempts)
                ):
                    selected = attempts[selected_attempt_index]
                    if isinstance(selected, dict):
                        selected_attempt = selected

                if selected_attempt is None:
                    first_attempt = attempts[0]
                    if isinstance(first_attempt, dict):
                        selected_attempt = first_attempt

                if selected_attempt is not None:
                    exception_type = selected_attempt.get("exception_type")
                    exception_message = selected_attempt.get("exception")
                    if exception_type and exception_message:
                        return f"{exception_type}: {exception_message}"
                    if exception_message:
                        return str(exception_message)

            if optimization_result.get("converged") is False:
                return "optimizer did not converge"

        return "fit marked unsuccessful without error details"

    failed: list[dict[str, Any]] = []
    for idx, model in enumerate(fitted_models):
        is_failure = model is None or not model.get("success", False)
        if not is_failure:
            continue

        failed_ages = np.asarray(ages_by_fit[idx], dtype=np.float64)
        failed.append(
            {
                "local_index": idx,
                "simulation_index": idx + start_index,
                "population": len(failed_ages),
                "ages": failed_ages,
                "reason": _failure_reason(model),
            },
        )

    failed.sort(key=lambda item: (-int(item["population"]), int(item["local_index"])))
    return failed[:max_items]


def iqr_plotting_subset(
    values_by_name: Mapping[str, NDArray[np.float64] | list[float]],
    whisker_width: float = 1.5,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.bool_]]:
    """Return a joint plotting subset with IQR outliers removed row-wise."""
    arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in values_by_name.items()
    }
    if not arrays:
        return {}, np.array([], dtype=bool)

    lengths = {array.shape[0] for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("All plotting arrays must have the same length")

    length = lengths.pop()
    if length == 0:
        return {name: array.copy() for name, array in arrays.items()}, np.zeros(
            0,
            dtype=bool,
        )

    mask = np.ones(length, dtype=bool)
    for array in arrays.values():
        finite_mask = np.isfinite(array)
        mask &= finite_mask

        finite_values = array[finite_mask]
        if finite_values.size == 0:
            mask &= False
            continue

        q1, q3 = np.percentile(finite_values, [25.0, 75.0])
        iqr = float(q3 - q1)
        if not np.isfinite(iqr) or iqr == 0.0:
            continue

        lower_bound = float(q1 - whisker_width * iqr)
        upper_bound = float(q3 + whisker_width * iqr)
        mask &= (array >= lower_bound) & (array <= upper_bound)

    return {name: array[mask] for name, array in arrays.items()}, mask


def boundary_corrected_kde_on_support(
    samples: NDArray[np.float64],
    support_upper: float,
    n_grid: int = 300,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Reflection KDE on [0, support_upper], normalized on support."""
    samples = np.asarray(samples, dtype=np.float64)
    finite_samples = samples[np.isfinite(samples)]
    if len(finite_samples) < 2 or np.std(finite_samples) == 0.0:
        return None, None

    kde = gaussian_kde(finite_samples)
    x_grid = np.linspace(0.0, float(support_upper), int(n_grid))
    y_kde = kde(x_grid) + kde(-x_grid) + kde(2.0 * support_upper - x_grid)

    normalization = np.trapezoid(y_kde, x_grid)
    if normalization > 0.0 and np.isfinite(normalization):
        y_kde = y_kde / normalization
    return x_grid, y_kde


def boundary_corrected_kde_at_zero(
    samples: NDArray[np.float64],
    n_grid: int = 300,
    upper_padding: float = 1.15,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Reflection KDE for nonnegative support [0, inf), normalized on grid."""
    samples = np.asarray(samples, dtype=np.float64)
    finite_samples = samples[np.isfinite(samples)]
    if len(finite_samples) < 2 or np.std(finite_samples) == 0.0:
        return None, None

    x_max = float(np.max(finite_samples))
    if not np.isfinite(x_max) or x_max <= 0.0:
        x_max = 1.0
    x_max *= float(upper_padding)

    kde = gaussian_kde(finite_samples)
    x_grid = np.linspace(0.0, x_max, int(n_grid))
    y_kde = kde(x_grid) + kde(-x_grid)

    normalization = np.trapezoid(y_kde, x_grid)
    if normalization > 0.0 and np.isfinite(normalization):
        y_kde = y_kde / normalization
    return x_grid, y_kde
