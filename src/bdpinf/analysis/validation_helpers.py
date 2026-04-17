"""Shared helpers for validation notebooks and scripts."""

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
    """Return metadata for the first failed fits and their simulation vectors."""
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
            },
        )
        if len(failed) >= max_items:
            break
    return failed


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
