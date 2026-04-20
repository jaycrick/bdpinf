"""Tests for shared validation helpers."""

# ruff: noqa: S101

import numpy as np

from inf_bp_texts.analysis import (
    FitModelConfig,
    boundary_corrected_kde_at_zero,
    boundary_corrected_kde_on_support,
    first_failed_fits,
    fit_single_model,
)


def test_boundary_corrected_kde_on_support_stays_on_domain() -> None:
    samples = np.array([10.0, 20.0, 25.0, 30.0, 35.0], dtype=np.float64)
    x_grid, y_kde = boundary_corrected_kde_on_support(samples, support_upper=100.0)

    assert x_grid is not None
    assert y_kde is not None
    assert np.all(x_grid >= 0.0)
    assert np.all(x_grid <= 100.0)
    assert np.all(y_kde >= 0.0)


def test_boundary_corrected_kde_at_zero_starts_at_zero() -> None:
    samples = np.array([0.001, 0.002, 0.004, 0.006], dtype=np.float64)
    x_grid, y_kde = boundary_corrected_kde_at_zero(samples)

    assert x_grid is not None
    assert y_kde is not None
    assert x_grid[0] == 0.0
    assert np.all(x_grid >= 0.0)
    assert np.all(y_kde >= 0.0)


def test_first_failed_fits_returns_original_indices() -> None:
    fitted_models = [
        {"success": True},
        None,
        {"success": False, "error_type": "RuntimeError", "error": "boom"},
        {"success": True},
        {
            "success": False,
            "optimization_result": {"converged": False},
        },
    ]
    ages = [
        np.array([1.0]),
        np.array([2.0, 3.0]),
        np.array([4.0]),
        np.array([5.0]),
        np.array([6.0, 7.0, 8.0]),
    ]

    failed = first_failed_fits(
        fitted_models=fitted_models,
        ages_by_fit=ages,
        start_index=10,
        max_items=3,
    )

    assert len(failed) == 3
    assert [item["simulation_index"] for item in failed] == [14, 11, 12]
    assert [item["population"] for item in failed] == [3, 2, 1]
    assert [item["reason"] for item in failed] == [
        "optimizer did not converge",
        "fit returned None (empty ages or fitting exception)",
        "RuntimeError: boom",
    ]


def test_fit_single_model_handles_empty_input() -> None:
    config = FitModelConfig(times=1000.0)
    result = fit_single_model(np.array([], dtype=np.float64), config)
    assert result is None
