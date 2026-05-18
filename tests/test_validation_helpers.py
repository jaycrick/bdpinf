"""Tests for shared validation helpers."""

# ruff: noqa: S101

import numpy as np

import inf_bp_texts.analysis.validation_helpers as validation_helpers_module
from inf_bp_texts.analysis import (
    FitModelConfig,
    boundary_corrected_kde_at_zero,
    boundary_corrected_kde_on_support,
    first_failed_fits,
    fit_single_model,
    iqr_plotting_subset,
)
from inf_bp_texts.analysis import validation_helpers as vh


def test_safe_mean_and_var():
    arr = np.array([1.0, 2.0, np.nan, np.inf, 3.0])
    assert vh.safe_mean(arr) == 2.0
    assert round(vh.safe_var(arr), 6) == round(np.var([1.0, 2.0, 3.0], ddof=1), 6)

    empty = np.array([np.nan, np.inf])
    assert np.isnan(vh.safe_mean(empty))
    assert np.isnan(vh.safe_var(empty))


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
        {
            "success": False,
            "optimization_result": {
                "converged": False,
                "selected_attempt_index": 0,
                "attempts": [
                    {
                        "exception_type": "RuntimeError",
                        "exception": "boom",
                    },
                ],
            },
        },
        {"success": True},
        {
            "success": False,
            "optimization_result": {
                "converged": False,
                "selected_attempt_index": 0,
                "attempts": [
                    {
                        "exception_type": None,
                        "exception": None,
                    },
                ],
            },
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


def test_fit_model_config_defaults_to_unbias_true() -> None:
    assert FitModelConfig(times=1000.0).unbias is True


def test_fit_single_model_forwards_unbias_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_fit_birth_death_processes(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(
        validation_helpers_module,
        "fit_birth_death_processes",
        fake_fit_birth_death_processes,
    )

    result = fit_single_model(
        np.array([1.0, 2.0], dtype=np.float64),
        FitModelConfig(times=10.0),
    )

    assert result == {"success": True}
    assert captured["unbias"] is True


def test_iqr_plotting_subset_removes_row_outliers() -> None:
    plotting_data, mask = iqr_plotting_subset(
        {
            "x": np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float64),
            "y": np.array([10.0, 11.0, 99.0, 12.0], dtype=np.float64),
        },
    )

    assert mask.tolist() == [True, True, False, False]
    assert plotting_data["x"].tolist() == [1.0, 2.0]
    assert plotting_data["y"].tolist() == [10.0, 11.0]


def test_iqr_plotting_subset_keeps_constant_series() -> None:
    plotting_data, mask = iqr_plotting_subset(
        {
            "x": np.array([5.0, 5.0, 5.0], dtype=np.float64),
            "y": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        },
    )

    assert mask.tolist() == [True, True, True]
    assert plotting_data["x"].tolist() == [5.0, 5.0, 5.0]
    assert plotting_data["y"].tolist() == [1.0, 2.0, 3.0]
