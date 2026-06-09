"""Tests for shared validation helpers."""

import numpy as np
import pytest
import torch

import inf_bp_texts.analysis.validation_helpers as validation_helpers_module
from inf_bp_texts.analysis import (
    FitModelConfig,
    boundary_corrected_kde_at_zero,
    boundary_corrected_kde_on_support,
    build_hist_bins,
    extract_epoch_parameter_arrays,
    first_failed_fits,
    fit_single_model,
    get_selected_attempt,
    has_ages_in_each_epoch,
    iqr_plotting_subset,
    robust_y_max,
    theoretical_age_density,
)
from inf_bp_texts.analysis import validation_helpers as vh
from inf_bp_texts.models.mathematical_functions import calculate_non_spine_ages_log_lik


class TestValidationHelpers:
    """Test shared validation helper functions."""

    def test_safe_mean_and_var(self):
        arr = np.array([1.0, 2.0, np.nan, np.inf, 3.0])
        assert vh.safe_mean(arr) == 2.0
        assert round(vh.safe_var(arr), 6) == round(np.var([1.0, 2.0, 3.0], ddof=1), 6)

        empty = np.array([np.nan, np.inf])
        assert np.isnan(vh.safe_mean(empty))
        assert np.isnan(vh.safe_var(empty))

    def test_boundary_corrected_kde_on_support_stays_on_domain(self) -> None:
        samples = np.array([10.0, 20.0, 25.0, 30.0, 35.0], dtype=np.float64)
        x_grid, y_kde = boundary_corrected_kde_on_support(samples, support_upper=100.0)

        assert x_grid is not None
        assert y_kde is not None
        assert np.all(x_grid >= 0.0)
        assert np.all(x_grid <= 100.0)
        assert np.all(y_kde >= 0.0)

    def test_boundary_corrected_kde_at_zero_starts_at_zero(self) -> None:
        samples = np.array([0.001, 0.002, 0.004, 0.006], dtype=np.float64)
        x_grid, y_kde = boundary_corrected_kde_at_zero(samples)

        assert x_grid is not None
        assert y_kde is not None
        assert x_grid[0] == 0.0
        assert np.all(x_grid >= 0.0)
        assert np.all(y_kde >= 0.0)

    def test_first_failed_fits_returns_original_indices(self) -> None:
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

    def test_fit_single_model_handles_empty_input(self) -> None:
        config = FitModelConfig(times=1000.0)
        result = fit_single_model(np.array([], dtype=np.float64), config)
        assert result is None

    def test_fit_model_config_defaults_to_unbias_true(self) -> None:
        assert FitModelConfig(times=1000.0).unbias is True

    def test_fit_single_model_forwards_unbias_default(self, monkeypatch) -> None:
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

    def test_iqr_plotting_subset_removes_row_outliers(self) -> None:
        plotting_data, mask = iqr_plotting_subset(
            {
                "x": np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float64),
                "y": np.array([10.0, 11.0, 99.0, 12.0], dtype=np.float64),
            },
        )

        assert mask.tolist() == [True, True, False, False]
        assert plotting_data["x"].tolist() == [1.0, 2.0]
        assert plotting_data["y"].tolist() == [10.0, 11.0]

    def test_iqr_plotting_subset_keeps_constant_series(self) -> None:
        plotting_data, mask = iqr_plotting_subset(
            {
                "x": np.array([5.0, 5.0, 5.0], dtype=np.float64),
                "y": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            },
        )

        assert mask.tolist() == [True, True, True]
        assert plotting_data["x"].tolist() == [5.0, 5.0, 5.0]
        assert plotting_data["y"].tolist() == [1.0, 2.0, 3.0]


class TestNewValidationHelpers:
    """Test recently extracted validation helper functions."""

    def test_has_ages_in_each_epoch_true_when_all_covered(self) -> None:
        ages = np.array([100.0, 400.0, 900.0])
        times = np.array([300.0, 1000.0])
        assert has_ages_in_each_epoch(ages, times) is True

    def test_has_ages_in_each_epoch_false_when_first_epoch_empty(self) -> None:
        ages = np.array([400.0, 900.0])
        times = np.array([300.0, 1000.0])
        assert has_ages_in_each_epoch(ages, times) is False

    def test_has_ages_in_each_epoch_false_when_later_epoch_empty(self) -> None:
        ages = np.array([100.0, 200.0])
        times = np.array([300.0, 1000.0])
        assert has_ages_in_each_epoch(ages, times) is False

    def test_extract_epoch_parameter_arrays_keys_present(self) -> None:
        models = [
            {
                "success": True,
                "birth_rates": np.array([0.05, 0.03]),
                "death_rates": np.array([0.02, 0.01]),
            },
        ]
        result = extract_epoch_parameter_arrays(models)
        assert set(result.keys()) == {
            "epoch_1_birth",
            "epoch_1_death",
            "epoch_2_birth",
            "epoch_2_death",
        }
        assert result["epoch_1_birth"][0] == pytest.approx(0.05)
        assert result["epoch_2_death"][0] == pytest.approx(0.01)

    def test_extract_epoch_parameter_arrays_excludes_none_and_failures(self) -> None:
        models = [
            None,
            {
                "success": False,
                "birth_rates": np.array([0.05, 0.03]),
                "death_rates": np.array([0.02, 0.01]),
            },
            {
                "success": True,
                "birth_rates": np.array([0.04, 0.02]),
                "death_rates": np.array([0.01, 0.005]),
            },
        ]
        result = extract_epoch_parameter_arrays(models)
        assert len(result["epoch_1_birth"]) == 1
        assert result["epoch_1_birth"][0] == pytest.approx(0.04)

    def test_build_hist_bins_covers_full_range(self) -> None:
        result = build_hist_bins(
            np.array([0.0, 500.0, 1000.0]),
            np.array([200.0, 800.0]),
        )
        assert result["start"] == 0.0
        assert result["end"] > 1000.0
        assert result["size"] > 0.0

    def test_build_hist_bins_empty_inputs_return_defaults(self) -> None:
        result = build_hist_bins(np.array([]), np.array([]))
        assert result["start"] == 0.0
        assert result["end"] == pytest.approx(1.0)
        assert result["size"] == pytest.approx(1.0 / 30)

    def test_robust_y_max_ignores_extreme_outlier(self) -> None:
        typical = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20, dtype=np.float64)
        with_outlier = np.concatenate([typical, np.array([1e9])])
        y_max = robust_y_max([with_outlier])
        assert y_max < 100.0

    def test_robust_y_max_falls_back_to_one_for_empty_input(self) -> None:
        assert robust_y_max([]) == pytest.approx(1.0)
        assert robust_y_max([None]) == pytest.approx(1.0)

    def test_theoretical_age_density_integrates_to_approximately_one(self) -> None:
        birth_rate = torch.tensor([0.1], dtype=torch.float64)
        death_rate = torch.tensor([0.05], dtype=torch.float64)
        T = 100.0
        times = torch.tensor([T], dtype=torch.float64)

        def log_lik_fn(age: torch.Tensor) -> torch.Tensor:
            return calculate_non_spine_ages_log_lik(birth_rate, death_rate, age, times)

        x_grid, y_vals = theoretical_age_density(
            log_lik_fn, support_upper=T, n_grid=1000
        )
        integral = float(np.trapezoid(y_vals, x_grid))
        assert 0.5 < integral < 1.5

    def test_get_selected_attempt_returns_indexed_attempt(self) -> None:
        result = {
            "optimization_result": {
                "selected_attempt_index": 1,
                "attempts": [{"nll": 1.0}, {"nll": 2.0}],
            }
        }
        attempt = get_selected_attempt(result)
        assert attempt is not None
        assert attempt["nll"] == pytest.approx(2.0)

    def test_get_selected_attempt_falls_back_to_first_on_invalid_index(self) -> None:
        result = {
            "optimization_result": {
                "selected_attempt_index": 99,
                "attempts": [{"nll": 1.0}, {"nll": 2.0}],
            }
        }
        attempt = get_selected_attempt(result)
        assert attempt is not None
        assert attempt["nll"] == pytest.approx(1.0)

    def test_get_selected_attempt_returns_none_when_no_attempts(self) -> None:
        result = {
            "optimization_result": {
                "selected_attempt_index": 0,
                "attempts": [],
            }
        }
        assert get_selected_attempt(result) is None

    def test_get_selected_attempt_returns_none_when_key_missing(self) -> None:
        assert get_selected_attempt({}) is None
