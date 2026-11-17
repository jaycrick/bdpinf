"""Tests for shared validation helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bdpinf.analysis.validation_helpers as validation_helpers_module
from bdpinf.analysis import (
    FitModelConfig,
    boundary_corrected_kde_at_zero,
    boundary_corrected_kde_on_support,
    build_hist_bins,
    fit_single_model,
    has_ages_in_each_epoch,
    iqr_plotting_subset,
    robust_y_max,
    theoretical_age_density,
)
from bdpinf.analysis import validation_helpers as vh
from bdpinf.inference import Constraints
from bdpinf.models.math_nested import (
    nested_terms,
    non_spine_log_density,
)


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

    def test_weighted_kde_on_support_stays_on_domain(self) -> None:
        samples = np.array([10.0, 20.0, 25.0, 30.0, 35.0], dtype=np.float64)
        weights = np.array([0.5, 0.5, 1.0, 1.0, 2.0], dtype=np.float64)
        x_grid, y_kde = boundary_corrected_kde_on_support(
            samples, support_upper=100.0, weights=weights
        )
        assert x_grid is not None
        assert y_kde is not None
        assert np.all(x_grid >= 0.0)
        assert np.all(x_grid <= 100.0)
        assert np.all(y_kde >= 0.0)
        # density should integrate to ~1 on support
        integral = float(np.trapezoid(y_kde, x_grid))
        assert abs(integral - 1.0) < 0.01

    def test_weighted_kde_at_zero_nonnegative(self) -> None:
        samples = np.array([0.001, 0.002, 0.004, 0.006], dtype=np.float64)
        weights = np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float64)
        x_grid, y_kde = boundary_corrected_kde_at_zero(samples, weights=weights)
        assert x_grid is not None
        assert y_kde is not None
        assert x_grid[0] == 0.0
        assert np.all(y_kde >= 0.0)

    def test_uniform_weights_reproduce_unweighted_on_support(self) -> None:
        samples = np.array([10.0, 20.0, 25.0, 30.0, 35.0], dtype=np.float64)
        uniform_weights = np.ones(len(samples), dtype=np.float64)
        x_unweighted, y_unweighted = boundary_corrected_kde_on_support(
            samples, support_upper=100.0
        )
        x_weighted, y_weighted = boundary_corrected_kde_on_support(
            samples, support_upper=100.0, weights=uniform_weights
        )
        assert x_unweighted is not None and y_unweighted is not None
        assert x_weighted is not None and y_weighted is not None
        np.testing.assert_allclose(y_weighted, y_unweighted, rtol=1e-10)

    def test_uniform_weights_reproduce_unweighted_at_zero(self) -> None:
        samples = np.array([0.001, 0.002, 0.004, 0.006], dtype=np.float64)
        uniform_weights = np.ones(len(samples), dtype=np.float64)
        x_unweighted, y_unweighted = boundary_corrected_kde_at_zero(samples)
        x_weighted, y_weighted = boundary_corrected_kde_at_zero(
            samples, weights=uniform_weights
        )
        assert x_unweighted is not None and y_unweighted is not None
        assert x_weighted is not None and y_weighted is not None
        np.testing.assert_allclose(y_weighted, y_unweighted, rtol=1e-10)

    def test_fit_single_model_handles_empty_input(self) -> None:
        config = FitModelConfig(times=1000.0)
        result = fit_single_model(np.array([], dtype=np.float64), config)
        assert result is None

    def test_fit_model_config_defaults_to_condition_on_survival(self) -> None:
        assert FitModelConfig(times=1000.0).condition_on_survival is True

    def test_fit_single_model_forwards_condition_on_survival_default(
        self, monkeypatch, make_fit_result
    ) -> None:
        captured: dict[str, object] = {}

        def fake_fit(*_args, **kwargs):
            captured.update(kwargs)
            return make_fit_result([0.5], [0.1])

        monkeypatch.setattr(validation_helpers_module, "fit", fake_fit)

        result = fit_single_model(
            np.array([1.0, 2.0], dtype=np.float64),
            FitModelConfig(times=10.0),
        )

        assert result is not None
        assert result.success
        assert captured["condition_on_survival"] is True

    def test_fit_single_model_forwards_optional_settings(
        self, monkeypatch, make_fit_result
    ) -> None:
        captured: dict[str, object] = {}

        def fake_fit(*_args, **kwargs):
            captured.update(kwargs)
            return make_fit_result([0.5], [0.1])

        monkeypatch.setattr(validation_helpers_module, "fit", fake_fit)
        constraints = Constraints(fixed={"birth.v0": 1e-12})

        fit_single_model(
            np.array([1.0, 2.0], dtype=np.float64),
            FitModelConfig(
                times=10.0,
                method="L-BFGS-B",
                n_starts=3,
                random_seed=11,
                constraints=constraints,
                initial_birth_rates=np.array([0.5]),
            ),
        )

        assert captured["method"] == "L-BFGS-B"
        assert captured["n_starts"] == 3
        assert captured["seed"] == 11
        assert captured["constraints"] is constraints
        assert captured["init"] == {"birth.v0": 0.5}

    def test_fit_single_model_logs_and_returns_none_on_exception(
        self, monkeypatch, caplog
    ) -> None:
        def boom(*_args, **_kwargs):
            raise RuntimeError("optimizer blew up")

        monkeypatch.setattr(validation_helpers_module, "fit", boom)

        with caplog.at_level("WARNING"):
            result = fit_single_model(
                np.array([1.0], dtype=np.float64), FitModelConfig(times=10.0)
            )

        assert result is None
        assert "optimizer blew up" in caplog.text

    def test_fit_single_model_handles_expected_fit_errors(
        self, monkeypatch, caplog
    ) -> None:
        def invalid(*_args, **_kwargs):
            raise ValueError("no finite start")

        monkeypatch.setattr(validation_helpers_module, "fit", invalid)

        with caplog.at_level("WARNING"):
            result = fit_single_model(
                np.array([1.0], dtype=np.float64), FitModelConfig(times=10.0)
            )

        assert result is None
        assert "no finite start" in caplog.text

    def test_fit_single_model_propagates_type_error(self, monkeypatch) -> None:
        def programming_error(*_args, **_kwargs):
            raise TypeError("unexpected keyword argument")

        monkeypatch.setattr(validation_helpers_module, "fit", programming_error)

        with pytest.raises(TypeError, match="unexpected keyword"):
            fit_single_model(
                np.array([1.0], dtype=np.float64), FitModelConfig(times=10.0)
            )

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
        birth_rate = jnp.array([0.1], dtype=jnp.float64)
        death_rate = jnp.array([0.05], dtype=jnp.float64)
        T = 100.0
        times = jnp.array([T], dtype=jnp.float64)

        def log_lik_fn(age: jax.Array) -> jax.Array:
            return non_spine_log_density(
                nested_terms(birth_rate, death_rate, times), age
            ).sum()

        x_grid, y_vals = theoretical_age_density(
            log_lik_fn, support_upper=T, n_grid=1000
        )
        integral = float(np.trapezoid(y_vals, x_grid))
        assert 0.5 < integral < 1.5
