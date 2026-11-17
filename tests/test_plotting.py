"""Tests for visualization plotting utilities."""

from collections.abc import Callable
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pytest

import bdpinf.inference.likelihood
from bdpinf.inference import FitResult
from bdpinf.visualization.plotting import (
    convergence_chart,
    rate_ci_panel,
    save_multi_epoch_birth_time_figure,
    save_multi_epoch_convergence_figure,
    save_multi_epoch_fit_figure,
)

_DEFAULT_NLL_HISTORY = [10.0, 8.0, 7.5]


def _fit_result(
    make_fit_result: Callable[..., FitResult],
    *,
    success: bool = True,
    nll_history: list[float] | None = None,
) -> FitResult:
    """Build a two-epoch fit result with the given convergence history."""
    return make_fit_result(
        [0.02, 0.03],
        [0.01, 0.015],
        times=[5.0, 10.0],
        converged=success,
        nll_history=_DEFAULT_NLL_HISTORY if nll_history is None else nll_history,
    )


class TestConvergenceChart:
    """Test the log-likelihood convergence line chart."""

    def test_returns_chart_with_expected_encodings(self) -> None:
        chart = convergence_chart([-10.0, -8.0, -7.5], title="Convergence")

        assert isinstance(chart, alt.Chart)
        spec = chart.to_dict()
        assert spec["encoding"]["x"]["field"] == "Iteration"
        assert spec["encoding"]["y"]["field"] == "Log-likelihood"
        assert spec["title"] == "Convergence"

    def test_iteration_axis_is_one_indexed(self) -> None:
        chart = convergence_chart([1.0, 2.0, 3.0], title="test")

        datasets = chart.to_dict()["datasets"]
        rows = next(iter(datasets.values()))
        assert [row["Iteration"] for row in rows] == [1, 2, 3]


class TestSaveMultiEpochConvergenceFigure:
    """Test the convergence-trace saver."""

    def test_writes_chart_for_successful_result(
        self, make_fit_result: Callable[..., FitResult], tmp_path: Path
    ) -> None:
        result = _fit_result(make_fit_result, nll_history=[10.0, 8.0, 7.5])

        save_multi_epoch_convergence_figure(
            "Test Case", result, "convergence_slug", figures_dir=tmp_path
        )

        assert (tmp_path / "convergence_slug.pdf").exists()

    def test_skips_unsuccessful_result(
        self, make_fit_result: Callable[..., FitResult], tmp_path: Path
    ) -> None:
        result = _fit_result(make_fit_result, success=False)

        save_multi_epoch_convergence_figure(
            "Test Case", result, "convergence_slug", figures_dir=tmp_path
        )

        assert not (tmp_path / "convergence_slug.pdf").exists()

    def test_skips_result_with_no_history(
        self, make_fit_result: Callable[..., FitResult], tmp_path: Path
    ) -> None:
        result = _fit_result(make_fit_result, nll_history=[])

        save_multi_epoch_convergence_figure(
            "Test Case", result, "convergence_slug", figures_dir=tmp_path
        )

        assert not (tmp_path / "convergence_slug.pdf").exists()


class TestSaveMultiEpochBirthTimeFigure:
    """Test the birth-time histogram saver."""

    def test_writes_chart_for_successful_result(
        self, make_fit_result: Callable[..., FitResult], tmp_path: Path
    ) -> None:
        result = _fit_result(
            make_fit_result,
        )
        epochs = [(0.0, 100.0, "Epoch 1"), (100.0, 200.0, "Epoch 2")]
        birth_years = np.array([20.0, 50.0, 80.0, 150.0])
        times = np.array([100.0, 200.0], dtype=np.float64)

        save_multi_epoch_birth_time_figure(
            "Test Case",
            result,
            epochs,
            birth_years,
            times,
            output_slug="birth_time_slug",
            figures_dir=tmp_path,
        )

        assert (tmp_path / "birth_time_slug.pdf").exists()

    def test_overlay_failure_still_saves_the_histogram(
        self,
        make_fit_result: Callable[..., FitResult],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        def invalid(*_args: object, **_kwargs: object) -> None:
            raise ValueError("invalid rates")

        monkeypatch.setattr(
            bdpinf.inference.likelihood,
            "calculate_unknown_spine_mixture_age_density_rates",
            invalid,
        )

        with caplog.at_level("WARNING"):
            save_multi_epoch_birth_time_figure(
                "Test Case",
                _fit_result(make_fit_result),
                [(0.0, 100.0, "Epoch 1"), (100.0, 200.0, "Epoch 2")],
                np.array([20.0, 50.0, 80.0, 150.0]),
                np.array([100.0, 200.0], dtype=np.float64),
                output_slug="birth_time_slug",
                figures_dir=tmp_path,
            )

        assert (tmp_path / "birth_time_slug.pdf").exists()
        assert "invalid rates" in caplog.text

    def test_type_error_in_overlay_propagates(
        self,
        make_fit_result: Callable[..., FitResult],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def broken(*_args: object, **_kwargs: object) -> None:
            raise TypeError("programming error")

        monkeypatch.setattr(
            bdpinf.inference.likelihood,
            "calculate_unknown_spine_mixture_age_density_rates",
            broken,
        )

        with pytest.raises(TypeError, match="programming error"):
            save_multi_epoch_birth_time_figure(
                "Test Case",
                _fit_result(make_fit_result),
                [(0.0, 100.0, "Epoch 1"), (100.0, 200.0, "Epoch 2")],
                np.array([20.0, 50.0, 80.0, 150.0]),
                np.array([100.0, 200.0], dtype=np.float64),
                output_slug="birth_time_slug",
                figures_dir=tmp_path,
            )

    def test_skips_unsuccessful_result(
        self, make_fit_result: Callable[..., FitResult], tmp_path: Path
    ) -> None:
        result = _fit_result(make_fit_result, success=False)
        epochs = [(0.0, 100.0, "Epoch 1")]
        birth_years = np.array([20.0])
        times = np.array([100.0], dtype=np.float64)

        save_multi_epoch_birth_time_figure(
            "Test Case",
            result,
            epochs,
            birth_years,
            times,
            output_slug="birth_time_slug",
            figures_dir=tmp_path,
        )

        assert not (tmp_path / "birth_time_slug.pdf").exists()

    def test_skips_when_no_observed_years(
        self, make_fit_result: Callable[..., FitResult], tmp_path: Path
    ) -> None:
        result = _fit_result(
            make_fit_result,
        )
        epochs = [(0.0, 100.0, "Epoch 1")]
        birth_years = np.array([], dtype=float)
        times = np.array([100.0], dtype=np.float64)

        save_multi_epoch_birth_time_figure(
            "Test Case",
            result,
            epochs,
            birth_years,
            times,
            output_slug="birth_time_slug",
            figures_dir=tmp_path,
        )

        assert not (tmp_path / "birth_time_slug.pdf").exists()


class TestRateCiPanel:
    """Test the dot-and-whisker rate/CI panel builder."""

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Period": ["Epoch 1", "Epoch 2", "Epoch 3"],
                "rate": [0.001, 0.05, 0.5],
                "lower": [0.0005, 0.02, 0.2],
                "upper": [0.002, 0.08, 0.8],
            }
        )

    def test_gridlines_enabled_on_linear_rate_axis(self) -> None:
        chart = rate_ci_panel(
            self._frame(), "Period", "rate", "lower", "upper", color="#4477AA"
        )

        spec = chart.to_dict()
        assert spec["layer"][0]["encoding"]["x"]["axis"]["grid"] is True

    def test_log_scale_ticks_are_powers_of_ten(self) -> None:
        chart = rate_ci_panel(
            self._frame(),
            "Period",
            "rate",
            "lower",
            "upper",
            color="#4477AA",
            log_scale=True,
        )

        spec = chart.to_dict()
        axis = spec["layer"][0]["encoding"]["x"]["axis"]
        assert axis["grid"] is True
        # Data spans 0.0005 .. 0.8 -> decades 1e-4 .. 1e0.
        assert axis["values"] == [1e-4, 1e-3, 1e-2, 1e-1, 1e0]

    def test_vertical_orient_places_rate_on_y_axis(self) -> None:
        chart = rate_ci_panel(
            self._frame(),
            "Period",
            "rate",
            "lower",
            "upper",
            color="#4477AA",
            orient="v",
        )

        spec = chart.to_dict()
        assert spec["layer"][0]["encoding"]["y"]["field"] == "rate"
        assert spec["layer"][0]["encoding"]["y"]["axis"]["grid"] is True


class TestSaveMultiEpochFitFigure:
    """Test the per-epoch birth/death rate bar-chart saver."""

    def test_writes_chart_for_successful_result(
        self, make_fit_result: Callable[..., FitResult], tmp_path: Path
    ) -> None:
        result = _fit_result(
            make_fit_result,
        )
        epochs = [(0.0, 100.0, "Epoch 1"), (100.0, 200.0, "Epoch 2")]

        save_multi_epoch_fit_figure(
            "Test Case", result, epochs, "fit_slug", figures_dir=tmp_path
        )

        assert (tmp_path / "fit_slug.pdf").exists()

    def test_skips_unsuccessful_result(
        self, make_fit_result: Callable[..., FitResult], tmp_path: Path
    ) -> None:
        result = _fit_result(make_fit_result, success=False)
        epochs = [(0.0, 100.0, "Epoch 1")]

        save_multi_epoch_fit_figure(
            "Test Case", result, epochs, "fit_slug", figures_dir=tmp_path
        )

        assert not (tmp_path / "fit_slug.pdf").exists()
