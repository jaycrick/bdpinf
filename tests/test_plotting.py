"""Tests for visualization plotting utilities."""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from inf_bp_texts.visualization.plotting import (
    convergence_chart,
    save_multi_epoch_birth_time_figure,
    save_multi_epoch_convergence_figure,
    save_multi_epoch_fit_figure,
)

_DEFAULT_NLL_HISTORY = [10.0, 8.0, 7.5]


def _fit_result(
    *, success: bool = True, nll_history: list[float] | None = None
) -> dict:
    """Build a minimal fit-result dict shaped like fit_birth_death_processes' output."""
    if nll_history is None:
        nll_history = _DEFAULT_NLL_HISTORY
    return {
        "success": success,
        "birth_rates": [0.02, 0.03],
        "death_rates": [0.01, 0.015],
        "optimization_result": {
            "attempts": [{"nll_history": nll_history}],
            "selected_attempt_index": 0,
        },
    }


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

    def test_writes_chart_for_successful_result(self, tmp_path: Path) -> None:
        result = _fit_result(nll_history=[10.0, 8.0, 7.5])

        save_multi_epoch_convergence_figure(
            "Test Case", result, "convergence_slug", figures_dir=tmp_path
        )

        assert (tmp_path / "convergence_slug.pdf").exists()

    def test_skips_unsuccessful_result(self, tmp_path: Path) -> None:
        result = _fit_result(success=False)

        save_multi_epoch_convergence_figure(
            "Test Case", result, "convergence_slug", figures_dir=tmp_path
        )

        assert not (tmp_path / "convergence_slug.pdf").exists()

    def test_skips_result_with_no_history(self, tmp_path: Path) -> None:
        result = _fit_result(nll_history=[])

        save_multi_epoch_convergence_figure(
            "Test Case", result, "convergence_slug", figures_dir=tmp_path
        )

        assert not (tmp_path / "convergence_slug.pdf").exists()


class TestSaveMultiEpochBirthTimeFigure:
    """Test the manuscript birth-time histogram saver."""

    def test_writes_chart_for_successful_result(self, tmp_path: Path) -> None:
        result = _fit_result()
        epochs = [(0.0, 100.0, "Epoch 1"), (100.0, 200.0, "Epoch 2")]
        manuscripts_df = pd.DataFrame({"parsed_year": [20.0, 50.0, 80.0, 150.0]})
        times = np.array([100.0, 200.0], dtype=np.float64)

        save_multi_epoch_birth_time_figure(
            "Test Case",
            result,
            epochs,
            manuscripts_df,
            times,
            "birth_time_slug",
            figures_dir=tmp_path,
        )

        assert (tmp_path / "birth_time_slug.pdf").exists()

    def test_skips_unsuccessful_result(self, tmp_path: Path) -> None:
        result = _fit_result(success=False)
        epochs = [(0.0, 100.0, "Epoch 1")]
        manuscripts_df = pd.DataFrame({"parsed_year": [20.0]})
        times = np.array([100.0], dtype=np.float64)

        save_multi_epoch_birth_time_figure(
            "Test Case",
            result,
            epochs,
            manuscripts_df,
            times,
            "birth_time_slug",
            figures_dir=tmp_path,
        )

        assert not (tmp_path / "birth_time_slug.pdf").exists()

    def test_skips_when_no_observed_years(self, tmp_path: Path) -> None:
        result = _fit_result()
        epochs = [(0.0, 100.0, "Epoch 1")]
        manuscripts_df = pd.DataFrame({"parsed_year": pd.Series([], dtype=float)})
        times = np.array([100.0], dtype=np.float64)

        save_multi_epoch_birth_time_figure(
            "Test Case",
            result,
            epochs,
            manuscripts_df,
            times,
            "birth_time_slug",
            figures_dir=tmp_path,
        )

        assert not (tmp_path / "birth_time_slug.pdf").exists()


class TestSaveMultiEpochFitFigure:
    """Test the per-epoch birth/death rate bar-chart saver."""

    def test_writes_chart_for_successful_result(self, tmp_path: Path) -> None:
        result = _fit_result()
        epochs = [(0.0, 100.0, "Epoch 1"), (100.0, 200.0, "Epoch 2")]

        save_multi_epoch_fit_figure(
            "Test Case", result, epochs, "fit_slug", figures_dir=tmp_path
        )

        assert (tmp_path / "fit_slug.pdf").exists()

    def test_skips_unsuccessful_result(self, tmp_path: Path) -> None:
        result = _fit_result(success=False)
        epochs = [(0.0, 100.0, "Epoch 1")]

        save_multi_epoch_fit_figure(
            "Test Case", result, epochs, "fit_slug", figures_dir=tmp_path
        )

        assert not (tmp_path / "fit_slug.pdf").exists()
