"""Tests for single-epoch analysis helper utilities."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from inf_bp_texts.analysis import (
    BoxPlotStyle,
    KdePlotConfig,
    compute_boxplot_summary,
    compute_parameter_summary,
    create_kde_density_plot,
    create_parameter_box_plot,
    extract_alive_ages,
    extract_rate_arrays,
    summarize_fit_success,
)
from inf_bp_texts.models.birth_death_process import BirthDeathProcess


def _layer_mark_types(spec: dict[str, Any]) -> set[str]:
    """Collect the mark-type strings from every layer of an Altair spec dict."""
    marks: set[str] = set()
    for layer in spec.get("layer", []):
        mark = layer.get("mark")
        if isinstance(mark, dict):
            marks.add(mark["type"])
        elif isinstance(mark, str):
            marks.add(mark)
    return marks


class TestSingleEpochAnalysis:
    """Test single-epoch analysis helpers."""

    def test_extract_alive_ages_sorts_by_spine_flag(self) -> None:
        simulation_df = pd.DataFrame(
            {
                "creation_time": [100.0, 50.0, 10.0],
                "death_time": [np.nan, np.nan, 80.0],
                "is_spine": [False, True, False],
            },
        )
        process = BirthDeathProcess(
            creation_times=[100.0, 50.0],
            epoch_end_times=200.0,
            manuscript_records=simulation_df,
        )

        ages = extract_alive_ages(process, time_span=200.0)
        # Alive rows are creation times [50.0, 100.0] after spine-first sorting.
        assert np.allclose(ages, np.array([150.0, 100.0], dtype=np.float64))

    def test_extract_rate_arrays_returns_empty_for_no_success(self) -> None:
        birth_rates, death_rates = extract_rate_arrays([None, {"success": False}])
        assert birth_rates.size == 0
        assert death_rates.size == 0

    def test_extract_rate_arrays_concatenates_successful_models(self) -> None:
        fitted_models = [
            {
                "success": True,
                "birth_rates": np.array([0.01]),
                "death_rates": np.array([0.002]),
            },
            None,
            {
                "success": True,
                "birth_rates": np.array([0.02]),
                "death_rates": np.array([0.003]),
            },
        ]

        birth_rates, death_rates = extract_rate_arrays(fitted_models)
        assert np.allclose(birth_rates, np.array([0.01, 0.02], dtype=np.float64))
        assert np.allclose(death_rates, np.array([0.002, 0.003], dtype=np.float64))

    def test_compute_parameter_summary_contains_expected_columns(self) -> None:
        summary = compute_parameter_summary(
            birth_rates=np.array([0.01, 0.02, 0.03]),
            death_rates=np.array([0.004, 0.005, 0.006]),
            true_birth_rate=0.02,
            true_death_rate=0.005,
        )

        assert list(summary["Parameter"]) == ["Birth Rate", "Death Rate"]
        assert set(summary.columns) == {
            "Parameter",
            "True Value",
            "Mean",
            "Median",
            "Std",
            "Min",
            "Max",
            "25%",
            "75%",
        }

    def test_compute_parameter_summary_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="Cannot summarize empty rate arrays"):
            compute_parameter_summary(
                birth_rates=np.array([]),
                death_rates=np.array([0.1]),
                true_birth_rate=0.1,
                true_death_rate=0.1,
            )

    def test_summarize_fit_success_counts_and_rate(self) -> None:
        fitted_models: list[dict[str, Any] | None] = [
            {
                "success": True,
                "birth_rates": np.array([0.01]),
                "death_rates": np.array([0.001]),
            },
            {"success": False},
            None,
        ]
        summary = summarize_fit_success(fitted_models, total_models=3)
        assert summary["successful"] == 1
        assert summary["failed"] == 2
        assert summary["success_rate"] == pytest.approx(33.3333333333)
        assert summary["warn_low_success"]

    def test_create_kde_density_plot_layers_density_and_true_rule(self) -> None:
        chart = create_kde_density_plot(
            rates=np.array([0.001, 0.002, 0.003, 0.004]),
            true_rate=0.0025,
            config=KdePlotConfig(
                estimate_axis_label="rate",
                true_label="True λ",
                density_color="#000000",
                true_line_color="#ff0000",
            ),
        )
        marks = _layer_mark_types(chart.to_dict())
        # Density curve, dashed true-value rule, and the true-value text label.
        assert {"line", "rule", "text"} <= marks

    def test_create_parameter_box_plot_layers_box_and_markers(self) -> None:
        chart = create_parameter_box_plot(
            birth_rates=np.array([0.01, 0.02, 0.03]),
            death_rates=np.array([0.004, 0.005, 0.006]),
            true_rates=(0.02, 0.005),
            style=BoxPlotStyle(
                base_color="#111111",
                true_color="#222222",
                mean_color="#333333",
                median_color="#444444",
            ),
        )
        marks = _layer_mark_types(chart.to_dict())
        # Box plot plus overlaid true/mean/median point markers.
        assert "boxplot" in marks
        assert "point" in marks

    def test_compute_boxplot_summary_matches_expected_shape(self) -> None:
        summary = compute_boxplot_summary(
            birth_rates=np.array([0.01, 0.02, 0.03]),
            death_rates=np.array([0.004, 0.005, 0.006]),
            true_birth_rate=0.02,
            true_death_rate=0.005,
        )
        assert summary.shape[0] == 2
        assert "RMSE" in summary.columns
