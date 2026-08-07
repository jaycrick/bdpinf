"""Golden-value and structural tests for compute_consistency_bands."""

import numpy as np
import pandas as pd
import pytest

from bdpinf.visualization.plotting import (
    compute_consistency_bands,
    plot_parameter_consistency,
)

_REGIMES = [{"name": "A", "birth_rate": 0.02}, {"name": "B", "birth_rate": 0.05}]
_T_VALUES = [100, 1000, 10000]


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic summary/results frames for two regimes over three T."""
    rng = np.random.default_rng(0)
    summary_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for regime in _REGIMES:
        for t in _T_VALUES:
            reps = rng.normal(regime["birth_rate"], 0.005, 30)
            raw_rows.extend(
                {"regime": regime["name"], "T": t, "success": True, "birth_estimate": r}
                for r in reps
            )
            summary_rows.append(
                {
                    "regime": regime["name"],
                    "T": t,
                    "birth_mean": float(np.mean(reps)),
                    "birth_var": float(np.var(reps, ddof=1)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(raw_rows)


def _bands() -> pd.DataFrame:
    summary_df, results_df = _synthetic_inputs()
    return compute_consistency_bands(
        summary_df,
        results_df,
        _REGIMES,
        bootstrap_samples=200,
        seed=7,
        parameter_name="birth",
        true_rate_key="birth_rate",
    )


class TestConsistencyBands:
    """Test compute_consistency_bands tidy output and the faceted renderer."""

    def test_tidy_shape_and_columns(self) -> None:
        bands = _bands()
        # 2 regimes x 3 T x 3 statistics.
        assert bands.shape == (18, 7)
        assert list(bands.columns) == [
            "regime",
            "T",
            "statistic",
            "estimate",
            "lower",
            "upper",
            "true_value",
        ]
        assert set(bands["statistic"]) == {"mean", "median", "variance"}

    def test_variance_has_no_true_reference(self) -> None:
        bands = _bands()
        assert bands.loc[bands["statistic"] == "variance", "true_value"].isna().all()
        assert (bands.loc[bands["statistic"] == "mean", "true_value"] > 0).all()

    @pytest.mark.regression
    def test_golden_bootstrap_bands(self) -> None:
        bands = _bands()
        mean_a100 = bands[
            (bands["regime"] == "A")
            & (bands["T"] == 100.0)
            & (bands["statistic"] == "mean")
        ].iloc[0]
        assert mean_a100["estimate"] == pytest.approx(0.0193925675, abs=1e-9)
        assert mean_a100["lower"] == pytest.approx(0.0179876579, abs=1e-9)
        assert mean_a100["upper"] == pytest.approx(0.0207710029, abs=1e-9)
        assert mean_a100["true_value"] == pytest.approx(0.02)

        var_b1000 = bands[
            (bands["regime"] == "B")
            & (bands["T"] == 1000.0)
            & (bands["statistic"] == "variance")
        ].iloc[0]
        assert var_b1000["estimate"] == pytest.approx(2.4330990e-05, abs=1e-12)

    def test_plot_parameter_consistency_facets_regime_and_statistic(self) -> None:
        summary_df, results_df = _synthetic_inputs()
        chart = plot_parameter_consistency(
            summary_df,
            results_df,
            _REGIMES,
            bootstrap_samples=50,
            seed=7,
            parameter_name="birth",
            true_rate_key="birth_rate",
            title="Birth consistency",
        )
        spec = chart.to_dict()
        assert spec["facet"]["row"]["field"] == "regime"
        assert spec["facet"]["column"]["field"] == "statistic"
        # Layered cell: CI band area, estimate line, true-value rule.
        marks = {
            layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]
            for layer in spec["spec"]["layer"]
        }
        assert {"area", "line", "rule"} <= marks
