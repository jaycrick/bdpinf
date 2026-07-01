"""Test suite for visualization.correlation_plots builders."""

import altair as alt
import numpy as np
import pandas as pd

from inf_bp_texts.visualization.altair_theme import DIVERGING_RDBU
from inf_bp_texts.visualization.correlation_plots import (
    correlation_heatmap_chart,
    posterior_rate_correlation,
)


class TestPosteriorRateCorrelation:
    """Test the births-then-deaths posterior correlation matrix builder."""

    def test_returns_square_labelled_symmetric_matrix(self) -> None:
        rng = np.random.default_rng(0)
        birth_samples = rng.normal(size=(500, 2))
        death_samples = rng.normal(size=(500, 2))
        epoch_names = ["Epoch 1", "Epoch 2"]

        corr = posterior_rate_correlation(birth_samples, death_samples, epoch_names)

        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (4, 4)
        assert list(corr.index) == ["λ Epoch 1", "λ Epoch 2", "μ Epoch 1", "μ Epoch 2"]
        assert list(corr.columns) == list(corr.index)
        np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)
        np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T)

    def test_drops_zero_variance_columns(self) -> None:
        rng = np.random.default_rng(1)
        birth_samples = np.column_stack([rng.normal(size=300), np.full(300, 0.5)])
        death_samples = rng.normal(size=(300, 2))
        epoch_names = ["Epoch 1", "Epoch 2"]

        corr = posterior_rate_correlation(birth_samples, death_samples, epoch_names)

        assert "λ Epoch 2" not in corr.index
        assert corr.shape == (3, 3)
        assert not corr.isna().to_numpy().any()

    def test_custom_symbols(self) -> None:
        rng = np.random.default_rng(2)
        birth_samples = rng.normal(size=(100, 1))
        death_samples = rng.normal(size=(100, 1))

        corr = posterior_rate_correlation(
            birth_samples,
            death_samples,
            ["1100-1800"],
            birth_symbol="B",
            death_symbol="D",
        )

        assert list(corr.index) == ["B 1100-1800", "D 1100-1800"]


class TestCorrelationHeatmapChart:
    """Test the diverging correlation heatmap builder."""

    def test_returns_layer_chart_with_rect_and_text(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.3], [0.3, 1.0]],
            index=["λ Epoch 1", "μ Epoch 1"],
            columns=["λ Epoch 1", "μ Epoch 1"],
        )

        chart = correlation_heatmap_chart(corr, title="Posterior rate correlations")

        assert isinstance(chart, alt.LayerChart)
        spec = chart.to_dict()
        marks = [layer["mark"] for layer in spec["layer"]]
        assert {"rect", "text"}.issubset(
            {m if isinstance(m, str) else m["type"] for m in marks}
        )

    def test_color_scale_matches_diverging_palette(self) -> None:
        corr = pd.DataFrame(
            [[1.0, -0.7], [-0.7, 1.0]], index=["a", "b"], columns=["a", "b"]
        )

        chart = correlation_heatmap_chart(corr, title="test")
        spec = chart.to_dict()

        rect_layer = next(
            layer
            for layer in spec["layer"]
            if (
                layer["mark"]
                if isinstance(layer["mark"], str)
                else layer["mark"]["type"]
            )
            == "rect"
        )
        color_scale = rect_layer["encoding"]["color"]["scale"]
        assert color_scale["domain"] == [-1, 1]
        assert color_scale["range"] == list(reversed(DIVERGING_RDBU))

    def test_text_format_and_labels(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.123], [0.123, 1.0]], index=["a", "b"], columns=["a", "b"]
        )

        chart = correlation_heatmap_chart(corr, title="test", text_format=".3f")
        spec = chart.to_dict()

        text_layer = next(
            layer
            for layer in spec["layer"]
            if (
                layer["mark"]
                if isinstance(layer["mark"], str)
                else layer["mark"]["type"]
            )
            == "text"
        )
        assert text_layer["encoding"]["text"]["format"] == ".3f"
