"""Tests for the date-range KDE comparison figure."""

import altair as alt
import numpy as np

from inf_bp_texts.visualization import plot_date_range_kde_comparison


class TestDateRangeKdeComparison:
    """Test plot_date_range_kde_comparison structure and scaling."""

    def test_layers_two_histograms_and_a_kde_line(self) -> None:
        rng = np.random.default_rng(0)
        midpoint = rng.normal(1000, 120, 40)
        grid = np.linspace(700, 1300, 200)
        density = np.exp(-0.5 * ((grid - 1000) / 120) ** 2) / (120 * np.sqrt(2 * np.pi))
        resampled = rng.normal(1000, 120, 40)

        chart = plot_date_range_kde_comparison(
            midpoint,
            grid,
            density,
            resampled,
            n=40,
            bin_width=50.0,
            title="Date ranges",
        )

        assert isinstance(chart, alt.LayerChart)
        spec = chart.to_dict()
        marks = [
            layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]
            for layer in spec["layer"]
        ]
        # Midpoint histogram, resampled histogram, KDE line.
        assert marks.count("bar") == 2
        assert marks.count("line") == 1

    def test_accepts_integer_year_arrays(self) -> None:
        # Resampled dates arrive as integer years; the builder must accept them.
        years = np.array([900, 950, 1000, 1050], dtype=np.int64)
        grid = np.linspace(900, 1050, 50)
        density = np.full(50, 1.0 / 150.0)
        chart = plot_date_range_kde_comparison(
            years, grid, density, years, n=4, bin_width=25.0, title="ints"
        )
        assert chart.to_dict()["layer"]
