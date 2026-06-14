"""Tests for the date-distribution exploration figures."""

import altair as alt
import numpy as np
import pandas as pd

from inf_bp_texts.visualization import (
    plot_date_range_kde_comparison,
    plot_grouped_date_distribution,
)


def _layer_mark_types(chart: alt.LayerChart) -> list[str]:
    """Return the mark-type string for each layer of an Altair layer chart."""
    return [
        layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]
        for layer in chart.to_dict()["layer"]
    ]


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

    def test_custom_midpoint_label_drives_color_domain(self) -> None:
        years = np.array([900.0, 1000.0, 1100.0])
        grid = np.linspace(900, 1100, 30)
        density = np.full(30, 1.0 / 200.0)
        chart = plot_date_range_kde_comparison(
            years,
            grid,
            density,
            years,
            n=3,
            bin_width=50.0,
            title="midyear",
            midpoint_label="Midpoint (midYear)",
        )
        domain = chart.to_dict()["layer"][0]["encoding"]["color"]["scale"]["domain"]
        assert domain[0] == "Midpoint (midYear)"


class TestGroupedDateDistribution:
    """Test the grouped (stacked-by-group) date-distribution figure."""

    def test_layers_histograms_kde_and_resampled_lines(self) -> None:
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            {
                "year": rng.normal(1000, 150, 90),
                "group": np.repeat(["Papyri", "Majuscules", "Minuscules"], 30),
            }
        )
        grid = np.linspace(600, 1400, 60)
        density = np.full(60, 1.0 / 800.0)
        mids = np.arange(650.0, 1400.0, 50.0)
        counts = rng.integers(0, 8, len(mids)).astype(float)

        chart = plot_grouped_date_distribution(
            frame,
            grid,
            density,
            mids,
            counts,
            n=90,
            bin_width=50.0,
            group_order=["Papyri", "Majuscules", "Minuscules"],
            group_colors=["#4477AA", "#EE6677", "#228833"],
            title="Grouped",
        )

        marks = _layer_mark_types(chart)
        # Stacked group histograms (one bar layer) + KDE line + resampled line.
        assert marks.count("bar") == 1
        assert marks.count("line") == 2
        domain = chart.to_dict()["layer"][0]["encoding"]["color"]["scale"]["domain"]
        assert domain == [
            "Papyri",
            "Majuscules",
            "Minuscules",
            "Weighted KDE",
            "Resampled dates",
        ]
