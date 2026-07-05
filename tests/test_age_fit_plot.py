"""Test suite for visualization.age_fit_plot helpers."""

import altair as alt
import numpy as np

from inf_bp_texts.visualization.age_fit_plot import (
    creation_year_fit_chart,
    plot_creation_year_fit,
)


class TestPlotCreationYearFit:
    """Test the observed-vs-fitted creation-year overlay chart."""

    def test_returns_layered_chart_with_density_overlay(self) -> None:
        ages = np.array([10.0, 40.0, 70.0, 90.0])
        times = np.array([100.0], dtype=np.float64)

        chart = plot_creation_year_fit(
            ages=ages,
            times=times,
            start_time=0.0,
            birth_rate=0.05,
            death_rate=0.01,
            title="test",
        )

        assert isinstance(chart, alt.LayerChart)
        assert len(chart.layer) == 2  # observed + expected-count histograms

    def test_excludes_point_mass_when_oldest_age_is_full_duration(self) -> None:
        """The spine's age is deterministically the full duration (a point mass).

        When the oldest observed manuscript's age equals that duration, the
        density grid must not evaluate exactly there, so the resulting
        expected-count overlay stays finite.
        """
        process_duration = 100.0
        ages = np.array([process_duration, 50.0, 20.0])
        times = np.array([process_duration], dtype=np.float64)

        chart = plot_creation_year_fit(
            ages=ages,
            times=times,
            start_time=0.0,
            birth_rate=0.05,
            death_rate=0.01,
            title="test",
        )

        expected_layer = chart.layer[1]
        counts = expected_layer.data["count"].to_numpy()
        assert np.all(np.isfinite(counts))


class TestCreationYearFitChart:
    """Test the shared observed/expected layered-histogram builder."""

    def test_default_labels_and_axis_titles(self) -> None:
        observed = np.array([10.0, 20.0, 30.0, 40.0])
        density_years = np.linspace(0.0, 50.0, 20)
        density_values = np.full(20, 0.02)

        chart = creation_year_fit_chart(
            observed, density_years, density_values, title="test"
        )

        spec = chart.to_dict()
        histogram_enc = spec["layer"][0]["encoding"]
        assert histogram_enc["x"]["title"] == "Creation Year"
        assert histogram_enc["y"]["title"] == "count"
        assert histogram_enc["color"]["scale"]["domain"] == ["Observed", "Expected"]

    def test_bars_use_binned_x_so_they_reach_the_zero_baseline(self) -> None:
        """Without bin="binned", Vega-Lite renders floating bars that never reach zero."""
        observed = np.array([10.0, 20.0, 30.0, 40.0])
        density_years = np.linspace(0.0, 50.0, 20)
        density_values = np.full(20, 0.02)

        chart = creation_year_fit_chart(
            observed, density_years, density_values, title="test"
        )

        spec = chart.to_dict()
        for layer in spec["layer"]:
            assert layer["encoding"]["x"]["bin"] == "binned"

    def test_x_domain_extends_to_cover_fitted_range(self) -> None:
        """The shared x-domain must reach the density grid even past observed data."""
        observed = np.array([100.0, 110.0, 120.0])
        density_years = np.linspace(0.0, 200.0, 50)
        density_values = np.full(50, 0.01)

        chart = creation_year_fit_chart(
            observed, density_years, density_values, title="test"
        )

        domain = chart.to_dict()["layer"][0]["encoding"]["x"]["scale"]["domain"]
        assert domain[0] <= 0.0
        assert domain[1] >= 200.0

    def test_both_layers_share_bin_edges_and_bin_count(self) -> None:
        observed = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        density_years = np.linspace(0.0, 20.0, 10)
        density_values = np.full(10, 0.05)

        chart = creation_year_fit_chart(
            observed, density_years, density_values, title="test", nbins=10
        )

        observed_data = chart.layer[0].data
        expected_data = chart.layer[1].data
        assert len(observed_data) == 10
        assert len(expected_data) == 10
        np.testing.assert_array_equal(
            observed_data["bin_start"].to_numpy(), expected_data["bin_start"].to_numpy()
        )

    def test_omits_expected_layer_for_degenerate_observed_years(self) -> None:
        """A single repeated observed year has no bin width to scale against."""
        chart = creation_year_fit_chart(np.full(5, 10.0), None, None, title="test")

        assert len(chart.layer) == 1

    def test_expected_counts_peak_near_true_value(self) -> None:
        """A smooth density bump at a known year should peak in a nearby bin."""
        observed = np.array([0.0, 100.0])
        density_years = np.linspace(0.0, 100.0, 500)
        density_values = np.exp(-0.5 * ((density_years - 50.0) / 5.0) ** 2)

        chart = creation_year_fit_chart(
            observed, density_years, density_values, title="test", nbins=10
        )

        expected_data = chart.layer[1].data
        peak_row = expected_data.loc[expected_data["count"].idxmax()]
        assert 40.0 <= peak_row["bin_start"] <= 60.0
