"""Test suite for visualization.age_fit_plot helpers."""

import altair as alt
import numpy as np

from inf_bp_texts.visualization.age_fit_plot import plot_creation_year_fit


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
        assert len(chart.layer) == 2  # noqa: PLR2004 - histogram + density layers

    def test_excludes_point_mass_when_oldest_age_is_full_duration(self) -> None:
        """The spine's age is deterministically the full duration (a point mass).

        When the oldest observed manuscript's age equals that duration, the
        density grid must not evaluate exactly there.
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

        density_layer = chart.layer[1]
        birth_years = density_layer.data["Creation year (CE)"].to_numpy()
        # The point-mass boundary (birth_year == start_time, i.e. age ==
        # process_duration) must be excluded; the grid's nearest point should
        # be nudged just inside the domain instead of landing exactly on it.
        assert not np.any(birth_years == 0.0)
