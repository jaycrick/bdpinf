"""Test suite for visualization.bayesian_plots builders."""

import altair as alt
import numpy as np

from inf_bp_texts.visualization.bayesian_plots import (
    PriorPosteriorPanel,
    RatePanel,
    TracePanel,
    posterior_rates_chart,
    ppc_chart,
    prior_posterior_chart,
    trace_chart,
)


class TestPriorPosteriorChart:
    """Test the prior-vs-posterior overlay chart."""

    def test_returns_hconcat_with_one_panel_per_input(self) -> None:
        rng = np.random.default_rng(0)
        panels = [
            PriorPosteriorPanel(
                label="Birth Rate",
                posterior_samples=rng.normal(size=200),
                prior_x=np.linspace(-5, 5, 50),
                prior_pdf=np.full(50, 0.1),
                color="#4477AA",
            ),
            PriorPosteriorPanel(
                label="Death Rate",
                posterior_samples=rng.normal(size=200),
                prior_x=np.linspace(-5, 5, 50),
                prior_pdf=np.full(50, 0.1),
                color="#EE6677",
            ),
        ]

        chart = prior_posterior_chart(panels, title="Prior vs Posterior")

        assert isinstance(chart, alt.HConcatChart)
        assert len(chart.hconcat) == 2
        for sub_chart in chart.hconcat:
            assert isinstance(sub_chart, alt.LayerChart)
            assert len(sub_chart.layer) == 2  # histogram + scaled prior line

    def test_omits_prior_overlay_for_degenerate_samples(self) -> None:
        """A single repeated sample value has no bin width to scale the prior by."""
        panels = [
            PriorPosteriorPanel(
                label="Constant",
                posterior_samples=np.full(10, 1.0),
                prior_x=np.linspace(-5, 5, 50),
                prior_pdf=np.full(50, 0.1),
                color="#4477AA",
            ),
        ]

        chart = prior_posterior_chart(panels, title="test")

        sub_chart = chart.hconcat[0]
        assert isinstance(sub_chart, alt.LayerChart)
        assert len(sub_chart.layer) == 1  # histogram only, no prior overlay


class TestPosteriorRatesChart:
    """Test the posterior-rate bar-with-credible-interval chart."""

    def test_returns_hconcat_with_one_panel_per_input(self) -> None:
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Epoch 1", "Epoch 2"],
                values=np.array([0.01, 0.02]),
                ci_lower=np.array([0.005, 0.015]),
                ci_upper=np.array([0.015, 0.025]),
                color="#4477AA",
            ),
            RatePanel(
                label="Death Rate",
                categories=["Epoch 1", "Epoch 2"],
                values=np.array([0.03, 0.04]),
                ci_lower=np.array([0.025, 0.035]),
                ci_upper=np.array([0.035, 0.045]),
                color="#EE6677",
            ),
        ]

        chart = posterior_rates_chart(panels, title="Posterior Rates")

        assert isinstance(chart, alt.HConcatChart)
        assert len(chart.hconcat) == 2
        first_panel = chart.hconcat[0]
        assert isinstance(first_panel, alt.LayerChart)
        datasets = chart.to_dict()["datasets"]
        first_panel_rows = next(iter(datasets.values()))
        assert [row["category"] for row in first_panel_rows] == ["Epoch 1", "Epoch 2"]

    def test_generalizes_beyond_two_categories(self) -> None:
        """N-way comparisons (e.g. families, subsets) must work, not just 2 epochs."""
        categories = ["Subset A", "Subset B", "Subset C"]
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=categories,
                values=np.array([0.01, 0.02, 0.03]),
                ci_lower=np.array([0.005, 0.015, 0.025]),
                ci_upper=np.array([0.015, 0.025, 0.035]),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test")

        datasets = chart.to_dict()["datasets"]
        panel_rows = next(iter(datasets.values()))
        assert len(panel_rows) == 3


class TestTraceChart:
    """Test the MCMC trace-plot grid."""

    def test_returns_concat_grid_with_legend_on_first_panel_only(self) -> None:
        rng = np.random.default_rng(0)
        panels = [
            TracePanel(label="log lambda epoch 0", chains=rng.normal(size=(3, 50))),
            TracePanel(label="log mu epoch 0", chains=rng.normal(size=(3, 50))),
        ]

        chart = trace_chart(panels, title="Traces", columns=2)

        assert isinstance(chart, alt.ConcatChart)
        assert len(chart.concat) == 2
        panel_specs = chart.to_dict()["concat"]
        assert panel_specs[0]["encoding"]["color"]["legend"] == {"title": None}
        assert panel_specs[1]["encoding"]["color"]["legend"] is None

    def test_long_frame_has_one_row_per_chain_per_draw(self) -> None:
        panels = [TracePanel(label="param", chains=np.zeros((4, 25)))]

        chart = trace_chart(panels, title="test")

        datasets = chart.to_dict()["datasets"]
        panel_rows = next(iter(datasets.values()))
        assert len(panel_rows) == 4 * 25


class TestPpcChart:
    """Test the posterior-predictive-check overlay chart."""

    def test_returns_layered_chart_with_histogram_and_observed_rule(self) -> None:
        rng = np.random.default_rng(0)
        replicated = rng.normal(loc=10.0, scale=2.0, size=500)

        chart = ppc_chart(replicated, observed_value=11.5, title="PPC")

        assert isinstance(chart, alt.LayerChart)
        assert len(chart.layer) == 2
        rule_data = chart.layer[1].data
        assert rule_data["Replicated statistic"].iloc[0] == 11.5
