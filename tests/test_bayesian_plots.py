"""Test suite for visualization.bayesian_plots builders."""

import altair as alt
import numpy as np
import pytest

from inf_bp_texts.visualization.bayesian_plots import (
    PriorPosteriorPanel,
    RatePanel,
    TracePanel,
    posterior_rates_chart,
    ppc_chart,
    prior_posterior_chart,
    prior_rate_samples,
    prior_rates_chart,
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
    """Test the posterior-rate ridgeline chart."""

    def test_returns_facet_chart_with_three_layers(self) -> None:
        rng = np.random.default_rng(0)
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Epoch 1", "Epoch 2"],
                samples=rng.lognormal(size=(300, 2)),
                color="#4477AA",
            ),
            RatePanel(
                label="Death Rate",
                categories=["Epoch 1", "Epoch 2"],
                samples=rng.lognormal(size=(300, 2)),
                color="#EE6677",
            ),
        ]

        chart = posterior_rates_chart(panels, title="Posterior Rates")

        assert isinstance(chart, alt.FacetChart)
        spec = chart.to_dict()
        assert len(spec["spec"]["layer"]) == 3  # area + median_point + ci_bar

    def test_facet_row_and_column_fields(self) -> None:
        rng = np.random.default_rng(1)
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Epoch 1", "Epoch 2", "Epoch 3"],
                samples=rng.lognormal(size=(200, 3)),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test", category_title="Epoch")

        spec = chart.to_dict()
        assert spec["facet"]["row"]["field"] == "category"
        assert spec["facet"]["column"]["field"] == "panel"

    def test_ci_bounds_are_95_quantile_interval(self) -> None:
        """Summary rows carry the equal-tailed 95% credible interval."""
        rng = np.random.default_rng(2)
        col = rng.lognormal(size=1000)
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Epoch 1"],
                samples=col[:, np.newaxis],
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test")

        spec = chart.to_dict()
        all_rows = [row for rows in spec["datasets"].values() for row in rows]
        summary_rows = [r for r in all_rows if r.get("row_type") == "summary"]
        assert len(summary_rows) == 1
        row = summary_rows[0]
        assert row["rate"] == pytest.approx(
            float(np.median(col))
        )  # rate=median for summary rows
        assert row["ci_low"] == pytest.approx(float(np.quantile(col, 0.025)))
        assert row["ci_high"] == pytest.approx(float(np.quantile(col, 0.975)))
        assert row["value"] == pytest.approx(0.0)  # baseline y-position

    def test_constant_column_keeps_summary_and_draw_rows(self) -> None:
        """A constant column still emits draw rows and a summary marker.

        Binning of constant draws happens in-spec (Vega-Lite), so all 100 draw
        rows are passed through to the spec unchanged.  The summary row carries
        the median (0.5) and the CI errorbar.
        """
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Constant"],
                samples=np.full((100, 1), 0.5),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test")

        spec = chart.to_dict()
        all_rows = [row for rows in spec["datasets"].values() for row in rows]
        draw_rows = [r for r in all_rows if r.get("row_type") == "draw"]
        summary_rows = [r for r in all_rows if r.get("row_type") == "summary"]
        assert len(draw_rows) == 100  # all draws passed through to Vega-Lite
        assert len(summary_rows) == 1
        assert summary_rows[0]["rate"] == pytest.approx(
            0.5
        )  # rate=median for summary rows

    def test_log_scale_produces_valid_spec(self) -> None:
        """log_scale=True builds a chart and the first x-encoding uses log scale."""
        rng = np.random.default_rng(3)
        panels = [
            RatePanel(
                label="Half-life",
                categories=["Epoch 1", "Epoch 2"],
                samples=rng.lognormal(mean=5, sigma=1, size=(200, 2)),
                color="#228833",
            ),
        ]

        chart = posterior_rates_chart(
            panels, title="Half-life", vertical=True, log_scale=True
        )

        spec = chart.to_dict()
        # vconcat → first FacetChart → layer[1] is median_point; its x encoding
        # should carry a log scale.
        layer = spec["vconcat"][0]["spec"]["layer"]
        # layer[1] is the median_point mark_point layer
        median_x = layer[1]["encoding"]["x"]
        assert median_x.get("scale", {}).get("type") == "log"

    def test_default_no_log_scale(self) -> None:
        """Default (log_scale=False) path leaves no log-scale in x encoding."""
        rng = np.random.default_rng(4)
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Epoch 1"],
                samples=rng.lognormal(size=(100, 1)),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test")

        spec = chart.to_dict()
        layer = spec["spec"]["layer"]
        median_x = layer[1]["encoding"]["x"]
        # No scale key at all, or type is not "log"
        assert median_x.get("scale", {}).get("type") != "log"


class TestPriorRateSamples:
    """Test the raw deterministic prior-quantile-sample helper."""

    def test_shapes_and_birth_death_split(self) -> None:
        locs = np.array([-1.0, -1.0, -10.0, -10.0])
        scales = np.array([0.1, 0.1, 0.1, 0.1])

        birth, death = prior_rate_samples(locs, scales, n_categories=2, n_quantiles=500)

        assert birth.shape == (500, 2)
        assert death.shape == (500, 2)
        assert np.median(birth) == pytest.approx(np.exp(-1.0), rel=0.05)
        assert np.median(death) == pytest.approx(np.exp(-10.0), rel=0.05)

    def test_deterministic_across_calls(self) -> None:
        locs = np.array([-2.0, -5.6])
        scales = np.array([0.6, 0.6])

        birth_a, death_a = prior_rate_samples(locs, scales, n_categories=1)
        birth_b, death_b = prior_rate_samples(locs, scales, n_categories=1)

        np.testing.assert_array_equal(birth_a, birth_b)
        np.testing.assert_array_equal(death_a, death_b)


class TestPriorRatesChart:
    """Test the prior-rate ridgeline chart."""

    def test_returns_vconcat_with_birth_and_death_panels(self) -> None:
        locs = np.array([-2.0, -2.0, -5.6, -5.6])
        scales = np.array([0.6, 0.6, 0.6, 0.6])

        chart = prior_rates_chart(locs, scales, ["Epoch 1", "Epoch 2"], title="Priors")

        assert isinstance(chart, alt.VConcatChart)
        spec = chart.to_dict()
        assert len(spec["vconcat"]) == 2  # birth panel + death panel

    def test_deterministic_across_calls(self) -> None:
        """Quantile samples come from fixed Hazen positions, not an RNG draw."""
        locs = np.array([-2.0, -5.6])
        scales = np.array([0.6, 0.6])

        chart_a = prior_rates_chart(locs, scales, ["Epoch 1"], title="test")
        chart_b = prior_rates_chart(locs, scales, ["Epoch 1"], title="test")

        assert chart_a.to_dict() == chart_b.to_dict()

    def test_log_scale_default_true(self) -> None:
        locs = np.array([-2.0, -5.6])
        scales = np.array([0.6, 0.6])

        chart = prior_rates_chart(locs, scales, ["Epoch 1"], title="test")

        spec = chart.to_dict()
        layer = spec["vconcat"][0]["spec"]["layer"]
        median_x = layer[1]["encoding"]["x"]
        assert median_x.get("scale", {}).get("type") == "log"

    def test_birth_death_index_convention(self) -> None:
        """First half of locs/scales is birth, second half is death."""
        locs = np.array([-1.0, -10.0])  # birth loc=-1, death loc=-10
        scales = np.array([0.1, 0.1])

        chart = prior_rates_chart(locs, scales, ["Epoch 1"], title="test")

        spec = chart.to_dict()
        birth_name = spec["vconcat"][0]["data"]["name"]
        death_name = spec["vconcat"][1]["data"]["name"]
        birth_rows = [
            r for r in spec["datasets"][birth_name] if r.get("row_type") == "draw"
        ]
        death_rows = [
            r for r in spec["datasets"][death_name] if r.get("row_type") == "draw"
        ]
        assert np.median([r["rate"] for r in birth_rows]) == pytest.approx(
            np.exp(-1.0), rel=0.05
        )
        assert np.median([r["rate"] for r in death_rows]) == pytest.approx(
            np.exp(-10.0), rel=0.05
        )


class TestTraceChart:
    """Test the MCMC trace-plot grid."""

    def test_returns_list_of_charts_with_zero_false_scale(self) -> None:
        rng = np.random.default_rng(0)
        panels = [
            TracePanel(label="log lambda epoch 0", chains=rng.normal(size=(3, 50))),
            TracePanel(label="log mu epoch 0", chains=rng.normal(size=(3, 50))),
        ]

        charts = trace_chart(panels)

        assert isinstance(charts, list)
        assert len(charts) == 2
        for chart in charts:
            assert isinstance(chart, alt.Chart)
            spec = chart.to_dict()
            assert spec["encoding"]["y"]["scale"] == {"zero": False}
            assert spec["encoding"]["color"]["legend"] == {"title": None}

    def test_long_frame_has_one_row_per_chain_per_draw(self) -> None:
        panels = [TracePanel(label="param", chains=np.zeros((4, 25)))]

        charts = trace_chart(panels)

        datasets = charts[0].to_dict()["datasets"]
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
