"""Test suite for visualization.bayesian_plots builders."""

import altair as alt
import numpy as np
import pytest

from bdpinf.visualization.bayesian_plots import (
    PriorPosteriorPanel,
    PriorPosteriorRatePanel,
    RatePanel,
    TracePanel,
    _histogram_ridge_rows,
    posterior_rates_chart,
    ppc_chart,
    prior_posterior_chart,
    prior_posterior_rates_chart,
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

    def test_constant_column_produces_normalized_bin_and_summary_row(self) -> None:
        """A constant column still emits a normalized bin ridge and a summary marker.

        Binning happens in numpy (:func:`_histogram_ridge_rows`), so a
        constant column produces up to ``_RIDGE_MAXBINS`` bin rows (all zero
        except the single bin containing the constant). With only one
        category/panel in this test, the figure-wide rescale
        (:func:`_rescale_bins_to_global_peak`) has nothing else to compare
        against, so that single bin still normalizes to 1.0 — the tallest bin
        in the (only) ridge peaks at 1.0. The summary row carries the median
        (0.5) and the CI errorbar.
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
        bin_rows = [r for r in all_rows if r.get("row_type") == "bin"]
        summary_rows = [r for r in all_rows if r.get("row_type") == "summary"]
        assert len(bin_rows) > 0
        assert max(r["value"] for r in bin_rows) == pytest.approx(1.0)
        assert len(summary_rows) == 1
        assert summary_rows[0]["rate"] == pytest.approx(
            0.5
        )  # rate=median for summary rows

    def test_ridge_density_is_invariant_to_draw_count(self) -> None:
        """Area-normalized density is unchanged when every draw is duplicated.

        :func:`_histogram_ridge_rows` normalizes by ``n * bin_width``, so
        multiplying draw count while preserving the underlying distribution
        (duplicating every draw the same number of times) leaves every bin's
        density value unchanged — this is what keeps ridges comparable
        across epochs/subsets regardless of how many posterior draws each
        has, matching the old peak-normalization's draw-count invariance
        without collapsing every ridge to the same peak height.
        """
        rng = np.random.default_rng(11)
        few = rng.lognormal(size=200)
        many = np.repeat(few, 50)

        few_rows = _histogram_ridge_rows(few, log_scale=False)
        many_rows = _histogram_ridge_rows(many, log_scale=False)

        assert [r["value"] for r in few_rows] == pytest.approx(
            [r["value"] for r in many_rows]
        )

    def test_ridge_height_reflects_relative_spread(self) -> None:
        """A tightly concentrated category renders taller than a diffuse one.

        Previously every ridge was peak-normalized independently, so all
        ridges read as the same height regardless of spread. After switching
        to area-normalization plus a shared figure-wide rescale
        (:func:`_rescale_bins_to_global_peak`), a low-variance category's
        peak density is higher (its mass concentrates in fewer bins) than a
        high-variance category's, so the low-variance ridge renders taller —
        this is the spread information the flat-height ridges were hiding.
        """
        rng = np.random.default_rng(13)
        tight = rng.normal(loc=1.0, scale=0.02, size=2000)
        wide = rng.normal(loc=1.0, scale=1.0, size=2000)
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Tight", "Wide"],
                samples=np.column_stack([tight, wide]),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test")

        spec = chart.to_dict()
        all_rows = [row for rows in spec["datasets"].values() for row in rows]
        bin_rows = [r for r in all_rows if r.get("row_type") == "bin"]
        tight_peak = max(r["value"] for r in bin_rows if r["category"] == "Tight")
        wide_peak = max(r["value"] for r in bin_rows if r["category"] == "Wide")
        assert tight_peak == pytest.approx(1.0)  # global tallest ridge
        assert wide_peak < tight_peak

    def test_ridge_y_scale_domain_fixed_to_unit_interval(self) -> None:
        """The shared facet y-scale domain is [0, 1] after the global-peak rescale."""
        rng = np.random.default_rng(12)
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Epoch 1"],
                samples=rng.lognormal(size=(100, 1)),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test", vertical=True)

        layer = chart.to_dict()["vconcat"][0]["spec"]["layer"]
        area_y = layer[0]["encoding"]["y"]
        assert area_y["scale"]["domain"] == [0.0, 1.0]

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

    def test_x_axis_title_has_no_stray_field_names(self) -> None:
        """Regression test: layering used to comma-join distinct field titles.

        Before the fix, the area's x title ("Rate per year") was joined with
        the untitled median/CI-bar x-encodings' own field names ("rate",
        "ci_low"), producing "Rate per year, rate, ci_low" in the exported
        axis title.
        """
        rng = np.random.default_rng(5)
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Epoch 1"],
                samples=rng.lognormal(size=(100, 1)),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test", vertical=True)

        spec = chart.to_dict()
        layer = spec["vconcat"][0]["spec"]["layer"]
        for encoded in (layer[0]["encoding"]["x"], layer[1]["encoding"]["x"]):
            assert encoded["title"] == "Rate per year"
        assert "ci_low" not in str(layer[2]["encoding"]["x"].get("title", ""))

    def test_facet_row_height_targets_200px_total(self) -> None:
        """Row height must scale so an n-category ridgeline nears 200px tall.

        FacetChart height is ignored by save_chart's width/height override
        (Vega-Lite derives it from row height * row count), so the only way
        to land near the project's usual 300x200 export target is to size
        the row height at chart-build time.
        """
        rng = np.random.default_rng(6)
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"],
                samples=rng.lognormal(size=(50, 4)),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test", vertical=True)

        row_height = chart.to_dict()["vconcat"][0]["spec"]["height"]
        assert row_height * 4 == pytest.approx(200.0)

    def test_legend_entries_present_and_labelled(self) -> None:
        """Distribution, median, and CI legends each live on a distinct channel.

        Regression test: the distribution (area) and CI (errorbar) legends
        used to share the ``opacity`` channel, so Vega-Lite merged them into
        one legend and discarded the errorbar's intended symbol. They must
        now be on different channels (``opacity`` / ``shape`` / ``color``) so
        each keeps its own symbol.
        """
        rng = np.random.default_rng(7)
        panels = [
            RatePanel(
                label="Birth Rate",
                categories=["Epoch 1"],
                samples=rng.lognormal(size=(100, 1)),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(panels, title="test", vertical=True)

        layer = chart.to_dict()["vconcat"][0]["spec"]["layer"]
        area_opacity = layer[0]["encoding"]["opacity"]
        median_shape = layer[1]["encoding"]["shape"]
        ci_color = layer[2]["encoding"]["color"]
        assert area_opacity["scale"]["domain"] == ["Posterior distribution"]
        assert median_shape["scale"]["domain"] == ["Median"]
        assert ci_color["scale"]["domain"] == ["95% CrI"]

    def test_distribution_label_override(self) -> None:
        """prior_rates_chart overrides the label so its ridge doesn't say "Posterior"."""
        rng = np.random.default_rng(8)
        panels = [
            RatePanel(
                label="Prior Rate",
                categories=["Epoch 1"],
                samples=rng.lognormal(size=(100, 1)),
                color="#4477AA",
            ),
        ]

        chart = posterior_rates_chart(
            panels, title="test", vertical=True, distribution_label="Prior distribution"
        )

        layer = chart.to_dict()["vconcat"][0]["spec"]["layer"]
        assert layer[0]["encoding"]["opacity"]["scale"]["domain"] == [
            "Prior distribution"
        ]


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
        # Summary rows carry the precomputed median directly under "rate".
        birth_rows = [
            r for r in spec["datasets"][birth_name] if r.get("row_type") == "summary"
        ]
        death_rows = [
            r for r in spec["datasets"][death_name] if r.get("row_type") == "summary"
        ]
        assert np.median([r["rate"] for r in birth_rows]) == pytest.approx(
            np.exp(-1.0), rel=0.05
        )
        assert np.median([r["rate"] for r in death_rows]) == pytest.approx(
            np.exp(-10.0), rel=0.05
        )


class TestPriorPosteriorRatesChart:
    """Test the prior-vs-posterior overlay ridgeline chart (same row per category)."""

    def test_returns_vconcat_with_row_facet_only(self) -> None:
        rng = np.random.default_rng(0)
        panels = [
            PriorPosteriorRatePanel(
                label="Birth Rate",
                categories=["Epoch 1", "Epoch 2"],
                prior_samples=rng.lognormal(size=(4000, 2)),
                posterior_samples=rng.lognormal(size=(1000, 2)),
                color="#4477AA",
            ),
        ]

        chart = prior_posterior_rates_chart(panels, title="Comparison")

        assert isinstance(chart, alt.VConcatChart)
        spec = chart.to_dict()
        assert len(spec["vconcat"]) == 1
        facet = spec["vconcat"][0]["facet"]
        assert facet["row"]["field"] == "category"
        assert "column" not in facet  # no column facet: prior/posterior share one row

    def test_four_layers_prior_outline_then_posterior_fill(self) -> None:
        rng = np.random.default_rng(1)
        panels = [
            PriorPosteriorRatePanel(
                label="Death Rate",
                categories=["Epoch 1"],
                prior_samples=rng.lognormal(size=(4000, 1)),
                posterior_samples=rng.lognormal(size=(1000, 1)),
                color="#EE6677",
            ),
        ]

        chart = prior_posterior_rates_chart(panels, title="test")

        layer = chart.to_dict()["vconcat"][0]["spec"]["layer"]
        assert len(layer) == 4  # prior_area, posterior_area, median_point, ci_bar
        prior_area, posterior_area = layer[0], layer[1]
        assert prior_area["mark"]["fillOpacity"] == 0
        assert prior_area["mark"]["strokeDash"] == [4, 4]
        assert "strokeDash" not in posterior_area["mark"]
        assert posterior_area["mark"]["fillOpacity"] == 0.8

    def test_only_posterior_gets_summary_rows(self) -> None:
        rng = np.random.default_rng(2)
        panels = [
            PriorPosteriorRatePanel(
                label="Birth Rate",
                categories=["Epoch 1", "Epoch 2"],
                prior_samples=rng.lognormal(size=(4000, 2)),
                posterior_samples=rng.lognormal(size=(1000, 2)),
                color="#4477AA",
            ),
        ]

        chart = prior_posterior_rates_chart(panels, title="test")

        spec = chart.to_dict()
        all_rows = [row for rows in spec["datasets"].values() for row in rows]
        prior_summary = [
            r for r in all_rows if r["kind"] == "Prior" and r["row_type"] == "summary"
        ]
        posterior_summary = [
            r
            for r in all_rows
            if r["kind"] == "Posterior" and r["row_type"] == "summary"
        ]
        assert len(prior_summary) == 0
        assert len(posterior_summary) == 2  # one per category

    def test_multi_panel_stacks_vertically(self) -> None:
        rng = np.random.default_rng(3)
        panels = [
            PriorPosteriorRatePanel(
                label="Birth Rate",
                categories=["Epoch 1"],
                prior_samples=rng.lognormal(size=(4000, 1)),
                posterior_samples=rng.lognormal(size=(1000, 1)),
                color="#4477AA",
            ),
            PriorPosteriorRatePanel(
                label="Death Rate",
                categories=["Epoch 1"],
                prior_samples=rng.lognormal(size=(4000, 1)),
                posterior_samples=rng.lognormal(size=(1000, 1)),
                color="#EE6677",
            ),
        ]

        chart = prior_posterior_rates_chart(panels, title="test")

        spec = chart.to_dict()
        assert len(spec["vconcat"]) == 2

    def test_log_scale_applies_to_posterior_median_layer(self) -> None:
        rng = np.random.default_rng(4)
        panels = [
            PriorPosteriorRatePanel(
                label="Half-life",
                categories=["Epoch 1"],
                prior_samples=rng.lognormal(mean=5, sigma=1, size=(4000, 1)),
                posterior_samples=rng.lognormal(mean=5, sigma=1, size=(1000, 1)),
                color="#228833",
            ),
        ]

        chart = prior_posterior_rates_chart(panels, title="test", log_scale=True)

        layer = chart.to_dict()["vconcat"][0]["spec"]["layer"]
        median_x = layer[2]["encoding"]["x"]  # layer[2] = median_point
        assert median_x.get("scale", {}).get("type") == "log"

    def test_x_axis_title_has_no_stray_field_names(self) -> None:
        """Same ", ci_low" comma-join regression as posterior_rates_chart."""
        rng = np.random.default_rng(9)
        panels = [
            PriorPosteriorRatePanel(
                label="Birth Rate",
                categories=["Epoch 1"],
                prior_samples=rng.lognormal(size=(4000, 1)),
                posterior_samples=rng.lognormal(size=(1000, 1)),
                color="#4477AA",
            ),
        ]

        chart = prior_posterior_rates_chart(panels, title="test", x_title="Rate")

        layer = chart.to_dict()["vconcat"][0]["spec"]["layer"]
        median_x = layer[2]["encoding"]["x"]  # median_point
        ci_x = layer[3]["encoding"]["x"]  # ci_bar
        assert median_x["title"] == "Rate"
        assert ci_x["title"] == "Rate"


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
