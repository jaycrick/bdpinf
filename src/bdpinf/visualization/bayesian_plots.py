"""Altair rendering functions for Bayesian posterior diagnostics.

The four `04_bayesian` notebooks each build the same handful of figure
archetypes (prior vs. posterior, posterior rate ridgeline plots, MCMC trace
plots, and posterior predictive checks) for different models and data subsets.
This module gives each archetype one tested builder instead of the per-notebook
``make_subplots`` blocks the migration replaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from inf_bp_texts.visualization.altair_theme import PTOLQUAL

_PANEL_WIDTH = 360
_PANEL_HEIGHT = 300

# Ridgeline (joyplot) layout constants
_RIDGE_STEP = 20  # pixel height of each facet row
_RIDGE_OVERLAP = 1  # fraction of _RIDGE_STEP that ridge peaks bleed into the row above
_RIDGE_MAXBINS = 30  # max bins for the in-spec Vega-Lite binned density


@dataclass(frozen=True)
class PriorPosteriorPanel:
    """One parameter's prior-density-vs-posterior-histogram panel."""

    label: str
    posterior_samples: NDArray[np.float64]
    prior_x: NDArray[np.float64]
    prior_pdf: NDArray[np.float64]
    color: str
    nbins: int = 40


def prior_posterior_chart(
    panels: Sequence[PriorPosteriorPanel],
    *,
    title: str,
    x_title: str = "log(rate)",
) -> alt.HConcatChart:
    """Side-by-side panels overlaying a posterior histogram with the prior density.

    The prior is a continuous density and the posterior is a count histogram;
    each panel rescales its prior curve into that panel's histogram-count
    units (bin width derived from the posterior samples' own range) so both
    series share one y-axis, mirroring the scaling done by
    :func:`~.age_fit_plot.creation_year_fit_chart` for its density overlay.
    """
    charts = []
    for panel in panels:
        samples = np.asarray(panel.posterior_samples, dtype=np.float64)
        hist = (
            alt.Chart(pd.DataFrame({"value": samples}))
            .mark_bar(color=panel.color, opacity=0.65)
            .encode(
                x=alt.X("value:Q", bin=alt.Bin(maxbins=panel.nbins), title=x_title),
                y=alt.Y("count():Q", title="Count"),
            )
        )
        layers: list[alt.Chart] = [hist]
        sample_min = float(samples.min()) if samples.size else 0.0
        sample_max = float(samples.max()) if samples.size else 0.0
        if samples.size and sample_max > sample_min:
            bin_width = (sample_max - sample_min) / panel.nbins
            scaled_pdf = np.asarray(panel.prior_pdf, dtype=np.float64) * (
                samples.size * bin_width
            )
            prior_line = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "value": np.asarray(panel.prior_x, dtype=np.float64),
                            "count": scaled_pdf,
                        }
                    )
                )
                .mark_line(strokeDash=[4, 4], strokeWidth=2, color=PTOLQUAL[2])
                .encode(x="value:Q", y="count:Q")
            )
            layers.append(prior_line)
        charts.append(
            alt.layer(*layers).properties(
                title=panel.label, width=_PANEL_WIDTH, height=_PANEL_HEIGHT
            )
        )
    return alt.hconcat(*charts).properties(title=title)


@dataclass(frozen=True)
class RatePanel:
    """One parameter's ridgeline panel of per-category posterior samples.

    ``samples`` has shape ``(n_draws, n_categories)``; ``samples[:, j]`` is
    the full posterior for ``categories[j]``.  The credible interval is the
    equal-tailed *ci_prob* interval (default 95%).
    """

    label: str
    categories: Sequence[str]
    samples: NDArray[np.float64]  # shape (n_draws, n_categories)
    color: str
    ci_prob: float = 0.95


def _panel_rows(panel: RatePanel) -> list[dict[str, object]]:
    """Build combined draw + summary rows for one RatePanel."""
    rows: list[dict[str, object]] = []
    samp = np.asarray(panel.samples, dtype=np.float64)
    tail = (1.0 - panel.ci_prob) / 2.0
    for j, cat in enumerate(list(panel.categories)):
        col = samp[:, j]
        median = float(np.median(col))
        ci_lo, ci_hi = (float(v) for v in np.quantile(col, [tail, 1.0 - tail]))
        rows.append(
            {
                "panel": panel.label,
                "category": cat,
                "rate": median,
                "value": 0.0,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "row_type": "summary",
            }
        )
        rows.extend(
            {
                "panel": panel.label,
                "category": cat,
                "rate": float(draw),
                "value": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "row_type": "draw",
            }
            for draw in col
        )
    return rows


def _ridgeline_layers(
    color_scale: alt.Scale,
    y_scale: alt.Scale,
    x_title: str,
    log_scale: bool = False,
) -> tuple[alt.Chart, alt.Chart, alt.Chart]:
    """Build the three ridgeline mark layers shared by both layout modes.

    Parameters
    ----------
    color_scale :
        Fill colour scale for the area layer.
    y_scale :
        Shared vertical scale (controls ridge row height/overlap).
    x_title :
        Label for the quantitative x-axis.
    log_scale :
        When ``True``, bin draws in log10 space and render the x-axis on a
        log scale.  Use for quantities that span orders of magnitude (e.g.
        half-lives).  The median and CI overlays also use a log-scale x-axis
        so they stay aligned with the binned-area ridge.
    """
    _x_scale = alt.Scale(type="log") if log_scale else alt.Undefined

    if log_scale:
        # Bin in log10 space so bins are equal-width on the log axis.
        # The calculate → bin → aggregate → impute pipeline mirrors the linear
        # path but operates on log10(rate) rather than rate directly.
        area = (
            alt.Chart()
            .transform_filter("datum.row_type === 'draw'")
            .transform_calculate(log_rate="log(datum.rate) / log(10)")
            .transform_bin(
                ["bin_max", "bin_min"],
                field="log_rate",
                bin=alt.Bin(maxbins=_RIDGE_MAXBINS),
            )
            .transform_aggregate(
                value="count()",
                groupby=["panel", "category", "bin_min", "bin_max"],
            )
            .transform_impute(
                impute="value",
                groupby=["panel", "category"],
                key="bin_min",
                value=0,
            )
            .transform_calculate(
                rate_min="pow(10, datum.bin_min)",
                rate_max="pow(10, datum.bin_max)",
            )
            .mark_area(
                interpolate="monotone",
                fillOpacity=0.8,
                stroke="lightgray",
                strokeWidth=0.5,
            )
            .encode(
                x=alt.X(
                    "rate_min:Q",
                    bin="binned",
                    title=x_title,
                    scale=_x_scale,
                ),
                x2=alt.X2("rate_max:Q"),
                y=alt.Y("value:Q", axis=None, scale=y_scale),
                fill=alt.Fill("panel:N", scale=color_scale, legend=None),
            )
        )
    else:
        area = (
            alt.Chart()
            .transform_filter("datum.row_type === 'draw'")
            .transform_bin(
                ["bin_max", "bin_min"],
                field="rate",
                bin=alt.Bin(maxbins=_RIDGE_MAXBINS),
            )
            .transform_aggregate(
                value="count()",
                groupby=["panel", "category", "bin_min", "bin_max"],
            )
            .transform_impute(
                impute="value",
                groupby=["panel", "category"],
                key="bin_min",
                value=0,
            )
            .mark_area(
                interpolate="monotone",
                fillOpacity=0.8,
                stroke="lightgray",
                strokeWidth=0.5,
            )
            .encode(
                x=alt.X("bin_min:Q", bin="binned", title=x_title),
                y=alt.Y("value:Q", axis=None, scale=y_scale),
                fill=alt.Fill("panel:N", scale=color_scale, legend=None),
            )
        )

    median_point = (
        alt.Chart()
        .transform_filter("datum.row_type === 'summary'")
        .mark_point(filled=True, size=40, color="black")
        .encode(
            x=alt.X("rate:Q", scale=_x_scale),
            y=alt.Y("value:Q", scale=y_scale),
        )
    )
    ci_bar = (
        alt.Chart()
        .transform_filter("datum.row_type === 'summary'")
        .mark_errorbar(ticks=True, color="black")
        .encode(
            x=alt.X("ci_low:Q", scale=_x_scale),
            x2=alt.X2("ci_high:Q"),
            y=alt.Y("value:Q", scale=y_scale),
        )
    )
    return area, median_point, ci_bar


def posterior_rates_chart(
    panels: Sequence[RatePanel],
    *,
    title: str,
    x_title: str = "Rate per year",
    category_title: str = "Category",
    vertical: bool = False,
    log_scale: bool = False,
) -> alt.FacetChart | alt.VConcatChart:
    """Ridgeline (joyplot) of per-category posterior rate distributions.

    Each row facets one category; columns facet parameter panels (birth vs.
    death) when ``vertical=False`` (default), or panels are stacked as separate
    facets when ``vertical=True``.  Within every cell a binned-count area ridge
    is drawn via Vega-Lite's ``transform_bin`` / ``transform_aggregate`` /
    ``transform_impute`` pipeline, with the posterior median overlaid as a
    filled ``mark_point`` and the equal-tailed credible interval as a ticked
    ``mark_errorbar`` along the row baseline.

    Parameters
    ----------
    panels:
        Per-parameter ridge panels; each carries raw posterior draws.
    title:
        Top-level figure title.
    x_title:
        Label for the rate (x) axis.
    category_title:
        Label for the category (row-facet) axis.
    vertical:
        When ``True``, stack panels vertically (one facet grid per panel)
        instead of side-by-side columns.
    log_scale:
        When ``True``, render the x-axis on a log scale and bin draws in
        log10 space so bins are equal-width on the axis.  Use for
        quantities that span several orders of magnitude (e.g. half-lives).
    """
    y_scale = alt.Scale(range=[_RIDGE_STEP, -_RIDGE_STEP * _RIDGE_OVERLAP])
    # Build category sort from the first panel so the y-axis preserves the
    # caller-provided order rather than defaulting to alphabetical.
    _category_sort = list(panels[0].categories) if panels else None

    if vertical:
        # One FacetChart per panel (row-facet only), then vconcat.
        # configure_* must be applied at the outermost level (vconcat), not on
        # individual FacetCharts, because Altair only allows config at top level.
        sub_charts: list[alt.FacetChart] = []
        for panel in panels:
            pscale = alt.Scale(domain=[panel.label], range=[panel.color])
            area, median_point, ci_bar = _ridgeline_layers(
                pscale, y_scale, x_title, log_scale
            )
            row_enc_v = alt.Row(
                "category:N",
                title=category_title,
                header=alt.Header(labelAngle=0, labelAlign="left"),
                sort=list(panel.categories),
            )
            sub_charts.append(
                alt.layer(area, median_point, ci_bar)
                .properties(height=_RIDGE_STEP)
                .facet(row=row_enc_v, data=pd.DataFrame(_panel_rows(panel)))
                .properties(title=panel.label, bounds="flush")
            )
        return (
            alt.vconcat(*sub_charts)
            .properties(title=title)
            .configure_facet(spacing=0)
            .configure_view(stroke=None)
        )

    # Default: column facet for panels, row facet for categories.
    color_scale = alt.Scale(
        domain=[p.label for p in panels],
        range=[p.color for p in panels],
    )
    area, median_point, ci_bar = _ridgeline_layers(
        color_scale, y_scale, x_title, log_scale
    )
    combined_df = pd.DataFrame([r for panel in panels for r in _panel_rows(panel)])
    row_enc = alt.Row(
        "category:N",
        title=category_title,
        header=alt.Header(labelAngle=0, labelAlign="left"),
        sort=_category_sort,
    )
    return (
        alt.layer(area, median_point, ci_bar)
        .properties(height=_RIDGE_STEP)
        .facet(
            row=row_enc,
            column=alt.Column("panel:N", title=None),
            data=combined_df,
        )
        .properties(title=title, bounds="flush")
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )


def prior_rate_samples(
    locs: NDArray[np.float64],
    scales: NDArray[np.float64],
    n_categories: int,
    n_quantiles: int = 4000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Deterministic quantile samples of log-normal birth/death priors.

    Rates are log-normal: ``rate = exp(Normal(loc, scale))``.  Rather than
    drawing random samples, each category's ridge is built from a fixed set of
    deterministic quantile points — standard-normal quantiles at Hazen
    plotting positions ``(i - 0.5) / n_quantiles``, transformed through
    ``exp(loc + scale * z)`` — so downstream charts are reproducible without
    an RNG seed.  This mirrors the deterministic quantile-resampling idiom
    used in :func:`~.analysis.date_density.quantile_resample_dates`.

    Parameters
    ----------
    locs, scales :
        Log-scale prior mean/std, shape ``[2 * n_categories]`` following the
        birth-then-death convention of
        :meth:`~inf_bp_texts.inference.bayesian.BayesianPriorConfig.resolve_tensors`
        (index ``i < n_categories`` is birth category ``i``; ``i >=
        n_categories`` is death category ``i - n_categories``).
    n_categories :
        Number of categories (epochs or subsets).
    n_quantiles :
        Number of deterministic quantile points per category ridge.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(birth_samples, death_samples)``, each shape
        ``[n_quantiles, n_categories]`` — usable directly as a
        :class:`RatePanel`'s ``samples``, including for splitting birth,
        death, and half-life into separate charts.
    """
    n = n_categories
    probs = (np.arange(1, n_quantiles + 1) - 0.5) / n_quantiles
    z = norm.ppf(probs)
    birth = np.exp(locs[:n][None, :] + scales[:n][None, :] * z[:, None])
    death = np.exp(locs[n:][None, :] + scales[n:][None, :] * z[:, None])
    return birth, death


def prior_rates_chart(
    locs: NDArray[np.float64],
    scales: NDArray[np.float64],
    categories: Sequence[str],
    *,
    title: str,
    category_title: str = "Epoch",
    n_quantiles: int = 4000,
    log_scale: bool = True,
    birth_color: str = PTOLQUAL[0],
    death_color: str = PTOLQUAL[1],
) -> alt.FacetChart | alt.VConcatChart:
    """Ridgeline of per-category prior rate distributions.

    Renders the priors in the same style as :func:`posterior_rates_chart`,
    combining birth and death into one figure.  Use :func:`prior_rate_samples`
    directly when birth, death, and half-life need separate figures.

    Parameters
    ----------
    locs, scales :
        Log-scale prior mean/std, shape ``[2 * n_categories]`` — see
        :func:`prior_rate_samples` for the birth-then-death index convention.
    categories :
        Category (epoch or subset) labels, length ``n_categories``.
    title :
        Top-level figure title.
    category_title :
        Label for the category (row-facet) axis.
    n_quantiles :
        Number of deterministic quantile points per category ridge.
    log_scale :
        Render the x-axis on a log scale (default ``True``, matching the
        log-normal prior support).
    birth_color, death_color :
        Fill colours for the birth-rate and death-rate panels.
    """
    birth, death = prior_rate_samples(locs, scales, len(categories), n_quantiles)
    return posterior_rates_chart(
        [
            RatePanel(
                label="Birth Rate λ (prior)",
                categories=categories,
                samples=birth,
                color=birth_color,
            ),
            RatePanel(
                label="Death Rate μ (prior)",
                categories=categories,
                samples=death,
                color=death_color,
            ),
        ],
        title=title,
        category_title=category_title,
        vertical=True,
        log_scale=log_scale,
    )


@dataclass(frozen=True)
class TracePanel:
    """One parameter's per-chain MCMC trace panel."""

    label: str
    chains: NDArray[np.float64]  # shape (n_chains, n_draws)


def trace_chart(
    panels: Sequence[TracePanel],
    *,
    y_title: str = "log(Rate)",
) -> list[alt.Chart]:
    """Per-chain trace lines, one independent chart per panel.

    Returns a plain list so callers can arrange charts with Quarto's
    ``# | layout-ncol:`` cell option (or the equivalent fenced-div syntax
    inside ``output: asis`` loops) rather than relying on Altair concatenation.

    The y-axis is fitted to the sampled range (``zero=False``) so trace
    variation is visible at full resolution without zero-anchored whitespace.
    """
    charts = []
    for panel in panels:
        chains = np.asarray(panel.chains, dtype=np.float64)
        n_chains, n_draws = chains.shape
        draws = np.arange(1, n_draws + 1)
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Draw": draws,
                        "value": chains[chain],
                        "Chain": f"Chain {chain + 1}",
                    }
                )
                for chain in range(n_chains)
            ],
            ignore_index=True,
        )
        charts.append(
            alt.Chart(frame)
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("Draw:Q", title="Draw"),
                # zero=False: log-rates are meaningful relative to themselves,
                # not to 0; matches the convention in plotting.py (log-likelihood
                # axis).  Without this the default zero=True wastes most of the
                # panel height on empty space below the sampled range.
                y=alt.Y(
                    "value:Q",
                    title=y_title,
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("Chain:N", legend=alt.Legend(title=None)),
            )
            .properties(title=panel.label, width=_PANEL_WIDTH, height=250)
        )
    return charts


def ppc_chart(
    replicated: NDArray[np.float64],
    observed_value: float,
    *,
    title: str,
    x_title: str = "Replicated statistic",
    replicated_label: str = "Replicated",
    observed_label: str = "Observed",
    nbins: int = 30,
) -> alt.LayerChart:
    """Histogram of a posterior-predictive replicated statistic vs. the observed value.

    Both series share one colour legend (replicated histogram vs. observed
    rule), the same idiom used by
    :func:`~.age_fit_plot.creation_year_fit_chart` for observed-vs-fitted
    overlays.
    """
    values = np.asarray(replicated, dtype=np.float64)
    color_scale = alt.Scale(
        domain=[replicated_label, observed_label],
        range=[PTOLQUAL[1], PTOLQUAL[0]],
    )
    legend = alt.Legend(title=None)

    histogram = (
        alt.Chart(pd.DataFrame({x_title: values, "series": replicated_label}))
        .mark_bar(opacity=0.65)
        .encode(
            x=alt.X(f"{x_title}:Q", bin=alt.Bin(maxbins=nbins), title=x_title),
            y=alt.Y("count():Q", title="Frequency"),
            color=alt.Color("series:N", scale=color_scale, legend=legend),
        )
    )
    observed_rule = (
        alt.Chart(
            pd.DataFrame({x_title: [float(observed_value)], "series": [observed_label]})
        )
        .mark_rule(strokeWidth=3, strokeDash=[4, 4])
        .encode(
            x=f"{x_title}:Q",
            color=alt.Color("series:N", scale=color_scale, legend=legend),
        )
    )
    return alt.layer(histogram, observed_rule).properties(title=title)
