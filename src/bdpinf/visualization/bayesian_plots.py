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

from inf_bp_texts.visualization.altair_theme import PTOLQUAL

_PANEL_WIDTH = 360
_PANEL_HEIGHT = 300

# Ridgeline (joyplot) layout constants
_RIDGE_STEP = 40  # pixel height of each facet row
_RIDGE_OVERLAP = 1  # fraction of _RIDGE_STEP that ridge peaks bleed into the row above
_RIDGE_KDE_GRID = 200  # KDE evaluation grid points per category


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


def posterior_rates_chart(
    panels: Sequence[RatePanel],
    *,
    title: str,
    x_title: str = "Rate per year",
    category_title: str = "Category",
) -> alt.FacetChart:
    """Ridgeline (joyplot) of per-category posterior rate distributions.

    Each row facets one category; each column facets one parameter panel
    (e.g. birth / death rate).  Within every cell a boundary-corrected KDE
    area ridge is drawn, with the posterior median overlaid as a filled
    ``mark_point`` and the equal-tailed credible interval as a ticked
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
    """
    from inf_bp_texts.analysis.validation_helpers import boundary_corrected_kde_at_zero

    color_scale = alt.Scale(
        domain=[p.label for p in panels],
        range=[p.color for p in panels],
    )

    # Build one combined DataFrame so Altair can place a single data source at
    # the facet top level (required by Altair when layers have no shared data).
    # Summary rows carry the median/CI marker positions; density rows carry the
    # KDE curve.  A "row_type" column lets each layer filter its own subset.
    combined_rows: list[dict[str, object]] = []

    for panel in panels:
        categories = list(panel.categories)
        samp = np.asarray(panel.samples, dtype=np.float64)
        tail = (1.0 - panel.ci_prob) / 2.0
        for j, cat in enumerate(categories):
            col = samp[:, j]
            median = float(np.median(col))
            ci_lo, ci_hi = (float(v) for v in np.quantile(col, [tail, 1.0 - tail]))
            # One summary row per (panel, category): rate=median for x-position.
            combined_rows.append(
                {
                    "panel": panel.label,
                    "category": cat,
                    "rate": median,
                    "density": 0.0,  # baseline y-position
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "row_type": "summary",
                }
            )
            x_vals, y_vals = boundary_corrected_kde_at_zero(col, n_grid=_RIDGE_KDE_GRID)
            if x_vals is None:
                continue
            for x, y in zip(x_vals, y_vals, strict=True):
                combined_rows.append(
                    {
                        "panel": panel.label,
                        "category": cat,
                        "rate": float(x),
                        "density": float(y),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "row_type": "density",
                    }
                )

    combined_df = pd.DataFrame(combined_rows)

    y_scale = alt.Scale(range=[_RIDGE_STEP, -_RIDGE_STEP * _RIDGE_OVERLAP])

    # Each layer uses alt.Chart() (no data) and inherits from the facet-level
    # data; transform_filter selects the relevant subset per layer.
    area = (
        alt.Chart()
        .transform_filter("datum.row_type === 'density'")
        .mark_area(
            interpolate="monotone",
            fillOpacity=0.8,
            stroke="lightgray",
            strokeWidth=0.5,
        )
        .encode(
            x=alt.X("rate:Q", title=x_title),
            y=alt.Y("density:Q", axis=None, scale=y_scale),
            fill=alt.Fill("panel:N", scale=color_scale, legend=None),
        )
    )
    median_point = (
        alt.Chart()
        .transform_filter("datum.row_type === 'summary'")
        .mark_point(filled=True, size=40, color="black")
        .encode(
            x=alt.X("rate:Q"),  # rate=median for summary rows; shares x scale with area
            y=alt.Y("density:Q", scale=y_scale),
        )
    )
    ci_bar = (
        alt.Chart()
        .transform_filter("datum.row_type === 'summary'")
        .mark_errorbar(ticks=True, color="black")
        .encode(
            x=alt.X("ci_low:Q"),
            x2=alt.X2("ci_high:Q"),
            y=alt.Y("density:Q", scale=y_scale),
        )
    )
    return (
        alt.layer(area, median_point, ci_bar)
        .facet(
            row=alt.Row(
                "category:N",
                title=category_title,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
            column=alt.Column("panel:N", title=None),
            data=combined_df,
        )
        .resolve_scale(y="independent")
        .properties(title=title, bounds="flush")
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )


@dataclass(frozen=True)
class TracePanel:
    """One parameter's per-chain MCMC trace panel."""

    label: str
    chains: NDArray[np.float64]  # shape (n_chains, n_draws)


def trace_chart(
    panels: Sequence[TracePanel],
    *,
    title: str,
    columns: int = 2,
    y_title: str = "log(Rate)",
) -> alt.ConcatChart:
    """Grid of per-chain trace lines, one panel per parameter/epoch shown."""
    charts = []
    for index, panel in enumerate(panels):
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
        # Only the first panel carries a legend; the colour-per-chain mapping is
        # identical across panels (same categorical values, shared theme range).
        legend = alt.Legend(title=None) if index == 0 else None
        charts.append(
            alt.Chart(frame)
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("Draw:Q", title="Draw"),
                y=alt.Y("value:Q", title=y_title),
                color=alt.Color("Chain:N", legend=legend),
            )
            .properties(title=panel.label, width=_PANEL_WIDTH, height=250)
        )
    return alt.concat(*charts, columns=columns).properties(title=title)


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
