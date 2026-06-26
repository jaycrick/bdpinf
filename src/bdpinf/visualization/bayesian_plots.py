"""Altair rendering functions for Bayesian posterior diagnostics.

The four `04_bayesian` notebooks each build the same handful of figure
archetypes (prior vs. posterior, posterior rates with credible intervals,
MCMC trace plots, and posterior predictive checks) for different models and
data subsets. This module gives each archetype one tested builder instead of
the per-notebook ``make_subplots`` blocks the migration replaces.
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
    """One parameter's bar-with-credible-interval panel."""

    label: str
    categories: Sequence[str]
    values: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    color: str


def posterior_rates_chart(
    panels: Sequence[RatePanel],
    *,
    title: str,
    x_title: str = "Category",
    y_title: str = "Rate per year",
) -> alt.HConcatChart:
    """Side-by-side bar charts of posterior point estimates with 95% CI whiskers."""
    charts = []
    for panel in panels:
        categories = list(panel.categories)
        frame = pd.DataFrame(
            {
                "category": categories,
                "value": np.asarray(panel.values, dtype=np.float64),
                "lower": np.asarray(panel.ci_lower, dtype=np.float64),
                "upper": np.asarray(panel.ci_upper, dtype=np.float64),
            }
        )
        base = alt.Chart(frame).encode(
            x=alt.X("category:N", title=x_title, sort=categories),
        )
        bars = base.mark_bar(color=panel.color, opacity=0.85).encode(
            y=alt.Y("value:Q", title=y_title),
        )
        whiskers = base.mark_rule(color="black", strokeWidth=1.5).encode(
            y=alt.Y("lower:Q", title=y_title),
            y2="upper:Q",
        )
        charts.append(
            (bars + whiskers).properties(
                title=panel.label, width=_PANEL_WIDTH, height=_PANEL_HEIGHT
            )
        )
    return alt.hconcat(*charts).properties(title=title)


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
