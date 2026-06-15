"""Altair helpers for manuscript date-distribution exploration figures.

The recurring "date-range KDE" comparison — a midpoint histogram, the
weighted date-range KDE curve, and a resampled-date histogram overlaid on one
axis — appears across several data-exploration notebooks, so it lives here as a
single reusable builder rather than being re-coded inline per notebook.
"""

from __future__ import annotations

from typing import Any, Literal

import altair as alt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from inf_bp_texts.visualization.altair_theme import PTOLQUAL

_MIDPOINT_LABEL = "Midpoint (parsed_year)"
_KDE_LABEL = "Weighted date-range KDE"
_RESAMPLED_LABEL = "Resampled dates"
_DIMMED_OPACITY = 0.15

LegendInteraction = Literal["none", "filter", "highlight"]


def plot_date_range_kde_comparison(
    midpoint_years: ArrayLike,
    year_grid: ArrayLike,
    density: ArrayLike,
    resampled_years: ArrayLike,
    *,
    n: int,
    bin_width: float,
    title: str,
    x_title: str = "Year (CE)",
    midpoint_label: str = _MIDPOINT_LABEL,
    interaction: LegendInteraction = "none",
) -> alt.LayerChart:
    """Overlay the midpoint histogram, weighted KDE, and resampled histogram.

    Parameters
    ----------
    midpoint_years : NDArray[np.float64]
        Per-manuscript midpoint (``parsed_year``) values.
    year_grid, density : NDArray[np.float64]
        The weighted date-range KDE grid and its density; the density is scaled
        to histogram counts as ``density * n * bin_width``.
    resampled_years : NDArray[np.float64]
        Deterministic quantile-resampled exact dates.
    n : int
        Manuscript count used to scale the KDE to count units.
    bin_width : float
        Histogram bin width in years (shared by both histograms).
    title : str
        Figure title.
    x_title : str
        X-axis label (default ``"Year (CE)"``).
    midpoint_label : str
        Legend label for the midpoint histogram (default
        ``"Midpoint (parsed_year)"``); pass e.g. ``"Midpoint (midYear)"`` for
        datasets keyed on a different column.
    interaction : {"none", "filter", "highlight"}
        Legend-click behaviour in the interactive (HTML) render. ``"filter"``
        shows only the clicked series; ``"highlight"`` keeps every series but dims
        the unselected ones. Both are inert in the static PDF export (an empty
        selection matches all series, so the default appearance is unchanged).
    """
    y_title = f"Manuscripts per {int(bin_width)}-year bin"
    color = alt.Color(
        "series:N",
        title=None,
        scale=alt.Scale(
            domain=[midpoint_label, _KDE_LABEL, _RESAMPLED_LABEL],
            range=[PTOLQUAL[0], PTOLQUAL[1], PTOLQUAL[2]],
        ),
        legend=alt.Legend(orient="top"),
    )
    series_select = alt.selection_point(fields=["series"], bind="legend")

    def _highlight_opacity(base: float) -> dict[str, Any]:
        # Preserve each layer's natural opacity when selected/default so the
        # static PDF still reads as overlaid; dim the rest on legend click.
        if interaction == "highlight":
            return {
                "opacity": alt.condition(
                    series_select, alt.value(base), alt.value(_DIMMED_OPACITY)
                )
            }
        return {}

    def _maybe_filter(chart: alt.Chart) -> alt.Chart:
        return (
            chart.transform_filter(series_select) if interaction == "filter" else chart
        )

    def _histogram(years: ArrayLike, label: str) -> alt.Chart:
        frame = pd.DataFrame(
            {x_title: np.asarray(years, dtype=np.float64), "series": label}
        )
        return _maybe_filter(
            alt.Chart(frame)
            .mark_bar(opacity=0.5)
            .encode(
                x=alt.X(
                    f"{x_title}:Q",
                    bin=alt.Bin(step=bin_width),
                    title=x_title,
                    axis=alt.Axis(format="d"),
                ),
                y=alt.Y("count():Q", title=y_title),
                color=color,
                **_highlight_opacity(0.5),
            )
        )

    kde_frame = pd.DataFrame(
        {
            x_title: np.asarray(year_grid, dtype=np.float64),
            "value": np.asarray(density, dtype=np.float64) * n * bin_width,
            "series": _KDE_LABEL,
        }
    )
    kde_line = _maybe_filter(
        alt.Chart(kde_frame)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(f"{x_title}:Q"),
            y=alt.Y("value:Q", title=y_title),
            color=color,
            **_highlight_opacity(1.0),
        )
    )

    layered = alt.layer(
        _histogram(midpoint_years, midpoint_label),
        _histogram(resampled_years, _RESAMPLED_LABEL),
        kde_line,
    ).properties(title=title)
    if interaction != "none":
        layered = layered.add_params(series_select)
    return layered
