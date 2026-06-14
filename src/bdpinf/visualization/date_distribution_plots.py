"""Altair helpers for manuscript date-distribution exploration figures.

The recurring "date-range KDE" comparison — a midpoint histogram, the
weighted date-range KDE curve, and a resampled-date histogram overlaid on one
axis — appears across several data-exploration notebooks, so it lives here as a
single reusable builder rather than being re-coded inline per notebook.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from inf_bp_texts.visualization.altair_theme import PTOLQUAL

_MIDPOINT_LABEL = "Midpoint (parsed_year)"
_KDE_LABEL = "Weighted date-range KDE"
_RESAMPLED_LABEL = "Resampled dates"


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
    """
    y_title = f"Manuscripts per {int(bin_width)}-year bin"
    color = alt.Color(
        "series:N",
        title=None,
        scale=alt.Scale(
            domain=[_MIDPOINT_LABEL, _KDE_LABEL, _RESAMPLED_LABEL],
            range=[PTOLQUAL[0], PTOLQUAL[1], PTOLQUAL[2]],
        ),
        legend=alt.Legend(orient="top"),
    )

    def _histogram(years: ArrayLike, label: str) -> alt.Chart:
        frame = pd.DataFrame(
            {x_title: np.asarray(years, dtype=np.float64), "series": label}
        )
        return (
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
            )
        )

    kde_frame = pd.DataFrame(
        {
            x_title: np.asarray(year_grid, dtype=np.float64),
            "value": np.asarray(density, dtype=np.float64) * n * bin_width,
            "series": _KDE_LABEL,
        }
    )
    kde_line = (
        alt.Chart(kde_frame)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(f"{x_title}:Q"),
            y=alt.Y("value:Q", title=y_title),
            color=color,
        )
    )

    return alt.layer(
        _histogram(midpoint_years, _MIDPOINT_LABEL),
        _histogram(resampled_years, _RESAMPLED_LABEL),
        kde_line,
    ).properties(title=title)
