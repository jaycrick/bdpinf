"""Altair helpers for manuscript date-distribution exploration figures.

The recurring "date-range KDE" comparison — a midpoint histogram, the
weighted date-range KDE curve, and a resampled-date histogram overlaid on one
axis — appears across several data-exploration notebooks, so it lives here as a
single reusable builder rather than being re-coded inline per notebook.
"""

from __future__ import annotations

from collections.abc import Sequence

import altair as alt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from inf_bp_texts.visualization.altair_theme import PTOLQUAL

_MIDPOINT_LABEL = "Midpoint (parsed_year)"
_KDE_LABEL = "Weighted date-range KDE"
_RESAMPLED_LABEL = "Resampled dates"
_GROUPED_KDE_LABEL = "Weighted KDE"
_GROUPED_RESAMPLED_LABEL = "Resampled dates"


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
        _histogram(midpoint_years, midpoint_label),
        _histogram(resampled_years, _RESAMPLED_LABEL),
        kde_line,
    ).properties(title=title)


def plot_grouped_date_distribution(
    midyear_frame: pd.DataFrame,
    year_grid: ArrayLike,
    density: ArrayLike,
    resampled_mids: ArrayLike,
    resampled_counts: ArrayLike,
    *,
    n: int,
    bin_width: float,
    group_order: Sequence[str],
    group_colors: Sequence[str],
    title: str,
    x_title: str = "Year (CE)",
) -> alt.LayerChart:
    """Stacked per-group midpoint histograms with KDE and resampled overlays.

    Parameters
    ----------
    midyear_frame : pd.DataFrame
        Long-form per-manuscript frame with a ``year`` column (midpoint) and a
        ``group`` column (group label); the histograms stack by group.
    year_grid, density : ArrayLike
        Weighted date-range KDE grid and density, scaled to counts via
        ``density * n * bin_width``.
    resampled_mids, resampled_counts : ArrayLike
        Bin midpoints and counts of the resampled dates, drawn as a dotted
        frequency-polygon line.
    n : int
        Manuscript count used to scale the KDE.
    bin_width : float
        Histogram bin width in years.
    group_order, group_colors : Sequence[str]
        Group labels (stacking order) and their matching colours.
    title : str
        Figure title.
    x_title : str
        X-axis label (default ``"Year (CE)"``).
    """
    y_title = f"Manuscripts per {int(bin_width)}-year bin"
    domain = [*group_order, _GROUPED_KDE_LABEL, _GROUPED_RESAMPLED_LABEL]
    color = alt.Color(
        "series:N",
        title=None,
        scale=alt.Scale(
            domain=domain,
            range=[*group_colors, PTOLQUAL[4], PTOLQUAL[5]],
        ),
        legend=alt.Legend(orient="top"),
    )

    histograms = (
        alt.Chart(midyear_frame.rename(columns={"year": x_title, "group": "series"}))
        .mark_bar(opacity=0.6)
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
    kde_line = (
        alt.Chart(
            pd.DataFrame(
                {
                    x_title: np.asarray(year_grid, dtype=np.float64),
                    "value": np.asarray(density, dtype=np.float64) * n * bin_width,
                    "series": _GROUPED_KDE_LABEL,
                }
            )
        )
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(f"{x_title}:Q"),
            y=alt.Y("value:Q", title=y_title),
            color=color,
        )
    )
    resampled_line = (
        alt.Chart(
            pd.DataFrame(
                {
                    x_title: np.asarray(resampled_mids, dtype=np.float64),
                    "value": np.asarray(resampled_counts, dtype=np.float64),
                    "series": _GROUPED_RESAMPLED_LABEL,
                }
            )
        )
        .mark_line(strokeWidth=1.5, strokeDash=[2, 2])
        .encode(
            x=alt.X(f"{x_title}:Q"),
            y=alt.Y("value:Q", title=y_title),
            color=color,
        )
    )

    return alt.layer(histograms, kde_line, resampled_line).properties(title=title)
