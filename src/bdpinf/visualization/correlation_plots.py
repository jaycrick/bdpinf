"""Posterior rate correlation matrix and its diverging heatmap.

:func:`posterior_rate_correlation` builds a labelled births-then-deaths
correlation matrix from per-epoch posterior draws (or any other set of
columns, e.g. section indicators). :func:`correlation_heatmap_chart` renders
any such square, labelled correlation matrix as an Altair heatmap using the
project's shared :data:`~bdpinf.visualization.altair_theme.DIVERGING_RDBU`
palette, centred at zero with the numeric value overlaid on each cell.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bdpinf.visualization.altair_theme import DIVERGING_RDBU


def posterior_rate_correlation(
    birth_samples: NDArray[np.float64],
    death_samples: NDArray[np.float64],
    epoch_names: list[str],
    *,
    birth_symbol: str = "λ",
    death_symbol: str = "μ",
) -> pd.DataFrame:
    """Correlation matrix of per-epoch birth and death rate posterior draws.

    Parameters
    ----------
    birth_samples, death_samples : NDArray[np.float64]
        Posterior draws, shape ``[n_draws, n_epochs]``.
    epoch_names : list[str]
        Epoch labels, length ``n_epochs``.
    birth_symbol, death_symbol : str
        Prefix used to label birth/death columns (e.g. ``"λ Ante Nicaea"``).

    Returns
    -------
    pd.DataFrame
        Square, labelled correlation matrix ordered births-then-deaths.
        Zero-variance columns (fixed/constrained parameters) are dropped
        before computing correlations, since they are undefined (0/0) and
        would otherwise surface as ``NaN`` rows/columns.
    """
    births = np.asarray(birth_samples, dtype=np.float64)
    deaths = np.asarray(death_samples, dtype=np.float64)

    labels = [f"{birth_symbol} {name}" for name in epoch_names] + [
        f"{death_symbol} {name}" for name in epoch_names
    ]
    stacked = np.hstack([births, deaths])

    keep = np.std(stacked, axis=0) > 0
    kept_labels = [label for label, k in zip(labels, keep, strict=True) if k]
    corr = np.corrcoef(stacked[:, keep], rowvar=False)

    return pd.DataFrame(corr, index=kept_labels, columns=kept_labels)


def correlation_heatmap_chart(
    corr: pd.DataFrame,
    *,
    title: str,
    width: int = 360,
    height: int = 360,
    text_format: str = ".2f",
) -> alt.LayerChart:
    """Render a square correlation matrix as a diverging heatmap with labels.

    Colour uses the shared
    :data:`~bdpinf.visualization.altair_theme.DIVERGING_RDBU` palette,
    reversed so the domain ``[-1, 1]`` maps low-to-high against the palette's
    red-to-blue order (``+1`` blue, ``0`` white, ``-1`` red). The numeric
    value is overlaid on each cell, switching to white text on saturated cells
    for contrast.

    Parameters
    ----------
    corr : pd.DataFrame
        Square correlation matrix; row/column labels come from the index and
        columns (see :func:`posterior_rate_correlation`).
    title : str
        Chart title.
    width, height : int
        Chart dimensions in pixels.
    text_format : str
        ``d3-format`` string for the overlaid correlation values.
    """
    labels = list(corr.index)
    corr_long = (
        corr.rename_axis("row")
        .reset_index()
        .melt(id_vars="row", var_name="col", value_name="correlation")
    )

    heat_base = alt.Chart(corr_long).encode(
        x=alt.X("col:N", title=None, sort=labels, axis=alt.Axis(grid=False)),
        y=alt.Y("row:N", title=None, sort=labels, axis=alt.Axis(grid=False)),
    )
    heat_cells = heat_base.mark_rect().encode(
        color=alt.Color(
            "correlation:Q",
            title="Correlation",
            scale=alt.Scale(domain=[-1, 1], range=list(reversed(DIVERGING_RDBU))),
        ),
    )
    heat_labels = heat_base.mark_text(baseline="middle").encode(
        text=alt.Text("correlation:Q", format=text_format),
        color=alt.condition(
            "abs(datum.correlation) > 0.5", alt.value("white"), alt.value("black")
        ),
    )
    return (heat_cells + heat_labels).properties(
        title=title, width=width, height=height
    )


__all__ = [
    "correlation_heatmap_chart",
    "posterior_rate_correlation",
]
