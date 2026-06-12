"""Observed-creation-year histogram with fitted birth-time density overlay.

This module provides a single reusable function,
:func:`plot_creation_year_fit`, that renders the observed manuscript creation
years as a histogram and overlays the theoretical birth-time density computed
from MLE birth/death rates.  The same function is used by both the
single-epoch and multi-epoch NTVMR inference notebooks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from inf_bp_texts.visualization.plotting import (
    apply_professional_styling,
    apply_tufte_axes,
    ptolqual,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def plot_creation_year_fit(
    ages: NDArray[np.float64],
    times: torch.Tensor,
    start_time: float,
    birth_rate: float,
    death_rate: float,
    *,
    title: str,
    nbins: int = 30,
) -> go.Figure:
    """Plot observed creation years against the theoretical birth-time density.

    Converts manuscript ages (relative to the last epoch end) to calendar
    creation years, draws a histogram, and overlays the
    :func:`~inf_bp_texts.inference.calculate_unknown_spine_mixture_birth_density`
    density scaled to match the histogram counts.

    Parameters
    ----------
    ages : NDArray[np.float64]
        Manuscript ages in years (``age = last_epoch_end - creation_year``).
    times : torch.Tensor
        Adjusted epoch end times from
        ``BirthDeathProcess.adjusted_epoch_end_times``.
    start_time : float
        Epoch start year in calendar CE, used to convert relative birth times
        back to calendar years (``calendar_year = start_time + birth_rel``).
    birth_rate : float
        MLE birth rate (λ) used for the theoretical overlay.
    death_rate : float
        MLE death rate (μ) used for the theoretical overlay.
    title : str
        Figure title.
    nbins : int
        Number of histogram bins (default 30).

    Returns
    -------
    go.Figure
        Plotly figure with professional styling applied.
    """
    import torch  # noqa: PLC0415

    from inf_bp_texts.inference import (  # noqa: PLC0415
        calculate_unknown_spine_mixture_birth_density,
    )

    fig = go.Figure()

    # Convert ages → calendar creation years.
    # age = times[-1] - birth_rel   ⇒   birth_rel = times[-1] - age
    # calendar_year = start_time + birth_rel
    observed_birth_rel: NDArray[np.float64] = float(times[-1]) - ages
    observed_birth_years: NDArray[np.float64] = start_time + observed_birth_rel

    fig.add_trace(
        go.Histogram(
            x=observed_birth_years,
            nbinsx=nbins,
            name="Observed Creation Years",
            marker_color=ptolqual[0],
            opacity=0.6,
            showlegend=True,
        )
    )

    n = len(ages)
    if n > 0 and birth_rate > 0.0 and death_rate >= 0.0 and ages.size > 0:
        try:
            age_grid = np.linspace(0.0, float(ages.max()), 400)
            birth_grid_rel = float(times[-1]) - age_grid
            birth_grid_years = start_time + birth_grid_rel

            theoretical_birth_density: NDArray[Any] = (
                calculate_unknown_spine_mixture_birth_density(
                    birth_rates=birth_rate,
                    death_rates=death_rate,
                    birth_time_grid=torch.as_tensor(
                        birth_grid_rel, dtype=torch.float64
                    ),
                    times=times,
                    num_manuscripts=n,
                )
                .detach()
                .cpu()
                .numpy()
            )

            y_min = float(observed_birth_years.min())
            y_max = float(observed_birth_years.max())
            bin_width = (y_max - y_min) / nbins if y_max > y_min else 1.0
            theoretical_density_scaled = theoretical_birth_density * n * bin_width

            fig.add_trace(
                go.Scatter(
                    x=birth_grid_years,
                    y=theoretical_density_scaled,
                    mode="lines",
                    name="Fitted Creation-Year Distribution",
                    line={"color": ptolqual[1], "width": 3},
                    showlegend=True,
                )
            )
        except Exception as exc:
            logger.warning("Could not add theoretical overlay: %s", exc)

    fig = apply_professional_styling(
        fig,
        title,
        "Creation year (CE)",
        "Frequency",
    )
    fig = apply_tufte_axes(fig)
    fig.update_xaxes(tickformat=",d")
    fig.update_yaxes(tickformat=",d")
    return fig  # type: ignore[return-value]
