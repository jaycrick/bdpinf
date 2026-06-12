"""Observed-creation-year histogram with fitted birth-time density overlay.

This module provides the shared Altair builder :func:`creation_year_fit_chart`
(observed-year histogram layered with a scaled theoretical density) and the
notebook-facing :func:`plot_creation_year_fit`, which computes the density from
MLE birth/death rates and delegates to the builder.  The same builder backs the
multi-epoch ``save_multi_epoch_birth_time_figure`` saver, so the histogram +
density-scaling logic lives in exactly one place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from inf_bp_texts.visualization.altair_theme import PTOLQUAL

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def _scale_density_to_histogram(
    density: NDArray[np.float64],
    n: int,
    year_min: float,
    year_max: float,
    nbins: int,
) -> NDArray[np.float64] | None:
    """Scale a probability density to histogram-count units.

    Returns ``None`` when all observations share one year (``year_max <=
    year_min``): any ``bin_width`` guess would then mis-scale the overlay by
    orders of magnitude, so the density curve should be omitted instead.
    """
    if year_max <= year_min:
        return None
    bin_width = (year_max - year_min) / nbins
    return np.asarray(density, dtype=np.float64) * n * bin_width


def creation_year_fit_chart(
    observed_years: NDArray[np.float64],
    density_years: NDArray[np.float64] | None,
    density_values: NDArray[np.float64] | None,
    *,
    title: str,
    nbins: int = 30,
    x_title: str = "Creation year (CE)",
    observed_label: str = "Observed Creation Years",
    density_label: str = "Fitted Creation-Year Distribution",
) -> alt.LayerChart:
    """Layer an observed-year histogram with an optional scaled density curve.

    The density curve (when supplied and scalable) is scaled into the same
    frequency units as the histogram so both share one y-axis, and both series
    share a colour legend.
    """
    observed = np.asarray(observed_years, dtype=np.float64)
    color_scale = alt.Scale(
        domain=[observed_label, density_label],
        range=[PTOLQUAL[0], PTOLQUAL[1]],
    )
    legend = alt.Legend(title=None)

    histogram = (
        alt.Chart(pd.DataFrame({x_title: observed, "series": observed_label}))
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X(
                f"{x_title}:Q",
                bin=alt.Bin(maxbins=nbins),
                title=x_title,
                axis=alt.Axis(format="d"),
            ),
            y=alt.Y("count():Q", title="Frequency"),
            color=alt.Color("series:N", scale=color_scale, legend=legend),
        )
    )

    layers: list[alt.Chart] = [histogram]
    if density_years is not None and density_values is not None and observed.size > 0:
        scaled = _scale_density_to_histogram(
            density_values,
            observed.size,
            float(observed.min()),
            float(observed.max()),
            nbins,
        )
        if scaled is None:
            logger.warning(
                "Skipping theoretical overlay: all %d observations share the same"
                " year.",
                observed.size,
            )
        else:
            density_curve = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            x_title: np.asarray(density_years, dtype=np.float64),
                            "Frequency": scaled,
                            "series": density_label,
                        }
                    )
                )
                .mark_line(strokeWidth=3)
                .encode(
                    x=alt.X(f"{x_title}:Q"),
                    y=alt.Y("Frequency:Q"),
                    color=alt.Color("series:N", scale=color_scale, legend=legend),
                )
            )
            layers.append(density_curve)

    return alt.layer(*layers).properties(title=title)


def plot_creation_year_fit(
    ages: NDArray[np.float64],
    times: torch.Tensor,
    start_time: float,
    birth_rate: float,
    death_rate: float,
    *,
    title: str,
    nbins: int = 30,
) -> alt.LayerChart:
    """Plot observed creation years against the theoretical birth-time density.

    Converts manuscript ages (relative to the last epoch end) to calendar
    creation years and overlays the
    :func:`~inf_bp_texts.inference.calculate_unknown_spine_mixture_birth_density`
    density scaled to the histogram counts.

    Parameters
    ----------
    ages : NDArray[np.float64]
        Manuscript ages in years (``age = last_epoch_end - creation_year``).
    times : torch.Tensor
        Adjusted epoch end times from
        ``BirthDeathProcess.adjusted_epoch_end_times``.
    start_time : float
        Epoch start year in calendar CE.
    birth_rate, death_rate : float
        MLE rates (λ, μ) used for the theoretical overlay.
    title : str
        Figure title.
    nbins : int
        Number of histogram bins (default 30).
    """
    import torch  # noqa: PLC0415

    from inf_bp_texts.inference import (  # noqa: PLC0415
        calculate_unknown_spine_mixture_birth_density,
    )

    # age = times[-1] - birth_rel  ⇒  calendar_year = start_time + (times[-1] - age)
    observed_birth_years = start_time + (float(times[-1]) - ages)

    density_years: NDArray[np.float64] | None = None
    density_values: NDArray[np.float64] | None = None
    n = len(ages)
    if n > 0 and birth_rate > 0.0 and death_rate >= 0.0 and ages.size > 0:
        try:
            age_grid = np.linspace(0.0, float(ages.max()), 400)
            birth_grid_rel = float(times[-1]) - age_grid
            density_years = start_time + birth_grid_rel
            birth_grid_tensor = torch.as_tensor(birth_grid_rel, dtype=torch.float64)
            density_values = (
                calculate_unknown_spine_mixture_birth_density(
                    birth_rates=birth_rate,
                    death_rates=death_rate,
                    birth_time_grid=birth_grid_tensor,
                    times=times,
                    num_manuscripts=n,
                )
                .detach()
                .cpu()
                .numpy()
            )
        except Exception as exc:  # noqa: BLE001 - overlay is optional; skip on failure
            logger.warning("Could not compute theoretical overlay: %s", exc)
            density_years = None
            density_values = None

    return creation_year_fit_chart(
        observed_birth_years,
        density_years,
        density_values,
        title=title,
        nbins=nbins,
    )
