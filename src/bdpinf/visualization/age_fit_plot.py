"""Observed-creation-year histogram with fitted expected-count overlay.

This module provides the shared Altair builder :func:`creation_year_fit_chart`
(observed-year histogram layered with an expected-count histogram derived from
a fitted density) and the notebook-facing :func:`plot_creation_year_fit`, which
computes the density from MLE birth/death rates and delegates to the builder.
The same builder backs the multi-epoch ``save_multi_epoch_birth_time_figure``
saver, so the histogram + expected-count logic lives in exactly one place.
"""

from __future__ import annotations

import logging

import altair as alt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bdpinf.visualization.altair_theme import PTOLQUAL

logger = logging.getLogger(__name__)


def _expected_counts_per_bin(
    bin_edges: NDArray[np.float64],
    density_years: NDArray[np.float64],
    density_values: NDArray[np.float64],
    n: int,
) -> NDArray[np.float64]:
    """Compute expected count per bin from a fitted density.

    Approximates the density's integral over each bin as ``density(mid) *
    bin_width * n`` — the density evaluated at the bin midpoint, treated as
    locally constant across the bin. ``density_years`` must be sorted
    ascending (the caller is responsible for sorting).
    """
    mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = float(bin_edges[1] - bin_edges[0])
    density_at_mid = np.interp(mids, density_years, density_values, left=0.0, right=0.0)
    return density_at_mid * bin_width * n


def creation_year_fit_chart(
    observed_years: NDArray[np.float64],
    density_years: NDArray[np.float64] | None,
    density_values: NDArray[np.float64] | None,
    *,
    title: str,
    nbins: int = 30,
    x_title: str = "Creation Year",
    observed_label: str = "Observed",
    density_label: str = "Expected",
) -> alt.LayerChart:
    """Layer an observed-year histogram with an expected-count histogram.

    The fitted density (when supplied) is converted to an expected count per
    bin — density at the bin midpoint times bin width times the observed
    sample size — so both series are true histograms sharing one set of bin
    edges, one y-axis (``count``), and one colour legend. The shared x-domain
    is extended to cover the fitted distribution's range, even where it
    reaches beyond the observed years.
    """
    observed = np.asarray(observed_years, dtype=np.float64)
    color_scale = alt.Scale(
        domain=[observed_label, density_label],
        range=[PTOLQUAL[0], PTOLQUAL[1]],
    )
    legend = alt.Legend(title=None, orient="top")

    lo = float(observed.min()) if observed.size else 0.0
    hi = float(observed.max()) if observed.size else 1.0

    has_density = (
        density_years is not None and density_values is not None and observed.size > 0
    )
    dy: NDArray[np.float64] | None = None
    dv: NDArray[np.float64] | None = None
    if has_density:
        dy = np.asarray(density_years, dtype=np.float64)
        dv = np.asarray(density_values, dtype=np.float64)
        order = np.argsort(dy)
        dy, dv = dy[order], dv[order]
        lo = min(lo, float(dy.min()))
        hi = max(hi, float(dy.max()))

    if hi <= lo:
        if has_density:
            logger.warning(
                "Skipping theoretical overlay: all %d observations share the same"
                " year.",
                observed.size,
            )
        has_density = False
        hi = lo + 1.0  # degenerate range: keep the histogram itself renderable

    x_scale = alt.Scale(domain=[lo, hi])
    edges = np.linspace(lo, hi, nbins + 1)

    observed_counts, _ = np.histogram(observed, bins=edges)
    histogram = (
        alt.Chart(
            pd.DataFrame(
                {
                    "bin_start": edges[:-1],
                    "bin_end": edges[1:],
                    "count": observed_counts,
                    "series": observed_label,
                }
            )
        )
        .mark_bar(opacity=0.3, binSpacing=0)
        .encode(
            x=alt.X(
                "bin_start:Q",
                bin="binned",
                title=x_title,
                scale=x_scale,
                axis=alt.Axis(format="d"),
            ),
            x2="bin_end:Q",
            y=alt.Y("count:Q", title="count", stack=None),
            color=alt.Color("series:N", scale=color_scale, legend=legend),
        )
    )

    layers: list[alt.Chart] = [histogram]
    if has_density and dy is not None and dv is not None:
        expected_counts = _expected_counts_per_bin(edges, dy, dv, observed.size)
        expected_bars = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "bin_start": edges[:-1],
                        "bin_end": edges[1:],
                        "count": expected_counts,
                        "series": density_label,
                    }
                )
            )
            .mark_bar(opacity=0.3, binSpacing=0)
            .encode(
                x=alt.X("bin_start:Q", bin="binned", scale=x_scale),
                x2="bin_end:Q",
                y=alt.Y("count:Q", stack=None),
                color=alt.Color("series:N", scale=color_scale, legend=legend),
            )
        )
        layers.append(expected_bars)

    return alt.layer(*layers).properties(title=title)


def plot_creation_year_fit(
    ages: NDArray[np.float64],
    times: NDArray[np.float64],
    start_time: float,
    birth_rate: float,
    death_rate: float,
    *,
    title: str,
    nbins: int = 30,
) -> alt.LayerChart:
    """Plot observed creation years against the theoretical birth-time density.

    Converts individual ages (relative to the last epoch end) to calendar
    creation years and overlays the
    :func:`~bdpinf.inference.calculate_unknown_spine_mixture_birth_density`
    density scaled to the histogram counts.

    Parameters
    ----------
    ages : NDArray[np.float64]
        Individual ages in years (``age = last_epoch_end - creation_year``).
    times : NDArray[np.float64]
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
    from bdpinf.inference import (
        calculate_unknown_spine_mixture_birth_density,
    )

    # age = times[-1] - birth_rel  ⇒  calendar_year = start_time + (times[-1] - age)
    observed_birth_years = start_time + (float(times[-1]) - ages)

    density_years: NDArray[np.float64] | None = None
    density_values: NDArray[np.float64] | None = None
    n = len(ages)
    if n > 0 and birth_rate > 0.0 and death_rate >= 0.0 and ages.size > 0:
        try:
            # age == process_duration is the spine individual's age: it is
            # deterministically alive for the full duration, so the mixture
            # density has a point mass there rather than a continuous value.
            # Exclude it from the grid so it does not distort the overlay's
            # scale.
            process_duration = float(times[-1])
            age_upper = float(ages.max())
            if np.isclose(age_upper, process_duration):
                age_upper = float(np.nextafter(process_duration, 0.0))
            age_grid = np.linspace(0.0, age_upper, 400)
            birth_grid_rel = float(times[-1]) - age_grid
            density_years = start_time + birth_grid_rel
            density_values = np.asarray(
                calculate_unknown_spine_mixture_birth_density(
                    birth_rates=birth_rate,
                    death_rates=death_rate,
                    birth_time_grid=birth_grid_rel,
                    times=times,
                    num_individuals=n,
                ),
                dtype=np.float64,
            )
        except (ValueError, FloatingPointError) as exc:  # overlay is optional
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
