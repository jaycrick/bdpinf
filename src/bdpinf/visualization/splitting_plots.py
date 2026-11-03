"""Altair rendering functions for multilevel splitting simulation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import pandas as pd

from bdpinf.visualization.altair_theme import PTOLQUAL

if TYPE_CHECKING:
    from bdpinf.simulation.conditioned import ConditionedTrajectory


def _point(color: str) -> alt.OverlayMarkDef:
    """Line-overlay point marker that matches its line colour.

    ``OverlayMarkDef`` does not inherit the parent line's colour, so it must be
    set explicitly or the points fall back to the theme's default category.
    """
    return alt.OverlayMarkDef(size=40, color=color)


def plot_conditioned_trajectories(
    trajectories: list[ConditionedTrajectory],
    *,
    max_trajectories: int = 12,
    title: str = "Conditioned Trajectories",
) -> alt.Chart:
    """Plot sample conditioned population paths, one coloured line per particle."""
    if not trajectories:
        placeholder = pd.DataFrame({"_x": [0.0], "_y": [0.0]})
        return (
            alt.Chart(placeholder)
            .mark_text(text="No conditioned trajectories available", color="#777777")
            .encode(
                x=alt.X("_x:Q", title="Time"),
                y=alt.Y("_y:Q", title="Population size"),
            )
            .properties(title=title)
        )

    n_show = min(max_trajectories, len(trajectories))
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "Time": np.asarray(trajectories[idx].times, dtype=float),
                    "Population size": np.asarray(
                        trajectories[idx].populations, dtype=float
                    ),
                    "Particle": f"Particle {trajectories[idx].particle_id}",
                }
            )
            for idx in range(n_show)
        ],
        ignore_index=True,
    )

    # Colour cycles through the ptolqual category range supplied by the theme.
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=1.6)
        .encode(
            x=alt.X("Time:Q", title="Time"),
            y=alt.Y("Population size:Q", title="Population size"),
            color=alt.Color("Particle:N", legend=alt.Legend(title=None)),
        )
        .properties(title=title)
    )


def plot_empirical_population_mean(
    *,
    level_times: np.ndarray,
    mean_populations: np.ndarray,
    title: str = "Empirical Mean Population Across Levels",
) -> alt.Chart:
    """Plot empirical ensemble mean population vs level times."""
    frame = pd.DataFrame(
        {
            "Level time": np.asarray(level_times, dtype=float),
            "Mean population": np.asarray(mean_populations, dtype=float),
        }
    )
    return (
        alt.Chart(frame)
        .mark_line(point=_point(PTOLQUAL[0]), strokeWidth=2, color=PTOLQUAL[0])
        .encode(
            x=alt.X("Level time:Q", title="Level time"),
            y=alt.Y("Mean population:Q", title="Mean population"),
        )
        .properties(title=title)
    )
