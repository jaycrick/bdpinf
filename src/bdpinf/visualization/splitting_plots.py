"""Plotly rendering functions for multilevel splitting simulation results."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import plotly.graph_objs as go

from inf_bp_texts.visualization.plotting import (
    apply_professional_styling,
    apply_tufte_axes,
    ptolqual,
)

if TYPE_CHECKING:
    from inf_bp_texts.simulation.splitting import ConditionedTrajectory


def plot_conditioned_trajectories(
    trajectories: list[ConditionedTrajectory],
    *,
    max_trajectories: int = 12,
    title: str = "Conditioned Trajectories",
) -> go.Figure:
    """Plot sample conditioned population paths."""
    fig = go.Figure()
    if not trajectories:
        fig.add_annotation(text="No conditioned trajectories available")
        return cast(
            "go.Figure",
            apply_professional_styling(
                fig,
                title,
                "Time",
                "Population size",
            ),
        )

    n_show = min(max_trajectories, len(trajectories))
    for idx in range(n_show):
        traj = trajectories[idx]
        color = ptolqual[idx % len(ptolqual)]
        fig.add_trace(
            go.Scatter(
                x=traj.times,
                y=traj.populations,
                mode="lines",
                line={"color": color, "width": 1.6},
                name=f"Particle {traj.particle_id}",
            ),
        )

    styled = cast(
        "go.Figure",
        apply_professional_styling(fig, title, "Time", "Population size"),
    )
    return cast("go.Figure", apply_tufte_axes(styled))


def plot_empirical_population_mean(
    *,
    level_times: np.ndarray,
    mean_populations: np.ndarray,
    title: str = "Empirical Mean Population Across Levels",
) -> go.Figure:
    """Plot empirical ensemble mean population vs level times."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=level_times,
            y=mean_populations,
            mode="lines+markers",
            line={"color": ptolqual[0], "width": 2.0},
            marker={"size": 6},
            name="Empirical mean",
        ),
    )
    styled = cast(
        "go.Figure",
        apply_professional_styling(fig, title, "Level time", "Mean population"),
    )
    return cast("go.Figure", apply_tufte_axes(styled))


def plot_level_survival_probabilities(
    *,
    level_times: np.ndarray,
    level_survival_probabilities: np.ndarray,
    title: str = "Estimated Survival Probabilities Across Levels",
) -> go.Figure:
    """Plot level-wise conditional survival probabilities."""
    if len(level_times) != len(level_survival_probabilities) + 1:
        msg = (
            "level_times must include t0; probabilities are defined for level intervals"
        )
        raise ValueError(msg)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=level_times[1:],
            y=level_survival_probabilities,
            mode="lines+markers",
            line={"color": ptolqual[1], "width": 2.0},
            marker={"size": 6},
            name="Level conditional survival",
        ),
    )
    styled = cast(
        "go.Figure",
        apply_professional_styling(fig, title, "Level time", "Probability"),
    )
    styled = cast("go.Figure", apply_tufte_axes(styled))
    styled.update_yaxes(range=[0.0, 1.0])
    return styled
