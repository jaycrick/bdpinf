"""Plotly rendering functions for birth-death parameter estimate visualizations."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from numpy.typing import NDArray

_MIN_KDE_SAMPLES = 2


@dataclass(frozen=True)
class KdePlotConfig:
    """Display and sampling options for KDE density plots."""

    estimate_axis_label: str
    true_label: str
    density_color: str
    true_line_color: str
    x_upper_multiplier: float = 10.0
    n_points: int = 100
    x_quantile_low: float = 0.005
    x_quantile_high: float = 0.995
    x_padding_fraction: float = 0.05


@dataclass(frozen=True)
class BoxPlotStyle:
    """Styling options for box plots with reference markers."""

    base_color: str
    true_color: str
    mean_color: str
    median_color: str


def create_kde_density_plot(
    rates: Sequence[float] | NDArray[np.float64],
    true_rate: float,
    config: KdePlotConfig,
) -> go.Figure:
    """Create a 1D KDE plot with true-value reference line and annotation."""
    values = np.asarray(rates, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < _MIN_KDE_SAMPLES:
        msg = "At least two rates are required to build a KDE plot"
        raise ValueError(msg)

    from inf_bp_texts.analysis.validation_helpers import (  # noqa: PLC0415
        boundary_corrected_kde_at_zero,
    )

    x_vals, y_vals = boundary_corrected_kde_at_zero(
        values,
        n_grid=config.n_points,
    )
    if x_vals is None or y_vals is None:
        msg = "Could not build boundary-corrected KDE from the provided rates"
        raise ValueError(msg)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name="Density",
            line={"color": config.density_color},
            showlegend=False,
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[true_rate, true_rate],
            y=[0, float(np.max(y_vals))],
            mode="lines",
            name=config.true_label,
            line={"color": config.true_line_color, "dash": "dash"},
            showlegend=False,
        ),
    )
    fig.add_annotation(
        x=true_rate,
        y=float(np.max(y_vals)) * 1.05,
        text=config.true_label,
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-30,
        xanchor="center",
        font={"color": config.true_line_color, "size": 11},
        bgcolor="rgba(255,255,255,0.75)",
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickformat=".4f", title=config.estimate_axis_label)
    fig.update_yaxes(rangemode="tozero", title="Density")
    return fig


def compute_boxplot_summary(
    birth_rates: Sequence[float] | NDArray[np.float64],
    death_rates: Sequence[float] | NDArray[np.float64],
    true_birth_rate: float,
    true_death_rate: float,
) -> pd.DataFrame:
    """Build a detailed summary table used alongside parameter box plots."""
    birth = np.asarray(birth_rates, dtype=np.float64)
    death = np.asarray(death_rates, dtype=np.float64)
    if birth.size == 0 or death.size == 0:
        msg = "Cannot compute boxplot summary for empty rate arrays"
        raise ValueError(msg)

    mse_birth = float(np.mean((birth - true_birth_rate) ** 2))
    mse_death = float(np.mean((death - true_death_rate) ** 2))

    return pd.DataFrame(
        {
            "Parameter": ["Birth Rate (λ)", "Death Rate (μ)"],
            "True Value": [true_birth_rate, true_death_rate],
            "Mean": [float(np.mean(birth)), float(np.mean(death))],
            "Median": [float(np.median(birth)), float(np.median(death))],
            "Std Dev": [float(np.std(birth)), float(np.std(death))],
            "Q1 (25%)": [
                float(np.percentile(birth, 25)),
                float(np.percentile(death, 25)),
            ],
            "Q3 (75%)": [
                float(np.percentile(birth, 75)),
                float(np.percentile(death, 75)),
            ],
            "Min": [float(np.min(birth)), float(np.min(death))],
            "Max": [float(np.max(birth)), float(np.max(death))],
            "Bias": [
                float(np.mean(birth) - true_birth_rate),
                float(np.mean(death) - true_death_rate),
            ],
            "Rel. Bias (%)": [
                float((np.mean(birth) - true_birth_rate) / true_birth_rate * 100),
                float((np.mean(death) - true_death_rate) / true_death_rate * 100),
            ],
            "MSE": [mse_birth, mse_death],
            "RMSE": [float(np.sqrt(mse_birth)), float(np.sqrt(mse_death))],
        },
    )


def create_parameter_box_plot(
    birth_rates: Sequence[float] | NDArray[np.float64],
    death_rates: Sequence[float] | NDArray[np.float64],
    true_rates: tuple[float, float],
    style: BoxPlotStyle,
) -> go.Figure:
    """Create a two-parameter box plot with true, mean, and median overlays."""
    box_data = [
        np.asarray(birth_rates, dtype=np.float64),
        np.asarray(death_rates, dtype=np.float64),
    ]
    labels = ["MLE Birth Rates", "MLE Death Rates"]
    means = [float(np.mean(box_data[0])), float(np.mean(box_data[1]))]
    medians = [float(np.median(box_data[0])), float(np.median(box_data[1]))]

    fig = go.Figure()
    for idx, data in enumerate(box_data):
        fig.add_trace(
            go.Box(
                y=data,
                name=labels[idx],
                boxmean=True,
                boxpoints="outliers",
                marker_color=style.base_color,
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[labels[idx]],
                y=[true_rates[idx]],
                mode="markers",
                marker={"color": style.true_color, "size": 8, "symbol": "circle"},
                name="True",
                showlegend=(idx == 0),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[labels[idx]],
                y=[means[idx]],
                mode="markers",
                marker={"color": style.mean_color, "size": 10, "symbol": "circle"},
                name="Mean",
                showlegend=(idx == 0),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[labels[idx]],
                y=[medians[idx]],
                mode="markers",
                marker={"color": style.median_color, "size": 10, "symbol": "square"},
                name="Median",
                showlegend=(idx == 0),
            ),
        )
    return fig
