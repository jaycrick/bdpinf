"""Altair rendering functions for birth-death parameter estimate visualizations."""

from collections.abc import Sequence
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
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
) -> alt.LayerChart:
    """Create a 1D KDE plot with a dashed true-value reference line and label.

    Vega-Lite has no arrow-annotation primitive, so the Plotly arrow pointing at
    the true value is reproduced as a dashed ``mark_rule`` plus a ``mark_text``
    label above it (a documented fidelity compromise of the Altair migration).
    """
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

    y_max = float(np.max(y_vals))
    density = (
        alt.Chart(
            pd.DataFrame(
                {
                    "estimate": np.asarray(x_vals, dtype=np.float64),
                    "density": np.asarray(y_vals, dtype=np.float64),
                }
            )
        )
        .mark_line(color=config.density_color)
        .encode(
            x=alt.X(
                "estimate:Q",
                title=config.estimate_axis_label,
                axis=alt.Axis(format=".4f"),
            ),
            y=alt.Y("density:Q", title="Density"),
        )
    )
    true_line = (
        alt.Chart(pd.DataFrame({"estimate": [float(true_rate)]}))
        .mark_rule(color=config.true_line_color, strokeDash=[4, 4])
        .encode(x="estimate:Q")
    )
    true_label = (
        alt.Chart(
            pd.DataFrame(
                {
                    "estimate": [float(true_rate)],
                    "density": [y_max * 1.05],
                    "label": [config.true_label],
                }
            )
        )
        .mark_text(dy=-2, baseline="bottom", color=config.true_line_color, fontSize=11)
        .encode(x="estimate:Q", y="density:Q", text="label:N")
    )
    return alt.layer(density, true_line, true_label)


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
) -> alt.LayerChart:
    """Create a two-parameter box plot with true and mean overlays.

    The median is already drawn as the box's centre line, so only true-value and
    mean markers are overlaid. The Plotly ``boxmean=True`` mean line (which
    ``mark_boxplot`` has no equivalent for) is reproduced by the explicit mean
    marker.
    """
    birth = np.asarray(birth_rates, dtype=np.float64)
    death = np.asarray(death_rates, dtype=np.float64)
    labels = ("MLE Birth Rates", "MLE Death Rates")

    box_frame = pd.DataFrame(
        {
            "Parameter": [labels[0]] * birth.size + [labels[1]] * death.size,
            "Rate": np.concatenate([birth, death]),
        }
    )
    box = (
        alt.Chart(box_frame)
        .mark_boxplot(color=style.base_color, size=40, outliers=True)
        .encode(
            x=alt.X("Parameter:N", title=None, sort=list(labels)),
            y=alt.Y("Rate:Q", title="Rate"),
        )
    )

    kinds = ("True", "Mean")
    marker_frame = pd.DataFrame(
        {
            "Parameter": list(labels) * 2,
            "value": [
                true_rates[0],
                true_rates[1],
                float(np.mean(birth)),
                float(np.mean(death)),
            ],
            "kind": ["True", "True", "Mean", "Mean"],
        }
    )
    color = alt.Color(
        "kind:N",
        title=None,
        scale=alt.Scale(domain=list(kinds), range=[style.true_color, style.mean_color]),
    )
    markers = (
        alt.Chart(marker_frame)
        .mark_point(filled=True, size=90, opacity=1.0)
        .encode(
            x=alt.X("Parameter:N", sort=list(labels)),
            y=alt.Y("value:Q"),
            color=color,
        )
    )
    return alt.layer(box, markers)
