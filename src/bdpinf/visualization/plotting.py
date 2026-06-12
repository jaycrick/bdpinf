"""Visualization functions for birth-death process analysis.

This module contains plotting and visualization utilities for birth-death
process models and their results.

"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from plotly.subplots import make_subplots

from inf_bp_texts.visualization.altair_theme import PTOLQUAL, save_chart

logger = logging.getLogger(__name__)

# Professional plotting style configuration
# Configure professional font family for all plots
FONT_FAMILY = "Libertinus Serif, Latin Modern Roman, Times New Roman, Times, serif"
FONT_SIZE = 13
TITLE_FONT_SIZE = 15

# Global font configuration for plotly
font_config = {"family": FONT_FAMILY, "size": FONT_SIZE, "color": "black"}
title_font_config = {"family": FONT_FAMILY, "size": TITLE_FONT_SIZE, "color": "black"}

# Professional plot styling with high luminance, low saturation background
# Using OKLCH values: L > 95, C < 2 (very bright, desaturated grey)
plot_bgcolor = "#FAFAFA"  # Very light grey, high luminance, low saturation
paper_bgcolor = "#FAFAFA"  # Same as plot background for consistency
gridcolor = "#D0D0D0"  # Medium grey for grid lines (darker than background)

ptolqual = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Violet
    "#BBBBBB",  # Gray
]


def apply_professional_styling(
    fig: Any,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 400,
    *,
    autosize: bool = True,
) -> Any:
    """Apply consistent professional styling to all figures.

    Args:
        fig: Plotly figure object
        title: Title for the figure
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Figure height in pixels (default: 400)

    Returns
    -------
        Updated figure object with professional styling applied

    """
    fig.update_layout(
        title={
            "text": title,
            "y": 0.995,
            "yanchor": "top",
            "pad": {"b": 18},
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font=font_config,
        title_font=title_font_config,
        xaxis={
            "gridcolor": gridcolor,
            "gridwidth": 0.5,
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        yaxis={
            "gridcolor": gridcolor,
            "gridwidth": 0.5,
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        # Tufte-like in-plot legend defaults (small, muted, inside plotting area)
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 0.02,
            "bgcolor": "rgba(250,250,250,0.6)",
            "bordercolor": "rgba(0,0,0,0.08)",
            "borderwidth": 0.5,
            "font": {"size": FONT_SIZE - 2, "color": "black"},
            "traceorder": "normal",
        },
        margin={"l": 50, "r": 15, "t": 15 if title else 8, "b": 40, "pad": 4},
        autosize=autosize,
    )
    return fig


def plot_parameter_consistency(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    regime_configs: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    seed: int,
    parameter_name: str,
    true_rate_key: str,
    title: str,
) -> go.Figure:
    """Create multi-row consistency plots for a parameter across regimes and T.

    This function mirrors the plotting helper from the validation notebook but
    is provided here so notebooks can import and reuse it.
    """
    regime_order = [regime_config["name"] for regime_config in regime_configs]
    regime_lookup = {cfg["name"]: cfg for cfg in regime_configs}

    def _fill_color(color: object) -> str:
        fill_color = "rgba(0,0,0,0.12)"
        try:
            if isinstance(color, str) and color.startswith("rgb"):
                nums = [int(x) for x in color.strip("rgb()").split(",")]
                fill_color = f"rgba({nums[0]},{nums[1]},{nums[2]},0.12)"
        except Exception:
            fill_color = "rgba(0,0,0,0.12)"
        return fill_color

    fig: go.Figure = make_subplots(
        rows=len(regime_order),
        cols=3,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=[
            title
            for regime in regime_order
            for title in (f"{regime} mean", f"{regime} median", f"{regime} variance")
        ],
    )

    from inf_bp_texts.analysis.spike_detection import seed_from_parts

    for row_index, regime_name in enumerate(regime_order, start=1):
        regime_summary = summary_df[summary_df["regime"] == regime_name].sort_values(
            "T",
        )
        regime_raw = results_df[
            (results_df["regime"] == regime_name) & (results_df["success"])
        ]
        true_value = float(regime_lookup[regime_name][true_rate_key])
        mean_color = ptolqual[(row_index - 1) % len(ptolqual)]
        median_color = ptolqual[(row_index) % len(ptolqual)]
        variance_color = ptolqual[(row_index + 2) % len(ptolqual)]

        t_values = []
        mean_lower = []
        mean_upper = []
        median_lower = []
        median_upper = []
        median_estimates = []
        var_lower = []
        var_upper = []

        for t in regime_summary["T"]:
            t = float(t)
            t_values.append(t)
            reps = regime_raw.loc[
                regime_raw["T"] == t,
                f"{parameter_name}_estimate",
            ].to_numpy()
            reps = reps[np.isfinite(reps)]
            if reps.size == 0:
                mean_lower.append(np.nan)
                mean_upper.append(np.nan)
                median_lower.append(np.nan)
                median_upper.append(np.nan)
                median_estimates.append(np.nan)
                var_lower.append(np.nan)
                var_upper.append(np.nan)
                continue

            rng_bs = np.random.default_rng(
                seed_from_parts(seed, "bootstrap", regime_name, parameter_name, t),
            )
            boot_stats_mean = np.array(
                [
                    np.mean(rng_bs.choice(reps, size=reps.size, replace=True))
                    for _ in range(int(bootstrap_samples))
                ],
            )
            boot_stats_median = np.array(
                [
                    np.median(rng_bs.choice(reps, size=reps.size, replace=True))
                    for _ in range(int(bootstrap_samples))
                ],
            )
            boot_stats_var = np.array(
                [
                    np.var(rng_bs.choice(reps, size=reps.size, replace=True), ddof=1)
                    for _ in range(int(bootstrap_samples))
                ],
            )

            mean_lower.append(float(np.percentile(boot_stats_mean, 2.5)))
            mean_upper.append(float(np.percentile(boot_stats_mean, 97.5)))
            median_lower.append(float(np.percentile(boot_stats_median, 2.5)))
            median_upper.append(float(np.percentile(boot_stats_median, 97.5)))
            median_estimates.append(float(np.median(reps)))
            var_lower.append(float(np.percentile(boot_stats_var, 2.5)))
            var_upper.append(float(np.percentile(boot_stats_var, 97.5)))

        # Mean panel
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=mean_lower,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            row=row_index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=mean_upper,
                mode="lines",
                fill="tonexty",
                fillcolor=_fill_color(mean_color),
                line=dict(width=0),
                showlegend=False,
            ),
            row=row_index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=regime_summary["T"],
                y=regime_summary[f"{parameter_name}_mean"],
                mode="lines+markers",
                line=dict(color=mean_color, width=2),
                marker=dict(size=7),
                name=f"{regime_name} mean",
                showlegend=row_index == 1,
            ),
            row=row_index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=regime_summary["T"],
                y=[true_value] * len(regime_summary),
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name=f"{regime_name} true",
                showlegend=row_index == 1,
            ),
            row=row_index,
            col=1,
        )

        # Median panel
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=median_lower,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            row=row_index,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=median_upper,
                mode="lines",
                fill="tonexty",
                fillcolor=_fill_color(median_color),
                line=dict(width=0),
                showlegend=False,
            ),
            row=row_index,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=median_estimates,
                mode="lines+markers",
                line=dict(color=median_color, width=2),
                marker=dict(size=7),
                name=f"{regime_name} median",
                showlegend=row_index == 1,
            ),
            row=row_index,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=[true_value] * len(t_values),
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name=f"{regime_name} true median",
                showlegend=False,
            ),
            row=row_index,
            col=2,
        )

        # Variance panel
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=var_lower,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            row=row_index,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=var_upper,
                mode="lines",
                fill="tonexty",
                fillcolor=_fill_color(variance_color),
                line=dict(width=0),
                showlegend=False,
            ),
            row=row_index,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=regime_summary["T"],
                y=regime_summary[f"{parameter_name}_var"],
                mode="lines+markers",
                line=dict(color=variance_color, width=2),
                marker=dict(size=7),
                name=f"{regime_name} variance",
                showlegend=row_index == 1,
            ),
            row=row_index,
            col=3,
        )

    for row_index in range(1, len(regime_order) + 1):
        fig.update_xaxes(type="log", title_text="T", row=row_index, col=1)
        fig.update_xaxes(type="log", title_text="T", row=row_index, col=2)
        fig.update_xaxes(type="log", title_text="T", row=row_index, col=3)
        fig.update_yaxes(title_text="Mean estimate", row=row_index, col=1)
        fig.update_yaxes(title_text="Median estimate", row=row_index, col=2)
        fig.update_yaxes(title_text="Variance", row=row_index, col=3)

    fig.update_layout(
        title=title,
        height=280 * len(regime_order),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(t=90, l=40, r=20, b=40),
    )
    fig = apply_tufte_axes(fig)
    return fig


def apply_tufte_axes(
    fig: Any,
    axis_color: str = "#BBBBBB",
    tick_color: str = "#777777",
    tick_len: int = 3,
    tick_width: int = 1,
    axis_line_width: int = 1,
) -> Any:
    """Apply Tufte-style minimalist axes to all subplots in a Plotly figure.

    - Removes grid lines and zero lines
    - Uses thin, subtle axis lines
    - Small outward ticks with muted color
    - Keeps automargins enabled
    """
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=axis_color,
        linewidth=axis_line_width,
        ticks="outside",
        ticklen=tick_len,
        tickwidth=tick_width,
        tickcolor=tick_color,
        automargin=True,
        mirror=False,
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=axis_color,
        linewidth=axis_line_width,
        ticks="outside",
        ticklen=tick_len,
        tickwidth=tick_width,
        tickcolor=tick_color,
        automargin=True,
        mirror=False,
    )
    return fig


def save_figure(
    fig: Any,
    filename: str,
    width: int = 550,
    height: int = 450,
    output_dir: str = "figures",
) -> Path | None:
    """Save plotly figure as PDF with A4-friendly dimensions.

    Args:
        fig: Plotly figure object
        filename: Output filename (without extension)
        width: Figure width in pixels (default: 800)
        height: Figure height in pixels (default: 600)
        output_dir: Directory to save figures (default: "figures")

    Returns
    -------
        Path to saved file if successful, None if failed

    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Calculate appropriate margins based on figure size
        # Smaller figures need tighter margins; larger figures can have more space
        if width < 250 or height < 200:
            # Very small figures: minimal margins but enough for axis labels
            margin_dict = {"l": 30, "r": 8, "t": 8, "b": 35, "pad": 2}
        elif width < 350 or height < 300:
            # Small figures: compact margins
            margin_dict = {"l": 40, "r": 10, "t": 10, "b": 40, "pad": 3}
        else:
            # Standard figures: comfortable margins
            margin_dict = {"l": 50, "r": 15, "t": 15, "b": 45, "pad": 4}

        # Update figure size and margins for consistent export
        fig.update_layout(
            width=width,
            height=height,
            font={"size": FONT_SIZE},  # Ensure readable font size
            margin=margin_dict,
            autosize=False,
        )
        # Ensure axes don't expand beyond plotting area
        fig.update_xaxes(automargin=True, fixedrange=False)
        fig.update_yaxes(automargin=True, fixedrange=False)

        # Set transparent backgrounds for clean PDF export
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        file_path = output_path / f"{filename}.pdf"
        fig.write_image(str(file_path))
        return file_path
    except Exception as e:
        logger.warning("Failed to save %s.pdf: %s", filename, e)
        return None


def save_multi_epoch_convergence_figure(
    case_name: str,
    result: dict[str, Any],
    output_slug: str,
    figures_dir: Path | None = None,
) -> None:
    """Save the log-likelihood convergence trace for one multi-epoch fit.

    Parameters
    ----------
    case_name : str
        Human-readable label used in the figure title.
    result : dict[str, Any]
        Fit result dict as returned by ``fit_birth_death_processes``.
    output_slug : str
        Filename stem passed to ``save_figure``.
    figures_dir : Path | None
        Directory for saved figures. Defaults to ``Path("figures")``.

    """
    from inf_bp_texts.analysis.validation_helpers import (  # noqa: PLC0415
        get_selected_attempt,
    )

    if not result.get("success"):
        return

    selected_attempt = get_selected_attempt(result)
    nll_history = (
        selected_attempt.get("nll_history", [])
        if isinstance(selected_attempt, dict)
        else []
    )
    if not nll_history:
        return

    ll_history = [-value for value in nll_history]
    iterations = list(range(1, len(ll_history) + 1))

    frame = pd.DataFrame({"Iteration": iterations, "Log-likelihood": ll_history})
    chart = (
        alt.Chart(frame)
        .mark_line(
            point=alt.OverlayMarkDef(size=30, color=PTOLQUAL[2]),
            strokeWidth=2,
            color=PTOLQUAL[2],
        )
        .encode(
            x=alt.X("Iteration:Q", title="Iteration", axis=alt.Axis(format="d")),
            y=alt.Y("Log-likelihood:Q", title="Log-likelihood"),
        )
        .properties(title=f"{case_name}: Log-likelihood Convergence")
    )
    save_chart(
        chart,
        output_slug,
        width=397,
        height=300,
        output_dir=str(figures_dir or "figures"),
    )


def save_multi_epoch_birth_time_figure(
    case_name: str,
    result: dict[str, Any],
    epochs: list[Any],
    manuscripts_df: pd.DataFrame,
    times: torch.Tensor,
    output_slug: str,
    figures_dir: Path | None = None,
) -> None:
    """Save manuscript birth-time histogram with fitted density overlay.

    Parameters
    ----------
    case_name : str
        Human-readable label used in the figure title.
    result : dict[str, Any]
        Fit result dict as returned by ``fit_birth_death_processes``.
    epochs : list[Any]
        List of ``(start, end, label)`` epoch tuples.
    manuscripts_df : pd.DataFrame
        DataFrame with a ``"parsed_year"`` column of manuscript creation years.
    times : torch.Tensor
        Epoch boundary times as a 1-D float64 tensor.
    output_slug : str
        Filename stem passed to ``save_figure``.
    figures_dir : Path | None
        Directory for saved figures. Defaults to ``Path("figures")``.

    """
    from inf_bp_texts.inference.likelihood import (  # noqa: PLC0415
        calculate_unknown_spine_mixture_birth_density,
    )

    if not result.get("success"):
        return

    birth_rates = np.asarray(result["birth_rates"], dtype=float)
    death_rates = np.asarray(result["death_rates"], dtype=float)
    observed_birth_years = manuscripts_df["parsed_year"].to_numpy(dtype=float)
    if observed_birth_years.size == 0:
        return

    start_year = float(epochs[0][0])
    plot_end_year = float(epochs[-1][0])
    birth_year_grid = np.linspace(start_year, plot_end_year, 200)
    birth_time_grid = birth_year_grid - start_year

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=observed_birth_years,
            nbinsx=50,
            name="Observed Birth Years",
            marker_color=ptolqual[0],
            opacity=0.6,
            showlegend=True,
        )
    )

    try:
        theoretical_birth_density = (
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=torch.as_tensor(birth_rates, dtype=torch.float64),
                death_rates=torch.as_tensor(death_rates, dtype=torch.float64),
                birth_time_grid=torch.as_tensor(birth_time_grid, dtype=torch.float64),
                times=times,
                num_manuscripts=len(observed_birth_years),
            )
            .detach()
            .cpu()
            .numpy()
        )
        hist_count = len(observed_birth_years)
        bin_width = (observed_birth_years.max() - observed_birth_years.min()) / 50
        if bin_width <= 0:
            bin_width = 1.0
        theoretical_density_scaled = theoretical_birth_density * hist_count * bin_width
        fig.add_trace(
            go.Scatter(
                x=birth_year_grid,
                y=theoretical_density_scaled,
                mode="lines",
                name="Fitted Birth-Time Distribution",
                line={"color": ptolqual[1], "width": 3},
                showlegend=True,
            )
        )
    except Exception as exc:
        logger.warning(
            "Could not generate theoretical overlay for %s: %s", case_name, exc
        )

    fig = apply_professional_styling(
        fig,
        f"{case_name}: Observed vs Fitted Manuscript Birth Times",
        "Manuscript birth year (CE)",
        "Frequency",
    )
    fig = apply_tufte_axes(fig)
    fig.update_xaxes(tickformat=",d")
    fig.update_yaxes(tickformat=",d")
    save_figure(
        fig,
        output_slug,
        width=397,
        height=300,
        output_dir=str(figures_dir or "figures"),
    )


def save_multi_epoch_fit_figure(
    case_name: str,
    result: dict[str, Any],
    epochs: list[Any],
    output_slug: str,
    figures_dir: Path | None = None,
) -> None:
    """Save a compact birth/death rate bar chart for one multi-epoch fit.

    Parameters
    ----------
    case_name : str
        Human-readable label used in the figure title.
    result : dict[str, Any]
        Fit result dict as returned by ``fit_birth_death_processes``.
    epochs : list[Any]
        List of ``(start, end, label)`` epoch tuples used to label bars.
    output_slug : str
        Filename stem passed to ``save_figure``.
    figures_dir : Path | None
        Directory for saved figures. Defaults to ``Path("figures")``.

    """
    if not result.get("success"):
        return

    birth_rates = np.asarray(result["birth_rates"], dtype=float)
    death_rates = np.asarray(result["death_rates"], dtype=float)
    epoch_labels = [f"{start}-{end}" for start, end, _ in epochs]

    def _rate_panel(rates: np.ndarray, color: str, subtitle: str) -> alt.LayerChart:
        frame = pd.DataFrame({"Epoch": epoch_labels, "Rate": rates.astype(float)})
        base = alt.Chart(frame).encode(
            # sort=None keeps epochs in chronological (input) order, not alphabetical.
            x=alt.X("Epoch:N", title="Epoch", sort=None),
            y=alt.Y("Rate:Q", title="Rate", axis=alt.Axis(format=".6f")),
        )
        bars = base.mark_bar(color=color)
        labels = base.mark_text(dy=-5, baseline="bottom").encode(
            text=alt.Text("Rate:Q", format=".6f"),
        )
        return (bars + labels).properties(title=subtitle, width=360, height=300)

    chart = alt.hconcat(
        _rate_panel(birth_rates, PTOLQUAL[0], f"{case_name}: Birth Rates (λ)"),
        _rate_panel(death_rates, PTOLQUAL[1], f"{case_name}: Death Rates (μ)"),
    ).properties(title=f"{case_name}: Multi-epoch Birth/Death Rates")
    save_chart(
        chart,
        output_slug,
        output_dir=str(figures_dir or "figures"),
    )
