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
from numpy.typing import NDArray

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


_CONSISTENCY_STATS = ("mean", "median", "variance")


def compute_consistency_bands(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    regime_configs: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    seed: int,
    parameter_name: str,
    true_rate_key: str,
) -> pd.DataFrame:
    """Compute tidy bootstrap CI bands for a parameter across regimes and T.

    Pure data preparation for :func:`plot_parameter_consistency`: for each
    regime and observation time ``T`` it bootstraps the 95% interval of the
    mean, median, and variance of the per-replicate estimates.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Per-regime/``T`` summary with ``regime``, ``T``,
        ``{parameter_name}_mean`` and ``{parameter_name}_var`` columns.
    results_df : pd.DataFrame
        Per-replicate results with ``regime``, ``T``, ``success`` and
        ``{parameter_name}_estimate`` columns.
    regime_configs : Sequence[Mapping[str, Any]]
        Regime definitions, each with a ``name`` and the ``true_rate_key``.
    bootstrap_samples : int
        Number of bootstrap resamples per statistic.
    seed : int
        Base seed; per-cell seeds are derived deterministically from it.
    parameter_name : str
        Parameter prefix (e.g. ``"birth"``) used to select columns.
    true_rate_key : str
        Key into each regime config giving the parameter's true value.

    Returns
    -------
    pd.DataFrame
        Long-form table with columns ``regime``, ``T``, ``statistic`` (one of
        ``mean``/``median``/``variance``), ``estimate``, ``lower``, ``upper``,
        and ``true_value`` (NaN for ``variance``, which has no reference line).
    """
    from inf_bp_texts.analysis.spike_detection import seed_from_parts  # noqa: PLC0415

    regime_order = [regime_config["name"] for regime_config in regime_configs]
    regime_lookup = {cfg["name"]: cfg for cfg in regime_configs}

    rows: list[dict[str, Any]] = []
    for regime_name in regime_order:
        regime_summary = summary_df[summary_df["regime"] == regime_name].sort_values(
            "T"
        )
        regime_raw = results_df[
            (results_df["regime"] == regime_name) & (results_df["success"])
        ]
        true_value = float(regime_lookup[regime_name][true_rate_key])

        for _, summary_row in regime_summary.iterrows():
            t = float(summary_row["T"])
            reps = regime_raw.loc[
                regime_raw["T"] == t,
                f"{parameter_name}_estimate",
            ].to_numpy()
            reps = reps[np.isfinite(reps)]

            estimates = {
                "mean": float(summary_row[f"{parameter_name}_mean"]),
                "median": float(np.median(reps)) if reps.size else np.nan,
                "variance": float(summary_row[f"{parameter_name}_var"]),
            }
            if reps.size == 0:
                bounds = {stat: (np.nan, np.nan) for stat in _CONSISTENCY_STATS}
            else:
                # Draw mean, then median, then variance resamples from the same
                # generator (this ordering fixes the golden bootstrap values).
                rng_bs = np.random.default_rng(
                    seed_from_parts(seed, "bootstrap", regime_name, parameter_name, t),
                )
                boot = {
                    "mean": np.array(
                        [
                            np.mean(rng_bs.choice(reps, size=reps.size, replace=True))
                            for _ in range(int(bootstrap_samples))
                        ],
                    ),
                    "median": np.array(
                        [
                            np.median(rng_bs.choice(reps, size=reps.size, replace=True))
                            for _ in range(int(bootstrap_samples))
                        ],
                    ),
                    "variance": np.array(
                        [
                            np.var(
                                rng_bs.choice(reps, size=reps.size, replace=True),
                                ddof=1,
                            )
                            for _ in range(int(bootstrap_samples))
                        ],
                    ),
                }
                bounds = {
                    stat: (
                        float(np.percentile(boot[stat], 2.5)),
                        float(np.percentile(boot[stat], 97.5)),
                    )
                    for stat in _CONSISTENCY_STATS
                }

            true_values = {"mean": true_value, "median": true_value, "variance": np.nan}
            for stat in _CONSISTENCY_STATS:
                lower, upper = bounds[stat]
                rows.append(
                    {
                        "regime": regime_name,
                        "T": t,
                        "statistic": stat,
                        "estimate": estimates[stat],
                        "lower": lower,
                        "upper": upper,
                        "true_value": true_values[stat],
                    },
                )

    return pd.DataFrame(
        rows,
        columns=[
            "regime",
            "T",
            "statistic",
            "estimate",
            "lower",
            "upper",
            "true_value",
        ],
    )


def plot_parameter_consistency(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    regime_configs: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    seed: int,
    parameter_name: str,
    true_rate_key: str,
    title: str,
) -> alt.FacetChart:
    """Facet a parameter's consistency across regimes (rows) and statistics (cols).

    Each cell shows the estimate line (with a bootstrap 95% CI band) versus a
    log-scaled observation time ``T``; the true value is a dashed horizontal
    rule (omitted for variance). Statistics use independent y-scales.
    """
    bands = compute_consistency_bands(
        summary_df,
        results_df,
        regime_configs,
        bootstrap_samples,
        seed,
        parameter_name,
        true_rate_key,
    )
    regime_order = [regime_config["name"] for regime_config in regime_configs]

    x = alt.X("T:Q", scale=alt.Scale(type="log"), title="T")
    color = alt.Color("regime:N", legend=None)

    ci_band = (
        alt.Chart()
        .mark_area(opacity=0.25)
        .encode(
            x=x,
            y=alt.Y("lower:Q", title=None),
            y2="upper:Q",
            color=color,
        )
    )
    estimate_line = (
        alt.Chart()
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("estimate:Q", title=None),
            color=color,
        )
    )
    true_rule = (
        alt.Chart()
        .transform_filter("isValid(datum.true_value)")
        .mark_rule(color="black", strokeDash=[4, 4])
        .encode(y="true_value:Q")
    )

    return (
        alt.layer(ci_band, estimate_line, true_rule, data=bands)
        .facet(
            column=alt.Column("statistic:N", sort=list(_CONSISTENCY_STATS), title=None),
            row=alt.Row("regime:N", sort=regime_order, title=None),
        )
        .resolve_scale(y="independent")
        .properties(title=title)
    )


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


def convergence_chart(ll_history: Sequence[float], *, title: str) -> alt.Chart:
    """Line chart of log-likelihood versus iteration for an optimizer trace.

    Parameters
    ----------
    ll_history : Sequence[float]
        Per-iteration log-likelihood values (already negated from the NLL).
    title : str
        Figure title.
    """
    iterations = list(range(1, len(ll_history) + 1))
    frame = pd.DataFrame({"Iteration": iterations, "Log-likelihood": list(ll_history)})
    return (
        alt.Chart(frame)
        .mark_line(
            point=alt.OverlayMarkDef(size=30, color=PTOLQUAL[2]),
            strokeWidth=2,
            color=PTOLQUAL[2],
        )
        .encode(
            x=alt.X("Iteration:Q", title="Iteration", axis=alt.Axis(format="d")),
            # Log-likelihood is meaningful relative to itself, not to zero, so
            # fit the axis to the data range instead of stretching down to 0.
            y=alt.Y(
                "Log-likelihood:Q",
                title="Log-likelihood",
                scale=alt.Scale(zero=False),
            ),
        )
        .properties(title=title)
    )


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
    chart = convergence_chart(
        ll_history, title=f"{case_name}: Log-likelihood Convergence"
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
    times: NDArray[np.float64],
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
    times : NDArray[np.float64]
        Epoch boundary times as a 1-D float64 array.
    output_slug : str
        Filename stem passed to ``save_figure``.
    figures_dir : Path | None
        Directory for saved figures. Defaults to ``Path("figures")``.

    """
    from inf_bp_texts.inference.likelihood import (  # noqa: PLC0415
        calculate_unknown_spine_mixture_birth_density,
    )
    from inf_bp_texts.visualization.age_fit_plot import (  # noqa: PLC0415
        creation_year_fit_chart,
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

    density_values: np.ndarray | None = None
    try:
        density_values = np.asarray(
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=np.asarray(birth_rates, dtype=np.float64),
                death_rates=np.asarray(death_rates, dtype=np.float64),
                birth_time_grid=np.asarray(birth_time_grid, dtype=np.float64),
                times=times,
                num_manuscripts=len(observed_birth_years),
            ),
            dtype=np.float64,
        )
    except Exception as exc:  # noqa: BLE001 - overlay is optional; skip on failure
        logger.warning(
            "Could not generate theoretical overlay for %s: %s", case_name, exc
        )
        density_values = None

    chart = creation_year_fit_chart(
        observed_birth_years,
        birth_year_grid if density_values is not None else None,
        density_values,
        title=f"{case_name}: Observed vs Fitted Manuscript Birth Times",
        nbins=50,
        x_title="Manuscript birth year (CE)",
        observed_label="Observed Birth Years",
        density_label="Fitted Birth-Time Distribution",
    )
    save_chart(
        chart,
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
