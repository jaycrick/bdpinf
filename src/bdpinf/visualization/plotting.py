"""
Visualization functions for birth-death process analysis.

This module contains plotting and visualization utilities for birth-death
process models and their results.

Examples:
    Plotting model fit results:

    >>> import numpy as np
    >>> from inf_bp_texts.visualization.plotting import plot_fit_results
    >>>
    >>> # Generate sample data
    >>> ages = np.array([50, 100, 150, 200, 250, 300])
    >>> birth_rate, death_rate = 0.5, 0.3
    >>> observation_time = 400
    >>>
    >>> # Create fit results plot
    >>> fig = plot_fit_results(ages, birth_rate, death_rate, observation_time)
    >>> fig.show()  # Display the plot

    Plotting simulation results:

    >>> from inf_bp_texts.visualization.plotting import plot_simulation_results
    >>>
    >>> # Plot simulation data
    >>> simulation_data = {'ages': ages, 'counts': [10, 8, 6, 4, 2, 1]}
    >>> fig = plot_simulation_results(simulation_data)
    >>> fig.show()
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..models.birth_death_process import BirthDeathProcess

# Professional plotting style configuration
# Configure professional font family for all plots
FONT_FAMILY = "Latin Modern Roman, Times New Roman, Times, serif"
FONT_SIZE = 11
TITLE_FONT_SIZE = 13

# Global font configuration for plotly
font_config = dict(family=FONT_FAMILY, size=FONT_SIZE, color="black")
title_font_config = dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE, color="black")

# Professional plot styling with high luminance, low saturation background
# Using OKLCH values: L > 95, C < 2 (very bright, desaturated grey)
plot_bgcolor = "#FAFAFA"  # Very light grey, high luminance, low saturation
paper_bgcolor = "#FAFAFA"  # Same as plot background for consistency
gridcolor = "#D0D0D0"  # Medium grey for grid lines (darker than background)

# Custom color palette for plots
my_blue = "#4477AA"
my_red = "#EE6677"
my_green = "#228833"
my_yellow = "#CCBB44"
my_cyan = "#66CCEE"
my_violet = "#AA3377"
my_gray = "#BBBBBB"

ptolqual = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Violet
    "#BBBBBB",  # Gray
]

# Global layout configuration
layout_config = dict(
    plot_bgcolor=plot_bgcolor,
    paper_bgcolor=paper_bgcolor,
    font=font_config,
    title_font=title_font_config,
    xaxis=dict(gridcolor=gridcolor, gridwidth=0.5, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=gridcolor, gridwidth=0.5, showgrid=True, zeroline=False),
)


def apply_professional_styling(
    fig: Any, title: str, xaxis_title: str, yaxis_title: str, height: int = 400
) -> Any:
    """
    Apply consistent professional styling to all figures.

    Args:
        fig: Plotly figure object
        title: Title for the figure
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Figure height in pixels (default: 400)

    Returns:
        Updated figure object with professional styling applied
    """
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font=font_config,
        title_font=title_font_config,
        xaxis=dict(gridcolor=gridcolor, gridwidth=0.5, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=gridcolor, gridwidth=0.5, showgrid=True, zeroline=False),
    )
    return fig


def save_figure(
    fig: Any,
    filename: str,
    width: int = 550,
    height: int = 450,
    output_dir: str = "figures",
) -> Optional[Path]:
    """
    Save plotly figure as PDF with A4-friendly dimensions.

    Args:
        fig: Plotly figure object
        filename: Output filename (without extension)
        width: Figure width in pixels (default: 800)
        height: Figure height in pixels (default: 600)
        output_dir: Directory to save figures (default: "figures")

    Returns:
        Path to saved file if successful, None if failed
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Update figure size for better A4 compatibility
        fig.update_layout(
            width=width,
            height=height,
            font=dict(size=11),  # Ensure readable font size
            margin=dict(l=50, r=50, t=60, b=50),  # Consistent margins
        )

        file_path = output_path / f"{filename}.pdf"
        fig.write_image(str(file_path))
        return file_path
    except Exception as e:
        print(f"⚠️ Failed to save {filename}.pdf: {e}")
        return None


def plot_fit_results(
    model: BirthDeathProcess,
    manuscript_data: pd.DataFrame,
    age_column: str = "age",
    count_column: Optional[str] = None,
) -> None:
    """
    Plot birth-death process fit results (requires plotly).

    Args:
        model: Fitted BirthDeathProcess instance
        manuscript_data: Manuscript data used for fitting
        age_column: Column name for ages
        count_column: Column name for counts (optional)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available. Cannot create plots.")
        return

    if not model.fitted or model.fit_results is None:
        print("Model must be fitted before plotting.")
        return

    # Extract ages and counts from data
    if age_column not in manuscript_data.columns:
        raise ValueError(f"Column '{age_column}' not found in manuscript_data")

    ages = manuscript_data[age_column].values
    ages_array = np.asarray(ages, dtype=np.float64)

    if count_column is not None and count_column in manuscript_data.columns:
        counts = manuscript_data[count_column].values
        counts_array = np.asarray(counts, dtype=np.float64)
    else:
        counts_array = np.ones_like(ages_array, dtype=np.float64)

    # Create age bins for visualization
    age_bins = np.arange(0, float(ages_array.max()) + 50, 50, dtype=np.float64)
    hist_counts, bin_edges = np.histogram(
        ages_array, bins=age_bins, weights=counts_array
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Generate prediction curve
    age_range = np.linspace(0, ages_array.max(), 100, dtype=np.float64)
    predicted_counts = model.predict(age_range)

    # Create subplot
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Observed vs Predicted Manuscript Counts", "Residuals"],
        vertical_spacing=0.12,
    )

    # Plot observed data
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=hist_counts,
            mode="markers",
            name="Observed",
            marker=dict(size=8, color="blue"),
        ),
        row=1,
        col=1,
    )

    # Plot predicted curve
    fig.add_trace(
        go.Scatter(
            x=age_range,
            y=predicted_counts,
            mode="lines",
            name="Predicted",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=1,
    )

    # Plot residuals
    predicted_at_data = model.predict(bin_centers)
    residuals = hist_counts - predicted_at_data

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(size=6, color="green"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Add zero line for residuals
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Update layout
    title_text = (
        f"Birth-Death Process Fit<br>λ={model.birth_rate:.4f}, μ={model.death_rate:.4f}"
    )

    fig.update_layout(
        height=600,
        title=title_text,
    )
    fig.update_xaxes(title_text="Manuscript Age (years)", row=2, col=1)
    fig.update_yaxes(title_text="Number of Manuscripts", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)

    fig.show()


def plot_simulation_results(
    simulation_df: pd.DataFrame,
    age_bins: Optional[NDArray[np.float64]] = None,
    title: str = "Birth-Death Process Simulation Results",
) -> None:
    """
    Plot results from birth-death process simulation.

    Args:
        simulation_df: DataFrame from simulate_birth_death_process
        age_bins: Optional age bins for histogram
        title: Plot title
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available. Cannot create plots.")
        return

    surviving_manuscripts = simulation_df[simulation_df["survived"]]

    if len(surviving_manuscripts) == 0:
        print("No surviving manuscripts to plot.")
        return

    # Create age bins if not provided
    if age_bins is None:
        max_age = surviving_manuscripts["age"].max()
        age_bins = np.arange(0, max_age + 50, 50, dtype=np.float64)

    # Create subplot
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Age Distribution (Surviving)",
            "Creation Times vs Ages",
            "Manuscript Timeline",
            "Survival vs Death Statistics",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # 1. Age histogram
    fig.add_trace(
        go.Histogram(
            x=surviving_manuscripts["age"],
            xbins=dict(
                start=age_bins[0], end=age_bins[-1], size=age_bins[1] - age_bins[0]
            ),
            name="Age Distribution",
            marker_color="blue",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Creation times vs ages scatter plot
    fig.add_trace(
        go.Scatter(
            x=surviving_manuscripts["creation_time"],
            y=surviving_manuscripts["age"],
            mode="markers",
            name="Surviving",
            marker=dict(size=6, color="green"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Timeline of all manuscripts
    survived = simulation_df[simulation_df["survived"]]
    died = simulation_df[~simulation_df["survived"]]

    fig.add_trace(
        go.Scatter(
            x=survived["creation_time"],
            y=[1] * len(survived),
            mode="markers",
            name="Survived",
            marker=dict(size=8, color="green"),
        ),
        row=2,
        col=1,
    )

    if len(died) > 0:
        fig.add_trace(
            go.Scatter(
                x=died["creation_time"],
                y=[0] * len(died),
                mode="markers",
                name="Died",
                marker=dict(size=8, color="red"),
            ),
            row=2,
            col=1,
        )

    # 4. Summary statistics bar chart
    total_created = len(simulation_df)
    total_survived = len(survived)
    # Remove unused variable
    # survival_rate = total_survived / total_created if total_created > 0 else 0

    fig.add_trace(
        go.Bar(
            x=["Created", "Survived", "Died"],
            y=[total_created, total_survived, total_created - total_survived],
            marker=dict(color=["blue", "green", "red"]),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        height=800,
        title=title,
        showlegend=True,
    )

    # Update axis labels
    fig.update_xaxes(title_text="Age (years)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.update_xaxes(title_text="Creation Time", row=1, col=2)
    fig.update_yaxes(title_text="Age (years)", row=1, col=2)

    fig.update_xaxes(title_text="Creation Time", row=2, col=1)
    fig.update_yaxes(title_text="Status", row=2, col=1)

    fig.update_xaxes(title_text="Category", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    fig.show()


def plot_parameter_comparison(
    fit_results_list: list[Dict[str, Any]],
    parameter_names: list[str],
    title: str = "Parameter Comparison",
) -> None:
    """
    Plot comparison of fitted parameters across different datasets.

    Args:
        fit_results_list: List of fit result dictionaries
        parameter_names: List of names for each dataset
        title: Plot title
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available. Cannot create plots.")
        return

    if len(fit_results_list) != len(parameter_names):
        raise ValueError("fit_results_list and parameter_names must have same length")

    # Extract parameters
    birth_rates: list[float] = []
    death_rates: list[float] = []
    log_likelihoods: list[float] = []
    valid_names: list[str] = []

    for i, result in enumerate(fit_results_list):
        if result["success"]:
            birth_rates.append(float(result["birth_rate"]))
            death_rates.append(float(result["death_rate"]))
            log_likelihoods.append(float(result["log_likelihood"]))
            valid_names.append(parameter_names[i])
        # Skip failed fits instead of adding None values

    if not birth_rates:
        print("No successful fits to plot.")
        return

    # Create subplot
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Birth Rates", "Death Rates", "Log-Likelihoods"],
        horizontal_spacing=0.1,
    )

    # Birth rates
    fig.add_trace(
        go.Bar(
            x=valid_names,
            y=birth_rates,
            name="Birth Rate",
            marker_color="blue",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Death rates
    fig.add_trace(
        go.Bar(
            x=valid_names,
            y=death_rates,
            name="Death Rate",
            marker_color="red",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Log-likelihoods
    fig.add_trace(
        go.Bar(
            x=valid_names,
            y=log_likelihoods,
            name="Log-Likelihood",
            marker_color="green",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # Update layout
    fig.update_layout(
        height=400,
        title=title,
    )

    # Update axis labels
    fig.update_yaxes(title_text="Birth Rate (λ)", row=1, col=1)
    fig.update_yaxes(title_text="Death Rate (μ)", row=1, col=2)
    fig.update_yaxes(title_text="Log-Likelihood", row=1, col=3)

    fig.show()


def plot_bootstrap_confidence_intervals(
    bootstrap_results: Dict[str, Any],
    title: str = "Bootstrap Confidence Intervals",
) -> None:
    """
    Plot bootstrap confidence intervals for parameters.

    Args:
        bootstrap_results: Results from bootstrap_confidence_intervals
        title: Plot title
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not available. Cannot create plots.")
        return

    if not bootstrap_results["success"]:
        print("Bootstrap analysis failed, cannot plot.")
        return

    original_fit = bootstrap_results["original_fit"]
    birth_rate_ci = bootstrap_results["birth_rate_ci"]
    death_rate_ci = bootstrap_results["death_rate_ci"]

    fig = go.Figure()

    # Birth rate
    fig.add_trace(
        go.Scatter(
            x=["Birth Rate"],
            y=[float(original_fit["birth_rate"])],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[birth_rate_ci[1] - float(original_fit["birth_rate"])],
                arrayminus=[float(original_fit["birth_rate"]) - birth_rate_ci[0]],
            ),
            mode="markers",
            marker=dict(size=10, color="blue"),
            name="Birth Rate",
        )
    )

    # Death rate
    fig.add_trace(
        go.Scatter(
            x=["Death Rate"],
            y=[float(original_fit["death_rate"])],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[death_rate_ci[1] - float(original_fit["death_rate"])],
                arrayminus=[float(original_fit["death_rate"]) - death_rate_ci[0]],
            ),
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Death Rate",
        )
    )

    confidence_level = bootstrap_results["confidence_level"]
    n_bootstrap = bootstrap_results["n_bootstrap"]

    fig.update_layout(
        title=f"{title}<br>({confidence_level * 100:.0f}% CI, {n_bootstrap} bootstrap samples)",
        xaxis_title="Parameter",
        yaxis_title="Value",
        showlegend=True,
        height=500,
    )

    fig.show()
