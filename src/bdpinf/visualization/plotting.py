"""Visualization functions for birth-death process analysis.

This module contains plotting and visualization utilities for birth-death
process models and their results.

"""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.graph_objs as go

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
layout_config = {
    "plot_bgcolor": plot_bgcolor,
    "paper_bgcolor": paper_bgcolor,
    "font": font_config,
    "title_font": title_font_config,
    "xaxis": {
        "gridcolor": gridcolor,
        "gridwidth": 0.5,
        "showgrid": False,
        "zeroline": False,
        "automargin": True,
    },
    "yaxis": {
        "gridcolor": gridcolor,
        "gridwidth": 0.5,
        "showgrid": False,
        "zeroline": False,
        "automargin": True,
    },
    "margin": {"l": 0, "r": 0, "t": 1, "b": 1, "autoexpand": True},
}


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

    Returns:
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


def apply_tufte_legend(
    fig: Any,
    position: str = "top right",
    bgcolor: str = "rgba(250,250,250,0.6)",
    bordercolor: str = "rgba(0,0,0,0.08)",
    font_size: int = FONT_SIZE - 2,
    *,
    showlegend: bool = True,
) -> Any:
    """Apply a small, Tufte-style internal legend to a Plotly figure.

    position: one of ('top right', 'top left', 'bottom right', 'bottom left')
    """
    pos_map = {
        "top right": {"x": 0.98, "y": 0.98, "xanchor": "right", "yanchor": "top"},
        "top left": {"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top"},
        "bottom right": {"x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"},
        "bottom left": {"x": 0.02, "y": 0.02, "xanchor": "left", "yanchor": "bottom"},
    }

    p = pos_map.get(position, pos_map["top right"])

    fig.update_layout(
        showlegend=showlegend,
        legend={
            "orientation": "v",
            "y": p["y"],
            "x": p["x"],
            "yanchor": p["yanchor"],
            "xanchor": p["xanchor"],
            "bgcolor": bgcolor,
            "bordercolor": bordercolor,
            "borderwidth": 0.5,
            "font": {"size": font_size, "color": "black"},
        },
    )

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

    Returns:
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
        print(f"⚠️ Failed to save {filename}.pdf: {e}")
        return None


def lineage_to_figure(
    lineage_display: pd.DataFrame,
    *,
    title: str = "Simulated Manuscript Tree in DFS Order",
    xaxis_title: str = "DFS order",
    yaxis_title: str = "Tree depth",
    height: int | None = None,
) -> Any:
    """Create a Plotly figure showing a lineage tree using DFS ordering.

    Args:
        lineage_display: DataFrame produced by `birth_death_lineage_to_display_frame`.
        title: Figure title.
        xaxis_title: X axis label.
        yaxis_title: Y axis label.
        height: Optional figure height in pixels. If None, computed from tree depth.

    Returns:
        Plotly Figure object.

    """
    if lineage_display.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Coordinates: x is dfs_order, y is depth
    tree_coords: dict[int, tuple[int, int]] = {}
    for row in lineage_display.itertuples(index=False):
        key = int(cast("int", row.manuscript_id))
        tree_coords[key] = (
            int(cast("int", row.dfs_order)),
            int(cast("int", row.depth)),
        )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for row in lineage_display.itertuples(index=False):
        if pd.isna(row.parent_id):
            continue
        parent_key = int(cast("int", row.parent_id))
        child_key = int(cast("int", row.manuscript_id))
        parent_x, parent_y = tree_coords[parent_key]
        child_x, child_y = tree_coords[child_key]
        edge_x.extend([parent_x, child_x, None])
        edge_y.extend([parent_y, child_y, None])

    node_x: list[int] = [
        tree_coords[int(cast("int", row.manuscript_id))][0]
        for row in lineage_display.itertuples(index=False)
    ]

    node_y: list[int] = [
        tree_coords[int(cast("int", row.manuscript_id))][1]
        for row in lineage_display.itertuples(index=False)
    ]

    node_text: list[str] = [
        f"m{int(node_id)}" for node_id in lineage_display["manuscript_id"]
    ]

    node_colors: list[str] = []
    for is_spine in lineage_display["is_spine"]:
        node_colors.append(ptolqual[0] if bool(is_spine) else ptolqual[2])

    node_hover: list[str] = []
    for row in lineage_display.itertuples(index=False):
        mid = int(cast("int", row.manuscript_id))
        parent_lbl = (
            "root" if pd.isna(row.parent_id) else f"m{int(cast('int', row.parent_id))}"
        )
        hover = (
            f"m{mid}"
            f"<br>parent: {parent_lbl}"
            f"<br>dfs order: {int(cast('int', row.dfs_order))}"
            f"<br>depth: {int(cast('int', row.depth))}"
            f"<br>creation time: {float(cast('float', row.creation_time)):.2f}"
        )
        node_hover.append(hover)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cast("Any", edge_x),
            y=cast("Any", edge_y),
            mode="lines",
            line=dict(color="#8A8A8A", width=1.4),
            hoverinfo="skip",
            showlegend=False,
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(
                size=11,
                color=node_colors,
                line=dict(color="#404040", width=0.6),
            ),
            hovertext=node_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ),
    )

    # Compute height safely; lineage_display is non-empty here
    if height is not None:
        computed_height: int = height
    else:
        # lineage_display["depth"].max() should be numeric when DataFrame non-empty
        max_depth_val = lineage_display["depth"].max()
        try:
            max_depth_int = int(max_depth_val)
        except Exception:
            max_depth_int = 0
        computed_height = max(420, 110 + 70 * int(max_depth_int + 1))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        height=computed_height,
    )

    fig = apply_professional_styling(fig, "", xaxis_title, yaxis_title)
    fig = apply_tufte_axes(fig)
    fig.update_yaxes(autorange="reversed", showgrid=True, gridcolor=gridcolor)
    fig.update_xaxes(showgrid=True, gridcolor=gridcolor)

    return fig
