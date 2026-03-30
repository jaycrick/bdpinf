"""Analysis helpers for notebook and report workflows."""

from .single_epoch import (
    BoxPlotStyle,
    KdePlotConfig,
    compute_boxplot_summary,
    compute_parameter_summary,
    create_kde_density_plot,
    create_parameter_box_plot,
    extract_alive_ages,
    extract_rate_arrays,
    summarize_fit_success,
)

__all__ = [
    "BoxPlotStyle",
    "KdePlotConfig",
    "compute_boxplot_summary",
    "compute_parameter_summary",
    "create_kde_density_plot",
    "create_parameter_box_plot",
    "extract_alive_ages",
    "extract_rate_arrays",
    "summarize_fit_success",
]
