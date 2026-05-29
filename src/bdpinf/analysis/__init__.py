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
from .spike_detection import (
    bootstrap_percentile_ci,
    compute_ci_width_table,
    select_extreme_replicates,
    select_max_ci_times,
)
from .validation_helpers import (
    FitModelConfig,
    boundary_corrected_kde_at_zero,
    boundary_corrected_kde_on_support,
    first_failed_fits,
    fit_single_model,
    iqr_plotting_subset,
)

__all__ = [
    "BoxPlotStyle",
    "FitModelConfig",
    "KdePlotConfig",
    "bootstrap_percentile_ci",
    "boundary_corrected_kde_at_zero",
    "boundary_corrected_kde_on_support",
    "compute_boxplot_summary",
    "compute_ci_width_table",
    "compute_parameter_summary",
    "create_kde_density_plot",
    "create_parameter_box_plot",
    "extract_alive_ages",
    "extract_rate_arrays",
    "first_failed_fits",
    "fit_single_model",
    "iqr_plotting_subset",
    "select_extreme_replicates",
    "select_max_ci_times",
    "summarize_fit_success",
]
