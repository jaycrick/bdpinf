"""Analysis helpers for notebook and report workflows."""

from inf_bp_texts.visualization.parameter_plots import (
    BoxPlotStyle,
    KdePlotConfig,
    compute_boxplot_summary,
    create_kde_density_plot,
    create_parameter_box_plot,
)

from .single_epoch import (
    compute_parameter_summary,
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
    build_hist_bins,
    extract_epoch_parameter_arrays,
    first_failed_fits,
    fit_single_model,
    get_selected_attempt,
    has_ages_in_each_epoch,
    iqr_plotting_subset,
    proportion_not_at_upper,
    robust_y_max,
    theoretical_age_density,
)

__all__ = [
    "BoxPlotStyle",
    "FitModelConfig",
    "KdePlotConfig",
    "bootstrap_percentile_ci",
    "boundary_corrected_kde_at_zero",
    "boundary_corrected_kde_on_support",
    "build_hist_bins",
    "compute_boxplot_summary",
    "compute_ci_width_table",
    "compute_parameter_summary",
    "create_kde_density_plot",
    "create_parameter_box_plot",
    "extract_alive_ages",
    "extract_epoch_parameter_arrays",
    "extract_rate_arrays",
    "first_failed_fits",
    "fit_single_model",
    "get_selected_attempt",
    "has_ages_in_each_epoch",
    "iqr_plotting_subset",
    "proportion_not_at_upper",
    "robust_y_max",
    "select_extreme_replicates",
    "select_max_ci_times",
    "summarize_fit_success",
    "theoretical_age_density",
]
