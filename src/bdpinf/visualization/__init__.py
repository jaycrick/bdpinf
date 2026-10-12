"""Visualization package for birth-death process analysis.

This package contains plotting and visualization utilities for birth-death
process models and their results.
"""

from .age_density_plots import (
    age_density_frame,
    rate_frame,
    scenario_panel,
    scenario_row,
)
from .age_fit_plot import plot_creation_year_fit
from .altair_theme import (
    DIVERGING_RDBU,
    PTOLQUAL,
    THEME_NAME,
    display_svg,
    get_figure_dir,
    register_export_fonts,
    save_chart,
    save_chart_html,
    set_figure_dir,
)
from .bayesian_plots import (
    PriorPosteriorPanel,
    PriorPosteriorRatePanel,
    RatePanel,
    TracePanel,
    posterior_rates_chart,
    ppc_chart,
    prior_posterior_chart,
    prior_posterior_rates_chart,
    prior_rate_samples,
    prior_rates_chart,
    trace_chart,
)
from .correlation_plots import correlation_heatmap_chart, posterior_rate_correlation
from .date_distribution_plots import plot_date_range_kde_comparison
from .parameter_plots import (
    BoxPlotStyle,
    KdePlotConfig,
    compute_boxplot_summary,
    create_kde_density_plot,
    create_parameter_box_plot,
)
from .plotting import (
    convergence_chart,
    rate_ci_panel,
    save_multi_epoch_birth_time_figure,
    save_multi_epoch_convergence_figure,
    save_multi_epoch_fit_figure,
)
from .splitting_plots import (
    plot_conditioned_trajectories,
    plot_empirical_population_mean,
    plot_level_survival_probabilities,
)

__all__ = [
    "DIVERGING_RDBU",
    "PTOLQUAL",
    "THEME_NAME",
    "BoxPlotStyle",
    "KdePlotConfig",
    "PriorPosteriorPanel",
    "PriorPosteriorRatePanel",
    "RatePanel",
    "TracePanel",
    "age_density_frame",
    "compute_boxplot_summary",
    "convergence_chart",
    "correlation_heatmap_chart",
    "create_kde_density_plot",
    "create_parameter_box_plot",
    "display_svg",
    "get_figure_dir",
    "plot_conditioned_trajectories",
    "plot_creation_year_fit",
    "plot_date_range_kde_comparison",
    "plot_empirical_population_mean",
    "plot_level_survival_probabilities",
    "posterior_rate_correlation",
    "posterior_rates_chart",
    "ppc_chart",
    "prior_posterior_chart",
    "prior_posterior_rates_chart",
    "prior_rate_samples",
    "prior_rates_chart",
    "rate_ci_panel",
    "rate_frame",
    "register_export_fonts",
    "save_chart",
    "save_chart_html",
    "save_multi_epoch_birth_time_figure",
    "save_multi_epoch_convergence_figure",
    "save_multi_epoch_fit_figure",
    "scenario_panel",
    "scenario_row",
    "set_figure_dir",
    "trace_chart",
]
