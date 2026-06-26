"""Visualization package for birth-death process analysis.

This package contains plotting and visualization utilities for birth-death
process models and their results.
"""

from .age_fit_plot import plot_creation_year_fit
from .altair_theme import (
    PTOLQUAL,
    THEME_NAME,
    display_svg,
    register_export_fonts,
    save_chart,
    save_chart_html,
)
from .bayesian_plots import (
    PriorPosteriorPanel,
    RatePanel,
    TracePanel,
    posterior_rates_chart,
    ppc_chart,
    prior_posterior_chart,
    trace_chart,
)
from .date_distribution_plots import plot_date_range_kde_comparison
from .era_shading import era_band_chart, year_to_century_coord
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
    "PTOLQUAL",
    "THEME_NAME",
    "BoxPlotStyle",
    "KdePlotConfig",
    "PriorPosteriorPanel",
    "RatePanel",
    "TracePanel",
    "convergence_chart",
    "rate_ci_panel",
    "display_svg",
    "era_band_chart",
    "plot_creation_year_fit",
    "plot_date_range_kde_comparison",
    "ppc_chart",
    "posterior_rates_chart",
    "prior_posterior_chart",
    "register_export_fonts",
    "save_chart",
    "save_chart_html",
    "trace_chart",
    "year_to_century_coord",
    "compute_boxplot_summary",
    "create_kde_density_plot",
    "create_parameter_box_plot",
    "plot_conditioned_trajectories",
    "plot_empirical_population_mean",
    "plot_level_survival_probabilities",
    "save_multi_epoch_birth_time_figure",
    "save_multi_epoch_convergence_figure",
    "save_multi_epoch_fit_figure",
]
