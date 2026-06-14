"""Visualization package for birth-death process analysis.

This package contains plotting and visualization utilities for birth-death
process models and their results.
"""

from .age_fit_plot import plot_creation_year_fit
from .altair_theme import (
    PTOLQUAL,
    THEME_NAME,
    register_export_fonts,
    save_chart,
)
from .date_distribution_plots import (
    plot_date_range_kde_comparison,
    plot_grouped_date_distribution,
)
from .parameter_plots import (
    BoxPlotStyle,
    KdePlotConfig,
    compute_boxplot_summary,
    create_kde_density_plot,
    create_parameter_box_plot,
)
from .plotting import (
    save_figure,
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
    "plot_creation_year_fit",
    "plot_date_range_kde_comparison",
    "plot_grouped_date_distribution",
    "register_export_fonts",
    "save_chart",
    "compute_boxplot_summary",
    "create_kde_density_plot",
    "create_parameter_box_plot",
    "plot_conditioned_trajectories",
    "plot_empirical_population_mean",
    "plot_level_survival_probabilities",
    "save_figure",
    "save_multi_epoch_birth_time_figure",
    "save_multi_epoch_convergence_figure",
    "save_multi_epoch_fit_figure",
]
