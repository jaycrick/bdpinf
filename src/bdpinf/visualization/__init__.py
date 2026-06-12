"""Visualization package for birth-death process analysis.

This package contains plotting and visualization utilities for birth-death
process models and their results.
"""

from .age_fit_plot import plot_creation_year_fit
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
    "BoxPlotStyle",
    "KdePlotConfig",
    "plot_creation_year_fit",
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
