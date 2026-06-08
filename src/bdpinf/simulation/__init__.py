"""Simulation package for birth-death process data generation.

This package contains simulation and prediction utilities for birth-death
process models.
"""

from .simulator import (
    simulate_bdp_ages_from_distributions,
    simulate_birth_death_process,
    simulate_birth_death_process_from_contour_process,
)
from .splitting import (
    SplittingConfig,
    plot_conditioned_trajectories,
    plot_empirical_population_mean,
    plot_level_survival_probabilities,
    select_level_times_for_band,
    simulate_subcritical_survival_splitting,
)

__all__ = [
    "SplittingConfig",
    "plot_conditioned_trajectories",
    "plot_empirical_population_mean",
    "plot_level_survival_probabilities",
    "select_level_times_for_band",
    "simulate_bdp_ages_from_distributions",
    "simulate_birth_death_process",
    "simulate_birth_death_process_from_contour_process",
    "simulate_subcritical_survival_splitting",
]
