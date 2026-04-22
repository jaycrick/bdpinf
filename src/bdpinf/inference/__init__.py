"""Inference package for birth-death process parameter estimation.

This package contains likelihood calculation, optimization, and fitting utilities
for birth-death process models.
"""

from .bayesian import (
    BayesianMCMCConfig,
    BayesianModelConfig,
    BayesianPriorConfig,
    fit_birth_death_processes_bayesian,
    fit_birth_death_processes_bayesian_from_dataframe,
)
from .fitting import (
    _get_crude_parameter_estimates,
    fit_birth_death_processes,
    fit_from_dataframe,
)
from .likelihood import (
    calculate_unknown_spine_mixture_age_density,
    log_likelihood,
    negative_log_likelihood,
)

__all__ = [
    "BayesianMCMCConfig",
    "BayesianModelConfig",
    "BayesianPriorConfig",
    "_get_crude_parameter_estimates",
    "calculate_unknown_spine_mixture_age_density",
    "fit_birth_death_processes",
    "fit_birth_death_processes_bayesian",
    "fit_birth_death_processes_bayesian_from_dataframe",
    "fit_from_dataframe",
    "log_likelihood",
    "negative_log_likelihood",
]
