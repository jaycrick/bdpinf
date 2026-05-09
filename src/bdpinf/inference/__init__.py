"""Inference package for birth-death process parameter estimation.

This package contains likelihood calculation, optimization, and fitting utilities
for birth-death process models.
"""

from .bayesian import (
    BayesianMCMCConfig,
    BayesianModelConfig,
    BayesianPriorConfig,
    fit_birth_death_processes_bayesian,
    fit_birth_death_processes_bayesian_from_data_frame,
)
from .fitting import (
    _get_crude_parameter_estimates,
    fit_birth_death_processes,
    fit_from_data_frame,
)
from .likelihood import (
    calculate_unknown_spine_mixture_age_density,
    calculate_unknown_spine_mixture_birth_density,
    log_likelihood,
    negative_log_likelihood,
)
from .uncertainty import (
    confidence_intervals_from_log_covariance,
    estimate_mle_covariance,
)

__all__ = [
    "BayesianMCMCConfig",
    "BayesianModelConfig",
    "BayesianPriorConfig",
    "_get_crude_parameter_estimates",
    "calculate_unknown_spine_mixture_age_density",
    "calculate_unknown_spine_mixture_birth_density",
    "confidence_intervals_from_log_covariance",
    "estimate_mle_covariance",
    "fit_birth_death_processes",
    "fit_birth_death_processes_bayesian",
    "fit_birth_death_processes_bayesian_from_data_frame",
    "fit_from_data_frame",
    "log_likelihood",
    "negative_log_likelihood",
]
