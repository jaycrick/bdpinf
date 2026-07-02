"""Inference package for birth-death process parameter estimation.

This package contains likelihood calculation, optimization, and fitting utilities
for birth-death process models.
"""

from ._fit_types import ConvergenceConfig
from .bayesian import (
    BayesianMCMCConfig,
    BayesianModelConfig,
    BayesianPriorConfig,
    fit_birth_death_processes_bayesian,
    fit_birth_death_processes_bayesian_from_data_frame,
    mle_init_log_rates,
)
from .fitting import (
    fit_birth_death_processes,
    fit_from_data_frame,
    get_crude_parameter_estimates,
)
from .likelihood import (
    calculate_unknown_spine_mixture_age_density,
    calculate_unknown_spine_mixture_birth_density,
    log_likelihood,
    negative_log_likelihood,
)
from .persistence import (
    load_posterior_samples,
    load_recovery_results,
    save_posterior_samples,
    save_recovery_results,
)
from .uncertainty import (
    confidence_intervals_from_log_covariance,
    estimate_mle_covariance,
)

__all__ = [
    "ConvergenceConfig",
    "BayesianMCMCConfig",
    "BayesianModelConfig",
    "BayesianPriorConfig",
    "get_crude_parameter_estimates",
    "calculate_unknown_spine_mixture_age_density",
    "calculate_unknown_spine_mixture_birth_density",
    "confidence_intervals_from_log_covariance",
    "estimate_mle_covariance",
    "fit_birth_death_processes",
    "fit_birth_death_processes_bayesian",
    "fit_birth_death_processes_bayesian_from_data_frame",
    "fit_from_data_frame",
    "load_posterior_samples",
    "load_recovery_results",
    "mle_init_log_rates",
    "log_likelihood",
    "negative_log_likelihood",
    "save_posterior_samples",
    "save_recovery_results",
]
