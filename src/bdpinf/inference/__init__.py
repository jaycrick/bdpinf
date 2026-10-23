"""Inference package for birth-death process parameter estimation.

This package contains likelihood calculation, optimization, and fitting utilities
for birth-death process models.
"""

from .bayesian import (
    BayesianMCMCConfig,
    BayesianPriorConfig,
    BayesianResult,
    fit_bayesian,
)
from .constraints import CompiledConstraints, Constraints
from .fitting import (
    FitResult,
    fit,
    get_crude_parameter_estimates,
    moment_initial_rates,
    profile_likelihood_interval,
)
from .likelihood import (
    AgeData,
    calculate_unknown_spine_mixture_age_density,
    calculate_unknown_spine_mixture_birth_density,
    log_likelihood,
    log_likelihood_rates,
)
from .persistence import (
    load_posterior_samples,
    load_recovery_results,
    save_posterior_samples,
    save_recovery_results,
)
from .rate_models import (
    BirthDeathModel,
    Param,
    PiecewiseConstant,
    PiecewiseLinear,
    RateModel,
)
from .uncertainty import (
    confidence_intervals_from_log_covariance,
    covariance_from_hessian,
)

__all__ = [
    "AgeData",
    "BayesianMCMCConfig",
    "BayesianPriorConfig",
    "BayesianResult",
    "BirthDeathModel",
    "CompiledConstraints",
    "Constraints",
    "FitResult",
    "Param",
    "PiecewiseConstant",
    "PiecewiseLinear",
    "RateModel",
    "calculate_unknown_spine_mixture_age_density",
    "calculate_unknown_spine_mixture_birth_density",
    "confidence_intervals_from_log_covariance",
    "covariance_from_hessian",
    "fit",
    "fit_bayesian",
    "get_crude_parameter_estimates",
    "load_posterior_samples",
    "load_recovery_results",
    "log_likelihood",
    "log_likelihood_rates",
    "moment_initial_rates",
    "profile_likelihood_interval",
    "save_posterior_samples",
    "save_recovery_results",
]
