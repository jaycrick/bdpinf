"""Inference package for birth-death process parameter estimation.

This package contains likelihood calculation, optimization, and fitting utilities
for birth-death process models.
"""

from .bayesian import (
    BayesianMCMCConfig,
    BayesianPriorConfig,
    BayesianResult,
    fit_bayesian,
    sample_prior,
)
from .constraints import CompiledConstraints, Constraints
from .fitting import (
    FitResult,
    fit,
    get_crude_parameter_estimates,
    moment_initial_rates,
    profile_likelihood_interval,
)
from .knots import (
    KnotFitResult,
    MethodResult,
    evenly_spaced_knots,
    fit_knots,
    neighbour_placements,
    nested_start,
    place_knots,
)
from .likelihood import (
    AgeData,
    calculate_unknown_spine_mixture_age_density,
    calculate_unknown_spine_mixture_birth_density,
    log_likelihood,
    log_likelihood_rates,
    range_bounds,
)
from .persistence import (
    load_posterior_samples,
    load_recovery_results,
    save_posterior_samples,
    save_recovery_results,
)
from .rate_models import (
    BirthDeathModel,
    FreeKnots,
    Param,
    PiecewiseConstantModel,
    PiecewiseLinearModel,
    RateModel,
    SplineKind,
    SplineModel,
)
from .selection import effective_df, information_criteria
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
    "FreeKnots",
    "KnotFitResult",
    "MethodResult",
    "Param",
    "PiecewiseConstantModel",
    "PiecewiseLinearModel",
    "RateModel",
    "SplineKind",
    "SplineModel",
    "calculate_unknown_spine_mixture_age_density",
    "calculate_unknown_spine_mixture_birth_density",
    "confidence_intervals_from_log_covariance",
    "covariance_from_hessian",
    "effective_df",
    "evenly_spaced_knots",
    "fit",
    "fit_bayesian",
    "fit_knots",
    "get_crude_parameter_estimates",
    "information_criteria",
    "load_posterior_samples",
    "load_recovery_results",
    "log_likelihood",
    "log_likelihood_rates",
    "moment_initial_rates",
    "neighbour_placements",
    "nested_start",
    "place_knots",
    "profile_likelihood_interval",
    "range_bounds",
    "sample_prior",
    "save_posterior_samples",
    "save_recovery_results",
]
