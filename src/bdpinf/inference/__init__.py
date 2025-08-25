"""
Inference package for birth-death process parameter estimation.

This package contains likelihood calculation, optimization, and fitting utilities
for birth-death process models.
"""

from .fitting import (
    bootstrap_confidence_intervals,
    fit_birth_death_process,
    fit_from_dataframe,
    get_crude_parameter_estimates,
)
from .likelihood import (
    log_likelihood,
    negative_log_likelihood,
)

__all__ = [
    "log_likelihood",
    "negative_log_likelihood",
    "fit_birth_death_process",
    "fit_from_dataframe",
    "get_crude_parameter_estimates",
    "bootstrap_confidence_intervals",
]
