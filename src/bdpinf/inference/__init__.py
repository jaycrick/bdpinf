"""Inference package for birth-death process parameter estimation.

This package contains likelihood calculation, optimization, and fitting utilities
for birth-death process models.
"""

from .fitting import (
    _get_crude_parameter_estimates,
    fit_birth_death_processes,
    fit_from_dataframe,
)
from .likelihood import log_likelihood, negative_log_likelihood

__all__ = [
    "_get_crude_parameter_estimates",
    "fit_birth_death_processes",
    "fit_from_dataframe",
    "log_likelihood",
    "negative_log_likelihood",
]
