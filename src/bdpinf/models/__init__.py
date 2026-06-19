"""Models package for birth-death process analysis.

This package contains the core model definitions and mathematical functions
for birth-death process modeling of manuscript data.
"""

from .birth_death_process import BirthDeathProcess
from .math_ages import calculate_ages_log_lik, calculate_non_spine_ages_log_lik
from .math_populations import (
    calculate_populations_log_lik,
    calculate_populations_log_pmf,
)

__all__ = [
    "BirthDeathProcess",
    "calculate_ages_log_lik",
    "calculate_non_spine_ages_log_lik",
    "calculate_populations_log_lik",
    "calculate_populations_log_pmf",
]
