"""Mathematical functions for birth-and-death process calculations.

Public re-export facade. Implementation is split across three private
submodules to keep each file focused:

- ``_math_extinction``: extinction probability and initial-lifetime likelihoods
- ``_math_ages``: spine/non-spine/combined manuscript age likelihoods
- ``_math_populations``: population count likelihood

All names that were previously importable from this module remain importable
from this module unchanged.
"""

from inf_bp_texts.models._math_ages import (
    _reverse_time,
    calculate_ages_log_lik,
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)
from inf_bp_texts.models._math_extinction import (
    _integrand_surv_from_weighted_birth_time,
    _pr_extinction_numerator,
    calculate_extinction_pr,
    calculate_log_lik_initial_lifetimes_given_extinction,
    calculate_log_lik_initial_lifetimes_given_non_extinction,
    calculate_log_lik_lifetimes,
    calculate_log_pr_extinction_given_initial_lifetimes,
)
from inf_bp_texts.models._math_populations import (
    _compute_populations_log_lik_multi_start,
    _population_geometric_parameter,
    calculate_populations_log_lik,
)

__all__ = [
    # Public functions
    "calculate_ages_log_lik",
    "calculate_extinction_pr",
    "calculate_log_lik_initial_lifetimes_given_extinction",
    "calculate_log_lik_initial_lifetimes_given_non_extinction",
    "calculate_log_lik_lifetimes",
    "calculate_log_pr_extinction_given_initial_lifetimes",
    "calculate_non_spine_ages_log_lik",
    "calculate_populations_log_lik",
    "calculate_spine_ages_log_lik",
    # Private helpers — re-exported for test access
    "_compute_populations_log_lik_multi_start",
    "_integrand_surv_from_weighted_birth_time",
    "_population_geometric_parameter",
    "_pr_extinction_numerator",
    "_reverse_time",
]
