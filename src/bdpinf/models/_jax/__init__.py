"""JAX likelihood spike — Phase 2a of the PyTorch → JAX migration.

This transient sub-package contains JAX re-expressions of the four torch math
modules used in Phase 2a to:

1. Run the numerical equivalence gate (``tests/test_jax_torch_equivalence.py``).
2. Benchmark JAX vs torch CPU throughput.

In Phase 2b the implementations move *in-place* into the parent modules
(``math_extinction.py``, ``math_ages.py``, ``math_populations.py``,
``inference/likelihood.py``) and this package is deleted.

The public names mirror the torch equivalents exactly so the equivalence tests
can import from both and compare.
"""

# Enable float64 globally on import — this module must be imported before
# any jnp computation or the default float32 truncates silently.
import jax

jax.config.update("jax_enable_x64", True)

from inf_bp_texts.models._jax.likelihood import log_likelihood
from inf_bp_texts.models._jax.math_ages import (
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)
from inf_bp_texts.models._jax.math_extinction import (
    calculate_extinction_pr,
    calculate_log_lik_initial_lifetimes_given_extinction,
    calculate_log_lik_initial_lifetimes_given_non_extinction,
    calculate_log_pr_extinction_given_initial_lifetimes,
    pr_extinction_numerator,
)
from inf_bp_texts.models._jax.math_populations import (
    calculate_populations_log_lik,
    population_geometric_parameter,
)

__all__ = [
    "calculate_extinction_pr",
    "calculate_log_lik_initial_lifetimes_given_extinction",
    "calculate_log_lik_initial_lifetimes_given_non_extinction",
    "calculate_log_pr_extinction_given_initial_lifetimes",
    "calculate_non_spine_ages_log_lik",
    "calculate_populations_log_lik",
    "calculate_spine_ages_log_lik",
    "log_likelihood",
    "population_geometric_parameter",
    "pr_extinction_numerator",
]
