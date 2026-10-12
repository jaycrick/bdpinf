r"""Age log-likelihood calculations for birth-death processes.

Thin array-in/array-out wrappers over the paper's closed forms in
:mod:`bdpinf.models.math_nested` (epoch-wise :math:`\mathcal W`).
With observation time :math:`T`:

- non-spine age :math:`A` (density on :math:`(0, T)`, no atom):
  :math:`f(a) = \lambda(T-a)\,\mathcal W(T-a)\,e^{-M(T-a,T]} / (\mathcal W(T) - 1)`;
- spine age :math:`A^*`:
  :math:`f^*(a) = \lambda(T-a)\,e^{-\Lambda(T-a,T]}\,L_\mu(T-a,T)` on
  :math:`(0, T)` plus an atom :math:`\Pr(A^* = T) = \mathcal W(T) e^{-\Lambda(0,T]}`
  evaluated for ``age >= T``.

For one epoch these reduce to Kück & Schuhmacher (2018) eqs. 3.6 and 3.8
(the paper's constant-rate corollaries), including the critical case
:math:`\lambda = \mu` exactly.
The multi-epoch likelihood needs no time reversal: :math:`\mathcal W` has an
exact epoch-wise recursion (the "restart principle").

All inputs are assumed float64 (``jax_enable_x64`` is set in __init__.py).
"""

import jax
from jaxtyping import Array, Float

from bdpinf.models.math_nested import (
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.models.validation import normalize_and_validate_inputs

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_non_spine_ages_log_lik(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    ages: Float[Array, " n"],
    times: Float[Array, " epochs"],
    *,
    return_array: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    r"""Log-likelihood of non-spine individual ages.

    Parameters
    ----------
    birth_rates, death_rates : jax.Array
        Shape (n_epochs,) float64, all non-negative.
    ages : jax.Array
        Shape (n,) float64, 0 ≤ age ≤ times[-1].
    times : jax.Array
        Shape (n_epochs,) float64, strictly increasing; ``times[-1]`` is :math:`T`.
    return_array : bool
        Per-age values if True, summed scalar if False.

    Returns
    -------
    jax.Array
        Scalar or (n,) float64.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times, ages = normalize_and_validate_inputs(
            birth_rates, death_rates, times, ages
        )
    terms = nested_terms(birth_rates, death_rates, times)
    log_density = non_spine_log_density(terms, ages)
    return log_density if return_array else log_density.sum()


def calculate_spine_ages_log_lik(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    ages: Float[Array, " n"],
    times: Float[Array, " epochs"],
    *,
    return_array: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    r"""Log-likelihood of spine individual ages (with boundary mass at T).

    Ages ``>= times[-1]`` are scored with the atom
    :math:`\Pr(A^* = T) = \mathcal W(T) e^{-\Lambda(0,T]}`, all others with the
    continuous density :math:`f^*`.

    Parameters
    ----------
    birth_rates, death_rates : jax.Array
        Shape (n_epochs,) float64.
    ages : jax.Array
        Shape (n,) float64, 0 ≤ age ≤ times[-1].
    times : jax.Array
        Shape (n_epochs,) float64, strictly increasing.
    return_array : bool
        Per-age values if True, summed scalar if False.

    Returns
    -------
    jax.Array
        Scalar or (n,) float64.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times, ages = normalize_and_validate_inputs(
            birth_rates, death_rates, times, ages
        )
    terms = nested_terms(birth_rates, death_rates, times)
    log_density = spine_log_density(terms, ages)
    return log_density if return_array else log_density.sum()


def calculate_ages_log_lik(
    birth_rates: jax.Array | float,
    death_rates: jax.Array | float,
    ages: jax.Array | float,
    times: jax.Array | float,
    *,
    are_spines: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    r"""Compute log-likelihood of individual ages under a birth-death model.

    Dispatches to spine or non-spine likelihood based on ``are_spines``.

    Parameters
    ----------
    birth_rates : jax.Array | float
        Birth rate(s) λ. Scalar for single epoch, 1D array for piecewise-constant.
    death_rates : jax.Array | float
        Death rate(s) μ. Same shape as birth_rates.
    ages : jax.Array | float
        Individual age(s) t. All ages must satisfy 0 ≤ t ≤ T.
    times : jax.Array | float
        Epoch end time(s) T. Scalar or strictly-increasing 1D array.
    are_spines : bool
        If True, compute spine likelihood. If False, compute non-spine. Default False.

    Returns
    -------
    jax.Array
        Scalar float64 summed log-likelihood. Returns -inf for invalid inputs.

    Raises
    ------
    ValueError
        If array inputs have mismatched lengths, times not strictly increasing,
        or ages outside [0, T].

    """
    if are_spines:
        return calculate_spine_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times,
            _skip_normalization=_skip_normalization,
        )
    return calculate_non_spine_ages_log_lik(
        birth_rates,
        death_rates,
        ages,
        times,
        _skip_normalization=_skip_normalization,
    )
