r"""Extinction probability calculations for birth-death processes.

Notation follows the paper:
$\Gamma(0,t] = \int_0^t (\lambda - \mu)$,
$I(T) = \int_0^T \mu(t)\, e^{-\Gamma(0,t]}\,dt$,
$\mathcal W(T) = e^{\Gamma(0,T]}(1 + I(T))$ and $\pi(T) = 1/\mathcal W(T)$.
$I(T) = P_0(T)/(1 - P_0(T))$ is the odds of extinction by $T$,
so $\Pr(Z_T = 0) = I(T) / (1 + I(T))$.
Extinction is almost sure as $T \to \infty$ iff $I(\infty)$ diverges
(Kendall's criterion, Kendall (1948); see ``bibliography.bib``).

The epoch-wise closed forms live in :mod:`bdpinf.models.math_nested`
(:func:`~bdpinf.models.math_nested.log_extinction_and_survival_pr` gives
$\ln P_0(T)$ and $\ln(1 - P_0(T))$); this module exposes the probability
built on them.

All inputs are assumed float64 (``jax_enable_x64`` is set in __init__.py).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bdpinf.models.math_nested import log_extinction_and_survival_pr, nested_terms


def calculate_extinction_pr(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    return_log_prob: bool = False,
) -> jax.Array:
    r"""Extinction probability $P_0(T) = \Pr(Z_T = 0) = I/(1+I)$, or its log.

    Formula: see the section "Piecewise-constant rates (exact)"
    of ``docs/likelihoods.qmd``.

    Evaluated as $\ln I - \operatorname{softplus}(\ln I)$, which stays
    accurate when $I$ is tiny.

    Parameters
    ----------
    birth_rates, death_rates : jax.Array | float
        Non-negative epoch rates, shape (n_epochs,) float64.
    times : jax.Array | float
        Strictly increasing epoch end times, shape (n_epochs,); ``times[-1]``
        is the observation time $T$.
    return_log_prob : bool
        If True return log P(Z_T = 0). Default False.

    Returns
    -------
    jax.Array
        Scalar float64.

    """
    log_pr, _ = log_extinction_and_survival_pr(
        nested_terms(birth_rates, death_rates, times)
    )
    return log_pr if return_log_prob else jnp.exp(log_pr)
