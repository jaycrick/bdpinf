r"""Extinction probability calculations for birth-death processes.

Notation follows the paper:
$\Gamma(0,t] = \int_0^t (\lambda - \mu)$,
$N(T) = \int_0^T \mu(t)\, e^{-\Gamma(0,t]}\,dt$,
$\mathcal W(T) = e^{\Gamma(0,T]}(1 + N(T))$ and $\pi(T) = 1/\mathcal W(T)$.
The extinction probability is $\Pr(Z_T = 0) = N(T) / (1 + N(T))$.
In the code $N(T)$ is called the *extinction numerator* (``ext_num``); it
equals $\mathcal W(T) - 1$ only after the rescaling above, so
$\pi(T) = 1/(1 + N(T))$ holds only when $\Gamma(0,T] = 0$.

The epoch-wise closed forms live in :mod:`bdpinf.models.math_nested`
(:func:`~bdpinf.models.math_nested.log_extinction_numerator` gives
$\ln N(T)$); this module exposes the probability built on them.

All inputs are assumed float64 (``jax_enable_x64`` is set in __init__.py).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bdpinf.models.math_nested import (
    log_extinction_numerator,
    nested_terms,
    restore_neg_inf,
)


def calculate_extinction_pr(
    birth_rates: Float[Array, " epochs"],
    death_rates: Float[Array, " epochs"],
    times: Float[Array, " epochs"],
    *,
    return_log_prob: bool = False,
) -> jax.Array:
    r"""Extinction probability $P_0(T) = \Pr(Z_T = 0) = N/(1+N)$, or its log.

    Evaluated as $\ln N - \operatorname{softplus}(\ln N)$, which stays
    accurate when $N$ is tiny.

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
    log_n = restore_neg_inf(
        log_extinction_numerator(nested_terms(birth_rates, death_rates, times))
    )
    log_pr = restore_neg_inf(log_n - jnp.logaddexp(0.0, log_n))
    return log_pr if return_log_prob else jnp.exp(log_pr)
