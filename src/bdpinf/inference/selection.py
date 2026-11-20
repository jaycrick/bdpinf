r"""Information criteria with a charge for searched knot locations.

Knot locations are irregular parameters: the likelihood is not a smooth,
regular function of them, and choosing them by search adds optimism beyond one
degree of freedom each.
:func:`effective_df` charges every free knot ``knot_cost`` degrees of freedom in
total (its coefficient plus ``knot_cost - 1`` for its location and the search),
and :func:`information_criteria` turns a log-likelihood and that count into
AIC, AICc and BIC.

Likelihood form
---------------
The criteria use $-2\hat\ell$ throughout.
The regression form with ``n * ln(RSS / n)`` is the Gaussian special case, and
``n ln(RSS/n)`` equals $-2\hat\ell$ up to a constant that does not depend on
the model, so the two rank models identically.

Degrees of freedom
------------------
For one spline of degree $d$ with $K$ knots, ``p_rates = d + 1 + K`` (the
polynomial coefficients plus one per knot), so
``effective_df(d + 1 + K, K, c) = d + 1 + c K``.

Choosing the knot cost ``c``
----------------------------
* ``c = 3.0`` (default): spaced free knots.
* ``c = 3.5`` to ``4.0``: unconstrained knots that may coalesce.
* ``c = 2.0`` to ``2.5``: a coarse grid of knot locations.

These are guidance, not derived constants.
Friedman (1991, MARS) charges about three degrees of freedom per knot *in
addition to* the coefficient (about 4 in the units of ``knot_cost``).
A specific constant from Spiriti et al. (free-knot splines) is not verified
here and is not claimed.

AICc
----
The small-sample correction is the Gaussian linear-regression one, so it is an
approximation for these likelihoods.
It is infinite when ``n - df - 1 <= 0``, and the usual rule recommends it
over AIC when ``n / df < 40`` (``aicc_recommended``).
"""

import math

__all__ = ["effective_df", "information_criteria"]

_AICC_RATIO = 40.0


def effective_df(p_rates: int, n_knots: int, knot_cost: float = 3.0) -> float:
    """Effective degrees of freedom of a model with searched knots.

    ``df = p_rates + (knot_cost - 1) * n_knots``: each knot is charged
    ``knot_cost`` in total, its coefficient (already in ``p_rates``) plus
    ``knot_cost - 1`` for its location and the search.
    For one spline of degree ``d`` with ``K`` knots this is ``d + 1 + c K``.
    See the module docstring for the guidance on ``knot_cost`` (3.0 for spaced
    free knots; 3.5 to 4.0 unconstrained or coalescing; 2.0 to 2.5 coarse
    grid) and the Friedman (1991) reference.

    Parameters
    ----------
    p_rates : int
        Free rate coefficients after fixed and tied parameters are removed
        (``BirthDeathModel.n_free`` minus the knot locations).
    n_knots : int
        Number of free knots searched (0 for fixed epochs: no knot charge).
    knot_cost : float
        Total charge per knot.

    Returns
    -------
    float
        The effective degrees of freedom.

    """
    return float(p_rates + (knot_cost - 1.0) * n_knots)


def information_criteria(log_lik: float, n: int, df: float) -> dict[str, float | bool]:
    """AIC, AICc and BIC from a log-likelihood, sample size and ``df``.

    The likelihood form is used: ``-2 * log_lik`` replaces ``n ln(RSS/n)``.
    ``aic = -2 ll + 2 df``; ``bic = -2 ll + ln(n) df``;
    ``aicc = aic + 2 df (df + 1) / (n - df - 1)``, which is ``inf`` when
    ``n - df - 1 <= 0`` (a Gaussian linear-regression approximation here).
    ``aicc_recommended`` is ``n / df < 40``.

    Parameters
    ----------
    log_lik : float
        Maximised log-likelihood.
    n : int
        Sample size.
    df : float
        Degrees of freedom, usually from :func:`effective_df`.

    Returns
    -------
    dict
        Keys ``aic``, ``aicc``, ``bic`` (floats) and ``aicc_recommended``
        (bool).

    """
    aic = -2.0 * log_lik + 2.0 * df
    room = n - df - 1.0
    aicc = aic + 2.0 * df * (df + 1.0) / room if room > 0 else math.inf
    return {
        "aic": float(aic),
        "aicc": float(aicc),
        "bic": float(-2.0 * log_lik + math.log(n) * df),
        "aicc_recommended": bool(df > 0 and n / df < _AICC_RATIO),
    }
