"""Uncertainty estimation helpers for birth-death process MLEs."""

import math
import warnings

import jax
import jax.numpy as jnp
import jax.scipy.special


def _is_positive_definite(matrix: jax.Array) -> bool:
    """Return True if ``matrix`` admits a Cholesky factorization (no NaNs)."""
    chol = jnp.linalg.cholesky(matrix)
    return not bool(jnp.any(jnp.isnan(chol)))


def _invert_observed_fisher(
    observed_fisher: jax.Array,
) -> jax.Array:
    """Invert the observed Fisher information with PD-checked jitter fallbacks.

    A non-positive-definite Hessian (e.g. at a saddle point, or when the
    optimizer stopped short of a true stationary point) can still invert
    cleanly while producing negative "variances" on the diagonal. We therefore
    require the Hessian (and each jittered variant) to be positive definite
    --- verified via Cholesky --- not merely NaN-free, before trusting its
    inverse. If no jitter level recovers positive-definiteness, fall back to
    the pseudo-inverse and warn, since the resulting covariance should not be
    treated as valid.
    """
    if observed_fisher.size == 0:
        return observed_fisher

    if _is_positive_definite(observed_fisher):
        result = jnp.linalg.inv(observed_fisher)
        if not bool(jnp.any(jnp.isnan(result))):
            return result

    identity = jnp.eye(observed_fisher.shape[0], dtype=observed_fisher.dtype)
    for jitter in (1e-10, 1e-8, 1e-6, 1e-4):
        jittered = observed_fisher + jitter * identity
        if not _is_positive_definite(jittered):
            continue
        result = jnp.linalg.inv(jittered)
        if not bool(jnp.any(jnp.isnan(result))):
            return result

    warnings.warn(
        "Observed Fisher information is not positive definite even after "
        "jitter; the fit may not be at a true stationary point. Falling back "
        "to the pseudo-inverse, but the resulting covariance/standard errors "
        "should not be treated as valid.",
        UserWarning,
        stacklevel=2,
    )
    return jnp.linalg.pinv(observed_fisher)


def covariance_from_hessian(hessian: jax.Array) -> jax.Array:
    """Covariance of the free parameters from the negative-log-likelihood Hessian.

    Inverts the observed Fisher information, with the jitter and pseudo-inverse
    fallbacks of :func:`_invert_observed_fisher`.
    """
    hessian = 0.5 * (hessian + hessian.T)
    covariance = _invert_observed_fisher(hessian)
    return 0.5 * (covariance + covariance.T)


def confidence_intervals_from_log_covariance(
    log_parameter_estimates: jax.Array,
    covariance_log_parameters: jax.Array,
    *,
    confidence_level: float = 0.95,
) -> jax.Array:
    """Build Wald confidence intervals on the original scale.

    Uses log-scale uncertainty and exponentiates the resulting interval bounds
    back to the original scale.
    """
    if log_parameter_estimates.size == 0:
        return jnp.empty((0, 2), dtype=log_parameter_estimates.dtype)

    standard_errors = jnp.sqrt(
        jnp.maximum(jnp.diag(covariance_log_parameters), 0.0),
    )
    z_value = math.sqrt(2.0) * float(
        jax.scipy.special.erfinv(jnp.array(confidence_level, dtype=jnp.float64))
    )

    lower = jnp.exp(log_parameter_estimates - z_value * standard_errors)
    upper = jnp.exp(log_parameter_estimates + z_value * standard_errors)
    return jnp.stack([lower, upper], axis=-1)
