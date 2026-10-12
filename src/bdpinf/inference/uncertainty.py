"""Uncertainty estimation helpers for birth-death process MLEs."""

import math
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy.special

from .likelihood import log_likelihood


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


def estimate_mle_covariance(
    free_log_params: jax.Array,
    n_epochs: int,
    full_log_params_fn: Callable[[jax.Array], jax.Array],
    non_spine_ages: jax.Array,
    times_tensor: jax.Array,
    *,
    spine_ages_tensor: jax.Array | None,
    population_sizes_tensor: jax.Array,
    initial_population: int,
    condition_on_survival: bool = True,
) -> jax.Array:
    """Estimate covariance of fitted log-parameters using the Hessian."""
    theta_hat = jnp.array(free_log_params, dtype=jnp.float64)

    def objective(theta_free: jax.Array) -> jax.Array:
        full_log_params = full_log_params_fn(theta_free)
        return -log_likelihood(
            birth_rates=full_log_params[:n_epochs],
            death_rates=full_log_params[n_epochs:],
            ages=non_spine_ages,
            times=times_tensor,
            spine_ages=spine_ages_tensor,
            ns=population_sizes_tensor,
            initial_population=initial_population,
            condition_on_survival=condition_on_survival,
            is_ordered=False,
            log_scale_params=True,
        )

    hessian_nll = jax.hessian(objective)(theta_hat)
    hessian_nll = 0.5 * (hessian_nll + hessian_nll.T)

    covariance_log_free = _invert_observed_fisher(hessian_nll)
    covariance_log_free = 0.5 * (covariance_log_free + covariance_log_free.T)

    jacobian_full_log = jax.jacobian(full_log_params_fn)(theta_hat)

    covariance_log_full = jacobian_full_log @ covariance_log_free @ jacobian_full_log.T
    covariance_log_full = 0.5 * (covariance_log_full + covariance_log_full.T)

    return covariance_log_full


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
