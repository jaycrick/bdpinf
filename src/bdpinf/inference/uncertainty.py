"""Uncertainty estimation helpers for birth-death process MLEs."""

import math
from collections.abc import Callable
from contextlib import suppress
from typing import Any, cast

import torch

from .likelihood import negative_log_likelihood


def _invert_observed_fisher(
    observed_fisher: torch.Tensor,
) -> torch.Tensor:
    """Invert the observed Fisher information with simple fallbacks."""
    if observed_fisher.numel() == 0:
        return observed_fisher.clone()

    with suppress(RuntimeError):
        torch.linalg.cond(observed_fisher)

    try:
        return cast("torch.Tensor", torch.linalg.inv(observed_fisher))  # pyright: ignore[reportAnyReturnType]
    except RuntimeError:
        pass

    identity = torch.eye(
        observed_fisher.shape[0],
        dtype=observed_fisher.dtype,
        device=observed_fisher.device,
    )
    for jitter in (1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return cast(
                "torch.Tensor",
                torch.linalg.inv(observed_fisher + jitter * identity),
            )  # pyright: ignore[reportAnyReturnType]
        except RuntimeError:
            continue

    return cast("torch.Tensor", torch.linalg.pinv(observed_fisher))


def estimate_mle_covariance(
    free_log_params: torch.Tensor,
    n_epochs: int,
    full_log_params_fn: Callable[[torch.Tensor], torch.Tensor],
    non_spine_ages: torch.Tensor,
    times_tensor: torch.Tensor,
    *,
    spine_ages_tensor: torch.Tensor | None,
    population_sizes_tensor: torch.Tensor,
    initial_population: int,
    unbias: bool = True,
) -> torch.Tensor:
    """Estimate covariance of fitted log-parameters using the Hessian."""
    theta_hat = cast("Any", free_log_params.detach().clone().to(dtype=torch.float64))

    def objective(theta_free: torch.Tensor) -> torch.Tensor:
        full_log_params = full_log_params_fn(theta_free)
        return negative_log_likelihood(
            birth_rates=full_log_params[:n_epochs],
            death_rates=full_log_params[n_epochs:],
            ages=non_spine_ages,
            times=times_tensor,
            spine_ages=spine_ages_tensor,
            ns=population_sizes_tensor,
            initial_population=initial_population,
            unbias=unbias,
            is_ordered=False,
            log_scale_params=True,
        )

    hessian_fn = cast("Any", torch.autograd.functional.hessian)
    hessian_nll = hessian_fn(objective, theta_hat)
    hessian_nll = 0.5 * (hessian_nll + hessian_nll.transpose(-1, -2))

    covariance_log_free = _invert_observed_fisher(hessian_nll)
    covariance_log_free = 0.5 * (
        covariance_log_free + covariance_log_free.transpose(-1, -2)
    )

    jacobian_fn = cast("Any", torch.autograd.functional.jacobian)
    theta_hat_detached = theta_hat.detach()
    jacobian_full_log = jacobian_fn(full_log_params_fn, theta_hat_detached)

    covariance_log_full = jacobian_full_log @ covariance_log_free @ jacobian_full_log.T
    covariance_log_full = 0.5 * (
        covariance_log_full + covariance_log_full.transpose(-1, -2)
    )

    return cast("torch.Tensor", covariance_log_full)


def confidence_intervals_from_log_covariance(
    log_parameter_estimates: torch.Tensor,
    covariance_log_parameters: torch.Tensor,
    *,
    confidence_level: float = 0.95,
) -> torch.Tensor:
    """Build Wald confidence intervals on the original scale.

    Uses log-scale uncertainty and exponentiates the resulting interval bounds
    back to the original scale.
    """
    if log_parameter_estimates.numel() == 0:
        return log_parameter_estimates.new_empty((0, 2))

    standard_errors = torch.sqrt(
        torch.clamp(torch.diag(covariance_log_parameters), min=0.0),
    )
    z_value = math.sqrt(2.0) * torch.erfinv(
        torch.tensor(
            confidence_level,
            dtype=log_parameter_estimates.dtype,
            device=log_parameter_estimates.device,
        ),
    )

    lower = torch.exp(log_parameter_estimates - z_value * standard_errors)
    upper = torch.exp(log_parameter_estimates + z_value * standard_errors)
    return torch.stack((lower, upper), dim=-1)
