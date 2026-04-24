"""Uncertainty estimation helpers for birth-death process MLEs."""

from collections.abc import Callable
from typing import Any, cast

import torch

from .likelihood import negative_log_likelihood


def _invert_observed_fisher(
    observed_fisher: torch.Tensor,
) -> torch.Tensor:
    """Invert the observed Fisher information with simple fallbacks."""
    if observed_fisher.numel() == 0:
        return observed_fisher.clone()

    try:
        torch.linalg.cond(observed_fisher)
    except RuntimeError:
        pass

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
) -> torch.Tensor:
    """Estimate covariance of fitted rates using the Hessian in free log space."""
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

    with torch.no_grad():
        theta_hat_detached = theta_hat.detach()
        full_log_params_hat = full_log_params_fn(theta_hat_detached)
        full_rates_hat = torch.exp(full_log_params_hat)

    jacobian_fn = cast("Any", torch.autograd.functional.jacobian)
    jacobian_full_log = jacobian_fn(full_log_params_fn, theta_hat_detached)

    jacobian_rate_from_log = torch.diag(full_rates_hat)
    jacobian_full_rate = jacobian_rate_from_log @ jacobian_full_log
    covariance_rates_full = (
        jacobian_full_rate @ covariance_log_free @ jacobian_full_rate.T
    )
    covariance_rates_full = 0.5 * (
        covariance_rates_full + covariance_rates_full.transpose(-1, -2)
    )

    return cast("torch.Tensor", covariance_rates_full)
