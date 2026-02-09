"""
Derivative functions for birth-and-death process calculations.

This module contains analytical derivatives used by the mathematical functions.
"""

import torch

from ..utils import ensure_1d_tensors


def calculate_populations_log_lik_derivatives(
    lambda_param: torch.Tensor | float,
    mu_param: torch.Tensor | float,
    time: torch.Tensor | float,
    ns: torch.Tensor | int,
) -> torch.Tensor:
    """
    Calculate derivatives of log population probabilities with respect to lambda and mu.

    Uses only PyTorch operations to preserve autograd support.
    """
    lambda_val, mu_val, time_t, ns = ensure_1d_tensors(lambda_param, mu_param, time, ns)

    d_lambda = torch.zeros_like(ns, dtype=torch.float64)
    d_mu = torch.zeros_like(ns, dtype=torch.float64)

    zero_mask = ns == 0
    non_zero_mask = ~zero_mask

    is_critical = torch.isclose(lambda_val, mu_val, rtol=1e-14).item()

    if is_critical:
        if torch.any(zero_mask):
            d_lambda[zero_mask] = 1 / (lambda_val * (1 + lambda_val * time_t))
            d_mu[zero_mask] = d_lambda[zero_mask]

        if torch.any(non_zero_mask):
            ns_nonzero = ns[non_zero_mask]
            numerator = ns_nonzero - 1 - 2 * lambda_val * time_t
            denominator = lambda_val * (1 + lambda_val * time_t)
            d_lambda[non_zero_mask] = numerator / denominator
            d_mu[non_zero_mask] = d_lambda[non_zero_mask]

        return torch.stack([torch.sum(d_lambda), torch.sum(d_mu)])

    growth_rate = lambda_val - mu_val
    exp_rT = torch.exp(growth_rate * time_t)

    if torch.any(zero_mask):
        alpha_t = mu_val * (exp_rT - 1) / (lambda_val * exp_rT - mu_val)

        d_alpha_d_lambda = (
            mu_val
            * exp_rT
            * (growth_rate * time_t - exp_rT + 1)
            / (lambda_val * exp_rT - mu_val) ** 2
        )

        d_alpha_d_mu = (
            exp_rT
            * (lambda_val * (exp_rT - 1) - mu_val * time_t * growth_rate)
            / (lambda_val * exp_rT - mu_val) ** 2
        )

        d_lambda[zero_mask] = d_alpha_d_lambda / alpha_t
        d_mu[zero_mask] = d_alpha_d_mu / alpha_t

    if torch.any(non_zero_mask):
        ns_nonzero = ns[non_zero_mask]

        alpha_t = mu_val * (exp_rT - 1) / (lambda_val * exp_rT - mu_val)
        beta_t = (lambda_val / mu_val) * alpha_t

        d_alpha_d_lambda = (
            mu_val
            * exp_rT
            * (growth_rate * time_t - exp_rT + 1)
            / (lambda_val * exp_rT - mu_val) ** 2
        )
        d_alpha_d_mu = (
            exp_rT
            * (lambda_val * (exp_rT - 1) - mu_val * time_t * growth_rate)
            / (lambda_val * exp_rT - mu_val) ** 2
        )

        d_beta_d_lambda = (alpha_t / mu_val) + (lambda_val / mu_val) * d_alpha_d_lambda
        d_beta_d_mu = (
            -(lambda_val / mu_val**2) * alpha_t + (lambda_val / mu_val) * d_alpha_d_mu
        )

        d_lambda_nonzero = (
            -d_alpha_d_lambda / (1 - alpha_t)
            - d_beta_d_lambda / (1 - beta_t)
            + (ns_nonzero - 1) * d_beta_d_lambda / beta_t
        )

        d_mu_nonzero = (
            -d_alpha_d_mu / (1 - alpha_t)
            - d_beta_d_mu / (1 - beta_t)
            + (ns_nonzero - 1) * d_beta_d_mu / beta_t
        )

        d_lambda[non_zero_mask] = d_lambda_nonzero
        d_mu[non_zero_mask] = d_mu_nonzero

    return torch.stack([torch.sum(d_lambda), torch.sum(d_mu)])


def calculate_non_spine_ages_log_lik_derivatives(
    lambda_param: torch.Tensor | float,
    mu_param: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    T: torch.Tensor | float,
) -> torch.Tensor:
    """
    Calculate derivatives of log age-density for non-spine manuscripts.

    Uses only PyTorch operations to preserve autograd support.
    """
    lambda_val, mu_val, ages_t_t, T_t = ensure_1d_tensors(
        lambda_param, mu_param, ages_t, T
    )

    is_critical = torch.isclose(lambda_val, mu_val, rtol=1e-14)

    if is_critical:
        d_mu = d_lambda = (T_t - ages_t_t) / (
            1 + lambda_val * (T_t - ages_t_t)
        ) - ages_t_t
    else:
        growth_rate = lambda_val - mu_val

        exp_lambda_t = torch.exp(-lambda_val * ages_t_t)
        exp_mu_t = torch.exp(-mu_val * ages_t_t)
        exp_growth_T = torch.exp(-growth_rate * T_t)

        N = lambda_val * exp_lambda_t - mu_val * exp_growth_T * exp_mu_t
        D = 1 - exp_growth_T

        dN_dlambda = (
            exp_lambda_t * (1 - lambda_val * ages_t_t)
            + mu_val * T_t * exp_growth_T * exp_mu_t
        )
        dD_dlambda = T_t * exp_growth_T
        d_lambda = dN_dlambda / N - dD_dlambda / D

        dN_dmu = -exp_growth_T * exp_mu_t * (1 + mu_val * (T_t - ages_t_t))
        dD_dmu = -T_t * exp_growth_T
        d_mu = dN_dmu / N - dD_dmu / D

    return torch.stack((torch.sum(d_lambda), torch.sum(d_mu)))


def calculate_spine_ages_log_lik_derivatives(
    lambda_param: torch.Tensor | float,
    mu_param: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    T: torch.Tensor | float,
) -> torch.Tensor:
    """
    Calculate derivatives of log age-density for spine manuscripts.

    Uses only PyTorch operations to preserve autograd support.
    """
    lambda_val, mu_val, ages_t_t, T_t = ensure_1d_tensors(
        lambda_param, mu_param, ages_t, T
    )

    at_boundary = ages_t_t == T_t
    not_at_boundary = torch.logical_not(at_boundary)
    d_lambda = torch.zeros_like(ages_t_t, dtype=torch.float64)
    d_mu = torch.zeros_like(ages_t_t, dtype=torch.float64)

    is_critical = torch.isclose(lambda_val, mu_val, rtol=1e-14)

    if is_critical:
        if torch.any(at_boundary):
            d_mu[at_boundary] = d_lambda[at_boundary] = 2 / lambda_val - T_t

        if torch.any(not_at_boundary):
            ages_interior = ages_t_t[not_at_boundary]
            d_mu[not_at_boundary] = d_lambda[not_at_boundary] = (
                2 / lambda_val - ages_interior
            )
    else:
        if torch.any(at_boundary):
            lambda_numerator = mu_val * (
                torch.exp(mu_val * T_t) * (T_t * (lambda_val - mu_val) + 1)
                - torch.exp(lambda_val * T_t)
            )
            denominator = (lambda_val - mu_val) * (
                lambda_val * torch.exp(lambda_val * T_t)
                - mu_val * torch.exp(mu_val * T_t)
            )
            d_lambda[at_boundary] = lambda_numerator / denominator
            mu_numerator = -lambda_val * (
                torch.exp(lambda_val * T_t) * (T_t * (lambda_val - mu_val) - 1)
                + torch.exp(mu_val * T_t)
            )
            d_mu[at_boundary] = mu_numerator / denominator
        if torch.any(not_at_boundary):
            B = torch.exp(-mu_val * T_t) - torch.exp(-lambda_val * T_t)

            d_lambda[not_at_boundary] = (
                1 / lambda_val
                + T_t * torch.exp(-lambda_val * T_t) / B
                - 1 / (lambda_val - mu_val)
            )

            d_mu[not_at_boundary] = (
                1 / mu_val
                - T_t * torch.exp(-mu_val * T_t) / B
                + 1 / (lambda_val - mu_val)
            )

    return torch.stack((torch.sum(d_lambda), torch.sum(d_mu)))


def calculate_ages_log_lik_derivatives(
    lambda_param: torch.Tensor | float,
    mu_param: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    T: torch.Tensor | float,
    are_spines: bool = False,
) -> torch.Tensor:
    """
    Compute derivatives of log age-density with respect to lambda and mu.
    """
    if are_spines:
        return calculate_spine_ages_log_lik_derivatives(
            lambda_param, mu_param, ages_t, T
        )
    return calculate_non_spine_ages_log_lik_derivatives(
        lambda_param, mu_param, ages_t, T
    )
