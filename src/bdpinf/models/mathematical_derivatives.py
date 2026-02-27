"""Derivative functions for birth-and-death process calculations.

This module contains analytical derivatives used by the mathematical functions.
Currently it is only used for validating the autograd results from PyTorch.
"""

import torch

from inf_bp_texts.utils import ensure_1d_tensors


def calculate_populations_log_lik_derivatives(
    birth_rate: torch.Tensor | float,
    death_rate: torch.Tensor | float,
    time: torch.Tensor | float,
    ns: torch.Tensor | int,
) -> torch.Tensor:
    """Calculate derivatives of log population probs with respect to lambda and mu."""
    birth_rate_val, death_rate_val, time_t, ns = ensure_1d_tensors(
        birth_rate,
        death_rate,
        time,
        ns,
    )

    d_birth_rate = torch.zeros_like(ns, dtype=torch.float64)
    d_death_rate = torch.zeros_like(ns, dtype=torch.float64)

    zero_mask = ns == 0
    non_zero_mask = torch.logical_not(zero_mask)

    is_critical = torch.isclose(birth_rate_val, death_rate_val, rtol=1e-14).item()

    if is_critical:
        if torch.any(zero_mask):
            d_birth_rate[zero_mask] = 1 / (
                birth_rate_val * (1 + birth_rate_val * time_t)
            )
            d_death_rate[zero_mask] = d_birth_rate[zero_mask]

        if torch.any(non_zero_mask):
            ns_nonzero = ns[non_zero_mask]
            numerator = ns_nonzero - 1 - 2 * birth_rate_val * time_t
            denominator = birth_rate_val * (1 + birth_rate_val * time_t)
            d_birth_rate[non_zero_mask] = numerator / denominator
            d_death_rate[non_zero_mask] = d_birth_rate[non_zero_mask]

        return torch.stack([torch.sum(d_birth_rate), torch.sum(d_death_rate)])

    growth_rate = birth_rate_val - death_rate_val
    exp_rT = torch.exp(growth_rate * time_t)

    if torch.any(zero_mask):
        alpha_t = (
            death_rate_val * (exp_rT - 1) / (birth_rate_val * exp_rT - death_rate_val)
        )

        d_alpha_d_birth_rate = (
            death_rate_val
            * exp_rT
            * (growth_rate * time_t - exp_rT + 1)
            / (birth_rate_val * exp_rT - death_rate_val) ** 2
        )

        d_alpha_d_death_rate = (
            exp_rT
            * (birth_rate_val * (exp_rT - 1) - death_rate_val * time_t * growth_rate)
            / (birth_rate_val * exp_rT - death_rate_val) ** 2
        )

        d_birth_rate[zero_mask] = d_alpha_d_birth_rate / alpha_t
        d_death_rate[zero_mask] = d_alpha_d_death_rate / alpha_t

    if torch.any(non_zero_mask):
        ns_nonzero = ns[non_zero_mask]

        alpha_t = (
            death_rate_val * (exp_rT - 1) / (birth_rate_val * exp_rT - death_rate_val)
        )
        beta_t = (birth_rate_val / death_rate_val) * alpha_t

        d_alpha_d_birth_rate = (
            death_rate_val
            * exp_rT
            * (growth_rate * time_t - exp_rT + 1)
            / (birth_rate_val * exp_rT - death_rate_val) ** 2
        )
        d_alpha_d_death_rate = (
            exp_rT
            * (birth_rate_val * (exp_rT - 1) - death_rate_val * time_t * growth_rate)
            / (birth_rate_val * exp_rT - death_rate_val) ** 2
        )

        d_beta_d_birth_rate = (alpha_t / death_rate_val) + (
            birth_rate_val / death_rate_val
        ) * d_alpha_d_birth_rate
        d_beta_d_death_rate = (
            -(birth_rate_val / death_rate_val**2) * alpha_t
            + (birth_rate_val / death_rate_val) * d_alpha_d_death_rate
        )

        d_birth_rate_nonzero = (
            -d_alpha_d_birth_rate / (1 - alpha_t)
            - d_beta_d_birth_rate / (1 - beta_t)
            + (ns_nonzero - 1) * d_beta_d_birth_rate / beta_t
        )

        d_death_rate_nonzero = (
            -d_alpha_d_death_rate / (1 - alpha_t)
            - d_beta_d_death_rate / (1 - beta_t)
            + (ns_nonzero - 1) * d_beta_d_death_rate / beta_t
        )

        d_birth_rate[non_zero_mask] = d_birth_rate_nonzero
        d_death_rate[non_zero_mask] = d_death_rate_nonzero

    return torch.stack([torch.sum(d_birth_rate), torch.sum(d_death_rate)])


def calculate_non_spine_ages_log_lik_derivatives(
    birth_rate: torch.Tensor | float,
    death_rate: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    T: torch.Tensor | float,
) -> torch.Tensor:
    """Calculate derivatives of log age-density for non-spine manuscripts.

    Uses only PyTorch operations to preserve autograd support.
    """
    birth_rate_val, death_rate_val, ages_t_t, T_t = ensure_1d_tensors(
        birth_rate,
        death_rate,
        ages_t,
        T,
    )

    is_critical = torch.isclose(birth_rate_val, death_rate_val, rtol=1e-14)

    if is_critical:
        d_death_rate = d_birth_rate = (T_t - ages_t_t) / (
            1 + birth_rate_val * (T_t - ages_t_t)
        ) - ages_t_t
    else:
        growth_rate = birth_rate_val - death_rate_val

        exp_birth_rate_t = torch.exp(-birth_rate_val * ages_t_t)
        exp_death_rate_t = torch.exp(-death_rate_val * ages_t_t)
        exp_growth_T = torch.exp(-growth_rate * T_t)

        N = (
            birth_rate_val * exp_birth_rate_t
            - death_rate_val * exp_growth_T * exp_death_rate_t
        )
        D = 1 - exp_growth_T

        dN_dlambda = (
            exp_birth_rate_t * (1 - birth_rate_val * ages_t_t)
            + death_rate_val * T_t * exp_growth_T * exp_death_rate_t
        )
        dD_dlambda = T_t * exp_growth_T
        d_birth_rate = dN_dlambda / N - dD_dlambda / D

        dN_dmu = (
            -exp_growth_T * exp_death_rate_t * (1 + death_rate_val * (T_t - ages_t_t))
        )
        dD_dmu = -T_t * exp_growth_T
        d_death_rate = dN_dmu / N - dD_dmu / D

    return torch.stack((torch.sum(d_birth_rate), torch.sum(d_death_rate)))


def calculate_spine_ages_log_lik_derivatives(
    birth_rate: torch.Tensor | float,
    death_rate: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    T: torch.Tensor | float,
) -> torch.Tensor:
    """Calculate derivatives of log age-density for spine manuscripts.

    Uses only PyTorch operations to preserve autograd support.
    """
    birth_rate_val, death_rate_val, ages_t_t, T_t = ensure_1d_tensors(
        birth_rate,
        death_rate,
        ages_t,
        T,
    )

    at_boundary = ages_t_t == T_t
    not_at_boundary = torch.logical_not(at_boundary)
    d_birth_rate = torch.zeros_like(ages_t_t, dtype=torch.float64)
    d_death_rate = torch.zeros_like(ages_t_t, dtype=torch.float64)

    is_critical = torch.isclose(birth_rate_val, death_rate_val, rtol=1e-14)

    if is_critical:
        if torch.any(at_boundary):
            d_death_rate[at_boundary] = d_birth_rate[at_boundary] = (
                2 / birth_rate_val - T_t
            )

        if torch.any(not_at_boundary):
            ages_interior = ages_t_t[not_at_boundary]
            d_death_rate[not_at_boundary] = d_birth_rate[not_at_boundary] = (
                2 / birth_rate_val - ages_interior
            )
    else:
        if torch.any(at_boundary):
            lambda_numerator = death_rate_val * (
                torch.exp(death_rate_val * T_t)
                * (T_t * (birth_rate_val - death_rate_val) + 1)
                - torch.exp(birth_rate_val * T_t)
            )
            denominator = (birth_rate_val - death_rate_val) * (
                birth_rate_val * torch.exp(birth_rate_val * T_t)
                - death_rate_val * torch.exp(death_rate_val * T_t)
            )
            d_birth_rate[at_boundary] = lambda_numerator / denominator
            mu_numerator = -birth_rate_val * (
                torch.exp(birth_rate_val * T_t)
                * (T_t * (birth_rate_val - death_rate_val) - 1)
                + torch.exp(death_rate_val * T_t)
            )
            d_death_rate[at_boundary] = mu_numerator / denominator
        if torch.any(not_at_boundary):
            B = torch.exp(-death_rate_val * T_t) - torch.exp(-birth_rate_val * T_t)

            d_birth_rate[not_at_boundary] = (
                1 / birth_rate_val
                + T_t * torch.exp(-birth_rate_val * T_t) / B
                - 1 / (birth_rate_val - death_rate_val)
            )

            d_death_rate[not_at_boundary] = (
                1 / death_rate_val
                - T_t * torch.exp(-death_rate_val * T_t) / B
                + 1 / (birth_rate_val - death_rate_val)
            )

    return torch.stack((torch.sum(d_birth_rate), torch.sum(d_death_rate)))


def calculate_ages_log_lik_derivatives(
    birth_rate: torch.Tensor | float,
    death_rate: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    T: torch.Tensor | float,
    *,
    are_spines: bool = False,
) -> torch.Tensor:
    """Compute derivatives of log age-density with respect to lambda and mu."""
    if are_spines:
        return calculate_spine_ages_log_lik_derivatives(
            birth_rate,
            death_rate,
            ages_t,
            T,
        )
    return calculate_non_spine_ages_log_lik_derivatives(
        birth_rate,
        death_rate,
        ages_t,
        T,
    )
