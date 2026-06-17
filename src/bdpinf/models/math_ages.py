"""Spine and non-spine manuscript age log-likelihood functions.

Imports the extinction/lifetime base layer from math_extinction.
"""

from typing import cast

import torch

from inf_bp_texts.models.math_extinction import (
    calculate_log_lik_initial_lifetimes_given_extinction,
    calculate_log_lik_initial_lifetimes_given_non_extinction,
)
from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import require_tensors


def _reverse_time(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Reverse time and swap rates for multi-epoch likelihood transformation.

    **Mathematical Principle:**

    The time-reversal trick exploits a fundamental duality: a non-spine age likelihood
    under forward-time evolution (ages observed at final time $T$) can be recomputed as
    an extinction-conditioned lifetime likelihood under backward-time evolution.

    **Transformation:**
    - Epoch times: $\tilde{T}_k = T_K - T_{K-k}$ (flip and shift)
    - Rates: $(\tilde{\lambda}_k, \tilde{\mu}_k) = (\mu_{K-k+1}, \lambda_{K-k+1})$
    (swap birth ↔ death)

    Parameters
    ----------
    birth_rates : torch.Tensor
        Birth rates $\lambda_k$ for each epoch (shape: (n_epochs,),
        dtype: torch.float64).
    death_rates : torch.Tensor
        Death rates $\mu_k$ for each epoch (shape: (n_epochs,), dtype: torch.float64).
    times : torch.Tensor
        Epoch end times $T_1, T_2, \ldots, T_K$ strictly increasing
        (shape: (n_epochs,), dtype: torch.float64).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - **inverted_times**: Reversed and shifted epoch boundaries
          $(\tilde{T}_1, \tilde{T}_2, \ldots, \tilde{T}_K)$
        - **reflected_birth_rates**: Swapped death rates (new birth rates)
        - **reflected_death_rates**: Swapped birth rates (new death rates)

    Notes
    -----
    Used internally by `calculate_non_spine_ages_log_lik` and
    `calculate_spine_ages_log_lik` to handle multi-epoch cases.
    Enables efficient dispatch to extinction-conditioned likelihood functions.

    """
    end_time = times[-1]
    inverted_times = torch.flip(
        end_time
        - torch.cat([torch.zeros(1, dtype=torch.float64, device=times.device), times])[
            :-1
        ],
        dims=[0],
    )
    reflected_birth_rates = torch.flip(death_rates, dims=[0])
    reflected_death_rates = torch.flip(birth_rates, dims=[0])
    return inverted_times, reflected_birth_rates, reflected_death_rates


def calculate_non_spine_ages_log_lik(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    return_array: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    r"""Calculate log-likelihood of the ages of the "non-spine" individual(s).

    In the constant rate case, the density for the age *a* is:
    $$
        \frac{\lambda * \exp(-\lambda a) - \mu * \exp(-(\lambda - \mu)T) * \exp(-\mu a)
        }{1 - \exp(-(\lambda - \mu)T)}
    $$

    given $\lambda \neq \mu$, otherwise the "critical" case is:
    $$
        \frac{\exp(-\lambda a) * (1 + \lambda(T-a))}{T}
    $$

    Parameters
    ----------
    birth_rates : torch.Tensor | float
        (Non-negative) birth rate(s) $\lambda$.
    death_rates : torch.Tensor | float
        (Non-negative) death rate(s) $\mu$.
    ages : torch.Tensor | float
        Age(s) $a$. All ages must satisfy $0 \leq a \leq T$.
    times : torch.Tensor | float
        Observation time(s) $T$. For multi-epoch case, the end of each epoch.
    return_array : bool, optional
        If True, return the per-age log-likelihood contributions instead of
        the summed value. Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood across ages.
        dtype: torch.float64.

    References
    ----------
    For the constant rate case see equation (3.6) from:
        Kück, F., & Schuhmacher, D. (2018).
        On the age of a randomly picked individual in a linear birth-and-death process.
        *Journal of Applied Probability, 55*(1), 82--93. doi:10.1017/jpr.2018.7

    """
    if not _skip_normalization:
        birth_rates, death_rates, times, ages = normalize_and_validate_inputs(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ages=ages,
        )
    else:
        require_tensors(
            birth_rates=birth_rates, death_rates=death_rates, times=times, ages=ages
        )
        birth_rates = cast(torch.Tensor, birth_rates)
        death_rates = cast(torch.Tensor, death_rates)
        times = cast(torch.Tensor, times)
        ages = cast(torch.Tensor, ages)
    end_time = times[-1]

    if len(times) > 1:
        inverted_times, reflected_birth_rates, reflected_death_rates = _reverse_time(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
        )
        return calculate_log_lik_initial_lifetimes_given_extinction(
            ages,
            reflected_birth_rates,
            reflected_death_rates,
            inverted_times,
            as_array=return_array,
            _skip_normalization=True,
        )

    birth_rate_val = birth_rates[0]
    death_rate_val = death_rates[0]
    _dev = ages.device
    _ninf = torch.tensor(float("-inf"), dtype=torch.float64, device=_dev)

    if birth_rate_val == 0 or death_rate_val <= 0:
        return _ninf

    is_critical = torch.isclose(birth_rate_val, death_rate_val, rtol=1e-12).item()

    # Check for numerical stability in parameter space
    if is_critical:  # λ ≈ μ
        # f(t) = [1 + λ(T-t)] / T * e^{-λt}t
        # average ensures both birth_rate_val and death_rate_val in the comp graph
        avg_rate = (birth_rate_val + death_rate_val) / 2
        log_numerator = -avg_rate * ages + torch.log1p(avg_rate * (end_time - ages))
        log_denominator = torch.log(end_time)

        # Check for numerical validity
        if not (
            torch.all(torch.isfinite(log_numerator)) and torch.isfinite(log_denominator)
        ):
            return _ninf

    else:  # λ ≠ μ
        growth_rate = birth_rate_val - death_rate_val
        prod_growth_rate_end_time = growth_rate * end_time

        numerator_term1 = birth_rate_val * torch.exp(-birth_rate_val * ages)
        numerator_term2 = (
            death_rate_val
            * torch.exp(-prod_growth_rate_end_time)
            * torch.exp(-death_rate_val * ages)
        )

        numerator = numerator_term1 - numerator_term2
        denominator = 1 - torch.exp(-prod_growth_rate_end_time)

        if not torch.all(torch.sign(numerator) == torch.sign(denominator)):
            return _ninf

        log_numerator = torch.log(torch.abs(numerator))
        log_denominator = torch.log(torch.abs(denominator))

        # Check for numerical validity
        if not (
            torch.all(torch.isfinite(log_numerator)) and torch.isfinite(log_denominator)
        ):
            return (
                torch.full_like(ages, float("-inf"), dtype=torch.float64)
                if return_array
                else _ninf
            )

    # Calculate final log density contributions
    log_density = log_numerator - log_denominator

    # Final numerical check
    if not torch.all(torch.isfinite(log_density)):
        return (
            torch.full_like(ages, float("-inf"), dtype=torch.float64)
            if return_array
            else _ninf
        )

    return log_density if return_array else torch.sum(log_density)


def calculate_spine_ages_log_lik(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    return_array: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    r"""Calculate log-likelihood of the age(s) of the special "spine" individual(s).

    In the constant rate case, the density for the age $a$ is:

    $$
        \frac{\lambda\mu(\exp(-\lambda a) - \exp(-\mu a))}{\lambda - \mu} \quad
        \text{for } a < T,
    $$
    and
    $$
        \frac{\lambda \exp(-\mu T) - \mu \exp(-\lambda T)}{\lambda - \mu} \quad
        \text{for } a = T.
    $$

    given $\lambda \neq \mu$, otherwise the "critical" case is:
    $$
        \lambda^2 * a * \exp(-\lambda a) \quad \text{for } a < T,
    $$
    and
    $$
        (1 + \lambda T) * \exp(-\lambda T) \quad \text{for } a = T.
    $$

    Parameters
    ----------
    birth_rates : torch.Tensor | float
        (Non-negative) birth rate(s) $\lambda$.
    death_rates : torch.Tensor | float
        (Non-negative) death rate(s) $\mu$.
    ages : torch.Tensor | float
        Age(s) $a$. All ages must satisfy $0 \leq a \leq T$.
    times : torch.Tensor | float
        Observation time(s) $T$. For multi-epoch case, the end of each epoch.
    return_array : bool, optional
        If True, return the per-age log-likelihood contributions instead of
        the summed value. Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood across ages.
        dtype: torch.float64.

    Notes
    -----
    For the constant rate case see equation (3.8) from
        Kück, F., & Schuhmacher, D. (2018).
        On the age of a randomly picked individual in a linear birth-and-death process.
        *Journal of Applied Probability, 55*(1), 82--93. doi:10.1017/jpr.2018.7

    """
    if not _skip_normalization:
        birth_rates, death_rates, times, ages = normalize_and_validate_inputs(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ages=ages,
        )
    else:
        require_tensors(
            birth_rates=birth_rates, death_rates=death_rates, times=times, ages=ages
        )
        birth_rates = cast(torch.Tensor, birth_rates)
        death_rates = cast(torch.Tensor, death_rates)
        times = cast(torch.Tensor, times)
        ages = cast(torch.Tensor, ages)
    end_time = times[-1]

    if len(times) > 1:
        inverted_times, reflected_birth_rates, reflected_death_rates = _reverse_time(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
        )
        return calculate_log_lik_initial_lifetimes_given_non_extinction(
            lifetimes=ages,
            birth_rates=reflected_birth_rates,
            death_rates=reflected_death_rates,
            times=inverted_times,
            as_array=return_array,
            _skip_normalization=True,
        )

    birth_rate = birth_rates[0]
    death_rate = death_rates[0]
    _dev = ages.device
    _ninf = torch.tensor(float("-inf"), dtype=torch.float64, device=_dev)

    if birth_rate < 0 or death_rate < 0:
        return _ninf

    is_critical = torch.isclose(birth_rate, death_rate, rtol=1e-12).item()
    at_boundary = ages == end_time
    not_at_boundary = torch.logical_not(at_boundary)

    if is_critical:  # λ ≈ μ
        # f(t) ≈ [1 + λ(T-t)]/T * e^{-λt}
        # Using average ensures both birth_rate and death_rate are in the comp graph
        avg_rate = (birth_rate + death_rate) / 2
        boundary_val = torch.log1p(avg_rate * end_time) - avg_rate * end_time
        log_density = torch.where(
            at_boundary,
            boundary_val,
            2 * torch.log(avg_rate) + torch.log(ages) - avg_rate * ages,
        )
        return log_density if return_array else torch.sum(log_density)
    # General case: λ ≠ μ
    log_numerator = torch.zeros_like(ages, dtype=torch.float64)
    denom_t = birth_rate - death_rate
    sign_denominator = torch.sign(denom_t)
    log_denominator = torch.log(torch.abs(denom_t))

    if torch.any(at_boundary):
        numerator_boundary = birth_rate * torch.exp(
            -death_rate * end_time,
        ) - death_rate * torch.exp(-birth_rate * end_time)

        if not torch.all(torch.sign(numerator_boundary) == sign_denominator):
            return _ninf

        log_numerator[at_boundary] = torch.log(torch.abs(numerator_boundary))

    if torch.any(not_at_boundary):
        numerator_1 = torch.exp(-death_rate * ages) - torch.exp(-birth_rate * ages)

        if not torch.all(torch.sign(numerator_1) == sign_denominator):
            return _ninf

        log_numerator[not_at_boundary] = (
            torch.log(birth_rate)
            + torch.log(death_rate)
            + torch.log(torch.abs(numerator_1[not_at_boundary]))
        )

    # Check for numerical validity
    if not (
        torch.all(torch.isfinite(log_numerator)) and torch.isfinite(log_denominator)
    ):
        return (
            torch.full_like(ages, float("-inf"), dtype=torch.float64)
            if return_array
            else _ninf
        )

    # Calculate final log density contributions
    log_density = log_numerator - log_denominator

    # Final numerical check
    if not torch.all(torch.isfinite(log_density)):
        return (
            torch.full_like(ages, float("-inf"), dtype=torch.float64)
            if return_array
            else _ninf
        )

    return log_density if return_array else torch.sum(log_density)


def calculate_ages_log_lik(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    are_spines: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    r"""Compute log-likelihood of manuscript ages under a birth-death model.

    Dispatches to spine or non-spine likelihood calculation based on ``are_spines``.
    All operations preserve PyTorch's computational graph for automatic differentiation.

    Parameters
    ----------
    birth_rates : torch.Tensor | float
        Birth rate(s) $\lambda$.
        Scalar for single epoch, 1D tensor for piecewise-constant rates.
        All values must be positive. Will be converted to torch.float64.
    death_rates : torch.Tensor | float
        Death rate(s) $\mu$.
        Scalar for single epoch, 1D tensor for piecewise-constant rates.
        All values must be positive. Will be converted to torch.float64.
    ages : torch.Tensor | float
        Manuscript age(s) $t$. Scalar or 1D tensor.
        All ages must satisfy $0 \leq t \leq T$.
        Will be converted to torch.float64.
    times : torch.Tensor | float
        Observation time(s) $T$. Scalar for single epoch, 1D tensor of strictly
        increasing epoch end times for piecewise-constant model.
        Will be converted to torch.float64.
    are_spines : bool, optional
        If True, computes spine likelihood (discrete mass at $T$, continuous for
        $0 \leq t < T$). If False, computes non-spine likelihood. Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood across all ages.
        dtype: torch.float64. Fully differentiable via PyTorch autograd.
        Returns -inf tensor for numerically invalid inputs.

    Raises
    ------
    ValueError
        If array inputs have mismatched lengths, times are not strictly increasing,
        or ages fall outside $[0, T]$.

    Notes
    -----
    - Handles $\lambda \approx \mu$ limiting case for numerical stability
      using ``log1p/expm1``.
    - Supports both single-epoch (scalar)
    and multi-epoch (array) piecewise-constant rates.
    - Multi-epoch cases use time-reversal transformation internally.
    - All operations maintain PyTorch's computational graph for gradient computation.
    - Internally delegates to ``calculate_spine_ages_log_lik`` or
      ``calculate_non_spine_ages_log_lik`` depending on ``are_spines`` flag.

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
