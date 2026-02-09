"""
Mathematical functions for birth-and-death process calculations.

This module contains pure mathematical functions for calculating birth-and-death
process probabilities and densities. These functions are isolated from the
main BirthDeathProcess class for better testability and modularity.
"""

import numpy as np
import scipy
import torch
from numpy.typing import NDArray

from ..utils import (
    ensure_1d_tensors,
    validate_matching_lengths,
    validate_matching_shapes,
)


def _validate_times_non_negative(times: torch.Tensor, *, error_message: str) -> None:
    if torch.any(times < 0):
        raise ValueError(error_message)


def _validate_times_strictly_increasing(
    times: torch.Tensor,
    *,
    error_message: str,
    require_positive_start: bool = False,
) -> None:
    if require_positive_start:
        diffs = torch.diff(torch.cat([torch.tensor([0.0], dtype=torch.float64), times]))
    else:
        diffs = torch.diff(times)
    if torch.any(diffs <= 0):
        raise ValueError(error_message)


def _validate_ages_in_range(
    ages_t: torch.Tensor,
    end_time: torch.Tensor,
    *,
    error_message: str,
) -> None:
    if torch.any(ages_t < 0) or torch.any(ages_t > end_time):
        raise ValueError(error_message)


def calculate_populations_log_lik(
    lambda_params: torch.Tensor | float,
    mu_params: torch.Tensor | float,
    times: torch.Tensor | float,
    ns: torch.Tensor | int,
) -> torch.Tensor:
    """
    Calculate the log-probability (log-likelihood) of observing given population counts
    under a birth-and-death process, with support for piecewise-constant rate epochs.

    This function computes log p_n(T) for each count n in `ns` using closed-form
    expressions for the birth-and-death process probabilities. The computed log-probabilities
    are summed and returned as a single torch.Tensor. Invalid or numerically
    unstable inputs return -inf tensor.

    Uses only PyTorch operations to preserve autograd support.

    Parameters
    ----------
    lambda_params : torch.Tensor | float
        Birth rate(s) λ. Scalar for constant rate, 1D tensor for piecewise-constant rates.
    mu_params : torch.Tensor | float
        Death rate(s) μ. Scalar for constant rate, 1D tensor for piecewise-constant rates.
    times : torch.Tensor | float
        Time(s) T. Scalar for single time point, 1D tensor of epoch end times.
        Must be ≥ 0 and strictly increasing.
    ns : torch.Tensor | int
        Non-negative population count(s) for which to evaluate log-probabilities.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing sum of log p_n(T) across provided `ns`.
        dtype: torch.float64. Fully differentiable via PyTorch autograd.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Array inputs must have
    matching lengths. Returns -inf tensor for numerical issues.
    """
    # Convert scalar inputs to tensors and ensure 1D arrays
    lambda_params, mu_params, times, ns = ensure_1d_tensors(
        lambda_params, mu_params, times, ns
    )

    # Input validation
    if torch.any(ns < 0):
        raise ValueError(f"Population counts ns must be non-negative. Got ns: {ns}")
    _validate_times_non_negative(
        times, error_message=f"Times must be non-negative. Got times: {times}"
    )
    validate_matching_lengths(
        lambda_params,
        mu_params,
        times,
        names=("lambda_params", "mu_params", "times"),
    )

    if len(times) > 1:
        # Multi-epoch case
        if torch.any(lambda_params < 0) or torch.any(mu_params < 0):
            return torch.tensor(float("-inf"), dtype=torch.float64)

        _validate_times_strictly_increasing(
            times,
            error_message=(
                "Epoch times must be strictly monotonically increasing. "
                f"Got time array: {times}"
            ),
            require_positive_start=False,
        )

        alpha_t = calculate_extinction_pr(lambda_params, mu_params, times)
        beta_t = 1 - _population_geometric_parameter(lambda_params, mu_params, times)
    else:
        # Single-epoch case
        lambda_val = lambda_params[0]
        mu_val = mu_params[0]
        time_val = times[0]

        # Validate positive parameters
        if lambda_val < 0 or mu_val < 0:
            return torch.tensor(float("-inf"), dtype=torch.float64)

        if torch.isclose(lambda_val, mu_val, rtol=1e-12):
            # Limiting case: λ ≈ μ
            avg_rate = (lambda_val + mu_val) / 2
            rate_time_product = avg_rate * time_val
            alpha_t = beta_t = rate_time_product / (1 + rate_time_product)
        else:
            # General case: λ ≠ μ
            growth_rate = lambda_val - mu_val
            exponent = growth_rate * time_val
            exp_term = torch.exp(exponent)

            if not torch.isfinite(exp_term):
                return torch.tensor(float("-inf"), dtype=torch.float64)

            denominator = lambda_val * exp_term - mu_val

            # Guard against division by zero
            if mu_val == 0.0:
                alpha_t = torch.tensor(0.0, dtype=torch.float64)
                beta_t = lambda_val * (exp_term - 1) / denominator
            else:
                alpha_t = mu_val * (exp_term - 1) / denominator
                beta_t = (lambda_val / mu_val) * alpha_t

    # Validate α_t and β_t are in valid range
    if not (0 <= alpha_t <= 1):
        raise ValueError(f"Computed α_t out of bounds [0, 1]: α_t={alpha_t}")
    if not (0 <= beta_t <= 1):
        raise ValueError(f"Computed β_t out of bounds [0, 1]: β_t={beta_t}")
    if beta_t == 0.0 and torch.any(ns > 1):
        return torch.tensor(float("-inf"), dtype=torch.float64)

    # Initialize log probability array
    log_pr = torch.zeros_like(ns, dtype=torch.float64)

    # Handle n == 0 case
    zero_mask = ns == 0
    if torch.any(zero_mask):
        log_pr[zero_mask] = torch.log(alpha_t)

    # Handle n > 0 case
    nonzero_mask = ~zero_mask
    if torch.any(nonzero_mask):
        log_term1 = torch.log1p(-alpha_t)
        log_term2 = torch.log1p(-beta_t)
        log_term3 = torch.where(
            ns == 1,
            torch.tensor(0.0, dtype=torch.float64),
            (ns - 1) * torch.log(beta_t),
        )

        # Check all terms are finite
        if not (
            torch.isfinite(log_term1)
            and torch.isfinite(log_term2)
            and torch.all(torch.isfinite(log_term3[nonzero_mask]))
        ):
            return torch.tensor(float("-inf"), dtype=torch.float64)

        # Broadcast full expression then mask to preserve original shape
        log_pr[nonzero_mask] = (log_term1 + log_term2 + log_term3)[nonzero_mask]

    # Final numerical check
    if not torch.all(torch.isfinite(log_pr)):
        return torch.tensor(float("-inf"), dtype=torch.float64)

    return torch.sum(log_pr)


def calculate_non_spine_ages_log_lik(
    lambda_params: torch.Tensor | float,
    mu_params: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    times: torch.Tensor | float,
) -> torch.Tensor:
    """
    Calculate log-likelihood of manuscript ages for non-spine individuals.

    Uses only PyTorch operations to preserve autograd support.

    Parameters
    ----------
    lambda_params : torch.Tensor | float
        Birth rate(s) λ. Scalar for single epoch, 1D tensor for piecewise-constant rates.
        All values must be positive.
    mu_params : torch.Tensor | float
        Death rate(s) μ. Scalar for single epoch, 1D tensor for piecewise-constant rates.
        All values must be positive.
    ages_t : torch.Tensor | float
        Manuscript age(s) t. Scalar or 1D tensor of ages (t ≥ 0).
        All ages must satisfy 0 ≤ t ≤ T.
    times : torch.Tensor | float
        Observation time(s) T. Scalar for single epoch, 1D tensor of strictly increasing
        epoch end times for piecewise-constant model.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood across ages.
        dtype: torch.float64. Fully differentiable via PyTorch autograd.
        Returns -inf for numerically invalid inputs.

    Raises
    ------
    ValueError
        If array inputs have mismatched lengths, times are not strictly increasing,
        or ages fall outside [0, T].

    Notes
    -----
    - Handles λ ≈ μ limiting case for numerical stability using log1p and expm1.
    - Supports single-epoch (scalar) or multi-epoch (array) piecewise-constant rates.
    - Multi-epoch case uses time-reversal transformation to reduce to extinction likelihood.
    - All operations preserve PyTorch's computational graph for gradient computation.

    Examples
    --------
    Single-epoch case:
    >>> lambda_val = torch.tensor(0.1, requires_grad=True)
    >>> mu_val = torch.tensor(0.05, requires_grad=True)
    >>> ages = torch.tensor([10.0, 20.0, 30.0])
    >>> T = 100.0
    >>> log_lik = calculate_non_spine_ages_log_lik(lambda_val, mu_val, ages, T)
    >>> log_lik.backward()  # Compute gradients w.r.t. lambda_val and mu_val
    """
    # Normalize to 1D arrays so scalar inputs still support indexing
    lambda_params, mu_params, times, ages_t = ensure_1d_tensors(
        lambda_params, mu_params, times, ages_t
    )

    validate_matching_shapes(
        lambda_params,
        mu_params,
        times,
        names=("lambda_params", "mu_params", "times"),
    )

    end_time = times[-1]

    _validate_times_strictly_increasing(
        times,
        error_message=f"times must be strictly increasing. Got times: {times}",
        require_positive_start=True,
    )
    _validate_ages_in_range(
        ages_t,
        end_time,
        error_message=f"ages_t must be in [0, {end_time}]. Got ages_t: {ages_t}",
    )

    validate_matching_lengths(
        lambda_params,
        mu_params,
        times,
        names=("lambda_params", "mu_params", "times"),
    )

    if len(times) > 1:
        # Multi-epoch case: use time-reversal transformation
        inverted_times = torch.flip(
            end_time
            - torch.cat([torch.tensor([0.0], dtype=torch.float64), times])[:-1],
            dims=[0],
        )
        reflected_birth_rates = torch.flip(mu_params, dims=[0])
        reflected_death_rates = torch.flip(lambda_params, dims=[0])
        return calculate_log_lik_initial_lifetimes_given_extinction(
            ages_t,
            reflected_birth_rates,
            reflected_death_rates,
            inverted_times,
        )

    # Single-epoch case
    # Extract scalar tensors from single-element arrays (atleast_1d guarantees arrays)
    lambda_val = lambda_params[0]
    mu_val = mu_params[0]

    # Validate positive parameters
    if lambda_val <= 0 or mu_val <= 0:
        return torch.tensor(float("-inf"), dtype=torch.float64)

    # Pre-compute isclose check once (avoids repeated calls in vectorized operations)
    is_critical = torch.isclose(lambda_val, mu_val, rtol=1e-12).item()

    # Check for numerical stability in parameter space
    if is_critical:
        # Limiting case: λ ≈ μ - use average to preserve gradient for both parameters
        # f(t) = [1 + λ(T-t)] / T * e^{-λt}
        # log f(t) = log(1 + λ(T-t)) - log(T) - λt
        # Using average ensures both lambda_val and mu_val are in the computation graph
        avg_rate = (lambda_val + mu_val) / 2
        log_numerator = -avg_rate * ages_t + torch.log1p(avg_rate * (end_time - ages_t))
        log_denominator = torch.log(end_time)

        # Check for numerical validity
        if not (
            torch.all(torch.isfinite(log_numerator)) and torch.isfinite(log_denominator)
        ):
            return torch.tensor(float("-inf"), dtype=torch.float64)

    else:
        # General case: λ ≠ μ
        growth_rate = lambda_val - mu_val
        growth_rate_T = growth_rate * end_time

        # Calculate numerator terms
        numerator_term1 = lambda_val * torch.exp(-lambda_val * ages_t)
        numerator_term2 = (
            mu_val * torch.exp(-growth_rate_T) * torch.exp(-mu_val * ages_t)
        )

        numerator = numerator_term1 - numerator_term2
        denominator = 1 - torch.exp(-growth_rate_T)

        # Check sign consistency
        if not torch.all(torch.sign(numerator) == torch.sign(denominator)):
            return torch.tensor(float("-inf"), dtype=torch.float64)

        log_numerator = torch.log(torch.abs(numerator))
        log_denominator = torch.log(torch.abs(denominator))

        # Check for numerical validity
        if not (
            torch.all(torch.isfinite(log_numerator)) and torch.isfinite(log_denominator)
        ):
            return torch.tensor(float("-inf"), dtype=torch.float64)

    # Calculate final log density
    log_density = torch.sum(log_numerator - log_denominator)

    # Final numerical check
    if not torch.isfinite(log_density):
        return torch.tensor(float("-inf"), dtype=torch.float64)

    return log_density


def calculate_spine_ages_log_lik(
    lambda_params: torch.Tensor | float,
    mu_params: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    times: torch.Tensor | float,
) -> torch.Tensor:
    """
    Calculate log-likelihood of manuscript ages for the "spine" individual.

    Uses only PyTorch operations to preserve autograd support.

    Parameters
    ----------
    lambda_params : torch.Tensor | float
        Birth rate(s) λ.
    mu_params : torch.Tensor | float
        Death rate(s) μ.
    ages_t : torch.Tensor | float
        Manuscript age(s) t (0 ≤ t ≤ T).
    times : torch.Tensor | float
        Observation time(s) T.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood across ages.
        dtype: torch.float64. Supports automatic differentiation.

    Notes
    -----
    Ages at T use discrete mass; ages < T use continuous density. Handles λ ≈ μ
    limiting case. Supports piecewise-constant rates via array inputs.
    """

    # Normalize to 1D arrays so scalar inputs still support indexing
    lambda_params, mu_params, times, ages_t = ensure_1d_tensors(
        lambda_params, mu_params, times, ages_t
    )

    validate_matching_shapes(
        lambda_params,
        mu_params,
        times,
        names=("lambda_params", "mu_params", "times"),
    )

    end_time = times[-1]

    _validate_times_strictly_increasing(
        times,
        error_message=f"times must be strictly increasing. Got times: {times}",
        require_positive_start=True,
    )
    _validate_ages_in_range(
        ages_t,
        end_time,
        error_message=f"ages_t must be in [0, {end_time}]. Got ages_t: {ages_t}",
    )

    validate_matching_lengths(
        lambda_params,
        mu_params,
        times,
        names=("lambda_params", "mu_params", "times"),
    )

    if len(times) > 1:
        inverted_times = torch.flip(
            end_time
            - torch.cat([torch.tensor([0.0], dtype=torch.float64), times])[:-1],
            dims=[0],
        )
        reflected_birth_rates = torch.flip(mu_params, dims=[0])
        reflected_death_rates = torch.flip(lambda_params, dims=[0])
        return calculate_log_lik_initial_lifetimes_given_non_extinction(
            lifetimes=ages_t,
            birth_rates=reflected_birth_rates,
            death_rates=reflected_death_rates,
            times=inverted_times,
        )

    # Handle parameter scaling
    # Extract scalar tensors from single-element arrays (atleast_1d guarantees arrays)
    lambda_t = lambda_params[0]
    mu_t = mu_params[0]

    # Validate positive parameters
    if lambda_t <= 0 or mu_t < 0:
        return torch.tensor(float("-inf"), dtype=torch.float64)

    # Pre-compute isclose check once (avoids repeated calls in vectorized operations)
    is_critical = torch.isclose(lambda_t, mu_t, rtol=1e-12).item()

    at_boundary = ages_t == end_time
    not_at_boundary = torch.logical_not(at_boundary)

    if is_critical:
        # Limiting case: λ ≈ μ - use average to preserve gradient for both parameters
        # When λ ≈ μ, f(t) ≈ [1 + λ(T-t)]/T * e^{-λt}
        # Using average ensures both lambda_t and mu_t are in the computation graph
        avg_rate = (lambda_t + mu_t) / 2
        boundary_val = torch.log1p(avg_rate * end_time) - avg_rate * end_time
        log_density = torch.where(
            at_boundary,
            boundary_val,
            2 * torch.log(avg_rate) + torch.log(ages_t) - avg_rate * ages_t,
        )
        return torch.sum(log_density)

    # General case: λ ≠ μ

    # Initialize arrays for numerator and denominator
    log_numerator = torch.zeros_like(ages_t, dtype=torch.float64)
    denom_t = lambda_t - mu_t
    sign_denominator = torch.sign(denom_t)
    log_denominator = torch.log(torch.abs(denom_t))

    if torch.any(at_boundary):
        numerator_boundary = lambda_t * torch.exp(-mu_t * end_time) - mu_t * torch.exp(
            -lambda_t * end_time
        )

        if not torch.all(torch.sign(numerator_boundary) == sign_denominator):
            return torch.tensor(float("-inf"), dtype=torch.float64)

        log_numerator[at_boundary] = torch.log(torch.abs(numerator_boundary))

    if torch.any(not_at_boundary):
        numerator_1 = torch.exp(-mu_t * end_time) - torch.exp(-lambda_t * end_time)

        if not torch.all(torch.sign(numerator_1) == sign_denominator):
            return torch.tensor(float("-inf"), dtype=torch.float64)

        log_numerator[not_at_boundary] = (
            torch.log(lambda_t) + torch.log(mu_t) + torch.log(torch.abs(numerator_1))
        )

    # Check for numerical validity
    if not (
        torch.all(torch.isfinite(log_numerator)) and torch.isfinite(log_denominator)
    ):
        return torch.tensor(float("-inf"), dtype=torch.float64)

    # Calculate final log density (sum across all ages)
    log_density = torch.sum(log_numerator - log_denominator)

    # Final numerical check
    if not torch.isfinite(log_density):
        return torch.tensor(float("-inf"), dtype=torch.float64)

    return log_density


def calculate_ages_log_lik(
    lambda_params: torch.Tensor | float,
    mu_params: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    times: torch.Tensor | float,
    are_spines: bool = False,
) -> torch.Tensor:
    """
    Compute log-likelihood of manuscript ages under a birth-death model.

    Dispatches to spine or non-spine likelihood calculation based on ``are_spines``.
    All operations preserve PyTorch's computational graph for automatic differentiation.

    Parameters
    ----------
    lambda_params : torch.Tensor | float
        Birth rate(s) λ. Scalar for single epoch, 1D tensor for piecewise-constant rates.
        All values must be positive. Will be converted to torch.float64.
    mu_params : torch.Tensor | float
        Death rate(s) μ. Scalar for single epoch, 1D tensor for piecewise-constant rates.
        All values must be positive. Will be converted to torch.float64.
    ages_t : torch.Tensor | float
        Manuscript age(s) t. Scalar or 1D tensor. All ages must satisfy 0 ≤ t ≤ T.
        Will be converted to torch.float64.
    times : torch.Tensor | float
        Observation time(s) T. Scalar for single epoch, 1D tensor of strictly increasing
        epoch end times for piecewise-constant model. Will be converted to torch.float64.
    are_spines : bool, optional
        If True, computes spine likelihood (discrete mass at T, continuous for t < T).
        If False, computes non-spine likelihood. Default False.

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
        or ages fall outside [0, T].

    Notes
    -----
    - Handles λ ≈ μ limiting case for numerical stability using log1p/expm1.
    - Supports both single-epoch (scalar) and multi-epoch (array) piecewise-constant rates.
    - Multi-epoch cases use time-reversal transformation internally.
    - All operations maintain PyTorch's computational graph for gradient computation.
    - Internally delegates to ``calculate_spine_ages_log_lik`` or
      ``calculate_non_spine_ages_log_lik`` depending on ``are_spines`` flag.

    Examples
    --------
    Single-epoch case with gradient computation:

    >>> lambda_val = torch.tensor(0.1, requires_grad=True)
    >>> mu_val = torch.tensor(0.05, requires_grad=True)
    >>> ages = torch.tensor([10.0, 20.0, 30.0])
    >>> T = 100.0
    >>> log_lik = calculate_ages_log_lik(lambda_val, mu_val, ages, T, are_spines=False)
    >>> log_lik.backward()  # Compute gradients w.r.t. lambda_val and mu_val
    >>> print(lambda_val.grad, mu_val.grad)
    """
    if are_spines:
        return calculate_spine_ages_log_lik(lambda_params, mu_params, ages_t, times)
    else:
        # calculate_non_spine_ages_log_lik requires array inputs
        # Wrap scalar parameters in single-element arrays for single-epoch case
        return calculate_non_spine_ages_log_lik(
            lambda_params,
            mu_params,
            ages_t,
            times,
        )


def calculate_extinction_pr(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    times: torch.Tensor | float,
    return_log_prob: bool = False,
) -> torch.Tensor:
    """
    Calculate extinction probability for a birth-death process.

    Parameters
    ----------
    birth_rates : torch.Tensor or float
        Birth rate(s) λ. Scalar or 1D tensor for piecewise-constant rates.
    death_rates : torch.Tensor or float
        Death rate(s) μ. Scalar or 1D tensor for piecewise-constant rates.
    times : torch.Tensor or float
        Observation time(s) T. Scalar for single time, 1D tensor for epoch end times.
        For arrays, must be monotonically increasing.
    return_log_prob : bool, optional
        If True, return log(α_t) instead of α_t. Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing extinction probability α_t
        (or log(α_t) if return_log_prob=True). dtype: torch.float64.
        Supports automatic differentiation.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Returns tensor
    for full PyTorch autograd support. All computation is in PyTorch.
    """
    # Convert inputs to tensors
    birth_rates_t, death_rates_t, times_t = ensure_1d_tensors(
        birth_rates, death_rates, times
    )

    _validate_times_non_negative(
        times_t, error_message=f"times must be non-negative. Got times: {times_t}"
    )

    _validate_times_strictly_increasing(
        times_t,
        error_message=f"times must be strictly increasing. Got times: {times_t}",
        require_positive_start=True,
    )

    pr_extinction_numerator = _pr_extinction_numerator(
        birth_rates=birth_rates_t,
        death_rates=death_rates_t,
        times=times_t,
    )
    if return_log_prob:
        log_alpha_t = torch.log(pr_extinction_numerator) - torch.log1p(
            pr_extinction_numerator
        )
        return log_alpha_t

    alpha_t = 1 - 1 / (1 + pr_extinction_numerator)

    return alpha_t


def calculate_log_lik_initial_lifetimes_given_extinction(
    lifetimes: torch.Tensor,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate log-likelihood of initial lifetime(s) given extinction.

    Uses only PyTorch operations to maintain autograd support.

    Parameters
    ----------
    lifetimes : torch.Tensor
        1D tensor of manuscript lifetimes (shape: (n,)), each ≤ times[-1].
    birth_rates : torch.Tensor
        1D tensor of birth rates λ per epoch (shape: (n_epochs,)).
    death_rates : torch.Tensor
        1D tensor of death rates μ per epoch; same length as ``birth_rates``.
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times matching rate arrays.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood values.
        dtype: torch.float64. Supports automatic differentiation.
    """

    n = len(lifetimes)

    log_prs_extinction_given_lifetimes = (
        calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes=lifetimes,
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
        )
    )
    log_liks_lifetimes = calculate_log_lik_lifetimes(
        lifetimes=lifetimes,
        birth_rates=birth_rates,
        times=times,
    )
    log_extinction_prob = calculate_extinction_pr(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        return_log_prob=True,
    )

    # Compute conditional log likelihood
    result = (
        torch.sum(log_prs_extinction_given_lifetimes)
        + log_liks_lifetimes
        - n * log_extinction_prob
    )
    return result


def calculate_log_lik_initial_lifetimes_given_non_extinction(
    lifetimes: torch.Tensor,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate log-likelihood of initial lifetime(s) given non-extinction.

    This implementation uses only PyTorch operations to preserve
    autograd support end-to-end.

    Parameters
    ----------
    lifetimes : torch.Tensor
        1D tensor of manuscript lifetimes (shape: (n,)), each ≤ times[-1].
    birth_rates : torch.Tensor
        1D tensor of birth rates λ per epoch (shape: (n_epochs,)).
    death_rates : torch.Tensor
        1D tensor of death rates μ per epoch; same length as ``birth_rates``.
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times matching rate arrays.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood values
        in torch.float64. Supports automatic differentiation.
    """

    n = len(lifetimes)

    log_prs_extinction_given_lifetimes = (
        calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes=lifetimes,
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            as_array=True,
        )
    )
    log_liks_lifetimes = calculate_log_lik_lifetimes(
        lifetimes=lifetimes,
        birth_rates=birth_rates,
        times=times,
    )
    log_extinction_prob = calculate_extinction_pr(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        return_log_prob=True,
    )

    log_prs_non_extinction_given_lifetimes = torch.log(
        -torch.expm1(log_prs_extinction_given_lifetimes)
    )

    # Compute conditional log likelihood
    result = (
        torch.sum(log_prs_non_extinction_given_lifetimes)
        + log_liks_lifetimes
        - n * torch.log(-torch.expm1(log_extinction_prob))
    )
    return result


def calculate_log_pr_extinction_given_initial_lifetimes(
    lifetimes: torch.Tensor | float,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
    as_array: bool = False,
) -> torch.Tensor:
    """
    Calculate log probability of extinction given initial lifetime(s).

    Parameters
    ----------
    lifetimes : torch.Tensor
        1D tensor of manuscript lifetimes (shape: (n,), dtype: torch.float64).
        Each lifetime must be ≤ times[-1].
    birth_rates : torch.Tensor
        1D tensor of birth rates λ per epoch (shape: (n_epochs,), dtype: torch.float64).
    death_rates : torch.Tensor
        1D tensor of death rates μ per epoch. Same shape as ``birth_rates``
        (dtype: torch.float64).
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times (shape: (n_epochs,),
        dtype: torch.float64). Must have same length as ``birth_rates``.
    as_array : bool, optional
        If True, return log probabilities per lifetime (aligned to the original
        ordering); if False, return their sum. Default False.

    Returns
    -------
    torch.Tensor
        If ``as_array`` is False, scalar tensor (shape ()) containing summed
        log extinction probability. If ``as_array`` is True, 1D tensor of log
        extinction probabilities aligned with ``lifetimes``. dtype: torch.float64.
        Supports automatic differentiation.

    Notes
    -----
    Uses adaptive quadrature (scipy.integrate.quad) for numerical integration.
    Handles multi-epoch piecewise-constant rates with cumulative integral caching
    for efficiency.
    """

    # Convert inputs to tensors
    lifetimes_t, birth_rates_t, death_rates_t, times_t = ensure_1d_tensors(
        lifetimes, birth_rates, death_rates, times
    )

    validate_matching_lengths(
        birth_rates_t,
        death_rates_t,
        times_t,
        names=("birth_rates", "death_rates", "times"),
    )

    birth_rates_array = birth_rates_t

    # Sort lifetimes to enable cumulative integral caching
    sorted_lifetimes, sort_indices = torch.sort(lifetimes_t)

    if sorted_lifetimes[-1] > times_t[-1]:
        raise ValueError(
            "All lifetimes must be less than or equal to the final time point."
            + f" Got max lifetime: {sorted_lifetimes[-1]}, final time: {times_t[-1]}"
        )

    log_probs = torch.zeros_like(sorted_lifetimes, dtype=torch.float64)

    # Build integration intervals from sorted lifetimes
    n_lifetimes = len(sorted_lifetimes)

    # Create interval boundaries: [0, sorted_lifetimes[0], sorted_lifetimes[1], ...]
    interval_boundaries = torch.cat(
        [torch.tensor([0.0], dtype=torch.float64), sorted_lifetimes]
    )

    # Create intervals: each row is [start, end]
    interval_starts = interval_boundaries[:-1]
    interval_ends = interval_boundaries[1:]

    # Identify non-zero width intervals
    nonzero_mask = interval_ends > interval_starts
    active_indices = torch.where(nonzero_mask)[0]

    cumulative_integrals = torch.zeros(n_lifetimes, dtype=torch.float64)

    if len(active_indices) > 0:
        # Use scipy's adaptive quadrature for integration
        # Detach tensors and convert to numpy for scipy
        birth_rates_np = birth_rates_t.detach().cpu().numpy()
        death_rates_np = death_rates_t.detach().cpu().numpy()
        times_np = times_t.detach().cpu().numpy()

        interval_integrals = []
        for idx in active_indices:
            a = float(interval_starts[idx].item())
            b = float(interval_ends[idx].item())
            result, _ = scipy.integrate.quad(
                lambda t: float(
                    _integrand_surv_from_weighted_birth_time_scalar(
                        t, birth_rates_np, death_rates_np, times_np
                    )
                ),
                a,
                b,
            )
            interval_integrals.append(result)

        # Build cumulative integrals
        interval_integral_full = torch.zeros(n_lifetimes, dtype=torch.float64)
        for i, idx in enumerate(active_indices):
            interval_integral_full[idx] = torch.tensor(
                interval_integrals[i], dtype=torch.float64
            )

        cumulative_integrals = torch.cumsum(interval_integral_full, dim=0)

    # Compute cumulative birth rates for each lifetime
    for i, lifetime in enumerate(sorted_lifetimes):
        # Find epochs that end before this lifetime
        epoch_mask = times_t < lifetime
        num_full_epochs = int(torch.sum(epoch_mask).item())

        # Build lifetime epochs
        full_epochs = times_t[epoch_mask]
        lifetime_epochs = torch.cat([full_epochs, lifetime.unsqueeze(0)])

        # Compute epoch durations
        lifetime_epochs_padded = torch.cat(
            [torch.tensor([0.0], dtype=torch.float64), lifetime_epochs]
        )
        epoch_durs = torch.diff(lifetime_epochs_padded)

        # Compute cumulative birth rate
        cum_birth_rate = torch.sum(
            birth_rates_array[: num_full_epochs + 1] * epoch_durs
        )

        log_probs[i] = cumulative_integrals[i] - cum_birth_rate

    if as_array:
        return log_probs

    return torch.sum(log_probs)


def calculate_log_lik_lifetimes(
    lifetimes: torch.Tensor,
    birth_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate log-likelihood of manuscript lifetime(s).

    Parameters
    ----------
    lifetimes : torch.Tensor
        1D tensor of lifetimes (shape: (n,), dtype: torch.float64). Each lifetime ≤ times[-1].
    birth_rates : torch.Tensor
        1D tensor of birth rates λ per epoch (shape: (n_epochs,), dtype: torch.float64).
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times (shape: (n_epochs,), dtype: torch.float64).

    Returns
    -------
    torch.Tensor
        Scalar tensor containing sum of log-likelihood values.
        Shape: (), dtype: torch.float64. Supports automatic differentiation.
    """

    # Convert inputs to tensors
    lifetimes_t, birth_rates_t, times_t = ensure_1d_tensors(
        lifetimes, birth_rates, times
    )

    lambda_val_array = birth_rates_t

    # Compute epoch starts and durations
    times_padded = torch.cat([torch.tensor([0.0], dtype=torch.float64), times_t])
    epoch_starts = times_padded[:-1]
    epoch_durations = times_t - epoch_starts

    # Compute cumulative growth at each epoch
    cum_growth_at_epochs = torch.cumsum(lambda_val_array * epoch_durations, dim=0)

    log_liks = torch.zeros_like(lifetimes_t, dtype=torch.float64)

    for i, lifetime in enumerate(lifetimes_t):
        if torch.isclose(lifetime, times_t[-1]):
            # Boundary case
            log_liks[i] = -cum_growth_at_epochs[-1]
            continue

        # Find valid epoch indices where epoch_starts < lifetime
        valid_mask = epoch_starts < lifetime
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        # The current epoch index is the last valid index
        current_epoch_idx = valid_indices[-1]
        current_lambda = lambda_val_array[current_epoch_idx]

        # Compute cumulative growth up to current lifetime
        cum_growth = (
            cum_growth_at_epochs[current_epoch_idx - 1]
            if current_epoch_idx > 0
            else torch.tensor(0.0, dtype=torch.float64)
        ) + (lifetime - epoch_starts[current_epoch_idx]) * current_lambda

        log_liks[i] = torch.log(current_lambda) - cum_growth

    return torch.sum(log_liks)


def _population_geometric_parameter(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate geometric parameter (1 - β_t) for offspring distribution.

    Private helper function for computing the conditional geometric parameter
    of the population distribution given survival.

    Parameters
    ----------
    birth_rates : torch.Tensor
        1D tensor of birth rates λ per epoch (shape: (n_epochs,), dtype: torch.float64).
    death_rates : torch.Tensor
        1D tensor of death rates μ per epoch; same shape as ``birth_rates`` (dtype: torch.float64).
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times (shape: (n_epochs,), dtype: torch.float64).

    Returns
    -------
    torch.Tensor
        Scalar tensor containing geometric parameter 1 - β_t ∈ (0, 1).
        Shape: (), dtype: torch.float64. Supports automatic differentiation.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Full support for
    PyTorch's autograd differentiation.
    """
    # Convert inputs to tensors
    birth_rates_t, death_rates_t, times_t = ensure_1d_tensors(
        birth_rates, death_rates, times
    )

    _validate_times_strictly_increasing(
        times_t,
        error_message="times array must be strictly increasing.",
        require_positive_start=True,
    )

    # Compute epoch durations
    times_padded = torch.cat([torch.tensor([0.0], dtype=torch.float64), times_t])
    epoch_durations = torch.diff(times_padded)

    lambda_val_array = birth_rates_t
    mu_val_array = death_rates_t

    growth_rates = lambda_val_array - mu_val_array
    cum_growth = torch.cumsum(growth_rates * epoch_durations, dim=0)

    geom_sum = _pr_extinction_numerator(birth_rates_t, death_rates_t, times_t)
    # Compute in log-domain to avoid overflow: cond = exp(-C) / (1 + geom_sum)
    # => log(cond) = -C - log1p(geom_sum)
    log_cond = -cum_growth[-1] - torch.log1p(geom_sum)
    cond_geom_prob = torch.exp(log_cond)
    return cond_geom_prob


def _integrand_surv_from_weighted_birth_time(
    birth_time: torch.Tensor | float,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Integrand for survival probability calculation (scalar version).

    Parameters
    ----------
    birth_time : torch.Tensor | float
        Scalar time at which birth occurred.
    birth_rates : torch.Tensor
        1D tensor of birth rates λ per epoch (shape: (n_epochs,), dtype: torch.float64).
    death_rates : torch.Tensor
        1D tensor of death rates μ per epoch; same shape as ``birth_rates`` (dtype: torch.float64).
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times (shape: (n_epochs,), dtype: torch.float64).

    Returns
    -------
    torch.Tensor
        Scalar tensor containing integrand value. Shape: (),
        dtype: torch.float64. Supports automatic differentiation.
    """
    # Convert inputs to tensors
    birth_time_t, birth_rates_t, death_rates_t, times_t = ensure_1d_tensors(
        birth_time, birth_rates, death_rates, times
    )

    # Find the index of the last epoch that starts before or at start_time
    times_shifted = times_t - birth_time_t
    valid_mask = times_shifted > 0
    valid_indices = torch.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError("birth_time exceeds all epoch end times.")

    trunc_times = times_shifted[valid_indices]
    trunc_birth_rates = birth_rates_t[valid_indices]
    trunc_death_rates = death_rates_t[valid_indices]

    ext_prob = calculate_extinction_pr(
        birth_rates=trunc_birth_rates,
        death_rates=trunc_death_rates,
        times=trunc_times,
    )

    return ext_prob * trunc_birth_rates[0]


def _integrand_surv_from_weighted_birth_time_scalar(
    birth_time: float,
    birth_rates_np: NDArray[np.float64],
    death_rates_np: NDArray[np.float64],
    times_np: NDArray[np.float64],
) -> float:
    """
    Scalar integrand for survival probability calculation (numpy version for scipy.integrate).

    Parameters
    ----------
    birth_time : float
        Time at which birth occurred (scalar).
    birth_rates_np : np.ndarray
        Birth rates λ for each epoch.
    death_rates_np : np.ndarray
        Death rates μ for each epoch.
    times_np : np.ndarray
        Epoch end times.

    Returns
    -------
    float
        Integrand value for this birth time.
    """
    import numpy as np

    shifted_times = times_np - birth_time
    valid_mask = shifted_times > 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return 0.0

    trunc_times = shifted_times[valid_indices]
    trunc_birth_rates = birth_rates_np[valid_indices]
    trunc_death_rates = death_rates_np[valid_indices]

    # Convert back to torch for calculation, but without gradients
    ext_prob = calculate_extinction_pr(
        birth_rates=torch.from_numpy(trunc_birth_rates),
        death_rates=torch.from_numpy(trunc_death_rates),
        times=torch.from_numpy(trunc_times),
    )

    return float(ext_prob.item() * trunc_birth_rates[0])


def _integrand_surv_from_weighted_birth_time_vectorized(
    birth_times: torch.Tensor,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized integrand for survival probability calculation.

    Computes integrand values for multiple birth times simultaneously.

    Parameters
    ----------
    birth_times : torch.Tensor
        1D tensor of times at which births occurred. Shape: (n_points,), dtype: torch.float64.
    birth_rates : torch.Tensor
        1D tensor of birth rates λ for each epoch. Shape: (n_epochs,), dtype: torch.float64.
    death_rates : torch.Tensor
        1D tensor of death rates μ for each epoch. Shape: (n_epochs,), dtype: torch.float64.
    times : torch.Tensor
        1D tensor of epoch end times. Shape: (n_epochs,), dtype: torch.float64.

    Returns
    -------
    torch.Tensor
        1D tensor of integrand values for each birth time. Shape: (n_points,),
        dtype: torch.float64. Supports automatic differentiation.
    """
    # Convert inputs to tensors
    birth_times_t, birth_rates_t, death_rates_t, times_t = ensure_1d_tensors(
        birth_times, birth_rates, death_rates, times
    )

    validate_matching_shapes(
        birth_rates_t,
        death_rates_t,
        times_t,
        names=("birth_rates", "death_rates", "times"),
    )
    n_points = len(birth_times_t)
    results = torch.zeros(n_points, dtype=torch.float64)

    # For each birth time, compute the extinction probability
    # This is the main computational bottleneck - we compute extinction probabilities
    # for different time-shifted parameter sets
    has_valid_indices = False
    for i, birth_time in enumerate(birth_times_t):
        shifted_times = times_t - birth_time
        valid_mask = shifted_times > 0
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            results[i] = torch.tensor(0.0, dtype=torch.float64)
            continue

        has_valid_indices = True
        trunc_times = shifted_times[valid_indices]
        trunc_birth_rates = birth_rates_t[valid_indices]
        trunc_death_rates = death_rates_t[valid_indices]

        ext_prob = calculate_extinction_pr(
            birth_rates=trunc_birth_rates,
            death_rates=trunc_death_rates,
            times=trunc_times,
        )

        results[i] = ext_prob * trunc_birth_rates[0]

    if not has_valid_indices:
        raise ValueError(
            "All birth times exceed all epoch end times. "
            f"Max birth_time: {torch.max(birth_times_t)}, final epoch time: {times_t[-1]}"
        )

    return results


def _pr_extinction_numerator(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate numerator of extinction probability integral.

    Private helper function for extinction probability calculation.

    Parameters
    ----------
    birth_rates : torch.Tensor
        1D tensor of birth rates λ per epoch (shape: (n_epochs,), dtype: torch.float64).
    death_rates : torch.Tensor
        1D tensor of death rates μ per epoch; same shape as ``birth_rates`` (dtype: torch.float64).
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times (shape: (n_epochs,), dtype: torch.float64).

    Returns
    -------
    torch.Tensor
        Scalar tensor containing numerator value for extinction probability.
        Shape: (), dtype: torch.float64. Supports automatic differentiation.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Full support for
    PyTorch's autograd differentiation.
    """

    birth_rates, death_rates, times = ensure_1d_tensors(birth_rates, death_rates, times)

    validate_matching_shapes(
        birth_rates,
        death_rates,
        times,
        names=("birth_rates", "death_rates", "times"),
    )

    epoch_durations = torch.diff(times, prepend=times.new_zeros(1))
    growth_rates = birth_rates - death_rates
    x = growth_rates * epoch_durations
    critical_mask = torch.abs(x) < 1e-10
    cum_growth = torch.cumsum(x, dim=0)
    trunc_cum_growth = torch.cat([cum_growth.new_zeros(1), cum_growth[:-1]])
    kernel = torch.where(
        condition=critical_mask,
        input=epoch_durations,
        other=-torch.expm1(-x) / growth_rates,
    )
    pr_extinction_numerator_summand = (
        death_rates * torch.exp(-trunc_cum_growth) * kernel
    )

    return pr_extinction_numerator_summand.sum()
