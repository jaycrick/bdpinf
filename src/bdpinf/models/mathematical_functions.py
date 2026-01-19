"""
Mathematical functions for birth-and-death process calculations.

This module contains pure mathematical functions for calculating birth-and-death
process probabilities and densities. These functions are isolated from the
main BirthDeathProcess class for better testability and modularity.
"""

import numpy as np
import scipy
import torch  # type: ignore


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
    are summed and returned as a single np.float64 value. Invalid or numerically
    unstable inputs return -inf (np.float64(-np.inf)).

    Parameters
    ----------
    lambda_params : float or NDArray[np.float64]
        Birth rate(s) λ. Scalar for constant rate, array for piecewise-constant rates.
        If `log_scale_params` is True, interpreted as log(λ).
    mu_params : float or NDArray[np.float64]
        Death rate(s) μ. Scalar for constant rate, array for piecewise-constant rates.
        If `log_scale_params` is True, interpreted as log(μ).
    times : float or NDArray[np.float64]
        Time(s) T at which to evaluate the distribution. Scalar for single time point,
        array for epoch end times in piecewise-constant model. Must be ≥ 0 and increasing.
    ns : int or NDArray[np.integer]
        Non-negative population count(s) for which to evaluate log-probabilities.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing sum of log p_n(T) across provided `ns`.
        dtype: torch.float64. Supports automatic differentiation.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Array inputs must have
    matching lengths. Returns -inf for numerical issues, raises exceptions for
    programming errors.
    """
    # Convert scalar inputs to arrays for consistent handling
    lambda_params = np.atleast_1d(lambda_params).ravel()
    mu_params = np.atleast_1d(mu_params).ravel()
    times = np.atleast_1d(times).ravel()
    ns = np.atleast_1d(ns).ravel()

    # Input validation
    if np.any(ns < 0):
        raise ValueError(f"Population counts ns must be non-negative. Got ns: {ns}")
    if np.any(times < 0):
        raise ValueError(f"Times must be non-negative. Got times: {times}")
    if not (len(lambda_params) == len(mu_params) == len(times)):
        raise ValueError(
            f"Array inputs must have matching lengths. "
            f"Got: lambda_params length={len(lambda_params)}, "
            f"mu_params length={len(mu_params)}, time length={len(times)}"
        )

    if len(times) > 1:
        if np.any(lambda_params < 0) or np.any(mu_params < 0):
            return np.float64(-np.inf)
        lambda_val_array = lambda_params
        mu_val_array = mu_params
        if not np.all(np.diff(times) > 0):
            raise ValueError(
                f"Epoch times must be strictly monotonically increasing. "
                f"Got time array: {times}"
            )

        alpha_t = calculate_extinction_pr(lambda_val_array, mu_val_array, times)

        beta_t = 1 - _population_geometric_parameter(
            lambda_val_array, mu_val_array, times
        )
    else:
        lambda_params = lambda_params[0]
        mu_params = mu_params[0]
        times = times[0]
        # Handle parameter scaling and validation
        if np.any(lambda_params < 0) or np.any(mu_params < 0):
            return np.float64(-np.inf)
        lambda_val = lambda_params
        mu_val = mu_params

        if np.isclose(lambda_val, mu_val, rtol=1e-12):
            # Limiting case: λ ≈ μ (includes λ = μ = 0)
            rate_time_product = lambda_val * times
            alpha_t = beta_t = np.float64(rate_time_product / (1 + rate_time_product))
        else:
            # General case: λ ≠ μ
            growth_rate = lambda_val - mu_val
            exponent = growth_rate * times
            exp_term = torch.exp(exponent)
            if not torch.isfinite(exp_term):
                return np.float64(-np.inf)
            denominator = lambda_val * exp_term - mu_val
            # Guard against division by zero
            if mu_val == 0.0:
                alpha_t = np.float64(0.0)
                beta_t = np.float64(lambda_val * (exp_term - 1) / denominator)
            else:
                alpha_t = np.float64(mu_val * (exp_term - 1) / denominator)

                # Calculate β_t = (λ/μ) * α_t
                beta_t = np.float64((lambda_val / mu_val) * alpha_t)

    # Validate α_t and β_t are in valid range (allow β_t = 0 when n ≤ 1)
    if not (0 <= alpha_t <= 1):
        raise ValueError(f"Computed α_t out of bounds [0, 1]: α_t={alpha_t}")
    if not (0 <= beta_t <= 1):
        raise ValueError(f"Computed β_t out of bounds [0, 1]: β_t={beta_t}")
    if beta_t == 0.0 and np.any(ns > 1):
        return np.float64(-np.inf)

    # Initialize log probability array
    log_pr = np.zeros_like(ns, dtype=np.float64)

    # Handle n == 0 case
    zero_mask = ns == 0
    if np.any(zero_mask):
        log_pr[zero_mask] = torch.log(alpha_t)

    # Handle n > 0 case
    nonzero_mask = np.invert(zero_mask)
    if np.any(nonzero_mask):
        log_term1 = torch.log1p(-alpha_t)
        log_term2 = torch.log1p(-beta_t)
        log_term3 = np.where(ns == 1, 0.0, (ns - 1) * torch.log(beta_t))

        # Check all terms are finite
        if not (
            torch.isfinite(log_term1)
            and torch.isfinite(log_term2)
            and np.all(torch.isfinite(log_term3[nonzero_mask]))
        ):
            return np.float64(-np.inf)

        # Broadcast full expression then mask to preserve original shape
        log_pr[nonzero_mask] = (log_term1 + log_term2 + log_term3)[nonzero_mask]

    # Final numerical check
    if not np.all(torch.isfinite(log_pr)):
        return np.float64(-np.inf)

    return np.float64(np.sum(log_pr))


def calculate_populations_log_lik_derivatives(
    lambda_param: torch.Tensor | float,
    mu_param: torch.Tensor | float,
    time: torch.Tensor | float,
    ns: torch.Tensor | int,
) -> torch.Tensor:
    """
    Calculate derivatives of log population probabilities with respect to λ and μ.

    Parameters
    ----------
    lambda_param : float
        Birth rate λ.
    mu_param : float
        Death rate μ.
    time : float
        Time t at which to evaluate (t ≥ 0).
    ns : int or NDArray[np.int64]
        Population count(s) to evaluate.

    Returns
    -------
    torch.Tensor
        Tensor [∂/∂λ, ∂/∂μ] containing summed derivatives across all `ns`.
        Shape: (2,), dtype: torch.float64. Supports automatic differentiation.

    Notes
    -----
    Vectorized over `ns`. Handles λ ≈ μ limiting case for numerical stability.
    """

    # Handle parameter scaling
    lambda_val = lambda_param
    mu_val = mu_param

    # Ensure ns is an array for consistent masking
    ns = np.atleast_1d(ns)

    d_lambda = np.zeros_like(ns, dtype=np.float64)
    d_mu = np.zeros_like(ns, dtype=np.float64)

    zero_mask = ns == 0
    non_zero_mask = np.invert(zero_mask)

    # Pre-compute isclose check once (avoids repeated calls in vectorized operations)
    is_critical = np.isclose(lambda_val, mu_val, rtol=1e-14)

    if is_critical:
        # Limiting case: λ ≈ μ
        # α_t = β_t = λT/(1 + λT)

        if np.any(zero_mask):
            # For n=0: log p_0 = log(α_t) = log(λT) - log(1 + λT)
            # ∂/∂λ = 1/λ - T/(1 + λT) = 1/(λ(1 + λT))
            d_lambda[zero_mask] = 1 / (lambda_val * (1 + lambda_val * time))
            # When λ = μ, both derivatives are the same
            d_mu[zero_mask] = d_lambda[zero_mask]

        if np.any(non_zero_mask):
            # Extract only non-zero n values
            ns_nonzero = ns[non_zero_mask]
            # For n>0: log p_n = (n-1)log(λT) - (n+1)log(1+λT)
            # ∂/∂λ = (n-1)/λ - (n+1)T/(1+λT) = [n - 1 - 2λT]/(λ(1+λT))
            numerator = ns_nonzero - 1 - 2 * lambda_val * time
            denominator = lambda_val * (1 + lambda_val * time)
            d_lambda[non_zero_mask] = numerator / denominator
            # When λ = μ, both derivatives are the same
            d_mu[non_zero_mask] = d_lambda[non_zero_mask]

        return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)

    # General case: λ ≠ μ
    growth_rate = lambda_val - mu_val
    exp_rT = torch.exp(growth_rate * time)

    if np.any(zero_mask):
        # For n=0: d log(α_t) / d param = (1/α_t) * dα_t/d param

        # α_t = μ(e^{rT} - 1) / (λe^{rT} - μ)
        alpha_t = mu_val * (exp_rT - 1) / (lambda_val * exp_rT - mu_val)

        # ∂α_t/∂λ = μe^{rT} [(λ - μ)T - e^{rT} + 1] / (λe^{rT} - μ)²
        d_alpha_d_lambda = (
            mu_val
            * exp_rT
            * (growth_rate * time - exp_rT + 1)
            / (lambda_val * exp_rT - mu_val) ** 2
        )

        # ∂α_t/∂μ = e^{rT}[λ(e^{rT} - 1) - μT(λ - μ)] / (λe^{rT} - μ)²
        d_alpha_d_mu = (
            exp_rT
            * (lambda_val * (exp_rT - 1) - mu_val * time * growth_rate)
            / (lambda_val * exp_rT - mu_val) ** 2
        )

        d_lambda[zero_mask] = d_alpha_d_lambda / alpha_t
        d_mu[zero_mask] = d_alpha_d_mu / alpha_t
    if np.any(non_zero_mask):
        # Extract only non-zero n values for computation
        ns_nonzero = ns[non_zero_mask]

        # For n>0: log p_n = log(1-α_t) + log(1-β_t) + (n-1)log(β_t)
        # where β_t = (λ/μ) * α_t

        alpha_t = mu_val * (exp_rT - 1) / (lambda_val * exp_rT - mu_val)
        beta_t = (lambda_val / mu_val) * alpha_t

        # Derivatives of α_t (from above)
        d_alpha_d_lambda = (
            mu_val
            * exp_rT
            * (growth_rate * time - exp_rT + 1)
            / (lambda_val * exp_rT - mu_val) ** 2
        )
        d_alpha_d_mu = (
            exp_rT
            * (lambda_val * (exp_rT - 1) - mu_val * time * growth_rate)
            / (lambda_val * exp_rT - mu_val) ** 2
        )

        # Derivatives of β_t = (λ/μ) * α_t
        # ∂β_t/∂λ = (1/μ)·α_t + (λ/μ)·∂α_t/∂λ
        d_beta_d_lambda = (alpha_t / mu_val) + (lambda_val / mu_val) * d_alpha_d_lambda

        # ∂β_t/∂μ = -(λ/μ²)·α_t + (λ/μ)·∂α_t/∂μ
        d_beta_d_mu = (
            -(lambda_val / mu_val**2) * alpha_t + (lambda_val / mu_val) * d_alpha_d_mu
        )

        # ∂/∂λ [log(1-α_t) + log(1-β_t) + (n-1)log(β_t)]
        #     = -1/(1-α_t)·∂α_t/∂λ - 1/(1-β_t)·∂β_t/∂λ + (n-1)/β_t·∂β_t/∂λ
        d_lambda_nonzero = (
            -d_alpha_d_lambda / (1 - alpha_t)
            - d_beta_d_lambda / (1 - beta_t)
            + (ns_nonzero - 1) * d_beta_d_lambda / beta_t
        )

        # ∂/∂μ [log(1-α_t) + log(1-β_t) + (n-1)log(β_t)]
        #     = -1/(1-α_t)·∂α_t/∂μ - 1/(1-β_t)·∂β_t/∂μ + (n-1)/β_t·∂β_t/∂μ
        d_mu_nonzero = (
            -d_alpha_d_mu / (1 - alpha_t)
            - d_beta_d_mu / (1 - beta_t)
            + (ns_nonzero - 1) * d_beta_d_mu / beta_t
        )

        d_lambda[non_zero_mask] = d_lambda_nonzero
        d_mu[non_zero_mask] = d_mu_nonzero

    return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)


def calculate_non_spine_ages_log_lik(
    lambda_params: torch.Tensor | float,
    mu_params: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    times: torch.Tensor | float,
) -> torch.Tensor:
    """
    Calculate log-likelihood of manuscript ages for non-spine individuals.

    Parameters
    ----------
    lambda_params : float or NDArray[np.float64]
        Birth rate(s) λ.
    mu_params : float or NDArray[np.float64]
        Death rate(s) μ.
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (t ≥ 0).
    times : float or NDArray[np.float64]
        Observation time(s) T. Must satisfy T ≥ t for all ages.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood across ages.
        dtype: torch.float64. Supports automatic differentiation.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Supports piecewise-constant
    rates via array inputs.
    """
    # Normalize to 1D arrays so scalar inputs still support indexing
    lambda_params = np.atleast_1d(np.asarray(lambda_params, dtype=np.float64)).ravel()
    mu_params = np.atleast_1d(np.asarray(mu_params, dtype=np.float64)).ravel()
    times = np.atleast_1d(np.asarray(times, dtype=np.float64)).ravel()
    ages_t = np.atleast_1d(np.asarray(ages_t, dtype=np.float64)).ravel()

    if not (lambda_params.shape == mu_params.shape == times.shape):
        raise ValueError(
            f"lambda_params, mu_params, and times must have the same shape. "
            f"Got shapes: lambda_params={lambda_params.shape}, "
            f"mu_params={mu_params.shape}, times={times.shape}"
        )

    end_time = times[-1]

    if np.any(np.diff(np.insert(times, 0, 0.0)) <= 0):
        raise ValueError(f"times must be strictly increasing. Got times: {times}")
    # Input validation
    if np.any(ages_t < 0) or np.any(ages_t > end_time):
        raise ValueError(f"ages_t must be in [0, {end_time}]. Got ages_t: {ages_t}")

    if len(lambda_params) != len(mu_params):
        raise ValueError(
            f"lambda_params and mu_params must have the same length. "
            f"Got lengths: lambda_param={len(lambda_params)}, "
            f"mu_param={len(mu_params)}"
        )

    if len(lambda_params) != len(times):
        raise ValueError(
            f"lambda_params and times must have the same length. "
            f"Got lengths: lambda_param={len(lambda_params)}, times={len(times)}"
        )

    if len(times) > 1:
        # # Multi-epoch case with piecewise-constant rates.
        # # Generalized closed-form using cumulative integrals of λ and μ.
        # epoch_starts = np.insert(times, 0, 0.0)[:-1]
        # epoch_durations = times - epoch_starts

        # cum_lambda_epochs = np.cumsum(lambda_params * epoch_durations)
        # cum_mu_epochs = np.cumsum(mu_params * epoch_durations)
        # cum_growth_T = np.sum((lambda_params - mu_params) * epoch_durations)

        # denom = 1 - torch.exp(-cum_growth_T)
        # if denom == 0.0:
        #     return np.float64(-np.inf)
        # log_denominator = torch.log(np.abs(denom))

        # log_numerators = np.zeros_like(ages_t, dtype=np.float64)

        # for i, t in enumerate(ages_t):
        #     # Find epoch index j with epoch_starts[j] < t <= times[j]
        #     j = int(np.searchsorted(times, t, side="left"))
        #     # Accumulated integrals up to start of epoch j
        #     cum_lambda_before = cum_lambda_epochs[j - 1] if j > 0 else 0.0
        #     cum_mu_before = cum_mu_epochs[j - 1] if j > 0 else 0.0

        #     # Add contribution within current epoch up to t
        #     cum_lambda_t = cum_lambda_before + lambda_params[j] * (t - epoch_starts[j])
        #     cum_mu_t = cum_mu_before + mu_params[j] * (t - epoch_starts[j])

        #     lam_t = lambda_params[j]
        #     mu_t = mu_params[j]

        #     term1 = lam_t * torch.exp(-cum_lambda_t)
        #     term2 = mu_t * torch.exp(-cum_growth_T) * torch.exp(-cum_mu_t)
        #     numer = term1 - term2

        #     # Ensure consistent sign with denominator; otherwise invalid
        #     if np.sign(numer) != np.sign(denom):
        #         return np.float64(-np.inf)
        #     log_numerators[i] = torch.log(np.abs(numer))

        # return np.float64(np.sum(log_numerators) - len(ages_t) * log_denominator)
        inverted_times = np.flip(end_time - np.insert(times, 0, 0.0)[:-1])
        reflected_birth_rates = np.flip(mu_params)
        reflected_death_rates = np.flip(lambda_params)
        return calculate_log_lik_initial_lifetimes_given_extinction(
            ages_t,
            reflected_birth_rates,
            reflected_death_rates,
            inverted_times,
        )

    # Handle parameter scaling
    # Extract scalar values from single-element arrays (atleast_1d guarantees arrays)
    lambda_scalar = float(lambda_params[0])
    mu_scalar = float(mu_params[0])

    # Validate positive parameters
    if lambda_scalar < 0 or mu_scalar <= 0:
        return np.float64(-np.inf)
    lambda_val = lambda_scalar
    mu_val = mu_scalar

    # Pre-compute isclose check once (avoids repeated calls in vectorized operations)
    is_critical = np.isclose(lambda_val, mu_val, rtol=1e-12)

    # Check for numerical stability in parameter space
    if is_critical:
        # Limiting case: λ ≈ μ

        log_numerator = -lambda_val * ages_t + torch.log(
            1 + lambda_val * (end_time - ages_t)
        )
        log_denominator = torch.log(end_time)

        # Check for numerical validity
        if not (
            np.all(torch.isfinite(log_numerator)) and torch.isfinite(log_denominator)
        ):
            return np.float64(-np.inf)

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

        if not np.any(np.sign(numerator) == np.sign(denominator)):
            return np.float64(-np.inf)

        log_numerator = torch.log(abs(numerator))
        log_denominator = torch.log(abs(denominator))

        # Check for numerical validity
        if not (
            torch.isfinite(log_numerator).all() and torch.isfinite(log_denominator)
        ):
            return np.float64(-np.inf)

    # Calculate final log density
    log_density = np.sum(log_numerator - log_denominator)

    # Final numerical check
    if not torch.isfinite(log_density):
        return np.float64(-np.inf)

    return np.float64(log_density)


def calculate_spine_ages_log_lik(
    lambda_params: torch.Tensor | float,
    mu_params: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    times: torch.Tensor | float,
) -> torch.Tensor:
    """
    Calculate log-likelihood of manuscript ages for the "spine" individual.

    Parameters
    ----------
    lambda_params : float or NDArray[np.float64]
        Birth rate(s) λ.
    mu_params : float or NDArray[np.float64]
        Death rate(s) μ.
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T).
    times : float or NDArray[np.float64]
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
    lambda_params = np.atleast_1d(np.asarray(lambda_params, dtype=np.float64)).ravel()
    mu_params = np.atleast_1d(np.asarray(mu_params, dtype=np.float64)).ravel()
    times = np.atleast_1d(np.asarray(times, dtype=np.float64)).ravel()
    ages_t = np.atleast_1d(np.asarray(ages_t, dtype=np.float64)).ravel()

    if not (lambda_params.shape == mu_params.shape == times.shape):
        raise ValueError(
            f"Array inputs must have matching shapes. "
            f"Got: lambda_params shape={lambda_params.shape}, "
            f"mu_params shape={mu_params.shape}, times shape={times.shape}"
        )

    end_time = times[-1]

    if np.any(np.diff(np.insert(times, 0, 0.0)) <= 0):
        raise ValueError(f"times must be strictly increasing. Got times: {times}")
    # Input validation
    if np.any(ages_t < 0) or np.any(ages_t > end_time):
        raise ValueError(f"ages_t must be in [0, {end_time}]. Got ages_t: {ages_t}")

    if len(lambda_params) != len(mu_params):
        raise ValueError(
            f"lambda_params and mu_params must have the same length. "
            f"Got lengths: lambda_param={len(lambda_params)}, "
            f"mu_param={len(mu_params)}"
        )

    if len(lambda_params) != len(times):
        raise ValueError(
            f"lambda_params and times must have the same length. "
            f"Got lengths: lambda_param={len(lambda_params)}, times={len(times)}"
        )

    if len(times) > 1:
        inverted_times = np.flip(end_time - np.insert(times, 0, 0.0)[:-1])
        reflected_birth_rates = np.flip(mu_params)
        reflected_death_rates = np.flip(lambda_params)
        return calculate_log_lik_initial_lifetimes_given_non_extinction(
            lifetimes=ages_t,
            birth_rates=reflected_birth_rates,
            death_rates=reflected_death_rates,
            times=inverted_times,
        )

    # Handle parameter scaling
    # Extract scalar values from single-element arrays (atleast_1d guarantees arrays)
    lambda_scalar = float(lambda_params[0])
    mu_scalar = float(mu_params[0])

    # Validate positive parameters
    if lambda_scalar <= 0 or mu_scalar < 0:
        return np.float64(-np.inf)
    lambda_val = lambda_scalar
    mu_val = mu_scalar

    # Pre-compute isclose check once (avoids repeated calls in vectorized operations)
    is_critical = np.isclose(lambda_val, mu_val, rtol=1e-12)

    at_boundary = ages_t == end_time
    not_at_boundary = np.invert(at_boundary)

    if is_critical:
        # Limiting case: λ ≈ μ
        boundary_val = torch.log1p(lambda_val * end_time) - lambda_val * end_time
        log_density = np.where(
            at_boundary,
            boundary_val,
            2 * torch.log(lambda_val) + torch.log(ages_t) - lambda_val * ages_t,
        )
        return np.sum(log_density, dtype=np.float64)

    # General case: λ ≠ μ

    # Initialize arrays for numerator and denominator
    log_numerator = np.zeros_like(ages_t, dtype=np.float64)
    denominator = lambda_val - mu_val
    sign_denominator = np.sign(denominator)
    log_denominator = torch.log(np.abs(denominator))

    if np.any(at_boundary):
        numerator_boundary = lambda_val * torch.exp(
            -mu_val * end_time
        ) - mu_val * torch.exp(-lambda_val * end_time)

        if not np.all(np.sign(numerator_boundary) == sign_denominator):
            return np.float64(-np.inf)

        log_numerator[at_boundary] = torch.log(np.abs(numerator_boundary))

    if np.any(not_at_boundary):
        numerator_1 = torch.exp(-mu_val * end_time) - torch.exp(-lambda_val * end_time)

        if not np.all(np.sign(numerator_1) == sign_denominator):
            return np.float64(-np.inf)

        log_numerator[not_at_boundary] = (
            torch.log(lambda_val) + torch.log(mu_val) + torch.log(np.abs(numerator_1))
        )

    # Check for numerical validity
    if not (np.all(torch.isfinite(log_numerator)) and torch.isfinite(log_denominator)):
        return np.float64(-np.inf)

    # Calculate final log density (sum across all ages)
    log_density = np.sum(log_numerator - log_denominator)

    # Final numerical check
    if not torch.isfinite(log_density):
        return np.float64(-np.inf)

    return np.float64(log_density)


def calculate_ages_log_lik(
    lambda_params: torch.Tensor | float,
    mu_params: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    times: torch.Tensor | float,
    are_spines: bool = False,
) -> torch.Tensor:
    """
    Compute log-likelihood of manuscript ages under a birth-death model.

    Parameters
    ----------
    lambda_params : float or NDArray[np.float64]
        Birth rate(s) λ.
    mu_params : float or NDArray[np.float64]
        Death rate(s) μ.
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T).
    times : float or NDArray[np.float64]
        Observation time(s) T.
    are_spines : bool, optional
        If True, use spine calculation; otherwise non-spine. Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood across ages.
        dtype: torch.float64. Supports automatic differentiation.
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


def calculate_non_spine_ages_log_lik_derivatives(
    lambda_param: torch.Tensor | float,
    mu_param: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    T: torch.Tensor | float,
) -> torch.Tensor:
    """
    Calculate derivatives of log age-density for non-spine manuscripts.

    Parameters
    ----------
    lambda_param : float
        Birth rate λ.
    mu_param : float
        Death rate μ.
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T).
    T : float
        Observation time T.

    Returns
    -------
    torch.Tensor
        Tensor [∂/∂λ, ∂/∂μ] containing summed derivatives across all ages.
        Shape: (2,), dtype: torch.float64. Supports automatic differentiation.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability.
    """

    # Handle parameter scaling
    lambda_val = lambda_param
    mu_val = mu_param

    # Pre-compute isclose check once (avoids repeated calls in vectorized operations)
    is_critical = np.isclose(lambda_val, mu_val, rtol=1e-14)

    # Check for numerical stability in parameter space
    if is_critical:
        # Limiting case: λ ≈ μ
        # f(t) = [1 + λ(T-t)] / T * e^{-λt}
        # log f(t) = log(1 + λ(T-t)) - log(T) - λt
        # ∂ log f(t)/∂λ = (T-t)/(1 + λ(T-t)) - t
        d_mu = d_lambda = (T - ages_t) / (1 + lambda_val * (T - ages_t)) - ages_t
    else:
        # General case: λ ≠ μ
        # log f(t) = log(λe^{-λt} - μe^{-(λ-μ)T}e^{-μt}) - log(1 - e^{-(λ-μ)T})
        # Let N = λe^{-λt} - μe^{-(λ-μ)T}e^{-μt}
        # Let D = 1 - e^{-(λ-μ)T}

        growth_rate = lambda_val - mu_val

        # Calculate N and D
        exp_lambda_t = torch.exp(-lambda_val * ages_t)
        exp_mu_t = torch.exp(-mu_val * ages_t)
        exp_growth_T = torch.exp(-growth_rate * T)

        N = lambda_val * exp_lambda_t - mu_val * exp_growth_T * exp_mu_t
        D = 1 - exp_growth_T

        # ∂N/∂λ = e^{-λt}(1 - λt) + μT·e^{-(λ-μ)T}e^{-μt}
        dN_dlambda = (
            exp_lambda_t * (1 - lambda_val * ages_t)
            + mu_val * T * exp_growth_T * exp_mu_t
        )

        # ∂D/∂λ = T·e^{-(λ-μ)T}
        dD_dlambda = T * exp_growth_T

        # ∂ log f(t)/∂λ = (1/N)·∂N/∂λ - (1/D)·∂D/∂λ
        d_lambda = dN_dlambda / N - dD_dlambda / D

        # ∂N/∂μ = -e^{-(λ-μ)T}e^{-μt}[1 + μ(T - t)]
        dN_dmu = -exp_growth_T * exp_mu_t * (1 + mu_val * (T - ages_t))

        # ∂D/∂μ = -T·e^{-(λ-μ)T}
        dD_dmu = -T * exp_growth_T

        # ∂ log f(t)/∂μ = (1/N)·∂N/∂μ - (1/D)·∂D/∂μ
        d_mu = dN_dmu / N - dD_dmu / D

    return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)


def calculate_spine_ages_log_lik_derivatives(
    lambda_param: torch.Tensor | float,
    mu_param: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    T: torch.Tensor | float,
) -> torch.Tensor:
    """
    Calculate derivatives of log age-density for spine manuscripts.

    Parameters
    ----------
    lambda_param : float
        Birth rate λ.
    mu_param : float
        Death rate μ.
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T). Values at T use boundary formula.
    T : float
        Observation time T.

    Returns
    -------
    torch.Tensor
        Tensor [∂/∂λ, ∂/∂μ] containing summed derivatives across all ages.
        Shape: (2,), dtype: torch.float64. Supports automatic differentiation.

    Notes
    -----
    Vectorized over ages. Handles λ ≈ μ limiting case for numerical stability.
    """
    # Handle parameter scaling
    lambda_val = lambda_param
    mu_val = mu_param

    at_boundary = ages_t == T
    not_at_boundary = np.invert(at_boundary)
    d_lambda = np.zeros_like(ages_t, dtype=np.float64)
    d_mu = np.zeros_like(ages_t, dtype=np.float64)

    # Pre-compute isclose check once (avoids repeated calls in vectorized operations)
    is_critical = np.isclose(lambda_val, mu_val, rtol=1e-14)

    # Check for numerical stability in parameter space
    if is_critical:
        # Limiting case: λ ≈ μ
        # At boundary: log f(T) = 2log(λ) + log(T) - λT, so ∂/∂λ = 2/λ - T
        # Interior: log f(t) = 2log(λ) + log(t) - λt, so ∂/∂λ = 2/λ - t

        if np.any(at_boundary):
            # d/dλ = 2/λ - T
            d_mu[at_boundary] = d_lambda[at_boundary] = 2 / lambda_val - T

        if np.any(not_at_boundary):
            # d/dλ = 2/λ - t
            ages_interior = (
                ages_t[not_at_boundary] if isinstance(ages_t, np.ndarray) else ages_t
            )
            d_mu[not_at_boundary] = d_lambda[not_at_boundary] = (
                2 / lambda_val - ages_interior
            )
    else:
        # General case: λ ≠ μ
        if np.any(at_boundary):
            # d/dλ = (μ (e^(μ T) (T (λ - μ) + 1) - e^(λ T)))/((λ - μ) (λ e^(λ T) - μ e^(μ T)))
            lambda_numerator = mu_val * (
                torch.exp(mu_val * T) * (T * (lambda_val - mu_val) + 1)
                - torch.exp(lambda_val * T)
            )
            denominator = (lambda_val - mu_val) * (
                lambda_val * torch.exp(lambda_val * T) - mu_val * torch.exp(mu_val * T)
            )
            d_lambda[at_boundary] = lambda_numerator / denominator
            # d/dμ = -(λ (e^(λ T) (T (λ - μ) - 1) + e^(μ T)))/((λ - μ) (λ e^(λ T) - μ e^(μ T)))
            mu_numerator = -lambda_val * (
                torch.exp(lambda_val * T) * (T * (lambda_val - mu_val) - 1)
                + torch.exp(mu_val * T)
            )
            d_mu[at_boundary] = mu_numerator / denominator
        if np.any(not_at_boundary):
            # Extract ages not at boundary for computation
            ages_interior = (
                ages_t[not_at_boundary] if isinstance(ages_t, np.ndarray) else ages_t
            )

            # Interior: log f(t) = log(λ) + log(μ) + log(e^{-μT} - e^{-λT}) - log(λ - μ)
            # Let B = e^{-μT} - e^{-λT}

            B = torch.exp(-mu_val * T) - torch.exp(-lambda_val * T)

            # ∂ log f(t)/∂λ = 1/λ + T·e^{-λT}/B - 1/(λ-μ)
            d_lambda[not_at_boundary] = (
                1 / lambda_val
                + T * torch.exp(-lambda_val * T) / B
                - 1 / (lambda_val - mu_val)
            )

            # ∂ log f(t)/∂μ = 1/μ - T·e^{-μT}/B + 1/(λ-μ)
            d_mu[not_at_boundary] = (
                1 / mu_val - T * torch.exp(-mu_val * T) / B + 1 / (lambda_val - mu_val)
            )

    return np.array([np.sum(d_lambda), np.sum(d_mu)], dtype=np.float64)


def calculate_ages_log_lik_derivatives(
    lambda_param: torch.Tensor | float,
    mu_param: torch.Tensor | float,
    ages_t: torch.Tensor | float,
    T: torch.Tensor | float,
    are_spines: bool = False,
) -> torch.Tensor:
    """
    Compute derivatives of log age-density with respect to λ and μ.

    Parameters
    ----------
    lambda_param : float
        Birth rate λ.
    mu_param : float
        Death rate μ.
    ages_t : float or NDArray[np.float64]
        Manuscript age(s) t (0 ≤ t ≤ T).
    T : float
        Observation time T.
    are_spines : bool, optional
        If True, use spine calculation; otherwise non-spine. Default False.

    Returns
    -------
    torch.Tensor
        Tensor [∂/∂λ, ∂/∂μ] containing summed derivatives across all ages.
        Shape: (2,), dtype: torch.float64. Supports automatic differentiation.
    """
    if are_spines:
        return calculate_spine_ages_log_lik_derivatives(
            lambda_param, mu_param, ages_t, T
        )
    else:
        return calculate_non_spine_ages_log_lik_derivatives(
            lambda_param, mu_param, ages_t, T
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
    birth_rates_t = torch.as_tensor(birth_rates, dtype=torch.float64)
    death_rates_t = torch.as_tensor(death_rates, dtype=torch.float64)
    times_t = torch.as_tensor(times, dtype=torch.float64)

    # Ensure 1D shape
    birth_rates_t = torch.atleast_1d(birth_rates_t).ravel()
    death_rates_t = torch.atleast_1d(death_rates_t).ravel()
    times_t = torch.atleast_1d(times_t).ravel()

    if torch.any(times_t < 0):
        raise ValueError(f"times must be non-negative. Got times: {times_t}")

    # Check monotonically increasing
    times_padded = torch.cat([torch.tensor([0.0], dtype=torch.float64), times_t])
    time_diffs = torch.diff(times_padded)
    if torch.any(time_diffs <= 0):
        raise ValueError(f"times must be strictly increasing. Got times: {times_t}")

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

    Parameters
    ----------
    lifetimes : NDArray[np.float64]
        1D array of manuscript lifetimes (shape: (n,)), each ≤ times[-1].
    birth_rates : NDArray[np.float64]
        1D array of birth rates λ per epoch (shape: (n_epochs,)).
    death_rates : NDArray[np.float64]
        1D array of death rates μ per epoch; same length as ``birth_rates``.
    times : NDArray[np.float64]
        1D, strictly increasing array of epoch end times matching rate arrays.

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
        np.sum(log_prs_extinction_given_lifetimes)
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

    Parameters
    ----------
    lifetimes : NDArray[np.float64]
        1D array of manuscript lifetimes (shape: (n,)), each ≤ times[-1].
    birth_rates : NDArray[np.float64]
        1D array of birth rates λ per epoch (shape: (n_epochs,)).
    death_rates : NDArray[np.float64]
        1D array of death rates μ per epoch; same length as ``birth_rates``.
    times : NDArray[np.float64]
        1D, strictly increasing array of epoch end times matching rate arrays.

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
        np.sum(log_prs_non_extinction_given_lifetimes)
        + log_liks_lifetimes
        - n * torch.log(-torch.expm1(log_extinction_prob))
    )
    return np.float64(result)


def calculate_log_pr_extinction_given_initial_lifetimes(
    lifetimes: torch.Tensor,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
    as_array: bool = False,
) -> torch.Tensor:
    """
    Calculate log probability of extinction given initial lifetime(s).

    Parameters
    ----------
    lifetimes : NDArray[np.float64]
        1D array of manuscript lifetimes (shape: (n,)), each ≤ times[-1].
    birth_rates : NDArray[np.float64]
        1D array of birth rates λ per epoch (shape: (n_epochs,)).
    death_rates : NDArray[np.float64]
        1D array of death rates μ per epoch; same length as ``birth_rates``.
    times : NDArray[np.float64]
        1D, strictly increasing array of epoch end times matching rate arrays.
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
    """

    if len(birth_rates) != len(death_rates) or len(birth_rates) != len(times):
        raise ValueError("Length of birth_rates, death_rates, and times must be equal.")

    birth_rates_array = birth_rates

    # Sort lifetimes to enable cumulative integral caching
    sorted_lifetimes = np.sort(lifetimes)

    if sorted_lifetimes[-1] > times[-1]:
        raise ValueError(
            "All lifetimes must be less than or equal to the final time point."
            + f" Got max lifetime: {sorted_lifetimes[-1]}, final time: {times[-1]}"
        )

    log_probs = np.zeros_like(sorted_lifetimes, dtype=np.float64)

    # Build integration intervals from sorted lifetimes
    # Intervals are [0, lifetime[0]], [lifetime[0], lifetime[1]], ...
    n_lifetimes = len(sorted_lifetimes)

    # Create interval boundaries: [0, sorted_lifetimes[0], sorted_lifetimes[1], ...]
    interval_boundaries = np.insert(sorted_lifetimes, 0, 0.0)

    # Create intervals: each row is [start, end]
    intervals = np.column_stack(
        [
            interval_boundaries[
                :-1
            ],  # Start points: [0, lifetime[0], lifetime[1], ...]
            interval_boundaries[1:],  # End points: [lifetime[0], lifetime[1], ...]
        ]
    )

    # Remove zero-width intervals (can occur with duplicate lifetimes)
    nonzero_mask = intervals[:, 1] > intervals[:, 0]
    active_intervals = intervals[nonzero_mask]

    if len(active_intervals) > 0:
        # Use scipy's fixed_quad for 15-point Gauss-Legendre quadrature
        interval_integrals = np.zeros(len(active_intervals), dtype=np.float64)

        for i, (a, b) in enumerate(active_intervals):
            result, _ = scipy.integrate.fixed_quad(
                _integrand_surv_from_weighted_birth_time_vectorized,
                a,
                b,
                args=(birth_rates, death_rates, times),
                n=5,
            )
            interval_integrals[i] = result

        # Map integrals back to full intervals array (including zero-width ones)
        full_interval_integrals = np.zeros(len(intervals), dtype=np.float64)
        full_interval_integrals[nonzero_mask] = interval_integrals

        # Compute cumulative integrals for each lifetime
        # The cumulative integral at lifetime[i] is the sum of all intervals up to and including i
        cumulative_integrals = np.cumsum(full_interval_integrals)
    else:
        cumulative_integrals = np.zeros(n_lifetimes, dtype=np.float64)

    # Compute cumulative birth rates for each lifetime (vectorized)
    for i, lifetime in enumerate(sorted_lifetimes):
        full_epochs = times[np.where(times < lifetime)[0]]
        num_full_epochs = len(full_epochs)
        lifetime_epochs = np.append(full_epochs, lifetime)
        if len(birth_rates_array[0 : num_full_epochs + 1]) != len(
            np.diff(np.insert(lifetime_epochs, 0, 0.0))
        ):
            raise ValueError(
                "Mismatch in lengths when computing cumulative birth rates."
                + " birth_rates_array length: "
                + f"{len(birth_rates_array[0 : num_full_epochs + 1])}, "
                + f" lifetime: {lifetime}, times: {times}.\n"
                + f" lifetime_epochs: {lifetime_epochs}"
            )
        cum_birth_rate = np.sum(
            birth_rates_array[0 : num_full_epochs + 1]
            * np.diff(np.insert(lifetime_epochs, 0, 0.0))
        )

        log_probs[i] = cumulative_integrals[i] - cum_birth_rate

    if as_array:
        return log_probs

    return np.sum(log_probs)


def calculate_log_lik_lifetimes(
    lifetimes: torch.Tensor,
    birth_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate log-likelihood of manuscript lifetime(s).

    Parameters
    ----------
    lifetimes : NDArray[np.float64]
        1D array of lifetimes (shape: (n,)), each ≤ times[-1].
    birth_rates : NDArray[np.float64]
        1D array of birth rates λ per epoch (shape: (n_epochs,)).
    times : NDArray[np.float64]
        1D, strictly increasing array of epoch end times matching ``birth_rates``.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing sum of log-likelihood values.
        dtype: torch.float64. Supports automatic differentiation.
    """

    # Convert inputs to tensors
    lifetimes_t = torch.as_tensor(lifetimes, dtype=torch.float64)
    birth_rates_t = torch.as_tensor(birth_rates, dtype=torch.float64)
    times_t = torch.as_tensor(times, dtype=torch.float64)

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
    birth_rates : NDArray[np.float64]
        1D array of birth rates λ per epoch (shape: (n_epochs,)).
    death_rates : NDArray[np.float64]
        1D array of death rates μ per epoch; same length as ``birth_rates``.
    times : NDArray[np.float64]
        1D, strictly increasing array of epoch end times (shape: (n_epochs,)).

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing geometric parameter 1 - β_t ∈ (0, 1).
        dtype: torch.float64. Supports automatic differentiation.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Returns tensor
    for full PyTorch autograd support.
    """
    epoch_durations = np.diff(np.insert(times, 0, 0.0))
    if np.any(epoch_durations < 0):
        raise ValueError("times array must be strictly increasing.")
    lambda_val_array = birth_rates
    mu_val_array = death_rates

    growth_rates = lambda_val_array - mu_val_array
    cum_growth = np.cumsum(growth_rates * epoch_durations)

    geom_sum = _pr_extinction_numerator(birth_rates, death_rates, times)
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
    birth_time : float
        Time at which birth occurred.
    birth_rates : NDArray[np.float64]
        1D array of birth rates λ per epoch (shape: (n_epochs,)).
    death_rates : NDArray[np.float64]
        1D array of death rates μ per epoch; same length as ``birth_rates``.
    times : NDArray[np.float64]
        1D, strictly increasing array of epoch end times (shape: (n_epochs,)).

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing integrand value.
        dtype: torch.float64. Supports automatic differentiation.
    """
    # Find the index of the last epoch that starts before or at start_time

    times = times - birth_time
    valid_indices = np.where(times > 0)[0]

    if len(valid_indices) == 0:
        raise ValueError("birth_time exceeds all epoch end times.")

    trunc_times = times[valid_indices]
    trunc_birth_rates = birth_rates[valid_indices]
    trunc_death_rates = death_rates[valid_indices]

    ext_prob = calculate_extinction_pr(
        birth_rates=trunc_birth_rates,
        death_rates=trunc_death_rates,
        times=trunc_times,
    )

    return np.float64(ext_prob * trunc_birth_rates[0])


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
    birth_times : NDArray[np.float64]
        Array of times at which births occurred. Shape: (n_points,)
    birth_rates : NDArray[np.float64]
        Birth rate(s) λ for each epoch. Shape: (n_epochs,)
    death_rates : NDArray[np.float64]
        Death rate(s) μ for each epoch. Shape: (n_epochs,)
    times : NDArray[np.float64]
        Epoch end time(s). Shape: (n_epochs,)

    Returns
    -------
    torch.Tensor
        1D tensor of integrand values for each birth time. Shape: (n_points,).
        dtype: torch.float64. Supports automatic differentiation.
    """

    if not (birth_rates.shape == death_rates.shape == times.shape):
        raise ValueError(
            "Length of birth_rates, death_rates, and times must be equal. "
            f"Got shapes {birth_rates.shape}, {death_rates.shape}, {times.shape}."
        )
    n_points = len(birth_times)
    results = np.zeros(n_points, dtype=np.float64)

    # For each birth time, compute the extinction probability
    # This is the main computational bottleneck - we compute extinction probabilities
    # for different time-shifted parameter sets
    for i, birth_time in enumerate(birth_times):
        shifted_times = times - birth_time
        valid_indices = np.where(shifted_times > 0)[0]

        if len(valid_indices) == 0:
            results[i] = 0.0
            continue

        trunc_times = shifted_times[valid_indices]
        trunc_birth_rates = birth_rates[valid_indices]
        trunc_death_rates = death_rates[valid_indices]

        ext_prob = calculate_extinction_pr(
            birth_rates=trunc_birth_rates,
            death_rates=trunc_death_rates,
            times=trunc_times,
        )

        results[i] = ext_prob * trunc_birth_rates[0]

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
    birth_rates : NDArray[np.float64]
        1D array of birth rates λ per epoch (shape: (n_epochs,)).
    death_rates : NDArray[np.float64]
        1D array of death rates μ per epoch; same length as ``birth_rates``.
    times : NDArray[np.float64]
        1D, strictly increasing array of epoch end times (shape: (n_epochs,)).

    Returns
    -------
    torch.Tensor
        Scalar tensor containing numerator value for extinction probability.
        Shape: (), dtype: torch.float64.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Supports automatic
    differentiation through PyTorch's autograd.
    """

    if not (birth_rates.shape == death_rates.shape == times.shape):
        raise ValueError(
            "Length of birth_rates, death_rates, and times must be equal. "
            f"Got shapes {birth_rates.shape}, {death_rates.shape}, {times.shape}."
        )

    # Convert to PyTorch tensors
    birth_rates_t = torch.as_tensor(birth_rates, dtype=torch.float64)
    death_rates_t = torch.as_tensor(death_rates, dtype=torch.float64)
    times_t = torch.as_tensor(times, dtype=torch.float64)

    # Compute epoch durations: prepend 0 to times, then take differences
    times_padded = torch.cat([torch.tensor([0.0], dtype=torch.float64), times_t])
    epoch_durations = torch.diff(times_padded)

    critical_mask = torch.isclose(birth_rates_t, death_rates_t, rtol=1e-12)
    growth_rates = birth_rates_t - death_rates_t
    cum_growth = torch.cumsum(growth_rates * epoch_durations, dim=0)
    trunc_cum_growth = torch.cat(
        [torch.tensor([0.0], dtype=torch.float64), cum_growth[:-1]]
    )

    # Stable evaluation of (1 - exp(-x)) / x via expm1 to avoid overflow/underflow
    x = growth_rates * epoch_durations
    stable_frac = -torch.expm1(-x) / torch.where(
        growth_rates == 0.0, torch.ones_like(growth_rates), growth_rates
    )
    kernel = torch.where(critical_mask, epoch_durations, stable_frac)
    pr_extinction_numerator_summand = (
        death_rates_t * torch.exp(-trunc_cum_growth) * kernel
    )

    return torch.sum(pr_extinction_numerator_summand)
