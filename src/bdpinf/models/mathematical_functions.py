"""Mathematical functions for birth-and-death process calculations.

This module contains pure mathematical functions for calculating birth-and-death
process probabilities and densities. These functions are isolated from the
main BirthDeathProcess class for better testability and modularity.
"""

import scipy
import torch

from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import ensure_1d_tensors


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
        end_time - torch.cat([torch.tensor([0.0], dtype=torch.float64), times])[:-1],
        dims=[0],
    )
    reflected_birth_rates = torch.flip(death_rates, dims=[0])
    reflected_death_rates = torch.flip(birth_rates, dims=[0])
    return inverted_times, reflected_birth_rates, reflected_death_rates


def _compute_populations_log_lik_multi_start(
    ns: torch.Tensor,
    nonzero_mask: torch.Tensor,
    n0: int,
    alpha_t: torch.Tensor,
    beta_t: torch.Tensor,
) -> torch.Tensor:
    r"""Compute log probabilities for populations when initial_population > 1.

    Helper function that handles the complex combinatorial calculation for
    the case when the process starts with more than one individual.

    Parameters
    ----------
    ns : torch.Tensor
        Population count(s) (shape: (n,), dtype: torch.float64).
    nonzero_mask : torch.Tensor
        Boolean mask indicating non-zero population counts.
    n0 : int
        Initial number of individuals (initial_population).
    alpha_t : torch.Tensor
        Extinction probability $\Pr(Z_T = 0)$.
    beta_t : torch.Tensor
        Geometric parameter for survival distribution.

    Returns
    -------
    torch.Tensor
        Log probabilities for non-zero population counts (shape: (n,)).

    """
    log_pr = torch.zeros_like(ns, dtype=torch.float64)

    log_alpha = (
        torch.log(alpha_t)
        if alpha_t > 0
        else torch.tensor(float("-inf"), dtype=torch.float64)
    )
    log_beta = (
        torch.log(beta_t)
        if beta_t > 0
        else torch.tensor(float("-inf"), dtype=torch.float64)
    )
    log_one_minus_alpha = (
        torch.log1p(-alpha_t)
        if alpha_t < 1
        else torch.tensor(float("-inf"), dtype=torch.float64)
    )
    log_one_minus_beta = (
        torch.log1p(-beta_t)
        if beta_t < 1
        else torch.tensor(float("-inf"), dtype=torch.float64)
    )
    log_one_minus = log_one_minus_alpha + log_one_minus_beta

    nonzero_indices = torch.where(nonzero_mask)[0]
    for idx in nonzero_indices:
        n_val = int(ns[idx].item())
        if n_val <= 0:
            log_pr[idx] = torch.tensor(float("-inf"), dtype=torch.float64)
            continue

        j_min = max(0, n0 - n_val)
        j_max = n0 - 1
        if j_min > j_max:
            msg = (
                f"Invalid bounds for summation index j: j_min={j_min}, j_max={j_max}. "
                f"n0={n0}, n_val={n_val}"
            )
            raise ValueError(msg)

        j = torch.arange(j_min, j_max + 1, dtype=torch.int64)
        j_f = j.to(torch.float64)
        n_f = torch.tensor(float(n_val), dtype=torch.float64)
        n0_f = torch.tensor(float(n0), dtype=torch.float64)

        log_binom_n0 = (
            torch.lgamma(n0_f + 1)
            - torch.lgamma(j_f + 1)
            - torch.lgamma(n0_f - j_f + 1)
        )
        log_binom_second = (
            torch.lgamma(n_f)
            - torch.lgamma(n0_f - j_f)
            - torch.lgamma(n_f - n0_f + j_f + 1)
        )

        log_alpha_term = j_f * log_alpha
        log_beta_term = (n_f - n0_f + j_f) * log_beta
        log_one_minus_term = (n0_f - j_f) * log_one_minus

        log_terms = (
            log_binom_n0
            + log_binom_second
            + log_alpha_term
            + log_beta_term
            + log_one_minus_term
        )
        log_pr[idx] = torch.logsumexp(log_terms, dim=0)

    return log_pr


def calculate_populations_log_lik(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ns: torch.Tensor | int,
    times: torch.Tensor | float,
    *,
    initial_population: int = 1,
    unbias: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    r"""Calculate log-probability of population counts under birth-death process.

    This function computes $\log \Pr(Z_T = n)$ for each count $n$ in `ns` using
    closed-form expressions for the birth-and-death process probabilities.
    The computed log-probabilities are summed and returned as a single
    torch.Tensor. Invalid or numerically unstable inputs return -inf tensor.

    Parameters
    ----------
    birth_rates : torch.Tensor | float
        Birth rate(s) $\lambda$. Scalar for constant rate, 1D tensor for
        piecewise-constant rates.
    death_rates : torch.Tensor | float
        Death rate(s) $\mu$. Scalar for constant rate, 1D tensor for
        piecewise-constant rates.
    times : torch.Tensor | float
        Time(s) $T$. Scalar for single time point, 1D tensor of epoch end
        times. Must be greater than zero and strictly increasing.
    ns : torch.Tensor | int
        Non-negative population count(s) for which to evaluate
        log-probabilities.
    initial_population : int
        The number of initial individuals at time 0. Default is 1.
    unbias : bool
        If True, condition on non-extinction for positive counts when the
        initial population is 1 by using the geometric mass with parameter
        $1 - \beta_t$ on the positive integers.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing sum of $\log \Pr(Z_T = n)$ across
        provided `ns`. dtype: torch.float64.

    Notes
    -----
    Under the basic birth-and-death process with constant rates,
    the probability of observing $n$ individuals at time $T$
    can be expressed in terms of parameters $\alpha_t$ and $\beta_T$,
    (Tavaré, 2018).

    $$
        \alpha_T := \Pr(Z_T = 0)
        = \frac{\mu(e^{(\mu - \lambda)T} - 1)}
        {\lambda e^{(\mu - \lambda)T} - \mu}
    $$ and
    $$
        \beta_T := \frac{\lambda}{\mu} \alpha_T
    $$

    For $\lambda = \mu$, the limiting case is:

    $$\alpha_T = \beta_T = \frac{\lambda T}{1 + \lambda T}.$$

    If the initial number of individuals is $n_0$, then we have
    $$
        \Pr(Z_T = n))
        = \sum_{i = \max(0, n_0 - n)}^{n_0 - 1} \binom{n_0}{i}
        \binom{n - 1}{n_0 - i - 1}
        \alpha_T^{i} \beta_T^{n - n_0 + i}
        [(1 - \alpha_T)(1 - \beta_T)]^{n_0 - i}
    $$

    References
    ----------
    - Tavaré, S. (2018). The linear birth--death process: an inferential
    retrospective. *Advances in Applied Probability, 50*(A), 253--269.
    doi:10.1017/apr.2018.84

    """
    # Convert scalar inputs to tensors and ensure 1D arrays
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
        )
    else:
        if not torch.is_tensor(birth_rates):
            msg = "birth_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)
    (ns,) = ensure_1d_tensors(ns)
    if initial_population <= 0:
        msg = "initial_population must be a positive integer."
        raise ValueError(msg)

    # Input validation
    if torch.any(ns < 0):
        msg = f"Population counts ns must be non-negative. Got ns: {ns}"
        raise ValueError(msg)

    if len(times) > 1:
        alpha_t = calculate_extinction_pr(
            birth_rates,
            death_rates,
            times,
            _skip_normalization=True,
        )
        beta_t = 1 - _population_geometric_parameter(
            birth_rates,
            death_rates,
            times,
            _skip_normalization=True,
        )
    else:
        # Single-epoch case
        birth_rate_val = birth_rates[0]
        death_rate_val = death_rates[0]
        time_val = times[0]

        if torch.isclose(birth_rate_val, death_rate_val, rtol=1e-12):
            # Limiting case: λ ≈ μ
            avg_rate = (birth_rate_val + death_rate_val) / 2
            rate_time_product = avg_rate * time_val
            alpha_t = beta_t = rate_time_product / (1 + rate_time_product)
        else:
            # General case: λ ≠ μ
            growth_rate = birth_rate_val - death_rate_val
            exponent = growth_rate * time_val
            exp_term = torch.exp(exponent)

            if not torch.isfinite(exp_term):
                return torch.tensor(float("-inf"), dtype=torch.float64)

            denominator = birth_rate_val * exp_term - death_rate_val

            # Guard against division by zero
            if death_rate_val == 0.0:
                alpha_t = torch.tensor(0.0, dtype=torch.float64)
                beta_t = birth_rate_val * (exp_term - 1) / denominator
            else:
                alpha_t = death_rate_val * (exp_term - 1) / denominator
                beta_t = (birth_rate_val / death_rate_val) * alpha_t

    # Validate alpha_t and beta_t are in valid range
    if not (0 <= alpha_t <= 1):
        msg = f"Computed alpha_t out of bounds [0, 1]: alpha_t={alpha_t}"
        raise ValueError(msg)
    if not (0 <= beta_t <= 1):
        msg = f"Computed β_t out of bounds [0, 1]: β_t={beta_t}"
        raise ValueError(msg)
    if beta_t == 0.0 and torch.any(ns > 1):
        return torch.tensor(float("-inf"), dtype=torch.float64)

    # Initialize log probability array
    log_pr = torch.zeros_like(ns, dtype=torch.float64)

    # Handle n == 0 case
    zero_mask = ns == 0
    if torch.any(zero_mask):
        log_pr[zero_mask] = torch.log(alpha_t) * initial_population

    # Handle n > 0 case
    nonzero_mask = torch.logical_not(zero_mask)
    if torch.any(nonzero_mask):
        if unbias and initial_population == 1:
            log_one_minus_beta = torch.log1p(-beta_t)
            log_beta = torch.log(beta_t)
            log_term3 = torch.where(
                ns == 1,
                torch.tensor(0.0, dtype=torch.float64),
                (ns - 1) * log_beta,
            )

            if not (
                torch.isfinite(log_one_minus_beta)
                and torch.all(torch.isfinite(log_term3[nonzero_mask]))
            ):
                return torch.tensor(float("-inf"), dtype=torch.float64)

            log_pr[nonzero_mask] = (log_one_minus_beta + log_term3)[nonzero_mask]
        elif initial_population == 1:
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
        else:
            # Multi-start case: initial_population > 1
            log_pr = _compute_populations_log_lik_multi_start(
                ns=ns,
                nonzero_mask=nonzero_mask,
                n0=int(initial_population),
                alpha_t=alpha_t,
                beta_t=beta_t,
            )

    # Final numerical check
    if not torch.all(torch.isfinite(log_pr)):
        return torch.tensor(float("-inf"), dtype=torch.float64)

    return torch.sum(log_pr)


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
        if not torch.is_tensor(birth_rates):
            msg = "birth_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(ages):
            msg = "ages must be a torch.Tensor"
            raise ValueError(msg)
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

    if birth_rate_val == 0 or death_rate_val <= 0:
        return torch.tensor(float("-inf"), dtype=torch.float64)

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
            return torch.tensor(float("-inf"), dtype=torch.float64)

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
            return torch.tensor(float("-inf"), dtype=torch.float64)

        log_numerator = torch.log(torch.abs(numerator))
        log_denominator = torch.log(torch.abs(denominator))

        # Check for numerical validity
        if not (
            torch.all(torch.isfinite(log_numerator)) and torch.isfinite(log_denominator)
        ):
            return (
                torch.full_like(ages, float("-inf"), dtype=torch.float64)
                if return_array
                else torch.tensor(float("-inf"), dtype=torch.float64)
            )

    # Calculate final log density contributions
    log_density = log_numerator - log_denominator

    # Final numerical check
    if not torch.all(torch.isfinite(log_density)):
        return (
            torch.full_like(ages, float("-inf"), dtype=torch.float64)
            if return_array
            else torch.tensor(float("-inf"), dtype=torch.float64)
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
        if not torch.is_tensor(birth_rates):
            msg = "birth_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(ages):
            msg = "ages must be a torch.Tensor"
            raise ValueError(msg)
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

    if birth_rate < 0 or death_rate < 0:
        return torch.tensor(float("-inf"), dtype=torch.float64)

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
            return torch.tensor(float("-inf"), dtype=torch.float64)

        log_numerator[at_boundary] = torch.log(torch.abs(numerator_boundary))

    if torch.any(not_at_boundary):
        numerator_1 = torch.exp(-death_rate * ages) - torch.exp(-birth_rate * ages)

        if not torch.all(torch.sign(numerator_1) == sign_denominator):
            return torch.tensor(float("-inf"), dtype=torch.float64)

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
            else torch.tensor(float("-inf"), dtype=torch.float64)
        )

    # Calculate final log density contributions
    log_density = log_numerator - log_denominator

    # Final numerical check
    if not torch.all(torch.isfinite(log_density)):
        return (
            torch.full_like(ages, float("-inf"), dtype=torch.float64)
            if return_array
            else torch.tensor(float("-inf"), dtype=torch.float64)
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
        Manuscript age(s) $t$. Scalar or 1D tensor. All ages must satisfy $0 \leq t \leq T$.
        Will be converted to torch.float64.
    times : torch.Tensor | float
        Observation time(s) $T$. Scalar for single epoch, 1D tensor of strictly increasing
        epoch end times for piecewise-constant model.
        Will be converted to torch.float64.
    are_spines : bool, optional
        If True, computes spine likelihood (discrete mass at $T$, continuous for $0 \leq t < T$).
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
        or ages fall outside $[0, T]$.

    Notes
    -----
    - Handles $\lambda \approx \mu$ limiting case for numerical stability using ``log1p/expm1``.
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


def calculate_extinction_pr(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    return_log_prob: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    r"""Calculate extinction probability for a birth-death process $Z$ at time $T$.

        Pr(Z_T = 0).

    Parameters
    ----------
    birth_rates : torch.Tensor or float
        Birth rate(s) $\lambda$. Scalar or 1D tensor for piecewise-constant rates.
    death_rates : torch.Tensor or float
        Death rate(s) $\mu$. Scalar or 1D tensor for piecewise-constant rates.
    times : torch.Tensor or float
        Observation time(s) $T$. Scalar for single time, 1D tensor for epoch end times.
        For arrays, must be monotonically increasing.
    return_log_prob : bool, optional
        If True, return $\log(\Pr(Z_T = 0))$ instead of $\Pr(Z_T = 0)$. Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing extinction probability $\Pr(Z_T = 0)$
        (or $\log(\Pr(Z_T = 0))$ if return_log_prob=True). dtype: torch.float64.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
        )
    else:
        if not torch.is_tensor(birth_rates):
            msg = "birth_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)

    pr_extinction_numerator = _pr_extinction_numerator(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        _skip_normalization=True,
    )
    if return_log_prob:
        return torch.log(pr_extinction_numerator) - torch.log1p(
            pr_extinction_numerator,
        )

    return 1 - 1 / (1 + pr_extinction_numerator)


def calculate_log_lik_initial_lifetimes_given_extinction(
    lifetimes: torch.Tensor,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
    *,
    as_array: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    """Calculate log-likelihood of initial lifetime(s) given extinction.

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
    as_array : bool, optional
        If True, return the per-lifetime contributions aligned with
        ``lifetimes`` instead of summing them. Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood values when
        ``as_array`` is False. If ``as_array`` is True, returns per-lifetime
        contributions aligned with ``lifetimes``. dtype: torch.float64.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times, lifetimes = normalize_and_validate_inputs(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ages=lifetimes,
        )
    else:
        if not torch.is_tensor(birth_rates):
            msg = "birth_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(lifetimes):
            msg = "lifetimes must be a torch.Tensor"
            raise ValueError(msg)

    n = len(lifetimes)

    log_prs_extinction_given_lifetimes = (
        calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes=lifetimes,
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            _skip_normalization=True,
        )
    )
    log_liks_lifetimes = calculate_log_lik_lifetimes(
        lifetimes=lifetimes,
        death_rates=death_rates,
        times=times,
        as_array=as_array,
        _skip_normalization=True,
    )
    log_extinction_prob = calculate_extinction_pr(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        return_log_prob=True,
        _skip_normalization=True,
    )

    # Compute conditional log likelihood
    if as_array:
        return (
            log_prs_extinction_given_lifetimes
            + log_liks_lifetimes
            - log_extinction_prob
        )

    return (
        torch.sum(log_prs_extinction_given_lifetimes)
        + log_liks_lifetimes
        - n * log_extinction_prob
    )


def calculate_log_lik_initial_lifetimes_given_non_extinction(
    lifetimes: torch.Tensor,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
    *,
    as_array: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    r"""Calculate log-likelihood of initial lifetime(s) given non-extinction.

    This implementation uses only PyTorch operations to preserve
    autograd support end-to-end.

    Parameters
    ----------
    lifetimes : torch.Tensor
        1D tensor of manuscript lifetimes (shape: (n,)), each ≤ times[-1].
    birth_rates : torch.Tensor
        1D tensor of birth rates $\lambda$ per epoch (shape: (n_epochs,)).
    death_rates : torch.Tensor
        1D tensor of death rates $\mu$ per epoch; same length as ``birth_rates``.
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times matching rate arrays.
    as_array : bool, optional
        If True, return the per-lifetime contributions aligned with
        ``lifetimes`` instead of summing them. Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood values when
        ``as_array`` is False. If ``as_array`` is True, returns per-lifetime
        contributions aligned with ``lifetimes``. dtype: torch.float64.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times, lifetimes = normalize_and_validate_inputs(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ages=lifetimes,
        )
    else:
        if not torch.is_tensor(birth_rates):
            msg = "birth_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(lifetimes):
            msg = "lifetimes must be a torch.Tensor"
            raise ValueError(msg)

    n = len(lifetimes)

    log_prs_extinction_given_lifetimes = (
        calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes=lifetimes,
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            as_array=True,
            _skip_normalization=True,
        )
    )
    log_liks_lifetimes = calculate_log_lik_lifetimes(
        lifetimes=lifetimes,
        death_rates=death_rates,
        times=times,
        as_array=as_array,
        _skip_normalization=True,
    )
    log_extinction_prob = calculate_extinction_pr(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        return_log_prob=True,
        _skip_normalization=True,
    )

    log_prs_non_extinction_given_lifetimes = torch.log(
        -torch.expm1(log_prs_extinction_given_lifetimes),
    )

    # Compute conditional log likelihood
    if as_array:
        return (
            log_prs_non_extinction_given_lifetimes
            + log_liks_lifetimes
            - torch.log(-torch.expm1(log_extinction_prob))
        )

    return (
        torch.sum(log_prs_non_extinction_given_lifetimes)
        + log_liks_lifetimes
        - n * torch.log(-torch.expm1(log_extinction_prob))
    )


def calculate_log_pr_extinction_given_initial_lifetimes(
    lifetimes: torch.Tensor | float,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
    *,
    as_array: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    r"""Calculate log probability of extinction given initial lifetime(s).

    Parameters
    ----------
    lifetimes : torch.Tensor
        1D tensor of manuscript lifetimes (shape: (n,), dtype: torch.float64).
        Each lifetime must be ≤ times[-1].
    birth_rates : torch.Tensor
        1D tensor of birth rates $\lambda$ per epoch
        (shape: (n_epochs,), dtype: torch.float64).
    death_rates : torch.Tensor
        1D tensor of death rates $\mu$ per epoch. Same shape as ``birth_rates``
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


    Notes
    -----
    Uses adaptive quadrature (scipy.integrate.quad) for numerical integration.
    Handles multi-epoch piecewise-constant rates with cumulative integral caching
    for efficiency.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times, lifetimes = normalize_and_validate_inputs(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ages=lifetimes,
        )
    else:
        if not torch.is_tensor(birth_rates):
            msg = "birth_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(lifetimes):
            msg = "lifetimes must be a torch.Tensor"
            raise ValueError(msg)

    birth_rates_array = birth_rates

    # Sort lifetimes to enable cumulative integral caching
    sorted_lifetimes, _ = torch.sort(lifetimes)

    if sorted_lifetimes[-1] > times[-1]:
        msg = (
            "All lifetimes must be less than or equal to the final time point."
            f" Got max lifetime: {sorted_lifetimes[-1]}, final time: {times[-1]}"
        )
        raise ValueError(
            msg,
        )

    log_probs = torch.zeros_like(sorted_lifetimes, dtype=torch.float64)

    # Build integration intervals from sorted lifetimes
    n_lifetimes = len(sorted_lifetimes)

    # Create interval boundaries: [0, sorted_lifetimes[0], sorted_lifetimes[1], ...]
    interval_boundaries = torch.cat(
        [torch.tensor([0.0], dtype=torch.float64), sorted_lifetimes],
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
        # Detach tensors and move to CPU for scipy
        birth_rates_cpu = birth_rates.detach().cpu()
        death_rates_cpu = death_rates.detach().cpu()
        times_cpu = times.detach().cpu()

        interval_integrals = []
        for idx in active_indices:
            a = float(interval_starts[idx].item())
            b = float(interval_ends[idx].item())
            result, _ = scipy.integrate.quad(
                lambda t: float(
                    _integrand_surv_from_weighted_birth_time(
                        t,
                        birth_rates_cpu,
                        death_rates_cpu,
                        times_cpu,
                    ),
                ),
                a,
                b,
            )
            interval_integrals.append(result)

        # Build cumulative integrals
        interval_integral_full = torch.zeros(n_lifetimes, dtype=torch.float64)
        for i, idx in enumerate(active_indices):
            interval_integral_full[idx] = torch.tensor(
                interval_integrals[i],
                dtype=torch.float64,
            )

        cumulative_integrals = torch.cumsum(interval_integral_full, dim=0)

    # Compute cumulative birth rates for each lifetime
    for i, lifetime in enumerate(sorted_lifetimes):
        # Find epochs that end before this lifetime
        epoch_mask = times < lifetime
        num_full_epochs = int(torch.sum(epoch_mask).item())

        # Build lifetime epochs
        full_epochs = times[epoch_mask]
        lifetime_epochs = torch.cat([full_epochs, lifetime.unsqueeze(0)])

        # Compute epoch durations
        lifetime_epochs_padded = torch.cat(
            [torch.tensor([0.0], dtype=torch.float64), lifetime_epochs],
        )
        epoch_durs = torch.diff(lifetime_epochs_padded)

        # Compute cumulative birth rate
        cum_birth_rate = torch.sum(
            birth_rates_array[: num_full_epochs + 1] * epoch_durs,
        )

        log_probs[i] = cumulative_integrals[i] - cum_birth_rate

    if as_array:
        return log_probs

    return torch.sum(log_probs)


def calculate_log_lik_lifetimes(
    lifetimes: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
    *,
    as_array: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    r"""Calculate log-likelihood of manuscript lifetime(s).

    Pr(lifetime)

    Parameters
    ----------
    lifetimes : torch.Tensor
        1D tensor of lifetimes (shape: (n,), dtype: torch.float64).
        Each lifetime ≤ times[-1].
    death_rates : torch.Tensor
        1D tensor of death rates $\mu$ per epoch
        (shape: (n_epochs,), dtype: torch.float64).
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times (shape: (n_epochs,),
        dtype: torch.float64).
    as_array : bool, optional
        If True, return the per-lifetime contributions aligned with
        ``lifetimes`` instead of summing them. Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing sum of log-likelihood values when
        ``as_array`` is False. If ``as_array`` is True, returns the per-lifetime
        contributions aligned with ``lifetimes``. Shape: (), or (n,).
        dtype: torch.float64.

    """
    # Convert inputs to tensors
    if not _skip_normalization:
        lifetimes, death_rates, times = ensure_1d_tensors(lifetimes, death_rates, times)
    else:
        if not torch.is_tensor(lifetimes):
            msg = "lifetimes must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)

    death_rate_val_array = death_rates

    # Compute epoch starts and durations
    times_padded = torch.cat([torch.tensor([0.0], dtype=torch.float64), times])
    epoch_starts = times_padded[:-1]
    epoch_durations = times - epoch_starts

    # Compute cumulative growth at each epoch
    cum_growth_at_epochs = torch.cumsum(death_rate_val_array * epoch_durations, dim=0)

    log_liks = torch.zeros_like(lifetimes, dtype=torch.float64)

    for i, lifetime in enumerate(lifetimes):
        if torch.isclose(lifetime, times[-1]):
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
        current_mu = death_rate_val_array[current_epoch_idx]

        # Compute cumulative growth up to current lifetime
        cum_growth = (
            cum_growth_at_epochs[current_epoch_idx - 1]
            if current_epoch_idx > 0
            else torch.tensor(0.0, dtype=torch.float64)
        ) + (lifetime - epoch_starts[current_epoch_idx]) * current_mu

        log_liks[i] = torch.log(current_mu) - cum_growth

    return log_liks if as_array else torch.sum(log_liks)


def _population_geometric_parameter(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
    *,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    """Calculate geometric parameter (1 - β_t) for offspring distribution.

    Private helper function for computing the conditional geometric parameter
    of the population distribution given survival.

    Parameters
    ----------
    birth_rates : torch.Tensor
        1D tensor of birth rates λ per epoch (shape: (n_epochs,), dtype: torch.float64).
    death_rates : torch.Tensor
        1D tensor of death rates μ per epoch; same shape as ``birth_rates``
        (dtype: torch.float64).
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times (shape: (n_epochs,),
        dtype: torch.float64).

    Returns
    -------
    torch.Tensor
        Scalar tensor containing geometric parameter 1 - β_t ∈ (0, 1).
        Shape: (), dtype: torch.float64.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Full support for
    PyTorch's autograd differentiation.

    """
    # Convert inputs to tensors
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
        )
    else:
        if not torch.is_tensor(birth_rates):
            msg = "birth_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)

    # Compute epoch durations
    times_padded = torch.cat([torch.tensor([0.0], dtype=torch.float64), times])
    epoch_durations = torch.diff(times_padded)

    birth_rate_val_array = birth_rates
    death_rate_val_array = death_rates

    growth_rates = birth_rate_val_array - death_rate_val_array
    cum_growth = torch.cumsum(growth_rates * epoch_durations, dim=0)

    geom_sum = _pr_extinction_numerator(
        birth_rates,
        death_rates,
        times,
        _skip_normalization=True,
    )
    log_cond = -cum_growth[-1] - torch.log1p(geom_sum)
    return torch.exp(log_cond)


def _integrand_surv_from_weighted_birth_time(
    birth_time: float,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> float:
    """Scalar integrand for survival probability calculation.

    Parameters
    ----------
    birth_time : float
        Time at which birth occurred (scalar).
    birth_rates : torch.Tensor
        Birth rates λ for each epoch.
    death_rates : torch.Tensor
        Death rates μ for each epoch.
    times : torch.Tensor
        Epoch end times.

    Returns
    -------
    float
        Integrand value for this birth time.

    """
    shifted_times = times - birth_time
    valid_mask = shifted_times > 0
    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() == 0:
        return 0.0
    trunc_times = shifted_times[valid_indices]
    trunc_birth_rates = birth_rates[valid_indices]
    trunc_death_rates = death_rates[valid_indices]
    ext_prob = calculate_extinction_pr(
        birth_rates=trunc_birth_rates,
        death_rates=trunc_death_rates,
        times=trunc_times,
        _skip_normalization=True,
    )
    return float(ext_prob * trunc_birth_rates[0])


def _pr_extinction_numerator(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
    *,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    """Calculate numerator of extinction probability integral.

    Private helper function for extinction probability calculation.

    Parameters
    ----------
    birth_rates : torch.Tensor
        1D tensor of birth rates λ per epoch (shape: (n_epochs,), dtype: torch.float64).
    death_rates : torch.Tensor
        1D tensor of death rates μ per epoch; same shape as ``birth_rates``
        (dtype: torch.float64).
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times (shape: (n_epochs,),
        dtype: torch.float64).

    Returns
    -------
    torch.Tensor
        Scalar tensor containing numerator value for extinction probability.
        Shape: (), dtype: torch.float64.

    Notes
    -----
    Handles λ ≈ μ limiting case for numerical stability. Full support for
    PyTorch's autograd differentiation.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
        )
    else:
        if not torch.is_tensor(birth_rates):
            msg = "birth_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(death_rates):
            msg = "death_rates must be a torch.Tensor"
            raise ValueError(msg)
        if not torch.is_tensor(times):
            msg = "times must be a torch.Tensor"
            raise ValueError(msg)

    epoch_durations = torch.diff(times, prepend=times.new_zeros(1))
    growth_rates = birth_rates - death_rates
    x = growth_rates * epoch_durations
    critical_mask = torch.isclose(x, torch.zeros_like(x), atol=1e-10)
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
