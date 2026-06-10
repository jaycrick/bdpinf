"""Population count log-likelihood functions.

Imports the extinction base layer from math_extinction.
"""

from typing import cast

import torch

from inf_bp_texts.models.math_extinction import (
    _pr_extinction_numerator,
    calculate_extinction_pr,
)
from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import ensure_1d_tensors, require_tensors


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
        require_tensors(birth_rates=birth_rates, death_rates=death_rates, times=times)

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
        require_tensors(birth_rates=birth_rates, death_rates=death_rates, times=times)
        birth_rates = cast(torch.Tensor, birth_rates)
        death_rates = cast(torch.Tensor, death_rates)
        times = cast(torch.Tensor, times)
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
