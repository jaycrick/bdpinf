"""Mathematical functions for birth-and-death process calculations.

This module contains pure mathematical functions for calculating birth-and-death
process probabilities and densities. These functions are isolated from the
main BirthDeathProcess class for better testability and modularity.
"""

from typing import NamedTuple

import scipy
import torch

from inf_bp_texts.utils import (
    ensure_1d_tensors,
    validate_matching_lengths,
    validate_matching_shapes,
)


def _validate_times_positive(times: torch.Tensor, *, error_message: str) -> None:
    if torch.any(times <= 0):
        raise ValueError(error_message)


def _validate_times_strictly_increasing(
    times: torch.Tensor,
    *,
    all_positive: bool = False,
) -> None:
    error_message = f"times must be strictly increasing. Got times: {times}"
    if all_positive:
        diffs = torch.diff(torch.cat([torch.tensor([0.0], dtype=torch.float64), times]))
    else:
        diffs = torch.diff(times)
    if torch.any(diffs <= 0):
        raise ValueError(error_message)


def _validate_ages_in_range(
    ages: torch.Tensor,
    end_time: torch.Tensor,
    *,
    error_message: str,
) -> None:
    if torch.any(ages < 0) or torch.any(ages > end_time):
        raise ValueError(error_message)


def _validate_rates_non_negative(
    birth_rates: torch.Tensor | None,
    death_rates: torch.Tensor | None,
) -> None:
    """Validate that birth and death rates are non-negative."""
    if birth_rates is not None and torch.any(birth_rates < 0):
        msg = f"birth_rates must be non-negative. Got birth_rates: {birth_rates}"
        raise ValueError(msg)
    if death_rates is not None and torch.any(death_rates < 0):
        msg = f"death_rates must be non-negative. Got death_rates: {death_rates}"
        raise ValueError(msg)


class ValidationConfig(NamedTuple):
    """Configuration for input validation."""

    validate_shapes: bool = True
    validate_lengths: bool = True
    require_times_positive: bool = True
    require_times_increasing: bool = True
    require_ages_in_range: bool = True
    require_rates_non_negative: bool = True


def _normalize_and_validate_inputs(
    birth_rates: torch.Tensor | float | None = None,
    death_rates: torch.Tensor | float | None = None,
    times: torch.Tensor | float | None = None,
    ages: torch.Tensor | float | None = None,
    *,
    config: ValidationConfig | None = None,
) -> tuple[torch.Tensor, ...]:
    # Build input list and labels
    inputs = []
    labels = []
    if birth_rates is not None:
        inputs.append(birth_rates)
        labels.append("birth_rates")
    if death_rates is not None:
        inputs.append(death_rates)
        labels.append("death_rates")
    if times is not None:
        inputs.append(times)
        labels.append("times")
    if ages is not None:
        inputs.append(ages)
        labels.append("ages")

    # Normalize to tensors
    normalized = ensure_1d_tensors(*inputs) if inputs else ()
    tensor_map = dict(zip(labels, normalized, strict=True))

    # Use default config if not provided
    if config is None:
        config = ValidationConfig()

    # Validate rates are non-negative
    if config.require_rates_non_negative:
        _validate_rates_non_negative(
            tensor_map.get("birth_rates"),
            tensor_map.get("death_rates"),
        )

    # Validate rate/time consistency
    _validate_rate_time_consistency(
        tensor_map,
        validate_lengths=config.validate_lengths,
        validate_shapes=config.validate_shapes,
    )

    # Validate times
    if times is not None:
        _validate_times_properties(
            tensor_map["times"],
            require_positive=config.require_times_positive,
            require_increasing=config.require_times_increasing,
        )

    # Validate ages
    if config.require_ages_in_range and ages is not None:
        if times is None:
            msg = "times is required for age range validation."
            raise ValueError(msg)
        end_time = tensor_map["times"][-1]
        _validate_ages_in_range(
            tensor_map["ages"],
            end_time,
            error_message=(
                f"ages must be in [0, {end_time}]. Got ages: {tensor_map['ages']}"
            ),
        )

    return tuple(tensor_map[label] for label in labels)


def _validate_rate_time_consistency(
    tensor_map: dict[str, torch.Tensor],
    *,
    validate_lengths: bool = True,
    validate_shapes: bool = True,
) -> None:
    """Validate consistency between rate and time tensors."""
    rate_time_labels = [
        label
        for label in ("birth_rates", "death_rates", "times")
        if label in tensor_map
    ]
    if not validate_lengths or len(rate_time_labels) <= 1:
        return

    validate_matching_lengths(
        *(tensor_map[label] for label in rate_time_labels),
        names=tuple(rate_time_labels),
    )
    if validate_shapes:
        validate_matching_shapes(
            *(tensor_map[label] for label in rate_time_labels),
            names=tuple(rate_time_labels),
        )


def _validate_times_properties(
    times: torch.Tensor,
    *,
    require_positive: bool = True,
    require_increasing: bool = True,
) -> None:
    """Validate time tensor properties."""
    if require_positive:
        _validate_times_positive(
            times,
            error_message=f"times must be positive. Got times: {times}",
        )
    if require_increasing:
        _validate_times_strictly_increasing(
            times,
            all_positive=require_positive,
        )


def _reverse_time(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    end_time = times[-1]
    inverted_times = torch.flip(
        end_time - torch.cat([torch.tensor([0.0], dtype=torch.float64), times])[:-1],
        dims=[0],
    )
    reflected_birth_rates = torch.flip(death_rates, dims=[0])
    reflected_death_rates = torch.flip(birth_rates, dims=[0])
    return inverted_times, reflected_birth_rates, reflected_death_rates


def calculate_populations_log_lik(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ns: torch.Tensor | int,
    times: torch.Tensor | float,
    *,
    starting_population: int = 1,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    r"""Calculate log-probability of population counts under birth-death process.

    This function computes log p_n(T) for each count n in `ns` using
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
    starting_population : int
        The number of starting individuals at time 0. Default is 1.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing sum of log p_n(T) across
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
        birth_rates, death_rates, times = _normalize_and_validate_inputs(
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
    if starting_population <= 0:
        msg = "starting_population must be a positive integer."
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
        log_pr[zero_mask] = torch.log(alpha_t) * starting_population

    # Handle n > 0 case
    nonzero_mask = torch.logical_not(zero_mask)
    if torch.any(nonzero_mask):
        if starting_population == 1:
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
            n0 = int(starting_population)
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

                i_min = max(0, n0 - n_val)
                i_max = n0 - 1
                if i_min > i_max:
                    log_pr[idx] = torch.tensor(float("-inf"), dtype=torch.float64)
                    continue

                i = torch.arange(i_min, i_max + 1, dtype=torch.int64)
                i_f = i.to(torch.float64)
                n_f = torch.tensor(float(n_val), dtype=torch.float64)
                n0_f = torch.tensor(float(n0), dtype=torch.float64)

                log_binom_n0 = (
                    torch.lgamma(n0_f + 1)
                    - torch.lgamma(i_f + 1)
                    - torch.lgamma(n0_f - i_f + 1)
                )
                log_binom_second = (
                    torch.lgamma(n_f)
                    - torch.lgamma(n0_f - i_f)
                    - torch.lgamma(n_f - n0_f + i_f + 1)
                )

                n0_minus_i = (n0_f - i_f).to(torch.float64)
                n_minus_n0_plus_i = (n_f - n0_f + i_f).to(torch.float64)

                log_alpha_term = torch.where(
                    i_f == 0,
                    torch.tensor(0.0, dtype=torch.float64),
                    i_f * log_alpha,
                )
                log_beta_term = torch.where(
                    n_minus_n0_plus_i == 0,
                    torch.tensor(0.0, dtype=torch.float64),
                    n_minus_n0_plus_i * log_beta,
                )
                log_one_minus_term = torch.where(
                    n0_minus_i == 0,
                    torch.tensor(0.0, dtype=torch.float64),
                    n0_minus_i * log_one_minus,
                )

                log_terms = (
                    log_binom_n0
                    + log_binom_second
                    + log_alpha_term
                    + log_beta_term
                    + log_one_minus_term
                )
                log_pr[idx] = torch.logsumexp(log_terms, dim=0)

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
    _skip_normalization: bool = False,
) -> torch.Tensor:
    """Calculate log-likelihood of the ages of the "non-spine" individual(s).

    In the constant rate case, the density for the age *a* is:

        (λ * exp(-λa) - μ * exp(-(λ - μ)T) * exp(-μa))
        / (1 - exp(-(λ - μ)T))

    given λ ≠ μ, otherwise the "critical" case is:

        exp(-λa) * (1 + λ(T-a)) / T

    Parameters
    ----------
    birth_rates : torch.Tensor | float
        (Non-negative) birth rate(s) *λ*.
    death_rates : torch.Tensor | float
        (Non-negative) death rate(s) *μ*.
    ages : torch.Tensor | float
        Age(s) *a*. All ages must satisfy 0 ≤ *a* ≤ *T*.
    times : torch.Tensor | float
        Observation time(s) *T*. For multi-epoch case, the end of each epoch.

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
        birth_rates, death_rates, times, ages = _normalize_and_validate_inputs(
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
            return torch.tensor(float("-inf"), dtype=torch.float64)

    # Calculate final log density
    log_density = torch.sum(log_numerator - log_denominator)

    # Final numerical check
    if not torch.isfinite(log_density):
        return torch.tensor(float("-inf"), dtype=torch.float64)

    return log_density


def calculate_spine_ages_log_lik(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    """Calculate log-likelihood of the age(s) of the special "spine" individual(s).

    In the constant rate case, the density for the age a is:

        λμ(exp(-λa) - exp(-μa)) / (λ - μ) for a < T, and
        (λ * exp(-μT) - μ * exp(-λT)) / (λ - μ) for a = T.

    given λ ≠ μ, otherwise the "critical" case is:

        λ^2 * a * exp(-λa) for a < T, and
        (1 + λT) * exp(-λT) for a = T.

    Parameters
    ----------
    birth_rates : torch.Tensor | float
        (Non-negative) birth rate(s) λ.
    death_rates : torch.Tensor | float
        (Non-negative) death rate(s) μ.
    ages : torch.Tensor | float
        Age(s) a. All ages must satisfy 0 ≤ a ≤ T.
    times : torch.Tensor | float
        Observation time(s) T. For multi-epoch case, the end of each epoch.

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
        birth_rates, death_rates, times, ages = _normalize_and_validate_inputs(
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
        return torch.sum(log_density)
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
        return torch.tensor(float("-inf"), dtype=torch.float64)

    # Calculate final log density (sum across all ages)
    log_density = torch.sum(log_numerator - log_denominator)

    # Final numerical check
    if not torch.isfinite(log_density):
        return torch.tensor(float("-inf"), dtype=torch.float64)

    return log_density


def calculate_ages_log_lik(
    birth_rates: torch.Tensor | float,
    death_rates: torch.Tensor | float,
    ages: torch.Tensor | float,
    times: torch.Tensor | float,
    *,
    are_spines: bool = False,
    _skip_normalization: bool = False,
) -> torch.Tensor:
    """Compute log-likelihood of manuscript ages under a birth-death model.

    Dispatches to spine or non-spine likelihood calculation based on ``are_spines``.
    All operations preserve PyTorch's computational graph for automatic differentiation.

    Parameters
    ----------
    birth_rates : torch.Tensor | float
        Birth rate(s) λ.
        Scalar for single epoch, 1D tensor for piecewise-constant rates.
        All values must be positive. Will be converted to torch.float64.
    death_rates : torch.Tensor | float
        Death rate(s) μ.
        Scalar for single epoch, 1D tensor for piecewise-constant rates.
        All values must be positive. Will be converted to torch.float64.
    ages : torch.Tensor | float
        Manuscript age(s) t. Scalar or 1D tensor. All ages must satisfy 0 ≤ t ≤ T.
        Will be converted to torch.float64.
    times : torch.Tensor | float
        Observation time(s) T. Scalar for single epoch, 1D tensor of strictly increasing
        epoch end times for piecewise-constant model.
        Will be converted to torch.float64.
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
        or ages fall outside [0, *T*].

    Notes
    -----
    - Handles λ ≈ μ limiting case for numerical stability using log1p/expm1.
    - Supports both single-epoch (scalar)
    and multi-epoch (array) piecewise-constant rates.
    - Multi-epoch cases use time-reversal transformation internally.
    - All operations maintain PyTorch's computational graph for gradient computation.
    - Internally delegates to ``calculate_spine_ages_log_lik`` or
      ``calculate_non_spine_ages_log_lik`` depending on ``are_spines`` flag.

    Examples
    --------
    Single-epoch case with gradient computation:

    >>> birth_rate_val = torch.tensor(0.1, requires_grad=True)
    >>> death_rate_val = torch.tensor(0.05, requires_grad=True)
    >>> ages = torch.tensor([10.0, 20.0, 30.0])
    >>> T = 100.0
    >>> log_lik = calculate_ages_log_lik(
    ...     birth_rate_val, death_rate_val, ages, T, are_spines=False
    ... )
    >>> log_lik.backward()  # Compute gradients w.r.t. birth_rate_val and death_rate_val
    >>> print(birth_rate_val.grad, death_rate_val.grad)

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
    """Calculate extinction probability for a birth-death process *Z* at time *T*.

        Pr(Z_T = 0).

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
        If True, return log(Pr(Z_T = 0)) instead of Pr(Z_T = 0). Default False.

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing extinction probability Pr(Z_T = 0)
        (or log(Pr(Z_T = 0)) if return_log_prob=True). dtype: torch.float64.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times = _normalize_and_validate_inputs(
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

    Returns
    -------
    torch.Tensor
        Scalar tensor (shape ()) containing summed log-likelihood values.
        dtype: torch.float64.

    """
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
    _skip_normalization: bool = False,
) -> torch.Tensor:
    """Calculate log-likelihood of initial lifetime(s) given non-extinction.

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
        in torch.float64.

    """
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
    """Calculate log probability of extinction given initial lifetime(s).

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


    Notes
    -----
    Uses adaptive quadrature (scipy.integrate.quad) for numerical integration.
    Handles multi-epoch piecewise-constant rates with cumulative integral caching
    for efficiency.

    """
    if not _skip_normalization:
        birth_rates, death_rates, times, lifetimes = _normalize_and_validate_inputs(
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
                    _integrand_surv_from_weighted_birth_time_scalar(
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
    _skip_normalization: bool = False,
) -> torch.Tensor:
    """Calculate log-likelihood of manuscript lifetime(s).

    Pr(lifetime)

    Parameters
    ----------
    lifetimes : torch.Tensor
        1D tensor of lifetimes (shape: (n,), dtype: torch.float64).
        Each lifetime ≤ times[-1].
    death_rates : torch.Tensor
        1D tensor of death rates μ per epoch (shape: (n_epochs,), dtype: torch.float64).
    times : torch.Tensor
        1D, strictly increasing tensor of epoch end times (shape: (n_epochs,),
        dtype: torch.float64).

    Returns
    -------
    torch.Tensor
        Scalar tensor containing sum of log-likelihood values.
        Shape: (), dtype: torch.float64.

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

    return torch.sum(log_liks)


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
        birth_rates, death_rates, times = _normalize_and_validate_inputs(
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
    birth_time: torch.Tensor | float,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Integrand for survival probability calculation (scalar version).

    Parameters
    ----------
    birth_time : torch.Tensor | float
        Scalar time at which birth occurred.
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
        Scalar tensor containing integrand value. Shape: (),
        dtype: torch.float64.

    """
    # Convert inputs to tensors
    birth_time, birth_rates, death_rates, times = ensure_1d_tensors(
        birth_time,
        birth_rates,
        death_rates,
        times,
    )

    # Find the index of the last epoch that starts before or at start_time
    times_shifted = times - birth_time
    valid_mask = times_shifted > 0
    valid_indices = torch.where(valid_mask)[0]

    if len(valid_indices) == 0:
        msg = "birth_time exceeds all epoch end times."
        raise ValueError(msg)

    trunc_times = times_shifted[valid_indices]
    trunc_birth_rates = birth_rates[valid_indices]
    trunc_death_rates = death_rates[valid_indices]

    ext_prob = calculate_extinction_pr(
        birth_rates=trunc_birth_rates,
        death_rates=trunc_death_rates,
        times=trunc_times,
        _skip_normalization=True,
    )

    return ext_prob * trunc_birth_rates[0]


def _integrand_surv_from_weighted_birth_time_scalar(
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
        birth_rates, death_rates, times = _normalize_and_validate_inputs(
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
