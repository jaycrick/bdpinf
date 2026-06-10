"""Extinction probability and initial-lifetime likelihood functions.

Foundation layer of the mathematical model — no dependencies on other
_math_* submodules.
"""

from typing import cast

import scipy
import torch

from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import ensure_1d_tensors, require_tensors


def pr_extinction_numerator(
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
        require_tensors(birth_rates=birth_rates, death_rates=death_rates, times=times)

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
        require_tensors(birth_rates=birth_rates, death_rates=death_rates, times=times)
        birth_rates = cast(torch.Tensor, birth_rates)
        death_rates = cast(torch.Tensor, death_rates)
        times = cast(torch.Tensor, times)

    extinction_num = pr_extinction_numerator(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        _skip_normalization=True,
    )
    if return_log_prob:
        return torch.log(extinction_num) - torch.log1p(
            extinction_num,
        )

    return 1 - 1 / (1 + extinction_num)


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
        require_tensors(lifetimes=lifetimes, death_rates=death_rates, times=times)

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
        require_tensors(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            lifetimes=lifetimes,
        )
        lifetimes = cast(torch.Tensor, lifetimes)

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
        require_tensors(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            lifetimes=lifetimes,
        )

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
        require_tensors(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            lifetimes=lifetimes,
        )

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
