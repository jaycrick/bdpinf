"""Extinction probability and initial-lifetime likelihood functions.

Foundation layer of the mathematical model — no dependencies on other
_math_* submodules.
"""

from typing import cast

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
    times_padded = torch.cat(
        [torch.zeros(1, dtype=torch.float64, device=times.device), times]
    )
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
            else torch.zeros(1, dtype=torch.float64, device=times.device).squeeze()
        ) + (lifetime - epoch_starts[current_epoch_idx]) * current_mu

        log_liks[i] = torch.log(current_mu) - cum_growth

    return log_liks if as_array else torch.sum(log_liks)


def _integrand_surv_from_weighted_birth_time(  # pyright: ignore[reportUnusedFunction]
    birth_time: float,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> float:
    """Integrand λ(t)·P_ext(t, T) at a single birth time t.

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


def _analytic_log_pr_extinction_per_lifetime(
    lifetimes: torch.Tensor,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    r"""Closed-form log P(offspring extinct by T | ancestor lifetime s) per lifetime.

    Replaces scipy adaptive quadrature with a pure-PyTorch analytic expression,
    enabling autograd and GPU computation.

    For birth time t in epoch k with Δ = τ_k − t remaining in that epoch, the
    extinction numerator is:

        ext_num(Δ) = μ_k * (1 − e^{−g_k Δ}) / g_k  +  e^{−g_k Δ} * tail_k

    where tail_k = extinction numerator from epoch k+1 onward.  Integrating
    λ_k * P_ext(t, T) = λ_k * ext_num / (1 + ext_num) over an epoch interval
    [a, b] yields the analytic formula:

        μ_k*(b−a) − ln[(1 + ext_num(τ_k−b)) / (1 + ext_num(τ_k−a))]

    Subtracting the birth-rate integral λ_k*(b−a) gives the contribution to
    log P per epoch: ln(1 + ext_upper) − ln(1 + ext_lower) − g_k*(b−a).

    Parameters
    ----------
    lifetimes : torch.Tensor
        1D tensor of ancestor lifetimes (shape (n,), dtype float64), each ≤ times[-1].
    birth_rates : torch.Tensor
        1D tensor of epoch birth rates (shape (n_epochs,), dtype float64).
    death_rates : torch.Tensor
        1D tensor of epoch death rates, same shape as birth_rates.
    times : torch.Tensor
        1D strictly-increasing epoch end times, same length as birth_rates.

    Returns
    -------
    torch.Tensor
        Per-lifetime log probabilities aligned with ``lifetimes`` (shape (n,)).

    """
    n_epochs = len(birth_rates)
    zero = birth_rates.new_zeros(())

    epoch_starts = torch.cat([zero.unsqueeze(0), times[:-1]])
    dur = times - epoch_starts  # epoch durations
    g = birth_rates - death_rates  # growth rates per epoch

    # Protect against g=0 division: use safe_g=1 in the denominator (masked out below)
    _atol = 1e-10
    critical = g.abs() < _atol
    safe_g = torch.where(critical, torch.ones_like(g), g)

    exp_neg_g_dur = torch.where(critical, torch.ones_like(g), torch.exp(-g * dur))
    kernel = torch.where(critical, dur, (1.0 - exp_neg_g_dur) / safe_g)

    # Build tail[k] = extinction numerator from epoch k+1 onward via backward pass.
    # tail[K-1] = 0; tail[k] = mu[k+1]*kernel[k+1] + exp(-g[k+1]*dur[k+1])*tail[k+1]
    tail_vals: list[torch.Tensor] = [zero]
    for k in range(n_epochs - 1, 0, -1):
        tail_vals.append(death_rates[k] * kernel[k] + exp_neg_g_dur[k] * tail_vals[-1])
    tail = torch.stack(list(reversed(tail_vals)))  # shape [n_epochs]

    # ext_from[k] = extinction numerator starting fresh at epoch k
    ext_from = death_rates * kernel + exp_neg_g_dur * tail

    # Per-epoch log-prob contribution for a FULL epoch k:
    # ln(1+tail[k]) - ln(1+ext_from[k]) - g[k]*dur[k]
    per_epoch = torch.log1p(tail) - torch.log1p(ext_from) - g * dur  # [n_epochs]

    # Prefix sums: prefix[k] = sum of contributions from full epochs 0..k-1
    prefix = torch.cat(
        [zero.unsqueeze(0), torch.cumsum(per_epoch, dim=0)]
    )  # [n_epochs+1]

    # Epoch index for each lifetime: first k with times[k] >= s_i
    epoch_of = torch.searchsorted(times.contiguous(), lifetimes.contiguous())
    epoch_of = epoch_of.clamp(max=n_epochs - 1)

    # Per-lifetime quantities
    remaining = times[epoch_of] - lifetimes  # Δ = τ_m - s_i
    g_lt = g[epoch_of]
    mu_lt = death_rates[epoch_of]
    tail_lt = tail[epoch_of]
    ext_from_lt = ext_from[epoch_of]
    critical_lt = critical[epoch_of]

    safe_g_lt = torch.where(critical_lt, torch.ones_like(g_lt), g_lt)
    exp_rem = torch.where(
        critical_lt, torch.ones_like(g_lt), torch.exp(-g_lt * remaining)
    )
    kernel_rem = torch.where(critical_lt, remaining, (1.0 - exp_rem) / safe_g_lt)
    ext_num_at_si = mu_lt * kernel_rem + exp_rem * tail_lt

    # Partial epoch: ln(1+ext_num_at_si) - ln(1+ext_from[m]) - g_m*(s_i-τ_{m-1})
    dur_partial = lifetimes - epoch_starts[epoch_of]
    partial = torch.log1p(ext_num_at_si) - torch.log1p(ext_from_lt) - g_lt * dur_partial

    return prefix[epoch_of] + partial


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
    Uses a closed-form analytic expression derived from the generating functional
    of a Poisson process. For each ancestor lifetime s, the probability that all
    offspring born in (0, s) eventually go extinct by T equals
    exp(∫₀ˢ λ(t) P_ext(t, T) dt − ∫₀ˢ λ(t) dt).  Both integrals over each
    piecewise-constant epoch have closed forms expressible in terms of the epoch
    extinction numerators, so no numerical quadrature is needed.  The
    implementation is fully vectorised over lifetimes, autograd-compatible, and
    device-agnostic (CPU/GPU).

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

    if lifetimes.max() > times[-1]:
        msg = (
            "All lifetimes must be less than or equal to the final time point."
            f" Got max lifetime: {lifetimes.max()}, final time: {times[-1]}"
        )
        raise ValueError(msg)

    log_probs = _analytic_log_pr_extinction_per_lifetime(
        lifetimes=lifetimes,
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
    )

    if as_array:
        return log_probs

    return log_probs.sum()


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
