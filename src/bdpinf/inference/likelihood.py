"""Likelihood calculation functions for birth-death process parameter estimation.

This module contains functions for calculating log-likelihood values for
birth-death process models, used in parameter fitting and model evaluation.
"""

import warnings
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from bdpinf.models.math_nested import (
    age_log_densities,
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.models.math_populations import calculate_populations_log_lik
from bdpinf.models.validation import normalize_and_validate_inputs
from bdpinf.utils import ensure_1d_tensor, ensure_1d_tensors


class SpineResolution(NamedTuple):
    """Outcome of :func:`resolve_spine_ages`."""

    spine_ages: NDArray[np.float64] | None
    non_spine_ages: NDArray[np.float64]
    use_unknown_spine: bool


def resolve_spine_ages(
    ages: ArrayLike,
    end_time: float,
    *,
    multiple_terminal: Literal["raise", "marginalize"] = "raise",
) -> SpineResolution:
    r"""Split ``ages`` into a spine and non-spine ages when no spine is labelled.

    Host-side (not traced).
    An age ``np.isclose`` to the observation time ``T`` is the spine, and is
    *snapped to exactly* ``T`` so that it is scored with the atom
    :math:`\Pr(A^*=T)` rather than the interior density.
    A single age is the spine.
    Otherwise the spine is unknown and is marginalised over.

    Parameters
    ----------
    ages : array-like
        All observed ages, with no spine labelled.
    end_time : float
        Observation time ``T`` (``times[-1]``).
    multiple_terminal : {"raise", "marginalize"}
        What to do if more than one age is at ``T``: raise ``ValueError``
        (the model assumes a unique spine) or treat the spine as unknown.

    """
    ages_np = np.asarray(ages, dtype=np.float64)
    if ages_np.size == 0:
        return SpineResolution(None, ages_np, True)
    at_final = np.isclose(ages_np, end_time)
    n_terminal = int(at_final.sum())
    if n_terminal > 1 and multiple_terminal == "raise":
        msg = (
            "More than one age equals the final observation time. "
            "Exactly one spine is assumed; provide explicit spine_ages."
        )
        raise ValueError(msg)
    if n_terminal == 1:
        return SpineResolution(
            np.full(1, end_time, dtype=np.float64), ages_np[~at_final], False
        )
    if ages_np.size == 1:
        return SpineResolution(ages_np.copy(), ages_np[:0], False)
    return SpineResolution(None, ages_np, True)


def log_likelihood(
    birth_rates: object,
    death_rates: object,
    ages: object,
    times: object,
    *,
    spine_ages: object | None = None,
    ns: object | None = None,
    initial_population: int = 1,
    unbias: bool = False,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> jnp.ndarray:
    """Compute the log-likelihood of observed individual ages in a birth--death process.

    This function evaluates the joint log-likelihood
    for a dataset of observed individual ages
    given a birth (λ) and death (μ) rate under a birth--death model.
    The likelihood is computed
    as the sum of two contributions:

        1. The log-probability of observing the total population configuration
            at observation time T (log P_n(T)).
        2. The sum of log age-densities for the observed individuals (Σ log f(t_i)),
            with an option to treat one or more observations as "spines"
            (special conditional ages).

    Behavior and notes
    - If log_scale_params is True, birth_rates and death_rates are interpreted as log(λ)
        and log(μ); otherwise they are interpreted on the original (positive) scale.
    - If ns is None, the function assumes each provided age corresponds to
        one individual (i.e., a count of 1 per age) and constructs ns accordingly.
    - If spine_ages is None and is_ordered is False, the function treats the spine
        as unknown among the provided ages and marginalizes over all possible
        spine positions (uniformly) using a numerically stable log-sum-exp term.
    - Exception: if spine_ages is None and any observed age equals the final
        observation time, that age is treated as the spine automatically.
        If more than one age equals the final observation time, a ValueError is
        raised because the model assumes a unique spine.
    - Exception: if spine_ages is None and there is exactly one observed age,
        that individual is treated as the spine automatically.
    - If is_ordered is True, the first provided age is treated as a spine
        (unique conditional distribution)
        and the remaining ages are treated as ordinary observations.
        is_ordered and spine_ages cannot both be provided.
    - If spine_ages is provided, those ages are treated as spines;
        ordinary ages are treated as non-spine observations.
    - The function validates input ranges: rates must be finite
        (and positive if not in log scale), observation_time must be > 0,
        and all ages/spine_ages must lie in [0, T].
    - Returns -inf for numerical issues
        (non-finite rates, invalid parameter domains, overflow).
    - Raises exceptions for programming errors
        (type mismatches, array shape errors, etc.).

    Parameters
    ----------
    birth_rates : float | jax.Array
        Birth rate parameter λ > 0, or log(λ) if log_scale_params is True.
    death_rates : float | jax.Array
        Death rate parameter μ > 0, or log(μ) if log_scale_params is True.
    ages : jax.Array | float
        Observed ages of individuals. A scalar is treated as a single-element array.
        All ages must satisfy 0 <= age <= observation_time.
    times : jax.Array | float
        Observation times (> 0) at which the population/ages are observed.
    spine_ages : jax.Array | None, optional
        Ages to be treated as spines (special conditional observations). If provided,
        these ages must also satisfy 0 <= age <= observation_time and are used
        explicitly. If not provided and is_ordered=False, the spine is treated as
        unknown among the provided ages and marginalized out. Mutually exclusive
        with is_ordered=True.
    ns : jax.Array | None, optional
        Counts per population/class that sum to the total number of ages. If None,
        a single population with one count per provided age is assumed.
    initial_population: int
        Initial population size.
    is_ordered : bool, optional
        If True, the first element of ages is treated as a spine and the rest as
        ordinary observations. Cannot be used together with spine_ages. When False
        and spine_ages is None, the unknown-spine marginal likelihood is used.
    log_scale_params : bool, optional
        If True, treat birth_rates and death_rates as log-scale parameters.

    Returns
    -------
    jnp.ndarray
        The computed total log-likelihood (log P_n(T) + Σ log f(t_i)).
        Returns -inf for numerical issues
        (non-positive rates, overflow, invalid domains).

    Raises
    ------
    ValueError
        If both is_ordered=True and spine_ages is provided, or if the sum of ns does
        not equal the number of provided ages.

    See Also
    --------
        calculate_populations_log_lik

    """
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    birth_rates, death_rates, ages, times = ensure_1d_tensors(
        birth_rates,
        death_rates,
        ages,
        times,
    )

    if not (birth_rates.shape == death_rates.shape == times.shape):
        msg = (
            "birth_rates, death_rates, and times must have the same shape, "
            f"got {birth_rates.shape}, {death_rates.shape}, {times.shape}."
        )
        raise ValueError(msg)

    if log_scale_params:
        birth_rates = jnp.exp(birth_rates)
        death_rates = jnp.exp(death_rates)
    elif bool(jnp.any(birth_rates <= 0)) or bool(jnp.any(death_rates <= 0)):
        return _ninf

    if spine_ages is not None:
        spine_ages = ensure_1d_tensor(spine_ages)
        if len(spine_ages) == 0:
            spine_ages = None
        else:
            if is_ordered:
                msg = "Cannot have both is_ordered=True and spine_ages provided."
                raise ValueError(msg)
            if ns is None:
                if len(spine_ages) > 1:
                    msg = (
                        "When multiple spine ages are provided, "
                        "population sizes (ns) must also be provided."
                    )
                    raise ValueError(msg)
                ns = jnp.array([len(ages) + len(spine_ages)], dtype=jnp.float64)
            if bool(jnp.any(spine_ages < 0)) or bool(jnp.any(spine_ages > times[-1])):
                msg = f"Spine ages must be in the range [0, {times[-1]}]."
                raise ValueError(msg)

    if float(times[0]) <= 0:
        msg = "Observation times must be positive."
        raise ValueError(msg)

    if bool(jnp.any(jnp.diff(times) < 0)):
        msg = "Observation times must be non-decreasing."
        raise ValueError(msg)

    if bool(jnp.any(ages < 0)) or bool(jnp.any(ages > times[-1])):
        msg = f"Ages must be in the range [0, {float(times[-1])}]."
        raise ValueError(msg)

    num_ages = len(ages) + (len(spine_ages) if spine_ages is not None else 0)
    if ns is not None:
        ns_arr = ensure_1d_tensor(ns)
        total_pop_size = int(float(ns_arr.sum()))
        if total_pop_size < num_ages:
            msg = "Sum of the populations must equal the number of ages provided."
            raise ValueError(msg)
        if total_pop_size > num_ages:
            warnings.warn(
                "Sum of the populations exceeds the number of ages provided; "
                "not all ages have been provided (age information is partial).",
                UserWarning,
                stacklevel=2,
            )
    else:
        ns_arr = jnp.array([num_ages], dtype=jnp.float64)
        total_pop_size = num_ages

    if not (
        bool(jnp.all(jnp.isfinite(birth_rates)))
        and bool(jnp.all(jnp.isfinite(death_rates)))
    ):
        return _ninf

    if spine_ages is None and not is_ordered and len(ages) > 0:
        resolved = resolve_spine_ages(ages, float(times[-1]))
        if resolved.spine_ages is not None:
            spine_ages = jnp.asarray(resolved.spine_ages)
            ages = jnp.asarray(resolved.non_spine_ages)

    use_unknown_spine_marginal = (not is_ordered) and (spine_ages is None)

    if is_ordered:
        spine_ages = ages[:1]
        ages = ages[1:]

    # Delegate the population + age-likelihood math to the shared core (see
    # _log_likelihood_core).  skip_normalization=False because this function
    # is never traced under jax.jit, so the math kernels' own
    # normalize_and_validate_inputs call is safe to run.
    return _log_likelihood_core(
        birth_rates,
        death_rates,
        non_spine_ages=ages,
        spine_ages=spine_ages,
        times=times,
        ns=ns_arr,
        initial_population=initial_population,
        unbias=unbias,
        use_unknown_spine_marginal=use_unknown_spine_marginal,
        skip_normalization=False,
    )


def _log_likelihood_core(
    birth_rates: jnp.ndarray,
    death_rates: jnp.ndarray,
    non_spine_ages: jnp.ndarray,
    spine_ages: jnp.ndarray | None,
    times: jnp.ndarray,
    ns: jnp.ndarray,
    *,
    initial_population: int = 1,
    unbias: bool = False,
    use_unknown_spine_marginal: bool = True,
    skip_normalization: bool = True,
) -> jnp.ndarray:
    """Shared log-likelihood core, used by both ``log_likelihood`` and the MLE hot path.

    All host-side validation and spine auto-detection (auto-detecting an
    unknown spine, splitting ``is_ordered`` inputs, etc.) must be completed by
    the caller before invoking this function — this function itself contains
    no host-side hazards (``bool(traced)``, ``np.isclose(traced)``, dynamic
    mask slicing).

    All JAX-branching on ``use_unknown_spine_marginal``, ``spine_ages is not
    None``, and array lengths uses Python-level conditionals that resolve at
    XLA trace time because these flags and shapes are static closure constants
    whenever this function is called under ``jax.jit`` (the MLE hot path, via
    ``_jittable_nll``).  Non-finite outputs are handled with ``jnp.where``
    instead of early ``bool()``-guarded returns, so the function is safe to
    call both inside and outside ``jax.jit``.

    Parameters
    ----------
    birth_rates, death_rates : jnp.ndarray
        Positive-scale rates (NOT log-scale), shape ``(n_epochs,)`` float64.
    non_spine_ages : jnp.ndarray
        Non-spine (or all, when marginalizing) ages, shape ``(m,)`` float64.
    spine_ages : jnp.ndarray | None
        Explicit spine ages if known, or ``None`` when ``use_unknown_spine_marginal``
        is True.
    times : jnp.ndarray
        Epoch end times, shape ``(n_epochs,)`` float64.
    ns : jnp.ndarray
        Population counts, shape ``(k,)`` float64.  Passed directly to
        ``calculate_populations_log_lik``; no host-side sum is computed here
        (closure JAX arrays are abstract inside ``jax.jit``).
    initial_population : int
        Initial population size (static, Python int).
    unbias : bool
        Condition on non-extinction for positive counts (static, Python bool).
    use_unknown_spine_marginal : bool
        If True, ``non_spine_ages`` contains all ages and the spine position
        is marginalized via logsumexp.  Static Python bool resolved at trace
        time.
    skip_normalization : bool
        Forwarded as ``_skip_normalization`` to the math kernels.  ``True``
        (the MLE hot path) skips ``normalize_and_validate_inputs``, which uses
        Python-level ``float()``/``bool()`` calls that are unsafe under
        ``jax.jit`` — the caller is responsible for having already validated
        and normalized its inputs.  ``False`` (``log_likelihood``, never
        traced) lets the math kernels re-validate.

    Returns
    -------
    jnp.ndarray
        Scalar float64 log-likelihood, or ``-inf`` for invalid parameters.

    """
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    log_P_n_T = calculate_populations_log_lik(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
        ns=ns,
        initial_population=initial_population,
        unbias=unbias,
        _skip_normalization=skip_normalization,
    )

    # NOTE: do NOT call int(float(ns.sum())) here — under jax.jit, closure
    # JAX arrays are abstract tracers.  The n_non_spine == 0 guard below
    # makes an early-return redundant: when there are no ages, ages_log_lik
    # stays 0.0 and the final jnp.where handles any -inf from log_P_n_T.
    ages_log_lik = jnp.float64(0.0)
    n_non_spine = len(non_spine_ages)  # static shape

    if skip_normalization is False:
        # Eager callers (``log_likelihood``) keep the shared input validation.
        birth_rates, death_rates, times = normalize_and_validate_inputs(
            birth_rates, death_rates, times
        )
    terms = nested_terms(birth_rates, death_rates, times)

    if use_unknown_spine_marginal and n_non_spine > 0:
        # Unknown-spine marginal (paper: P(Z_T) * sum_j f*(a_j) prod_{i!=j} f(a_i),
        # up to the 1/n for the uniformly chosen spine): every age is a spine
        # candidate; both densities come from one shared pass.
        log_f, log_f_spine = age_log_densities(terms, non_spine_ages)
        a = log_f_spine - log_f
        a_max = jax.lax.stop_gradient(jnp.max(a))
        log_sum_exp = jnp.log(jnp.sum(jnp.exp(a - a_max))) + a_max
        ages_log_lik = log_f.sum() - jnp.log(jnp.float64(n_non_spine)) + log_sum_exp
    else:
        # Known or pre-resolved spine.
        if spine_ages is not None:
            ages_log_lik = ages_log_lik + spine_log_density(terms, spine_ages).sum()
        if n_non_spine > 0:
            ages_log_lik = (
                ages_log_lik + non_spine_log_density(terms, non_spine_ages).sum()
            )

    total = log_P_n_T + ages_log_lik
    return jnp.where(jnp.isfinite(total), total, _ninf)


def calculate_unknown_spine_mixture_age_density(
    birth_rates: object,
    death_rates: object,
    age_grid: object,
    times: object,
    num_individuals: int,
) -> jnp.ndarray:
    """Compute age density for a random individual when the spine is unknown.

    Assumes exactly one spine among ``num_individuals`` observations and uses
    the mixture

        (1 / n) * ((n - 1) * f_non_spine(a) + f_spine(a)).

    This helper is useful for plotting theoretical age distributions under the
    same unknown-spine assumption used by the likelihood.

    Parameters
    ----------
    birth_rates : jax.Array | float
        Birth rate(s) λ.
    death_rates : jax.Array | float
        Death rate(s) μ.
    age_grid : jax.Array | float
        Age grid values where the density is evaluated.
    times : jax.Array | float
        Observation time(s). Scalar for single epoch or 1D epoch end times.
    num_individuals : int
        Number of individuals ``n`` in the mixture formula. Must be positive.

    Returns
    -------
    jnp.ndarray
        Density evaluated at each point in ``age_grid`` (shape matches grid).

    Raises
    ------
    ValueError
        If ``num_individuals`` is not positive.

    """
    if num_individuals <= 0:
        msg = "num_individuals must be a positive integer."
        raise ValueError(msg)

    birth_rates, death_rates, age_grid, times = ensure_1d_tensors(
        birth_rates,
        death_rates,
        age_grid,
        times,
    )

    log_n = jnp.log(jnp.float64(num_individuals))
    if num_individuals > 1:
        log_n_minus_one = jnp.log(jnp.float64(num_individuals - 1))
    else:
        log_n_minus_one = jnp.array(-jnp.inf, dtype=jnp.float64)

    log_non_spine, log_spine = age_log_densities(
        nested_terms(birth_rates, death_rates, times), age_grid
    )
    ab = jnp.stack([log_n_minus_one + log_non_spine, log_spine])
    return jnp.exp(jax.scipy.special.logsumexp(ab, axis=0) - log_n)


def calculate_unknown_spine_mixture_birth_density(
    birth_rates: object,
    death_rates: object,
    birth_time_grid: object,
    times: object,
    num_individuals: int,
) -> jnp.ndarray:
    """Compute birth-time density for a random individual when the spine is unknown.

    This transforms the age-density mixture to a density over birth times by the
    change of variables birth_time = T - age, where T = times[-1]. The Jacobian
    is 1 in absolute value, so the birth-time density at b is the age density at
    age = T - b evaluated with the same mixture formula.

    Parameters
    ----------
    birth_rates, death_rates, birth_time_grid, times, num_individuals
        Same semantics as :func:`calculate_unknown_spine_mixture_age_density`,
        except that ``birth_time_grid`` is interpreted as times since the
        model `start_time` (i.e., in the same coordinate system as ``times``).

    Returns
    -------
    jnp.ndarray
        Density evaluated at each point in ``birth_time_grid`` (shape matches grid).

    """
    if num_individuals <= 0:
        msg = "num_individuals must be a positive integer."
        raise ValueError(msg)

    birth_rates, death_rates, birth_time_grid, times = ensure_1d_tensors(
        birth_rates,
        death_rates,
        birth_time_grid,
        times,
    )

    T = times[-1]

    if bool(jnp.any(birth_time_grid < 0)) or bool(jnp.any(birth_time_grid > T)):
        msg = f"Birth times must be in the range [0, {T}]. Got: {birth_time_grid}"
        raise ValueError(msg)

    age_grid = T - birth_time_grid

    return calculate_unknown_spine_mixture_age_density(
        birth_rates=birth_rates,
        death_rates=death_rates,
        age_grid=age_grid,
        times=times,
        num_individuals=num_individuals,
    )


def negative_log_likelihood(
    birth_rates: object,
    death_rates: object,
    ages: object,
    times: object,
    *,
    spine_ages: object | None = None,
    ns: object | None = None,
    initial_population: int = 1,
    unbias: bool = False,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> jnp.ndarray:
    """Compute the negative log-likelihood for use with numerical optimizers.

    This function is a thin wrapper around log_likelihood that negates its
    value so it can be minimized by optimizers.
    It expects the model's birth and death rates
    (or their logarithms, see log_scale_params).

    Parameters
    ----------
    birth_rates : float | jax.Array
        Birth rate parameter(s) λ > 0, or log(λ) if log_scale_params is True.
    death_rates : float | jax.Array
        Death rate parameter(s) μ > 0, or log(μ) if log_scale_params is True.
    ages : jax.Array | float
        Ages of observed individuals (or a single age).
    times : jax.Array | float
        Observation times (> 0) at which the population/ages are observed.
    spine_ages : jax.Array | None, optional
        Optional spine ages required by some likelihood formulations.
    ns : jax.Array | None, optional
        Optional counts/weights associated with each age.
    initial_population: int
        Initial population size.
    is_ordered : bool, default False
        If True, interpret ``ages`` as ordered-event data.
    log_scale_params : bool, default False
        If True, parameters are provided in log-space.

    Returns
    -------
    jnp.ndarray
        The negative log-likelihood (i.e. -log_likelihood).

    """
    return -log_likelihood(
        birth_rates=birth_rates,
        death_rates=death_rates,
        ages=ages,
        times=times,
        spine_ages=spine_ages,
        ns=ns,
        initial_population=initial_population,
        unbias=unbias,
        is_ordered=is_ordered,
        log_scale_params=log_scale_params,
    )
