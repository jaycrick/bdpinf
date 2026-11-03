"""Age samplers for birth-death process models.

Draws spine and non-spine ages directly from the exact age distributions
(rejection sampling against the nested-integral densities),
without simulating individual genealogies.
The forward simulators that return a ``BirthDeathProcess`` live in
:mod:`bdpinf.simulation.forward`.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from bdpinf.models.math_extinction import calculate_extinction_pr
from bdpinf.models.math_nested import (
    NestedTerms,
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.models.math_populations import population_geometric_parameter
from bdpinf.process import BirthDeathProcess

MAX_REJECTION_ATTEMPTS = 200_000


class _NonSpineSamplingContext(NamedTuple):
    birth_rate: float
    death_rate: float
    max_time: float
    denom_lambda: float
    envelope_constant: float
    terms: NestedTerms


def _as_scalar_float(value: object, *, name: str) -> float:
    try:
        scalar_value = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be convertible to a scalar float"
        raise ValueError(msg) from exc
    return scalar_value


def _draw_from_density_by_rejection(
    density_fn: Callable[[float], float],
    max_time: float,
    rng: np.random.Generator,
    *,
    grid_size: int = 2048,
    max_attempts: int = 200_000,
) -> float:
    # A dense grid gives a robust envelope estimate for smooth age densities.
    grid = np.linspace(0.0, max_time, num=grid_size, endpoint=False)
    density_vals = np.array([density_fn(float(age)) for age in grid], dtype=np.float64)

    if not np.all(np.isfinite(density_vals)):
        msg = "Density envelope contains non-finite values"
        raise RuntimeError(msg)

    envelope = 1.05 * float(np.max(density_vals))
    if envelope <= 0.0:
        msg = "Density envelope must be strictly positive"
        raise RuntimeError(msg)

    for _ in range(max_attempts):
        candidate_age = float(rng.uniform(0.0, max_time))
        candidate_density = float(density_fn(candidate_age))
        if not np.isfinite(candidate_density) or candidate_density < 0.0:
            continue
        if rng.uniform() * envelope <= candidate_density:
            return candidate_age

    msg = "Rejection sampler exhausted max_attempts without acceptance"
    raise RuntimeError(msg)


def _sample_spine_ages(
    birth_rate: float,
    death_rate: float,
    max_time: float,
    rng: np.random.Generator,
    *,
    n_samples: int,
) -> list[float]:
    if n_samples < 0:
        msg = "n_samples must be non-negative"
        raise ValueError(msg)
    if n_samples == 0:
        return []

    # P(spine_age = T): discrete boundary mass at observation time.
    is_critical = np.isclose(birth_rate, death_rate, rtol=1e-12)
    if is_critical:
        avg_rate = 0.5 * (birth_rate + death_rate)
        boundary_mass = (1.0 + avg_rate * max_time) * np.exp(-avg_rate * max_time)
    else:
        denominator = birth_rate - death_rate
        numerator = (birth_rate * np.exp(-death_rate * max_time)) - (
            death_rate * np.exp(-birth_rate * max_time)
        )
        boundary_mass = numerator / denominator
    boundary_mass = float(np.clip(boundary_mass, 0.0, 1.0))

    continuous_mass = max(1.0 - boundary_mass, np.finfo(float).tiny)

    terms = nested_terms(birth_rate, death_rate, max_time)

    def normalized_continuous_density(age: float) -> float:
        log_lik = spine_log_density(terms, age).sum()
        if not bool(jnp.isfinite(log_lik)):
            return 0.0
        density = float(jnp.exp(log_lik))
        if density < 0.0 or not np.isfinite(density):
            return 0.0
        return float(density / continuous_mass)

    sampled_ages: list[float] = []
    for _ in range(n_samples):
        if rng.uniform() < boundary_mass:
            sampled_ages.append(max_time)
            continue
        sampled_ages.append(
            _draw_from_density_by_rejection(
                density_fn=normalized_continuous_density,
                max_time=max_time,
                rng=rng,
            ),
        )

    return sampled_ages


def _build_non_spine_sampling_context(
    birth_rate: float,
    death_rate: float,
    max_time: float,
) -> _NonSpineSamplingContext:

    exp_lambda_t = np.exp(-birth_rate * max_time)
    denom_lambda = 1.0 - exp_lambda_t
    if denom_lambda <= 0.0:
        msg = "Invalid truncated-exponential proposal normalization"
        raise RuntimeError(msg)

    growth_rate = birth_rate - death_rate
    is_critical = np.isclose(growth_rate, 0.0, rtol=1e-12)
    if is_critical:
        # Critical case envelope: ((1 + λT) / T) * exp(-λa)
        proposal_scale = (1.0 + birth_rate * max_time) / max_time
    else:
        exp_growth = np.exp(-growth_rate * max_time)
        denom_growth = 1.0 - exp_growth
        if denom_growth == 0.0:
            msg = "Invalid non-spine envelope normalization"
            raise RuntimeError(msg)
        proposal_scale = (birth_rate - death_rate * exp_growth) / denom_growth

    envelope_constant = proposal_scale * (denom_lambda / birth_rate)
    if envelope_constant <= 0.0 or not np.isfinite(envelope_constant):
        msg = "Invalid non-spine envelope constant"
        raise RuntimeError(msg)

    return _NonSpineSamplingContext(
        birth_rate=birth_rate,
        death_rate=death_rate,
        max_time=max_time,
        denom_lambda=denom_lambda,
        envelope_constant=envelope_constant,
        terms=nested_terms(birth_rate, death_rate, max_time),
    )


def _sample_single_non_spine_age(
    context: _NonSpineSamplingContext,
    rng: np.random.Generator,
    *,
    max_attempts: int = MAX_REJECTION_ATTEMPTS,
) -> float:
    for _ in range(max_attempts):
        u = rng.uniform()
        candidate_age = -np.log(1.0 - u * context.denom_lambda) / context.birth_rate

        proposal_density = context.birth_rate * np.exp(
            -context.birth_rate * candidate_age,
        )
        proposal_density /= context.denom_lambda
        log_lik = non_spine_log_density(context.terms, float(candidate_age)).sum()
        if not bool(jnp.isfinite(log_lik)):
            continue
        target_density = float(jnp.exp(log_lik))
        if target_density <= 0.0 or proposal_density <= 0.0:
            continue

        accept_ratio = target_density / (context.envelope_constant * proposal_density)
        if not np.isfinite(accept_ratio) or accept_ratio <= 0.0:
            continue
        if rng.uniform() <= min(1.0, accept_ratio):
            return float(candidate_age)

    msg = "Non-spine rejection sampler exhausted max_attempts without acceptance"
    raise RuntimeError(msg)


def _sample_non_spine_ages(
    birth_rate: float,
    death_rate: float,
    max_time: float,
    rng: np.random.Generator,
    *,
    n_samples: int,
) -> list[float]:
    if n_samples < 0:
        msg = "n_samples must be non-negative"
        raise ValueError(msg)
    if n_samples == 0:
        return []

    context = _build_non_spine_sampling_context(
        birth_rate=birth_rate,
        death_rate=death_rate,
        max_time=max_time,
    )

    return [
        _sample_single_non_spine_age(context=context, rng=rng) for _ in range(n_samples)
    ]


def simulate_bdp_ages_from_distributions(
    birth_rate: object,
    death_rate: object,
    max_time: object,
    rng: np.random.Generator,
    ensure_non_extinct: bool = False,
) -> BirthDeathProcess:
    """Sample population size and individual ages from BDP-derived distributions.

    The population size draw follows the existing extinction/geometric logic.
    Conditional on a non-zero population, this function samples exactly one spine
    age and ``n-1`` non-spine ages on ``[0, T]``.

    Parameters
    ----------
    birth_rate
        Scalar birth-rate tensor.
    death_rate
        Scalar death-rate tensor.
    max_time
        Scalar observation-time tensor.
    rng
        NumPy random number generator.
    ensure_non_extinct
        If True, force a non-zero population draw by sampling
        from the conditional geometric component only.

    Returns
    -------
        BirthDeathProcess instance containing the sampled age groups.

    Raises
    ------
        ValueError: If rates or max_time are not positive, or tensors are not scalar.
        RuntimeError: If rejection sampling cannot find an accepted draw.

    """
    br = _as_scalar_float(birth_rate, name="birth_rate")
    dr = _as_scalar_float(death_rate, name="death_rate")
    mt = _as_scalar_float(max_time, name="max_time")

    if br <= 0.0 or dr <= 0.0:
        msg = "birth_rate and death_rate must be positive"
        raise ValueError(msg)
    if mt <= 0.0:
        msg = "max_time must be positive"
        raise ValueError(msg)

    _b = jnp.array([br], dtype=jnp.float64)
    _d = jnp.array([dr], dtype=jnp.float64)
    _t = jnp.array([mt], dtype=jnp.float64)

    pop_size: int
    if ensure_non_extinct:
        pop_size = rng.geometric(float(population_geometric_parameter(_b, _d, _t)))
    elif rng.uniform() < float(calculate_extinction_pr(_b, _d, _t)):
        pop_size = 0
    else:
        pop_size = rng.geometric(float(population_geometric_parameter(_b, _d, _t)))

    return _ages_at_population_size(br, dr, mt, pop_size, rng)


def _ages_at_population_size(
    br: float,
    dr: float,
    mt: float,
    pop_size: int,
    rng: np.random.Generator,
) -> BirthDeathProcess:
    """Sample one spine age and ``pop_size - 1`` non-spine ages."""
    ages: dict[str, list[float]] = {"spine": [], "non_spine": []}
    if pop_size > 0:
        ages["spine"] = _sample_spine_ages(br, dr, mt, rng=rng, n_samples=1)
        ages["non_spine"] = _sample_non_spine_ages(
            br, dr, mt, rng=rng, n_samples=pop_size - 1
        )

    return BirthDeathProcess.from_age_samples(
        spine_ages=ages["spine"],
        non_spine_ages=ages["non_spine"],
        epoch_end_times=mt,
        birth_rates=br,
        death_rates=dr,
    )


def simulate_bdp_ages_given_size(
    birth_rate: object,
    death_rate: object,
    max_time: object,
    population_size: int,
    rng: np.random.Generator,
) -> BirthDeathProcess:
    """Sample individual ages for a fixed population size.

    Like :func:`simulate_bdp_ages_from_distributions` but conditional on
    ``population_size`` instead of drawing it: one spine age and
    ``population_size - 1`` non-spine ages on ``[0, T]``.

    Parameters
    ----------
    birth_rate, death_rate, max_time
        Positive scalars.
    population_size
        Number of surviving individuals (at least 1).
    rng
        NumPy random number generator.

    Returns
    -------
        BirthDeathProcess instance containing the sampled age groups.

    Raises
    ------
        ValueError: If rates or max_time are not positive, or
            ``population_size`` is less than 1.

    """
    br = _as_scalar_float(birth_rate, name="birth_rate")
    dr = _as_scalar_float(death_rate, name="death_rate")
    mt = _as_scalar_float(max_time, name="max_time")
    if br <= 0.0 or dr <= 0.0:
        msg = "birth_rate and death_rate must be positive"
        raise ValueError(msg)
    if mt <= 0.0:
        msg = "max_time must be positive"
        raise ValueError(msg)
    if population_size < 1:
        msg = "population_size must be at least 1"
        raise ValueError(msg)
    return _ages_at_population_size(br, dr, mt, int(population_size), rng)
