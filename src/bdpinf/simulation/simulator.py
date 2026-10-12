"""Simulation functions for birth-death process models.

This module contains functions for simulating individual data from birth-death
processes and making predictions using fitted models.

Examples
--------
    Simulating individual creation and survival with constant rates:

    >>> import numpy as np
    >>> from bdpinf.simulation.simulator import simulate_birth_death_process
    >>>
    >>> # Set simulation parameters
    >>> birth_rate = 0.5
    >>> death_rate = 0.3
    >>> observation_time = 500
    >>>
    >>> # Run simulation with constant rates
    >>> records = simulate_birth_death_process(
    ...     birth_rate, death_rate, observation_time, random_seed=42
    ... )
    >>> print(f"Simulated {len(records)} surviving records")

    Simulating with time-varying rates:

    >>> # Rates change at t=200 and t=400
    >>> birth_rates = np.array([0.5, 0.3, 0.6])
    >>> death_rates = np.array([0.3, 0.4, 0.2])
    >>> times = np.array([200, 400, 600])
    >>>
    >>> records = simulate_birth_death_process(
    ...     birth_rates, death_rates, times, random_seed=42
    ... )
    >>> print(f"Simulated {len(records)} records")

    Predicting future individual counts:

    >>> from bdpinf.simulation.simulator import predict_individual_counts
    >>>
    >>> # Current individual ages
    >>> current_ages = np.array([100, 200, 300, 400])
    >>>
    >>> # Predict counts in 100 years
    >>> predicted_count = predict_individual_counts(
    ...     current_ages, birth_rate, death_rate, future_time=100
    ... )
    >>> print(f"Predicted future records: {predicted_count}")

"""

import contextlib
import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.typing import NDArray

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


def simulate_birth_death_process(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> BirthDeathProcess:
    """Simulate individual data from the birth-death process with time-varying rates.

    This creates a proper birth-death process simulation where records
    are born at rate λ and die at rate μ. The rates can change at specified
    time points. The simulation runs until the final time is reached and
    records both creation times and survival status.

    Args:
        birth_rates: Birth rate(s) (λ) for individual creation. Either a single
            float or an array of rates corresponding to time periods.
        death_rates: Death rate(s) (μ) for individual loss. Either a single
            float or an array of rates corresponding to time periods.
        times: End time(s) for the simulation. Either a single float or an array
            of time points at which rates change. Must be the same length as
            birth_rates and death_rates if arrays are provided.
        random_seed: Random seed for reproducibility

    Returns
    -------
        BirthDeathProcess instance containing the simulated individual records

    """
    # Convert inputs to arrays for uniform handling
    birth_rates_arr = np.atleast_1d(birth_rates)
    death_rates_arr = np.atleast_1d(death_rates)
    times_arr = np.atleast_1d(times)

    # Validate inputs
    if not (len(birth_rates_arr) == len(death_rates_arr) == len(times_arr)):
        msg = "birth_rates, death_rates, and times must have the same length"
        raise ValueError(
            msg,
        )

    if np.any(birth_rates_arr <= 0) or np.any(death_rates_arr <= 0):
        msg = "Birth rates and death rates must be positive"
        raise ValueError(msg)

    if np.any(times_arr <= 0):
        msg = "Times must be positive"
        raise ValueError(msg)

    if len(times_arr) > 1 and np.any(np.diff(times_arr) <= 0):
        msg = "Times must be strictly increasing"
        raise ValueError(msg)

    # Base RNG used to derive per-attempt seeds for reproducibility
    base_seed_rng = np.random.default_rng(random_seed)

    if ensure_non_extinct and max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")

    attempts_allowed = max_retries if ensure_non_extinct else 0
    # We allow (attempts_allowed + 1) attempts: initial + retries
    for _attempt in range(attempts_allowed + 1):
        # Create a per-attempt RNG so retries produce different draws but are
        # reproducible when `random_seed` is provided.
        if random_seed is None:
            rng = np.random.default_rng()
        else:
            child_seed = int(base_seed_rng.integers(0, 2**31 - 1))
            rng = np.random.default_rng(child_seed)

        # Initialize tracking variables for this attempt
        current_time = 0.0
        individuals_alive: list[tuple[float, int]] = [
            (0.0, 0),
        ]  # (creation_time, individual_id)
        all_records: list[dict[str, Any]] = [
            {
                "individual_id": 0,
                "parent_id": None,
                "creation_time": 0.0,
                "death_time": None,
                "is_spine": False,
            },
        ]  # Record of all records
        new_individual_id = 1

        # Initialize rate tracking
        period_idx = 0
        birth_rate = float(birth_rates_arr[period_idx])
        death_rate = float(death_rates_arr[period_idx])
        max_time = float(times_arr[-1])

        birth_death_rate = birth_rate + death_rate
        prop_birth = birth_rate / birth_death_rate
        n_alive = len(individuals_alive)
        total_rate = n_alive * birth_death_rate

        while current_time < max_time:
            if total_rate <= 0:
                break

            # Time to next event
            dt = rng.exponential(1.0 / total_rate)
            current_time += dt

            # Check if we've crossed into a new period and update rates
            if period_idx < len(times_arr) - 1 and current_time > times_arr[period_idx]:
                # Set current time to the period boundary
                current_time = times_arr[period_idx]
                # Update to next period's parameters
                period_idx += 1
                birth_rate = float(birth_rates_arr[period_idx])
                death_rate = float(death_rates_arr[period_idx])
                birth_death_rate = birth_rate + death_rate
                prop_birth = birth_rate / birth_death_rate
                total_rate = len(individuals_alive) * birth_death_rate
                # Continue to start of loop to generate dt with new rates
                continue

            if current_time >= max_time:
                break

            # Determine if event is birth or death
            if rng.random() < prop_birth:
                # Birth event
                new_entry = (current_time, new_individual_id)
                # Insert at a random position other than the first.
                if len(individuals_alive) == 1:
                    individuals_alive.append(new_entry)
                    parent_id = individuals_alive[0][1]
                else:
                    insert_idx = int(rng.integers(1, len(individuals_alive) + 1))
                    individuals_alive.insert(insert_idx, new_entry)
                    parent_id = individuals_alive[insert_idx - 1][1]

                all_records.append(
                    {
                        "individual_id": new_individual_id,
                        "parent_id": parent_id,
                        "creation_time": current_time,
                        "death_time": None,
                        "is_spine": False,
                    },
                )
                new_individual_id += 1
                total_rate += birth_death_rate
            else:
                # Death event
                death_idx = rng.integers(0, len(individuals_alive))
                _, dead_id = individuals_alive.pop(death_idx)

                # Update the record
                for individual in all_records:
                    if individual["individual_id"] == dead_id:
                        individual["death_time"] = current_time
                        break

                total_rate -= birth_death_rate

        # Process surviving records
        spine_id = individuals_alive[0][1] if individuals_alive else None
        if spine_id is not None:
            for individual in all_records:
                if individual["individual_id"] == spine_id:
                    individual["is_spine"] = True
                    break

        # Create DataFrame and add derived columns
        df = pd.DataFrame(all_records)

        # Add survived column (True if death_time is None)
        df["survived"] = df["death_time"].isna()

        # Add age column for surviving records (vectorized)
        df["age"] = np.where(df["survived"], max_time - df["creation_time"], np.nan)

        # Add lifespan column for dead records (vectorized)
        df["lifespan"] = np.where(
            ~df["survived"],
            df["death_time"] - df["creation_time"],
            np.nan,
        )

        # If we don't need to ensure non-extinction, return first draw.
        if not ensure_non_extinct:
            return BirthDeathProcess.from_records(
                df,
                epoch_end_times=times_arr,
                birth_rates=birth_rates_arr,
                death_rates=death_rates_arr,
            )

        # If we require non-extinction, check if at least one survived.
        if df["survived"].any():
            return BirthDeathProcess.from_records(
                df,
                epoch_end_times=times_arr,
                birth_rates=birth_rates_arr,
                death_rates=death_rates_arr,
            )

        # Otherwise, loop to retry until attempts exhausted.

    # If we reach here, retries were exhausted without producing a non-extinct draw.
    msg = (
        f"simulate_birth_death_process: exhausted {attempts_allowed} retries "
        "without producing a non-extinct population"
    )
    raise RuntimeError(msg)


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

    Args:
        birth_rate: Scalar birth-rate tensor.
        death_rate: Scalar death-rate tensor.
        max_time: Scalar observation-time tensor.
        rng: NumPy random number generator.
        ensure_non_extinct: If True, force a non-zero population draw by sampling
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


# ---------------------------------------------------------------------------
# Lifespan-at-birth simulator (JAX/JIT)
# ---------------------------------------------------------------------------
# Each individual's entire lifespan is decided at the moment of birth by
# drawing from a piecewise-exponential distribution (one rate per epoch).
# The event loop then advances by the minimum of:
#   (a) next scheduled death (smallest upcoming death time in the buffer),
#   (b) next birth        (Exp(n_alive * λ) from current time),
#   (c) next epoch boundary.
# This differs from the classical Gillespie (which draws a single competing
# exponential for all individuals) but is stochastically equivalent within
# each epoch by the memoryless property of the exponential distribution.
#
# The numeric core runs inside jax.lax.while_loop for JIT speed.
# Parent-ID genealogy and spine assignment are reconstructed on the host
# (Python/NumPy) from the birth/death times and insertion uniforms.
#
# Entry points:
#   simulate_birth_death_lifespans_arrays  – raw JAX arrays, no DataFrame
#   simulate_birth_death_lifespans         – full BirthDeathProcess (same API
#                                            as simulate_birth_death_process)
# ---------------------------------------------------------------------------


class LifespanSimulationArrays(NamedTuple):
    """Raw per-individual arrays from the lifespan-at-birth simulator.

    Attributes
    ----------
    birth_times : jax.Array
        Float64 array of shape ``(n_born,)`` — creation time of each individual.
    death_times : jax.Array
        Float64 array of shape ``(n_born,)`` — pre-sampled death time (``inf``
        when the individual survives to the terminal time).
    birth_epochs : jax.Array
        Int32 array of shape ``(n_born,)`` — epoch index in which each
        individual was born.
    survived : jax.Array
        Bool array of shape ``(n_born,)`` — ``True`` iff the individual is
        alive at the terminal time.
    n_born : int
        Total number of individuals born (including the root).

    Notes
    -----
    Number from epoch ``e`` surviving to terminal time::

        jnp.sum((birth_epochs == e) & survived)

    This is the quantity required by the multilevel-splitting estimator.

    """

    birth_times: jax.Array
    death_times: jax.Array
    birth_epochs: jax.Array
    survived: jax.Array
    n_born: int


class _LifespanLoopState(NamedTuple):
    """Mutable state carried through the JAX while_loop."""

    key: jax.Array  # PRNGKey
    current_time: jax.Array  # float64 scalar
    n_alive: jax.Array  # int32 scalar — individuals alive at current_time
    n_born: jax.Array  # int32 scalar — total ever born (= next free buffer slot)
    epoch_idx: jax.Array  # int32 scalar — current epoch index
    overflow: jax.Array  # bool scalar — True iff max_population was exceeded
    birth_time_buf: jax.Array  # float64 [max_population]
    death_time_buf: jax.Array  # float64 [max_population] (inf = survives)
    insert_unif_buf: jax.Array  # float64 [max_population] (U[0,1) for genealogy)
    birth_epoch_buf: jax.Array  # int32 [max_population]
    n_alive_at_birth_buf: jax.Array  # int32 [max_population] — n_alive before birth


def _sample_piecewise_exponential_death_time(
    birth_time: jax.Array,
    death_rates: jax.Array,
    epoch_end_times: jax.Array,
    terminal_time: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Sample one individual's death time under piecewise-constant hazard.

    Draws ``E ~ Exp(1)`` and walks the epoch boundaries starting from
    ``birth_time``, accumulating ``rate_e * duration_e`` per epoch until the
    total hazard reaches ``E`` (inversion method).  Returns ``jnp.inf`` when
    the total hazard over ``[birth_time, terminal_time]`` is less than ``E``
    (i.e. the individual survives to the terminal time).

    JIT-safe: implemented as a ``lax.scan`` over the static epoch count.

    """
    key, subkey = jax.random.split(key)
    # Draw the target cumulative hazard (exponential quantile)
    E = jax.random.exponential(subkey, dtype=jnp.float64)

    # Epoch start times: 0 for epoch 0, epoch_end_times[k-1] for epoch k
    epoch_starts = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float64), epoch_end_times[:-1]]
    )

    # Per-epoch scan: accumulate hazard; solve for death within the first
    # epoch that pushes accumulated hazard past E.
    def _scan_epoch(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        xs: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        accumulated, found, death_time = carry
        epoch_start, epoch_end, rate = xs[0], xs[1], xs[2]

        # Portion of this epoch after birth and before terminal time
        interval_start = jnp.maximum(birth_time, epoch_start)
        interval_end = jnp.minimum(epoch_end, terminal_time)
        duration = jnp.maximum(jnp.float64(0.0), interval_end - interval_start)

        hazard_this_epoch = rate * duration
        new_accumulated = accumulated + hazard_this_epoch

        # Solve: interval_start + (E - accumulated_before) / rate = death_time
        remaining = E - accumulated
        safe_rate = jnp.where(rate > 0.0, rate, jnp.float64(1.0))
        candidate_death = interval_start + remaining / safe_rate

        # Death falls in this epoch iff hazard here tips us over E (and not
        # already found in an earlier epoch, and there is a nonzero interval).
        in_epoch = (~found) & (new_accumulated >= E) & (duration > 0.0) & (rate > 0.0)
        new_found = found | in_epoch
        new_death_time = jnp.where(in_epoch, candidate_death, death_time)

        return (new_accumulated, new_found, new_death_time), None

    xs = jnp.stack([epoch_starts, epoch_end_times, death_rates], axis=1)  # [n, 3]
    (_, found, death_time), _ = lax.scan(
        _scan_epoch,
        (jnp.float64(0.0), jnp.bool_(False), jnp.inf),
        xs,
    )
    # Survives to terminal time → return inf sentinel
    return jnp.where(found, death_time, jnp.inf)


@functools.partial(jax.jit, static_argnums=(4,))
def _simulate_lifespans_core(
    birth_rates: jax.Array,  # float64 [n_epochs]
    death_rates: jax.Array,  # float64 [n_epochs]
    epoch_end_times: jax.Array,  # float64 [n_epochs]; last entry = terminal_time
    key: jax.Array,
    max_population: int,  # STATIC — determines buffer size and causes recompile
) -> _LifespanLoopState:
    """JIT-compiled lifespan-at-birth event loop.

    Uses ``jax.lax.while_loop`` to advance a single realisation of the
    birth-death process until ``terminal_time = epoch_end_times[-1]``.

    Returns the final ``_LifespanLoopState``; only the first ``n_born``
    entries of each buffer are meaningful.  The ``overflow`` flag is ``True``
    if the process would have required more than ``max_population`` individuals.

    """
    terminal_time: jax.Array = epoch_end_times[-1]
    n_epochs: int = epoch_end_times.shape[0]  # static at compile time

    # ---- Initialise with the root individual (id 0, born at t=0) ----
    key, k_root_death = jax.random.split(key)
    root_death_time = _sample_piecewise_exponential_death_time(
        birth_time=jnp.float64(0.0),
        death_rates=death_rates,
        epoch_end_times=epoch_end_times,
        terminal_time=terminal_time,
        key=k_root_death,
    )

    birth_time_buf = (
        jnp.full(max_population, jnp.inf, dtype=jnp.float64).at[0].set(jnp.float64(0.0))
    )
    death_time_buf = (
        jnp.full(max_population, jnp.inf, dtype=jnp.float64).at[0].set(root_death_time)
    )
    insert_unif_buf = jnp.zeros(max_population, dtype=jnp.float64)
    birth_epoch_buf = jnp.zeros(max_population, dtype=jnp.int32)
    n_alive_at_birth_buf = jnp.zeros(max_population, dtype=jnp.int32)  # 0 for root

    init_state = _LifespanLoopState(
        key=key,
        current_time=jnp.float64(0.0),
        n_alive=jnp.int32(1),
        n_born=jnp.int32(1),
        epoch_idx=jnp.int32(0),
        overflow=jnp.bool_(False),
        birth_time_buf=birth_time_buf,
        death_time_buf=death_time_buf,
        insert_unif_buf=insert_unif_buf,
        birth_epoch_buf=birth_epoch_buf,
        n_alive_at_birth_buf=n_alive_at_birth_buf,
    )

    # ---- Event loop ----
    def _cond(state: _LifespanLoopState) -> jax.Array:
        return (state.current_time < terminal_time) & ~state.overflow

    def _body(state: _LifespanLoopState) -> _LifespanLoopState:
        (
            key,
            current_time,
            n_alive,
            n_born,
            epoch_idx,
            overflow,
            birth_time_buf,
            death_time_buf,
            insert_unif_buf,
            birth_epoch_buf,
            n_alive_at_birth_buf,
        ) = state

        key, k_birth_dt, k_death_time, k_insert = jax.random.split(key, 4)

        # Current epoch (clamp to avoid OOB on last epoch)
        safe_idx = jnp.minimum(epoch_idx, n_epochs - 1)
        b_rate: jax.Array = birth_rates[safe_idx]
        t_epoch_end: jax.Array = epoch_end_times[safe_idx]

        # ---- Next birth time ----
        total_birth_rate = n_alive.astype(jnp.float64) * b_rate
        dt_birth = jax.random.exponential(k_birth_dt, dtype=jnp.float64) / jnp.where(
            total_birth_rate > 0.0, total_birth_rate, jnp.float64(1.0)
        )
        next_birth = jnp.where(
            total_birth_rate > 0.0,
            current_time + dt_birth,
            jnp.inf,
        )

        # ---- Next scheduled death (smallest future death time in buffer) ----
        indices = jnp.arange(max_population)
        future_death_mask = (
            (indices < n_born)
            & jnp.isfinite(death_time_buf)
            & (death_time_buf > current_time)
        )
        next_death = jnp.min(jnp.where(future_death_mask, death_time_buf, jnp.inf))

        # ---- Choose next event ----
        next_event = jnp.minimum(jnp.minimum(next_birth, next_death), t_epoch_end)

        # Event type (mutually exclusive; epoch has highest priority on exact ties)
        is_epoch = next_event >= t_epoch_end - jnp.float64(1e-12)
        is_death = (~is_epoch) & (next_death <= next_birth)
        is_birth = (~is_epoch) & (~is_death)

        # Advance time; clamp to terminal_time so the while-loop condition fires
        new_time = jnp.where(next_event >= terminal_time, terminal_time, next_event)

        # ---- Epoch crossing ----
        # Only bump epoch_idx when this is an internal boundary (< terminal_time)
        new_epoch_idx = jnp.where(
            is_epoch & (t_epoch_end < terminal_time),
            epoch_idx + 1,
            epoch_idx,
        )

        # ---- Birth event ----
        can_do_birth = is_birth & (n_born < max_population)
        new_overflow = overflow | (is_birth & (n_born >= max_population))

        # Sample this new individual's death time and insert uniform
        new_indiv_death = _sample_piecewise_exponential_death_time(
            birth_time=next_event,
            death_rates=death_rates,
            epoch_end_times=epoch_end_times,
            terminal_time=terminal_time,
            key=k_death_time,
        )
        new_insert_u = jax.random.uniform(k_insert, dtype=jnp.float64)

        # Write to buffers at index n_born; guard OOB by clamping write index
        # (when can_do_birth is False the written value is ignored by jnp.where)
        safe_write_idx = jnp.minimum(n_born, max_population - 1)
        new_birth_time_buf = jnp.where(
            can_do_birth,
            birth_time_buf.at[safe_write_idx].set(next_event),
            birth_time_buf,
        )
        new_death_time_buf = jnp.where(
            can_do_birth,
            death_time_buf.at[safe_write_idx].set(new_indiv_death),
            death_time_buf,
        )
        new_insert_unif_buf = jnp.where(
            can_do_birth,
            insert_unif_buf.at[safe_write_idx].set(new_insert_u),
            insert_unif_buf,
        )
        new_birth_epoch_buf = jnp.where(
            can_do_birth,
            birth_epoch_buf.at[safe_write_idx].set(epoch_idx),
            birth_epoch_buf,
        )
        new_n_alive_at_birth_buf = jnp.where(
            can_do_birth,
            n_alive_at_birth_buf.at[safe_write_idx].set(n_alive),
            n_alive_at_birth_buf,
        )
        new_n_born = jnp.where(can_do_birth, n_born + 1, n_born)

        # ---- Update n_alive ----
        new_n_alive = jnp.where(
            is_birth & can_do_birth,
            n_alive + 1,
            jnp.where(is_death, jnp.maximum(n_alive - 1, 0), n_alive),
        )

        return _LifespanLoopState(
            key=key,
            current_time=new_time,
            n_alive=new_n_alive,
            n_born=new_n_born,
            epoch_idx=new_epoch_idx,
            overflow=new_overflow,
            birth_time_buf=new_birth_time_buf,
            death_time_buf=new_death_time_buf,
            insert_unif_buf=new_insert_unif_buf,
            birth_epoch_buf=new_birth_epoch_buf,
            n_alive_at_birth_buf=new_n_alive_at_birth_buf,
        )

    return lax.while_loop(_cond, _body, init_state)


def _reconstruct_genealogy(
    birth_times: np.ndarray,
    death_times: np.ndarray,
    insert_uniforms: np.ndarray,
    birth_epochs: np.ndarray,
    n_alive_at_birth: np.ndarray,
    terminal_time: float,
) -> pd.DataFrame:
    """Replay the event stream to assign parent IDs and spine.

    Replicates the random-insert genealogy convention of
    ``simulate_birth_death_process``: new individuals are inserted at position
    ``1 + floor(u * n_alive)`` (clamped to ``n_alive``) in the ordered
    alive-list, with their parent being the individual at ``insert_pos - 1``.
    The spine is ``alive_list[0]`` at termination.

    Parameters
    ----------
    birth_times : np.ndarray, shape (n_born,)
    death_times : np.ndarray, shape (n_born,)
        ``inf`` for survivors.
    insert_uniforms : np.ndarray, shape (n_born,)
        Pre-sampled U[0,1) values for each individual's insertion position.
    birth_epochs : np.ndarray, shape (n_born,)
        Epoch index in which each individual was born.
    n_alive_at_birth : np.ndarray, shape (n_born,)
        Population size just before each individual's birth (from JIT state).
    terminal_time : float

    Returns
    -------
    pd.DataFrame
        Canonical eight-column records table (individual_id, parent_id,
        creation_time, death_time, is_spine, survived, age, lifespan) plus
        ``birth_epoch``.

    """
    n_born = len(birth_times)
    survived_mask = ~np.isfinite(death_times) | (death_times > terminal_time)

    # ---- Build chronological event list ----
    # Each element: (time, kind, individual_id)
    # kind: 0 = death, 1 = birth  (deaths processed before births at same time)
    events: list[tuple[float, int, int]] = []
    for i in range(n_born):
        events.append((float(birth_times[i]), 1, i))
        if np.isfinite(death_times[i]) and death_times[i] <= terminal_time:
            events.append((float(death_times[i]), 0, i))
    events.sort(key=lambda e: (e[0], e[1]))

    # ---- Replay ----
    records: list[dict[str, Any]] = []
    alive_list: list[int] = []  # ordered list of alive individual IDs
    parent_map: dict[int, int | None] = {}

    for _t, kind, i in events:
        if kind == 1:  # birth
            n = int(n_alive_at_birth[i])
            if n == 0:
                # Root individual: no parent, simply append
                parent_map[i] = None
                alive_list.append(i)
            elif len(alive_list) == 1:
                parent_map[i] = alive_list[0]
                alive_list.append(i)
            else:
                # Insert at 1 + floor(u * n), clamped to n
                insert_pos = min(1 + int(float(insert_uniforms[i]) * n), n)
                parent_map[i] = alive_list[insert_pos - 1]
                alive_list.insert(insert_pos, i)
        else:  # death
            # already removed is harmless (shouldn't happen in a valid simulation)
            with contextlib.suppress(ValueError):
                alive_list.remove(i)

    # ---- Spine = alive_list[0] at termination ----
    spine_id = alive_list[0] if alive_list else None

    # ---- Build records DataFrame ----
    for i in range(n_born):
        raw_death = float(death_times[i])
        died = np.isfinite(raw_death) and raw_death <= terminal_time
        records.append(
            {
                "individual_id": i,
                "parent_id": parent_map.get(i),
                "creation_time": float(birth_times[i]),
                "death_time": raw_death if died else None,
                "is_spine": i == spine_id,
                "survived": survived_mask[i],
                "age": (
                    terminal_time - float(birth_times[i])
                    if survived_mask[i]
                    else np.nan
                ),
                "lifespan": (raw_death - float(birth_times[i])) if died else np.nan,
                "birth_epoch": int(birth_epochs[i]),
            }
        )

    return pd.DataFrame(records)


def simulate_birth_death_lifespans_arrays(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
    *,
    max_population: int = 200_000,
) -> LifespanSimulationArrays:
    """Simulate a birth-death process and return raw per-individual arrays.

    Runs the lifespan-at-birth Gillespie algorithm via a JIT-compiled JAX
    event loop.  Each individual's death time is pre-sampled at birth from a
    piecewise-exponential distribution, so the number of individuals surviving
    from each epoch to the terminal time is directly accessible from the
    returned arrays.

    This is the low-level entry point intended for downstream analysis such
    as multilevel splitting.  For a full ``BirthDeathProcess`` object with
    genealogy use :func:`simulate_birth_death_lifespans` instead.

    Args:
        birth_rates: Per-individual birth rate(s) λ.  Scalar or 1-D array
            (one value per epoch).
        death_rates: Per-individual death rate(s) μ.  Same length as
            ``birth_rates``.
        times: Epoch end time(s).  Scalar or 1-D array strictly increasing.
            The last entry is the terminal time T.
        random_seed: Integer seed for JAX PRNG reproducibility, or ``None``
            for a nondeterministic draw.
        max_population: Maximum number of individuals that can be born.
            Raises ``RuntimeError`` on overflow.  Recompiles the JIT kernel
            whenever a new value is first used.

    Returns
    -------
    LifespanSimulationArrays
        Named tuple with fields ``birth_times``, ``death_times``,
        ``birth_epochs``, ``survived``, ``n_born``.  Arrays are JAX arrays
        of length ``n_born``.

    Raises
    ------
    ValueError
        If input arrays have incompatible lengths, non-positive rates, or
        non-positive / non-increasing times.
    RuntimeError
        If the simulated population exceeds ``max_population``.

    """
    birth_rates_arr = np.atleast_1d(np.asarray(birth_rates, dtype=np.float64))
    death_rates_arr = np.atleast_1d(np.asarray(death_rates, dtype=np.float64))
    times_arr = np.atleast_1d(np.asarray(times, dtype=np.float64))

    if not (len(birth_rates_arr) == len(death_rates_arr) == len(times_arr)):
        msg = "birth_rates, death_rates, and times must have the same length"
        raise ValueError(msg)
    if np.any(birth_rates_arr <= 0) or np.any(death_rates_arr <= 0):
        msg = "Birth rates and death rates must be positive"
        raise ValueError(msg)
    if np.any(times_arr <= 0):
        msg = "Times must be positive"
        raise ValueError(msg)
    if len(times_arr) > 1 and np.any(np.diff(times_arr) <= 0):
        msg = "Times must be strictly increasing"
        raise ValueError(msg)
    if max_population < 1:
        msg = "max_population must be at least 1"
        raise ValueError(msg)

    br_jax = jnp.array(birth_rates_arr, dtype=jnp.float64)
    dr_jax = jnp.array(death_rates_arr, dtype=jnp.float64)
    et_jax = jnp.array(times_arr, dtype=jnp.float64)

    if random_seed is None:
        key = jax.random.key(int(np.random.default_rng().integers(0, 2**31 - 1)))
    else:
        key = jax.random.key(int(random_seed))

    final_state = _simulate_lifespans_core(br_jax, dr_jax, et_jax, key, max_population)

    if bool(final_state.overflow):
        msg = (
            f"simulate_birth_death_lifespans_arrays: population exceeded "
            f"max_population={max_population}. "
            "Increase max_population or use a lower birth rate / shorter time span."
        )
        raise RuntimeError(msg)

    n_born = int(final_state.n_born)
    terminal_time = float(times_arr[-1])
    survived = final_state.death_time_buf[:n_born] > terminal_time

    return LifespanSimulationArrays(
        birth_times=final_state.birth_time_buf[:n_born],
        death_times=final_state.death_time_buf[:n_born],
        birth_epochs=final_state.birth_epoch_buf[:n_born],
        survived=survived,
        n_born=n_born,
    )


def simulate_birth_death_lifespans(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
    max_population: int = 200_000,
) -> BirthDeathProcess:
    """Simulate a birth-death process using the lifespan-at-birth algorithm.

    A JIT-compiled alternative to :func:`simulate_birth_death_process` with
    the same public API.  Each individual's entire lifespan is decided at
    birth from a piecewise-exponential distribution.  The numeric core runs
    inside ``jax.lax.while_loop``; parent-ID genealogy and spine assignment
    are reconstructed on the host from the stored per-individual birth/death
    times and insertion uniforms.

    The returned :class:`~bdpinf.process.BirthDeathProcess` contains an
    additional ``birth_epoch`` column in its ``records`` table,
    recording the epoch in which each individual was born.

    Args:
        birth_rates: Birth rate(s) (λ). Scalar or 1-D array (one per epoch).
        death_rates: Death rate(s) (μ). Same length as ``birth_rates``.
        times: Epoch end time(s). Scalar or strictly-increasing 1-D array.
        random_seed: Integer seed for reproducibility, or ``None``.
        ensure_non_extinct: If ``True``, retry until at least one individual
            survives to the terminal time.
        max_retries: Maximum number of retry attempts when
            ``ensure_non_extinct=True``.
        max_population: Maximum total individuals that may be born; passed
            to the JIT core as a static buffer size.

    Returns
    -------
    BirthDeathProcess
        Simulated process with full genealogy and spine assignment.

    Raises
    ------
    ValueError
        On invalid inputs (lengths, non-positive rates, non-increasing times).
    RuntimeError
        If ``ensure_non_extinct=True`` and all retries produce an extinct
        population, or if ``max_population`` is exceeded.

    """
    birth_rates_arr = np.atleast_1d(np.asarray(birth_rates, dtype=np.float64))
    death_rates_arr = np.atleast_1d(np.asarray(death_rates, dtype=np.float64))
    times_arr = np.atleast_1d(np.asarray(times, dtype=np.float64))

    if not (len(birth_rates_arr) == len(death_rates_arr) == len(times_arr)):
        msg = "birth_rates, death_rates, and times must have the same length"
        raise ValueError(msg)
    if np.any(birth_rates_arr <= 0) or np.any(death_rates_arr <= 0):
        msg = "Birth rates and death rates must be positive"
        raise ValueError(msg)
    if np.any(times_arr <= 0):
        msg = "Times must be positive"
        raise ValueError(msg)
    if len(times_arr) > 1 and np.any(np.diff(times_arr) <= 0):
        msg = "Times must be strictly increasing"
        raise ValueError(msg)
    if ensure_non_extinct and max_retries < 0:
        msg = "max_retries must be a non-negative integer"
        raise ValueError(msg)
    if max_population < 1:
        msg = "max_population must be at least 1"
        raise ValueError(msg)

    terminal_time = float(times_arr[-1])
    br_jax = jnp.array(birth_rates_arr, dtype=jnp.float64)
    dr_jax = jnp.array(death_rates_arr, dtype=jnp.float64)
    et_jax = jnp.array(times_arr, dtype=jnp.float64)

    base_rng = np.random.default_rng(random_seed)
    attempts = max_retries if ensure_non_extinct else 0

    for _attempt in range(attempts + 1):
        if random_seed is None:
            key = jax.random.key(int(np.random.default_rng().integers(0, 2**31 - 1)))
        else:
            key = jax.random.key(int(base_rng.integers(0, 2**31 - 1)))

        final_state = _simulate_lifespans_core(
            br_jax, dr_jax, et_jax, key, max_population
        )

        if bool(final_state.overflow):
            msg = (
                f"simulate_birth_death_lifespans: population exceeded "
                f"max_population={max_population}."
            )
            raise RuntimeError(msg)

        n_born = int(final_state.n_born)
        birth_times_np = np.asarray(final_state.birth_time_buf[:n_born])
        death_times_np = np.asarray(final_state.death_time_buf[:n_born])
        insert_uniforms_np = np.asarray(final_state.insert_unif_buf[:n_born])
        birth_epochs_np = np.asarray(final_state.birth_epoch_buf[:n_born])
        n_alive_at_birth_np = np.asarray(final_state.n_alive_at_birth_buf[:n_born])

        survived_mask = ~np.isfinite(death_times_np) | (death_times_np > terminal_time)

        if ensure_non_extinct and not survived_mask.any():
            continue

        records_df = _reconstruct_genealogy(
            birth_times=birth_times_np,
            death_times=death_times_np,
            insert_uniforms=insert_uniforms_np,
            birth_epochs=birth_epochs_np,
            n_alive_at_birth=n_alive_at_birth_np,
            terminal_time=terminal_time,
        )

        if not ensure_non_extinct:
            return BirthDeathProcess.from_records(
                records_df,
                epoch_end_times=times_arr,
                birth_rates=birth_rates_arr,
                death_rates=death_rates_arr,
            )

        if records_df["survived"].any():
            return BirthDeathProcess.from_records(
                records_df,
                epoch_end_times=times_arr,
                birth_rates=birth_rates_arr,
                death_rates=death_rates_arr,
            )

    msg = (
        f"simulate_birth_death_lifespans: exhausted {attempts} retries "
        "without producing a non-extinct population"
    )
    raise RuntimeError(msg)


if __name__ == "__main__":
    for _ in range(10):
        sampled_ages = simulate_bdp_ages_from_distributions(
            birth_rate=0.5,
            death_rate=0.3,
            max_time=5.0,
            rng=np.random.default_rng(),
        )
        print(sampled_ages)
