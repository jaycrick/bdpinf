"""Bayesian inference utilities for birth-death process parameters using NumPyro.

This module drives full posterior inference (NumPyro NUTS) over birth and death
rates, using the JAX math layer via ``numpyro.factor``.

Key design choices
------------------
* A single ``free_log_rates`` sample site holds the reduced-dimension latent
  vector (only the free parameters under ``ParameterConstraints``).  The full
  ``2 * n_epochs`` log-rate vector is reconstructed differentiably inside the
  model using ``_expand_log_rates``, so equality-group and fixed constraints
  are honoured *during* sampling rather than post-hoc.
* ``BayesianPriorConfig`` supports per-rate-family and per-epoch loc/scale
  overrides on top of scalar fallbacks, enabling weakly-informative priors
  that differ between birth and death rates or between epochs.
* Multi-chain inference uses ``chain_method="vectorized"`` (JAX vmap over
  chains) which is safe on both CPU and GPU without subprocess overhead.
* Data tensors in ``BayesianModelData`` are stored as **NumPy arrays** so that
  they remain concrete (not abstract tracers) inside NumPyro's JIT-compiled
  ``lax.while_loop``.  The model function converts them to JAX at the boundary
  before calling the math layer.
* The model calls ``calculate_populations_log_lik`` and the age-likelihood
  functions *directly* (with ``_skip_normalization=True``) rather than going
  through ``log_likelihood``, avoiding validation code that uses Python-level
  ``float()`` / ``bool()`` / ``int()`` on JAX arrays — which fail under JIT.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpy.typing import NDArray
from numpyro.infer import MCMC, NUTS

from bdpinf.inference.fitting import ParameterConstraints
from bdpinf.inference.likelihood import (
    _log_likelihood_core,  # pyright: ignore[reportPrivateUsage]
    resolve_spine_ages,
)
from bdpinf.models.validation import normalize_and_validate_inputs

logger = logging.getLogger(__name__)

# Substitute for log(0) when a fixed-param value is zero: large negative keeps
# arithmetic finite without being exactly -inf.
_LOG_ZERO: float = -40.0


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianPriorConfig:
    """Normal prior configuration for log birth/death rates.

    The scalar ``loc`` / ``scale`` fields serve as fallbacks for both rate
    families.  Per-family overrides (``birth_loc``, ``death_loc``, …) take
    precedence.  Each per-family value may be a scalar (broadcast to all
    epochs) or a sequence of length ``n_epochs`` (per-epoch prior).

    Parameters
    ----------
    loc : float
        Fallback prior mean on log-rates (both families).
    scale : float
        Fallback prior standard deviation on log-rates (both families).
    birth_loc : float | Sequence[float] | None
        Prior mean(s) for log-birth-rates.  ``None`` → fall back to ``loc``.
    birth_scale : float | Sequence[float] | None
        Prior std(s) for log-birth-rates.  ``None`` → fall back to ``scale``.
    death_loc : float | Sequence[float] | None
        Prior mean(s) for log-death-rates.  ``None`` → fall back to ``loc``.
    death_scale : float | Sequence[float] | None
        Prior std(s) for log-death-rates.  ``None`` → fall back to ``scale``.

    Examples
    --------
    Weakly informative, tighter prior on death rates:

    >>> cfg = BayesianPriorConfig(loc=-6.0, scale=3.0, death_scale=1.5)

    Per-epoch birth priors for a two-epoch model:

    >>> cfg = BayesianPriorConfig(birth_loc=[-5.0, -7.0], birth_scale=1.5)

    """

    loc: float = -6.0
    scale: float = 2.0
    birth_loc: float | Sequence[float] | None = None
    birth_scale: float | Sequence[float] | None = None
    death_loc: float | Sequence[float] | None = None
    death_scale: float | Sequence[float] | None = None

    def resolve_tensors(
        self,
        n_epochs: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(loc_array, scale_array)`` of shape ``[2 * n_epochs]``.

        Index convention: ``i < n_epochs`` → birth epoch ``i``; ``i >= n_epochs``
        → death epoch ``i - n_epochs``.

        Parameters
        ----------
        n_epochs : int
            Number of epochs in the model.

        Returns
        -------
        tuple[NDArray, NDArray]
            ``(locs, scales)`` — both float64, shape ``[2 * n_epochs]``.

        """

        def _resolve(
            override: float | Sequence[float] | None,
            fallback: float,
            n: int,
        ) -> NDArray[np.float64]:
            if override is None:
                return np.full((n,), fallback, dtype=np.float64)
            if isinstance(override, (int, float)):
                return np.full((n,), float(override), dtype=np.float64)
            vec = np.array(list(override), dtype=np.float64)
            if vec.shape[0] != n:
                msg = (
                    f"Prior sequence length {vec.shape[0]} does not match n_epochs={n}"
                )
                raise ValueError(msg)
            return vec

        birth_locs = _resolve(self.birth_loc, self.loc, n_epochs)
        death_locs = _resolve(self.death_loc, self.loc, n_epochs)
        birth_scales = _resolve(self.birth_scale, self.scale, n_epochs)
        death_scales = _resolve(self.death_scale, self.scale, n_epochs)

        locs = np.concatenate([birth_locs, death_locs])
        scales = np.concatenate([birth_scales, death_scales])
        return locs, scales


@dataclass(frozen=True)
class BayesianMCMCConfig:
    """NUTS/MCMC sampler configuration.

    Parameters
    ----------
    num_samples : int
        Posterior samples per chain after warm-up.
    warmup_steps : int
        NUTS adaptation (warm-up) steps; discarded.
    num_chains : int
        Number of independent chains.  Multiple chains are parallelised via
        JAX vmap (``chain_method="vectorized"``), which is safe on both CPU
        and GPU.
    target_accept_prob : float
        NUTS dual-averaging target acceptance probability.
    max_tree_depth : int
        Maximum leapfrog tree depth per step.
    disable_progressbar : bool
        Suppress the per-step progress display.

    """

    num_samples: int = 800
    warmup_steps: int = 400
    num_chains: int = 1
    target_accept_prob: float = 0.7
    max_tree_depth: int = 10
    disable_progressbar: bool = True


@dataclass(frozen=True)
class BayesianModelConfig:
    """Model-level options for Bayesian birth-death inference.

    Parameters
    ----------
    is_ordered : bool
        If ``True``, the first age in each group is treated as the ancestral
        spine.
    spine_ages : jax.Array | NDArray | None
        Explicit spine ages (mutually exclusive with ``is_ordered=True``).
    population_sizes : jax.Array | NDArray | None
        Observed per-epoch counts.  Inferred from ``list_of_ages`` if
        ``None``.
    initial_population : int
        Size of the ancestral population at time 0.
    condition_on_survival : bool
        Condition on non-extinction when computing the population likelihood.
    rng_seed : int | None
        Seed for the JAX PRNG (reproducibility).
    constraints : ParameterConstraints | None
        Fixed parameters and equality groups.  When provided, only the free
        parameters are sampled; constraints are applied *inside* the NumPyro
        model (not post-hoc).
    init_log_rates : jax.Array | NDArray | None
        Starting values for the ``free_log_rates`` sample site (shape
        ``[n_free]``).  When provided, NUTS warmup begins at this point
        rather than sampling from the prior.  For multi-chain runs a small
        per-chain perturbation (±0.1 in log-space) is added automatically.

    """

    is_ordered: bool = False
    spine_ages: jax.Array | NDArray[np.float64] | None = None
    population_sizes: jax.Array | NDArray[np.float64] | None = None
    initial_population: int = 1
    condition_on_survival: bool = True
    rng_seed: int | None = None
    constraints: ParameterConstraints | None = None
    init_log_rates: jax.Array | NDArray[np.float64] | None = None


class BayesianFitKwargs(TypedDict, total=False):
    """Typed kwargs for Bayesian fitting wrappers."""

    model_config: BayesianModelConfig
    mcmc_config: BayesianMCMCConfig
    prior_config: BayesianPriorConfig


# ---------------------------------------------------------------------------
# Internal data container for the NumPyro model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianModelData:
    """Pre-validated data and metadata required by the NumPyro model.

    All array fields are **NumPy** arrays (not JAX arrays).  This ensures they
    remain concrete constants inside NumPyro's JIT-compiled ``lax.while_loop``
    rather than becoming abstract tracers.  JAX arrays closed over inside the
    traced model become abstract; NumPy arrays do not.

    Python-level flags (``use_unknown_spine_marginal``, ``n_non_spine_ages``,
    ``total_pop_size``) are pre-computed at construction time so the model
    function itself contains no Python-level branching on JAX values.
    """

    n_epochs: int
    non_spine_ages: NDArray[np.float64]  # pre-validated, boundary-detected
    times: NDArray[np.float64]
    spine_ages: NDArray[np.float64] | None  # explicit or auto-detected
    ns: NDArray[np.int32]  # population sizes per epoch
    initial_population: int
    condition_on_survival: bool
    # Prior arrays for the free parameters only (already sliced to free indices)
    free_prior_locs: NDArray[np.float64]  # shape [n_free]
    free_prior_scales: NDArray[np.float64]  # shape [n_free]
    # Constraint metadata (precomputed for speed inside the model)
    constraints: ParameterConstraints | None
    free_indices: list[int]
    # Pre-computed Python-level flags (avoids Python branching on JAX tracers)
    use_unknown_spine_marginal: bool
    n_non_spine_ages: int  # len(non_spine_ages)
    has_spine: bool  # spine_ages is not None


# ---------------------------------------------------------------------------
# Constraint expansion helpers
# ---------------------------------------------------------------------------


def _expand_log_rates(
    free_log_rates: jax.Array,
    constraints: ParameterConstraints | None,
    n_total: int,
    free_indices: list[int],
) -> jax.Array:
    """Expand a ``[n_free]`` free-log-rate vector to ``[n_total]``.

    Called inside the NumPyro model on each NUTS step — must be differentiable
    with respect to ``free_log_rates``.  Fixed-param entries are inserted as
    constants; equality-group non-representatives are aliased via ``jnp.stack``.

    """
    if constraints is None:
        return free_log_rates

    pos_of = {idx: pos for pos, idx in enumerate(free_indices)}
    rep_of: dict[int, int] = {
        idx: min(g) for g in constraints.equality_groups for idx in g
    }

    parts: list[jax.Array] = []
    for idx in range(n_total):
        if idx in constraints.fixed_params:
            val = constraints.fixed_params[idx]
            log_val = math.log(val) if val > 0.0 else _LOG_ZERO
            parts.append(jnp.array(log_val, dtype=free_log_rates.dtype))
        else:
            src = rep_of.get(idx, idx)
            parts.append(free_log_rates[pos_of[src]])

    return jnp.stack(parts)


def _expand_log_rates_batch(
    free_log_samples: NDArray[np.float64],
    constraints: ParameterConstraints | None,
    n_total: int,
    free_indices: list[int],
) -> NDArray[np.float64]:
    """Expand a ``[..., n_free]`` batch to ``[..., n_total]`` by filling constraints."""
    if constraints is None:
        return free_log_samples

    pos_of = {idx: pos for pos, idx in enumerate(free_indices)}
    rep_of: dict[int, int] = {
        idx: min(g) for g in constraints.equality_groups for idx in g
    }

    batch_shape = free_log_samples.shape[:-1]
    full = np.zeros((*batch_shape, n_total), dtype=np.float64)

    for idx in range(n_total):
        if idx in constraints.fixed_params:
            val = constraints.fixed_params[idx]
            full[..., idx] = math.log(val) if val > 0.0 else _LOG_ZERO
        else:
            src = rep_of.get(idx, idx)
            full[..., idx] = free_log_samples[..., pos_of[src]]

    return full


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianLikelihoodModel:
    """Callable NumPyro model for MCMC chains.

    Samples a single ``free_log_rates`` site of shape ``[n_free]``, expands
    it to the full ``[2 * n_epochs]`` log-rate vector (applying any
    ``ParameterConstraints`` differentiably), and computes the JAX
    log-likelihood via ``numpyro.factor``.

    Data is stored as NumPy arrays in ``model_data`` to remain concrete inside
    NumPyro's JIT-compiled ``lax.while_loop``.  Python-level flags are
    pre-computed so the model contains no abstract-value branching.
    """

    model_data: BayesianModelData

    def __call__(self) -> None:
        """Evaluate the NumPyro model for one MCMC draw."""
        md = self.model_data
        n_epochs = md.n_epochs
        n_total = 2 * n_epochs

        free_log_rates = numpyro.sample(
            "free_log_rates",
            dist.Normal(
                # NumPy arrays → concrete JAX constants in the prior
                jnp.asarray(md.free_prior_locs),
                jnp.asarray(md.free_prior_scales),
            ).to_event(1),
        )

        full_log_rates = _expand_log_rates(
            free_log_rates, md.constraints, n_total, md.free_indices
        )
        birth_rates = jnp.exp(full_log_rates[:n_epochs])
        death_rates = jnp.exp(full_log_rates[n_epochs:])

        # Convert NumPy data to JAX for computation.  These become JAX
        # constants (not abstract tracers) when created from NumPy inside the
        # JIT-compiled model, so all downstream math is grad-safe.
        times_jax = jnp.asarray(md.times)
        ns_jax = jnp.asarray(md.ns)

        # Same shared core as the MLE hot path (population + ages, with the
        # spine marginalised when unknown); data are pre-validated.
        total_log_lik = _log_likelihood_core(
            birth_rates,
            death_rates,
            non_spine_ages=jnp.asarray(md.non_spine_ages),
            spine_ages=jnp.asarray(md.spine_ages) if md.has_spine else None,
            times=times_jax,
            ns=ns_jax,
            initial_population=md.initial_population,
            condition_on_survival=md.condition_on_survival,
            use_unknown_spine_marginal=md.use_unknown_spine_marginal,
            skip_normalization=True,
        )
        safe_log_lik = jnp.where(
            jnp.isfinite(total_log_lik),
            total_log_lik,
            jnp.array(-1e20, dtype=jnp.float64),
        )
        numpyro.factor("obs_log_likelihood", safe_log_lik)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_age_inputs(
    list_of_ages: list[jax.Array | NDArray[np.float64] | list[float]],
    times_np: NDArray[np.float64],
    *,
    is_ordered: bool,
    spine_ages: jax.Array | NDArray[np.float64] | None,
    population_sizes: jax.Array | NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.int32], bool]:
    """Prepare age and population data as NumPy arrays.

    Handles spine auto-detection (ages at the final observation time) and
    returns the ``use_unknown_spine_marginal`` flag.

    Returns
    -------
    tuple
        ``(non_spine_ages, spine_ages_or_none, ns, use_unknown_spine_marginal)``
        All arrays are NumPy; ``ns`` has dtype int32.

    """
    age_arrays = [
        np.asarray(ages, dtype=np.float64).reshape(-1) for ages in list_of_ages
    ]
    if not age_arrays or not any(a.size > 0 for a in age_arrays):
        msg = "No valid data points for fitting"
        raise ValueError(msg)

    if is_ordered and spine_ages is not None:
        msg = "Cannot have both is_ordered=True and spine_ages provided."
        raise ValueError(msg)

    inferred_ns = np.array([len(ages) for ages in age_arrays], dtype=np.int32)

    spine_np: NDArray[np.float64] | None = (
        np.asarray(spine_ages, dtype=np.float64).reshape(-1)
        if spine_ages is not None
        else None
    )

    if is_ordered:
        spine_list: list[NDArray[np.float64]] = []
        non_spine_list: list[NDArray[np.float64]] = []
        for ages in age_arrays:
            if ages.size == 0:
                continue
            spine_list.append(ages[:1])
            if ages.size > 1:
                non_spine_list.append(ages[1:])

        spine_np = (
            np.concatenate(spine_list) if spine_list else np.empty(0, dtype=np.float64)
        )
        non_spine_np = (
            np.concatenate(non_spine_list)
            if non_spine_list
            else np.empty(0, dtype=np.float64)
        )
        ns = population_sizes if population_sizes is not None else inferred_ns
        return non_spine_np, spine_np, np.asarray(ns, dtype=np.int32), False

    # Non-ordered: concatenate all ages, then do boundary detection
    all_ages = (
        np.concatenate(age_arrays) if age_arrays else np.empty(0, dtype=np.float64)
    )
    end_time = float(times_np[-1])

    if spine_np is not None:
        # Explicit spine provided — no boundary detection needed
        ns = population_sizes if population_sizes is not None else inferred_ns
        return all_ages, spine_np, np.asarray(ns, dtype=np.int32), False

    # Boundary detection: age at final time → inferred spine
    resolved = resolve_spine_ages(all_ages, end_time)
    spine_np = resolved.spine_ages
    all_ages = resolved.non_spine_ages
    use_unknown_spine = resolved.use_unknown_spine

    ns = population_sizes if population_sizes is not None else inferred_ns
    return all_ages, spine_np, np.asarray(ns, dtype=np.int32), use_unknown_spine


def _run_nuts_mcmc(
    model: BayesianLikelihoodModel,
    mcmc_config: BayesianMCMCConfig,
    rng_seed: int | None,
    *,
    init_log_rates: jax.Array | NDArray[np.float64] | None = None,
) -> MCMC:
    """Configure and run NumPyro NUTS, returning the fitted MCMC object."""
    seed = rng_seed if rng_seed is not None else 0
    rng_key = jax.random.PRNGKey(seed)

    nuts_kernel = NUTS(
        model,
        target_accept_prob=mcmc_config.target_accept_prob,
        max_tree_depth=mcmc_config.max_tree_depth,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=mcmc_config.warmup_steps,
        num_samples=mcmc_config.num_samples,
        num_chains=mcmc_config.num_chains,
        chain_method="vectorized",
        progress_bar=not mcmc_config.disable_progressbar,
    )

    init_params: dict[str, jax.Array] | None = None
    if init_log_rates is not None:
        init = np.asarray(init_log_rates, dtype=np.float64)
        n_chains = mcmc_config.num_chains
        if n_chains > 1:
            noise_key, rng_key = jax.random.split(rng_key)
            noise = np.array(
                jax.random.normal(noise_key, shape=(n_chains, init.shape[0])) * 0.1
            )
            init = np.broadcast_to(init, (n_chains, init.shape[0])) + noise
        init_params = {"free_log_rates": jnp.array(init)}

    mcmc.run(rng_key, init_params=init_params, extra_fields=("diverging",))
    return mcmc


def _num_divergences(mcmc: MCMC) -> int:
    """Count total divergent transitions across all chains."""
    extra = mcmc.get_extra_fields()
    diverging = extra.get("diverging")
    if diverging is None:
        return 0
    return int(jnp.sum(diverging))


def _metric_as_array(
    summary_dict: dict[str, Any],
    metric_name: str,
    expected_length: int,
) -> NDArray[np.float64]:
    """Normalize a scalar/vector diagnostic metric into a fixed-length ndarray."""
    metric_value = summary_dict.get(metric_name)
    if metric_value is None:
        return np.full(expected_length, np.nan, dtype=np.float64)
    metric_array = np.asarray(metric_value, dtype=np.float64)
    if metric_array.ndim == 0:
        return np.full(expected_length, float(metric_array), dtype=np.float64)
    return metric_array.reshape(-1)


def _build_diagnostics_payload(
    posterior_summary_by_chain: dict[str, Any],
    *,
    n_epochs: int,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Build R-hat and ESS arrays for each rate family."""
    birth_stats = posterior_summary_by_chain.get("birth_rates", {})
    death_stats = posterior_summary_by_chain.get("death_rates", {})
    return {
        "birth_rates": {
            "r_hat": _metric_as_array(birth_stats, "r_hat", n_epochs),
            "n_eff": _metric_as_array(birth_stats, "n_eff", n_epochs),
        },
        "death_rates": {
            "r_hat": _metric_as_array(death_stats, "r_hat", n_epochs),
            "n_eff": _metric_as_array(death_stats, "n_eff", n_epochs),
        },
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def mle_init_log_rates(
    mle_result: dict[str, Any],
    constraints: ParameterConstraints | None,
    n_epochs: int,
) -> NDArray[np.float64]:
    """Extract free log-rates from an MLE result for use as NUTS initialization.

    Converts the MLE birth/death rate estimates to log-space, assembles the
    full ``[2 * n_epochs]`` log-rate vector, and slices to the free-parameter
    indices under ``constraints``.  The result can be passed directly as
    ``BayesianModelConfig(init_log_rates=...)``.

    Parameters
    ----------
    mle_result : dict
        Return value of :func:`bdpinf.inference.fit_birth_death_processes`.
    constraints : ParameterConstraints | None
        Same constraints used for the Bayesian fit.  ``None`` → all parameters
        are free.
    n_epochs : int
        Number of epochs.

    Returns
    -------
    NDArray[np.float64]
        Free log-rates, shape ``[n_free]``, float64.

    """
    n_total = 2 * n_epochs
    birth = np.asarray(mle_result["birth_rates"], dtype=np.float64)
    death = np.asarray(mle_result["death_rates"], dtype=np.float64)
    full_rates = np.concatenate([birth, death])
    # Clamp to avoid log(0)
    full_rates = np.clip(full_rates, 1e-300, None)
    log_full = np.log(full_rates)

    if constraints is None:
        return log_full
    free_indices = constraints.get_free_param_indices(n_total)
    return log_full[free_indices]


def fit_birth_death_processes_bayesian(
    list_of_ages: list[jax.Array | NDArray[np.float64] | list[float]],
    times: jax.Array | NDArray[np.float64] | float,
    *,
    model_config: BayesianModelConfig | None = None,
    mcmc_config: BayesianMCMCConfig | None = None,
    prior_config: BayesianPriorConfig | None = None,
) -> dict[str, Any]:
    """Fit birth-death rates using Bayesian inference with NumPyro NUTS.

    Parameters
    ----------
    list_of_ages : list[jax.Array | ndarray | list[float]]
        Age observations grouped by population/epoch.
    times : jax.Array | ndarray | float
        Observation time(s) — one value per epoch.
    model_config : BayesianModelConfig | None
        Model options: ordered inputs, spine ages, population sizes, initial
        population, condition_on_survival, constraints, and RNG seed.  Defaults to
        ``BayesianModelConfig()``.
    mcmc_config : BayesianMCMCConfig | None
        NUTS/MCMC configuration.  Defaults to ``BayesianMCMCConfig()``.
    prior_config : BayesianPriorConfig | None
        Normal prior location/scale on log-rates.  Defaults to
        ``BayesianPriorConfig()``.

    Returns
    -------
    dict[str, Any]
        Keys:

        ``"success"``
            Always ``True``.
        ``"birth_rates"``
            Posterior mean birth rates, shape ``[n_epochs]``.
        ``"death_rates"``
            Posterior mean death rates, shape ``[n_epochs]``.
        ``"birth_rates_90_ci"``
            5th–95th percentile credible interval, shape ``[n_epochs, 2]``.
        ``"death_rates_90_ci"``
            Same for death rates.
        ``"posterior_samples"``
            Dict ``{"birth_rates": ndarray, "death_rates": ndarray}`` — flat
            over chains, shape ``[n_samples, n_epochs]``.
        ``"posterior_samples_by_chain"``
            Same, grouped: shape ``[n_chains, n_samples, n_epochs]``.
        ``"posterior_log_samples"``
            Log-space posterior samples (flat), same dict structure.  Useful
            for prior-vs-posterior plots.
        ``"posterior_log_samples_by_chain"``
            Same, grouped.
        ``"posterior_summary"``
            NumPyro ``diagnostics.summary()`` dict (rate-space, flat).
        ``"posterior_summary_by_chain"``
            Same dict (rate-space, by-chain).
        ``"diagnostics"``
            Dict ``{"birth_rates": {"r_hat", "n_eff"}, "death_rates": ...}``.
        ``"num_divergences"``
            Total divergence count across all chains.
        ``"mcmc_config"``
            Dict echo of the MCMC settings used.
        ``"n_epochs"``
            Number of epochs in the model.

    Raises
    ------
    ValueError
        If ``initial_population <= 0``, constraints are invalid, or the
        ``prior_config`` sequence lengths do not match ``n_epochs``.

    """
    model_config = model_config or BayesianModelConfig()
    mcmc_config = mcmc_config or BayesianMCMCConfig()
    prior_config = prior_config or BayesianPriorConfig()

    if model_config.initial_population <= 0:
        msg = "initial_population must be a positive integer"
        raise ValueError(msg)

    (times_jax,) = normalize_and_validate_inputs(times=times)
    times_np = np.asarray(times_jax, dtype=np.float64)
    n_epochs = int(times_np.shape[0])
    n_total = 2 * n_epochs

    # Validate constraints before building anything
    constraints = model_config.constraints
    if constraints is not None:
        constraints.validate(n_total)
        free_indices = constraints.get_free_param_indices(n_total)
    else:
        free_indices = list(range(n_total))

    # Resolve prior arrays for all 2*n_epochs params, then slice to free
    all_prior_locs, all_prior_scales = prior_config.resolve_tensors(n_epochs)
    free_prior_locs = all_prior_locs[free_indices]
    free_prior_scales = all_prior_scales[free_indices]

    # Prepare age and population data (host-side; all NumPy)
    non_spine_np, spine_np, ns_np, use_unknown_spine_marginal = _prepare_age_inputs(
        list_of_ages,
        times_np,
        is_ordered=model_config.is_ordered,
        spine_ages=model_config.spine_ages,
        population_sizes=model_config.population_sizes,
    )

    if model_config.initial_population > 1 and use_unknown_spine_marginal:
        msg = (
            "Unknown-spine marginal is not supported for initial_population > 1. "
            "Provide explicit spine_ages or use is_ordered=True."
        )
        raise ValueError(msg)

    model_data = BayesianModelData(
        n_epochs=n_epochs,
        non_spine_ages=non_spine_np,
        times=times_np,
        spine_ages=spine_np,
        ns=ns_np,
        initial_population=model_config.initial_population,
        condition_on_survival=model_config.condition_on_survival,
        free_prior_locs=free_prior_locs,
        free_prior_scales=free_prior_scales,
        constraints=constraints,
        free_indices=free_indices,
        use_unknown_spine_marginal=use_unknown_spine_marginal,
        n_non_spine_ages=int(non_spine_np.shape[0]),
        has_spine=spine_np is not None,
    )

    model = BayesianLikelihoodModel(model_data=model_data)

    mcmc = _run_nuts_mcmc(
        model,
        mcmc_config,
        model_config.rng_seed,
        init_log_rates=model_config.init_log_rates,
    )

    # ---- Extract and reshape posterior samples --------------------------------

    posterior_flat = mcmc.get_samples(group_by_chain=False)
    posterior_by_chain = mcmc.get_samples(group_by_chain=True)

    free_flat = np.asarray(posterior_flat["free_log_rates"], dtype=np.float64)
    free_by_chain = np.asarray(posterior_by_chain["free_log_rates"], dtype=np.float64)

    # Expand free params → full [2*n_epochs] log-rate vectors
    full_log_flat = _expand_log_rates_batch(
        free_flat, constraints, n_total, free_indices
    )
    full_log_by_chain = _expand_log_rates_batch(
        free_by_chain, constraints, n_total, free_indices
    )

    # Split into birth / death
    log_birth_flat = full_log_flat[:, :n_epochs]
    log_death_flat = full_log_flat[:, n_epochs:]
    birth_flat = np.exp(log_birth_flat)
    death_flat = np.exp(log_death_flat)

    log_birth_by_chain = full_log_by_chain[..., :n_epochs]
    log_death_by_chain = full_log_by_chain[..., n_epochs:]
    birth_by_chain = np.exp(log_birth_by_chain)
    death_by_chain = np.exp(log_death_by_chain)

    # ---- Credible intervals --------------------------------------------------

    birth_ci = np.quantile(birth_flat, [0.05, 0.95], axis=0).T  # [n_epochs, 2]
    death_ci = np.quantile(death_flat, [0.05, 0.95], axis=0).T

    # ---- Divergences ---------------------------------------------------------

    num_divergences = _num_divergences(mcmc)

    # ---- Summary (rate-space, for diagnostics) ------------------------------

    # diagnostics.summary requires shape [n_chains, n_samples, ...] for ESS/R-hat
    posterior_summary = numpyro.diagnostics.summary(
        {"birth_rates": birth_by_chain, "death_rates": death_by_chain},
        prob=0.9,
    )
    posterior_summary_by_chain = posterior_summary
    diagnostics_payload = _build_diagnostics_payload(
        posterior_summary_by_chain,
        n_epochs=n_epochs,
    )

    return {
        "success": True,
        "birth_rates": birth_flat.mean(axis=0),
        "death_rates": death_flat.mean(axis=0),
        "birth_rates_90_ci": birth_ci,
        "death_rates_90_ci": death_ci,
        "posterior_samples": {
            "birth_rates": birth_flat,
            "death_rates": death_flat,
        },
        "posterior_samples_by_chain": {
            "birth_rates": birth_by_chain,
            "death_rates": death_by_chain,
        },
        "posterior_log_samples": {
            "birth_rates": log_birth_flat,
            "death_rates": log_death_flat,
        },
        "posterior_log_samples_by_chain": {
            "birth_rates": log_birth_by_chain,
            "death_rates": log_death_by_chain,
        },
        "posterior_summary": posterior_summary,
        "posterior_summary_by_chain": posterior_summary_by_chain,
        "diagnostics": diagnostics_payload,
        "num_divergences": num_divergences,
        "mcmc_config": {
            "num_samples": mcmc_config.num_samples,
            "warmup_steps": mcmc_config.warmup_steps,
            "num_chains": mcmc_config.num_chains,
            "target_accept_prob": mcmc_config.target_accept_prob,
            "max_tree_depth": mcmc_config.max_tree_depth,
        },
        "n_epochs": n_epochs,
    }
