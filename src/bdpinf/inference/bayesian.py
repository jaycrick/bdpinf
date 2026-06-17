"""Bayesian inference utilities for birth-death process parameters using Pyro.

This module drives full posterior inference (Pyro NUTS) over birth and death
rates, reusing the validated PyTorch likelihood verbatim via ``pyro.factor``.

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
* ``device`` on ``BayesianModelConfig`` moves all data tensors to the
  requested device (e.g. ``"cuda"``).  When using a GPU, set
  ``num_chains=1``; Pyro spawns OS processes for multi-chain runs, and
  CUDA contexts are not safely forkable.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, Unpack

import numpy as np
import pandas as pd
import pyro  # no type stubs — unknown members suppressed in pyproject.toml
import pyro.distributions as dist  # no type stubs
import torch
from numpy.typing import NDArray
from pyro.infer import MCMC, NUTS  # no type stubs
from pyro.infer.mcmc.util import summary  # no type stubs

from inf_bp_texts.inference.fitting import ParameterConstraints
from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import ensure_1d_tensor

from .likelihood import log_likelihood

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
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loc_tensor, scale_tensor)`` of shape ``[2 * n_epochs]``.

        Index convention: ``i < n_epochs`` → birth epoch ``i``; ``i >= n_epochs``
        → death epoch ``i - n_epochs``.

        Parameters
        ----------
        n_epochs : int
            Number of epochs in the model.
        device : str
            PyTorch device string (e.g. ``"cpu"`` or ``"cuda"``).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(locs, scales)`` — both float64, shape ``[2 * n_epochs]``, on
            ``device``.

        """

        def _resolve(
            override: float | Sequence[float] | None,
            fallback: float,
            n: int,
        ) -> torch.Tensor:
            if override is None:
                return torch.full((n,), fallback, dtype=torch.float64, device=device)
            if isinstance(override, (int, float)):
                return torch.full(
                    (n,), float(override), dtype=torch.float64, device=device
                )
            vec = torch.tensor(list(override), dtype=torch.float64, device=device)
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

        locs = torch.cat([birth_locs, death_locs])
        scales = torch.cat([birth_scales, death_scales])
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
        Number of independent chains.  For GPU runs, use ``num_chains=1``
        (Pyro spawns OS processes for multi-chain and CUDA contexts are not
        safely forkable).
    target_accept_prob : float
        NUTS dual-averaging target acceptance probability.
    max_tree_depth : int
        Maximum leapfrog tree depth per step.
    jit_compile : bool
        JIT-compile the log-potential for speed (experimental).
    disable_progressbar : bool
        Suppress the per-step progress display.

    """

    num_samples: int = 800
    warmup_steps: int = 400
    num_chains: int = 1
    target_accept_prob: float = 0.7
    max_tree_depth: int = 10
    jit_compile: bool = False
    disable_progressbar: bool = True


@dataclass(frozen=True)
class BayesianModelConfig:
    """Model-level options for Bayesian birth-death inference.

    Parameters
    ----------
    is_ordered : bool
        If ``True``, the first age in each group is treated as the ancestral
        spine.
    spine_ages : torch.Tensor | None
        Explicit spine ages (mutually exclusive with ``is_ordered=True``).
    population_sizes : torch.Tensor | None
        Observed per-epoch counts.  Inferred from ``list_of_ages`` if
        ``None``.
    initial_population : int
        Size of the ancestral population at time 0.
    unbias : bool
        Condition on non-extinction when computing the population likelihood.
    rng_seed : int | None
        Seed for ``pyro.set_rng_seed`` (reproducibility).
    constraints : ParameterConstraints | None
        Fixed parameters and equality groups.  When provided, only the free
        parameters are sampled; constraints are applied *inside* the Pyro
        model (not post-hoc).
    device : str
        PyTorch device for inference (``"cpu"`` or ``"cuda"``).  All data
        tensors are moved to this device before sampling.
    init_log_rates : torch.Tensor | None
        Starting values for the ``free_log_rates`` sample site (shape
        ``[n_free]``).  When provided, NUTS warmup begins at this point
        rather than sampling from the prior.  Strongly recommended when the
        prior is diffuse and the posterior is far from the prior mean (e.g.
        pass the log of the MLE estimates for the free parameters).  For
        multi-chain runs a small per-chain perturbation (±0.1 in log-space)
        is added automatically so chains explore different regions.

    """

    is_ordered: bool = False
    spine_ages: torch.Tensor | None = None
    population_sizes: torch.Tensor | None = None
    initial_population: int = 1
    unbias: bool = True
    rng_seed: int | None = None
    constraints: ParameterConstraints | None = None
    device: str = "cpu"
    init_log_rates: torch.Tensor | None = None


class BayesianFitKwargs(TypedDict, total=False):
    """Typed kwargs for Bayesian fitting wrappers."""

    model_config: BayesianModelConfig
    mcmc_config: BayesianMCMCConfig
    prior_config: BayesianPriorConfig


# ---------------------------------------------------------------------------
# Internal data container for the Pyro model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianModelData:
    """Normalized tensors and metadata required by ``BayesianLikelihoodModel``.

    All tensors are on the target device and float64.
    """

    n_epochs: int
    non_spine_ages: torch.Tensor
    times_tensor: torch.Tensor
    spine_ages_tensor: torch.Tensor | None
    population_sizes_tensor: torch.Tensor
    initial_population: int
    unbias: bool
    # Prior tensors for the free parameters only (already sliced to free indices)
    free_prior_locs: torch.Tensor  # shape [n_free]
    free_prior_scales: torch.Tensor  # shape [n_free]
    # Constraint metadata (precomputed for speed inside the model)
    constraints: ParameterConstraints | None
    free_indices: list[int]  # precomputed free param indices
    device: str


# ---------------------------------------------------------------------------
# Constraint expansion helpers
# ---------------------------------------------------------------------------


def _expand_log_rates(
    free_log_rates: torch.Tensor,
    constraints: ParameterConstraints | None,
    n_total: int,
    free_indices: list[int],
) -> torch.Tensor:
    """Expand a ``[n_free]`` free-log-rate vector to ``[n_total]``.

    This is called *inside* the Pyro model on each NUTS step, so it must be
    differentiable with respect to ``free_log_rates``.  Fixed-param entries
    are inserted as constants (no gradient needed); equality-group
    non-representatives are aliased to their representative's tensor element
    (``torch.stack`` of slices preserves the autograd graph).

    Parameters
    ----------
    free_log_rates : torch.Tensor
        Sampled free log-rates, shape ``[n_free]``.
    constraints : ParameterConstraints | None
        Constraint specification.  ``None`` → return ``free_log_rates``
        unchanged (unconstrained case).
    n_total : int
        Total parameter count (``2 * n_epochs``).
    free_indices : list[int]
        Precomputed free-param indices (``constraints.get_free_param_indices``
        or ``list(range(n_total))`` when unconstrained).

    Returns
    -------
    torch.Tensor
        Full log-rate vector, shape ``[n_total]``, on the same device and
        dtype as ``free_log_rates``.

    """
    if constraints is None:
        return free_log_rates

    pos_of = {idx: pos for pos, idx in enumerate(free_indices)}
    rep_of: dict[int, int] = {
        idx: min(g) for g in constraints.equality_groups for idx in g
    }

    parts: list[torch.Tensor] = []
    for idx in range(n_total):
        if idx in constraints.fixed_params:
            val = constraints.fixed_params[idx]
            log_val = math.log(val) if val > 0.0 else _LOG_ZERO
            parts.append(
                torch.tensor(
                    log_val,
                    dtype=free_log_rates.dtype,
                    device=free_log_rates.device,
                )
            )
        else:
            src = rep_of.get(idx, idx)
            parts.append(free_log_rates[pos_of[src]])

    return torch.stack(parts)


def _expand_log_rates_batch(
    free_log_samples: torch.Tensor,
    constraints: ParameterConstraints | None,
    n_total: int,
    free_indices: list[int],
) -> torch.Tensor:
    """Expand a ``[..., n_free]`` batch of free-log-rate samples to ``[..., n_total]``.

    Used in post-processing to reconstruct the full rate arrays from the single
    ``free_log_rates`` sample site (which may cover only the constrained
    free-parameter subspace).

    Parameters
    ----------
    free_log_samples : torch.Tensor
        Batched free log-rate samples, shape ``[..., n_free]``
        (e.g. ``[n_samples, n_free]`` or ``[n_chains, n_samples, n_free]``).
    constraints : ParameterConstraints | None
        Constraint specification.
    n_total : int
        Total parameter count (``2 * n_epochs``).
    free_indices : list[int]
        Precomputed free-param indices.

    Returns
    -------
    torch.Tensor
        Expanded batch, shape ``[..., n_total]``.

    """
    if constraints is None:
        return free_log_samples

    pos_of = {idx: pos for pos, idx in enumerate(free_indices)}
    rep_of: dict[int, int] = {
        idx: min(g) for g in constraints.equality_groups for idx in g
    }

    batch_shape = free_log_samples.shape[:-1]
    full = torch.zeros(
        *batch_shape,
        n_total,
        dtype=free_log_samples.dtype,
        device=free_log_samples.device,
    )

    for idx in range(n_total):
        if idx in constraints.fixed_params:
            val = constraints.fixed_params[idx]
            log_val = math.log(val) if val > 0.0 else _LOG_ZERO
            full[..., idx] = log_val
        else:
            src = rep_of.get(idx, idx)
            full[..., idx] = free_log_samples[..., pos_of[src]]

    return full


# ---------------------------------------------------------------------------
# Pyro model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianLikelihoodModel:
    """Pickle-safe callable Pyro model for multiprocessing MCMC chains.

    Samples a single ``free_log_rates`` site of shape ``[n_free]``, expands
    it to the full ``[2 * n_epochs]`` log-rate vector (applying any
    ``ParameterConstraints`` differentiably), and injects the existing
    PyTorch log-likelihood via ``pyro.factor``.
    """

    model_data: BayesianModelData

    def __call__(self) -> None:
        """Evaluate the Pyro model for one MCMC draw."""
        free_log_rates = pyro.sample(
            "free_log_rates",
            dist.Normal(
                self.model_data.free_prior_locs,
                self.model_data.free_prior_scales,
            ).to_event(1),
        )

        n_epochs = self.model_data.n_epochs
        n_total = 2 * n_epochs
        full_log_rates = _expand_log_rates(
            free_log_rates,
            self.model_data.constraints,
            n_total,
            self.model_data.free_indices,
        )

        log_birth = full_log_rates[:n_epochs]
        log_death = full_log_rates[n_epochs:]

        try:
            total_log_lik = log_likelihood(
                birth_rates=log_birth,
                death_rates=log_death,
                ages=self.model_data.non_spine_ages,
                times=self.model_data.times_tensor,
                spine_ages=self.model_data.spine_ages_tensor,
                ns=self.model_data.population_sizes_tensor,
                initial_population=self.model_data.initial_population,
                unbias=self.model_data.unbias,
                is_ordered=False,
                log_scale_params=True,
            )
        except (ValueError, RuntimeError):
            total_log_lik = torch.tensor(float("-inf"), dtype=torch.float64)

        safe_log_lik = torch.where(
            torch.isfinite(total_log_lik),
            total_log_lik,
            torch.tensor(-1e20, dtype=torch.float64),
        )
        pyro.factor("obs_log_likelihood", safe_log_lik)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_age_inputs(
    list_of_ages: list[torch.Tensor | NDArray[np.float64] | list[float]],
    *,
    is_ordered: bool,
    spine_ages: torch.Tensor | None,
    population_sizes: torch.Tensor | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Prepare non-spine ages, spine ages, and population-size vector.

    Parameters
    ----------
    list_of_ages : list
        Age observations grouped by population/epoch.
    is_ordered : bool
        If ``True``, the first age of each group is the spine.
    spine_ages : torch.Tensor | None
        Explicit spine ages (mutually exclusive with ``is_ordered``).
    population_sizes : torch.Tensor | None
        Override per-group population counts.
    device : str
        Target device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
        ``(non_spine_ages, spine_ages_tensor, population_sizes_tensor)`` — all
        on ``device``, float64 (int64 for counts).

    """
    age_tensors = [
        ensure_1d_tensor(torch.as_tensor(ages, dtype=torch.float64).to(device))
        for ages in list_of_ages
    ]
    if not age_tensors or not any(t.numel() > 0 for t in age_tensors):
        msg = "No valid data points for fitting"
        raise ValueError(msg)

    if is_ordered and spine_ages is not None:
        msg = "Cannot have both is_ordered=True and spine_ages provided."
        raise ValueError(msg)

    inferred_population_sizes = torch.tensor(
        [len(ages) for ages in age_tensors],
        dtype=torch.int64,
        device=device,
    )

    spine_ages_tensor: torch.Tensor | None = (
        ensure_1d_tensor(spine_ages.to(device)) if spine_ages is not None else None
    )

    non_spine_ages: torch.Tensor
    if is_ordered:
        spine_list: list[torch.Tensor] = []
        non_spine_list: list[torch.Tensor] = []
        for ages in age_tensors:
            if len(ages) == 0:
                continue
            spine_list.append(ages[:1])
            if len(ages) > 1:
                non_spine_list.append(ages[1:])

        spine_ages_tensor = (
            torch.cat(spine_list).to(dtype=torch.float64, device=device)
            if spine_list
            else torch.empty(0, dtype=torch.float64, device=device)
        )
        non_spine_ages = (
            torch.cat(non_spine_list).to(dtype=torch.float64, device=device)
            if non_spine_list
            else torch.empty(0, dtype=torch.float64, device=device)
        )
    else:
        non_spine_ages = torch.cat(age_tensors).to(dtype=torch.float64, device=device)

    if population_sizes is None:
        population_sizes_tensor = inferred_population_sizes
    else:
        population_sizes_tensor = torch.as_tensor(
            population_sizes, dtype=torch.int64, device=device
        )
        if len(population_sizes_tensor) != len(age_tensors):
            msg = (
                "Length of population_sizes must match list_of_ages length. "
                f"Got {len(population_sizes_tensor)} and {len(age_tensors)}."
            )
            raise ValueError(msg)

    return non_spine_ages, spine_ages_tensor, population_sizes_tensor


def _run_nuts_mcmc(
    model: Callable[[], None],
    mcmc_config: BayesianMCMCConfig,
    rng_seed: int | None,
    device: str,
    *,
    init_log_rates: torch.Tensor | None = None,
) -> MCMC:
    """Configure and run Pyro NUTS, returning the fitted MCMC object."""
    if device != "cpu" and mcmc_config.num_chains > 1:
        warnings.warn(
            f"GPU device '{device}' with num_chains={mcmc_config.num_chains}: "
            "Pyro spawns OS processes for multi-chain runs and CUDA contexts are "
            "not safely forkable.  Set num_chains=1 for GPU inference.",
            UserWarning,
            stacklevel=3,
        )

    if rng_seed is not None:
        pyro.set_rng_seed(rng_seed)

    nuts_kernel = NUTS(
        model,
        target_accept_prob=mcmc_config.target_accept_prob,
        max_tree_depth=mcmc_config.max_tree_depth,
        jit_compile=mcmc_config.jit_compile,
        ignore_jit_warnings=True,
    )

    initial_params: dict[str, torch.Tensor] | None = None
    if init_log_rates is not None:
        init = init_log_rates.to(dtype=torch.float64, device=device)
        if mcmc_config.num_chains > 1:
            # Different per-chain start to break symmetry; ±0.1 log-space noise
            gen = torch.Generator()
            if rng_seed is not None:
                gen.manual_seed(rng_seed)
            noise = (
                torch.randn(
                    mcmc_config.num_chains,
                    init.shape[0],
                    dtype=torch.float64,
                    device=device,
                    generator=gen,
                )
                * 0.1
            )
            init = init.unsqueeze(0).expand(mcmc_config.num_chains, -1) + noise
        initial_params = {"free_log_rates": init}

    mp_context = "spawn" if mcmc_config.num_chains > 1 else None
    mcmc = MCMC(
        nuts_kernel,
        num_samples=mcmc_config.num_samples,
        warmup_steps=mcmc_config.warmup_steps,
        num_chains=mcmc_config.num_chains,
        disable_progbar=mcmc_config.disable_progressbar,
        mp_context=mp_context,
        initial_params=initial_params,
    )
    mcmc.run()
    return mcmc


def _num_divergences_from_diagnostics(diagnostics: dict[str, Any]) -> int:
    """Extract total divergence count from Pyro diagnostics payload."""
    divergences = diagnostics.get("divergences", {})
    if isinstance(divergences, dict):
        return sum(len(chain_divs) for chain_divs in divergences.values())
    if isinstance(divergences, Sequence):
        return len(divergences)
    return int(divergences) if divergences else 0


def _metric_as_array(
    summary_dict: dict[str, Any],
    metric_name: str,
    expected_length: int,
) -> NDArray[np.float64]:
    """Normalize a scalar/vector diagnostic metric into a fixed-length ndarray."""
    metric_value = summary_dict.get(metric_name)
    if metric_value is None:
        return np.full(expected_length, np.nan, dtype=np.float64)
    if isinstance(metric_value, torch.Tensor):
        metric_value = metric_value.detach().cpu()
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
    birth_stats = posterior_summary_by_chain["birth_rates"]
    death_stats = posterior_summary_by_chain["death_rates"]
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
) -> torch.Tensor:
    """Extract free log-rates from an MLE result for use as NUTS initialization.

    Converts the MLE birth/death rate estimates to log-space, assembles the
    full ``[2 * n_epochs]`` log-rate vector, and slices to the free-parameter
    indices under ``constraints``.  The result can be passed directly as
    ``BayesianModelConfig(init_log_rates=...)``.

    Parameters
    ----------
    mle_result : dict
        Return value of :func:`inf_bp_texts.inference.fit_birth_death_processes`.
    constraints : ParameterConstraints | None
        Same constraints used for the Bayesian fit.  ``None`` → all parameters
        are free.
    n_epochs : int
        Number of epochs.

    Returns
    -------
    torch.Tensor
        Free log-rates, shape ``[n_free]``, float64.

    """
    n_total = 2 * n_epochs
    birth = np.asarray(mle_result["birth_rates"], dtype=np.float64)
    death = np.asarray(mle_result["death_rates"], dtype=np.float64)
    full_rates = np.concatenate([birth, death])
    # Clamp to avoid log(0)
    full_rates = np.clip(full_rates, 1e-300, None)
    log_full = torch.tensor(np.log(full_rates), dtype=torch.float64)

    if constraints is None:
        return log_full
    free_indices = constraints.get_free_param_indices(n_total)
    return log_full[free_indices]


def fit_birth_death_processes_bayesian(
    list_of_ages: list[torch.Tensor | NDArray[np.float64] | list[float]],
    times: torch.Tensor | float,
    *,
    model_config: BayesianModelConfig | None = None,
    mcmc_config: BayesianMCMCConfig | None = None,
    prior_config: BayesianPriorConfig | None = None,
) -> dict[str, Any]:
    """Fit birth-death rates using Bayesian inference with Pyro NUTS.

    Parameters
    ----------
    list_of_ages : list[torch.Tensor | ndarray | list[float]]
        Age observations grouped by population/epoch.
    times : torch.Tensor | float
        Observation time(s) — one value per epoch.
    model_config : BayesianModelConfig | None
        Model options: ordered inputs, spine ages, population sizes, initial
        population, unbias, constraints, RNG seed, and device.  Defaults to
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
            Pyro ``summary()`` dict (rate-space, flat).
        ``"posterior_summary_by_chain"``
            Pyro ``summary()`` dict (rate-space, by-chain).
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

    device = model_config.device

    if model_config.initial_population <= 0:
        msg = "initial_population must be a positive integer"
        raise ValueError(msg)

    (times_tensor,) = normalize_and_validate_inputs(times=times)
    times_tensor = times_tensor.to(device)

    n_epochs = len(times_tensor)
    n_total = 2 * n_epochs

    # Validate constraints before building anything
    constraints = model_config.constraints
    if constraints is not None:
        constraints.validate(n_total)
        free_indices = constraints.get_free_param_indices(n_total)
    else:
        free_indices = list(range(n_total))

    # Resolve prior tensors for all 2*n_epochs params, then slice to free
    all_prior_locs, all_prior_scales = prior_config.resolve_tensors(n_epochs, device)
    free_prior_locs = all_prior_locs[free_indices]
    free_prior_scales = all_prior_scales[free_indices]

    # Prepare age and population data
    non_spine_ages, spine_ages_tensor, population_sizes_tensor = _prepare_age_inputs(
        list_of_ages,
        is_ordered=model_config.is_ordered,
        spine_ages=model_config.spine_ages,
        population_sizes=model_config.population_sizes,
        device=device,
    )

    model_data = BayesianModelData(
        n_epochs=n_epochs,
        non_spine_ages=non_spine_ages,
        times_tensor=times_tensor,
        spine_ages_tensor=spine_ages_tensor,
        population_sizes_tensor=population_sizes_tensor,
        initial_population=model_config.initial_population,
        unbias=model_config.unbias,
        free_prior_locs=free_prior_locs,
        free_prior_scales=free_prior_scales,
        constraints=constraints,
        free_indices=free_indices,
        device=device,
    )

    model = BayesianLikelihoodModel(model_data=model_data)

    mcmc = _run_nuts_mcmc(
        model,
        mcmc_config,
        model_config.rng_seed,
        device,
        init_log_rates=model_config.init_log_rates,
    )

    # ---- Extract and reshape posterior samples --------------------------------

    posterior_flat = mcmc.get_samples(group_by_chain=False)
    posterior_by_chain = mcmc.get_samples(group_by_chain=True)

    free_flat = posterior_flat["free_log_rates"].to(dtype=torch.float64)  # [S, n_free]
    free_by_chain = posterior_by_chain["free_log_rates"].to(
        dtype=torch.float64
    )  # [C, S, n_free]

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
    birth_flat = torch.exp(log_birth_flat)
    death_flat = torch.exp(log_death_flat)

    log_birth_by_chain = full_log_by_chain[..., :n_epochs]
    log_death_by_chain = full_log_by_chain[..., n_epochs:]
    birth_by_chain = torch.exp(log_birth_by_chain)
    death_by_chain = torch.exp(log_death_by_chain)

    # ---- Credible intervals --------------------------------------------------

    quantiles = torch.tensor(
        [0.05, 0.95], dtype=torch.float64, device=birth_flat.device
    )
    birth_ci = torch.quantile(birth_flat, quantiles, dim=0).transpose(0, 1)
    death_ci = torch.quantile(death_flat, quantiles, dim=0).transpose(0, 1)

    # ---- Divergences ---------------------------------------------------------

    diagnostics = mcmc.diagnostics()
    num_divergences = _num_divergences_from_diagnostics(diagnostics)

    # ---- Pyro summary (rate-space, for diagnostics) -------------------------

    posterior_summary = summary(
        {"birth_rates": birth_flat, "death_rates": death_flat},
        prob=0.9,
        group_by_chain=False,
    )
    posterior_summary_by_chain = summary(
        {"birth_rates": birth_by_chain, "death_rates": death_by_chain},
        prob=0.9,
        group_by_chain=True,
    )
    diagnostics_payload = _build_diagnostics_payload(
        posterior_summary_by_chain,
        n_epochs=n_epochs,
    )

    return {
        "success": True,
        "birth_rates": birth_flat.mean(dim=0).cpu().numpy(),
        "death_rates": death_flat.mean(dim=0).cpu().numpy(),
        "birth_rates_90_ci": birth_ci.cpu().numpy(),
        "death_rates_90_ci": death_ci.cpu().numpy(),
        "posterior_samples": {
            "birth_rates": birth_flat.cpu().numpy(),
            "death_rates": death_flat.cpu().numpy(),
        },
        "posterior_samples_by_chain": {
            "birth_rates": birth_by_chain.cpu().numpy(),
            "death_rates": death_by_chain.cpu().numpy(),
        },
        "posterior_log_samples": {
            "birth_rates": log_birth_flat.cpu().numpy(),
            "death_rates": log_death_flat.cpu().numpy(),
        },
        "posterior_log_samples_by_chain": {
            "birth_rates": log_birth_by_chain.cpu().numpy(),
            "death_rates": log_death_by_chain.cpu().numpy(),
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


def fit_birth_death_processes_bayesian_from_data_frame(
    df: pd.DataFrame,
    times: float | NDArray[np.float64] | torch.Tensor,
    *,
    age_column: str | None = None,
    **fit_kwargs: Unpack[BayesianFitKwargs],
) -> dict[str, Any]:
    """Run Bayesian birth-death inference from a DataFrame age column.

    Automatically detects an ``age``, ``ages``, or ``age_*`` column if
    ``age_column`` is not specified.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing age observations.
    times : float | ndarray | torch.Tensor
        Observation time(s).
    age_column : str | None
        Column name containing ages.  Auto-detected if ``None``.
    **fit_kwargs : BayesianFitKwargs
        Forwarded to :func:`fit_birth_death_processes_bayesian`.

    Returns
    -------
    dict[str, Any]
        Same return schema as :func:`fit_birth_death_processes_bayesian`.

    """
    if age_column is None or age_column not in df.columns:
        exact_matches = [col for col in df.columns if col in ("age", "ages")]
        prefix_matches = [col for col in df.columns if col.startswith("age_")]
        candidates = exact_matches or prefix_matches
        if not candidates:
            msg = (
                f"Column '{age_column if age_column is not None else 'of ages'}' "
                "not found, and no 'age', 'ages', or 'age_*' columns were found."
            )
            raise ValueError(msg)
        if len(candidates) > 1:
            msg = f"Multiple age columns found: {candidates}"
            raise ValueError(msg)
        age_column = candidates[0]

    ages_array = df[age_column].to_numpy(dtype=np.float64, copy=True)
    ages_tensor = torch.as_tensor(ages_array, dtype=torch.float64)
    times_tensor = torch.as_tensor(
        np.asarray(times, dtype=np.float64), dtype=torch.float64
    )

    return fit_birth_death_processes_bayesian(
        list_of_ages=[ages_tensor],
        times=times_tensor,
        **fit_kwargs,
    )
