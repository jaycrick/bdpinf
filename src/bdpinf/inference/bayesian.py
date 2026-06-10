"""Bayesian inference utilities for birth-death process parameters using Pyro."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, Unpack

import numpy as np
import pandas as pd
import pyro  # type: ignore[import-not-found]
import pyro.distributions as dist  # type: ignore[import-not-found]
import torch
from numpy.typing import NDArray
from pyro.infer import MCMC, NUTS  # type: ignore[import-not-found]
from pyro.infer.mcmc.util import summary  # type: ignore[import-not-found]

from inf_bp_texts.models.validation import normalize_and_validate_inputs
from inf_bp_texts.utils import ensure_1d_tensor

from .likelihood import log_likelihood


@dataclass(frozen=True)
class BayesianPriorConfig:
    """Prior configuration for Bayesian birth-death inference."""

    loc: float = -6.0
    scale: float = 2.0


@dataclass(frozen=True)
class BayesianMCMCConfig:
    """MCMC configuration for Pyro NUTS."""

    num_samples: int = 800
    warmup_steps: int = 400
    num_chains: int = 1
    target_accept_prob: float = 0.7
    max_tree_depth: int = 10
    jit_compile: bool = False
    disable_progressbar: bool = True


@dataclass(frozen=True)
class BayesianModelConfig:
    """Model-level options for birth-death Bayesian inference."""

    is_ordered: bool = False
    spine_ages: torch.Tensor | None = None
    population_sizes: torch.Tensor | None = None
    initial_population: int = 1
    unbias: bool = True
    rng_seed: int | None = None


class BayesianFitKwargs(TypedDict, total=False):
    """Typed kwargs for Bayesian fitting wrappers."""

    model_config: BayesianModelConfig
    mcmc_config: BayesianMCMCConfig
    prior_config: BayesianPriorConfig


@dataclass(frozen=True)
class BayesianModelData:
    """Normalized tensors required to evaluate Bayesian model likelihood."""

    n_epochs: int
    non_spine_ages: torch.Tensor
    times_tensor: torch.Tensor
    spine_ages_tensor: torch.Tensor | None
    population_sizes_tensor: torch.Tensor
    initial_population: int
    unbias: bool


@dataclass(frozen=True)
class BayesianLikelihoodModel:
    """Pickle-safe callable Pyro model for multiprocessing MCMC chains."""

    model_data: BayesianModelData
    prior_config: BayesianPriorConfig

    def __call__(self) -> None:
        """Evaluate the Pyro model likelihood for one MCMC draw."""
        prior_loc_tensor = torch.tensor(self.prior_config.loc, dtype=torch.float64)
        prior_scale_tensor = torch.tensor(self.prior_config.scale, dtype=torch.float64)

        log_birth_rates = pyro.sample(
            "log_birth_rates",
            dist.Normal(prior_loc_tensor, prior_scale_tensor)
            .expand([self.model_data.n_epochs])
            .to_event(1),
        )
        log_death_rates = pyro.sample(
            "log_death_rates",
            dist.Normal(prior_loc_tensor, prior_scale_tensor)
            .expand([self.model_data.n_epochs])
            .to_event(1),
        )
        try:
            total_log_lik = log_likelihood(
                birth_rates=log_birth_rates,
                death_rates=log_death_rates,
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


def _prepare_age_inputs(
    list_of_ages: list[torch.Tensor | NDArray[np.float64] | list[float]],
    *,
    is_ordered: bool,
    spine_ages: torch.Tensor | None,
    population_sizes: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Prepare non-spine ages, spine ages, and population-size vector."""
    age_tensors = [
        ensure_1d_tensor(torch.as_tensor(ages, dtype=torch.float64))
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
    )

    non_spine_ages: torch.Tensor
    spine_ages_tensor = ensure_1d_tensor(spine_ages) if spine_ages is not None else None

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
            torch.cat(spine_list).to(dtype=torch.float64)
            if spine_list
            else torch.empty(0, dtype=torch.float64)
        )
        non_spine_ages = (
            torch.cat(non_spine_list).to(dtype=torch.float64)
            if non_spine_list
            else torch.empty(0, dtype=torch.float64)
        )
    else:
        non_spine_ages = torch.cat(age_tensors).to(dtype=torch.float64)

    if population_sizes is None:
        population_sizes_tensor = inferred_population_sizes
    else:
        population_sizes_tensor = torch.as_tensor(population_sizes, dtype=torch.int64)
        if len(population_sizes_tensor) != len(age_tensors):
            msg = (
                "Length of population_sizes must match list_of_ages length. "
                f"Got {len(population_sizes_tensor)} and {len(age_tensors)}."
            )
            raise ValueError(msg)

    return non_spine_ages, spine_ages_tensor, population_sizes_tensor


def _build_pyro_model(
    model_data: BayesianModelData,
    prior_config: BayesianPriorConfig,
) -> Callable[[], None]:
    """Create a Pyro model closure for NUTS sampling."""
    return BayesianLikelihoodModel(model_data=model_data, prior_config=prior_config)


def _run_nuts_mcmc(
    model: Callable[[], None],
    mcmc_config: BayesianMCMCConfig,
    rng_seed: int | None,
) -> MCMC:
    """Run NUTS MCMC and return the fitted MCMC object."""
    if rng_seed is not None:
        pyro.set_rng_seed(rng_seed)

    nuts_kernel = NUTS(
        model,
        target_accept_prob=mcmc_config.target_accept_prob,
        max_tree_depth=mcmc_config.max_tree_depth,
        jit_compile=mcmc_config.jit_compile,
        ignore_jit_warnings=True,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=mcmc_config.num_samples,
        warmup_steps=mcmc_config.warmup_steps,
        num_chains=mcmc_config.num_chains,
        disable_progbar=mcmc_config.disable_progressbar,
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
    """Normalize scalar/vector diagnostics into a fixed-length ndarray."""
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
    """Build r-hat and effective sample size arrays for each rate family."""
    birth_chain_stats = posterior_summary_by_chain["birth_rates"]
    death_chain_stats = posterior_summary_by_chain["death_rates"]
    return {
        "birth_rates": {
            "r_hat": _metric_as_array(birth_chain_stats, "r_hat", n_epochs),
            "n_eff": _metric_as_array(birth_chain_stats, "n_eff", n_epochs),
        },
        "death_rates": {
            "r_hat": _metric_as_array(death_chain_stats, "r_hat", n_epochs),
            "n_eff": _metric_as_array(death_chain_stats, "n_eff", n_epochs),
        },
    }


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
        Observation time(s), one value per epoch.
    model_config : BayesianModelConfig | None
        Model options such as ordered input handling and initial population.
    mcmc_config : BayesianMCMCConfig | None
        NUTS and MCMC sampler configuration.
    prior_config : BayesianPriorConfig | None
        Normal prior location/scale on log birth/death rates.

    Returns
    -------
    dict[str, Any]
        Posterior summaries and samples for birth/death rates.

    """
    model_config = model_config or BayesianModelConfig()
    mcmc_config = mcmc_config or BayesianMCMCConfig()
    prior_config = prior_config or BayesianPriorConfig()

    if model_config.initial_population <= 0:
        msg = "initial_population must be a positive integer"
        raise ValueError(msg)

    (times_tensor,) = normalize_and_validate_inputs(times=times)
    non_spine_ages, spine_ages_tensor, population_sizes_tensor = _prepare_age_inputs(
        list_of_ages,
        is_ordered=model_config.is_ordered,
        spine_ages=model_config.spine_ages,
        population_sizes=model_config.population_sizes,
    )

    n_epochs = len(times_tensor)
    model_data = BayesianModelData(
        n_epochs=n_epochs,
        non_spine_ages=non_spine_ages,
        times_tensor=times_tensor,
        spine_ages_tensor=spine_ages_tensor,
        population_sizes_tensor=population_sizes_tensor,
        initial_population=model_config.initial_population,
        unbias=model_config.unbias,
    )
    model = _build_pyro_model(
        model_data,
        prior_config,
    )
    mcmc = _run_nuts_mcmc(
        model,
        mcmc_config,
        model_config.rng_seed,
    )

    posterior = mcmc.get_samples(group_by_chain=False)
    posterior_by_chain = mcmc.get_samples(group_by_chain=True)

    log_birth_samples = posterior["log_birth_rates"].to(dtype=torch.float64)
    log_death_samples = posterior["log_death_rates"].to(dtype=torch.float64)
    birth_samples = torch.exp(log_birth_samples)
    death_samples = torch.exp(log_death_samples)

    log_birth_samples_by_chain = posterior_by_chain["log_birth_rates"].to(
        dtype=torch.float64,
    )
    log_death_samples_by_chain = posterior_by_chain["log_death_rates"].to(
        dtype=torch.float64,
    )
    birth_samples_by_chain = torch.exp(log_birth_samples_by_chain)
    death_samples_by_chain = torch.exp(log_death_samples_by_chain)

    quantiles = torch.tensor([0.05, 0.95], dtype=torch.float64)
    birth_ci = torch.quantile(birth_samples, quantiles, dim=0).transpose(0, 1)
    death_ci = torch.quantile(death_samples, quantiles, dim=0).transpose(0, 1)

    diagnostics = mcmc.diagnostics()
    num_divergences = _num_divergences_from_diagnostics(diagnostics)

    posterior_summary = summary(
        {
            "birth_rates": birth_samples,
            "death_rates": death_samples,
        },
        prob=0.9,
        group_by_chain=False,
    )

    posterior_summary_by_chain = summary(
        {
            "birth_rates": birth_samples_by_chain,
            "death_rates": death_samples_by_chain,
        },
        prob=0.9,
        group_by_chain=True,
    )
    diagnostics_payload = _build_diagnostics_payload(
        posterior_summary_by_chain,
        n_epochs=n_epochs,
    )

    return {
        "success": True,
        "birth_rates": birth_samples.mean(dim=0).cpu().numpy(),
        "death_rates": death_samples.mean(dim=0).cpu().numpy(),
        "birth_rates_90_ci": birth_ci.cpu().numpy(),
        "death_rates_90_ci": death_ci.cpu().numpy(),
        "posterior_samples": {
            "birth_rates": birth_samples.cpu().numpy(),
            "death_rates": death_samples.cpu().numpy(),
        },
        "posterior_samples_by_chain": {
            "birth_rates": birth_samples_by_chain.cpu().numpy(),
            "death_rates": death_samples_by_chain.cpu().numpy(),
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
    }


def fit_birth_death_processes_bayesian_from_data_frame(
    df: pd.DataFrame,
    times: float | NDArray[np.float64] | torch.Tensor,
    *,
    age_column: str | None = None,
    **fit_kwargs: Unpack[BayesianFitKwargs],
) -> dict[str, Any]:
    """Run Bayesian birth-death inference from a DataFrame age column."""
    if age_column is None or age_column not in df.columns:
        exact_matches = [col for col in df.columns if col in ("age", "ages")]
        prefix_matches = [col for col in df.columns if col.startswith("age_")]
        candidates = exact_matches or prefix_matches
        if not candidates:
            msg = (
                f"Column '{age_column if age_column is not None else 'of ages'}' "
                "not found, and no 'age', 'ages', or 'age_*' "
                "columns were found."
            )
            raise ValueError(msg)
        if len(candidates) > 1:
            msg = f"Multiple age columns found: {candidates}"
            raise ValueError(msg)
        age_column = candidates[0]

    ages_tensor = torch.as_tensor(
        df[age_column].to_numpy(dtype=np.float64, copy=True),
        dtype=torch.float64,
    )
    times_tensor, ages_tensor = normalize_and_validate_inputs(
        times=torch.as_tensor(times, dtype=torch.float64),
        ages=ages_tensor,
    )

    return fit_birth_death_processes_bayesian(
        list_of_ages=[ages_tensor],
        times=times_tensor,
        **fit_kwargs,
    )
