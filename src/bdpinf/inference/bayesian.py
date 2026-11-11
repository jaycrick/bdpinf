"""Bayesian inference for birth-death rate models using NumPyro NUTS.

:func:`fit_bayesian` samples the posterior of the parameters of a
:class:`~bdpinf.inference.rate_models.BirthDeathModel` given
:class:`~bdpinf.inference.likelihood.AgeData`, with the same parameterisation and
:class:`~bdpinf.inference.constraints.Constraints` as the maximum likelihood
fitter :func:`~bdpinf.inference.fitting.fit`.

Key design choices
------------------
* A single ``free_params`` sample site holds the reduced-dimension latent vector
  on the unconstrained scale (log for rates), with independent Normal priors
  (:class:`BayesianPriorConfig`).
  The full natural-scale parameter vector is rebuilt differentiably inside the
  model by :meth:`CompiledConstraints.expand`, so equality-group and fixed
  constraints are honoured *during* sampling rather than post-hoc.
* Multi-chain inference uses ``chain_method="vectorized"`` (JAX vmap over
  chains) which is safe on both CPU and GPU without subprocess overhead.
* Data tensors are stored as **NumPy arrays** so that they remain concrete (not
  abstract tracers) inside NumPyro's JIT-compiled ``lax.while_loop``.
* The model calls the shared likelihood core
  (:func:`~bdpinf.inference.likelihood.log_likelihood_rates`) on pre-validated
  :class:`AgeData`, avoiding host-side validation that fails under JIT.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpy.typing import NDArray
from numpyro.infer import MCMC, NUTS

from bdpinf.inference.constraints import CompiledConstraints, Constraints
from bdpinf.inference.fitting import FitResult
from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.inference.rate_models import BirthDeathModel
from bdpinf.models.math_general import DEFAULT_NODES

__all__ = [
    "BayesianMCMCConfig",
    "BayesianPriorConfig",
    "BayesianResult",
    "fit_bayesian",
    "sample_prior",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianPriorConfig:
    """Independent Normal priors on the unconstrained parameters.

    For rate (log-scale) parameters the prior is on the log rate; for a rate
    family with a minimum value ``m > 0`` (``min_birth``, ``min_death``) it is on
    ``log(rate - m)``, the unconstrained coordinate, so the rate is always above
    ``m``.
    The scalar ``loc`` / ``scale`` fields are fallbacks for every parameter.
    Per-family overrides (``birth_loc``, ``death_scale``, ...) take precedence;
    each may be a scalar (broadcast to the family's parameters) or a sequence
    with one value per parameter of that family.
    ``by_name`` overrides single parameters by their full name and wins over
    everything else except ``exponential``, which replaces the Normal prior of
    its parameters altogether.

    Parameters
    ----------
    loc : float
        Fallback prior mean.
    scale : float
        Fallback prior standard deviation.
    birth_loc, birth_scale, death_loc, death_scale : float | Sequence[float] | None
        Per-family overrides; ``None`` falls back to ``loc`` / ``scale``.
    by_name : Mapping[str, tuple[float, float]]
        ``{"birth.v0": (loc, scale), ...}``.
    knot_loc, knot_scale : float
        Normal prior on the logit of each free knot time within its range.
        The default (0, 1.7) is close to uniform over the range.
    exponential : Mapping[str, float]
        ``{"birth.v3": 0.01, ...}``: parameter name to the mean of an
        Exponential prior on the natural-scale rate.
        The density on the unconstrained log scale includes the Jacobian
        (``log p(u) = -log m - exp(u) / m + u``).
        It overrides every Normal setting for that name.
        Only log-transform (rate) parameters without a minimum value are allowed
        (``ValueError`` if the family has ``min_value > 0``), they must not be fixed,
        and the name of a tied group member puts the prior on the group's single
        free coordinate.

    Examples
    --------
    Weakly informative, tighter prior on death rates:

    >>> cfg = BayesianPriorConfig(loc=-6.0, scale=3.0, death_scale=1.5)

    Per-epoch birth priors for a two-epoch piecewise-constant model:

    >>> cfg = BayesianPriorConfig(birth_loc=[-5.0, -7.0], birth_scale=1.5)

    """

    loc: float = -6.0
    scale: float = 2.0
    birth_loc: float | Sequence[float] | None = None
    birth_scale: float | Sequence[float] | None = None
    death_loc: float | Sequence[float] | None = None
    death_scale: float | Sequence[float] | None = None
    by_name: Mapping[str, tuple[float, float]] = dataclasses.field(default_factory=dict)
    knot_loc: float = 0.0
    knot_scale: float = 1.7
    exponential: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def resolve_tensors(
        self, n_epochs: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(locs, scales)`` of shape ``[2 * n_epochs]``.

        For a piecewise-constant model.
        Index ``i < n_epochs`` is the birth rate of epoch ``i``, the rest are the
        death rates; equivalent to :meth:`resolve` for
        ``BirthDeathModel.piecewise_constant`` with ``n_epochs`` epochs.
        """
        return self.resolve(
            BirthDeathModel.piecewise_constant([float(k + 1) for k in range(n_epochs)])
        )

    def resolve(
        self, model: BirthDeathModel
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(locs, scales)`` over every parameter of ``model``.

        Raises
        ------
        ValueError
            If a sequence override has the wrong length, ``by_name`` has an
            unknown name, or ``exponential`` names an unknown or non-log
            parameter or has a non-positive mean.

        Notes
        -----
        The returned Normal settings ignore ``exponential``; use
        :meth:`resolve_exponential` for those parameters.

        """
        self._check_exponential(model)

        def _family(
            override: float | Sequence[float] | None, fallback: float, n: int
        ) -> NDArray[np.float64]:
            if override is None:
                return np.full((n,), fallback, dtype=np.float64)
            if isinstance(override, (int, float)):
                return np.full((n,), float(override), dtype=np.float64)
            vec = np.array(list(override), dtype=np.float64)
            if vec.shape[0] != n:
                msg = f"Prior sequence length {vec.shape[0]} does not match {n} values"
                raise ValueError(msg)
            return vec

        n_birth, n_death, n_knots = model.n_birth, model.n_death, model.n_knots
        locs = np.concatenate(
            [
                _family(self.birth_loc, self.loc, n_birth),
                _family(self.death_loc, self.loc, n_death),
                np.full(n_knots, self.knot_loc, dtype=np.float64),
            ]
        )
        scales = np.concatenate(
            [
                _family(self.birth_scale, self.scale, n_birth),
                _family(self.death_scale, self.scale, n_death),
                np.full(n_knots, self.knot_scale, dtype=np.float64),
            ]
        )
        for name, (loc, scale) in self.by_name.items():
            if name not in model.names:
                msg = f"Unknown parameter '{name}' in by_name"
                raise ValueError(msg)
            locs[model.names.index(name)] = loc
            scales[model.names.index(name)] = scale
        return locs, scales

    def _check_exponential(self, model: BirthDeathModel) -> None:
        """Validate the names and means in ``exponential`` against ``model``."""
        transforms = {p.name: p.transform for p in model.params}
        lowers = {p.name: p.lower for p in model.params}
        for name, mean in self.exponential.items():
            if name not in transforms:
                msg = f"Unknown parameter '{name}' in exponential"
                raise ValueError(msg)
            if transforms[name] != "log":
                msg = (
                    f"Exponential prior needs a log-transform parameter, but "
                    f"'{name}' uses '{transforms[name]}'"
                )
                raise ValueError(msg)
            if lowers[name] > 0:
                msg = (
                    f"Exponential prior on '{name}' is not supported with a "
                    f"minimum value ({lowers[name]}); use a Normal prior"
                )
                raise ValueError(msg)
            if not (np.isfinite(mean) and mean > 0):
                msg = f"Exponential prior mean for '{name}' must be positive"
                raise ValueError(msg)

    def resolve_exponential(
        self, model: BirthDeathModel, compiled: CompiledConstraints
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return ``(slots, means)`` of the free coordinates with Exponential priors.

        ``slots`` index the free (reduced) parameter vector, so a parameter in an
        equality group maps to the group's single free coordinate.

        Raises
        ------
        ValueError
            For an invalid ``exponential`` entry, a fixed parameter, or two
            tied parameters with different means.

        """
        self._check_exponential(model)
        index = {name: i for i, name in enumerate(model.names)}
        means: dict[int, float] = {}
        for name, mean in self.exponential.items():
            i = index[name]
            if compiled.is_fixed[i]:
                msg = f"Exponential prior on '{name}', which is fixed"
                raise ValueError(msg)
            slot = int(compiled.source[i])
            if slot in means and not np.isclose(means[slot], mean):
                msg = f"Tied parameters have different exponential means ('{name}')"
                raise ValueError(msg)
            means[slot] = float(mean)
        slots = np.array(sorted(means), dtype=np.int64)
        return slots, np.array([means[k] for k in slots], dtype=np.float64)


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


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianResult:
    """Posterior samples and diagnostics from :func:`fit_bayesian`.

    Attributes
    ----------
    model : BirthDeathModel
        The sampled family.
    compiled : CompiledConstraints
        Constraint maps used while sampling.
    samples_by_chain : NDArray
        Natural-scale draws, shape ``[n_chains, n_samples, n_params]``,
        ordered as ``model.params``.
    free_by_chain : NDArray
        Unconstrained draws of the sampled (free) parameters,
        shape ``[n_chains, n_samples, n_free]``.
    summary : dict[str, dict[str, Any]]
        NumPyro ``diagnostics.summary`` per non-fixed parameter name (mean, std,
        quantiles, ``n_eff``, ``r_hat``).
    num_divergences : int
        Divergent transitions across all chains.
    mcmc_config : dict[str, Any]
        Echo of the sampler settings.

    """

    model: BirthDeathModel
    compiled: CompiledConstraints
    samples_by_chain: NDArray[np.float64]
    free_by_chain: NDArray[np.float64]
    summary: dict[str, dict[str, Any]]
    num_divergences: int
    mcmc_config: dict[str, Any]

    @property
    def samples(self) -> NDArray[np.float64]:
        """Natural-scale draws pooled over chains, shape ``[n_draws, n_params]``."""
        return self.samples_by_chain.reshape(-1, self.samples_by_chain.shape[-1])

    @property
    def params(self) -> dict[str, float]:
        """Posterior means by parameter name."""
        means = self.samples.mean(axis=0)
        return dict(zip(self.model.names, (float(v) for v in means), strict=True))

    @property
    def birth_samples(self) -> NDArray[np.float64]:
        """Pooled draws of the birth parameters, shape ``[n_draws, n_birth]``."""
        return self.samples[:, : self.model.n_birth]

    @property
    def death_samples(self) -> NDArray[np.float64]:
        """Pooled draws of the death parameters, shape ``[n_draws, n_death]``."""
        return self.samples[
            :, self.model.n_birth : self.model.n_birth + self.model.n_death
        ]

    @property
    def knot_samples(self) -> NDArray[np.float64] | None:
        """Pooled draws of the free knot times (sorted), or ``None`` for fixed epochs.

        Shape ``[n_draws, n_knots]``.
        """
        if self.model.knots is None:
            return None
        return self.samples[:, self.model.n_birth + self.model.n_death :]

    @property
    def posterior_samples(self) -> dict[str, NDArray[np.float64]]:
        """Pooled draws as ``{"birth_rates": ..., "death_rates": ...}``.

        The layout expected by
        :func:`~bdpinf.inference.persistence.save_posterior_samples`
        (epoch rates for a piecewise-constant model).
        """
        return {"birth_rates": self.birth_samples, "death_rates": self.death_samples}

    def credible_intervals(self, prob: float = 0.9) -> dict[str, tuple[float, float]]:
        """Central credible intervals by parameter name."""
        lo, hi = np.quantile(self.samples, [0.5 - prob / 2, 0.5 + prob / 2], axis=0)
        return {
            name: (float(a), float(b))
            for name, a, b in zip(self.model.names, lo, hi, strict=True)
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the posterior as the dict layout used by the persistence helpers.

        With free knot times, ``knot_times`` (posterior means), ``knot_times_90_ci``
        and ``posterior_samples["knot_times"]`` are added.

        Rates are split into ``birth_rates`` and ``death_rates`` (the epoch rates
        of a piecewise-constant model; the knot values of other families), with
        posterior means, 90 % credible intervals, pooled and per-chain samples on
        the natural and log scale, and R-hat/ESS per family.
        Log samples floor the rates at $e^{-40}$,
        so a parameter fixed at zero stays finite.
        """
        n = self.model.n_birth
        birth_c = self.samples_by_chain[..., :n]
        death_c = self.samples_by_chain[..., n : n + self.model.n_death]
        birth, death = self.birth_samples, self.death_samples

        def log(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.log(np.clip(x, np.exp(-40.0), None))

        def family(names: list[str], key: str) -> NDArray[np.float64]:
            return np.array(
                [float(self.summary.get(name, {}).get(key, np.nan)) for name in names]
            )

        diagnostics = {
            fam: {
                "r_hat": family(names, "r_hat"),
                "n_eff": family(names, "n_eff"),
            }
            for fam, names in (
                ("birth_rates", self.model.birth_names),
                ("death_rates", self.model.death_names),
            )
        }
        knot_times = self.knot_samples
        out: dict[str, Any] = {
            "success": True,
            "birth_rates": birth.mean(axis=0),
            "death_rates": death.mean(axis=0),
            "birth_rates_90_ci": np.quantile(birth, [0.05, 0.95], axis=0).T,
            "death_rates_90_ci": np.quantile(death, [0.05, 0.95], axis=0).T,
            "posterior_samples": {"birth_rates": birth, "death_rates": death},
            "posterior_samples_by_chain": {
                "birth_rates": birth_c,
                "death_rates": death_c,
            },
            "posterior_log_samples": {
                "birth_rates": log(birth),
                "death_rates": log(death),
            },
            "posterior_log_samples_by_chain": {
                "birth_rates": log(birth_c),
                "death_rates": log(death_c),
            },
            "posterior_summary": self.summary,
            "diagnostics": diagnostics,
            "num_divergences": self.num_divergences,
            "mcmc_config": dict(self.mcmc_config),
            "n_epochs": n,
        }
        if knot_times is not None:
            out["knot_times"] = knot_times.mean(axis=0)
            out["knot_times_90_ci"] = np.quantile(knot_times, [0.05, 0.95], axis=0).T
            out["posterior_samples"]["knot_times"] = knot_times
        return out

    @property
    def r_hat(self) -> dict[str, float]:
        """Split R-hat by (non-fixed) parameter name."""
        return {k: float(v["r_hat"]) for k, v in self.summary.items()}

    @property
    def n_eff(self) -> dict[str, float]:
        """Effective sample size by (non-fixed) parameter name."""
        return {k: float(v["n_eff"]) for k, v in self.summary.items()}


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelData:
    """NumPy-backed copy of :class:`AgeData` (concrete inside NumPyro's loops)."""

    non_spine_ages: NDArray[np.float64]
    spine_ages: NDArray[np.float64] | None
    ns: NDArray[np.int32]
    end_time: float
    use_unknown_spine: bool
    initial_population: int
    upper_ages: NDArray[np.float64] | None = None
    weights: NDArray[np.float64] | None = None

    @classmethod
    def from_age_data(cls, data: AgeData) -> _ModelData:
        return cls(
            non_spine_ages=np.asarray(data.non_spine_ages, dtype=np.float64),
            spine_ages=(
                None
                if data.spine_ages is None
                else np.asarray(data.spine_ages, dtype=np.float64)
            ),
            ns=np.asarray(data.ns, dtype=np.int32),
            end_time=data.end_time,
            use_unknown_spine=data.use_unknown_spine,
            initial_population=data.initial_population,
            upper_ages=(
                None
                if data.upper_ages is None
                else np.asarray(data.upper_ages, dtype=np.float64)
            ),
            weights=(
                None
                if data.weights is None
                else np.asarray(data.weights, dtype=np.float64)
            ),
        )

    def to_age_data(self) -> AgeData:
        return AgeData(
            non_spine_ages=jnp.asarray(self.non_spine_ages),
            spine_ages=None
            if self.spine_ages is None
            else jnp.asarray(self.spine_ages),
            ns=jnp.asarray(self.ns),
            end_time=self.end_time,
            use_unknown_spine=self.use_unknown_spine,
            initial_population=self.initial_population,
            upper_ages=None
            if self.upper_ages is None
            else jnp.asarray(self.upper_ages),
            weights=None if self.weights is None else jnp.asarray(self.weights),
        )


@dataclass(frozen=True)
class _LikelihoodModel:
    """Callable NumPyro model for MCMC chains."""

    data: _ModelData
    model: BirthDeathModel
    compiled: CompiledConstraints
    prior_locs: NDArray[np.float64]  # [n_free]
    prior_scales: NDArray[np.float64]  # [n_free]
    condition_on_survival: bool
    n_nodes: int
    exp_slots: NDArray[np.int64] = dataclasses.field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    exp_means: NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def __call__(self) -> None:
        normal = dist.Normal(
            jnp.asarray(self.prior_locs), jnp.asarray(self.prior_scales)
        )
        if len(self.exp_slots):
            keep = np.ones(len(self.prior_locs), dtype=bool)
            keep[self.exp_slots] = False
            normal = normal.mask(jnp.asarray(keep))
        free = numpyro.sample("free_params", normal.to_event(1))
        if len(self.exp_slots):
            u = jnp.take(free, jnp.asarray(self.exp_slots))
            means = jnp.asarray(self.exp_means)
            numpyro.factor(
                "exponential_prior", jnp.sum(-jnp.log(means) - jnp.exp(u) / means + u)
            )
        birth, death = self.model.rates(self.compiled.expand(free))
        log_lik = log_likelihood_rates(
            self.data.to_age_data(),
            birth,
            death,
            condition_on_survival=self.condition_on_survival,
            n_nodes=self.n_nodes,
        )
        safe = jnp.where(jnp.isfinite(log_lik), log_lik, jnp.array(-1e20, jnp.float64))
        numpyro.factor("obs_log_likelihood", safe)


def _run_nuts(
    model: _LikelihoodModel,
    mcmc_config: BayesianMCMCConfig,
    rng_seed: int | None,
    init_free: NDArray[np.float64] | None,
) -> MCMC:
    """Configure and run NumPyro NUTS, returning the fitted MCMC object."""
    rng_key = jax.random.PRNGKey(rng_seed if rng_seed is not None else 0)
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
    if init_free is not None:
        init = np.asarray(init_free, dtype=np.float64)
        n_chains = mcmc_config.num_chains
        if n_chains > 1:
            noise_key, rng_key = jax.random.split(rng_key)
            noise = np.array(
                jax.random.normal(noise_key, shape=(n_chains, init.shape[0])) * 0.1
            )
            init = np.broadcast_to(init, (n_chains, init.shape[0])) + noise
        init_params = {"free_params": jnp.array(init)}
    mcmc.run(rng_key, init_params=init_params, extra_fields=("diverging",))
    return mcmc


def fit_bayesian(
    data: AgeData,
    model: BirthDeathModel,
    *,
    constraints: Constraints | None = None,
    prior: BayesianPriorConfig | None = None,
    mcmc: BayesianMCMCConfig | None = None,
    init: FitResult | NDArray[np.float64] | None = None,
    rng_seed: int | None = None,
    condition_on_survival: bool = True,
    n_nodes: int = DEFAULT_NODES,
) -> BayesianResult:
    """Sample the posterior of a birth-death rate model with NUTS.

    Parameters
    ----------
    data : AgeData
        Observed ages.
    model : BirthDeathModel
        Parametric family for the rates, on the same observation window.
    constraints : Constraints, optional
        Fixed parameters and equality groups; only free parameters are sampled.
    prior : BayesianPriorConfig, optional
        Normal priors on the unconstrained parameters, with Exponential priors
        on the natural-scale rate for the names in ``prior.exponential``.
    mcmc : BayesianMCMCConfig, optional
        Sampler settings.
    init : FitResult | array-like, optional
        Where to start NUTS: a maximum-likelihood fit of the same model, or a
        vector of free unconstrained values.
        For several chains a small per-chain perturbation is added.
    rng_seed : int, optional
        Seed for the JAX PRNG.
    condition_on_survival : bool
        Condition the population-count term on non-extinction.
    n_nodes : int
        Gauss-Legendre nodes for non-piecewise-constant rates.

    Returns
    -------
    BayesianResult
        With free knot times (:class:`~bdpinf.inference.rate_models.FreeKnots`)
        the knot draws are sorted within each draw, which for a shared range
        removes label switching exactly; ``free_by_chain`` is left unsorted.
        Piecewise-constant rates make the likelihood jump where a knot crosses
        an observed ``T - age``, which NUTS handles less efficiently (check
        ``num_divergences``); continuous piecewise-linear rates sample more
        smoothly.

    Raises
    ------
    ValueError
        For an unknown-spine marginal with ``initial_population > 1``, an
        invalid constraint, or a mismatched prior.

    """
    mcmc = mcmc or BayesianMCMCConfig()
    prior = prior or BayesianPriorConfig()
    if data.initial_population <= 0:
        msg = "initial_population must be a positive integer"
        raise ValueError(msg)
    if data.initial_population > 1 and data.use_unknown_spine:
        msg = (
            "Unknown-spine marginal is not supported for initial_population > 1. "
            "Provide explicit spine_ages or use is_ordered=True."
        )
        raise ValueError(msg)
    if abs(model.end_time - data.end_time) > 1e-9 * max(1.0, data.end_time):
        msg = "Model window does not match the data observation time."
        raise ValueError(msg)

    compiled = (constraints or Constraints()).compile(model.params)
    locs, scales = prior.resolve(model)
    exp_slots, exp_means = prior.resolve_exponential(model, compiled)
    likelihood_model = _LikelihoodModel(
        data=_ModelData.from_age_data(data),
        model=model,
        compiled=compiled,
        prior_locs=locs[compiled.free_indices],
        prior_scales=scales[compiled.free_indices],
        condition_on_survival=condition_on_survival,
        n_nodes=n_nodes,
        exp_slots=exp_slots,
        exp_means=exp_means,
    )

    init_free: NDArray[np.float64] | None
    if isinstance(init, FitResult):
        init_free = np.asarray(compiled.reduce(init.natural), dtype=np.float64)
    elif init is not None:
        init_free = np.asarray(init, dtype=np.float64)
    else:
        init_free = None

    sampler = _run_nuts(likelihood_model, mcmc, rng_seed, init_free)

    free_by_chain = np.asarray(
        sampler.get_samples(group_by_chain=True)["free_params"], dtype=np.float64
    )
    n_chains, n_draws, _ = free_by_chain.shape
    expand = jax.jit(jax.vmap(compiled.expand))
    natural = model.canonical(
        expand(jnp.asarray(free_by_chain.reshape(-1, free_by_chain.shape[-1])))
    ).reshape(n_chains, n_draws, -1)

    varying = {
        name: natural[..., i]
        for i, name in enumerate(model.names)
        if not compiled.is_fixed[i]
    }
    summary = numpyro.diagnostics.summary(varying, prob=0.9) if varying else {}
    extra = sampler.get_extra_fields().get("diverging")
    return BayesianResult(
        model=model,
        compiled=compiled,
        samples_by_chain=natural,
        free_by_chain=free_by_chain,
        summary=summary,
        num_divergences=0 if extra is None else int(jnp.sum(extra)),
        mcmc_config={
            "num_samples": mcmc.num_samples,
            "warmup_steps": mcmc.warmup_steps,
            "num_chains": mcmc.num_chains,
            "target_accept_prob": mcmc.target_accept_prob,
            "max_tree_depth": mcmc.max_tree_depth,
        },
    )


def sample_prior(
    model: BirthDeathModel,
    prior: BayesianPriorConfig | None = None,
    n: int = 1000,
    *,
    constraints: Constraints | None = None,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Draw from the prior that :func:`fit_bayesian` would use.

    Free coordinates are Normal on the unconstrained scale, except those named in
    ``prior.exponential``, which are drawn as Exponential rates; equality groups
    share one draw and fixed parameters repeat their fixed value.

    Parameters
    ----------
    model : BirthDeathModel
        The rate families (and free knots, if any).
    prior : BayesianPriorConfig, optional
        Prior settings (default: :class:`BayesianPriorConfig`).
    n : int
        Number of draws.
    constraints : Constraints, optional
        Fixed parameters and equality groups, as for :func:`fit_bayesian`.
    seed : int, optional
        Seed for the NumPy generator.

    Returns
    -------
    dict[str, NDArray]
        Natural-scale draws of shape ``[n]`` by full parameter name (knot times
        sorted within each draw).

    Raises
    ------
    ValueError
        For an invalid prior or constraint.

    """
    prior = prior or BayesianPriorConfig()
    compiled = (constraints or Constraints()).compile(model.params)
    locs, scales = prior.resolve(model)
    slots, means = prior.resolve_exponential(model, compiled)
    rng = np.random.default_rng(seed)
    free = rng.normal(
        locs[compiled.free_indices], scales[compiled.free_indices], (n, compiled.n_free)
    )
    if len(slots):
        free[:, slots] = np.log(rng.exponential(means, (n, len(slots))))
    expand = jax.jit(jax.vmap(compiled.expand))
    natural = model.canonical(expand(jnp.asarray(free)))
    return {name: natural[:, i] for i, name in enumerate(model.names)}
