"""Likelihood calculation functions for birth-death process parameter estimation.

This module contains functions for calculating log-likelihood values for
birth-death process models, used in parameter fitting and model evaluation.
"""

import dataclasses
import warnings
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from bdpinf.models.math_age_masses import range_log_masses
from bdpinf.models.math_general import (
    DEFAULT_NODES,
    AgeTerms,
    age_log_densities,
    non_spine_log_density,
    spine_log_density,
    terms_for,
)
from bdpinf.models.math_nested import nested_terms
from bdpinf.models.math_populations import (
    calculate_populations_log_lik_from_terms,
    check_conditioning,
)
from bdpinf.models.rate_functions import RateFunction
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
    $\Pr(A^*=T)$ rather than the interior density.
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


def range_bounds(
    early_year: ArrayLike,
    late_year: ArrayLike,
    start_year: float | None = None,
    observation_year: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ``[lower, upper)`` creation-year interval of every range.

    An integer-year range ``[early, late]`` is the interval
    ``[early, late + 1)``.
    With ``start_year`` and ``observation_year`` the interval is clipped to the
    support ``[start_year, observation_year]`` and kept at least one year wide,
    so a range lying wholly outside the support still contributes its nearest
    year.
    The bounds are aligned with the input rows: nothing is merged or reordered
    (unlike :meth:`AgeData.from_year_ranges`, whose ``from_ranges`` step
    merges equal ranges), so callers can attach them to a data frame.

    Parameters
    ----------
    early_year, late_year : array-like
        Earliest and latest creation year of each individual
        (``late_year`` below ``early_year`` is raised to it).
    start_year : float, optional
        First year of the support.
        Without it the lower bound is ``early_year``, not capped at ``upper - 1``.
    observation_year : float, optional
        Last year of the support, the cap on the upper bound.

    Returns
    -------
    lower, upper : tuple of numpy.ndarray
        Per-row interval bounds, in calendar years.

    Raises
    ------
    ValueError
        If the arrays differ in length.

    """
    early = np.asarray(early_year, dtype=np.float64).reshape(-1)
    late = np.asarray(late_year, dtype=np.float64).reshape(-1)
    if early.shape != late.shape:
        msg = "early_year and late_year must have the same length"
        raise ValueError(msg)
    late = np.maximum(late, early)
    upper = late + 1.0
    lower = early
    if observation_year is not None:
        upper = np.minimum(upper, observation_year)
    if start_year is not None:
        lower = np.maximum(np.minimum(lower, upper - 1.0), start_year)
    if observation_year is not None or start_year is not None:
        upper = np.maximum(upper, lower + 1.0)
    return lower, upper


@dataclasses.dataclass(frozen=True)
class AgeData:
    """Validated age observations in the form the likelihood core consumes.

    All host-side work (splitting off a labelled spine, detecting a spine at the
    observation time, population counts) happens once in the constructors,
    so evaluating the likelihood under ``jit``/``grad`` needs no host branches.

    Attributes
    ----------
    non_spine_ages : jax.Array
        Non-spine ages, or all ages when the spine is unknown.
    spine_ages : jax.Array | None
        Known spine ages.
    ns : jax.Array
        Population counts, one per group.
    end_time : float
        Observation time ``T``.
    use_unknown_spine : bool
        Whether the spine is marginalised over ``non_spine_ages``.
    initial_population : int
        Number of individuals at time 0.
    n_non_spine : int | jax.Array | None
        Number of genuine entries in ``non_spine_ages`` when it has been padded
        (see :meth:`padded`); ``None`` means all entries are genuine.
    upper_ages : jax.Array | None
        Set for interval-censored data (see :meth:`from_ranges`): each age is
        known only to lie in ``[non_spine_ages, upper_ages]``.
    weights : jax.Array | None
        For ranges, how many observations share each distinct range
        (zero for padding).

    """

    non_spine_ages: jax.Array
    spine_ages: jax.Array | None
    ns: jax.Array
    end_time: float
    use_unknown_spine: bool
    initial_population: int = 1
    n_non_spine: int | jax.Array | None = None
    upper_ages: jax.Array | None = None
    weights: jax.Array | None = None

    @property
    def has_ranges(self) -> bool:
        """Whether the ages are interval-censored (see :meth:`from_ranges`)."""
        return self.upper_ages is not None

    @property
    def n_observations(self) -> int:
        """Number of observed ages (spine included)."""
        if self.weights is not None:
            return round(float(np.asarray(self.weights).sum()))
        n_spine = 0 if self.spine_ages is None else len(self.spine_ages)
        n_rest = (
            len(self.non_spine_ages)
            if self.n_non_spine is None
            else int(self.n_non_spine)
        )
        return n_rest + n_spine

    def padded(self, size: int) -> "AgeData":
        """Return a copy whose ``non_spine_ages`` is padded to ``size`` entries.

        The padding is masked out of the likelihood, so values and gradients are
        unchanged, but jitted functions then see the same array shape for every
        sample size in the same bucket and compile only once.
        """
        n = (
            len(self.non_spine_ages)
            if self.n_non_spine is None
            else int(self.n_non_spine)
        )
        if size < n:
            msg = f"Cannot pad {n} ages to a smaller size {size}."
            raise ValueError(msg)
        # Pad on the host: jnp ops on a new length would each trigger a compile.
        current = np.asarray(self.non_spine_ages, dtype=np.float64)
        filler = np.full(size - len(current), 0.5 * self.end_time, dtype=np.float64)
        if self.upper_ages is not None and self.weights is not None:
            # Padding ranges are valid (lo < hi) and carry zero weight.
            upper = np.asarray(self.upper_ages, dtype=np.float64)
            weights = np.asarray(self.weights, dtype=np.float64)
            return dataclasses.replace(
                self,
                non_spine_ages=jnp.asarray(np.concatenate([current, filler])),
                upper_ages=jnp.asarray(np.concatenate([upper, filler + 1.0])),
                weights=jnp.asarray(np.concatenate([weights, np.zeros_like(filler)])),
                n_non_spine=n,
            )
        return dataclasses.replace(
            self,
            non_spine_ages=jnp.asarray(np.concatenate([current, filler])),
            n_non_spine=n,
        )

    @classmethod
    def from_groups(
        cls,
        list_of_ages: list[jax.Array] | list[NDArray[np.float64]],
        end_time: float,
        *,
        spine_ages: ArrayLike | None = None,
        is_ordered: bool = False,
        population_sizes: ArrayLike | None = None,
        initial_population: int = 1,
    ) -> "AgeData":
        """Build from age arrays per group.

        Parameters
        ----------
        list_of_ages : list of array-like
            Age arrays per group; at least one must be non-empty.
        end_time : float
            Observation time ``T`` (positive).
        spine_ages : array-like, optional
            Explicit spine ages.
        is_ordered : bool
            If True, the first element of every array is its spine.
        population_sizes : array-like, optional
            Counts per group; defaults to the group lengths (plus one per group
            when ``spine_ages`` is given).
        initial_population : int
            Individuals at time 0.

        Raises
        ------
        ValueError
            If all groups are empty, ``population_sizes`` has the wrong length,
            or ages fall outside ``[0, end_time]``.

        """
        if end_time <= 0:
            msg = "Observation time must be positive."
            raise ValueError(msg)
        # Host-side NumPy throughout: jnp ops on data-dependent shapes would each
        # trigger an XLA compile.
        groups = [np.asarray(a, dtype=np.float64).reshape(-1) for a in list_of_ages]
        if not any(len(g) > 0 for g in groups):
            msg = "No valid data points for fitting"
            raise ValueError(msg)
        if population_sizes is not None:
            ns = np.asarray(population_sizes, dtype=np.int32).reshape(-1)
            if ns.size != len(groups):
                msg = "Length of population_sizes must match length of list_of_ages"
                raise ValueError(msg)
        else:
            ns = np.array(
                [len(g) + int(spine_ages is not None) for g in groups],
                dtype=np.int32,
            )

        spine: NDArray[np.float64] | None
        if is_ordered and spine_ages is None:
            heads = [float(g[0]) for g in groups if len(g) > 0]
            tails = [g[1:] for g in groups if len(g) > 1]
            spine = np.array(heads, dtype=np.float64)
            rest = np.concatenate(tails) if tails else np.empty(0, dtype=np.float64)
        else:
            spine = (
                None
                if spine_ages is None
                else np.asarray(spine_ages, dtype=np.float64).reshape(-1)
            )
            rest = np.concatenate([g for g in groups if len(g) > 0])

        everything = rest if spine is None else np.concatenate([rest, spine])
        if np.any(everything < 0) or np.any(everything > end_time):
            msg = f"Ages must be in the range [0, {end_time}]."
            raise ValueError(msg)

        use_unknown = False
        if spine is None and len(rest) > 0:
            resolved = resolve_spine_ages(
                rest, end_time, multiple_terminal="marginalize"
            )
            spine = resolved.spine_ages
            rest = resolved.non_spine_ages
            use_unknown = resolved.use_unknown_spine
        return cls(
            non_spine_ages=jnp.asarray(rest, dtype=jnp.float64),
            spine_ages=None if spine is None else jnp.asarray(spine, dtype=jnp.float64),
            ns=jnp.asarray(ns),
            end_time=float(end_time),
            use_unknown_spine=use_unknown,
            initial_population=initial_population,
        )

    @classmethod
    def from_ranges(
        cls,
        lower_ages: ArrayLike,
        upper_ages: ArrayLike,
        end_time: float,
        *,
        population_size: int | None = None,
        initial_population: int = 1,
    ) -> "AgeData":
        r"""Build from ages that are only known to lie in ranges.

        Each observation is a range $[lo, hi]$ of ages, and its likelihood is the
        age density averaged uniformly over the range.
        An exact year $y$ is the range $[y, y+1)$ (the lower age is
        ``current_year - upper_year``).
        Ranges are not clipped, because the data cannot know the support of the
        birth rate; clip them to it first.
        The spine is always marginalised over, and duplicate ranges are merged
        into counts.

        Parameters
        ----------
        lower_ages, upper_ages : array-like
            Youngest and oldest age of each observation, with
            ``0 <= lower < upper <= end_time``.
        end_time : float
            Observation time ``T`` (positive).
        population_size : int, optional
            Population count; defaults to the number of observations.
        initial_population : int
            Individuals at time 0.

        Raises
        ------
        ValueError
            If there are no ranges, the arrays differ in length, or a range
            falls outside ``[0, end_time]`` or has no width.

        """
        if end_time <= 0:
            msg = "Observation time must be positive."
            raise ValueError(msg)
        lo = np.asarray(lower_ages, dtype=np.float64).reshape(-1)
        hi = np.asarray(upper_ages, dtype=np.float64).reshape(-1)
        if lo.size == 0:
            msg = "No valid data points for fitting"
            raise ValueError(msg)
        if lo.shape != hi.shape:
            msg = "lower_ages and upper_ages must have the same length"
            raise ValueError(msg)
        if np.any(lo < 0) or np.any(hi > end_time) or np.any(hi <= lo):
            msg = f"Ranges must satisfy 0 <= lower < upper <= {end_time}."
            raise ValueError(msg)
        pairs, counts = np.unique(
            np.stack([lo, hi], axis=1), axis=0, return_counts=True
        )
        n = int(lo.size) if population_size is None else int(population_size)
        return cls(
            non_spine_ages=jnp.asarray(pairs[:, 0]),
            spine_ages=None,
            ns=jnp.asarray([n], dtype=jnp.int32),
            end_time=float(end_time),
            use_unknown_spine=True,
            initial_population=initial_population,
            upper_ages=jnp.asarray(pairs[:, 1]),
            weights=jnp.asarray(counts, dtype=jnp.float64),
        )

    @classmethod
    def from_year_ranges(
        cls,
        early_year: ArrayLike,
        late_year: ArrayLike,
        *,
        start_year: float,
        observation_year: float,
        initial_population: int = 1,
    ) -> "AgeData":
        r"""Build range-censored ages from calendar-year date ranges.

        An individual dated ``[early, late]`` was created in some year of
        ``[early, late + 1)``, so its age, measured back from
        ``observation_year``, lies in
        ``[observation_year - (late + 1), observation_year - early]``.
        Each range is clipped to the birth support
        ``[start_year, observation_year]`` and kept at least one year wide, so a
        range lying wholly outside the support still contributes its nearest
        year.  Then it is passed to :meth:`from_ranges` with end time
        ``observation_year - start_year``.
        The per-row bounds come from :func:`range_bounds`.

        Parameters
        ----------
        early_year, late_year : array-like
            Earliest and latest creation year of each individual
            (``late_year`` below ``early_year`` is raised to it).
        start_year : float
            First year of the birth support (time zero of the model).
        observation_year : float
            Year of observation, the end of the last epoch.
        initial_population : int
            Individuals at time 0.

        Raises
        ------
        ValueError
            If the arrays differ in length or are empty, or
            ``observation_year <= start_year``.

        """
        if np.size(early_year) != np.size(late_year):
            msg = "early_year and late_year must have the same length"
            raise ValueError(msg)
        if observation_year <= start_year:
            msg = "observation_year must be later than start_year"
            raise ValueError(msg)
        lower, upper = range_bounds(early_year, late_year, start_year, observation_year)
        return cls.from_ranges(
            observation_year - upper,
            observation_year - lower,
            float(observation_year - start_year),
            initial_population=initial_population,
        )

    @classmethod
    def from_data_frame(
        cls,
        df: "Any",
        end_time: float,
        *,
        age_column: str | None = None,
        **kwargs: Any,
    ) -> "AgeData":
        """Build from one age column of a DataFrame (auto-detected if unnamed).

        Extra keyword arguments go to :meth:`from_groups`.
        """
        if age_column is None or age_column not in df.columns:
            exact = [col for col in df.columns if col in ("age", "ages")]
            prefixed = [col for col in df.columns if col.startswith("age_")]
            candidates = exact or prefixed
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
        ages = jnp.asarray(df[age_column].to_numpy(dtype=np.float64, copy=True))
        return cls.from_groups([ages], end_time, **kwargs)


_NEGATIVE_TOLERANCE = 1e-12


def _guarded(rate: RateFunction) -> tuple[RateFunction, jax.Array]:
    """Return ``(rate_to_evaluate, is_non_negative)``.

    Duck-typed: only rates that expose ``min_values()`` (the spline families)
    can dip below zero between knots; every other family is non-negative by
    construction and passes through.
    An infeasible rate is replaced by a harmless flat one (``flat_unless``),
    so that no NaN gradient leaks through the final ``jnp.where``.
    """
    min_values = getattr(rate, "min_values", None)
    if min_values is None:
        return rate, jnp.asarray(True)
    ok = jnp.min(jax.lax.stop_gradient(min_values())) >= -_NEGATIVE_TOLERANCE
    return rate.flat_unless(ok), ok  # type: ignore[attr-defined]


def log_likelihood_rates(
    data: AgeData,
    birth: RateFunction,
    death: RateFunction,
    *,
    condition_on_survival: bool = True,
    n_nodes: int = DEFAULT_NODES,
) -> jax.Array:
    """Log-likelihood of ``data`` under arbitrary birth and death rate functions.

    Formulas: see the sections "Population term" and
    "Unknown Spine (Default Inference Path)" of ``docs/likelihoods.qmd``.

    Piecewise-constant rates on shared epochs use the exact epoch-wise backend,
    anything else uses Gauss-Legendre quadrature
    (see :func:`bdpinf.models.math_general.terms_for`).
    Safe under ``jit`` and ``grad``; returns ``-inf`` for invalid parameters.
    A rate that exposes ``min_values()`` (the spline rates) and is negative
    anywhere is invalid: this is the single non-negativity guard for the fitters,
    the Bayesian model and direct calls.

    Raises
    ------
    ValueError
        If ``condition_on_survival`` is True and ``data.initial_population > 1``.
    """
    check_conditioning(data.initial_population, condition_on_survival)
    safe_birth, birth_ok = _guarded(birth)
    safe_death, death_ok = _guarded(death)
    value = _log_likelihood_rates(
        data,
        safe_birth,
        safe_death,
        condition_on_survival=condition_on_survival,
        n_nodes=n_nodes,
    )
    return jnp.where(birth_ok & death_ok, value, -jnp.inf)


def _log_likelihood_rates(
    data: AgeData,
    birth: RateFunction,
    death: RateFunction,
    *,
    condition_on_survival: bool,
    n_nodes: int,
) -> jax.Array:
    terms = terms_for(birth, death, n_nodes=n_nodes)
    if data.upper_ages is not None and data.weights is not None:
        return _range_log_likelihood(
            terms,
            data.non_spine_ages,
            data.upper_ages,
            data.weights,
            data.ns,
            initial_population=data.initial_population,
            condition_on_survival=condition_on_survival,
        )
    return _log_likelihood_core(
        terms,
        data.non_spine_ages,
        data.spine_ages,
        data.ns,
        initial_population=data.initial_population,
        condition_on_survival=condition_on_survival,
        use_unknown_spine_marginal=data.use_unknown_spine,
        n_valid=data.n_non_spine,
    )


def log_likelihood(
    birth_rates: object,
    death_rates: object,
    ages: object,
    times: object,
    *,
    spine_ages: object | None = None,
    ns: object | None = None,
    initial_population: int = 1,
    condition_on_survival: bool = True,
    is_ordered: bool = False,
    log_scale_params: bool = False,
) -> jnp.ndarray:
    """Compute the log-likelihood of observed individual ages in a birth--death process.

    Formulas: see the sections "Population term" and
    "Unknown Spine (Default Inference Path)" of ``docs/likelihoods.qmd``.

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
    initial_population : int
        Initial population size.
    condition_on_survival : bool, optional
        If True (default), condition the population-count term on the process
        not having gone extinct by ``T`` (the observed individuals imply survival).
        Only affects the count term.
        Only defined for ``initial_population == 1``:
        with more than one initial individual it raises,
        so pass ``condition_on_survival=False``.
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
        If both is_ordered=True and spine_ages is provided, if the sum of ns does
        not equal the number of provided ages,
        or if ``condition_on_survival`` is True and ``initial_population > 1``.

    See Also
    --------
        calculate_populations_log_lik

    """
    check_conditioning(initial_population, condition_on_survival)
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    birth_rates, death_rates, ages, times = ensure_1d_tensors(
        birth_rates,
        death_rates,
        ages,
        times,
    )
    _check_rate_time_shapes(birth_rates, death_rates, times)

    if log_scale_params:
        birth_rates = jnp.exp(birth_rates)
        death_rates = jnp.exp(death_rates)
    elif bool(jnp.any(birth_rates <= 0)) or bool(jnp.any(death_rates <= 0)):
        return _ninf

    spine_ages, ns = _resolve_explicit_spine(
        spine_ages, ns, ages, times, is_ordered=is_ordered
    )
    _check_times_and_ages(ages, times)
    ns_arr = _resolve_population_counts(ns, ages, spine_ages)

    if not (
        bool(jnp.all(jnp.isfinite(birth_rates)))
        and bool(jnp.all(jnp.isfinite(death_rates)))
    ):
        return _ninf

    ages, spine_ages, use_unknown_spine_marginal = _resolve_unknown_spine(
        ages, spine_ages, times, is_ordered=is_ordered
    )

    birth_rates, death_rates, times = normalize_and_validate_inputs(
        birth_rates, death_rates, times
    )
    return _log_likelihood_core(
        nested_terms(birth_rates, death_rates, times),
        ages,
        spine_ages,
        ns_arr,
        initial_population=initial_population,
        condition_on_survival=condition_on_survival,
        use_unknown_spine_marginal=use_unknown_spine_marginal,
    )


def _check_rate_time_shapes(
    birth_rates: jax.Array, death_rates: jax.Array, times: jax.Array
) -> None:
    """Raise ``ValueError`` unless the rates and times share one shape."""
    if not (birth_rates.shape == death_rates.shape == times.shape):
        msg = (
            "birth_rates, death_rates, and times must have the same shape, "
            f"got {birth_rates.shape}, {death_rates.shape}, {times.shape}."
        )
        raise ValueError(msg)


def _resolve_explicit_spine(
    spine_ages: object | None,
    ns: object | None,
    ages: jax.Array,
    times: jax.Array,
    *,
    is_ordered: bool,
) -> tuple[jax.Array | None, object | None]:
    """Validate explicit ``spine_ages`` (empty means none); default ``ns`` for one."""
    if spine_ages is None:
        return None, ns
    spine_arr = ensure_1d_tensor(spine_ages)
    if len(spine_arr) == 0:
        return None, ns
    if is_ordered:
        msg = "Cannot have both is_ordered=True and spine_ages provided."
        raise ValueError(msg)
    if ns is None:
        if len(spine_arr) > 1:
            msg = (
                "When multiple spine ages are provided, "
                "population sizes (ns) must also be provided."
            )
            raise ValueError(msg)
        ns = jnp.array([len(ages) + len(spine_arr)], dtype=jnp.float64)
    if bool(jnp.any(spine_arr < 0)) or bool(jnp.any(spine_arr > times[-1])):
        msg = f"Spine ages must be in the range [0, {times[-1]}]."
        raise ValueError(msg)
    return spine_arr, ns


def _check_times_and_ages(ages: jax.Array, times: jax.Array) -> None:
    """Raise ``ValueError`` for bad observation times or ages outside ``[0, T]``."""
    if float(times[0]) <= 0:
        msg = "Observation times must be positive."
        raise ValueError(msg)

    if bool(jnp.any(jnp.diff(times) < 0)):
        msg = "Observation times must be non-decreasing."
        raise ValueError(msg)

    if bool(jnp.any(ages < 0)) or bool(jnp.any(ages > times[-1])):
        msg = f"Ages must be in the range [0, {float(times[-1])}]."
        raise ValueError(msg)


def _resolve_population_counts(
    ns: object | None, ages: jax.Array, spine_ages: jax.Array | None
) -> jax.Array:
    """Return the population counts, defaulting to one per age; warn if partial."""
    num_ages = len(ages) + (len(spine_ages) if spine_ages is not None else 0)
    if ns is None:
        return jnp.array([num_ages], dtype=jnp.float64)
    ns_arr = ensure_1d_tensor(ns)
    total_pop_size = int(float(ns_arr.sum()))
    if total_pop_size < num_ages:
        msg = "Sum of the populations must equal the number of ages provided."
        raise ValueError(msg)
    if total_pop_size > num_ages:
        # Called straight from ``log_likelihood``, so level 3 is its caller.
        warnings.warn(
            "Sum of the populations exceeds the number of ages provided; "
            "not all ages have been provided (age information is partial).",
            UserWarning,
            stacklevel=3,
        )
    return ns_arr


def _resolve_unknown_spine(
    ages: jax.Array,
    spine_ages: jax.Array | None,
    times: jax.Array,
    *,
    is_ordered: bool,
) -> tuple[jax.Array, jax.Array | None, bool]:
    """Pick the spine (ordered, at ``T``, or a lone age) and whether to marginalise."""
    if spine_ages is None and not is_ordered and len(ages) > 0:
        resolved = resolve_spine_ages(ages, float(times[-1]))
        if resolved.spine_ages is not None:
            spine_ages = jnp.asarray(resolved.spine_ages)
            ages = jnp.asarray(resolved.non_spine_ages)

    use_unknown_spine_marginal = (not is_ordered) and (spine_ages is None)

    if is_ordered:
        spine_ages = ages[:1]
        ages = ages[1:]
    return ages, spine_ages, use_unknown_spine_marginal


def _range_log_likelihood(
    terms: AgeTerms,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    weights: jnp.ndarray,
    ns: jnp.ndarray,
    *,
    initial_population: int = 1,
    condition_on_survival: bool = True,
) -> jnp.ndarray:
    """Log-likelihood of ages known only to lie in ``[lo, hi]`` (unknown spine).

    Formula: see the section "Unknown Spine (Default Inference Path)"
    of ``docs/likelihoods.qmd``.

    ``weights`` count the observations sharing each range (0 for padding).
    The unknown-spine mixture is a sum of products, so integrating it over a box
    of ranges replaces every density by the matching range mass: the likelihood
    is exact.
    """
    check_conditioning(initial_population, condition_on_survival)
    log_p = calculate_populations_log_lik_from_terms(
        terms,
        ns,
        initial_population=initial_population,
        condition_on_survival=condition_on_survival,
    )
    log_f, log_f_spine = range_log_masses(terms, lo, hi)
    log_width = jnp.log(hi - lo)
    log_f = log_f - log_width
    log_f_spine = log_f_spine - log_width
    live = weights > 0
    w_safe = jnp.where(live, weights, 1.0)
    ratio = jnp.where(live, jnp.log(w_safe) + log_f_spine - log_f, -jnp.inf)
    ratio_max = jax.lax.stop_gradient(jnp.max(ratio))
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(ratio - ratio_max))) + ratio_max
    ages_ll = jnp.sum(jnp.where(live, weights * log_f, 0.0)) - jnp.log(jnp.sum(weights))
    total = log_p + ages_ll + log_sum_exp
    return jnp.where(jnp.isfinite(total), total, -jnp.inf)


def _log_likelihood_core(
    terms: AgeTerms,
    non_spine_ages: jnp.ndarray,
    spine_ages: jnp.ndarray | None,
    ns: jnp.ndarray,
    *,
    initial_population: int = 1,
    condition_on_survival: bool = True,
    use_unknown_spine_marginal: bool = True,
    n_valid: int | jax.Array | None = None,
) -> jnp.ndarray:
    """Shared log-likelihood core for ``log_likelihood`` and the fitters.

    Formulas: see the sections "Population term" and
    "Unknown Spine (Default Inference Path)" of ``docs/likelihoods.qmd``.

    All host-side validation and spine resolution must be completed by the
    caller (see :class:`AgeData`); this function contains no host-side hazards
    (``bool(traced)``, ``np.isclose(traced)``, dynamic mask slicing), so it is
    safe under ``jax.jit`` and ``jax.grad``.
    The flags and array lengths are static, so the Python-level branching on
    them resolves at trace time.
    Non-finite outputs are mapped to ``-inf`` with ``jnp.where``.

    Parameters
    ----------
    terms : AgeTerms
        Precomputed terms for the rates (exact epoch-wise or quadrature, see
        :func:`bdpinf.models.math_general.terms_for`).
    non_spine_ages : jnp.ndarray
        Non-spine (or all, when marginalizing) ages, shape ``(m,)`` float64.
    spine_ages : jnp.ndarray | None
        Explicit spine ages if known, or ``None`` when ``use_unknown_spine_marginal``
        is True.
    ns : jnp.ndarray
        Population counts, shape ``(k,)``.
    initial_population : int
        Initial population size (static, Python int).
    condition_on_survival : bool
        Condition on non-extinction for positive counts (static, Python bool).
        Raises ``ValueError`` with ``initial_population > 1``.
    use_unknown_spine_marginal : bool
        If True, ``non_spine_ages`` contains all ages and the spine position
        is marginalized via logsumexp.  Static Python bool resolved at trace
        time.
    n_valid : int | jax.Array | None
        If given, only the first ``n_valid`` entries of ``non_spine_ages`` are
        genuine (the rest is padding and is masked out).  May be traced.

    Returns
    -------
    jnp.ndarray
        Scalar float64 log-likelihood, or ``-inf`` for invalid parameters.

    """
    check_conditioning(initial_population, condition_on_survival)
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    log_P_n_T = calculate_populations_log_lik_from_terms(
        terms,
        ns,
        initial_population=initial_population,
        condition_on_survival=condition_on_survival,
    )

    ages_log_lik = jnp.float64(0.0)
    n_non_spine = len(non_spine_ages)  # static (padded) shape
    valid = None if n_valid is None else jnp.arange(n_non_spine) < n_valid

    def masked_sum(x: jax.Array) -> jax.Array:
        return x.sum() if valid is None else jnp.where(valid, x, 0.0).sum()

    if use_unknown_spine_marginal and n_non_spine > 0:
        # Unknown-spine marginal (paper: P(Z_T) * sum_j f*(a_j) prod_{i!=j} f(a_i),
        # up to the 1/n for the uniformly chosen spine): every age is a spine
        # candidate; both densities come from one shared pass.
        log_f, log_f_spine = age_log_densities(terms, non_spine_ages)
        a = log_f_spine - log_f
        if valid is not None:
            a = jnp.where(valid, a, -jnp.inf)
        a_max = jax.lax.stop_gradient(jnp.max(a))
        log_sum_exp = jnp.log(jnp.sum(jnp.exp(a - a_max))) + a_max
        count = n_non_spine if n_valid is None else n_valid
        ages_log_lik = (
            masked_sum(log_f)
            - jnp.log(jnp.asarray(count, dtype=jnp.float64))
            + log_sum_exp
        )
    else:
        # Known or pre-resolved spine.
        if spine_ages is not None:
            ages_log_lik = ages_log_lik + spine_log_density(terms, spine_ages).sum()
        if n_non_spine > 0:
            ages_log_lik = ages_log_lik + masked_sum(
                non_spine_log_density(terms, non_spine_ages)
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


def calculate_unknown_spine_mixture_age_density_rates(
    birth: RateFunction,
    death: RateFunction,
    age_grid: object,
    num_individuals: int,
    *,
    n_nodes: int = DEFAULT_NODES,
) -> jnp.ndarray:
    """Age density of a random individual when the spine is unknown.

    Same as :func:`calculate_unknown_spine_mixture_age_density` for arbitrary
    :class:`~bdpinf.models.rate_functions.RateFunction` birth and death rates.
    """
    if num_individuals <= 0:
        msg = "num_individuals must be a positive integer."
        raise ValueError(msg)
    (age_grid_arr,) = ensure_1d_tensors(age_grid)
    log_n = jnp.log(jnp.float64(num_individuals))
    if num_individuals > 1:
        log_n_minus_one = jnp.log(jnp.float64(num_individuals - 1))
    else:
        log_n_minus_one = jnp.array(-jnp.inf, dtype=jnp.float64)
    log_non_spine, log_spine = age_log_densities(
        terms_for(birth, death, n_nodes=n_nodes), age_grid_arr
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
