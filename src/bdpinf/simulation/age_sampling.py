r"""Sample observed ages directly from the exact age laws for any rate functions.

For birth and death rates $\lambda(t)$, $\mu(t)$ given as
:class:`~bdpinf.models.rate_functions.RateFunction` objects, a surviving
population at time $T$ has size $Z_T$ with $\Pr(Z_T = 0) = P_0(T)$ and, given
survival, $Z_T \sim \mathrm{Geometric}(\pi(T))$.
One individual (the spine) has age law $f^*$ plus an atom at $T$, and the
other $Z_T - 1$ ages are i.i.d. with the non-spine density $f$
(:mod:`bdpinf.models.math_general`).
Sampling inverts the numerically integrated CDFs on a fine grid, so it needs no
particle simulation and works for piecewise-linear, sinusoidal or any other
rate family.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from bdpinf.models import math_general, math_nested
from bdpinf.models.rate_functions import RateFunction

__all__ = ["SampledAges", "sample_ages_in_ranges", "simulate_ages_from_rates"]


@dataclass(frozen=True)
class SampledAges:
    """Ages of one simulated surviving population.

    Attributes
    ----------
    spine_age : float | None
        Age of the spine individual; ``None`` if the population died out.
    non_spine_ages : NDArray
        Ages of the other individuals.
    end_time : float
        Observation time $T$.

    """

    spine_age: float | None
    non_spine_ages: NDArray[np.float64]
    end_time: float

    @property
    def population(self) -> int:
        """Number of surviving individuals $Z_T$."""
        return 0 if self.spine_age is None else 1 + len(self.non_spine_ages)

    @property
    def all_ages(self) -> NDArray[np.float64]:
        """All ages, spine first (the layout ``AgeData(is_ordered=True)`` expects)."""
        if self.spine_age is None:
            return np.empty(0, dtype=np.float64)
        return np.concatenate([[self.spine_age], self.non_spine_ages])


def _inverse_cdf(
    grid: NDArray[np.float64], density: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Trapezoid CDF on ``grid`` (starting at 0) and its total mass."""
    steps = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    cdf = np.concatenate([[0.0], np.cumsum(steps)])
    return grid, cdf, float(cdf[-1])


def simulate_ages_from_rates(
    birth: RateFunction,
    death: RateFunction,
    rng: np.random.Generator,
    *,
    ensure_non_extinct: bool = True,
    n_grid: int = 4001,
) -> SampledAges:
    r"""Draw one population's ages from the exact age laws.

    Parameters
    ----------
    birth, death : RateFunction
        Birth and death rates on the same window ``[0, T]``.
    rng : numpy.random.Generator
        Source of randomness.
    ensure_non_extinct : bool
        Condition on survival ($Z_T \ge 1$).
        If False, extinction occurs with probability $P_0(T)$ and is returned as
        an empty population.
    n_grid : int
        Grid points for inverting the age CDFs; the error is $O(1/n^2)$.

    Returns
    -------
    SampledAges

    """
    end = float(birth.end_time)
    terms = math_general.terms_for(birth, death)
    log_p0, _ = math_nested.log_extinction_and_survival_pr(terms)
    if not ensure_non_extinct and rng.uniform() < float(np.exp(log_p0)):
        return SampledAges(None, np.empty(0, dtype=np.float64), end)

    pi = float(np.exp(-math_nested.log_nested_w_total(terms)))
    size = int(rng.geometric(min(max(pi, 1e-300), 1.0)))

    grid = np.linspace(0.0, end, n_grid)
    jgrid = jnp.asarray(grid)
    non_spine_pdf = np.exp(np.asarray(math_general.non_spine_log_density(terms, jgrid)))
    spine_pdf = np.exp(
        np.asarray(math_general.spine_log_density(terms, jgrid, atom_at_boundary=False))
    )
    atom = float(np.exp(math_general.spine_log_atom(terms)))

    # Non-spine ages: invert the normalised CDF.
    _, cdf_ns, mass_ns = _inverse_cdf(grid, non_spine_pdf)
    ns_ages = np.interp(rng.uniform(size=size - 1), cdf_ns / mass_ns, grid)

    # Spine age: continuous part has mass 1 - atom, the atom sits at T.
    _, cdf_sp, mass_sp = _inverse_cdf(grid, spine_pdf)
    u = rng.uniform()
    continuous_mass = max(1.0 - atom, 0.0)
    spine_age = (
        end
        if u >= continuous_mass
        else float(np.interp(u / continuous_mass, cdf_sp / mass_sp, grid))
    )
    return SampledAges(spine_age, np.asarray(ns_ages, dtype=np.float64), end)


def sample_ages_in_ranges(
    birth: RateFunction,
    death: RateFunction,
    lower: ArrayLike,
    upper: ArrayLike,
    rng: np.random.Generator,
    *,
    n_draws: int,
    n_grid: int = 4001,
) -> NDArray[np.float64]:
    r"""Draw ages known only to lie in ranges, from the non-spine age density.

    Each age $a_i$ has the non-spine density $f$ of the given rates
    (:func:`~bdpinf.models.math_general.non_spine_log_density`) truncated to
    ``[lower[i], upper[i]]``.
    Sampling inverts the trapezoid CDF on a grid: the CDF is interpolated at the
    range endpoints, uniforms are drawn between those two values and mapped back
    through the inverse CDF.
    A range with zero or non-finite mass (for example, one outside ``[0, T]``)
    falls back to a uniform draw on the range clipped to ``[0, T]``.
    The draws are vectorised, so many draws of thousands of ranges need no
    Python loop.

    Parameters
    ----------
    birth, death : RateFunction
        Birth and death rates on the same window ``[0, T]``.
    lower, upper : array-like
        Range endpoints, shape ``[n]``, in age units, with ``lower <= upper``.
    rng : numpy.random.Generator
        Source of randomness.
    n_draws : int
        Number of independent draws of all ``n`` ages.
    n_grid : int
        Grid points for the CDF.

    Returns
    -------
    NDArray
        Ages of shape ``[n_draws, n]``, each inside its range.

    Raises
    ------
    ValueError
        If ``lower`` and ``upper`` differ in shape, are not 1-D, or are not ordered.

    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape or lo.ndim != 1:
        msg = "lower and upper must be 1-D arrays of the same length"
        raise ValueError(msg)
    if np.any(lo > hi):
        msg = "lower must not exceed upper"
        raise ValueError(msg)
    end = float(birth.end_time)
    lo, hi = np.clip(lo, 0.0, end), np.clip(hi, 0.0, end)

    terms = math_general.terms_for(birth, death)
    grid = np.linspace(0.0, end, n_grid)
    pdf = np.exp(
        np.asarray(math_general.non_spine_log_density(terms, jnp.asarray(grid)))
    )
    _, cdf, total = _inverse_cdf(grid, pdf)
    if not (np.isfinite(total) and total > 0):
        return lo + (hi - lo) * rng.uniform(size=(n_draws, lo.size))
    cdf = cdf / total
    f_lo, f_hi = np.interp(lo, grid, cdf), np.interp(hi, grid, cdf)
    mass = f_hi - f_lo
    usable = np.isfinite(mass) & (mass > 0)

    unit = rng.uniform(size=(n_draws, lo.size))
    inverted = np.interp(f_lo + unit * mass, cdf, grid)
    uniform = lo + (hi - lo) * unit
    return np.clip(np.where(usable, inverted, uniform), lo, hi)
