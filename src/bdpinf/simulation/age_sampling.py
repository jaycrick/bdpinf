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
from numpy.typing import NDArray

from bdpinf.models import math_general, math_nested
from bdpinf.models.rate_functions import RateFunction

__all__ = ["SampledAges", "simulate_ages_from_rates"]


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
