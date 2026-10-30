r"""Interval-censored (binned) ages against exact ages: speed and MLE accuracy.

Each observed age is known only to lie in a range ``[lo, hi]``, and its
contribution to the likelihood is the age density averaged uniformly over that
range, $(1/w)\int_{lo}^{hi} f(a)\,da$ with $w = hi - lo$.
For piecewise-constant rates the bin masses have closed forms
(no quadrature), using $\mathcal W' = \gamma\mathcal W + \mu$:

- non-spine: $G(s) = (\mathcal W(s) - 1)\,e^{-M(s,T]}$ has
  $G' = \lambda\mathcal W e^{-M}$,
  so $\Pr(lo \le A \le hi) = [G(T-lo) - G(T-hi)] / (\mathcal W(T) - 1)$;
- spine: $H(s) = e^{-\Lambda(s,T]}(L_\mu(s,T) + e^{\Gamma(s,T]})$ has
  $H' = \lambda e^{-\Lambda} L_\mu$, $H(T) = 1$ and $H(0) = $ the atom at $A^* = T$,
  so the bin containing $T$ takes the founder's atom.

The unknown-spine mixture is a sum of products, so integrating it over a box of
bins replaces every density by the matching bin mass: the binned likelihood is
exact.

Sections (run all, or pick with ``--only``):

``check``
    Closed-form masses against Gauss-Legendre quadrature, masses summing to one,
    and the fine-bin limit against the exact-age likelihood.
``speed``
    Compile and steady-state time of the value, gradient and Hessian for exact
    ages, binned ages (closed form, per observation and aggregated by bin) and
    binned ages by quadrature.
``fit``
    Simulated replicates binned at ``T/20``: time to MLE convergence, bias,
    RMSE, Wald coverage and the information lost to binning.
    Constant rates first, then two epochs with a fixed knot.

Usage
-----
    uv run python scripts/bench_binned_ages.py --quick
    uv run python scripts/bench_binned_ages.py --only check speed
    uv run python scripts/bench_binned_ages.py --out _output/bench_binned_ages.json
"""

import argparse
import dataclasses
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from bench_fitting import _compile_and_run, _table

from bdpinf.inference import AgeData, BirthDeathModel, Constraints
from bdpinf.inference import fitting as fitting_module
from bdpinf.inference.constraints import CompiledConstraints
from bdpinf.inference.likelihood import log_likelihood_rates
from bdpinf.inference.rate_models import PiecewiseConstant
from bdpinf.inference.uncertainty import covariance_from_hessian
from bdpinf.models.math_general import (
    DEFAULT_NODES,
    AgeTerms,
    non_spine_log_density,
    spine_log_atom,
    spine_log_density,
    terms_for,
)
from bdpinf.models.math_nested import (
    LOG_ZERO,
    NestedTerms,
    _epoch_index,
    _log_kappa_minus,
    _log_or_floor,
    _row,
    log_nested_w_minus_one,
)
from bdpinf.models.math_populations import calculate_populations_log_lik_from_terms
from bdpinf.simulation import simulate_ages_from_rates

jax.config.update("jax_enable_x64", True)

SECTIONS = ("check", "speed", "fit")
N_BINS = 20
_PENALTY = fitting_module._PENALTY
_GL_NODES = 16


# ---------------------------------------------------------------------------
# Closed-form bin masses (piecewise-constant rates)
# ---------------------------------------------------------------------------


def _log_l_before(terms: NestedTerms) -> jax.Array:
    r"""$\ln L(\tau_{k-1})$ per epoch (``LOG_ZERO`` for the first epoch)."""
    durations = terms.times - terms.starts
    l_terms = (
        _log_or_floor(terms.birth_rates)
        - terms.gamma_before
        + _log_kappa_minus(terms.growth, durations)
    )
    cumulative = jax.lax.cumlogsumexp(l_terms)
    return jnp.concatenate([jnp.full((1,), LOG_ZERO), cumulative[:-1]])


def _log_g(terms: NestedTerms, log_l_before: jax.Array, s: jax.Array) -> jax.Array:
    r"""$\ln G(s)$ with $G = (\mathcal W - 1)e^{-M(s,T]}$."""
    row = _row(terms, s)
    u = s - row.start
    rem = row.time - s
    k = _epoch_index(terms.times, s)
    partial = (
        _log_or_floor(row.birth) - row.gamma_before + _log_kappa_minus(row.growth, u)
    )
    log_l = jnp.logaddexp(jnp.take(log_l_before, k), partial)
    gamma_s = row.gamma_before + row.growth * u
    return gamma_s + log_l - (row.death * rem + row.mu_after)


def _log_h(terms: NestedTerms, s: jax.Array) -> jax.Array:
    r"""$\ln H(s)$, the CDF of the spine's birth time (atom at ``s = 0``)."""
    row = _row(terms, s)
    u = s - row.start
    rem = row.time - s
    partial = (
        row.log_death
        + (terms.gamma_total - row.gamma_end)
        + row.growth * rem
        + _log_kappa_minus(row.growth, rem)
    )
    log_l_mu = jnp.logaddexp(partial, row.log_spine_after)
    gamma_to_end = terms.gamma_total - (row.gamma_before + row.growth * u)
    return -(row.birth * rem + row.lam_after) + jnp.logaddexp(log_l_mu, gamma_to_end)


def _log_diff(upper: jax.Array, lower: jax.Array, *, lower_is_zero: jax.Array) -> Any:
    """``log(exp(upper) - exp(lower))``, with ``lower`` treated as 0 where flagged."""
    delta = jnp.where(lower_is_zero, -jnp.inf, lower - upper)
    return upper + jnp.log1p(-jnp.exp(delta))


def closed_form_log_masses(
    terms: NestedTerms, lo: jax.Array, hi: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Log masses of ``[lo, hi]`` under the non-spine and spine age laws."""
    end = terms.times[-1]
    s_big = jnp.clip(end - lo, 0.0, end)
    s_small = jnp.clip(end - hi, 0.0, end)
    log_l_before = _log_l_before(terms)
    never = jnp.zeros_like(lo, dtype=bool)
    non_spine = _log_diff(
        _log_g(terms, log_l_before, s_big),
        _log_g(terms, log_l_before, s_small),
        lower_is_zero=never,
    ) - log_nested_w_minus_one(terms)
    spine = _log_diff(
        _log_h(terms, s_big), _log_h(terms, s_small), lower_is_zero=hi >= end
    )
    return non_spine, spine


def _end_time(terms: AgeTerms) -> jax.Array:
    """Observation time of either kind of terms."""
    return terms.times[-1] if isinstance(terms, NestedTerms) else terms.edges[-1]


def _interior_breakpoints(terms: AgeTerms) -> jax.Array:
    """Interior epoch boundaries (birth times where the rates are not smooth)."""
    if isinstance(terms, NestedTerms):
        return terms.times[:-1]
    return terms.edges[1:-1]


def quadrature_log_masses(
    terms: AgeTerms, lo: jax.Array, hi: jax.Array, n_nodes: int = _GL_NODES
) -> tuple[jax.Array, jax.Array]:
    """Return the same masses by Gauss-Legendre quadrature of each bin's densities.

    Each bin is cut at the ages ``T - tau`` where the rates change, so no panel
    contains a kink and the masses stay smooth when a knot moves inside a bin.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    nodes, weights = jnp.asarray(nodes), jnp.asarray(weights)
    end = _end_time(terms)
    cuts = jnp.clip(
        end - _interior_breakpoints(terms)[None, :], lo[:, None], hi[:, None]
    )
    points = jnp.sort(jnp.concatenate([lo[:, None], cuts, hi[:, None]], axis=1), axis=1)
    left, right = points[:, :-1], points[:, 1:]
    half = 0.5 * (right - left)
    x = left[..., None] + half[..., None] * (nodes + 1.0)
    flat = x.reshape(-1)

    def integrate(log_density: jax.Array) -> jax.Array:
        dens = jnp.exp(log_density.reshape(x.shape))
        return jnp.sum(jnp.sum(weights * dens, axis=-1) * half, axis=-1)

    non_spine = integrate(non_spine_log_density(terms, flat))
    continuous = integrate(spine_log_density(terms, flat, atom_at_boundary=False))
    atom = jnp.where(hi >= end, jnp.exp(spine_log_atom(terms)), 0.0)
    return jnp.log(non_spine), jnp.log(continuous + atom)


def binned_log_likelihood(
    terms: AgeTerms,
    lo: jax.Array,
    hi: jax.Array,
    weights: jax.Array,
    ns: jax.Array,
    *,
    quadrature: bool = False,
    condition_on_survival: bool = True,
) -> jax.Array:
    """Log-likelihood of ages known only to lie in ``[lo, hi]`` (unknown spine).

    ``weights`` count how many observations share each interval, so one row per
    distinct bin (``weights`` = counts) and one row per observation
    (``weights`` = 1, or 0 for padding) give the same value.
    The interval density is the bin mass divided by the width.
    """
    log_p = calculate_populations_log_lik_from_terms(
        terms, ns, initial_population=1, condition_on_survival=condition_on_survival
    )
    # Only piecewise-constant rates have the closed form; others integrate.
    use_closed = isinstance(terms, NestedTerms) and not quadrature
    masses = closed_form_log_masses if use_closed else quadrature_log_masses
    log_f, log_f_spine = masses(terms, lo, hi)
    log_width = jnp.log(hi - lo)
    log_f = log_f - log_width
    log_f_spine = log_f_spine - log_width
    live = weights > 0
    w_safe = jnp.where(live, weights, 1.0)
    ratio = jnp.where(live, jnp.log(w_safe) + log_f_spine - log_f, -jnp.inf)
    ratio_max = jax.lax.stop_gradient(jnp.max(ratio))
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(ratio - ratio_max))) + ratio_max
    total_weight = jnp.sum(weights)
    ages_ll = jnp.sum(jnp.where(live, weights * log_f, 0.0)) - jnp.log(total_weight)
    total = log_p + ages_ll + log_sum_exp
    return jnp.where(jnp.isfinite(total), total, -jnp.inf)


# ---------------------------------------------------------------------------
# Scenarios and data
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A model with its true parameters and the growth-rate contrasts."""

    name: str
    model: BirthDeathModel
    true: np.ndarray
    labels: tuple[str, ...]
    growth: np.ndarray  # rows map the natural vector to growth rates
    growth_labels: tuple[str, ...]
    horizon: float


def make_scenarios(
    horizon: float = 10.0, keys: tuple[str, ...] | None = None
) -> list[Scenario]:
    """Constant rates, two constant epochs, and two continuous linear epochs.

    The knot is fixed at ``T/2`` throughout, which is a bin edge.
    """
    scenarios = [
        Scenario(
            "constant",
            BirthDeathModel.piecewise_constant([horizon]),
            np.array([1.0, 0.5]),
            ("lambda", "mu"),
            np.array([[1.0, -1.0]]),
            ("gamma",),
            horizon,
        ),
        Scenario(
            "two epochs (knot at T/2)",
            BirthDeathModel.piecewise_constant([0.5 * horizon, horizon]),
            np.array([1.2, 0.7, 0.4, 0.4]),
            ("lambda1", "lambda2", "mu1", "mu2"),
            np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]),
            ("gamma1", "gamma2"),
            horizon,
        ),
        Scenario(
            "linear (2 epochs, continuous)",
            BirthDeathModel.piecewise_linear([0.5 * horizon, horizon]),
            np.array([1.0, 1.4, 0.9, 0.3, 0.5, 0.4]),
            ("b0", "b1", "b2", "d0", "d1", "d2"),
            np.hstack([np.eye(3), -np.eye(3)]),
            ("gamma(0)", "gamma(T/2)", "gamma(T)"),
            horizon,
        ),
    ]
    if keys is None:
        return scenarios
    return [sc for sc in scenarios if sc.name.split()[0] in keys]


def edges_for(horizon: float) -> np.ndarray:
    """Bin edges of width ``T/20`` with the last edge exactly ``T``."""
    return np.linspace(0.0, horizon, N_BINS + 1)


def simulate_population(
    scenario: Scenario, rng: np.random.Generator, min_n: int = 30
) -> np.ndarray:
    """Ages of one surviving population (founder included) from the exact laws."""
    birth, death = scenario.model.rates(jnp.asarray(scenario.true))
    while True:
        sample = simulate_ages_from_rates(birth, death, rng)
        if sample.population >= min_n:
            return np.asarray(sample.all_ages, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class BinnedData:
    """Binned ages in both layouts: per observation and aggregated by bin."""

    lo: np.ndarray
    hi: np.ndarray
    counts: np.ndarray  # per bin, shape (N_BINS,)
    edges: np.ndarray
    n: int

    @classmethod
    def from_ages(cls, ages: np.ndarray, horizon: float) -> "BinnedData":
        """Assign each age to its ``T/20`` bin (the last bin includes ``T``)."""
        edges = edges_for(horizon)
        idx = np.minimum(np.searchsorted(edges, ages, side="right") - 1, N_BINS - 1)
        counts = np.bincount(idx, minlength=N_BINS).astype(np.float64)
        return cls(edges[idx], edges[idx + 1], counts, edges, len(ages))

    def per_observation(self, size: int | None = None) -> tuple[np.ndarray, ...]:
        """``(lo, hi, weights)`` per observation, padded with zero weights."""
        size = self.n if size is None else size
        pad = size - self.n
        lo = np.concatenate([self.lo, np.full(pad, self.edges[0])])
        hi = np.concatenate([self.hi, np.full(pad, self.edges[1])])
        w = np.concatenate([np.ones(self.n), np.zeros(pad)])
        return lo, hi, w

    def aggregated(self) -> tuple[np.ndarray, ...]:
        """``(lo, hi, counts)`` with one row per bin."""
        return self.edges[:-1], self.edges[1:], self.counts

    def midpoint_data(self, horizon: float) -> AgeData:
        """Midpoints as ``AgeData``, only used for the moment starting point."""
        mid = 0.5 * (self.lo + self.hi)
        return AgeData(
            non_spine_ages=jnp.asarray(mid),
            spine_ages=None,
            ns=jnp.asarray([self.n], dtype=jnp.int32),
            end_time=horizon,
            use_unknown_spine=True,
        )


# ---------------------------------------------------------------------------
# Objectives and the shared MLE driver
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Problem:
    """Callables of the free (unconstrained) parameters plus the optimum scale."""

    nll: Callable[[jax.Array], jax.Array]
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]]
    hessian: Callable[[jax.Array], jax.Array]
    n_obs: int


def exact_problem(
    data: AgeData, model: BirthDeathModel, compiled: CompiledConstraints
) -> Problem:
    """Build the exact-age objective exactly as ``fit`` does."""
    objective = fitting_module._objective(
        model,
        compiled,
        has_spine=data.spine_ages is not None,
        use_unknown_spine=data.use_unknown_spine,
        initial_population=data.initial_population,
        end_time=data.end_time,
        condition_on_survival=True,
        n_nodes=DEFAULT_NODES,
    )
    n_rest = len(data.non_spine_ages)
    padded = data.padded(fitting_module._bucket(n_rest))
    runtime = (
        jnp.asarray(padded.non_spine_ages),
        jnp.zeros(0) if data.spine_ages is None else jnp.asarray(data.spine_ages),
        jnp.asarray(data.ns),
        jnp.asarray(compiled.fixed_natural),
        jnp.asarray(n_rest),
        jnp.zeros(0),
        jnp.zeros(0),
    )
    return Problem(
        lambda u: objective.nll(u, *runtime),
        lambda u: objective.value_and_grad(u, *runtime),
        lambda u: objective.hessian(u, *runtime),
        max(1, data.n_observations),
    )


_BINNED_CACHE: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}


def binned_functions(
    model: BirthDeathModel, compiled: CompiledConstraints, *, quadrature: bool
) -> tuple[Any, Any, Any]:
    """Jitted nll, value-and-grad and Hessian with the bins as runtime arguments."""
    key = (model, compiled.structure_key, quadrature)
    if key in _BINNED_CACHE:
        return _BINNED_CACHE[key]

    def nll(
        u: jax.Array,
        fixed: jax.Array,
        lo: jax.Array,
        hi: jax.Array,
        w: jax.Array,
        ns: jax.Array,
    ) -> jax.Array:
        birth, death = model.rates(compiled.expand(u, fixed))
        terms = terms_for(birth, death, n_nodes=DEFAULT_NODES)
        ll = binned_log_likelihood(terms, lo, hi, w, ns, quadrature=quadrature)
        return jnp.where(jnp.isfinite(ll), -ll, _PENALTY)

    funcs = (
        jax.jit(nll),
        jax.jit(jax.value_and_grad(nll)),
        jax.jit(jax.jacfwd(jax.jacfwd(nll))),
    )
    _BINNED_CACHE[key] = funcs
    return funcs


def binned_problem(
    arrays: tuple[np.ndarray, ...],
    n: int,
    model: BirthDeathModel,
    compiled: CompiledConstraints,
    *,
    quadrature: bool = False,
) -> Problem:
    """Binned objective for ``(lo, hi, weights)`` arrays."""
    nll, vg, hess = binned_functions(model, compiled, quadrature=quadrature)
    runtime = (
        jnp.asarray(compiled.fixed_natural),
        *(jnp.asarray(a) for a in arrays),
        jnp.asarray([n], dtype=jnp.int32),
    )
    return Problem(
        lambda u: nll(u, *runtime),
        lambda u: vg(u, *runtime),
        lambda u: hess(u, *runtime),
        n,
    )


@dataclasses.dataclass
class Solved:
    """One MLE: estimate, covariance, and what it cost."""

    natural: np.ndarray
    covariance: np.ndarray
    log_likelihood: float
    converged: bool
    n_iter: int
    n_evals: int
    seconds: float


def solve(
    problem: Problem,
    compiled: CompiledConstraints,
    starts: list[np.ndarray],
    *,
    compute_cov: bool = True,
) -> Solved:
    """Optimise from each start with ``fit``'s optimiser and keep the best."""
    grad_tol = 1e-6 * problem.n_obs
    began = time.perf_counter()
    outcomes = [
        fitting_module._optimise_start(
            u0,
            problem.value_and_grad,
            problem.hessian,
            method="trust-exact",
            grad_tol=grad_tol,
            max_iter=200,
        )
        for u0 in starts
    ]
    seconds = time.perf_counter() - began
    best = min(outcomes, key=lambda o: o.nll)
    x = jnp.asarray(best.x)
    natural = np.asarray(compiled.expand(x), dtype=np.float64)
    covariance = np.full((len(natural), len(natural)), np.nan)
    if compute_cov:
        hess = np.asarray(problem.hessian(x), dtype=np.float64)
        cov_free = np.asarray(covariance_from_hessian(jnp.asarray(hess)))
        jac = np.asarray(jax.jacobian(compiled.expand)(x))
        covariance = jac @ cov_free @ jac.T
    return Solved(
        natural=natural,
        covariance=covariance,
        log_likelihood=-best.nll if best.nll < _PENALTY else float("-inf"),
        converged=bool(best.grad_norm <= grad_tol and best.nll < _PENALTY),
        n_iter=sum(o.n_iter for o in outcomes),
        n_evals=sum(o.n_evals for o in outcomes),
        seconds=seconds,
    )


def starting_points(
    problem: Problem,
    model: BirthDeathModel,
    compiled: CompiledConstraints,
    moments: tuple[float, float] | None,
    knot_times: list[float] | None = None,
) -> list[np.ndarray]:
    """Return the moment start and its birth/death swap (``fit`` with 2 starts)."""
    for lam, mu in [m for m in (moments, (0.8, 0.4)) if m is not None]:
        natural = model.constant_natural(lam, mu, knot_times)
        first = np.asarray(compiled.reduce(natural), dtype=np.float64)
        if float(problem.nll(jnp.asarray(first))) >= _PENALTY:
            continue
        starts = [first]
        swapped = model.swapped(natural)
        if swapped is not None:
            second = np.asarray(compiled.reduce(swapped), dtype=np.float64)
            if float(problem.nll(jnp.asarray(second))) < _PENALTY:
                starts.append(second)
        return starts
    msg = "No finite starting point"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def run_check(keys: tuple[str, ...] | None) -> list[dict[str, Any]]:
    """Validate the closed forms before trusting any timing or fit."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(1)
    for scenario in make_scenarios(keys=keys):
        horizon = scenario.horizon
        edges = edges_for(horizon)
        lo, hi = jnp.asarray(edges[:-1]), jnp.asarray(edges[1:])
        # A bin straddling the knot and a narrow interior bin as extra cases.
        lo_x = jnp.concatenate([lo, jnp.array([4.4, 7.01, 0.0])])
        hi_x = jnp.concatenate([hi, jnp.array([5.6, 7.02, horizon])])
        for label, natural in (
            ("true", scenario.true),
            (
                "perturbed",
                scenario.true * np.resize([1.3, 0.8, 1.2, 0.9], len(scenario.true)),
            ),
        ):
            birth, death = scenario.model.rates(jnp.asarray(natural))
            terms = terms_for(birth, death)
            if isinstance(terms, NestedTerms):
                ns_cf, sp_cf = closed_form_log_masses(terms, lo_x, hi_x)
            else:  # no closed form: 16-node masses against the 48-node reference
                ns_cf, sp_cf = quadrature_log_masses(terms, lo_x, hi_x)
            # Quadrature reference, split at the knot so no panel has a kink.
            knot = 0.5 * horizon
            lo_a, hi_a = lo_x, jnp.minimum(hi_x, knot)
            lo_b, hi_b = jnp.maximum(lo_x, knot), hi_x
            lo_a = jnp.minimum(lo_a, hi_a)
            hi_b = jnp.maximum(hi_b, lo_b)
            ref = [
                quadrature_log_masses(terms, lo_p, hi_p, n_nodes=48)
                for lo_p, hi_p in ((lo_a, hi_a), (lo_b, hi_b))
            ]
            ns_q = jnp.log(sum(jnp.exp(r[0]) for r in ref))
            sp_q = jnp.log(sum(jnp.exp(r[1]) for r in ref))
            smooth = np.ones(len(lo_x), dtype=bool)
            if not isinstance(terms, NestedTerms):  # kinked panels, unsplit
                smooth[[N_BINS, N_BINS + 2]] = False
            err_ns = float(jnp.max(jnp.abs((ns_cf - ns_q)[smooth])))
            err_sp = float(jnp.max(jnp.abs((sp_cf - sp_q)[smooth])))
            sum_ns = float(jnp.sum(jnp.exp(ns_cf[:N_BINS])))
            sum_sp = float(jnp.sum(jnp.exp(sp_cf[:N_BINS])))
            rows.append(
                {
                    "scenario": scenario.name,
                    "params": label,
                    "max_abs_err_log_mass_A": err_ns,
                    "max_abs_err_log_mass_Astar": err_sp,
                    "sum_mass_A": sum_ns,
                    "sum_mass_Astar": sum_sp,
                }
            )
            assert err_ns < 1e-7 and err_sp < 1e-7, (
                scenario.name,
                label,
                err_ns,
                err_sp,
            )
            assert abs(sum_ns - 1) < 1e-9 and abs(sum_sp - 1) < 1e-9

        # Fine-bin limit: the binned density-scale likelihood tends to the exact one.
        ages = simulate_population(scenario, rng)
        ages = ages[ages < horizon]
        n = len(ages)
        ns = jnp.asarray([n], dtype=jnp.int32)
        exact = AgeData(
            non_spine_ages=jnp.asarray(ages),
            spine_ages=None,
            ns=ns,
            end_time=horizon,
            use_unknown_spine=True,
        )
        birth, death = scenario.model.rates(jnp.asarray(scenario.true))
        reference = float(log_likelihood_rates(exact, birth, death))
        terms = terms_for(birth, death)
        for width in (horizon / 200, horizon / 20000):
            lo_f = np.clip(ages - 0.5 * width, 0.0, horizon - width)
            value = float(
                binned_log_likelihood(
                    terms,
                    jnp.asarray(lo_f),
                    jnp.asarray(lo_f + width),
                    jnp.ones(n),
                    ns,
                )
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "params": f"fine bins, width T/{round(horizon / width)}",
                    "exact_ll": reference,
                    "binned_ll": value,
                    "abs_diff": abs(value - reference),
                }
            )
        assert abs(value - reference) < 1e-4 * max(1.0, abs(reference)), (
            value,
            reference,
        )
    return rows


def _bootstrap_ages(scenario: Scenario, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` ages resampled from one simulated population (timing only)."""
    base = simulate_population(scenario, rng, min_n=40)
    base = base[base < scenario.horizon]
    return rng.choice(base, size=n, replace=True)


def run_speed(
    sizes: tuple[int, ...], repeats: int, keys: tuple[str, ...] | None
) -> list[dict[str, Any]]:
    """Per-evaluation cost of the value, gradient and Hessian."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(2)
    for scenario in make_scenarios(keys=keys):
        model, horizon = scenario.model, scenario.horizon
        compiled = Constraints().compile(model.params)
        u = jnp.asarray(np.log(scenario.true) + 0.05)
        for n in sizes:
            ages = _bootstrap_ages(scenario, n, rng)
            binned = BinnedData.from_ages(ages, horizon)
            exact = AgeData(
                non_spine_ages=jnp.asarray(ages),
                spine_ages=None,
                ns=jnp.asarray([n], dtype=jnp.int32),
                end_time=horizon,
                use_unknown_spine=True,
            )
            size = fitting_module._bucket(n)
            per_obs = binned.per_observation(size)
            closed = isinstance(model.birth, PiecewiseConstant)
            problems = {
                "exact ages": exact_problem(exact, model, compiled),
                "binned, per observation": binned_problem(per_obs, n, model, compiled),
                "binned, aggregated": binned_problem(
                    binned.aggregated(), n, model, compiled
                ),
            }
            if closed:
                problems["binned, quadrature (per obs)"] = binned_problem(
                    per_obs, n, model, compiled, quadrature=True
                )
            for name, problem in problems.items():
                row: dict[str, Any] = {
                    "scenario": scenario.name,
                    "n_ages": n,
                    "variant": name,
                }
                for label, fn in (
                    ("value", problem.nll),
                    ("grad", problem.value_and_grad),
                    ("hess", problem.hessian),
                ):
                    first, run_ms = _compile_and_run(fn, u, repeats=repeats)
                    row[f"{label}_compile_s"] = first
                    row[f"{label}_ms"] = run_ms
                rows.append(row)
    return rows


def _coverage(est: np.ndarray, se: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(est - truth) <= 1.96 * se))


def run_fit(
    replicates: int, seed: int, keys: tuple[str, ...] | None
) -> dict[str, list[dict[str, Any]]]:
    """Simulate, bin at ``T/20``, fit both ways and summarise."""
    tables: dict[str, list[dict[str, Any]]] = {}
    for scenario in make_scenarios(keys=keys):
        model, horizon = scenario.model, scenario.horizon
        compiled = Constraints().compile(model.params)
        rng = np.random.default_rng(seed)
        # Per-observation binned fits only add timing for the closed form; for
        # quadrature they are slow and give the same estimates as aggregated.
        with_per_obs = isinstance(model.birth, PiecewiseConstant)
        methods = ("exact ages", "binned (aggregated)") + (
            ("binned (per obs)",) if with_per_obs else ()
        )
        solved: dict[str, list[Solved]] = {m: [] for m in methods}
        first_call: dict[str, float] = {}
        for rep in range(replicates):
            ages = simulate_population(scenario, rng)
            exact = AgeData.from_groups([ages], horizon)
            binned = BinnedData.from_ages(ages, horizon)
            mid = binned.midpoint_data(horizon)
            bucket = fitting_module._bucket(binned.n)
            problems = {
                methods[0]: (
                    exact_problem(exact, model, compiled),
                    fitting_module.moment_initial_rates(exact),
                ),
                methods[1]: (
                    binned_problem(binned.aggregated(), binned.n, model, compiled),
                    fitting_module.moment_initial_rates(mid),
                ),
            }
            if with_per_obs:
                problems[methods[2]] = (
                    binned_problem(
                        binned.per_observation(bucket), binned.n, model, compiled
                    ),
                    fitting_module.moment_initial_rates(mid),
                )
            for name, (problem, moments) in problems.items():
                starts = starting_points(problem, model, compiled, moments)
                began = time.perf_counter()
                result = solve(problem, compiled, starts)  # may include compilation
                if rep == 0:
                    first_call[name] = time.perf_counter() - began
                # Warm timing: the second solve reuses every compiled function.
                result = solve(problem, compiled, starts)
                solved[name].append(result)

        ok = np.array(
            [all(solved[m][r].converged for m in methods) for r in range(replicates)]
        )
        est = {m: np.array([s.natural for s in solved[m]]) for m in methods}
        cov = {m: np.array([s.covariance for s in solved[m]]) for m in methods}
        truth = scenario.true
        growth = scenario.growth

        timing_rows = []
        for m in methods:
            runs = solved[m]
            timing_rows.append(
                {
                    "method": m,
                    "converged": f"{sum(s.converged for s in runs)}/{replicates}",
                    "median_iters": statistics.median(s.n_iter for s in runs),
                    "median_evals": statistics.median(s.n_evals for s in runs),
                    "median_fit_s": statistics.median(s.seconds for s in runs),
                    "first_call_s": first_call[m],
                }
            )
        tables[f"{scenario.name}: time to convergence (2 starts)"] = timing_rows

        acc_rows = []
        for m in methods:
            for j, label in enumerate(scenario.labels):
                e, se = est[m][ok, j], np.sqrt(cov[m][ok, j, j])
                acc_rows.append(
                    {
                        "method": m,
                        "param": label,
                        "true": float(truth[j]),
                        "bias": float(np.mean(e - truth[j])),
                        "median_err": float(np.median(e - truth[j])),
                        "frac_at_zero": float(np.mean(e < 1e-3)),
                        "rmse": float(np.sqrt(np.mean((e - truth[j]) ** 2))),
                        "median_se": float(np.median(se)),
                        "coverage95": _coverage(e, se, truth[j]),
                    }
                )
            for g, label in enumerate(scenario.growth_labels):
                vec = growth[g]
                e = est[m][ok] @ vec
                se = np.sqrt(np.einsum("i,rij,j->r", vec, cov[m][ok], vec))
                acc_rows.append(
                    {
                        "method": m,
                        "param": label,
                        "true": float(vec @ truth),
                        "bias": float(np.mean(e - vec @ truth)),
                        "median_err": float(np.median(e - vec @ truth)),
                        "frac_at_zero": 0.0,
                        "rmse": float(np.sqrt(np.mean((e - vec @ truth) ** 2))),
                        "median_se": float(np.median(se)),
                        "coverage95": _coverage(e, se, vec @ truth),
                    }
                )
        tables[
            f"{scenario.name}: accuracy ({int(ok.sum())} replicates, all converged)"
        ] = acc_rows

        loss_rows = []
        for j, label in enumerate(scenario.labels):
            se_e = np.sqrt(cov[methods[0]][ok, j, j])
            se_b = np.sqrt(cov[methods[1]][ok, j, j])
            diff = est[methods[1]][ok, j] - est[methods[0]][ok, j]
            same = (
                est[methods[1]][ok, j] - est[methods[2]][ok, j]
                if with_per_obs
                else np.zeros(1)
            )
            loss_rows.append(
                {
                    "param": label,
                    "median_se_ratio_binned_over_exact": float(np.median(se_b / se_e)),
                    "median_abs_diff_over_exact_se": float(
                        np.median(np.abs(diff) / se_e)
                    ),
                    "max_abs_agg_vs_per_obs": float(np.max(np.abs(same))),
                }
            )
        tables[f"{scenario.name}: information lost to binning"] = loss_rows
    return tables


# ---------------------------------------------------------------------------


def main() -> None:
    """Run the requested sections and print Markdown tables."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=("constant", "two", "linear"),
        default=None,
        help="restrict to these scenarios (default: all)",
    )
    parser.add_argument("--quick", action="store_true", help="fewer repeats/replicates")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--out", type=Path, default=Path("_output/bench_binned_ages.json")
    )
    args = parser.parse_args()

    keys = tuple(args.scenarios) if args.scenarios else None
    results: dict[str, Any] = {"backend": jax.default_backend()}
    if "check" in args.only:
        results["check"] = run_check(keys)
        masses = [r for r in results["check"] if "sum_mass_A" in r]
        fine = [r for r in results["check"] if "sum_mass_A" not in r]
        _table(masses, "Closed-form bin masses vs quadrature")
        _table(fine, "Fine-bin limit vs exact-age likelihood")
    if "speed" in args.only:
        sizes = (100, 1000) if args.quick else (100, 1000, 10000)
        results["speed"] = run_speed(sizes, 5 if args.quick else 30, keys)
        _table(results["speed"], "Cost per evaluation")
    if "fit" in args.only:
        results["fit"] = run_fit(10 if args.quick else 50, args.seed, keys)
        for title, rows in results["fit"].items():
            _table(rows, title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
