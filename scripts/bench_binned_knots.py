r"""Joint Newton over rates and a knot time, on binned and on exact ages.

With exact ages and piecewise-constant rates the likelihood jumps whenever the
knot crosses an observed birth time $T - a$, so ``fit_knots`` profiles the knot.
With ages known only to lie in bins, every observation contributes a bin mass,
which is an integral of the rates and therefore continuous in the knot.
(It is not differentiable where the knot crosses a bin edge: the first
derivative has a finite jump there, with the kinks a bin width apart.)
This script asks whether a plain trust-region Newton over rates *and* knot then
matches the profile search, and how much faster it is.

Methods per replicate:

``exact/profile``
    ``fit_knots`` with its defaults (64-point scan, Nelder-Mead, joint polish).
``exact/newton``
    ``fit`` with the knot free, from three knot starts.
``binned/profile``
    The ``fit_knots`` search (scan, Nelder-Mead on the logit knot, final fit)
    on the binned objective.
``binned/newton``
    Joint trust-region Newton from the same three knot starts.

Sections (run all, or pick with ``--only``):

``smooth``
    The likelihood in the knot: jumps, derivative kinks, and autodiff against
    finite differences, exact against binned.
``fit``
    Speed and final answers.

Usage
-----
    uv run python scripts/bench_binned_knots.py --quick
    uv run python scripts/bench_binned_knots.py --scenarios pc
"""

import argparse
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from bench_binned_ages import (
    BinnedData,
    Scenario,
    binned_log_likelihood,
    binned_problem,
    simulate_population,
    solve,
    starting_points,
)
from bench_fitting import _table

from bdpinf.inference import AgeData, BirthDeathModel, Constraints, FreeKnots, fit
from bdpinf.inference import fitting as fitting_module
from bdpinf.inference.knots import fit_knots
from bdpinf.inference.likelihood import log_likelihood_rates
from bdpinf.models.math_general import DEFAULT_NODES, terms_for

jax.config.update("jax_enable_x64", True)

SECTIONS = ("smooth", "fit")
HORIZON = 10.0
TRUE_TAU = 3.7
TAU_STARTS = (2.5, 5.0, 7.5)
MIN_POPULATION = 100


@dataclasses.dataclass(frozen=True)
class KnotScenario:
    """A free-knot model, its truth, and the constraint tying the death rate."""

    key: str
    scenario: Scenario
    equal: list[list[str]]
    run_exact: bool

    @property
    def model(self) -> BirthDeathModel:
        """The free-knot model."""
        return self.scenario.model

    @property
    def n_params(self) -> int:
        """Natural-scale parameter count, knot last."""
        return len(self.scenario.true)

    @property
    def reported(self) -> list[int]:
        """Indices to report: births, the first (tied) death rate, then the knot."""
        n_birth = self.model.n_birth
        return [*range(n_birth), n_birth, self.n_params - 1]


def make_knot_scenarios() -> list[KnotScenario]:
    """Piecewise-constant and continuous piecewise-linear, one free knot."""
    knots = FreeKnots(1, end_time=HORIZON)
    pc = BirthDeathModel.piecewise_constant(knots)
    pl = BirthDeathModel.piecewise_linear(knots)
    return [
        KnotScenario(
            "pc",
            Scenario(
                "piecewise constant",
                pc,
                np.array([1.5, 0.6, 0.4, 0.4, TRUE_TAU]),
                ("birth.v0", "birth.v1", "death.v0", "death.v1", "tau0"),
                np.zeros((0, 5)),
                (),
                HORIZON,
            ),
            [["death.v0", "death.v1"]],
            run_exact=True,
        ),
        KnotScenario(
            "pl",
            Scenario(
                "piecewise linear",
                pl,
                np.array([0.8, 1.6, 0.6, 0.4, 0.4, 0.4, TRUE_TAU]),
                (
                    "birth.knot0",
                    "birth.knot1",
                    "birth.knot2",
                    "death.knot0",
                    "death.knot1",
                    "death.knot2",
                    "tau0",
                ),
                np.zeros((0, 7)),
                (),
                HORIZON,
            ),
            [["death.knot0", "death.knot1", "death.knot2"]],
            run_exact=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Smoothness of the likelihood in the knot
# ---------------------------------------------------------------------------


def run_smooth(keys: tuple[str, ...] | None) -> list[dict[str, Any]]:
    """Scan the log-likelihood in the knot with the rates at their true values."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(11)
    for ks in make_knot_scenarios():
        if keys and ks.key not in keys:
            continue
        model, true = ks.model, jnp.asarray(ks.scenario.true)
        ages = simulate_population(ks.scenario, rng, min_n=MIN_POPULATION)
        binned = BinnedData.from_ages(ages, HORIZON)
        lo, hi, counts = (jnp.asarray(a) for a in binned.aggregated())
        ns = jnp.asarray([binned.n], dtype=jnp.int32)
        exact = AgeData.from_groups([ages], HORIZON)

        def binned_ll(
            tau: jax.Array,
            model: BirthDeathModel = model,
            true: jax.Array = true,
            arrays: tuple[jax.Array, ...] = (lo, hi, counts, ns),
        ) -> jax.Array:
            birth, death = model.rates(true.at[-1].set(tau))
            terms = terms_for(birth, death, n_nodes=DEFAULT_NODES)
            return binned_log_likelihood(terms, *arrays)

        def exact_ll(
            tau: jax.Array,
            model: BirthDeathModel = model,
            true: jax.Array = true,
            exact: AgeData = exact,
        ) -> jax.Array:
            birth, death = model.rates(true.at[-1].set(tau))
            return log_likelihood_rates(exact, birth, death, n_nodes=DEFAULT_NODES)

        grid = jnp.linspace(1.0, 9.0, 8001)  # step 0.001
        h = float(grid[1] - grid[0])
        # Probe inside the smooth cells, away from the kinks at bin edges.
        draws = rng.uniform(1.0, 9.0, 40)
        width = HORIZON / 20
        draws = draws[np.abs(draws - width * np.round(draws / width)) > 0.02]
        probes = jnp.asarray(draws[:25])
        for name, func in (("exact ages", exact_ll), ("binned", binned_ll)):
            values = np.asarray(jax.jit(jax.vmap(func))(grid))
            grads = np.asarray(jax.jit(jax.vmap(jax.grad(func)))(grid))
            # Trapezoid residual: O(h^3) for a C^1 curve, the jump size otherwise.
            resid = np.abs(np.diff(values) - 0.5 * h * (grads[:-1] + grads[1:]))
            grad_jumps = np.abs(np.diff(grads))
            step = 1e-4
            fd = np.array(
                [
                    (float(func(p + step)) - float(func(p - step))) / (2 * step)
                    for p in probes
                ]
            )
            ad = np.asarray(jax.vmap(jax.grad(func))(probes))
            rel_err = np.abs(ad - fd) / (1.0 + np.abs(fd))
            rows.append(
                {
                    "scenario": ks.scenario.name,
                    "likelihood": name,
                    "max_value_jump": float(resid.max()),
                    "n_value_jumps_over_1e-3": int((resid > 1e-3).sum()),
                    "max_grad_jump": float(grad_jumps.max()),
                    "n_grad_jumps_over_1": int((grad_jumps > 1.0).sum()),
                    "max_abs_grad": float(np.abs(grads).max()),
                    "median_ad_vs_fd_rel_err": float(np.median(rel_err)),
                    "frac_ad_vs_fd_below_1e-5": float(np.mean(rel_err < 1e-5)),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Fitting methods
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Outcome:
    """One method on one replicate."""

    natural: np.ndarray
    log_likelihood: float
    converged: bool
    seconds: float
    work: int  # inner fits for a profile, optimiser iterations for Newton
    tau_se: float = float("nan")


def binned_newton(
    ks: KnotScenario, binned: BinnedData, moments: tuple[float, float] | None
) -> Outcome:
    """Joint trust-region Newton over rates and knot from three knot starts."""
    model = ks.model
    compiled = Constraints(equal=ks.equal).compile(model.params)
    problem = binned_problem(binned.aggregated(), binned.n, model, compiled)
    starts: list[np.ndarray] = []
    for tau in TAU_STARTS:
        starts += starting_points(problem, model, compiled, moments, [tau])
    result = solve(problem, compiled, starts)
    return Outcome(
        result.natural,
        result.log_likelihood,
        result.converged,
        result.seconds,
        result.n_iter,
        float(np.sqrt(result.covariance[-1, -1])),
    )


def binned_profile(
    ks: KnotScenario,
    binned: BinnedData,
    moments: tuple[float, float] | None,
    *,
    n_candidates: int = 64,
    n_refine: int = 2,
    maxfev: int = 60,
    polish: bool = False,
) -> Outcome:
    """Run the ``fit_knots`` search on the binned objective: scan, refine, final fit.

    With ``polish``, finish with a joint Newton fit of rates and knot from the
    profile optimum, kept if it converges without lowering the likelihood.
    """
    model = ks.model
    tried: dict[float, float] = {}
    best: dict[str, Any] = {"ll": -np.inf, "natural": None}
    inner = {"fits": 0}
    arrays = binned.aggregated()

    def fit_at(tau: float, *, compute_cov: bool = False) -> Any:
        compiled = Constraints(fixed={"tau0": tau}, equal=ks.equal).compile(
            model.params
        )
        problem = binned_problem(arrays, binned.n, model, compiled)
        if best["natural"] is None:
            starts = starting_points(problem, model, compiled, moments, [tau])
        else:
            warm = np.array(best["natural"])
            warm[-1] = tau
            starts = [np.asarray(compiled.reduce(jnp.asarray(warm)), dtype=np.float64)]
            if float(problem.nll(jnp.asarray(starts[0]))) >= fitting_module._PENALTY:
                starts = starting_points(problem, model, compiled, moments, [tau])
        inner["fits"] += 1
        return solve(problem, compiled, starts, compute_cov=compute_cov)

    def profile(tau: float) -> float:
        key = round(float(tau), 9)
        if key in tried:
            return tried[key]
        result = fit_at(float(tau))
        tried[key] = result.log_likelihood
        if result.log_likelihood > best["ll"]:
            best["ll"], best["natural"] = result.log_likelihood, result.natural
        return result.log_likelihood

    began = time.perf_counter()
    candidates = np.linspace(0.0, HORIZON, n_candidates + 2)[1:-1]
    for tau in candidates:
        profile(tau)
    order = np.argsort([-tried[round(float(t), 9)] for t in candidates])
    for tau in candidates[order[:n_refine]]:
        frac = float(np.clip(tau / HORIZON, 1e-6, 1 - 1e-6))
        scipy.optimize.minimize(
            lambda u: -profile(HORIZON / (1.0 + np.exp(-u[0]))),
            [np.log(frac / (1 - frac))],
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-9, "maxfev": maxfev},
        )
    tau_best = float(best["natural"][-1])
    final = fit_at(tau_best, compute_cov=True)
    tau_se = float("nan")
    if polish:
        compiled = Constraints(equal=ks.equal).compile(model.params)
        problem = binned_problem(arrays, binned.n, model, compiled)
        start = np.asarray(
            compiled.reduce(jnp.asarray(final.natural)), dtype=np.float64
        )
        joint = solve(problem, compiled, [start])
        inner["fits"] += joint.n_iter
        if joint.converged and joint.log_likelihood >= final.log_likelihood - 1e-9:
            final = joint
            tau_se = float(np.sqrt(joint.covariance[-1, -1]))
    seconds = time.perf_counter() - began
    return Outcome(
        final.natural,
        final.log_likelihood,
        final.converged,
        seconds,
        inner["fits"],
        tau_se,
    )


def exact_profile(ks: KnotScenario, data: AgeData) -> Outcome:
    """Run the existing ``fit_knots`` with its defaults."""
    began = time.perf_counter()
    result = fit_knots(data, ks.model, constraints=Constraints(equal=ks.equal), seed=0)
    seconds = time.perf_counter() - began
    return Outcome(
        np.asarray(result.fit.natural, dtype=np.float64),
        result.log_likelihood,
        bool(result.fit.converged),
        seconds,
        len(result.profile_points),
    )


def exact_newton(ks: KnotScenario, data: AgeData) -> Outcome:
    """Run the existing ``fit`` with the knot free, from three knot starts."""
    model = ks.model
    lam, mu = fitting_module.moment_initial_rates(data) or (0.8, 0.4)
    best: Any = None
    iters = 0
    began = time.perf_counter()
    for tau in TAU_STARTS:
        result = fit(
            data,
            model,
            constraints=Constraints(equal=ks.equal),
            init=model.constant_natural(lam, mu, [tau]),
            n_starts=2,
            compute_covariance=False,
        )
        iters += result.n_iter
        if best is None or result.log_likelihood > best.log_likelihood:
            best = result
    seconds = time.perf_counter() - began
    return Outcome(
        np.asarray(best.natural, dtype=np.float64),
        best.log_likelihood,
        bool(best.converged),
        seconds,
        iters,
    )


def _summarise(
    ks: KnotScenario, outcomes: dict[str, list[Outcome]], first_call: dict[str, float]
) -> dict[str, list[dict[str, Any]]]:
    """Build the speed, answer and agreement tables for one scenario."""
    title = ks.scenario.name
    truth = ks.scenario.true
    tables: dict[str, list[dict[str, Any]]] = {}
    names = list(outcomes)

    tables[f"{title}: speed"] = [
        {
            "method": m,
            "converged": f"{sum(o.converged for o in outcomes[m])}/{len(outcomes[m])}",
            "median_seconds": statistics.median(o.seconds for o in outcomes[m][1:]),
            "median_work": statistics.median(o.work for o in outcomes[m][1:]),
            "first_call_seconds": first_call[m],
        }
        for m in names
    ]

    answers = []
    for m in names:
        est = np.array([o.natural for o in outcomes[m]])
        for j in ks.reported:
            err = est[:, j] - truth[j]
            answers.append(
                {
                    "method": m,
                    "param": ks.scenario.labels[j],
                    "true": float(truth[j]),
                    "bias": float(err.mean()),
                    "median_abs_err": float(np.median(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                }
            )
    tables[f"{title}: answers vs truth"] = answers

    agreement = []
    pairs = [
        ("exact/newton", "exact/profile"),
        ("binned/newton", "binned/profile"),
        ("binned/hybrid", "binned/profile"),
        ("binned/hybrid64", "binned/profile"),
    ]
    for newton, profile in pairs:
        if newton not in outcomes or profile not in outcomes:
            continue
        d_tau = np.array(
            [
                a.natural[-1] - b.natural[-1]
                for a, b in zip(outcomes[newton], outcomes[profile], strict=True)
            ]
        )
        d_ll = np.array(
            [
                a.log_likelihood - b.log_likelihood
                for a, b in zip(outcomes[newton], outcomes[profile], strict=True)
            ]
        )
        agreement.append(
            {
                "pair": f"{newton} - {profile}",
                "median_abs_dtau": float(np.median(np.abs(d_tau))),
                "max_abs_dtau": float(np.abs(d_tau).max()),
                "median_dll": float(np.median(d_ll)),
                "min_dll": float(d_ll.min()),
                "frac_newton_within_1e-3": float(np.mean(d_ll > -1e-3)),
                "frac_newton_higher_by_1e-3": float(np.mean(d_ll > 1e-3)),
            }
        )
    tables[f"{title}: Newton against profile"] = agreement

    for name in ("binned/newton", "binned/hybrid"):
        if name not in outcomes:
            continue
        runs = outcomes[name]
        tau_hat = np.array([o.natural[-1] for o in runs])
        se = np.array([o.tau_se for o in runs])
        tables[f"{title}: {name} knot Wald interval"] = [
            {
                "median_se": float(np.nanmedian(se)),
                "empirical_sd": float(np.std(tau_hat, ddof=1)),
                "coverage95": float(
                    np.nanmean(np.abs(tau_hat - TRUE_TAU) <= 1.96 * se)
                ),
                "n_finite_se": int(np.isfinite(se).sum()),
            }
        ]
    return tables


def run_fit(
    replicates: int, seed: int, keys: tuple[str, ...] | None
) -> dict[str, list[dict[str, Any]]]:
    """Simulate, then run all four methods on every replicate."""
    tables: dict[str, list[dict[str, Any]]] = {}
    for ks in make_knot_scenarios():
        if keys and ks.key not in keys:
            continue
        rng = np.random.default_rng(seed)
        methods = [
            "binned/newton",
            "binned/profile",
            "binned/hybrid",
            "binned/hybrid64",
        ]
        if ks.run_exact:
            methods = ["exact/newton", "exact/profile", *methods]
        outcomes: dict[str, list[Outcome]] = {m: [] for m in methods}
        for rep in range(replicates):
            ages = simulate_population(ks.scenario, rng, min_n=MIN_POPULATION)
            binned = BinnedData.from_ages(ages, HORIZON)
            exact = AgeData.from_groups([ages], HORIZON)
            moments = fitting_module.moment_initial_rates(binned.midpoint_data(HORIZON))
            runners = {
                "exact/newton": lambda e=exact, ks=ks: exact_newton(ks, e),
                "exact/profile": lambda e=exact, ks=ks: exact_profile(ks, e),
                "binned/newton": lambda b=binned, m=moments, ks=ks: binned_newton(
                    ks, b, m
                ),
                "binned/profile": lambda b=binned, m=moments, ks=ks: binned_profile(
                    ks, b, m
                ),
                "binned/hybrid": lambda b=binned, m=moments, ks=ks: binned_profile(
                    ks, b, m, n_candidates=12, n_refine=1, maxfev=20, polish=True
                ),
                "binned/hybrid64": lambda b=binned, m=moments, ks=ks: binned_profile(
                    ks, b, m, polish=True
                ),
            }
            for name in methods:
                outcomes[name].append(runners[name]())
            print(f"{ks.key}: replicate {rep + 1}/{replicates} done", flush=True)
        first_call = {m: outcomes[m][0].seconds for m in methods}
        tables.update(_summarise(ks, outcomes, first_call))
    return tables


# ---------------------------------------------------------------------------


def main() -> None:
    """Run the requested sections and print Markdown tables."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--scenarios", nargs="*", choices=("pc", "pl"), default=None)
    parser.add_argument("--quick", action="store_true", help="fewer replicates")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--out", type=Path, default=Path("_output/bench_binned_knots.json")
    )
    args = parser.parse_args()
    keys = tuple(args.scenarios) if args.scenarios else None

    results: dict[str, Any] = {"backend": jax.default_backend()}
    if "smooth" in args.only:
        results["smooth"] = run_smooth(keys)
        _table(results["smooth"], "The likelihood in the knot (rates at truth)")
    if "fit" in args.only:
        replicates = args.replicates or (4 if args.quick else 20)
        results["fit"] = run_fit(replicates, args.seed, keys)
        for title, rows in results["fit"].items():
            _table(rows, title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
