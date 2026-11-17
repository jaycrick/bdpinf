"""Benchmark the fitting stack: derivatives, quadrature and optimisers.

Sections (run all, or pick with ``--only``):

``derivatives``
    Compile and run time of the gradient and Hessian variants (reverse,
    forward-over-reverse, reverse-over-reverse, forward-over-forward), and their
    accuracy against central finite differences.
``explicit``
    Mathematica-derived closed-form scores (single epoch, constant rates, the only
    case where explicit derivatives exist) against JAX autodiff: agreement,
    per-evaluation cost, and full-fit wall time cold and warm.
``quadrature``
    Gauss-Legendre node count against a 512-node reference, for moderate and
    extreme piecewise-linear rates.
``optimisers``
    Trust-region Newton against L-BFGS-B against Newton root-finding on the score
    (``scipy.optimize.root``) from identical starts.

Usage
-----
    uv run python scripts/bench_fitting.py --quick
    uv run python scripts/bench_fitting.py --only explicit optimisers
    uv run python scripts/bench_fitting.py --out _output/bench_fitting.json
"""

import argparse
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
import scipy.optimize

from bdpinf.inference import AgeData, BirthDeathModel, Constraints, fit
from bdpinf.inference import fitting as fitting_module
from bdpinf.inference.likelihood import log_likelihood_rates
from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate
from bdpinf.simulation import simulate_ages_from_rates

jax.config.update("jax_enable_x64", True)

# The optimiser benchmark times the jitted objective that ``fit`` itself builds.
_prepare_problem = fitting_module._prepare_problem  # noqa: SLF001  # benchmarks fitting internals

SECTIONS = ("derivatives", "explicit", "quadrature", "optimisers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median_ms(fn: Callable[[], Any], repeats: int) -> float:
    """Median wall time of ``fn`` in milliseconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def _compile_and_run(
    fn: Callable[..., Any], *args: Any, repeats: int
) -> tuple[float, float]:
    """Return (first-call seconds, steady-state milliseconds) of a jitted ``fn``."""
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - start
    return first, _median_ms(lambda: fn(*args), repeats)


def _simulate(
    birth: PiecewiseConstantRate | PiecewiseLinearRate,
    death: PiecewiseConstantRate | PiecewiseLinearRate,
    seed: int,
    min_n: int = 40,
    max_n: int | None = None,
) -> AgeData:
    """Draw ordered ages (spine first) from the exact age laws."""
    rng = np.random.default_rng(seed)
    while True:
        sample = simulate_ages_from_rates(birth, death, rng)
        if sample.population >= min_n and (max_n is None or sample.population <= max_n):
            return AgeData.from_groups(
                [sample.all_ages], float(birth.end_time), is_ordered=True
            )


def _constant_pair(
    birth: float, death: float, horizon: float
) -> tuple[PiecewiseConstantRate, PiecewiseConstantRate]:
    times = jnp.array([horizon])
    return (
        PiecewiseConstantRate(jnp.array([birth]), times),
        PiecewiseConstantRate(jnp.array([death]), times),
    )


def _objective(
    data: AgeData, model: BirthDeathModel
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array]:
    """Return a plain negative log-likelihood of log-parameters and a start."""
    compiled = Constraints().compile(model.params)

    def nll(u: jax.Array) -> jax.Array:
        birth, death = model.rates(compiled.expand(u))
        return -log_likelihood_rates(data, birth, death)

    start = compiled.reduce(model.constant_natural(0.9, 0.5))
    return nll, start


def _finite_difference(
    fn: Callable[[jax.Array], jax.Array], x: jax.Array, step: float = 1e-5
) -> np.ndarray:
    """Central-difference gradient of a scalar function."""
    x_np = np.asarray(x)
    out = np.zeros_like(x_np)
    for i in range(x_np.size):
        up, down = x_np.copy(), x_np.copy()
        up[i] += step
        down[i] -= step
        out[i] = (float(fn(jnp.asarray(up))) - float(fn(jnp.asarray(down)))) / (
            2 * step
        )
    return out


def _grad_component(
    fn: Callable[[jax.Array], jax.Array], index: int
) -> Callable[[jax.Array], jax.Array]:
    """Return ``u -> d fn / d u[index]``."""
    return lambda u: jax.grad(fn)(u)[index]


def _table(rows: list[dict[str, Any]], title: str) -> None:
    """Print rows as a Markdown table."""
    print(f"\n### {title}\n")
    if not rows:
        print("(no rows)")
        return
    columns = list(rows[0])
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            cells.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        print("| " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def bench_derivatives(repeats: int) -> list[dict[str, Any]]:
    """Compile/run time and accuracy of the derivative variants."""
    rows: list[dict[str, Any]] = []
    cases = {
        "PC K=1": (BirthDeathModel.piecewise_constant([6.0]), (0.9, 0.5)),
        "PC K=3": (BirthDeathModel.piecewise_constant([2.0, 4.0, 6.0]), (0.9, 0.5)),
        "PL K=2": (BirthDeathModel.piecewise_linear([3.0, 6.0]), (0.9, 0.5)),
        "PL K=4": (BirthDeathModel.piecewise_linear([1.5, 3.0, 4.5, 6.0]), (0.9, 0.5)),
    }
    for label, (model, _) in cases.items():
        birth = model.birth.build(model.birth.constant_values(1.0))
        death = model.death.build(model.death.constant_values(0.4))
        data = _simulate(birth, death, seed=7, min_n=150)
        nll, x = _objective(data, model)
        x = x + 0.05
        reference_grad = _finite_difference(nll, x)
        variants: dict[str, Callable[[jax.Array], jax.Array]] = {
            "grad: reverse": jax.jit(jax.grad(nll)),
            "grad: forward": jax.jit(jax.jacfwd(nll)),
            "hess: fwd-over-rev": jax.jit(jax.hessian(nll)),
            "hess: rev-over-rev": jax.jit(jax.jacrev(jax.jacrev(nll))),
            "hess: fwd-over-fwd": jax.jit(jax.jacfwd(jax.jacfwd(nll))),
        }
        reference_hess = np.stack(
            [
                _finite_difference(
                    lambda u, i=i, f=nll: jax.grad(f)(u)[i], x, step=1e-4
                )
                for i in range(x.size)
            ]
        )
        for name, fn in variants.items():
            first, run_ms = _compile_and_run(fn, x, repeats=repeats)
            value = np.asarray(fn(x))
            if name.startswith("grad"):
                err = np.max(
                    np.abs(value - reference_grad) / (1 + np.abs(reference_grad))
                )
            else:
                err = np.max(
                    np.abs(value - reference_hess) / (1 + np.abs(reference_hess))
                )
            rows.append(
                {
                    "model": label,
                    "n_params": int(x.size),
                    "n_ages": data.n_observations,
                    "variant": name,
                    "compile_s": first,
                    "run_ms": run_ms,
                    "max_rel_err_vs_fd": float(err),
                }
            )
    return rows


def _explicit_value_and_grad(
    lam: float, mu: float, *, horizon: float, spine: float, ages: np.ndarray, n: int
) -> tuple[float, np.ndarray]:
    """Closed-form log-likelihood and (lam, mu) score for constant rates.

    Scores are the Mathematica ``D[Log[...], {{l, m}}]`` of the closed forms of
    ``AgeDistCommon.wl`` (known spine below ``T``, survival-conditioned count).
    """
    e = np.exp((lam - mu) * horizon)
    pi = (lam - mu) / (lam * e - mu)
    a = ages
    el, em = np.exp(lam * horizon), np.exp(mu * horizon)
    value = np.log(pi) + (n - 1) * np.log1p(-pi)
    value += np.log(
        (lam * np.exp(lam * (horizon - a)) - mu * np.exp(mu * (horizon - a)))
        / (el - em)
    ).sum()
    value += np.log(
        lam * mu * (np.exp(-mu * spine) - np.exp(-lam * spine)) / (lam - mu)
    )

    common = (e * lam + mu * (n - 1) - lam * n) / (
        (e - 1) * (lam - mu) * (e * lam - mu)
    )
    s_pop = common * np.array(
        [
            (-(e * lam**2 * horizon) + mu * (-1 + e * (1 + lam * horizon))) / lam,
            1 + e * (-1 + lam * horizon - mu * horizon),
        ]
    )
    den = (el - em) * (
        np.exp(a * mu + lam * horizon) * lam - np.exp(a * lam + mu * horizon) * mu
    )
    s_a_lam = el * (
        np.exp(a * mu + lam * horizon) * (1 - a * lam)
        + np.exp(a * lam + mu * horizon) * mu * horizon
        + np.exp(mu * (a + horizon)) * (-1 + a * lam - lam * horizon)
    )
    s_a_mu = -em * (
        np.exp(a * lam + mu * horizon) * (-1 + a * mu)
        - np.exp(a * mu + lam * horizon) * lam * horizon
        + np.exp(lam * (a + horizon)) * (1 - a * mu + mu * horizon)
    )
    s_a = np.array([(s_a_lam / den).sum(), (s_a_mu / den).sum()])
    ea_l, ea_m = np.exp(spine * lam), np.exp(spine * mu)
    s_s = np.array(
        [
            ((-ea_l + ea_m) * mu - spine * ea_m * lam * (-lam + mu))
            / ((ea_l - ea_m) * lam * (lam - mu)),
            (-(ea_m * lam) + ea_l * (lam - spine * lam * mu + spine * mu**2))
            / ((ea_l - ea_m) * (lam - mu) * mu),
        ]
    )
    return float(value), s_pop + s_a + s_s


def bench_explicit(repeats: int, quick: bool) -> list[dict[str, Any]]:
    """Explicit Mathematica scores against JAX autodiff (single epoch, constant)."""
    rows: list[dict[str, Any]] = []
    sizes = (150, 1500) if quick else (150, 1500, 15000)
    for target_n in sizes:
        # Longer windows give larger populations.
        t_here = {150: 7.0, 1500: 11.0, 15000: 15.0}[target_n]
        birth, death = _constant_pair(1.0, 0.6, t_here)
        data = _simulate(birth, death, seed=3, min_n=target_n // 2)
        spine = float(data.spine_ages[0])  # type: ignore[index]
        ages = np.asarray(data.non_spine_ages)
        n = data.n_observations
        times = jnp.array([t_here])

        def nll(
            theta: jax.Array, data: AgeData = data, times: jax.Array = times
        ) -> jax.Array:
            return -log_likelihood_rates(
                data,
                PiecewiseConstantRate(theta[:1], times),
                PiecewiseConstantRate(theta[1:], times),
            )

        theta = jnp.array([1.05, 0.55])
        auto = jax.jit(jax.value_and_grad(nll))
        compile_s, auto_ms = _compile_and_run(auto, theta, repeats=repeats)
        value, grad = auto(theta)
        explicit_value, explicit_grad = _explicit_value_and_grad(
            1.05, 0.55, horizon=t_here, spine=spine, ages=ages, n=n
        )
        explicit_ms = _median_ms(
            lambda a=ages, s=spine, n=n, h=t_here: _explicit_value_and_grad(
                1.05, 0.55, horizon=h, spine=s, ages=a, n=n
            ),
            repeats,
        )
        rows.append(
            {
                "n_ages": n,
                "autodiff_compile_s": compile_s,
                "autodiff_eval_ms": auto_ms,
                "explicit_eval_ms": explicit_ms,
                "value_rel_diff": abs(float(-value) - explicit_value)
                / abs(explicit_value),
                "grad_rel_diff": float(
                    np.max(
                        np.abs(-np.asarray(grad) - explicit_grad)
                        / (1 + np.abs(explicit_grad))
                    )
                ),
            }
        )
    return rows


def bench_explicit_fit(repeats: int) -> list[dict[str, Any]]:
    """Full single-epoch fit wall time: cold and warm ``fit`` vs explicit L-BFGS."""
    rows: list[dict[str, Any]] = []
    horizon = 9.0
    model = BirthDeathModel.piecewise_constant([horizon])
    birth, death = _constant_pair(1.0, 0.6, horizon)
    datasets = [
        _simulate(birth, death, seed=s, min_n=80, max_n=120)
        for s in range(10, 10 + repeats)
    ]

    cold_start = time.perf_counter()
    fit(datasets[0], model, compute_covariance=False)
    cold = time.perf_counter() - cold_start
    warm = []
    for data in datasets[1:]:
        start = time.perf_counter()
        fit(data, model, compute_covariance=False)
        warm.append(time.perf_counter() - start)

    explicit_times = []
    for data in datasets:
        spine = float(data.spine_ages[0])  # type: ignore[index]
        ages = np.asarray(data.non_spine_ages)
        n = data.n_observations

        def fun(
            u: np.ndarray, spine: float = spine, ages: np.ndarray = ages, n: int = n
        ) -> tuple[float, np.ndarray]:
            lam, mu = np.exp(u)
            try:
                value, grad = _explicit_value_and_grad(
                    lam, mu, horizon=horizon, spine=spine, ages=ages, n=n
                )
            except FloatingPointError:
                return 1e20, np.zeros(2)
            if not np.isfinite(value):
                return 1e20, np.zeros(2)
            return -value, -grad * np.exp(u)

        start = time.perf_counter()
        with np.errstate(all="ignore"):
            scipy.optimize.minimize(
                fun,
                np.log([0.8, 0.4]),
                jac=True,
                method="L-BFGS-B",
                options={"gtol": 1e-8},
            )
        explicit_times.append(time.perf_counter() - start)

    rows.append(
        {
            "fit_cold_s": cold,
            "fit_warm_same_bucket_median_s": statistics.median(warm),
            "explicit_lbfgs_median_s": statistics.median(explicit_times),
        }
    )
    return rows


def _loglik_with_nodes(
    data: AgeData,
    birth: PiecewiseLinearRate,
    death: PiecewiseLinearRate,
    n_nodes: int,
) -> Callable[[], jax.Array]:
    """Return a nullary function evaluating the log-likelihood with ``n_nodes``."""
    return lambda: log_likelihood_rates(data, birth, death, n_nodes=n_nodes)


def bench_quadrature(repeats: int) -> list[dict[str, Any]]:
    """Gauss-Legendre accuracy against a 512-node reference."""
    rows: list[dict[str, Any]] = []
    times = jnp.array([3.0, 6.0])
    scenarios = {
        "moderate": ([1.0, 1.6, 0.8], [0.4, 0.7, 0.2]),
        "steep": ([4.0, 8.0, 2.0], [1.0, 6.0, 0.5]),
        "extreme": ([40.0, 80.0, 20.0], [10.0, 60.0, 5.0]),
    }
    for label, (b_knots, d_knots) in scenarios.items():
        birth = PiecewiseLinearRate.continuous(jnp.array(b_knots), times)
        death = PiecewiseLinearRate.continuous(jnp.array(d_knots), times)
        mild = PiecewiseLinearRate.continuous(jnp.array([1.0, 1.6, 0.8]), times)
        mild_d = PiecewiseLinearRate.continuous(jnp.array([0.4, 0.7, 0.2]), times)
        data = _simulate(mild, mild_d, seed=5, min_n=60)
        reference = float(log_likelihood_rates(data, birth, death, n_nodes=512))
        for nodes in (8, 16, 32, 64, 128):
            fn = jax.jit(_loglik_with_nodes(data, birth, death, nodes))
            _, run_ms = _compile_and_run(fn, repeats=repeats)
            value = float(fn())
            rows.append(
                {
                    "scenario": label,
                    "n_nodes": nodes,
                    "abs_error_loglik": abs(value - reference)
                    if np.isfinite(value) and np.isfinite(reference)
                    else float("nan"),
                    "run_ms": run_ms,
                }
            )
    return rows


def _root_find(problem: Any, x0: np.ndarray, tol: float) -> dict[str, Any]:
    """Newton/hybrid root of the score using the exact Hessian as Jacobian."""
    evals = 0

    def score(x: np.ndarray) -> np.ndarray:
        nonlocal evals
        evals += 1
        return np.asarray(problem.value_and_grad(jnp.asarray(x))[1])

    def jac(x: np.ndarray) -> np.ndarray:
        return np.asarray(problem.hessian(jnp.asarray(x)))

    start = time.perf_counter()
    result = scipy.optimize.root(score, x0, jac=jac, method="hybr", tol=tol)
    elapsed = time.perf_counter() - start
    x = np.asarray(result.x)
    value = float(problem.value_and_grad(jnp.asarray(x))[0])
    hess = np.asarray(problem.hessian(jnp.asarray(x)))
    return {
        "ll": -value,
        "grad_norm": float(np.linalg.norm(score(x))),
        "is_max": bool(np.all(np.linalg.eigvalsh(0.5 * (hess + hess.T)) > 0)),
        "evals": evals,
        "seconds": elapsed,
    }


def bench_optimisers(seeds: int) -> list[dict[str, Any]]:
    """Trust-region Newton vs L-BFGS-B vs Newton root-finding on the score."""
    rows: list[dict[str, Any]] = []
    scenarios: dict[str, tuple[BirthDeathModel, Any, Any, Constraints | None]] = {}
    one = BirthDeathModel.piecewise_constant([8.0])
    scenarios["constant"] = (one, *_constant_pair(0.9, 0.4, 8.0), None)
    scenarios["constant, near boundary (mu=0.02)"] = (
        one,
        *_constant_pair(0.9, 0.02, 8.0),
        None,
    )
    two = BirthDeathModel.piecewise_constant([4.0, 8.0])
    times2 = jnp.array([4.0, 8.0])
    scenarios["two epochs"] = (
        two,
        PiecewiseConstantRate(jnp.array([0.9, 0.7]), times2),
        PiecewiseConstantRate(jnp.array([0.3, 0.5]), times2),
        None,
    )
    lin = BirthDeathModel.piecewise_linear([4.0, 8.0])
    scenarios["linear birth, const death"] = (
        lin,
        PiecewiseLinearRate.continuous(jnp.array([0.8, 1.4, 0.8]), times2),
        PiecewiseLinearRate.continuous(jnp.array([0.4, 0.4, 0.4]), times2),
        Constraints(equal=[["death.knot0", "death.knot1", "death.knot2"]]),
    )
    for label, (model, birth, death, constraints) in scenarios.items():
        stats: dict[str, list[Any]] = {k: [] for k in ("newton", "lbfgs", "root")}
        for seed in range(seeds):
            data = _simulate(birth, death, seed=100 + seed, min_n=60)
            for method, key in (("trust-exact", "newton"), ("L-BFGS-B", "lbfgs")):
                # First call warms the compiled objective; the second is timed.
                fit(
                    data,
                    model,
                    constraints=constraints,
                    method=method,
                    compute_covariance=False,
                )
                start = time.perf_counter()
                result = fit(
                    data,
                    model,
                    constraints=constraints,
                    method=method,
                    compute_covariance=False,
                )
                stats[key].append(
                    {
                        "converged": result.converged,
                        "interior": result.at_interior_optimum,
                        "ll": result.log_likelihood,
                        "evals": result.n_function_evals,
                        "seconds": time.perf_counter() - start,
                    }
                )
            compiled = (constraints or Constraints()).compile(model.params)
            problem = _prepare_problem(
                data,
                model,
                compiled,
                condition_on_survival=True,
                n_nodes=32,
                pad_to_bucket=True,
                hessian_mode="fwd-fwd",
                tol=1e-6,
            )
            moments = fitting_module.moment_initial_rates(data) or (0.8, 0.4)
            x0 = np.asarray(compiled.reduce(model.constant_natural(*moments)))
            reference = stats["newton"][-1]["ll"]
            root = _root_find(problem, x0, 1e-10)
            root["success"] = bool(
                root["grad_norm"] < 1e-4 * max(1, data.n_observations)
                and root["is_max"]
                and root["ll"] > reference - 1e-4
            )
            stats["root"].append(root)
        for key, entries in stats.items():
            ll_best = [
                max(s["ll"] for s in group)
                for group in zip(*(stats[k] for k in stats), strict=True)
            ]
            ok = [
                (e.get("converged", e.get("success", False))) and e["ll"] > best - 1e-4
                for e, best in zip(entries, ll_best, strict=True)
            ]
            gaps = [best - e["ll"] for e, best in zip(entries, ll_best, strict=True)]
            rows.append(
                {
                    "scenario": label,
                    "method": key,
                    "reached_best_optimum": f"{sum(ok)}/{len(ok)}",
                    "median_ll_gap": statistics.median(gaps),
                    "max_ll_gap": max(gaps),
                    "median_evals": statistics.median(e["evals"] for e in entries),
                    "median_seconds": statistics.median(e["seconds"] for e in entries),
                }
            )
    return rows


# ---------------------------------------------------------------------------


def main() -> None:
    """Run the requested benchmark sections and print Markdown tables."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--quick", action="store_true", help="fewer repeats/seeds")
    parser.add_argument("--out", type=Path, default=Path("_output/bench_fitting.json"))
    args = parser.parse_args()
    repeats = 5 if args.quick else 30
    seeds = 3 if args.quick else 10

    results: dict[str, Any] = {"backend": jax.default_backend()}
    if "derivatives" in args.only:
        results["derivatives"] = bench_derivatives(repeats)
        _table(results["derivatives"], "Derivative variants")
    if "explicit" in args.only:
        results["explicit"] = bench_explicit(repeats, args.quick)
        _table(results["explicit"], "Explicit (Mathematica) scores vs autodiff")
        results["explicit_fit"] = bench_explicit_fit(4 if args.quick else 10)
        _table(results["explicit_fit"], "Single-epoch fit wall time")
    if "quadrature" in args.only:
        results["quadrature"] = bench_quadrature(repeats)
        _table(results["quadrature"], "Gauss-Legendre node count")
    if "optimisers" in args.only:
        results["optimisers"] = bench_optimisers(seeds)
        _table(results["optimisers"], "Optimisers")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
