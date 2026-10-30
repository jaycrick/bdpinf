r"""Benchmark the ``fit_knots`` search methods with several free knots (NTVMR).

The truth is one fit of continuous piecewise-linear birth and death rates to the
NTVMR continuous-text date ranges with every epoch boundary fixed at its current
date (``scripts/prepare_ntvmr_benchmark.py`` writes it to
``scripts/data/ntvmr_benchmark.json``, so nt-manuscripts is not needed here).
Each replicate is a parametric bootstrap sample from that truth: a non-extinct
population whose ages become date ranges of widths resampled from the real ranges.
Every method then fits all five interior knots and the rates, starting from the
current dates, through :func:`bdpinf.inference.fit_knots` (one method per call, so
the timings are independent).

The summary reports, per method, the time, how often it reaches the best
log-likelihood found for the replicate, the accuracy of the knots and rates, and
the same hit rates for every set of two and three methods (a set hits when at
least one member does).

Usage
-----
    uv run python scripts/bench_ntvmr_knots.py --truth --starts 1
    uv run python scripts/bench_ntvmr_knots.py --replicates 1 --workers 5
    uv run python scripts/bench_ntvmr_knots.py --real
    uv run python scripts/bench_ntvmr_knots.py --shard 3/200 --workers 5
    uv run python scripts/bench_ntvmr_knots.py --merge

The Hessian of this model takes minutes to compile, so a persistent JAX
compilation cache is enabled (``_output/jax_cache``, override with
``JAX_COMPILATION_CACHE_DIR``) that every process and array task shares.
With ``--workers`` the methods of a replicate run in parallel single-thread
processes.
"""

import argparse
import concurrent.futures
import itertools
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(Path("_output/jax_cache").resolve())
)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import jax.numpy as jnp
import numpy as np
from bench_fitting import _table

from bdpinf.inference import AgeData, BirthDeathModel, Constraints, FreeKnots, fit
from bdpinf.inference.knots import fit_knots
from bdpinf.inference.likelihood import log_likelihood_rates
from bdpinf.simulation import simulate_ages_from_rates

METHODS = ("brent", "de-profile", "grid-newton", "nelder-mead", "profile")
HIT_TOLS = (1e-3, 0.1, 1.0)
N_CANDIDATES = 64  # a 12-point scan is too sparse in five dimensions
MAX_SIZE_FACTOR = 5  # memory guard: drop populations larger than this times n
DATA_PATH = Path(__file__).parent / "data" / "ntvmr_benchmark.json"
SIZE_EDGES = (100, 500, 2000)


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------


def load_spec(
    path: Path = DATA_PATH, out_dir: Path = Path("_output")
) -> dict[str, Any]:
    """Read the prepared data; the truth may come from ``--truth`` output instead."""
    spec = json.loads(path.read_text())
    if not spec.get("truth"):
        fitted = out_dir / "ntvmr_truth.json"
        if not fitted.exists():
            msg = f"No truth in {path}; run --truth first (writes {fitted})"
            raise FileNotFoundError(msg)
        spec.update(json.loads(fitted.read_text()))
    return spec


def build_model(spec: dict[str, Any]) -> BirthDeathModel:
    """Piecewise-linear rates whose interior knots are free within their bounds."""
    knots = FreeKnots(
        len(spec["knot_bounds"]),
        float(spec["horizon"]),
        bounds=[tuple(b) for b in spec["knot_bounds"]],
        initial=spec["knot_initial"],
    )
    return BirthDeathModel.piecewise_linear(knots, continuous=True)


def free_constraints(spec: dict[str, Any]) -> Constraints:
    """Birth is zero at the final knot; everything else, knots included, is free."""
    return Constraints(fixed={spec["zero_birth_name"]: 0.0})


def fixed_constraints(spec: dict[str, Any]) -> Constraints:
    """Return the truth fit's constraints: the interior knots are fixed too."""
    knots = dict(zip(spec["knot_names"], spec["knot_initial"], strict=True))
    return Constraints(fixed={spec["zero_birth_name"]: 0.0, **knots})


def clip_years(
    lower: np.ndarray, upper: np.ndarray, start: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ranges ``[lower, upper)`` in ``[start, end]``, at least a year wide."""
    upper = np.minimum(np.asarray(upper, dtype=np.int64), end)
    lower = np.minimum(np.asarray(lower, dtype=np.int64), upper - 1)
    lower = np.maximum(lower, start)
    upper = np.maximum(upper, lower + 1)
    return lower, upper


def real_data(spec: dict[str, Any]) -> AgeData:
    """Return the real date ranges as ``AgeData`` (ages counted back from the end)."""
    ranges = np.asarray(spec["ranges"], dtype=np.float64)
    counts = ranges[:, 2].astype(int)
    end = spec["end_year"]
    return AgeData.from_ranges(
        np.repeat(end - ranges[:, 1], counts),
        np.repeat(end - ranges[:, 0], counts),
        float(spec["horizon"]),
    )


def simulate_replicate(
    spec: dict[str, Any], model: BirthDeathModel, rng: np.random.Generator
) -> AgeData:
    """One non-extinct bootstrap population with each age turned into a range."""
    birth, death = model.rates(jnp.asarray(spec["truth"]))
    ranges = np.asarray(spec["ranges"], dtype=np.float64)
    widths = ranges[:, 1] - ranges[:, 0]
    probs = ranges[:, 2] / ranges[:, 2].sum()
    cap = MAX_SIZE_FACTOR * int(ranges[:, 2].sum())
    for _ in range(200):
        sample = simulate_ages_from_rates(birth, death, rng, ensure_non_extinct=True)
        if sample.population <= cap:
            break
    else:
        msg = "No simulated population within the size cap"
        raise RuntimeError(msg)
    end, start = spec["end_year"], spec["start_year"]
    ages = np.asarray(sample.all_ages, dtype=np.float64)
    year = np.floor(end - ages).astype(np.int64)
    width = rng.choice(widths, size=len(year), p=probs).astype(np.int64)
    offset = np.floor(rng.uniform(size=len(year)) * width).astype(np.int64)
    lower, upper = clip_years(year - offset, year - offset + width, start, end)
    return AgeData.from_ranges(end - upper, end - lower, float(spec["horizon"]))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def run_method(
    spec: dict[str, Any],
    model: BirthDeathModel,
    data: AgeData,
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Fit rates and knots with one search method; failures are recorded, not raised."""
    began = time.perf_counter()
    try:
        result = fit_knots(
            data,
            model,
            constraints=free_constraints(spec),
            methods=method,
            n_candidates=N_CANDIDATES,
            seed=seed,
        )
    except (ValueError, RuntimeError) as exc:
        return {
            "ok": False,
            "error": str(exc)[:200],
            "log_likelihood": float("-inf"),
            "seconds": time.perf_counter() - began,
        }
    seconds = time.perf_counter() - began
    se = result.fit.standard_errors or {}
    return {
        "ok": True,
        "log_likelihood": float(result.log_likelihood),
        "natural": [float(result.fit.params[n]) for n in model.names],
        "se": [float(se.get(n, float("nan"))) for n in model.names],
        "converged": bool(result.fit.converged),
        "seconds": seconds,
        "n_fits": int(result.method_results[method].n_fits),
    }


def _fit_job(job: tuple[dict[str, Any], int, int, str]) -> tuple[str, dict[str, Any]]:
    """Fit one method on replicate ``rep`` (rebuilt from its seed) in a worker."""
    spec, rep, seed, method = job
    model = build_model(spec)
    data = simulate_replicate(spec, model, np.random.default_rng([seed, rep]))
    return method, run_method(spec, model, data, method, seed + rep)


def run_replicates(
    spec: dict[str, Any],
    replicates: list[int],
    seed: int,
    methods: tuple[str, ...],
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Simulate and fit each replicate with every method.

    With ``workers > 1`` the methods of a replicate run in parallel processes.
    """
    model = build_model(spec)
    truth = jnp.asarray(spec["truth"])
    pool = None
    if workers > 1:
        # Single-thread workers: the parallelism is across methods.
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        )
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
    records = []
    for rep in replicates:
        data = simulate_replicate(spec, model, np.random.default_rng([seed, rep]))
        birth, death = model.rates(truth)
        record: dict[str, Any] = {
            "replicate": rep,
            "n": data.n_observations,
            "distinct_ranges": len(data.non_spine_ages),
            "truth_log_likelihood": float(log_likelihood_rates(data, birth, death)),
            "methods": {},
        }
        if pool is None:
            for method in methods:
                got = run_method(spec, model, data, method, seed + rep)
                record["methods"][method] = got
                print(
                    f"  rep {rep} {method}: {got['seconds']:.1f}s "
                    f"ll={got['log_likelihood']:.3f}",
                    flush=True,
                )
        else:
            futures = [
                pool.submit(_fit_job, (spec, rep, seed, method)) for method in methods
            ]
            for future in concurrent.futures.as_completed(futures):
                method, got = future.result()
                record["methods"][method] = got
                print(
                    f"  rep {rep} {method}: {got['seconds']:.1f}s "
                    f"ll={got['log_likelihood']:.3f}",
                    flush=True,
                )
            record["methods"] = {m: record["methods"][m] for m in methods}
        records.append(record)
        print(f"replicate {rep} done (n = {data.n_observations})", flush=True)
    if pool is not None:
        pool.shutdown()
    return records


def fit_truth(spec: dict[str, Any], starts: int, seed: int = 209) -> dict[str, Any]:
    """Fit the real ranges with every knot fixed, printing the time and iterations."""
    model = build_model(spec)
    data = real_data(spec)
    began = time.perf_counter()
    result = fit(
        data,
        model,
        constraints=fixed_constraints(spec),
        n_starts=starts,
        perturb_std=0.1,
        seed=seed,
    )
    seconds = time.perf_counter() - began
    print(
        f"truth fit: {starts} start(s), {seconds:.1f}s, "
        f"ll={result.log_likelihood:.4f}, iterations={result.n_iter}, "
        f"evaluations={result.n_function_evals}, success={result.success}",
        flush=True,
    )
    for k, start in enumerate(result.starts):
        print(f"  start {k}: ll={start.log_likelihood:.4f}, iterations={start.n_iter}")
    return {
        "names": list(model.names),
        "truth": [float(result.params[n]) for n in model.names],
        "truth_log_likelihood": float(result.log_likelihood),
        "truth_fit_seconds": seconds,
        "truth_fit_starts": starts,
    }


def run_real(
    spec: dict[str, Any], methods: tuple[str, ...], seed: int
) -> list[dict[str, Any]]:
    """Fit the real data with each method (knots free), after the fixed-knot truth."""
    model = build_model(spec)
    data = real_data(spec)
    names = spec["knot_names"]
    index = [list(model.names).index(n) for n in names]
    rows = [
        {
            "method": "fixed knots (truth)",
            "log_likelihood": spec["truth_log_likelihood"],
            "seconds": spec.get("truth_fit_seconds", float("nan")),
            **{
                n: spec["truth"][i] + spec["start_year"]
                for n, i in zip(names, index, strict=True)
            },
        }
    ]
    for method in methods:
        got = run_method(spec, model, data, method, seed)
        rows.append(
            {
                "method": method,
                "log_likelihood": got["log_likelihood"],
                "seconds": got["seconds"],
                **{
                    n: (got["natural"][i] + spec["start_year"] if got["ok"] else np.nan)
                    for n, i in zip(names, index, strict=True)
                },
            }
        )
        print(f"  real {method}: {got['seconds']:.1f}s", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _gaps(records: list[dict[str, Any]], methods: tuple[str, ...]) -> np.ndarray:
    """Return log-likelihood minus the best per replicate, (replicates, methods)."""
    ll = np.array(
        [[r["methods"][m]["log_likelihood"] for m in methods] for r in records]
    )
    ll = np.nan_to_num(ll, nan=-np.inf)
    truth = np.array([r["truth_log_likelihood"] for r in records])
    best = np.maximum(ll.max(axis=1), truth)
    return ll - best[:, None]


def summarise(
    records: list[dict[str, Any]], spec: dict[str, Any], methods: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    """Speed, hit rates, accuracy, size effects and the best method sets."""
    model = build_model(spec)
    names = list(model.names)
    truth = np.array(spec["truth"])
    knot_idx = [names.index(n) for n in spec["knot_names"]]
    rate_idx = [
        i for i, n in enumerate(names) if n not in spec["knot_names"] and truth[i] > 0.0
    ]
    gaps = _gaps(records, methods)
    sizes = np.array([r["n"] for r in records])
    secs = np.array([[r["methods"][m]["seconds"] for m in methods] for r in records])
    speed, knots, rates, by_size = [], [], [], []
    for j, method in enumerate(methods):
        got = [r["methods"][method] for r in records]
        ok = np.array([g["ok"] for g in got])
        row: dict[str, Any] = {
            "method": method,
            "fitted": f"{int(ok.sum())}/{len(got)}",
            "median_seconds": float(np.median(secs[:, j])),
            "p90_seconds": float(np.percentile(secs[:, j], 90)),
            "total_seconds": float(secs[:, j].sum()),
            "median_fits": float(np.median([g.get("n_fits", np.nan) for g in got])),
        }
        for tol in HIT_TOLS:
            row[f"hit_{tol:g}"] = float(np.mean(gaps[:, j] >= -tol))
        row["median_gap"] = float(np.median(gaps[:, j]))
        row["worst_gap"] = float(np.min(gaps[:, j]))
        speed.append(row)
        if not ok.any():
            continue
        est = np.array([g["natural"] for g in got if g["ok"]])
        se = np.array([g["se"] for g in got if g["ok"]])
        for i in knot_idx:
            err = est[:, i] - truth[i]
            usable = np.isfinite(se[:, i]) & (se[:, i] > 0)
            knots.append(
                {
                    "method": method,
                    "knot": names[i],
                    "true_year": float(truth[i] + spec["start_year"]),
                    "bias": float(np.mean(err)),
                    "median_abs_err": float(np.median(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "wald_coverage95": float(
                        np.mean(np.abs(err[usable]) <= 1.96 * se[usable, i])
                    )
                    if usable.any()
                    else float("nan"),
                }
            )
        rel = np.abs(est[:, rate_idx] - truth[rate_idx]) / truth[rate_idx]
        rates.append(
            {
                "method": method,
                "median_rel_err": float(np.median(rel)),
                "p90_rel_err": float(np.percentile(rel, 90)),
            }
        )
        edges = (0, *SIZE_EDGES, np.inf)
        for lo, hi in itertools.pairwise(edges):
            mask = (sizes >= lo) & (sizes < hi)
            if not mask.any():
                continue
            fitted = [g for g, m in zip(got, mask, strict=True) if m and g["ok"]]
            knot_err = [
                abs(g["natural"][i] - truth[i]) for g in fitted for i in knot_idx
            ]
            by_size.append(
                {
                    "method": method,
                    "n_range": f"{lo}-{hi:g}",
                    "replicates": int(mask.sum()),
                    "hit_0.001": float(np.mean(gaps[mask, j] >= -1e-3)),
                    "median_knot_err": float(np.median(knot_err))
                    if knot_err
                    else float("nan"),
                }
            )
    sets = _method_sets(gaps, secs, methods)
    return {
        f"ntvmr: speed and hit rate ({len(records)} replicates)": speed,
        "ntvmr: knot accuracy (years)": knots,
        "ntvmr: rate accuracy (relative error, non-zero rates)": rates,
        "ntvmr: by sample size": by_size,
        "ntvmr: best single methods, pairs and triples": sets,
        "ntvmr: sample size n (min, q1, median, q3, max)": [
            dict(
                zip(
                    ["min", "q1", "median", "q3", "max"],
                    np.percentile(sizes, [0, 25, 50, 75, 100]).tolist(),
                    strict=True,
                )
            )
        ],
    }


def _method_sets(
    gaps: np.ndarray, secs: np.ndarray, methods: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Every set of one, two and three methods, best first within each size."""
    out = []
    for size in (1, 2, 3):
        rows = []
        for combo in itertools.combinations(range(len(methods)), size):
            top = gaps[:, combo].max(axis=1)
            row: dict[str, Any] = {
                "size": size,
                "methods": " + ".join(methods[j] for j in combo),
            }
            for tol in HIT_TOLS:
                row[f"hit_{tol:g}"] = float(np.mean(top >= -tol))
            row["median_seconds"] = float(np.median(secs[:, combo].sum(axis=1)))
            rows.append(row)
        rows.sort(key=lambda r: (-r["hit_0.001"], -r["hit_0.1"], r["median_seconds"]))
        out += rows
    return out


def report_diagnostics(out_dir: Path, stem: str) -> None:
    """Tabulate the diagnostic outputs: truth fits, one replicate and the real data."""
    truths = [
        {
            "file": path.name,
            "starts": got["truth_fit_starts"],
            "seconds": got["truth_fit_seconds"],
            "log_likelihood": got["truth_log_likelihood"],
        }
        for path in sorted(out_dir.glob("ntvmr_truth*.json"))
        for got in [json.loads(path.read_text())]
    ]
    _print(truths, "diagnostic: truth fits (all knots fixed)")
    reps = []
    for path in sorted(out_dir.glob(f"{stem}_pilot_*.json")):
        for rec in json.loads(path.read_text()):
            for method, got in rec["methods"].items():
                reps.append(
                    {
                        "method": method,
                        "n": rec["n"],
                        "ranges": rec["distinct_ranges"],
                        "seconds": got["seconds"],
                        "rate_fits": got.get("n_fits", ""),
                        "log_likelihood": got["log_likelihood"],
                        "truth_log_likelihood": rec["truth_log_likelihood"],
                    }
                )
    _print(reps, "diagnostic: one bootstrap replicate, each method")
    real = [
        row
        for path in sorted(out_dir.glob(f"{stem}_real_*.json"))
        for row in json.loads(path.read_text())
        if not row["method"].startswith("fixed")
    ]
    _print(real, "diagnostic: real data, each method (knots free)")


def _print(rows: list[dict[str, Any]], title: str) -> None:
    """Print a table whose rows may lack some columns."""
    columns = list(dict.fromkeys(k for row in rows for k in row))
    _table([{c: row.get(c, "") for c in columns} for row in rows], title)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main() -> None:
    """Run replicates (optionally one shard), the real-data fits, or the merge."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument(
        "--shard", default=None, help="i/N: replicates r with r %% N == i"
    )
    parser.add_argument("--merge", action="store_true", help="merge shard files")
    parser.add_argument("--real", action="store_true", help="fit the real data")
    parser.add_argument(
        "--truth", action="store_true", help="fit the truth, knots fixed"
    )
    parser.add_argument("--starts", type=int, default=5, help="starts for --truth")
    parser.add_argument("--workers", type=int, default=1, help="processes over methods")
    parser.add_argument("--tag", default=None, help="suffix for the output file")
    parser.add_argument("--verbose", action="store_true", help="log fit_knots progress")
    parser.add_argument(
        "--report", action="store_true", help="tabulate diagnostic runs"
    )
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("_output"))
    args = parser.parse_args()
    methods = tuple(args.methods or METHODS)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "bench_ntvmr_knots"
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.report:
        report_diagnostics(out_dir, stem)
        return
    if args.truth:
        spec = json.loads(args.data.read_text())
        found = fit_truth(spec, args.starts)
        name = (
            "ntvmr_truth.json" if args.tag is None else f"ntvmr_truth_{args.tag}.json"
        )
        (out_dir / name).write_text(json.dumps(found, indent=1))
        print(f"Wrote {out_dir / name}")
        return
    spec = load_spec(args.data, out_dir)

    if args.real:
        rows = run_real(spec, methods, args.seed)
        _print(rows, "ntvmr real data: knot years by method")
        (out_dir / f"{stem}_real.json").write_text(
            json.dumps(rows, indent=2, default=float)
        )
        return
    if args.merge:
        records: list[dict[str, Any]] = []
        for path in sorted(out_dir.glob(f"{stem}_shard*.json")):
            records += json.loads(path.read_text())
        records.sort(key=lambda r: r["replicate"])
        tables = summarise(records, spec, methods)
        for title, rows in tables.items():
            _print(rows, title)
        real = out_dir / f"{stem}_real.json"
        if real.exists():
            _print(
                json.loads(real.read_text()), "ntvmr real data: knot years by method"
            )
        (out_dir / f"{stem}.json").write_text(
            json.dumps(tables, indent=2, default=float)
        )
        print(f"\nMerged {len(records)} replicates into {out_dir / (stem + '.json')}")
        return

    index, count = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    mine = [r for r in range(args.replicates) if r % count == index]
    records = run_replicates(spec, mine, args.seed, methods, args.workers)
    pilot = (
        f"{stem}_pilot.json" if args.tag is None else f"{stem}_pilot_{args.tag}.json"
    )
    name = f"{stem}_shard{index}.json" if args.shard else pilot
    (out_dir / name).write_text(json.dumps(records, indent=2, default=float))
    if not args.shard and len(methods) > 1:
        for title, rows in summarise(records, spec, methods).items():
            _print(rows, title)
    print(f"\nWrote {out_dir / name}")


if __name__ == "__main__":
    main()
