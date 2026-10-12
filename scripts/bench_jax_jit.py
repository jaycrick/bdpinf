"""Post-JIT JAX benchmark for the birth-death process likelihood.

Measures the speedup from explicit jax.jit compilation of the NLL function
that the optimizer uses, across the same (epochs × n_ages) grid as the
Phase 2a eager benchmark stored in _output/jax_torch_benchmarks.json.

Results are keyed by `jax.default_backend()` ("cpu" or "gpu") and merged
into the output file rather than overwritten, so a CPU run and a GPU run
accumulate side by side and a CPU-vs-GPU speedup can be computed once both
are present. Backend is selected by the launching environment, e.g.:

Usage
-----
    uv run python scripts/bench_jax_jit.py
    JAX_PLATFORMS=cuda uv run python scripts/bench_jax_jit.py
    uv run python scripts/bench_jax_jit.py --n-repeats 20
"""

import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

# Must be set before `import jax` — jax reads JAX_PLATFORMS into its config at
# import time, so setting this any later (e.g. via `import bdpinf`) is
# too late and lets JAX auto-select GPU whenever CUDA is visible.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import typer

import bdpinf  # noqa: F401 — triggers jax_enable_x64  # pyright: ignore[reportUnusedImport]
from bdpinf import get_project_root
from bdpinf.inference import fit_birth_death_processes
from bdpinf.models.math_nested import (
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.models.math_populations import calculate_populations_log_lik

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Benchmark grid — matches Phase 2a
# ---------------------------------------------------------------------------

EPOCH_CONFIGS = [1, 2, 6]
N_AGES_CONFIGS = [10, 100, 1000]
RATE = 0.05  # typical manuscript birth/death rate (1/yr)


def _make_params(n_epochs: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (birth_rates, death_rates, times) for n_epochs."""
    birth_rates = jnp.full(n_epochs, RATE, dtype=jnp.float64)
    death_rates = jnp.full(n_epochs, RATE * 0.8, dtype=jnp.float64)
    times = jnp.array(
        [float(200 * (k + 1) / n_epochs) for k in range(n_epochs)], dtype=jnp.float64
    )
    return birth_rates, death_rates, times


def _make_ages(n_ages: int, t_max: float) -> tuple[jax.Array, jax.Array]:
    """Return (spine_ages, non_spine_ages) totalling n_ages."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.0, t_max, size=n_ages)
    raw_sorted = np.sort(raw)
    spine = jnp.array(raw_sorted[-1:], dtype=jnp.float64)
    non_spine = jnp.array(raw_sorted[:-1], dtype=jnp.float64)
    return spine, non_spine


def _make_ns(n_ages: int) -> jax.Array:
    return jnp.array([float(n_ages)], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Inner NLL function — pure JAX, safe to jit
# ---------------------------------------------------------------------------


def _make_nll_fn(
    non_spine_ages: jax.Array,
    spine_ages: jax.Array,
    ns: jax.Array,
    times: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    """Return a closure nll(log_params) → scalar.

    Mirrors what _compute_nll does inside fit_birth_death_processes:
    log_params = [log_lambda_1, ..., log_lambda_k, log_mu_1, ..., log_mu_k].
    """
    n_epochs = len(times)

    def nll(log_params: jax.Array) -> jax.Array:
        birth_rates = jnp.exp(log_params[:n_epochs])
        death_rates = jnp.exp(log_params[n_epochs:])
        pop_ll = calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ns=ns,
            initial_population=1,
            unbias=True,
            _skip_normalization=True,
        )
        spine_ll = spine_log_density(
            nested_terms(birth_rates, death_rates, times), spine_ages
        ).sum()
        non_spine_ll = non_spine_log_density(
            nested_terms(birth_rates, death_rates, times), non_spine_ages
        ).sum()
        return -(pop_ll + spine_ll + non_spine_ll)

    return nll


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_call(
    fn: Callable[..., Any], arg: object, n_repeats: int
) -> tuple[float, float]:
    """Return (cold_s, warm_mean_s) for fn(arg).

    Cold time includes JAX dispatch/JIT compilation on first call.
    """
    t0 = time.perf_counter()
    result = fn(arg)
    jax.block_until_ready(result)
    cold = time.perf_counter() - t0

    times_list = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        r = fn(arg)
        jax.block_until_ready(r)
        times_list.append(time.perf_counter() - t0)

    return cold, float(np.mean(times_list))


def _time_val_and_grad(
    fn: Callable[..., Any], arg: jax.Array, n_repeats: int
) -> tuple[float, float]:
    """Like _time_call but for functions returning (value, grad) tuples."""
    t0 = time.perf_counter()
    val, grad = fn(arg)
    jax.block_until_ready(val)
    jax.block_until_ready(grad)
    cold = time.perf_counter() - t0

    times_list = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        v, g = fn(arg)
        jax.block_until_ready(v)
        jax.block_until_ready(g)
        times_list.append(time.perf_counter() - t0)

    return cold, float(np.mean(times_list))


# ---------------------------------------------------------------------------
# Backend-keyed merge + cross-backend speedup summary
# ---------------------------------------------------------------------------


def _merge_backend_block(
    out: Path, section: str, backend: str, block: dict[str, Any]
) -> dict[str, Any]:
    """Load `out`, set payload[section][backend] = block, return the full payload."""
    payload: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
    payload.setdefault(section, {})
    payload[section][backend] = block
    return payload


def _speedup_summary(
    payload_section: dict[str, Any], row_key: tuple[str, ...], warm_metric: str
) -> list[dict[str, Any]] | None:
    """Pairwise cpu-vs-gpu speedup on `warm_metric`, matched by `row_key` fields.

    Returns None if either backend is missing from `payload_section`.
    """
    if "cpu" not in payload_section or "gpu" not in payload_section:
        return None

    cpu_rows = {
        tuple(r[k] for k in row_key): r for r in payload_section["cpu"]["results"]
    }
    gpu_rows = {
        tuple(r[k] for k in row_key): r for r in payload_section["gpu"]["results"]
    }

    summary = []
    for key in sorted(cpu_rows.keys() & gpu_rows.keys()):
        cpu_t = cpu_rows[key][warm_metric]
        gpu_t = gpu_rows[key][warm_metric]
        row = dict(zip(row_key, key, strict=True))
        row["cpu_" + warm_metric] = cpu_t
        row["gpu_" + warm_metric] = gpu_t
        row["gpu_speedup"] = round(cpu_t / gpu_t, 2) if gpu_t > 0 else None
        summary.append(row)
    return summary


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


N_REPEATS_OPT = typer.Option(10, help="Warm repeats per measurement.")
OUTPUT_OPT: Path | None = typer.Option(None, help="Output JSON path.")


@app.command()
def main(
    n_repeats: int = N_REPEATS_OPT,
    output: Path = OUTPUT_OPT,
) -> None:
    """Run post-JIT JAX benchmark and save results."""
    default_out = get_project_root() / "_output" / "jax_jit_benchmarks.json"  # pyright: ignore[reportUnusedVariable]
    out: Path = output if output is not None else default_out  # pyright: ignore[reportUnnecessaryComparison]

    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    typer.echo(f"JAX backend: {backend}, devices: {devices}")
    typer.echo(f"n_repeats={n_repeats}, output={out}")

    results = []

    for n_epochs in EPOCH_CONFIGS:
        birth_rates, death_rates, times = _make_params(n_epochs)
        t_max = float(times[-1])
        log_params_init = jnp.concatenate([jnp.log(birth_rates), jnp.log(death_rates)])

        for n_ages in N_AGES_CONFIGS:
            typer.echo(f"  epochs={n_epochs}, n_ages={n_ages} ...", nl=False)
            spine_ages, non_spine_ages = _make_ages(n_ages, t_max)
            ns = _make_ns(n_ages)

            nll_fn = _make_nll_fn(non_spine_ages, spine_ages, ns, times)
            vag_fn = jax.value_and_grad(nll_fn)
            jit_nll_fn = jax.jit(nll_fn)
            jit_vag_fn = jax.jit(vag_fn)

            # Eager forward
            eager_ll_cold, eager_ll_warm = _time_call(
                nll_fn, log_params_init, n_repeats
            )

            # Eager value_and_grad
            eager_vag_cold, eager_vag_warm = _time_val_and_grad(
                vag_fn, log_params_init, n_repeats
            )

            # JIT forward
            jit_ll_cold, jit_ll_warm = _time_call(
                jit_nll_fn, log_params_init, n_repeats
            )

            # JIT value_and_grad
            jit_vag_cold, jit_vag_warm = _time_val_and_grad(
                jit_vag_fn, log_params_init, n_repeats
            )

            row = {
                "epochs": n_epochs,
                "n_ages": n_ages,
                "eager_ll_cold_s": round(eager_ll_cold, 6),
                "eager_ll_warm_s": round(eager_ll_warm, 6),
                "eager_vag_cold_s": round(eager_vag_cold, 6),
                "eager_vag_warm_s": round(eager_vag_warm, 6),
                "jit_ll_cold_s": round(jit_ll_cold, 6),
                "jit_ll_warm_s": round(jit_ll_warm, 6),
                "jit_vag_cold_s": round(jit_vag_cold, 6),
                "jit_vag_warm_s": round(jit_vag_warm, 6),
                "speedup_ll": round(eager_ll_warm / jit_ll_warm, 2)
                if jit_ll_warm > 0
                else None,
                "speedup_vag": round(eager_vag_warm / jit_vag_warm, 2)
                if jit_vag_warm > 0
                else None,
            }
            results.append(row)

            e_ll = eager_ll_warm * 1000
            j_ll = jit_ll_warm * 1000
            e_vg = eager_vag_warm * 1000
            j_vg = jit_vag_warm * 1000
            typer.echo(
                f" ll: eager={e_ll:.1f}ms jit={j_ll:.1f}ms ({row['speedup_ll']}×)"
                f" | vag: eager={e_vg:.1f}ms jit={j_vg:.1f}ms ({row['speedup_vag']}×)"
            )

    block = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_repeats": n_repeats,
        "platform": {
            "jax_version": jax.__version__,
            "jax_devices": devices,
        },
        "results": results,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _merge_backend_block(out, "results", backend, block)
    speedup = _speedup_summary(
        payload["results"], ("epochs", "n_ages"), "jit_ll_warm_s"
    )
    if speedup is not None:
        payload["results"]["_speedup"] = speedup
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(f"\nResults written to {out} (backend={backend})")

    # Print summary table
    typer.echo("\n### Likelihood (forward): eager vs JIT warm (ms)")
    typer.echo(f"{'epochs':>6} {'n_ages':>7} {'eager':>10} {'JIT':>10} {'speedup':>8}")
    for r in results:
        typer.echo(
            f"{r['epochs']:>6} {r['n_ages']:>7}"
            f" {r['eager_ll_warm_s'] * 1000:>10.2f}"
            f" {r['jit_ll_warm_s'] * 1000:>10.2f}"
            f" {r['speedup_ll']:>8}×"
        )

    typer.echo("\n### value_and_grad: eager vs JIT warm (ms)")
    typer.echo(f"{'epochs':>6} {'n_ages':>7} {'eager':>10} {'JIT':>10} {'speedup':>8}")
    for r in results:
        typer.echo(
            f"{r['epochs']:>6} {r['n_ages']:>7}"
            f" {r['eager_vag_warm_s'] * 1000:>10.2f}"
            f" {r['jit_vag_warm_s'] * 1000:>10.2f}"
            f" {r['speedup_vag']:>8}×"
        )

    if speedup is not None:
        typer.echo("\n### CPU vs GPU (JIT forward warm, ms)")
        typer.echo(
            f"{'epochs':>6} {'n_ages':>7} {'cpu':>10} {'gpu':>10} {'gpu_speedup':>11}"
        )
        for r in speedup:
            typer.echo(
                f"{r['epochs']:>6} {r['n_ages']:>7}"
                f" {r['cpu_jit_ll_warm_s'] * 1000:>10.3f}"
                f" {r['gpu_jit_ll_warm_s'] * 1000:>10.3f}"
                f" {r['gpu_speedup']:>10}×"
            )


N_REPEATS_FITTER_OPT = typer.Option(5, help="Warm repeats per fitter cell.")
OUTPUT_FITTER_OPT: Path | None = typer.Option(None, help="Output JSON path.")


@app.command()
def bench_fitter(
    n_repeats: int = N_REPEATS_FITTER_OPT,
    output: Path = OUTPUT_FITTER_OPT,
) -> None:
    """Benchmark fit_birth_death_processes end-to-end (JIT-compiled optimizer).

    Each fit call builds a fresh jax.jit(value_and_grad(...)) closure.  The
    first optimizer step within a fit triggers XLA compilation; subsequent
    steps hit the JIT cache.  JAX also caches across fit calls that share
    the same input shapes, so the second fit call for a given (epochs, n_ages)
    cell avoids recompilation.

    Uses LBFGS only (no Adam) and skips Hessian estimation for a clean
    measurement of the optimizer loop.
    """
    default_out = get_project_root() / "_output" / "jax_jit_benchmarks.json"  # pyright: ignore[reportUnusedVariable]
    out: Path = output if output is not None else default_out  # pyright: ignore[reportUnnecessaryComparison]

    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    typer.echo(f"JAX backend: {backend}, devices: {devices}")
    typer.echo(f"n_repeats={n_repeats}, output={out}")

    fitter_results = []

    for n_epochs in EPOCH_CONFIGS:
        _, _, times_jax = _make_params(n_epochs)
        t_max = float(times_jax[-1])
        times_np = np.array(times_jax)

        for n_ages in N_AGES_CONFIGS:
            typer.echo(f"  epochs={n_epochs}, n_ages={n_ages} ...", nl=False)
            spine_ages_jax, non_spine_ages_jax = _make_ages(n_ages, t_max)
            # Combine into a single flat age list (fitter auto-detects spine)
            all_ages = np.concatenate(
                [np.array(non_spine_ages_jax), np.array(spine_ages_jax)]
            )

            _fit_kwargs: dict[str, Any] = {
                "optimizers": ["LBFGS"],
                "estimate_uncertainty": False,
                "unbias": True,
            }

            # Cold: first call — includes JIT compilation on the first optimizer step
            t0 = time.perf_counter()
            cold_result = fit_birth_death_processes([all_ages], times_np, **_fit_kwargs)
            cold_s = time.perf_counter() - t0
            _attempts = cold_result.get("optimization_result", {}).get("attempts", [])
            cold_iters: int | None = (
                _attempts[0].get("iterations") if _attempts else None
            )

            # Warm: n_repeats subsequent calls — JIT-compiled XLA kernel reused
            warm_times: list[float] = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                fit_birth_death_processes([all_ages], times_np, **_fit_kwargs)
                warm_times.append(time.perf_counter() - t0)

            warm_s = float(np.mean(warm_times))
            typer.echo(
                f" cold={cold_s * 1000:.0f}ms warm={warm_s * 1000:.0f}ms"
                f" (iters={cold_iters})"
            )

            fitter_results.append(
                {
                    "epochs": n_epochs,
                    "n_ages": n_ages,
                    "fitter_cold_s": round(cold_s, 3),
                    "fitter_warm_s": round(warm_s, 3),
                    "lbfgs_iters": cold_iters,
                }
            )

    block = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_repeats": n_repeats,
        "platform": {"jax_version": jax.__version__, "jax_devices": devices},
        "note": (
            "LBFGS only, estimate_uncertainty=False. "
            "Cold includes JIT compilation on first optimizer step; "
            "warm reuses JAX compilation cache (same input shapes)."
        ),
        "results": fitter_results,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _merge_backend_block(out, "fitter_results", backend, block)
    speedup = _speedup_summary(
        payload["fitter_results"], ("epochs", "n_ages"), "fitter_warm_s"
    )
    if speedup is not None:
        payload["fitter_results"]["_speedup"] = speedup
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(f"\nFitter results written to {out} (backend={backend})")

    typer.echo("\n### End-to-end fitter wall-clock time (LBFGS, ms)")
    typer.echo(f"{'epochs':>6} {'n_ages':>7} {'cold':>10} {'warm':>10} {'iters':>6}")
    for r in fitter_results:
        typer.echo(
            f"{r['epochs']:>6} {r['n_ages']:>7}"
            f" {r['fitter_cold_s'] * 1000:>10.0f}"
            f" {r['fitter_warm_s'] * 1000:>10.0f}"
            f" {r['lbfgs_iters']!s:>6}"
        )

    if speedup is not None:
        typer.echo("\n### CPU vs GPU (fitter warm, ms)")
        typer.echo(
            f"{'epochs':>6} {'n_ages':>7} {'cpu':>10} {'gpu':>10} {'gpu_speedup':>11}"
        )
        for r in speedup:
            typer.echo(
                f"{r['epochs']:>6} {r['n_ages']:>7}"
                f" {r['cpu_fitter_warm_s'] * 1000:>10.1f}"
                f" {r['gpu_fitter_warm_s'] * 1000:>10.1f}"
                f" {r['gpu_speedup']:>10}×"
            )


if __name__ == "__main__":
    app()
