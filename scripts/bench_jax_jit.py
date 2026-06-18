"""Post-JIT JAX benchmark for the birth-death process likelihood.

Measures the speedup from explicit jax.jit compilation of the NLL function
that the optimizer uses, across the same (epochs × n_ages) grid as the
Phase 2a eager benchmark stored in _output/jax_torch_benchmarks.json.

Usage
-----
    uv run python scripts/bench_jax_jit.py
    uv run python scripts/bench_jax_jit.py --n-repeats 20
"""

# ruff: noqa: E402

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import typer

import inf_bp_texts  # noqa: F401 — triggers jax_enable_x64  # pyright: ignore[reportUnusedImport]
from inf_bp_texts import get_project_root
from inf_bp_texts.models.math_ages import (
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)
from inf_bp_texts.models.math_populations import calculate_populations_log_lik

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
        spine_ll = calculate_spine_ages_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=spine_ages,
            times=times,
            _skip_normalization=True,
        )
        non_spine_ll = calculate_non_spine_ages_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=non_spine_ages,
            times=times,
            _skip_normalization=True,
        )
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
# Main benchmark
# ---------------------------------------------------------------------------


N_REPEATS_OPT = typer.Option(10, help="Warm repeats per measurement.")
OUTPUT_OPT: Path | None = typer.Option(None, help="Output JSON path.")  # pyright: ignore[reportAssignmentType]


@app.command()
def main(
    n_repeats: int = N_REPEATS_OPT,
    output: Path = OUTPUT_OPT,
) -> None:
    """Run post-JIT JAX benchmark and save results."""
    default_out = get_project_root() / "_output" / "jax_jit_benchmarks.json"  # pyright: ignore[reportUnusedVariable]
    out: Path = output if output is not None else default_out  # pyright: ignore[reportUnnecessaryComparison]

    devices = [str(d) for d in jax.devices()]
    typer.echo(f"JAX devices: {devices}")
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

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_repeats": n_repeats,
        "platform": {
            "jax_version": jax.__version__,
            "jax_devices": devices,
        },
        "results": results,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(f"\nResults written to {out}")

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


if __name__ == "__main__":
    app()
