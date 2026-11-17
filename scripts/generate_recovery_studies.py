r"""Generate and persist the simulate-then-fit "recovery study" datasets.

This script is an **offline artifact generator**, mirroring the application
repo's (nt-manuscripts) ``scripts/generate_bootstrap_kde_samples.py``: run it
once (ideally on an HPC
node with many cores) to produce ``recovery_samples/<study>.npz`` under the
project root. The heaviest ``qmd_scripts/validation`` notebooks then load
these files via
:func:`~bdpinf.inference.load_recovery_results` instead of re-running
thousands of simulate-then-fit replicates on every render.

Studies
-------
``single_epoch_bulk_inference`` (~3 999 replicates)
    validation/single_epoch/06_bulk_inference.qmd
``single_epoch_asymptotic_consistency`` (60 000 replicates)
    validation/single_epoch/07_asymptotic_consistency.qmd
``single_epoch_fixed_time_consistency`` (200 replicates x 3 regimes x 50 fits)
    validation/single_epoch/09_asymptotic_consistency_fixed_time.qmd
``multi_epoch_two_rate`` (up to 1 000 replicates)
    validation/multi_epoch/03_bulk_inference.qmd
``multi_epoch_single_rate`` (up to 1 000 replicates)
    validation/multi_epoch/04_bulk_inference_single_rate.qmd
``multi_epoch_zero_birth`` (1 000 replicates)
    validation/multi_epoch/05_bulk_inference_zero_birth.qmd

The per-replicate simulate+fit logic itself lives in
:mod:`bdpinf.analysis.single_epoch_recovery`,
:mod:`bdpinf.analysis.multi_epoch_recovery`, and
:mod:`bdpinf.analysis.asymptotic_consistency` — this script only wires
up each study's parameter grid, runs it via ``joblib.Parallel``, and persists
the tidy result table.

Usage examples
--------------
::

    # Full run (all five studies, production replicate counts):
    uv run python scripts/generate_recovery_studies.py

    # Smoke test — scale every study down to ~10 replicates:
    uv run python scripts/generate_recovery_studies.py --smoke-test

    # Just the single biggest win:
    uv run python scripts/generate_recovery_studies.py \\
        --study single_epoch_bulk_inference
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------------------
# Per-study generators
# ---------------------------------------------------------------------------


def _generate_single_epoch_bulk_inference(
    num_sims: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``06_bulk_inference.qmd``: 1 birth/death rate, single epoch."""
    from bdpinf.analysis import run_single_epoch_recovery_replicate
    from bdpinf.utils import derive_worker_seeds

    seed = 42
    birth_rate = 0.5
    death_rate = birth_rate - 0.4
    time_span = 15.0
    sim_seeds = derive_worker_seeds(seed, num_sims)

    # The notebook simulates `num_sims` replicates but only ever fits
    # `list_of_ages[1:]`; each replicate's RNG stream is independent
    # (jax.random.split-derived), so skipping index 0 entirely reproduces the
    # same fitted results without wasting a simulation.
    rows = Parallel(n_jobs=-1)(
        delayed(run_single_epoch_recovery_replicate)(
            i, sim_seeds[i], birth_rate, death_rate, time_span
        )
        for i in range(1, num_sims)
    )
    df = pd.DataFrame(rows)
    metadata = {
        "n_simulated": num_sims,
        "n_kept": len(df),
        "birth_rate": birth_rate,
        "death_rate": death_rate,
        "time_span": time_span,
        "seed": seed,
    }
    return df, metadata


def _generate_single_epoch_asymptotic_consistency(
    num_simulations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``07_asymptotic_consistency.qmd``: 4 regimes x 15 observation times."""
    from bdpinf.analysis.asymptotic_consistency import (
        FittingRunConfig,
        run_single_replicate,
    )

    seed = 42
    time_grid = np.logspace(-0.5, 0.5, num=15)
    regime_configs: list[dict[str, Any]] = [
        {
            "name": "Supercritical",
            "sampling": "process",
            "birth_rate": 5.0,
            "death_rate": 2.0,
        },
        {
            "name": "Almost critical",
            "sampling": "process",
            "birth_rate": 5.0,
            "death_rate": 4.0,
        },
        {
            "name": "Critical",
            "sampling": "distribution",
            "birth_rate": 5.0,
            "death_rate": 5.0,
        },
        {
            "name": "Subcritical",
            "sampling": "distribution",
            "birth_rate": 5.0,
            "death_rate": 6.0,
        },
    ]
    fit_run_config = FittingRunConfig(seed=seed, method="trust-exact", tol=1e-6)

    rows: list[dict[str, object]] = []
    for regime_config in regime_configs:
        for observation_time in time_grid:
            rows.extend(
                Parallel(n_jobs=-1)(
                    delayed(run_single_replicate)(
                        regime_config=regime_config,
                        observation_time=float(observation_time),
                        simulation_index=simulation_index,
                        fit_config=fit_run_config,
                    )
                    for simulation_index in range(num_simulations)
                )
            )
    df = pd.DataFrame(rows)
    metadata = {"num_simulations": num_simulations, "seed": seed}
    return df, metadata


def _generate_single_epoch_fixed_time_consistency(
    num_simulations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``09_asymptotic_consistency_fixed_time.qmd``: fixed T, growing sample."""
    from bdpinf.analysis.asymptotic_consistency import (
        FittingRunConfig,
        fixed_time_size_grid,
        population_size_quantile,
        run_fixed_time_replicate,
    )

    seed = 42
    observation_time = 2.0
    regime_configs: list[dict[str, Any]] = [
        {"name": "Subcritical", "birth_rate": 5.0, "death_rate": 6.0},
        {"name": "Critical", "birth_rate": 5.0, "death_rate": 5.0},
        {"name": "Supercritical", "birth_rate": 5.0, "death_rate": 2.0},
    ]
    # The supercritical population distribution sets the size grid for all regimes.
    n_max = population_size_quantile(5.0, 2.0, observation_time, q=0.99)
    size_grid = fixed_time_size_grid(n_max, num_points=50)
    fit_run_config = FittingRunConfig(seed=seed, method="trust-exact", tol=1e-6)

    rows: list[dict[str, object]] = []
    for regime_config in regime_configs:
        for replicate_rows in Parallel(n_jobs=-1)(
            delayed(run_fixed_time_replicate)(
                regime_config=regime_config,
                observation_time=observation_time,
                size_grid=size_grid,
                simulation_index=simulation_index,
                fit_config=fit_run_config,
            )
            for simulation_index in range(num_simulations)
        ):
            rows.extend(replicate_rows)
    df = pd.DataFrame(rows)
    metadata = {
        "num_simulations": num_simulations,
        "seed": seed,
        "observation_time": observation_time,
        "n_max": n_max,
        "size_grid": size_grid.tolist(),
    }
    return df, metadata


def _generate_multi_epoch_two_rate(
    num_sims: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``03_bulk_inference.qmd``: two-epoch process, distinct rates per epoch."""
    from bdpinf.analysis import FitModelConfig, run_two_rate_replicate
    from bdpinf.utils import derive_worker_seeds

    seed = 42
    birth_rates_true = np.array([0.5, 0.7], dtype=float)
    death_rates_true = np.array([0.1, 0.15], dtype=float)
    times = np.array([7.5, 15.0], dtype=float)
    sim_seeds = derive_worker_seeds(seed, num_sims)

    fit_config = FitModelConfig(
        times=times,
        is_ordered=True,
        random_seed=10925,
        initial_birth_rates=birth_rates_true,
        initial_death_rates=death_rates_true,
    )
    rows = Parallel(n_jobs=-1)(
        delayed(run_two_rate_replicate)(
            i,
            sim_seeds[i],
            birth_rates_true,
            death_rates_true,
            times,
            fit_config=fit_config,
        )
        for i in range(num_sims)
    )
    kept = [row for row in rows if row is not None]
    df = pd.DataFrame(kept)
    metadata = {"n_simulated": num_sims, "n_kept": len(kept), "seed": seed}
    return df, metadata


def _generate_multi_epoch_single_rate(
    num_sims: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``04_bulk_inference_single_rate.qmd``: single-rate data fit with two epochs."""
    from bdpinf.analysis import FitModelConfig, run_single_rate_replicate
    from bdpinf.utils import derive_worker_seeds

    seed = 42
    birth_rate = 0.5
    death_rate = birth_rate - 0.4
    time_span = 15.0
    fit_times = np.array([time_span / 2.0, time_span], dtype=float)
    birth_rates_true = np.array([birth_rate, birth_rate], dtype=float)
    death_rates_true = np.array([death_rate, death_rate], dtype=float)
    sim_seeds = derive_worker_seeds(seed, num_sims)

    fit_config = FitModelConfig(
        times=fit_times,
        is_ordered=True,
        random_seed=10925,
        initial_birth_rates=birth_rates_true,
        initial_death_rates=death_rates_true,
    )
    rows = Parallel(n_jobs=-1)(
        delayed(run_single_rate_replicate)(
            i,
            sim_seeds[i],
            birth_rate,
            death_rate,
            time_span,
            fit_times=fit_times,
            fit_config=fit_config,
        )
        for i in range(num_sims)
    )
    kept = [row for row in rows if row is not None]
    df = pd.DataFrame(kept)
    metadata = {"n_simulated": num_sims, "n_kept": len(kept), "seed": seed}
    return df, metadata


def _generate_multi_epoch_zero_birth(
    num_sims: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``05_bulk_inference_zero_birth.qmd``: structural-zero birth, shared death."""
    from bdpinf.analysis import FitModelConfig, run_zero_birth_replicate
    from bdpinf.inference import Constraints
    from bdpinf.utils import derive_worker_seeds

    birth_rate_epoch_1_true = 1.0
    death_rate_true = birth_rate_epoch_1_true - 0.9
    birth_rate_epoch_2_true = 0.0
    times = np.array([7.5, 15.0], dtype=float)

    # Two independent jax.random.split-derived seed streams (first-epoch
    # simulation and post-split death-time draws), matching the notebook.
    first_epoch_seeds = derive_worker_seeds(42000, num_sims)
    death_time_seeds = derive_worker_seeds(52000, num_sims)

    birth_rates_true = np.array([birth_rate_epoch_1_true, birth_rate_epoch_2_true])
    death_rates_true = np.array([death_rate_true, death_rate_true])
    fit_config = FitModelConfig(
        times=times,
        is_ordered=True,
        random_seed=10925,
        initial_birth_rates=birth_rates_true,
        initial_death_rates=death_rates_true,
    )
    param_constraints = Constraints(
        fixed={"birth.v1": 0.0},
        equal=[["death.v0", "death.v1"]],
    )

    rows = Parallel(n_jobs=-1)(
        delayed(run_zero_birth_replicate)(
            i,
            first_epoch_seeds[i],
            death_time_seeds[i],
            birth_rate_epoch_1_true,
            death_rate_true,
            times=times,
            fit_config=fit_config,
            param_constraints=param_constraints,
        )
        for i in range(num_sims)
    )
    df = pd.DataFrame(rows)
    metadata = {"n_simulated": num_sims, "seed": 42000}
    return df, metadata


# ---------------------------------------------------------------------------
# Registry: study key -> (generator, default replicate count)
# ---------------------------------------------------------------------------

_STUDIES: dict[str, tuple[Any, int]] = {
    "single_epoch_bulk_inference": (_generate_single_epoch_bulk_inference, 4000),
    "single_epoch_asymptotic_consistency": (
        _generate_single_epoch_asymptotic_consistency,
        1000,
    ),
    "single_epoch_fixed_time_consistency": (
        _generate_single_epoch_fixed_time_consistency,
        200,
    ),
    "multi_epoch_two_rate": (_generate_multi_epoch_two_rate, 1000),
    "multi_epoch_single_rate": (_generate_multi_epoch_single_rate, 1000),
    "multi_epoch_zero_birth": (_generate_multi_epoch_zero_birth, 1000),
}
_ALL_KEYS: list[str] = list(_STUDIES.keys())


@app.command()
def main(
    study: Annotated[
        list[str] | None,
        typer.Option(
            "--study",
            help=(
                "Study key(s) to generate. May be repeated. "
                f"Choices: {', '.join(_ALL_KEYS)}. Default: all."
            ),
        ),
    ] = None,
    smoke_test: Annotated[
        bool,
        typer.Option(
            "--smoke-test",
            help="Scale every study down to ~10 replicates, to check the wiring.",
        ),
    ] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help="Output directory. Defaults to <project_root>/recovery_samples/.",
        ),
    ] = None,
) -> None:
    """Generate and save recovery-study (simulate-then-fit) datasets."""
    from bdpinf import find_project_root
    from bdpinf.inference import save_recovery_results

    project_root = find_project_root()
    output_dir = out_dir or (project_root / "recovery_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    keys_to_run: list[str] = list(study) if study else _ALL_KEYS
    invalid = [k for k in keys_to_run if k not in _STUDIES]
    if invalid:
        typer.echo(f"Unknown study/studies: {invalid}. Valid: {_ALL_KEYS}", err=True)
        raise typer.Exit(code=1)

    total_start = time.perf_counter()

    for key in keys_to_run:
        generator, default_n = _STUDIES[key]
        n = 10 if smoke_test else default_n

        typer.echo(f"\n[{key}] generating {n} replicates …")
        t0 = time.perf_counter()

        results_df, metadata = generator(n)

        out_path = output_dir / f"{key}.npz"
        save_recovery_results(results_df, out_path, metadata=metadata)

        elapsed = time.perf_counter() - t0
        size_mb = out_path.stat().st_size / 1e6
        typer.echo(
            f"  Saved {out_path.name}  ({len(results_df)} rows, {size_mb:.2f} MB)  "
            f"in {elapsed:.1f}s"
        )

    total_elapsed = time.perf_counter() - total_start
    typer.echo(f"\nDone. Total time: {total_elapsed:.1f}s")
    typer.echo(f"Output directory: {output_dir}")


if __name__ == "__main__":
    app()
