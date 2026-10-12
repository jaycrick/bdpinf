"""Helpers for identifying and reproducing spike-driving fits."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

MIN_VARIANCE_SAMPLE_SIZE = 2


def seed_from_parts(*parts: object) -> int:
    """Create a stable integer seed from arbitrary parts.

    Used across validation modules to derive deterministic RNG seeds.
    """
    seed_material = "|".join(
        (
            int(part).to_bytes(8, "little", signed=True).hex()
            if isinstance(part, (np.integer, int))
            else (
                np.float64(part).hex()
                if isinstance(part, (np.floating, float))
                else str(part)
            )
        )
        for part in parts
    )
    digest = hashlib.blake2b(seed_material.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def replicate_seed(
    seed: int,
    regime: str,
    sampling: str,
    observation_time: float,
    simulation_index: int,
) -> int:
    """Return the replicate-level seed used for sampling and fitting.

    Parameters mirror how seeds were constructed in the notebooks.
    """
    return seed_from_parts(
        seed,
        regime,
        sampling,
        np.float64(observation_time).hex(),
        int(simulation_index),
    )


def bootstrap_seed(replicate_seed: int) -> int:
    """Return a bootstrap seed derived from a replicate seed."""
    return seed_from_parts(replicate_seed, "bootstrap")


def bootstrap_percentile_ci(
    values: Sequence[float] | np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
    statistic: str,
) -> dict[str, float]:
    """Return a bootstrap percentile CI summary for a 1D sample."""
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]

    if sample.size == 0:
        return {
            "estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "ci_width": np.nan,
        }

    rng = np.random.default_rng(int(seed))
    if statistic == "mean":
        estimate = float(np.mean(sample))
        boot_stats = np.array(
            [
                float(np.mean(rng.choice(sample, size=sample.size, replace=True)))
                for _ in range(int(bootstrap_samples))
            ],
            dtype=np.float64,
        )
    elif statistic == "median":
        estimate = float(np.median(sample))
        boot_stats = np.array(
            [
                float(np.median(rng.choice(sample, size=sample.size, replace=True)))
                for _ in range(int(bootstrap_samples))
            ],
            dtype=np.float64,
        )
    elif statistic == "variance":
        if sample.size < MIN_VARIANCE_SAMPLE_SIZE:
            return {
                "estimate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "ci_width": np.nan,
            }

        estimate = float(np.var(sample, ddof=1))
        boot_stats = np.array(
            [
                float(
                    np.var(rng.choice(sample, size=sample.size, replace=True), ddof=1),
                )
                for _ in range(int(bootstrap_samples))
            ],
            dtype=np.float64,
        )
    else:
        msg = f"Unsupported statistic: {statistic}"
        raise ValueError(msg)

    finite_boot_stats = boot_stats[np.isfinite(boot_stats)]
    if finite_boot_stats.size == 0:
        return {
            "estimate": estimate,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "ci_width": np.nan,
        }

    ci_lower, ci_upper = np.percentile(finite_boot_stats, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_width": float(ci_upper - ci_lower),
    }


def compute_ci_width_table(
    results_df: pd.DataFrame,
    regime_configs: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Compute bootstrap CI widths for birth/death estimates over time."""
    regime_lookup = {config["name"]: config for config in regime_configs}
    rows: list[dict[str, Any]] = []

    for (regime, sampling, observation_time), group in results_df.groupby(
        ["regime", "sampling", "T"],
        sort=False,
    ):
        observation_time_float = float(cast("float", observation_time))
        successful = group[group["success"]]
        for parameter_name, true_rate_key in (
            ("birth", "birth_rate"),
            ("death", "death_rate"),
        ):
            estimates = successful[f"{parameter_name}_estimate"].to_numpy(
                dtype=np.float64,
            )
            for metric_name, statistic in (
                ("mean", "mean"),
                ("median", "median"),
                ("variance", "variance"),
            ):
                ci_summary = bootstrap_percentile_ci(
                    estimates,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed_from_parts(
                        seed,
                        "bootstrap",
                        regime,
                        sampling,
                        np.float64(observation_time_float).hex(),
                        parameter_name,
                        metric_name,
                    ),
                    statistic=statistic,
                )
                rows.append(
                    {
                        "regime": regime,
                        "sampling": sampling,
                        "T": observation_time_float,
                        "parameter_name": parameter_name,
                        "metric_name": metric_name,
                        "n_replicates": len(group),
                        "n_successful": len(successful),
                        "true_value": float(regime_lookup[regime][true_rate_key]),
                        **ci_summary,
                    },
                )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["regime", "parameter_name", "metric_name", "T"],
        )
        .reset_index(drop=True)
    )


def select_max_ci_times(ci_width_df: pd.DataFrame) -> pd.DataFrame:
    """Select the time point with the widest CI for each scenario/metric pair."""
    if ci_width_df.empty:
        return ci_width_df.copy()

    selected_rows: list[pd.Series] = []
    group_columns = ["regime", "sampling", "parameter_name", "metric_name"]
    for _, group in ci_width_df.groupby(group_columns, sort=False):
        finite_group = group[np.isfinite(group["ci_width"])]
        if finite_group.empty:
            continue
        max_width = float(finite_group["ci_width"].max())
        selected = (
            finite_group[finite_group["ci_width"] == max_width]
            .sort_values(
                ["T"],
            )
            .head(1)
        )
        if not selected.empty:
            selected_rows.append(selected.iloc[0])

    if not selected_rows:
        return ci_width_df.iloc[0:0].copy()

    return (
        pd.DataFrame(selected_rows)
        .sort_values(
            ["regime", "parameter_name", "metric_name"],
        )
        .reset_index(drop=True)
    )


def select_extreme_replicates(
    results_df: pd.DataFrame,
    target_times_df: pd.DataFrame,
    *,
    num_simulations: int,
    include_lowest: bool = True,
) -> pd.DataFrame:
    """Select the top-k extreme fits for each target time and parameter."""
    if target_times_df.empty:
        return target_times_df.iloc[0:0].copy()

    k = min(max(5, int(np.ceil(num_simulations * 0.025))), 20)
    selection_rows: list[dict[str, Any]] = []
    sides = ["highest", "lowest"] if include_lowest else ["highest"]

    for _, target_row in target_times_df.iterrows():
        scenario_mask = (
            (results_df["regime"] == target_row["regime"])
            & (results_df["sampling"] == target_row["sampling"])
            & np.isclose(
                results_df["T"].to_numpy(dtype=np.float64),
                float(target_row["T"]),
            )
            & (results_df["success"])
        )
        scenario_rows = results_df.loc[scenario_mask].copy()
        for parameter_name in ("birth", "death"):
            estimate_column = f"{parameter_name}_estimate"
            for side in sides:
                ascending = side == "lowest"
                ordered = scenario_rows.sort_values(
                    [estimate_column, "simulation_index"],
                    ascending=[ascending, True],
                ).head(k)

                for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
                    selection_rows.append(
                        {
                            "regime": row["regime"],
                            "sampling": row["sampling"],
                            "T": float(row["T"]),
                            "simulation_index": int(row["simulation_index"]),
                            "population_size": int(row["population_size"]),
                            "birth_estimate": float(row["birth_estimate"]),
                            "death_estimate": float(row["death_estimate"]),
                            "replicate_seed": row.get("replicate_seed", np.nan),
                            "selection_parameter": parameter_name,
                            "selection_side": side,
                            "selection_rank": int(rank),
                            "selection_estimate": float(row[estimate_column]),
                            "target_parameter": target_row["parameter_name"],
                            "target_metric": target_row["metric_name"],
                            "target_time": float(target_row["T"]),
                            "target_ci_width": float(target_row["ci_width"]),
                            "target_true_value": float(target_row["true_value"]),
                            "target_n_successful": int(target_row["n_successful"]),
                            "target_n_replicates": int(target_row["n_replicates"]),
                            "selection_key": (
                                f"{target_row['parameter_name']}:"
                                f"{target_row['metric_name']}"
                            ),
                        },
                    )

    if not selection_rows:
        return results_df.iloc[0:0].copy()

    return (
        pd.DataFrame(selection_rows)
        .sort_values(
            [
                "regime",
                "sampling",
                "target_time",
                "selection_parameter",
                "selection_side",
                "selection_rank",
                "simulation_index",
            ],
        )
        .reset_index(drop=True)
    )
