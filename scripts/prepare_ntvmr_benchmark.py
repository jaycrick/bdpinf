r"""Prepare the NTVMR multi-knot benchmark data (needs the nt-manuscripts repo).

Reads the cleaned manuscript table, takes the continuous-text subset
(groups 1-3), and writes ``scripts/data/ntvmr_benchmark.json`` with:

- the distinct date ranges with counts (each manuscript is the years
  ``[earlyYear, lateYear + 1)``, clipped to the support ``[start, end)``);
- the truth: a fit of continuous piecewise-linear birth and death rates with
  every epoch boundary fixed at its current date;
- the epoch dates and the allowed range of each interior knot (the midpoints
  between neighbouring current dates).

``bench_ntvmr_knots.py`` then needs nothing from nt-manuscripts.

Usage
-----
Run it in the nt-manuscripts environment, from the bdpinf repo root:

    uv run --project ../nt-manuscripts python scripts/prepare_ntvmr_benchmark.py
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
from bench_ntvmr_knots import (
    DATA_PATH,
    build_model,
    clip_years,
    fixed_constraints,
    real_data,
)
from nt_manuscripts.constants import MANUSCRIPT_EPOCHS_SIX_ERA

from bdpinf.inference import fit

SEED = 209  # as in ntvmr_linear_rates_inference.qmd


def main() -> None:
    """Build the benchmark data and fit the truth."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("../nt-manuscripts/manuscript_analysis_cleaned.csv"),
    )
    parser.add_argument("--out", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="write the ranges and bounds only; fit the truth later with "
        "bench_ntvmr_knots.py --truth (for example on the HPC)",
    )
    args = parser.parse_args()

    start = int(MANUSCRIPT_EPOCHS_SIX_ERA[0][0])
    ends = [int(e[1]) for e in MANUSCRIPT_EPOCHS_SIX_ERA]
    end = ends[-1]
    frame = pd.read_csv(args.csv).dropna(subset=["earlyYear", "lateYear"])
    frame = frame[frame["group"].isin([1, 2, 3])]
    lower, upper = clip_years(
        frame["earlyYear"].to_numpy(), frame["lateYear"].to_numpy() + 1, start, end
    )
    pairs, counts = np.unique(
        np.stack([lower, upper], axis=1), axis=0, return_counts=True
    )

    interior = ends[:-1]  # 325 ... 1516
    edges = [start, *interior, end]
    bounds = [
        [
            0.5 * (edges[k - 1] + edges[k]) - start,
            0.5 * (edges[k] + edges[k + 1]) - start,
        ]
        for k in range(1, len(edges) - 1)
    ]
    spec = {
        "description": "NTVMR continuous text (groups 1-3), date ranges, six-era knots",
        "start_year": start,
        "end_year": end,
        "horizon": float(end - start),
        "epoch_ends": ends,
        "n": int(counts.sum()),
        "knot_names": [f"tau{k}" for k in range(len(interior))],
        "knot_initial": [float(k - start) for k in interior],
        "knot_bounds": bounds,
        "zero_birth_name": f"birth.knot{len(ends)}",
        "ranges": [
            [int(a), int(b), int(c)] for (a, b), c in zip(pairs, counts, strict=True)
        ],
        "truth": [],
        "names": [],
        "truth_log_likelihood": None,
    }
    model = build_model(spec)
    spec["names"] = list(model.names)
    if not args.no_fit:
        data = real_data(spec)
        truth = fit(
            data,
            model,
            constraints=fixed_constraints(spec),
            n_starts=5,
            perturb_std=0.1,
            seed=SEED,
        )
        spec["truth"] = [float(truth.params[n]) for n in model.names]
        spec["truth_log_likelihood"] = float(truth.log_likelihood)
        print(f"success={truth.success}, log-likelihood={truth.log_likelihood:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=1))
    print(f"n={spec['n']}, distinct ranges={len(pairs)}")
    print(f"knot bounds (years after {start}): {bounds}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
