"""Phase 4 review: export the two hard migrated figures to PDF for eyeballing.

Run from the repo root:

    uv run python docs/design/altair_spike/phase4_review.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inf_bp_texts.visualization import save_chart
from inf_bp_texts.visualization.genealogy import plot_genealogy_tree
from inf_bp_texts.visualization.plotting import plot_parameter_consistency

OUT_DIR = Path(__file__).parent


def _genealogy() -> None:
    records = pd.DataFrame(
        {
            "manuscript_id": [0, 1, 2, 3, 4, 5],
            "parent_id": [None, 0, 0, 2, 2, 3],
            "creation_time": [0.0, 0.8, 0.3, 2.0, 3.0, 3.6],
            "death_time": [0.5, 2.6, np.nan, np.nan, 4.1, np.nan],
            "survived": [False, False, True, True, False, True],
            "is_spine": [False, False, False, False, False, False],
        }
    )
    chart = plot_genealogy_tree(records, start_time=0.0, simulation_end_time=5.0)
    save_chart(chart, "p4_genealogy", width=700, output_dir=OUT_DIR)


def _consistency() -> None:
    rng = np.random.default_rng(0)
    regimes = [{"name": "A", "birth_rate": 0.02}, {"name": "B", "birth_rate": 0.05}]
    summary_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for regime in regimes:
        for t in (100, 1000, 10000):
            reps = rng.normal(regime["birth_rate"], 0.005, 30)
            raw_rows.extend(
                {"regime": regime["name"], "T": t, "success": True, "birth_estimate": r}
                for r in reps
            )
            summary_rows.append(
                {
                    "regime": regime["name"],
                    "T": t,
                    "birth_mean": float(np.mean(reps)),
                    "birth_var": float(np.var(reps, ddof=1)),
                }
            )
    chart = plot_parameter_consistency(
        pd.DataFrame(summary_rows),
        pd.DataFrame(raw_rows),
        regimes,
        bootstrap_samples=200,
        seed=7,
        parameter_name="birth",
        true_rate_key="birth_rate",
        title="Birth-rate consistency across observation time",
    )
    save_chart(chart, "p4_consistency", output_dir=OUT_DIR)


def main() -> None:
    _genealogy()
    _consistency()
    print(f"Phase 4 review PDFs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
