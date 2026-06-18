"""Phase 3 review: export the moderate migrated figures to PDF for eyeballing.

Run from the repo root:

    uv run python docs/design/altair_spike/phase3_review.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inf_bp_texts.visualization import save_chart
from inf_bp_texts.visualization.age_fit_plot import creation_year_fit_chart
from inf_bp_texts.visualization.parameter_plots import (
    BoxPlotStyle,
    KdePlotConfig,
    create_kde_density_plot,
    create_parameter_box_plot,
)

OUT_DIR = Path(__file__).parent
PTOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]


def main() -> None:
    rng = np.random.default_rng(3)

    # 1. KDE density plot with dashed true-value rule + text label.
    rates = rng.gamma(shape=4.0, scale=0.0006, size=400)
    save_chart(
        create_kde_density_plot(
            rates=rates,
            true_rate=0.0025,
            config=KdePlotConfig(
                estimate_axis_label="Birth rate (λ)",
                true_label="True λ",
                density_color=PTOL[0],
                true_line_color=PTOL[1],
            ),
        ),
        "p3_kde_density",
        width=460,
        height=320,
        output_dir=OUT_DIR,
    )

    # 2. Box plot with box + true/mean/median markers.
    birth = rng.normal(0.012, 0.002, 120)
    death = rng.normal(0.006, 0.0015, 120)
    save_chart(
        create_parameter_box_plot(
            birth_rates=birth,
            death_rates=death,
            true_rates=(0.012, 0.006),
            style=BoxPlotStyle(
                base_color=PTOL[0],
                true_color=PTOL[1],
                mean_color=PTOL[2],
                median_color=PTOL[3],
            ),
        ),
        "p3_box_plot",
        width=460,
        height=340,
        output_dir=OUT_DIR,
    )

    # 3. Consolidated creation-year histogram + scaled density overlay.
    observed = rng.normal(900.0, 80.0, 300)
    density_years = np.linspace(observed.min(), observed.max(), 200)
    density_values = np.exp(-0.5 * ((density_years - 900.0) / 80.0) ** 2) / (
        80.0 * np.sqrt(2 * np.pi)
    )
    save_chart(
        creation_year_fit_chart(
            observed,
            density_years,
            density_values,
            title="Demo: Observed vs Fitted Creation Years",
            nbins=30,
        ),
        "p3_creation_year_fit",
        width=520,
        height=340,
        output_dir=OUT_DIR,
    )

    print(f"Phase 3 review PDFs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
