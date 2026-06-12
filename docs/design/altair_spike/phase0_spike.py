"""Phase 0 spike for the Plotly→Altair migration.

Reproduces two figures in Altair to validate the migration premise before
committing to the full port (see PLOTLY_TO_ALTAIR_IMPLEMENTATION_PLAN.md, Phase 0):

1. An *easy* figure — the empirical-mean line plot from ``splitting_plots`` — to
   confirm vl-convert PDF export quality matches kaleido.
2. The *risky* figure — the genealogy tree — to confirm the shapes-based Plotly
   drawing is expressible declaratively in Altair via ``mark_rule`` segments
   built from the existing tidy ``genealogy_dfs_order`` frame.

Both are exported to PDF via vl-convert (Altair) and kaleido (Plotly) so the
two can be eyeballed side-by-side in ``docs/design/altair_spike/``.

Run from the repo root:

    uv run python docs/design/altair_spike/phase0_spike.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from inf_bp_texts.visualization.genealogy import (
    GenealogyPlotConfig,
    genealogy_dfs_order,
    plot_genealogy_tree,
)
from inf_bp_texts.visualization.splitting_plots import plot_empirical_population_mean

OUT_DIR = Path(__file__).parent

# ptolqual palette (ported verbatim from visualization.plotting; the real
# migration will register this as an Altair theme in Phase 1).
PTOLQUAL = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Violet
    "#BBBBBB",  # Gray
]
FONT = "Libertinus Serif, Latin Modern Roman, Times New Roman, Times, serif"


def _tufte_axis(**kwargs: object) -> alt.Axis:
    """Tufte-style axis: outward muted ticks, no gridlines, thin domain line."""
    return alt.Axis(
        grid=False,
        domainColor="#BBBBBB",
        domainWidth=1,
        tickColor="#777777",
        tickSize=3,
        labelFont=FONT,
        titleFont=FONT,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Figure 1 (easy): empirical mean population vs level time
# --------------------------------------------------------------------------
def spike_empirical_mean() -> None:
    """Easy line figure — Plotly vs Altair, both → PDF."""
    rng = np.random.default_rng(0)
    level_times = np.linspace(0.0, 10.0, 12)
    mean_populations = 100.0 * np.exp(-0.18 * level_times) + rng.normal(0, 1.5, 12)

    # --- Plotly reference (existing helper) ---
    plotly_fig = plot_empirical_population_mean(
        level_times=level_times,
        mean_populations=mean_populations,
        title="Empirical Mean Population Across Levels",
    )
    plotly_fig.update_layout(width=550, height=400)
    plotly_fig.write_image(str(OUT_DIR / "easy_plotly.pdf"))

    # --- Altair candidate ---
    df = pd.DataFrame({"level_time": level_times, "mean_population": mean_populations})
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=40), color=PTOLQUAL[0], strokeWidth=2)
        .encode(
            x=alt.X("level_time:Q", title="Level time", axis=_tufte_axis()),
            y=alt.Y("mean_population:Q", title="Mean population", axis=_tufte_axis()),
        )
        .properties(
            width=470,
            height=330,
            title="Empirical Mean Population Across Levels",
        )
        .configure_view(stroke=None)
        .configure_title(font=FONT, fontSize=15)
    )
    chart.save(str(OUT_DIR / "easy_altair.pdf"), engine="vl-convert")
    print("[easy]  wrote easy_plotly.pdf and easy_altair.pdf")


# --------------------------------------------------------------------------
# Figure 2 (risky): genealogy tree
# --------------------------------------------------------------------------
def _sample_genealogy() -> pd.DataFrame:
    """Return a small multi-generation tree with survivors and deaths."""
    return pd.DataFrame(
        {
            "manuscript_id": [1, 2, 3, 4, 5, 6],
            "parent_id": [None, 1, 1, 2, 3, 3],
            "creation_time": [0.0, 80.0, 150.0, 220.0, 300.0, 360.0],
            "death_time": [np.nan, 260.0, np.nan, np.nan, 410.0, np.nan],
            "survived": [True, False, True, True, False, True],
            "is_spine": [False, False, False, False, False, False],
        }
    )


def spike_genealogy() -> None:
    """Risky tree figure — Plotly shapes vs Altair mark_rule, both → PDF."""
    records = _sample_genealogy()
    start_time, end_time = 0.0, 500.0

    # --- Plotly reference (existing helper) ---
    plotly_fig = plot_genealogy_tree(
        records,
        start_time=start_time,
        simulation_end_time=end_time,
        config=GenealogyPlotConfig(width=700),
    )
    plotly_fig.write_image(str(OUT_DIR / "tree_plotly.pdf"))

    # --- Altair candidate: build tidy segment frames from the SAME dfs frame ---
    frame = genealogy_dfs_order(
        records, start_time=start_time, simulation_end_time=end_time
    )
    parent_y = dict(zip(frame["manuscript_id"], frame["dfs_index"], strict=True))

    # Lifetime segments: horizontal rule from birth_time → plot_death_time at y=dfs_index.
    def _category(row: pd.Series) -> str:
        if row["survived"]:
            return "spine" if row["is_spine"] else "survivor"
        return "extinct"

    lifetimes = frame.assign(category=frame.apply(_category, axis=1))

    # Connector segments: vertical dotted rule from parent's y to child's y at child birth_time.
    connectors = frame[frame["parent_id"].notna()].copy()
    connectors["parent_y"] = connectors["parent_id"].map(parent_y)

    color_scale = alt.Scale(
        domain=["extinct", "survivor", "spine"],
        range=["#000000", PTOLQUAL[2], PTOLQUAL[2]],
    )
    y_enc = alt.Y(
        "dfs_index:O",
        sort="ascending",
        title="DFS order index",
        axis=_tufte_axis(),
    )

    lifetime_marks = (
        alt.Chart(lifetimes)
        .mark_rule(strokeWidth=2.5)
        .encode(
            x=alt.X("birth_time:Q", title="Time", axis=_tufte_axis()),
            x2="plot_death_time:Q",
            y=y_enc,
            color=alt.Color("category:N", scale=color_scale, legend=None),
        )
    )
    connector_marks = (
        alt.Chart(connectors)
        .mark_rule(strokeDash=[2, 2], strokeWidth=1.25, color="#666666")
        .encode(
            x="birth_time:Q",
            y=alt.Y("parent_y:O"),
            y2="dfs_index:O",
        )
    )
    tree = (
        (connector_marks + lifetime_marks)
        .properties(width=620, height=max(220, 40 * len(frame)), title="Genealogy Tree")
        .configure_view(stroke=None)
        .configure_title(font=FONT, fontSize=15)
    )
    tree.save(str(OUT_DIR / "tree_altair.pdf"), engine="vl-convert")
    print("[tree]  wrote tree_plotly.pdf and tree_altair.pdf")


if __name__ == "__main__":
    spike_empirical_mean()
    spike_genealogy()
    print(f"\nSpike PDFs written to {OUT_DIR}")
