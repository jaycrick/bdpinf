"""Phase 2 review: export the five migrated easy figures to PDF for eyeballing.

Run from the repo root:

    uv run python docs/design/altair_spike/phase2_review.py

Writes one PDF per figure into this directory (gitignored). These back Review
Gate 1 in the Plotly→Altair implementation plan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inf_bp_texts.visualization import (
    plot_conditioned_trajectories,
    plot_empirical_population_mean,
    plot_level_survival_probabilities,
    save_chart,
    save_multi_epoch_convergence_figure,
    save_multi_epoch_fit_figure,
)

OUT_DIR = Path(__file__).parent


class _Traj:
    """Minimal stand-in for simulation.splitting.ConditionedTrajectory."""

    def __init__(self, particle_id: int, times: np.ndarray, populations: np.ndarray):
        self.particle_id = particle_id
        self.times = times
        self.populations = populations


def main() -> None:
    rng = np.random.default_rng(1)

    # 1. Conditioned trajectories (multi-series line, colour per particle).
    trajectories = []
    for pid in range(8):
        t = np.linspace(0.0, 50.0, 200)
        pop = np.maximum(
            0.0, 20.0 + np.cumsum(rng.normal(0.0, 1.0, t.size)) + 0.4 * pid
        )
        trajectories.append(_Traj(pid, t, pop))
    save_chart(
        plot_conditioned_trajectories(trajectories, max_trajectories=8),
        "p2_conditioned_trajectories",
        width=520,
        height=360,
        output_dir=OUT_DIR,
    )

    # 2. Empirical mean population (single line + points).
    level_times = np.linspace(0.0, 12.0, 14)
    mean_pop = 80.0 * np.exp(-0.16 * level_times) + rng.normal(0, 1.2, level_times.size)
    save_chart(
        plot_empirical_population_mean(
            level_times=level_times, mean_populations=mean_pop
        ),
        "p2_empirical_mean",
        width=470,
        height=330,
        output_dir=OUT_DIR,
    )

    # 3. Level survival probabilities (line + points, y in [0, 1]).
    probs = np.clip(np.linspace(0.95, 0.4, 13) + rng.normal(0, 0.02, 13), 0, 1)
    save_chart(
        plot_level_survival_probabilities(
            level_times=level_times, level_survival_probabilities=probs
        ),
        "p2_level_survival",
        width=470,
        height=330,
        output_dir=OUT_DIR,
    )

    # 4. Convergence trace (saves itself via save_chart internally).
    nll = list(380.0 - 60.0 * (1.0 - np.exp(-0.25 * np.arange(20))))
    result_conv = {
        "success": True,
        "optimization_result": {
            "selected_attempt_index": 0,
            "attempts": [{"nll_history": nll}],
        },
    }
    save_multi_epoch_convergence_figure(
        "Demo Case", result_conv, "p2_convergence", figures_dir=OUT_DIR
    )

    # 5. Multi-epoch fit bars (two panels, data labels).
    epochs = [(300, 600, "early"), (600, 900, "mid"), (900, 1200, "late")]
    result_fit = {
        "success": True,
        "birth_rates": [0.012345, 0.008210, 0.004567],
        "death_rates": [0.003210, 0.006540, 0.009870],
    }
    save_multi_epoch_fit_figure(
        "Demo Case", result_fit, epochs, "p2_fit_bars", figures_dir=OUT_DIR
    )

    print(f"Phase 2 review PDFs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
