# Plotly → Altair Migration: Completed (Historical)

**Status:** complete · **Completed:** 2026-06-25

This document tracked the migration of the visualization layer from Plotly to
Altair.
All phases are now done: Altair is the only charting library used by
`src/bdpinf/visualization/` and by every notebook under `qmd_scripts/`.

## What changed in the final phase

The original record of this plan understated the remaining work: it listed
only one un-migrated notebook (`kashouh_family_k_bayesian_inference.qmd`), but
all four notebooks under `qmd_scripts/core_analysis/04_bayesian/` were still
pure Plotly (`go.Figure`/`make_subplots`, `fig.show()`,
`save_figure`/`apply_professional_styling`/`apply_tufte_axes`).

The four recurring Bayesian figure archetypes (prior-vs-posterior, posterior
rates with credible intervals, MCMC trace plots, posterior predictive checks)
were extracted into tested Altair builders in
`src/bdpinf/visualization/bayesian_plots.py`
(`prior_posterior_chart`, `posterior_rates_chart`, `trace_chart`, `ppc_chart`,
covered by `tests/test_bayesian_plots.py`), and all four notebooks were
migrated to call them via the existing `save_chart`/`display_svg` chokepoint
in `altair_theme.py`.

Phase 6 (removing Plotly entirely) followed:

- Dropped the dead Plotly-shaped helpers from
  `src/bdpinf/visualization/plotting.py`
  (`apply_professional_styling`, `apply_tufte_axes`, `save_figure`) and
  rewrote `tests/test_plotting.py` to cover the surviving Altair builders
  (`convergence_chart`, the `save_multi_epoch_*` savers) instead.
- Dropped `plotly`, `kaleido`, `plotly-stubs` from `pyproject.toml` (`altair`
  and `vl-convert-python` were already `[project]` dependencies, not dev-group
  ones as an earlier draft of this plan assumed).
- Scrubbed remaining Plotly references from comments/docstrings across
  `visualization/`, `analysis/validation_helpers.py`, `CLAUDE.md`,
  `great-docs.yml`, and `README.md`.

Gate passed: `grep -rn "plotly\|kaleido" src tests qmd_scripts pyproject.toml`
returns nothing outside historical artifacts (`.jupyter_cache/`, this
document).
