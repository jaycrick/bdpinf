# Plotly → Altair Migration: Remaining Work

**Status:** in progress · **Last updated:** 2026-06-22

Phases 0--4 (spike, theme/export chokepoint, easy/moderate/hard helper
migration) and Phase 5 (18 of 19 notebooks) are done.
Altair is the house
plotting library; Plotly survives only in the two locations below.

## What remains

1. **Last Plotly notebook** --- `04_bayesian/kashouh_family_k_bayesian_inference.qmd`:
   4 figures (`prior_posterior` 2×2 KDE, `posterior_rates` bar, `trace_plots`
   2×2 line, `ppc` line+band), all currently `make_subplots` + `go.Scatter`.
   Slow (multi-chain NUTS, rendered on CPU via `scripts/render_cpu_light.slurm`
   --- GPU benchmarked slower, see `docs/design/GPU_RENDERING_PROTOCOL.md`);
   convert using the same Altair theme chokepoint
   (`visualization/altair_theme.py`) and helpers used by the other
   `04_bayesian` notebooks.

2. **Phase 6 --- remove Plotly**, once (1) is done and the remaining Plotly
   references are gone:
   - Drop remaining Plotly usage in `src/inf_bp_texts/visualization/plotting.py`
     and `tests/test_plotting.py`.
   - Drop `plotly>=6.5.2`, `kaleido>=1.2.0`, `plotly-stubs>=0.1.2` from
     `pyproject.toml`; promote `altair`/`vl-convert-python` from the dev group
     to `[project]` dependencies; `uv sync`.
   - Gate: `grep -rn "plotly\|kaleido" src tests qmd_scripts pyproject.toml`
     must be empty, then `uv run pre-commit run --all-files` and
     `uv run pytest` (coverage ≥80%).

**Rollback:** the migration is additive until Phase 6 --- any earlier phase can
be reverted independently.
Phase 6 is the point of no easy return.
