# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `analysis.subset_inference` --- `build_manuscript_subsets`/`fit_manuscript_subset`
  split the cleaned NTVMR data into six canonical subsets (non-lectionaries
  combined, lectionaries, and four per-section non-lectionary analyses) and fit
  each via MLE with per-epoch rates/SEs/95% CIs.
- `analysis.date_density` --- `explode_date_ranges`, `manuscript_date_density`,
  `quantile_resample_dates`, `resample_dates_from_ranges` for weighted KDE over
  manuscript date *ranges* (year-explosion + 1/n weights) instead of midpoints;
  `data_cleaning.add_resampled_year_column` populates the resulting
  `resampledYear` column. See `docs/design/KDE_DATE_RANGE_HANDLING.md`.
- `visualization.age_fit_plot.plot_creation_year_fit` and
  `visualization.date_distribution_plots.plot_date_range_kde_comparison` --- shared
  Altair builders for the recurring creation-year/date-range figures, replacing
  duplicated inline plotting code across notebooks.
- `visualization.altair_theme` --- registers the house Altair theme (ptolqual
  palette, Tufte axes) and `save_chart`, the vl-convert export chokepoint;
  first step of an in-progress Plotly→Altair migration (most helpers and 18/19
  notebooks now use Altair; see `docs/design/PLOTLY_TO_ALTAIR_IMPLEMENTATION_PLAN.md`
  for the one remaining notebook and the final dependency removal).
- `inference.bayesian` --- per-family prior loc/scale on `BayesianPriorConfig`,
  in-model parameter constraints and GPU/CUDA backend selection on
  `BayesianModelConfig`, and `mle_init_log_rates()` for MLE-warm-started NUTS.
- `qmd_scripts/core_analysis/04_bayesian/` --- four Bayesian inference notebooks
  (Kashouh FamilyK, Kashouh single-epoch, NTVMR single-epoch, NTVMR six-epoch)
  and a `gpu` dependency group for rendering them on an HPC GPU node.
- `scripts/render_gpu.slurm`, `scripts/render_cpu_light.slurm`,
  `scripts/render_cpu_parallel.slurm` --- SLURM batch scripts covering all
  notebooks in three resource-appropriate buckets (GPU, light CPU, parallel
  CPU), plus `scripts/render_all.sh` to submit them as a dependency chain; see
  `docs/design/GPU_RENDERING_PROTOCOL.md`.
- `utils.derive_worker_seeds()` (JAX-based independent seeding for joblib
  workers) and `utils.format_ordinal()`; `schemas` module (Pydantic v2 input
  validation); `_paths.get_project_root()` for robust path resolution in
  notebooks; `require_tensors` decorator for uniform array-type validation.
- CI lint job (ruff + type-check) added to GitHub Actions; `CITATION.cff`,
  `CHANGELOG.md`, `CONTRIBUTING.md` provenance files.
- NTVMR single-/multi-epoch notebooks now run six dataset subsets (was one)
  with full per-subset diagnostics and a cross-subset comparison.

### Changed

- **Migrated the numerical/optimization core from PyTorch to JAX**: math
  kernels (`models/math_*`), the MLE optimizer layer (`optimistix`/`optax`
  replacing the PyTorch LBFGS closure), and Bayesian inference (NumPyro NUTS
  replacing Pyro) are now pure JAX, float64, and JIT-compiled. The MLE hot path
  is JIT-compiled end-to-end, giving a 65×–10,000× per-call speedup and a
  60×–280× end-to-end fitter speedup on multi-epoch fits; the extinction
  probability integral was replaced with a closed-form analytic expression
  (310× speedup, also autograd- and GPU-compatible). `torch`/`pyro-ppl` are no
  longer dependencies. Benchmarks and the still-open real-GPU verification item
  are recorded in `docs/design/JAX_MIGRATION_BENCHMARKS.md`.
- `models.math_populations.calculate_populations_log_pmf` is now a public
  helper (per-element log P(Z_T = n_i)); `calculate_populations_log_lik`
  delegates to it.
- Notebook type-checking switched from `pyright` to `basedpyright`.
- All `qmd_scripts/` notebooks renamed for a consistent `<dataset>_<analysis>.qmd`
  / `<nn>_<topic>.qmd` convention; all core-analysis notebooks use
  `get_project_root()` instead of hardcoded relative paths; notebook seeds
  unified to `NOTEBOOK_SEED`/`SIMULATION_SEED` constants.
- CI pytest step no longer uses `continue-on-error: true` (test failures now
  fail the build).

### Fixed

- `inference._optimizer_runners` --- L-BFGS-B `ftol` was divided by `1e-8`,
  causing scipy to stop after one iteration and report false convergence
  failures on large-NLL (e.g. six-epoch) problems.
- `models.math_extinction.calculate_log_lik_initial_lifetimes_given_extinction`
  --- a missing `as_array=True` on an internal call made the multi-epoch
  combined log-likelihood incorrect (golden value updated).
- `models.mathematical_derivatives.calculate_spine_ages_log_lik_derivatives`
  (pre-JAX-migration) --- interior branch used epoch end time `T` instead of
  manuscript age `t`, giving wrong gradients off epoch boundaries.
- `analysis.subset_inference` --- zero-variance date ranges no longer crash KDE
  resampling (falls back to `midYear`); per-epoch manuscript counts now use an
  inclusive upper bound so manuscripts dated exactly at the final year aren't
  dropped.
- `visualization.age_fit_plot.plot_creation_year_fit` --- skips the density
  overlay (with a warning) when all observed years are identical, instead of
  rendering it at the wrong scale; density grid no longer evaluates exactly at
  the spine's age (a point mass), which previously produced a spurious spike.
- Several notebooks indexed `BirthDeathProcess` results with DataFrame-style
  boolean masks; replaced with `.surviving_manuscripts`/`.extinct_manuscripts`.
- `BirthDeathProcess.equals()` returned a tensor; callers now get/expect `bool`.
- Simulator could produce ages slightly exceeding `T`; now filtered.

### Removed

- `BirthDeathProcess.__getitem__` (dict/DataFrame-style subscripting) --- use
  `.spine_ages`, `.non_spine_ages`, `.surviving_manuscripts`,
  `.extinct_manuscripts`, `.manuscript_records` instead.
- `BirthDeathProcess.__eq__`'s dict-comparison branch.
- `fit_birth_death_processes_bayesian()`'s `unbias` keyword --- pass
  `model_config=BayesianModelConfig(unbias=...)` instead.
- Dead code in `visualization.plotting` (`layout_config`, unused colour
  constants, never-called `apply_tufte_legend`), superseded by the Altair theme.

## [0.1.0] --- initial release
