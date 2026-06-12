# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `visualization.altair_theme` — new module beginning the Plotly→Altair migration
  (Phase 1). Registers and enables the `inf_bp_texts` Altair theme (ptolqual palette,
  serif font stack, muted `#FAFAFA` background, non-bold titles/labels, Tufte axes) and
  provides `save_chart`, the vl-convert PDF/SVG/PNG export chokepoint — pure and
  fail-loud, unlike the Plotly `save_figure`. Re-exported from `inf_bp_texts.visualization`.
  The 5000-row Altair guard is disabled (these are static publication figures).

### Changed

- `visualization.splitting_plots` (`plot_conditioned_trajectories`,
  `plot_empirical_population_mean`, `plot_level_survival_probabilities`) and
  `visualization.plotting` (`save_multi_epoch_convergence_figure`,
  `save_multi_epoch_fit_figure`) — migrated to Altair (Phase 2, the easy figures). The
  `plot_*` functions now return `alt.Chart`; the `save_*` functions render via Altair and
  export through `save_chart`. Call signatures are unchanged, so notebooks and the
  splitting test are unaffected.

### Removed

- `visualization.plotting` — deleted dead code: the unused `layout_config` dict, the
  seven unused `my_*` colour constants, and the never-called `apply_tufte_legend`
  (superseded by the Altair theme). No public call sites existed.

### Fixed

- `analysis.subset_inference.build_manuscript_subsets` — `_make()` now catches
  `ValueError` from the KDE resampler (triggered when a subset has zero-variance date
  ranges, e.g. Revelation with a handful of same-year manuscripts) and falls back to
  `midYear`, preventing a hard crash at notebook render time.
- `analysis.subset_inference.fit_manuscript_subset` — the per-epoch manuscript count
  (`epoch_n`) now uses an inclusive upper bound (`<=`) for the last epoch, so
  manuscripts dated exactly at the final year are no longer silently excluded.
- `visualization.age_fit_plot.plot_creation_year_fit` — when all observed creation
  years are identical the theoretical density overlay is now skipped entirely (with a
  warning) rather than rendered with an incorrect `bin_width = 1.0` scale factor that
  mis-scaled the curve by orders of magnitude.

### Added

- `analysis.subset_inference` — new module with `build_manuscript_subsets` (splits the
  cleaned NTVMR DataFrame into six canonical subsets with independent per-subset KDE
  resampling) and `fit_manuscript_subset` (builds a BirthDeathProcess, fits via MLE,
  and returns a `SubsetFitResult` dataclass with per-epoch rates, SEs, and 95% CIs).
  `ManuscriptSubset` and `SubsetFitResult` also re-exported from `inf_bp_texts.analysis`.
- `visualization.age_fit_plot.plot_creation_year_fit` — shared helper that renders
  observed creation-year histograms with the theoretical birth-time density overlay
  (replaces duplicated inline code across inference notebooks); re-exported from
  `inf_bp_texts.visualization`.
- NTVMR `ntvmr_single_epoch_inference.qmd` and `ntvmr_birth_death_inference.qmd` now
  run the analysis on **six dataset subsets**: non-lectionaries combined (groups 1–3,
  the new default), lectionaries (group 4), and four per-section non-lectionary analyses
  (Gospels e, Acts+Catholic Epistles a, Pauline p, Revelation r).  Each subset gets
  full per-subset diagnostics (convergence + age-fit figures) and the notebooks end with
  a cross-subset comparison (single-epoch: grouped bar + table; multi-epoch: 2-panel λ/μ
  line plot over epochs + birth-rate and death-rate summary tables).

- `analysis.date_density.resample_dates_from_ranges` — convenience wrapper over
  `quantile_resample_dates` that accepts lists/arrays and defaults `n_samples` to
  `len(early)`; re-exported from `inf_bp_texts.analysis`.
- `manuscript_data_exploration.qmd` — NTVMR subset analysis: filters `docID > 50000`
  (versional/named items), adds a `group` column (`docID // 10000`), stacks the
  Biblical Books histogram and the Manuscripts by Century chart by group
  (papyri/majuscules/minuscules/lectionaries), adds a non-lectionary (groups 1–3)
  date-range KDE plot, a lectionary (group 4) date-range KDE plot, and two 2×2
  faceted per-section date distribution figures (non-lectionary and lectionary) with
  100-year bins, shared legend, and shared axis titles.

- `analysis.date_density` — three new functions for manuscript date-range handling:
  `explode_date_ranges` (year-explosion + 1/n weights), `manuscript_date_density`
  (weighted boundary-corrected KDE in calendar-year space), and
  `quantile_resample_dates` (Hazen-position deterministic resampling).  All three
  are re-exported from `inf_bp_texts.analysis`.
- `data_cleaning.add_resampled_year_column` — populates a resampled-year column in
  manuscript DataFrames; now accepts keyword arguments `early_col`, `late_col`,
  `mid_col`, and `out_col` (all default to NTVMR names for backward compatibility)
  so it can be applied to any dataset with arbitrary column names.
- Kashouh families **jB** and **K** now use weighted date-range KDE `resampledYear`
  creation dates for inference (single-epoch, multi-epoch, and Bayesian notebooks);
  midpoint `parsed_year` is retained for plots and axes.  New KDE overlay figures
  (`kashouh_family_k_date_range_kde`, `kashouh_family_jb_date_range_kde`) added to
  the Kashouh data-exploration notebook.
- Kashouh multi-epoch Case 1-3 fits (`kashouh_multi_epoch_inference.qmd`) now pass
  `resampledYear` (KDE-resampled) rather than `parsed_year` (midyear) as
  `creation_times` via `prepare_multi_epoch_fit`, matching all other Kashouh and
  NTVMR inference notebooks.  The helper also clamps `start_time` to the century
  floor of the earliest resampled year to prevent creation-before-epoch-start errors.
- `boundary_corrected_kde_on_support` and `boundary_corrected_kde_at_zero` now accept
  an optional `weights` parameter (`analysis.validation_helpers`); unweighted behaviour
  is unchanged.
- Standard `pyright` added as a dev dependency for type-checking `.qmd` notebook
  cells via `lintquarto -l ruff pyright`; `pyrightconfig.json` suppresses torch-stub
  false positives (mirroring `[tool.basedpyright]`); `basedpyright` invocations
  updated to use `--project pyproject.toml` so the two type-checker configs remain
  independent; per-file ruff ignores added for notebook-specific rules
  (`D100`/`D101`/`D103`/`B018`/`B905`/`E501`/`I001`)

### Fixed

- Several notebooks used `sim[sim["death_time"].isna()]` DataFrame-style indexing on
  `BirthDeathProcess` objects returned by `simulate_birth_death_process()`; replaced
  with `.surviving_manuscripts` / `.extinct_manuscripts` properties
  (`validation/multi_epoch/02_single_fit_debug`, `06_sensitivity_benchmark`,
  `bayesian_multi_epoch/01–04`)
- `seed` undefined in `validation/single_epoch/03_optimisation_framework.qmd`;
  replaced with `NOTEBOOK_SEED`
- Possibly-unbound `birth_ses_*` / `death_ses_*` / `birth_cis_*` / `death_cis_*`
  variables in `core_analysis/03_multi_epoch/kashouh_multi_epoch_inference.qmd`
  when the fit fails; `None`-initialise in the `else` branch
- Possibly-unbound `*_constrained` posterior variables in
  `core_analysis/04_bayesian/kashouh_family_k_bayesian_inference.qmd`; initialise in
  the `else` branch and add `assert` guards in subsequent `if success:` blocks
- `dict[str, object]` type annotation on `section_analysis_dict` in
  `manuscript_data_exploration.qmd` caused a cascade of false type errors; removed
  the annotation so `dict[str, Any]` is inferred from the function return type
- `bins` passed as `ndarray[int_]` to `pd.cut()` in `kashouh_multi_epoch_inference.qmd`
  (pandas-stubs expect `list[float]`); converted via `[float(b) for b in bins]`
- `creation_times` passed as `list[Scalar]` to `BirthDeathProcess()` in
  `kashouh_multi_epoch_inference.qmd`; converted with `.astype(float).tolist()`
- Unused imports (`typing.Callable`, `typing.Any`, `torch`, `re`, `robust_y_max`,
  `get_selected_attempt`, duplicate `pathlib.Path`) removed across multiple notebooks
- Unused loop variables renamed to `_var` convention (`regime_index`, `time_index`,
  `strategy_idx`, `section_col`, `n_mss`, `start`, `end`) across multiple notebooks

- All 23 `qmd_scripts/` notebooks renamed: validation files drop their `methods_validation_[multi_epoch_][bayesian_]` prefix and gain a `<nn>_` ordering prefix; four `03_multi_epoch/` files made consistent with the `<dataset>_multi_epoch_inference.qmd` convention
- `schemas` module with Pydantic v2 models for validated input types
- `_paths.get_project_root()` helper (exported from the public API) for
  robust path resolution in Quarto notebooks using `QUARTO_PROJECT_DIR`
- `require_tensors` decorator for uniform tensor-type validation
- Logging via `logging.getLogger(__name__)` throughout the inference layer;
  silent `except` blocks in the optimizer loop now emit `logger.warning`
- Compile-time tracking cells added to remaining validation notebooks (G3)
- CI lint job (ruff + mypy) added to the GitHub Actions workflow
- `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md` provenance files

### Removed

- `BirthDeathProcess.__getitem__`: dict/DataFrame subscript interface removed;
  use `.spine_ages`, `.non_spine_ages`, `.surviving_manuscripts`,
  `.extinct_manuscripts`, and `.manuscript_records` properties instead
- `BirthDeathProcess.__eq__` dict branch: comparison against `{"spine": [...],
  "non_spine": [...]}` dicts no longer supported
- `fit_birth_death_processes_bayesian()` `unbias` keyword argument removed;
  pass `model_config=BayesianModelConfig(unbias=...)` instead

### Changed

- All Quarto core-analysis notebooks now use `get_project_root()` instead
  of hardcoded `../../../` relative paths
- Notebook seed constants unified to `NOTEBOOK_SEED` / `SIMULATION_SEED`
  named constants; global `np.random.seed()` calls removed
- Optimizer diagnostic helpers extracted to `_capture_grad_norm()` and
  `_capture_param_snapshot()`, eliminating duplicated code between the
  LBFGS and first-order optimizer runners
- CI pytest step no longer uses `continue-on-error: true`; test failures
  now fail the build

### Fixed

- Tensor dtype mismatch in regression likelihood tests
  (`torch.float64` now explicit on expected values)
- `np.trapz` replaced with `np.trapezoid` for NumPy 2.x compatibility
- `BirthDeathProcess.equals()` returns a tensor — callers now use `bool()`
- Simulator could produce ages > T; integration tests now filter them

## [0.1.0] — initial release
