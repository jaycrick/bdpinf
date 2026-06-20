# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- `inference._optimizer_runners` — L-BFGS-B `ftol` was divided by `1e-8`
  (yielding `ftol=10`), causing scipy to stop after a single iteration and
  triggering false convergence failures for large-NLL problems (e.g. six-epoch
  manuscript fits).  Fixed by passing `ftol` directly and accepting
  `opt_result.success` as a valid convergence signal alongside the gradient and
  absolute-NLL-change criteria.

### Added

- `inference.bayesian` — extended `BayesianPriorConfig` with per-family loc/scale
  (`birth_loc`, `birth_scale`, `death_loc`, `death_scale`); added `constraints:
  ParameterConstraints | None` and `device: str` to `BayesianModelConfig` so NUTS
  samples only free log-rates and expands them inside the model (statistically
  correct in-model constraints, replacing post-hoc sample projection); GPU/CUDA
  device support with a `num_chains > 1` warning; `mp_context="spawn"` for
  multi-chain runs to avoid PyTorch autograd + fork incompatibility.  Result dict
  adds `posterior_log_samples`, `posterior_log_samples_by_chain`, `n_epochs`.
- `qmd_scripts/core_analysis/04_bayesian/` — five Bayesian inference notebooks:
  `kashouh_family_k_bayesian_inference.qmd` (rewritten; in-model constraints, CPU/GPU
  benchmark), `kashouh_single_epoch_bayesian.qmd`, `ntvmr_single_epoch_bayesian.qmd`,
  `ntvmr_birth_death_bayesian.qmd` (six-epoch), `ntvmr_initial_population_bayesian.qmd`
  (N₀=150 sensitivity).
- `tests/test_regression_likelihoods.py` — multi-epoch (2-epoch) golden-value
  regression fixtures for spine ages, non-spine ages, population, and combined
  log-likelihood.
- `tests/test_bayesian_mle_consistency.py` — MLE-vs-posterior consistency baseline
  (marked `slow`).
- `inference.bayesian` — `mle_init_log_rates()` helper extracts free-parameter
  log-rates from an MLE result for use as NUTS warmup start point; added
  `init_log_rates` field to `BayesianModelConfig` and per-chain ±0.1 noise for
  multi-chain runs; wider default prior (`scale=4.0`); reduced `max_tree_depth=8`.
- `utils.format_ordinal` --- formats an integer with its ordinal suffix (`1st`,
  `2nd`, `3rd`, `4th`, `11th`-`13th`, ...); used for century x-axis labels in
  `manuscript_data_exploration.qmd`'s "Extant Manuscripts by Century" chart.
- `models.birth_death_process.BirthDeathProcess` --- constructing from
  `manuscript_records` now validates that every manuscript's `creation_time` falls
  strictly after its parent's `creation_time` and strictly before its parent's
  `death_time` (when set), raising `ValueError` on violation.

- `visualization.date_distribution_plots.plot_date_range_kde_comparison` --- new reusable
  Altair builder for the recurring date-range figure (midpoint histogram + weighted KDE
  curve + resampled-date histogram), re-exported from `inf_bp_texts.visualization`
  (Phase 5).
  Configurable `midpoint_label`, an `interaction` mode (`"none"`/`"filter"`/
  `"highlight"`) that binds a legend `selection_point` (`"filter"` shows only the clicked
  series, `"highlight"` dims the unselected ones; both HTML-only, PDFs keep the static
  default), and an `x_domain` that pins the bin extent and x scale so the axis does not
  rescale when a series is filtered out.
  Replaces inline per-notebook `go.Figure` copies.

- `visualization.altair_theme` --- the theme now draws faint horizontal (y-axis) gridlines
  (`config.axisY`, `#E8E8E8`) in a Tufte style; the x-axis stays gridless.

- Data-exploration notebooks migrated to Altair with HTML interactivity:
  `manuscript_data_exploration.qmd` (books/by-century stacked histograms with legend
  filtering and Lectionaries stacked on top; section-correlation **heatmap** via
  `mark_rect` + `mark_text`; date-range KDE charts with filter/highlight legends; 2×2
  faceted per-section distributions with y-axis zoom; age-vs-content scatter with a
  `gaNum` hover tooltip) and `kashouh_data_exploration.qmd` (parsed-date histogram that
  filters to the clicked source; FamilyK/FamilyJB legends filter to the clicked series).

- `visualization.altair_theme` --- new module beginning the Plotly→Altair migration
  (Phase 1).
  Registers and enables the `inf_bp_texts` Altair theme (ptolqual palette,
  serif font stack, muted `#FAFAFA` background, non-bold titles/labels, Tufte axes) and
  provides `save_chart`, the vl-convert PDF/SVG/PNG export chokepoint --- pure and
  fail-loud, unlike the Plotly `save_figure`.
  Re-exported from `inf_bp_texts.visualization`.
  The 5000-row Altair guard is disabled (these are static publication figures).

### Changed

- **Phase 3 — JIT optimizer + CUDA packaging**: MLE optimizer hot path is now
  JIT-compiled (`inference/fitting.py`, `inference/likelihood.py`,
  `inference/_optimizer_runners.py`).  A jittable NLL core
  (`_log_likelihood_jit_core`) bypasses host-side validation and `bool(traced)`
  guards; `jax.jit(jax.value_and_grad(...))` is compiled once per fit and reused
  across all LBFGS-B and Adam/SGD iterations, eliminating 320–870 ms per-step
  eager overhead for multi-epoch fits.  End-to-end fitter benchmark
  (`scripts/bench_jax_jit.py bench-fitter`): at 100 optimizer steps a 6-epoch
  fit completes in ~1.4 s (JIT) vs ~91 s (old eager) — a 63× end-to-end
  speedup; at 500 steps ~280×.  A `gpu` `[dependency-groups]` entry
  (`jax[cuda12]>=0.4.38; sys_platform=="linux"`) allows opt-in CUDA install
  without affecting the cross-platform CPU lockfile.  GPU verified attempted on
  A40-12Q vGPU — still blocked by XLA CUDA VMM incompatibility; documented in
  `docs/design/JAX_MIGRATION_BENCHMARKS.md`.

- **Post-JIT benchmark**: added `scripts/bench_jax_jit.py` measuring explicit
  `jax.jit` speedup over the eager path on the same (epochs × n_ages) grid as
  the Phase 2a torch comparison. JIT reduces per-call NLL time from ~8–346 ms
  (eager) to < 0.5 ms (warm JIT) — 65×–10,453× speedup depending on epoch count.
  Results recorded in `_output/jax_jit_benchmarks.json` and
  `docs/design/JAX_MIGRATION_BENCHMARKS.md`.

- **Notebook torch removal**: migrated all default-render and Bayesian notebooks
  from `torch.tensor`/`torch.as_tensor` calls to JAX/NumPy equivalents; removed
  `.detach().cpu().numpy()` chains from `confidence_intervals_from_log_covariance`
  and `calculate_unknown_spine_mixture_birth_density` call sites; replaced
  `BirthDeathProcess.manuscript_ages.numpy()` with `np.asarray(...)`. All
  notebooks in `qmd_scripts/` and `docs/` now lint-clean without `torch`
  installed; `kashouh_family_k_bayesian_inference.qmd` GPU detection switched
  from `torch.cuda.is_available()` to `jax.devices()`.

- **JAX_PLATFORMS CPU default in `__init__.py`**: `os.environ.setdefault("JAX_PLATFORMS", "cpu")`
  is now set before `import jax` so that fresh subprocess workers spawned by
  `joblib.Parallel` do not trigger a fatal CUDA initialization error when
  `jax[cuda12]` is installed but the GPU is unavailable (e.g. on the A40-12Q
  vGPU with XLA VMM restrictions).  Explicit `JAX_PLATFORMS=cuda` set by the
  caller before import is still honoured.

- **`models.math_populations.calculate_populations_log_pmf`**: new public helper
  returns per-element log P(Z_T = n_i) as a shape-(m,) array instead of the sum.
  `calculate_populations_log_lik` delegates to it for the standard
  (`initial_population=1, unbias=False`) path, eliminating the code duplication.
  Notebooks' population-PMF loops replaced with a single vectorised call — one JAX
  dispatch instead of N per-support-point dispatches.

- **Validation lazy error messages** (`models.validation`): error-message f-strings
  in `_validate_times_positive` and `_validate_ages_in_range` are now computed
  inside the failure branch, not at the call site.  Avoids materialising array
  values into strings on the happy path.

- **Removed stale torch detection** (`utils.ensure_1d_tensor`, `utils.json_safe`):
  dropped `hasattr(x, "detach")` guard and the `torch.Tensor` branch — torch is
  no longer a dependency.

- **Phase 2e — cutover & cleanup**: removed `torch`, `torchvision`, `pyro-ppl` from
  project dependencies and `types-torch` from dev; promoted JAX stack (`jax`,
  `jaxtyping`, `numpyro`, `optax`, `optimistix`) to main dependencies; removed the
  transient `models._jax` spike package and `tests/test_jax_torch_equivalence.py`
  equivalence gate; all remaining `torch` usages in test files replaced with JAX/numpy
  equivalents; updated `CLAUDE.md` design note from "PyTorch everywhere" to "JAX
  everywhere"; basedpyright reaches 0 errors (fixed 6 pre-existing issues surfaced
  by the type-check pass); `models.mathematical_derivatives` removed from `_quarto.yml`
  API reference (module was deleted in Phase 2b.4); `scripts/bench_jax_vs_torch.py`
  retired (benchmark data archived in `docs/design/JAX_MIGRATION_BENCHMARKS.md`).

- `inference.fitting`, `inference._optimizer_runners`, `inference._fit_types`,
  `inference.uncertainty` — Phase 2c: optimizer layer migrated from PyTorch
  (LBFGS closure / `nn.Parameter`) to JAX + optimistix BFGS/LBFGS and optax Adam;
  `estimate_mle_covariance` uses `jax.hessian`; result-dict schema and public API
  are unchanged.
- `inference.bayesian` — Phase 2d: fully migrated from Pyro/PyTorch to
  NumPyro/JAX.  `BayesianLikelihoodModel` calls JAX math functions directly
  (`calculate_populations_log_lik`, `calculate_non_spine_ages_log_lik`,
  `calculate_spine_ages_log_lik`, all with `_skip_normalization=True`) to avoid
  `ConcretizationTypeError` inside NumPyro's JIT-compiled `lax.while_loop`.
  Model data stored as NumPy arrays (not JAX arrays) to remain concrete in the
  NUTS trace.  `chain_method="vectorized"` (JAX vmap) replaces Pyro's subprocess
  multi-chain.  The `device` parameter is accepted for API compatibility but
  ignored (JAX uses its default backend).  `BayesianPriorConfig.resolve_tensors`
  now returns NumPy arrays instead of `torch.Tensor`.

- `models.math_extinction` — replaced `scipy.integrate.quad` loop in
  `calculate_log_pr_extinction_given_initial_lifetimes` with a closed-form
  analytic expression derived from the Poisson generating functional; fully
  vectorised over lifetimes, autograd-compatible, and device-agnostic (CPU/GPU).
  Performance: 433 ms/call → 1.4 ms/call (310× speedup); enables practical NUTS
  sampling for multi-epoch models.  The `_integrand_surv_from_weighted_birth_time`
  helper is retained for testing.
- `kashouh_data_exploration.qmd` --- renamed the `source` field/legend/column to
  `Family` throughout (coverage table, parsed-date-distribution chart), and shortened
  its values from `"Family jB"`/`"Family K"` to `"jB"`/`"k"` (kept `"other"` for the
  remaining ALAGM-5 classifications).
- `ntvmr_single_epoch_inference.qmd` and `ntvmr_birth_death_inference.qmd` --- the
  Lectionary (group 4) subset is now fit on an epoch grid starting at **380 CE**
  instead of 100 CE (the genre only emerges in the late 4th century), still ending by
  2000 CE; KDE-resampled lectionary dates before 380 are clipped to the new start.
- `analysis.subset_inference.build_manuscript_subsets` --- dropped the `(1–3)`/`(4)`
  group-number suffixes from the `"Non-Lectionary"`/`"Lectionary"` subset labels (the
  per-section labels `"Gospels (e)"`, `"Acts + Catholic Epistles (a)"`,
  `"Pauline Epistles (p)"`, `"Revelation (r)"` are unchanged); this label flows into
  every print statement, chart title, and table column in the two NTVMR inference
  notebooks.

- `visualization.splitting_plots` (`plot_conditioned_trajectories`,
  `plot_empirical_population_mean`, `plot_level_survival_probabilities`) and
  `visualization.plotting` (`save_multi_epoch_convergence_figure`,
  `save_multi_epoch_fit_figure`) --- migrated to Altair (Phase 2, the easy figures).
  The
  `plot_*` functions now return `alt.Chart`; the `save_*` functions render via Altair and
  export through `save_chart`.
  Call signatures are unchanged, so notebooks and the
  splitting test are unaffected.

- `visualization.parameter_plots` (`create_kde_density_plot`,
  `create_parameter_box_plot`) and `visualization.age_fit_plot`
  (`plot_creation_year_fit`) --- migrated to Altair (Phase 3, moderate figures), now
  returning `alt.LayerChart`.
  The KDE arrow annotation becomes a dashed `mark_rule` +
  `mark_text` label; the box plot drops the redundant median dot (the box already draws
  the median line) and reproduces `boxmean` as an explicit mean marker.
  The
  observed-creation-year histogram + density logic is consolidated into a single shared
  `visualization.age_fit_plot.creation_year_fit_chart` builder reused by both
  `plot_creation_year_fit` and `save_multi_epoch_birth_time_figure` (removing duplicated
  density-scaling code).
  The two NTVMR inference notebooks switch the age-fit figure to
  `display(...)` + `save_chart(...)`.

- `visualization.genealogy.plot_genealogy_tree` and
  `models.BirthDeathProcess.plot_genealogy_tree` --- migrated to Altair (Phase 4), now
  returning `alt.LayerChart` instead of a Plotly figure.
  Lifetimes are horizontal
  `mark_rule` segments colour/size-coded by lineage (extinct / surviving / spine) on a
  per-row ordinal axis labelled by `manuscript_id`; parent→child connectors are dashed
  vertical `mark_rule` segments.
  `BirthDeathProcess` no longer imports Plotly.

- `visualization.plotting.plot_parameter_consistency` --- migrated to Altair (Phase 4),
  now returning an `alt.FacetChart` faceted by regime (rows) × statistic (columns) with
  `mark_area` bootstrap CI bands, log-scaled `T`, and dashed true-value rules.
  The
  bootstrap statistics are extracted into a new pure, testable
  `visualization.plotting.compute_consistency_bands` returning a tidy long-form frame
  (regime, T, statistic, estimate, lower, upper, true_value), replacing the previous
  \~270-line hand-built 36-trace subplot grid.
  `visualization.plotting` no longer imports
  Plotly figure constructors.
  The asymptotic-consistency notebook switches to
  `display(...)`.

- Notebook type-checking switched from `pyright` to `basedpyright` via
  `lintquarto -l ruff basedpyright`; `lintquarto` bumped to ≥0.13.1; the standalone
  `pyright` dev dependency and `pyrightconfig.json` have been removed.

### Removed

- `visualization.plotting` --- deleted dead code: the unused `layout_config` dict, the
  seven unused `my_*` colour constants, and the never-called `apply_tufte_legend`
  (superseded by the Altair theme).
  No public call sites existed.

### Fixed

- `models.math_extinction.calculate_log_lik_initial_lifetimes_given_extinction` —
  internal call to `calculate_log_pr_extinction_given_initial_lifetimes` was missing
  `as_array=True`, so the `return_array=True` path returned a scalar SUM of
  log-probabilities rather than a per-lifetime array. This caused the multi-epoch
  combined log-likelihood (used in `inference.likelihood.log_likelihood`) to be
  incorrect; golden value updated from −178.5901579681 to −177.7867323299.
- `models.mathematical_derivatives.calculate_spine_ages_log_lik_derivatives` —
  the interior (non-boundary) branch used `T` (epoch end time) instead of the
  manuscript age `t` in the formulas for `B_t`, `exp(-b*T)`, and `exp(-d*T)`, giving
  incorrect gradient values for any age that does not lie exactly at the epoch boundary.

- `analysis.subset_inference.build_manuscript_subsets` --- `_make()` now catches
  `ValueError` from the KDE resampler (triggered when a subset has zero-variance date
  ranges, e.g. Revelation with a handful of same-year manuscripts) and falls back to
  `midYear`, preventing a hard crash at notebook render time.
- `analysis.subset_inference.fit_manuscript_subset` --- the per-epoch manuscript count
  (`epoch_n`) now uses an inclusive upper bound (`<=`) for the last epoch, so
  manuscripts dated exactly at the final year are no longer silently excluded.
- `visualization.age_fit_plot.plot_creation_year_fit` --- when all observed creation
  years are identical the theoretical density overlay is now skipped entirely (with a
  warning) rather than rendered with an incorrect `bin_width = 1.0` scale factor that
  mis-scaled the curve by orders of magnitude.
- `visualization.age_fit_plot.plot_creation_year_fit` --- the density grid no longer
  evaluates exactly at `age == process_duration` (the spine manuscript's age is
  deterministically the full duration, a point mass rather than a continuous density
  value); the grid's upper bound is nudged just inside the domain instead, preventing a
  spurious spike from dominating the overlay's scale.

### Added

- `analysis.subset_inference` --- new module with `build_manuscript_subsets` (splits the
  cleaned NTVMR DataFrame into six canonical subsets with independent per-subset KDE
  resampling) and `fit_manuscript_subset` (builds a BirthDeathProcess, fits via MLE,
  and returns a `SubsetFitResult` dataclass with per-epoch rates, SEs, and 95% CIs).
  `ManuscriptSubset` and `SubsetFitResult` also re-exported from `inf_bp_texts.analysis`.

- `visualization.age_fit_plot.plot_creation_year_fit` --- shared helper that renders
  observed creation-year histograms with the theoretical birth-time density overlay
  (replaces duplicated inline code across inference notebooks); re-exported from
  `inf_bp_texts.visualization`.

- NTVMR `ntvmr_single_epoch_inference.qmd` and `ntvmr_birth_death_inference.qmd` now
  run the analysis on **six dataset subsets**: non-lectionaries combined (groups 1--3,
  the new default), lectionaries (group 4), and four per-section non-lectionary analyses
  (Gospels e, Acts+Catholic Epistles a, Pauline p, Revelation r).
  Each subset gets
  full per-subset diagnostics (convergence + age-fit figures) and the notebooks end with
  a cross-subset comparison (single-epoch: grouped bar + table; multi-epoch: 2-panel λ/μ
  line plot over epochs + birth-rate and death-rate summary tables).

- `analysis.date_density.resample_dates_from_ranges` --- convenience wrapper over
  `quantile_resample_dates` that accepts lists/arrays and defaults `n_samples` to
  `len(early)`; re-exported from `inf_bp_texts.analysis`.

- `manuscript_data_exploration.qmd` --- NTVMR subset analysis: filters `docID > 50000`
  (versional/named items), adds a `group` column (`docID // 10000`), stacks the
  Biblical Books histogram and the Manuscripts by Century chart by group
  (papyri/majuscules/minuscules/lectionaries), adds a non-lectionary (groups 1--3)
  date-range KDE plot, a lectionary (group 4) date-range KDE plot, and two 2×2
  faceted per-section date distribution figures (non-lectionary and lectionary) with
  100-year bins, shared legend, and shared axis titles.

- `analysis.date_density` --- three new functions for manuscript date-range handling:
  `explode_date_ranges` (year-explosion + 1/n weights), `manuscript_date_density`
  (weighted boundary-corrected KDE in calendar-year space), and
  `quantile_resample_dates` (Hazen-position deterministic resampling).
  All three
  are re-exported from `inf_bp_texts.analysis`.

- `data_cleaning.add_resampled_year_column` --- populates a resampled-year column in
  manuscript DataFrames; now accepts keyword arguments `early_col`, `late_col`,
  `mid_col`, and `out_col` (all default to NTVMR names for backward compatibility)
  so it can be applied to any dataset with arbitrary column names.

- Kashouh families **jB** and **K** now use weighted date-range KDE `resampledYear`
  creation dates for inference (single-epoch, multi-epoch, and Bayesian notebooks);
  midpoint `parsed_year` is retained for plots and axes.
  New KDE overlay figures
  (`kashouh_family_k_date_range_kde`, `kashouh_family_jb_date_range_kde`) added to
  the Kashouh data-exploration notebook.

- Kashouh multi-epoch Case 1-3 fits (`kashouh_multi_epoch_inference.qmd`) now pass
  `resampledYear` (KDE-resampled) rather than `parsed_year` (midyear) as
  `creation_times` via `prepare_multi_epoch_fit`, matching all other Kashouh and
  NTVMR inference notebooks.
  The helper also clamps `start_time` to the century
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
- `BirthDeathProcess.__eq__` dict branch: comparison against `{"spine": [...],   "non_spine": [...]}` dicts no longer supported
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
- `BirthDeathProcess.equals()` returns a tensor --- callers now use `bool()`
- Simulator could produce ages > T; integration tests now filter them

## [0.1.0] --- initial release
