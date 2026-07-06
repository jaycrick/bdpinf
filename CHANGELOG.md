# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `scripts/resume_render_all.sh`: resumes the `render_all.sh` job chain when
  `generate_recovery_studies.slurm` has already completed, and resumes
  `generate_bootstrap_kde_samples.slurm` at the specific subgroups it did not
  reach before a prior run timed out.

### Fixed

- `scripts/generate_bootstrap_kde_samples.slurm`: raised `--time` from 2h to
  5h (the job was timing out partway through the 7-subgroup sweep at \~2000s/
  subgroup) and right-sized `--cpus-per-task`/`--mem-per-cpu` down to match
  observed utilization (single-threaded, \~5.5 GB peak).

- `scripts/generate_recovery_studies.slurm`, `scripts/render_cpu_light.slurm`,
  `scripts/render_cpu_parallel.slurm`: right-sized `--cpus-per-task`,
  `--mem-per-cpu`, and `--time` to match observed CPU/memory utilization from
  completed runs, freeing up shared cluster resources.

- `inf_bp_texts.data.data_cleaning.run_cleaning_pipeline`: single canonical
  entry point for producing `manuscript_analysis_cleaned.csv`, now the only
  code path `inf-bp-texts ntvmr clean` uses.
  Composes a docID-based group
  filter, explicit removal of "merged" manuscripts (both `earlyYear` and
  `lateYear` are 0 --- see `identify_merged_manuscripts`, confirmed examples
  P14/P58/P67/074/081), the existing XML-based date cleaning, manual
  corrections loaded from the new version-controlled
  `data/manual_corrections.yaml` (via `load_manual_corrections`, replacing
  the two previously duplicated hardcoded correction dicts), and
  `fill_orphaned_book_sections` (sets a manuscript's section flag when a book
  from that section is present but the section itself is unmarked), and
  `clear_lectionary_sections` (lectionaries, `group == 4`, never carry
  `section_e`/`a`/`p`/`r` --- only `section_K`/`section_l` apply to them;
  this corrects both raw content-overview mis-parses and any orphaned-book
  fill that would otherwise apply to a lectionary's excerpted reading).
  `manuscript_data_exploration.qmd` no longer computes or writes the cleaned
  CSV; it is now a read-only review surface over the CLI's output.

- `qmd_scripts/core_analysis/04_bayesian/ntvmr_birth_death_bayesian.qmd`:
  new "All-Subsets Rate Comparison (Bayesian Medians)" section --- a
  multi-series `mark_line(point=True)` chart (one line per subset, epoch on
  the x-axis) following the posterior median at each epoch, mirroring the
  non-Bayesian point-estimate comparison in the multi-epoch MLE notebook;
  saved as `04a_subset_comparison_{birth,death,halflife}_median`.

- `qmd_scripts/core_analysis/04_bayesian/ntvmr_birth_death_bayesian.qmd`:
  "Rates by Epoch" birth-rate panels now also save a variant excluding the
  Modern Era epoch (`04a_rates_<subset>_birth_no_modern`), so the
  earlier-epoch spread isn't compressed by the widest posterior.
  "Rates by Tradition (per epoch)" birth/death/half-life panels likewise
  gain a genre-only variant excluding the Non-Lectionary and Lectionary
  whole-corpus splits (`04a_trad_<epoch>_{birth,death,halflife}_no_corpus`),
  so Gospels/Acts/Pauline/Revelation aren't dwarfed by the larger corpora.
  The "All-Subsets Rate Comparison (Bayesian Medians)" birth/death/half-life
  line charts gain the same genre-only variant
  (`04a_subset_comparison_{birth,death,halflife}_median_no_corpus`), and all
  six panels (full-subset and genre-only) now always show a legend, split
  into two columns via `alt.Legend(columns=2)` so entries wrap onto multiple
  rows instead of one long row.

- `inference.persistence.save_recovery_results`/`load_recovery_results`:
  schema-agnostic `.npz` round-trip for tidy per-replicate recovery-study
  tables, alongside the existing posterior-sample persistence.
  New
  `analysis.single_epoch_recovery` and `analysis.multi_epoch_recovery`
  modules factor the simulate-then-fit replicate logic out of the five
  heaviest `qmd_scripts/validation` notebooks, and
  `scripts/generate_recovery_studies.py` (+ `scripts/generate_recovery_studies.slurm`)
  precomputes them once instead of on every render --- see
  `docs/design/RECOVERY_STUDY_DATA_PIPELINE.md`.

- `visualization.altair_theme.save_chart` gains a `keep_title` keyword
  (default `False`): saved PDFs now have their title stripped by default via
  the new `_untitled()` helper, since figures are captioned separately in
  notebook prose; pass `keep_title=True` to preserve the baked-in title.
  Handles single, layered, concat, and facet charts.

- `visualization.bayesian_plots.posterior_rates_chart` gains a
  `distribution_label` keyword (default `"Posterior distribution"`) and now
  attaches a top-oriented legend ("Posterior distribution" / "95% CrI") to
  the ridge area and CI-bar layers; `prior_rates_chart` overrides it to
  `"Prior distribution"`.
  Per-row facet height is now computed from the
  category count so the exported figure lands near the project's usual
  `300x200`, instead of ignoring `save_chart`'s `height=` (which facets can't
  honour directly).

### Fixed

- `visualization.bayesian_plots._panel_rows`/`_ridgeline_layers` and
  `_prior_posterior_rows`/`_prior_posterior_layers` --- ridge areas are now
  binned and peak-normalized in numpy (`_histogram_ridge_rows`) instead of
  via Vega-Lite's in-spec `transform_bin`/`transform_aggregate`/
  `transform_impute` pipeline, so every ridge peaks at `1.0` regardless of
  draw count or spread; the shared facet y-scale domain is fixed to
  `[0, 1]`.
  Fixes prior/posterior and cross-epoch/cross-subset ridge plots
  rendering with wildly mismatched or flattened areas.

- `visualization.bayesian_plots._ridgeline_layers` --- the "95% CrI" legend
  entry rendered as a filled square (indistinguishable from the
  distribution legend) because it shared the `opacity` channel with the
  ridge area, and Vega-Lite merges same-channel legends across layers,
  keeping only one symbol.
  The median point and CI bar now use their own
  `shape`/`color` channels respectively, so the distribution, median (filled
  circle), and 95% CrI (black line) legends render as three distinct
  entries.
  Affects all `04a_rates_*`/`04a_halflife_*`/`04a_trad_*` figures
  built via `posterior_rates_chart`.

- `qmd_scripts/validation/single_epoch/06_bulk_inference.qmd` --- the joint
  λ/μ rate heatmap (`vse06_rate_heatmap_*`) dropped its custom
  `_anchored_bin()` helper in favour of Altair's `alt.Bin(maxbins=25)`,
  `_heatmap_subset()` now returns arrays instead of a dict, and the x-axis
  domain starts at the filtered minimum instead of 0 (y stays anchored at
  0).
  The death/birth-rate non-boundary histograms
  (`vse06_{birth,death}_rate_histogram_non_boundary`) now show gridlines
  only at round-number major ticks (`_major_gridline_values`) instead of one
  per histogram bin.

- `t₁/₂ (years)` / `t₁₂ (years)` half-life axis titles and panel labels
  (multi-epoch MLE and Bayesian notebooks) replaced with the single Unicode
  character `t½` for consistent, correctly-kerned rendering.

- `visualization.bayesian_plots._ridgeline_layers`/`_prior_posterior_layers`
  --- the median-point and CI-bar x-encodings left `title` unset, so
  layering them with the ridge area (whose x carries the real axis title)
  made Vega-Lite comma-join the three layers' distinct field names into an
  axis title like `"Rate per year, rate, ci_low"`; both now repeat the
  area's title.

- `visualization.age_fit_plot.creation_year_fit_chart` --- replaced the
  scaled continuous density-curve overlay with a true expected-count
  histogram sharing the observed histogram's bin edges (density evaluated at
  each bin midpoint × bin width × sample size), so the figure reads as a
  layered histogram ("Observed" vs "Expected") instead of a bar+line mix;
  the shared x-domain now extends to cover the fitted distribution's range
  even where it reaches past the observed data.
  Used by all `02a_age_fit_*`,
  `02b_*_ages`, and the multi-epoch birth-time saver.

- `visualization.age_fit_plot.creation_year_fit_chart` --- both histogram
  layers were missing `bin="binned"` on their `x` encoding, so Vega-Lite
  rendered each bin as a small floating rectangle instead of a true bar
  reaching the zero baseline; opacity dropped to `0.3` with `binSpacing=0`
  to match the project's other overlaid-histogram figures.
  Affects the
  same `02a_age_fit_*`/`02b_*_ages`/birth-time figures as above.

- `visualization.plotting.rate_ci_panel` --- the rate axis now always shows
  gridlines, and log-scale panels (death rates, half-lives) restrict ticks
  to clean powers of ten instead of Vega-Lite's default log ticks, which
  could overlap or omit decades entirely across several orders of magnitude.

- `visualization.correlation_plots.posterior_rate_correlation` and
  `correlation_heatmap_chart`: births-then-deaths correlation matrix of
  per-epoch posterior draws, rendered as a diverging heatmap (colourblind-
  friendly RdBu palette, `DIVERGING_RDBU` in `altair_theme`, centred at zero)
  with the correlation value overlaid on each cell.
  Wired into all four
  `04_bayesian` notebooks; the pre-existing `01a_section_correlation_heatmap`
  in `manuscript_data_exploration.qmd` now goes through the same builder for a
  consistent house style instead of an inline Vega `redblue` scheme.

- `visualization.bayesian_plots.prior_posterior_rates_chart` and
  `PriorPosteriorRatePanel`: ridgeline overlay of prior (dashed outline) vs.
  posterior (filled), one row per category --- unlike `posterior_rates_chart`,
  prior and posterior share one row-facet cell (no column facet) so they can
  be compared directly at a glance.
  Added to `ntvmr_birth_death_bayesian.qmd`
  under both existing posterior tabsets ("Rates by Epoch (per tradition)" and
  "Rates by Tradition (per epoch)"), split into birth/death/half-life; each
  row uses that subset's own actual fitted prior (six-era vs.
  lectionary-durable), looked up from `bayesian_configs`.

- `visualization.bayesian_plots.prior_rates_chart` and
  `prior_rate_samples`: ridgeline (joyplot) of per-epoch prior rate
  distributions, in the same style as `posterior_rates_chart`.
  Renders
  deterministic quantile samples (Hazen plotting positions) of the log-normal
  birth/death priors rather than random draws; `prior_rate_samples` exposes
  the raw birth/death sample arrays so callers can split birth, death, and
  half-life into separate figures.
  Wired into all four `04_bayesian`
  notebooks (`ntvmr_birth_death_bayesian`, `ntvmr_single_epoch_bayesian`,
  `kashouh_single_epoch_bayesian`, `kashouh_family_k_bayesian_inference`).
  In `ntvmr_birth_death_bayesian.qmd`, both the prior-rate ridgelines and the
  existing posterior rate ridgelines (by tradition and by epoch) are now
  split into separate birth/death/half-life figures, matching the layout of
  `ntvmr_birth_death_inference.qmd`'s MLE rate plots.

- `manuscript_survival_proportion.qmd` rewritten to use a
  **Horvitz-Thompson (HT) inverse-probability-weighting estimator** in place of
  the multilevel-splitting conditioned sampler.
  For each surviving manuscript
  $i$ born at year $y_i$, the survival probability
  $P_i = \exp(-\int_{y_i}^T \mu\,\mathrm{d}t)$ is computed analytically from
  the posterior death-rate schedule; total manuscripts ever created is estimated
  as $\hat N = \sum 1/P_i$ and the proportion alive as $n / \hat N$.
  Posterior
  uncertainty is propagated over all 4,000 NUTS draws; date uncertainty is
  propagated over 2,000 bootstrap-resampled creation years.
  Also fixes a data
  loading bug: the subset filter now uses `group.isin([1,2,3])` (matching
  `build_manuscript_subsets`) rather than `section_l != 1`.

- `analysis.bootstrap_date_density.bootstrap_epoch_counts` /
  `bootstrap_target_ranges`: bucket bootstrap resampled years into epochs and
  return the 2.5th--97.5th percentile count interval as `[lo, hi]` windows
  (last era always unconditioned); used by `docs/date_uncertainty.qmd`.

- `analysis.bootstrap_date_density`: new module for bootstrap KDE-quantile
  date resampling.
  Adds `bootstrap_quantile_resample_dates` (vectorized
  reimplementation of the weighted boundary-corrected KDE + Hazen-quantile
  pipeline from `quantile_resample_dates`, batched over 100,000 bootstrap
  draws without calling scipy in a loop), `save_bootstrap_samples` /
  `load_bootstrap_samples` (compressed `.npz` persistence of the
  `[n_bootstrap, n]` `doc_id_samples` / `resampled_years` arrays), and the
  internal `_build_weight_matrix` / `_resample_batch` helpers.
  The identity
  multiplicity exactly reproduces `quantile_resample_dates`.

- `scripts/generate_bootstrap_kde_samples.py`: Typer CLI that runs
  `bootstrap_quantile_resample_dates` over the seven standard manuscript
  subgroups (full, non_lectionary, lectionary, section_e/a/p/r) and writes
  the results to `bootstrap_samples/<subgroup>.npz` at the project root.

- `docs/date_uncertainty.qmd`: new documentation page covering the full
  manuscript date uncertainty pipeline --- year explosion, boundary-corrected
  weighted KDE, Hazen-quantile inversion, and the bootstrap vectorized
  reimplementation with mathematical derivations; added to the Docs navbar
  in `_quarto.yml`.

- README: new "Manuscript date uncertainty" section explaining the
  deterministic KDE-quantile pipeline and bootstrap extension, with code
  snippets for generation and loading.

- `.gitignore`: `bootstrap_samples/` (large generated artifacts, regenerate
  via the CLI script).

- `simulation.survival_conditioning`: new module for conditioned survival
  estimation via multilevel splitting.
  Adds `_simulate_era_core` (JIT-compiled
  single-era event loop resumable from an arbitrary alive-population state),
  `sample_conditioned_survival_process` (filter + clone + advance per era to
  produce matched trajectories), `sample_conditioned_survival_per_posterior`
  (one matched trajectory per Bayesian posterior draw, returns a tidy DataFrame),
  `build_target_ranges` (observed survivor counts → inclusive per-era bounds with
  optional tolerance), and result dataclasses `MatchedSurvivalSample` /
  `ConditionedSurvivalResult` / `EraConditioningDiagnostic`.

- `inference.persistence`: `save_posterior_samples` / `load_posterior_samples`
  --- thin `.npz` wrappers for persisting Bayesian posterior rate samples to disk
  so downstream notebooks can skip MCMC re-runs.

- `qmd_scripts/core_analysis/04_bayesian/ntvmr_birth_death_bayesian.qmd`: added
  a "Save Posterior Samples" cell that writes each subset's posterior draws to
  `posterior_samples/{subset}.npz` after the MCMC completes.

- `qmd_scripts/core_analysis/03_multi_epoch/manuscript_survival_proportion.qmd`:
  new notebook estimating the proportion of NT manuscripts currently alive vs
  ever created, with per-era breakdown and posterior uncertainty; loads
  `posterior_samples/{subset}.npz` and runs the conditioned sampler.

- `simulation.simulate_birth_death_lifespans`: new JAX/JIT-compiled birth-death
  simulator using the lifespan-at-birth algorithm.
  Each individual's death time is
  pre-sampled at birth from a piecewise-exponential distribution; the event loop runs
  inside `jax.lax.while_loop`.
  API mirrors `simulate_birth_death_process`; the
  returned `BirthDeathProcess.manuscript_records` includes a `birth_epoch` column.

- `simulation.simulate_birth_death_lifespans_arrays`: raw-array entry point (no
  DataFrame overhead) that exposes per-individual `birth_times`, `death_times`,
  `birth_epochs`, and `survived` flags --- the hook needed by a future multilevel-splitting
  estimator to count epoch-specific survivors.

- `simulation.LifespanSimulationArrays`: `NamedTuple` returned by the above.

- `qmd_scripts/validation/single_epoch/01_simulation_checks.qmd` and
  `qmd_scripts/validation/multi_epoch/01_simulation_checks.qmd`: lifespan-at-birth
  simulator added as a third comparator alongside direct sampling and the Gillespie
  full-process; includes population histogram, spine/non-spine KDE panels, and
  Anderson-Darling/KS goodness-of-fit tests against both existing methods.

### Changed

- `qmd_scripts/core_analysis/03_multi_epoch/ntvmr_birth_death_inference.qmd`:
  death-rate axis title corrected from `μ (loss probability/year)` to `μ` (a
  birth-death rate is not bounded by 1 and is not a probability per year).

- `visualization.bayesian_plots.posterior_rates_chart`: gains a `log_scale` keyword
  argument; when `True`, draws are binned in log10 space and both the area ridge
  and the median/CI overlays are rendered on a log-scale x-axis.
  All existing call
  sites default to `log_scale=False` so behaviour is unchanged.

- `visualization.altair_theme.save_chart` / `save_chart_html`: `output_dir` now
  defaults to `get_project_root() / "figures"` (was cwd-relative `"figures"`), so
  figures no longer scatter into per-notebook subdirectories when run via
  `quarto preview` or individually.

- All `qmd_scripts/` notebooks: `figures_dir` now resolves to a mirrored subpath
  under the repo-root `figures/` tree (e.g. `figures/core_analysis/04_bayesian/`),
  and all `save_chart` / `save_chart_html` / `save_multi_epoch_*` calls thread
  `output_dir=figures_dir`.
  Figure stems are shortened to a `<prefix>_<descriptive>`
  pattern (e.g. `04b_prior_posterior.pdf`) with the prefix encoding the source
  notebook; this resolves filename collisions that existed across sibling notebooks.

- `visualization.bayesian_plots.posterior_rates_chart` now renders overlapping
  ridgeline (joyplot) plots with correct per-cell height (`step = 20` px) and
  in-spec Vega-Lite binned density (`transform_bin` → `transform_aggregate` →
  `transform_impute`) replacing the Python-side boundary-corrected KDE; the
  posterior median is shown as a `mark_point` and the equal-tailed 95% credible
  interval as a ticked `mark_errorbar` along each row's baseline; `RatePanel`
  carries raw posterior `samples` (shape `n_draws × n_categories`) instead of
  pre-computed summary arrays.
  All four `04_bayesian/` notebooks updated.

- `visualization.bayesian_plots.trace_chart` now returns `list[alt.Chart]`
  (one independent chart per `TracePanel`) instead of a single `alt.ConcatChart`
  grid; the y-axis no longer anchors at zero (`scale=alt.Scale(zero=False)`)
  so trace variation fills the panel.
  All four `04_bayesian/` notebooks updated
  to loop over the list and arrange panels via Quarto `# | layout-ncol: 2`
  (simple cells) or `::: {layout-ncol=2}` fenced divs (panel-tabset loops).

### Added

- `analysis.latex_tables`: new module providing a generic booktabs + siunitx
  LaTeX table renderer (`rates_latex_table`) and four assembly helpers
  (`subset_fits_latex_by_epoch`, `subset_fits_latex_by_subset`,
  `bayesian_results_latex_by_epoch`, `bayesian_results_latex_by_subset`) that
  emit `\multirow` parameter groups, `S[table-format=…]` numeric columns,
  and `\num{}`-bracketed CI/CrI columns.
  All names re-exported from
  `inf_bp_texts.analysis`.

- `analysis.latex_tables`: four new half-life table helpers
  (`subset_fits_halflife_latex_by_epoch`, `subset_fits_halflife_latex_by_subset`,
  `bayesian_results_halflife_latex_by_epoch`, `bayesian_results_halflife_latex_by_subset`)
  that append a second $t_{1/2} = \ln 2 / \mu$ table (in whole years) alongside
  each rates table.
  MLE uncertainty uses the delta method and CI inversion;
  Bayesian uncertainty is derived directly from the death-rate posterior samples.
  Both multi-epoch notebooks (`03_multi_epoch`, `04_bayesian`) updated to emit
  the half-life table within each LaTeX tab.

- Half-life companion figures added to both multi-epoch notebooks alongside every
  death-rate figure: MLE notebook uses `rate_ci_panel` (dot-and-whisker, log scale)
  for the per-tradition and per-epoch tabsets plus a line chart for the cross-subset
  comparison; Bayesian notebook uses a `posterior_rates_chart` ridgeline (log scale)
  for both tabsets.
  Half-lives are `ln 2 / μ` with CI derived from inverting the
  death-rate CI (MLE) or transforming posterior samples directly (Bayesian).

- `analysis.subset_inference`: `BayesianSubsetConfig` frozen dataclass plus two
  new functions --- `fit_manuscript_subset_bayesian` (picklable per-worker
  function running MLE→NUTS for one subset) and `fit_manuscript_subsets_bayesian`
  (parallel driver using `joblib.Parallel(n_jobs=-1)`, mirroring the existing
  validation-notebook convention).

- `qmd_scripts/core_analysis/03_multi_epoch/ntvmr_birth_death_inference.qmd`:
  added two Quarto tabset panels comparing rates --- "Rates by Epoch (per
  Tradition)" and "Rates by Tradition (per Epoch)" --- plus a "LaTeX Tables"
  section with copiable booktabs/siunitx code blocks in both slicings.

- `qmd_scripts/core_analysis/04_bayesian/ntvmr_birth_death_bayesian.qmd`:
  expanded to run all six subsets in parallel (`TRIAL_SUBSET = None`); added
  tabset panels for posterior rates by tradition (per epoch) and by epoch (per
  tradition); added trace-plot tabset (one tab per tradition); added per-subset
  LaTeX table section with mean/median/95 % CrI columns; aligned lectionary
  subset to 380 CE epoch start (matching the MLE notebook); fixed prose to
  reflect actual config (`target_accept=0.8`, `max_tree_depth=8`, NumPyro).

- `qmd_scripts/validation/single_epoch/06_bulk_inference.qmd`: reordered figure
  flow (histograms → heatmaps → scatter); histograms and scatter plots now have
  interactive legend toggles for boundary/non-boundary groups; two new λ-vs-μ
  count heatmaps added (all fits and non-boundary only) with the true parameter
  value marked by a cross; scatter plots now IQR-filtered before display so the
  interactive Vega-Lite spec stays small; "Mean Fitted Rates vs Population Size"
  section header corrected to "Fitted Rates vs Mean Population Age".

- `data.ntvmr.download_all_manuscripts` now runs a fast connectivity preflight
  via `check_ntvmr_reachable()` before starting the download loop; if the server
  is unreachable it returns immediately with `stats["preflight_failed"] = 1` instead
  of grinding out 200 silent connection failures.
  Error stats now include
  `connection_errors` (network/exception) and `error_responses` (server returned
  `<error>` XML) sub-counts alongside the existing `errors` total.
  `NTVMR_GET_URL`
  constant replaces the two duplicated inline f-strings.

- `cli ntvmr download-all` gains `--verbose`/`-v` and `--log-file PATH` options;
  `--log-file` attaches a `FileHandler` to the `inf_bp_texts` package logger so all
  progress and error detail is persisted to a file.
  The summary line now shows the
  error sub-counts, and a human-readable "server appears unreachable" hint is printed
  when `preflight_failed` is set.

- `visualization.era_shading` --- `era_band_chart` and `year_to_century_coord`:
  reusable helpers that build alternating era background-band and label layers
  from any `EpochDefinition` tuple (e.g. `MANUSCRIPT_EPOCHS_SIX_ERA`) for
  figures using a quantitative century axis; re-exported from
  `inf_bp_texts.visualization`.

- `visualization.altair_theme.save_chart_html` --- exports an Altair chart to a
  self-contained interactive HTML (Vega + vega-embed inlined); re-exported from
  `inf_bp_texts.visualization`.

- `figures/mde_temporal_distribution_eras.pdf` --- era-shaded variant of the
  temporal distribution figure with alternating bands and era labels from
  `MANUSCRIPT_EPOCHS_SIX_ERA`; generated by `manuscript_data_exploration.qmd`.

- `figures/mde_temporal_distribution.html` --- interactive HTML export of the
  temporal distribution figure (legend-filter, brush, tooltips preserved);
  generated by `manuscript_data_exploration.qmd`.

- `constants.DEATH_LOG_PRIOR_LOC_SIX_ERA`, `DEATH_LOG_PRIOR_LOC_DURABLE`,
  `BIRTH_LOG_PRIOR_LOC`, `LOG_RATE_PRIOR_SCALE` --- material-informed Bayesian
  prior constants for log birth/death rates; death-rate means encode the
  papyrus-to-parchment material transition across the six canonical manuscript
  eras and are aligned to `MANUSCRIPT_EPOCHS_SIX_ERA`.
  Applied across the four
  real-data Bayesian notebooks in `qmd_scripts/core_analysis/04_bayesian/`.

- `constants.TEXT_UND_TEXTWERT_PAUL` --- authoritative *Text und Textwert* Pauline
  apparatus manuscript list (Gregory-Aland IDs) stored as an immutable constant;
  used to validate the pipeline's XML-derived `section_p` classification in
  `manuscript_data_exploration.qmd`.

- `analysis.subset_inference` --- `build_manuscript_subsets`/`fit_manuscript_subset`
  split the cleaned NTVMR data into six canonical subsets (non-lectionaries
  combined, lectionaries, and four per-section non-lectionary analyses) and fit
  each via MLE with per-epoch rates/SEs/95% CIs.

- `analysis.date_density` --- `explode_date_ranges`, `manuscript_date_density`,
  `quantile_resample_dates`, `resample_dates_from_ranges` for weighted KDE over
  manuscript date *ranges* (year-explosion + 1/n weights) instead of midpoints;
  `data_cleaning.add_resampled_year_column` populates the resulting
  `resampledYear` column.
  See `docs/design/KDE_DATE_RANGE_HANDLING.md`.

- `visualization.age_fit_plot.plot_creation_year_fit` and
  `visualization.date_distribution_plots.plot_date_range_kde_comparison` --- shared
  Altair builders for the recurring creation-year/date-range figures, replacing
  duplicated inline plotting code across notebooks.

- `visualization.altair_theme` --- registers the house Altair theme (ptolqual
  palette, Tufte axes) and `save_chart`, the vl-convert export chokepoint;
  Altair is now the only charting library used by the package and every
  notebook (see `docs/design/PLOTLY_TO_ALTAIR_IMPLEMENTATION_PLAN.md`).

- `visualization.bayesian_plots` --- `prior_posterior_chart`,
  `posterior_rates_chart`, `trace_chart`, `ppc_chart`: tested Altair builders
  for the figure archetypes repeated across the four `04_bayesian/` notebooks,
  completing the Plotly→Altair migration for those notebooks.

- `visualization.altair_theme.display_svg` --- renders a chart to static SVG
  (via the same vl-convert path as `save_chart`) instead of an interactive
  `display(...)`.
  Data-heavy charts (per-iteration optimizer histories, bulk
  simulation scatters) now use it in `optimizer_comparison.qmd` and the
  `bulk_inference*`/`03_optimisation_framework` validation notebooks, since
  inlining their full Vega-Lite data as a multi-MB single line was choking
  pandoc's markdown reader during render.

- `inference.bayesian` --- per-family prior loc/scale on `BayesianPriorConfig`,
  in-model parameter constraints on `BayesianModelConfig`, and
  `mle_init_log_rates()` for MLE-warm-started NUTS.

- `qmd_scripts/core_analysis/04_bayesian/` --- four Bayesian inference notebooks
  (Kashouh FamilyK, Kashouh single-epoch, NTVMR single-epoch, NTVMR six-epoch)
  and an optional `gpu` dependency group for ad-hoc CPU-vs-GPU benchmarking
  (not used for routine rendering --- see Changed).

- `scripts/render_cpu_light.slurm`, `scripts/render_cpu_parallel.slurm` ---
  SLURM batch scripts covering all notebooks, including the Bayesian ones, in
  two resource-appropriate buckets (light/serial and embarrassingly-parallel),
  plus `scripts/render_all.sh` to submit them as a dependency chain; see
  `docs/design/GPU_RENDERING_PROTOCOL.md`.

- `utils.derive_worker_seeds()` (JAX-based independent seeding for joblib
  workers) and `utils.format_ordinal()`; `schemas` module (Pydantic v2 input
  validation); `_paths.get_project_root()` for robust path resolution in
  notebooks.

- CI lint job (ruff + type-check) added to GitHub Actions; `CITATION.cff`,
  `CHANGELOG.md`, `CONTRIBUTING.md` provenance files.

- NTVMR single-/multi-epoch notebooks now run six dataset subsets (was one)
  with full per-subset diagnostics and a cross-subset comparison.

### Changed

- **Migrated the numerical/optimization core from PyTorch to JAX**: math
  kernels (`models/math_*`), the MLE optimizer layer (`optimistix`/`optax`
  replacing the PyTorch LBFGS closure), and Bayesian inference (NumPyro NUTS
  replacing Pyro) are now pure JAX, float64, and JIT-compiled.
  The MLE hot path
  is JIT-compiled end-to-end, giving a 65×--10,000× per-call speedup and a
  60×--280× end-to-end fitter speedup on multi-epoch fits; the extinction
  probability integral was replaced with a closed-form analytic expression
  (310× speedup, also autograd- and GPU-compatible).
  `torch`/`pyro-ppl` are no
  longer dependencies.
  Benchmarked CPU vs. real-GPU (A100) execution across the
  JIT MLE hot path and Bayesian NUTS fits (varying sample count, chain count,
  and a 67× larger multi-subset NTVMR model): CPU won every configuration
  tested, by roughly 1.6×--8×, with NUTS's control-flow-heavy tree-building
  the likely cause (suits CPU better than GPU's SIMD-throughput model,
  independent of problem size).
  `scripts/render_gpu.slurm` was removed as a
  result --- the Bayesian notebooks now render on CPU via
  `scripts/render_cpu_light.slurm`.
  See `docs/design/JAX_MIGRATION_BENCHMARKS.md`
  and `docs/design/GPU_RENDERING_PROTOCOL.md` for the numbers.
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
- `_quarto.yml`'s `render:` list no longer excludes the Bayesian (NumPyro/NUTS)
  notebooks --- a plain `quarto render` of the whole site now executes them
  too.
  Use the new `scripts/render_*.slurm` (see Added) to render large
  batches efficiently on HPC instead of a full site render on a laptop.
- `inference.bayesian` --- removed the no-op `BayesianModelConfig.device` and
  `BayesianMCMCConfig.jit_compile` fields (JAX/NumPyro always control the
  backend/JIT themselves); use the `JAX_PLATFORMS` environment variable
  instead.
  `BayesianPriorConfig.resolve_tensors()` dropped its unused
  `device` parameter to match.
- `inference.likelihood` --- the JIT-compiled MLE-hot-path core
  (`_log_likelihood_jit_core`) and the validating `log_likelihood()` front-end
  no longer duplicate the population/age-marginal math; `log_likelihood()` now
  delegates to the same shared core (renamed `_log_likelihood_core`), with
  validation/spine-auto-detection staying in the front-end only.

### Fixed

- `inference.fitting`/`inference._optimizer_runners` --- multi-optimizer fits
  (e.g. the default `("LBFGS", "Adam")`) now warm-start each subsequent
  optimizer from the running best parameters instead of the original initial
  guess, and `fit_birth_death_processes()` returns the globally best-NLL
  attempt's parameters (with `selected_attempt_index` reported consistently)
  instead of whichever attempt happened to run last.
- `visualization.altair_theme.display_svg` --- fixed a `TypeError` raised on
  every call (`a bytes-like object is required, not 'str'`): vl-convert
  returns the SVG mimebundle entry as `str`, but the function buffered it into
  an `io.BytesIO`; switched to `io.StringIO` to match.
- `inference._optimizer_runners` --- convergence thresholds are now
  sample-size invariant: the gradient-norm threshold scales with the number of
  observations and the NLL-improvement checks (LBFGS plateau, Adam/SGD
  patience) use a relative tolerance instead of a fixed absolute one.
  The
  patience (no-improvement) criterion for Adam/SGD now also requires a small
  gradient norm, and the first-order optimizer gained an "already converged at
  the initial point" guard (mirroring LBFGS) so an already-optimal start no
  longer raises a spurious "did not improve" error.
- `inference.uncertainty._invert_observed_fisher` --- a non-positive-definite
  observed Fisher information (e.g. at a saddle point) could invert cleanly
  and silently yield negative variances; positive-definiteness is now checked
  via Cholesky before trusting an inversion, with a warning when none of the
  jitter fallbacks recover it.
  `fit_birth_death_processes()` also warns when
  uncertainty is estimated for a fit that did not converge.
- `inference.fitting.get_crude_parameter_estimates` --- the birth/death-rate
  "shrink toward each other" step computed the death-rate update from the
  already-shrunk birth rate, making the pull asymmetric; both updates now use
  the original (pre-shrink) values.
- `inference._optimizer_runners` --- L-BFGS-B `ftol` was divided by `1e-8`,
  causing scipy to stop after one iteration and report false convergence
  failures on large-NLL (e.g. six-epoch) problems.
- `models.math_extinction.calculate_log_lik_initial_lifetimes_given_extinction`
  --- a missing `as_array=True` on an internal call made the multi-epoch
  combined log-likelihood incorrect (golden value updated).
- `models.mathematical_derivatives.calculate_spine_ages_log_lik_derivatives`
  (pre-JAX-migration) --- interior branch used epoch end time `T` instead of
  manuscript age `t`, giving wrong gradients off epoch boundaries.
- `kashouh_family_k_bayesian_inference.qmd` --- `import jax` ran before
  `from inf_bp_texts import ...`, so the package's CPU-default
  `os.environ.setdefault("JAX_PLATFORMS", "cpu")` never took effect once the
  `gpu` dependency group was installed; the notebook would silently render on
  GPU regardless of intent.
  Fixed by setting the env var directly before
  `import jax`, matching `scripts/bench_jax_jit.py`.
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
- `data.ntvmr` --- NTVMR requests now send browser-like (Firefox) headers
  instead of urllib's default `Python-urllib/x.y` `User-Agent`, which some
  servers reject; added a `_download_url()` helper (using `Request`/`urlopen`)
  since `urlretrieve` cannot carry custom headers.

### Removed

- `BirthDeathProcess.__getitem__` (dict/DataFrame-style subscripting) --- use
  `.spine_ages`, `.non_spine_ages`, `.surviving_manuscripts`,
  `.extinct_manuscripts`, `.manuscript_records` instead.
- `BirthDeathProcess.__eq__`'s dict-comparison branch.
- `fit_birth_death_processes_bayesian()`'s `unbias` keyword --- pass
  `model_config=BayesianModelConfig(unbias=...)` instead.
- Dead code in `visualization.plotting` (`layout_config`, unused colour
  constants, never-called `apply_tufte_legend`), superseded by the Altair theme.
- `plotly`, `kaleido`, `plotly-stubs` dependencies and the remaining Plotly
  helpers in `visualization.plotting` (`apply_professional_styling`,
  `apply_tufte_axes`, `save_figure`) --- the Plotly→Altair migration is
  complete; use `visualization.altair_theme.save_chart`/`display_svg` instead.

## [0.1.0] --- initial release
