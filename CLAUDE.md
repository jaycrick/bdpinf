# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git commit messages

Every commit message must begin with exactly one of these shortcode prefixes, followed by a single space and the description:

  | Prefix  | Use for                        |
  | ------- | ------------------------------ |
  | `:zap:` | features and primary concerns  |
  | `:bug:` | bug fixes                      |
  | `:tv:`  | CI, tooling, and configuration |
  | `:cop:` | tests and linting              |
  | `:abc:` | documentation                  |

Use the colon-wrapped shortcode (e.g. `:zap:`), not the Unicode emoji character.

Example: `:zap: Add users endpoint, fixes #27`

**Commit granularity:** Prefer bundled commits that group related code, tests, and docs into a single commit.
Choose the prefix that reflects the commit's primary concern.
Do not split a feature into separate per-prefix commits (e.g. one `:zap:` for the code, one `:cop:` for its tests, one `:abc:` for its docs) unless those pieces genuinely belong in separate commits for other reasons.

**CHANGELOG.md**: Update `CHANGELOG.md` at the repo root whenever a commit adds a new feature, changes public API, or fixes a user-visible bug.
Add a bullet under `## [Unreleased]` --- one line describing the change and which module it affects.
No need to touch `CHANGELOG.md` for pure tooling/CI changes (`:tv:`) or test-only commits (`:cop:`).

## Prose style

Write prose in `.qmd`, `.md` files and docstrings with semantic line breaks (one thought per line; see the `sembr` skill, enabled for this project in `.claude/settings.json`).
This matches `wrap = "semantic"` in `panache.toml`.

## Commands

```bash
# Install dependencies
uv sync --all-groups

# Run tests (slow tests excluded by default; coverage ≥80% enforced)
uv run pytest
uv run pytest tests/test_fitting.py -v        # single file
uv run pytest -m slow                          # run slow tests only
uv run pytest -m regression                    # golden-value regression tests
uv run pytest -m recovery                      # simulate-then-fit recovery tests (implies slow)
uv run pytest -k "test_name"                   # single test by name

# Linting and formatting
# Always run format and auto-fix first — let ruff correct what it can automatically
# before checking compliance, rather than hand-editing formatting or style issues.
# Import sorting for src/tests is handled by ruff's own "I" rules (see
# [tool.ruff.lint] select in pyproject.toml) — do not also run standalone
# `isort` over src/tests: isort splits any imported name that carries a
# trailing inline comment (e.g. `# pyright: ignore[...]`) onto its own
# import statement, which undoes ruff's merge and leaves files failing CI's
# `ruff check src tests` (CI does not run isort). isort is still used by
# panache to format Python chunks inside `.qmd` files (see panache.toml) —
# that pipeline runs isort before ruff, so it stays self-consistent.
uv run ruff format src tests
uv run ruff check --fix src tests
uv run basedpyright --project pyproject.toml src/  # strict mode; 0 errors enforced

# Markdown/Quarto prose linting and formatting (config in panache.toml)
uv run panache format .          # auto-fix formatting in place
uv run panache lint --fix .      # auto-fix lint issues
uv run panache lint --check .    # CI mode; no changes, exit 1 on issues

# All at once (pre-commit)
uv run pre-commit run --all-files

# Lint Python cells in Quarto notebooks — run whenever a .qmd file is created or updated
uv run lintquarto -l ruff basedpyright -p qmd_scripts docs -e .venv _output

# Research notebook site — always use uv run to pick up the venv; plain `quarto`
# uses system Python and fails with ModuleNotFoundError
uv run quarto render            # full site build → _output/ (cached cells skipped)
uv run quarto preview           # incremental preview with live reload

# API reference site (separate from the notebook site above; config in great-docs.yml)
uv run great-docs build         # regenerate the reference site → great-docs/_site
uv run great-docs preview       # live preview
```

## Tests

### Canonical file layout

Every `tests/test_*.py` file follows this order, top to bottom:

```python
"""<One-line module docstring ending in a period.>"""

# (optional) module-level pragmas that are NOT ruff, e.g.
# mypy: disable-error-code=import-untyped

import <stdlib>
import <third-party>

from bdpinf... import ...

# module-level constants, fixtures, and non-test helper functions stay here


class TestSubject:
    """Test <subject>."""

    def test_something(self) -> None:
        ...
```

### Conventions

- **Module docstring** is the first statement, single line, ends with `.`.
  `D100` is enforced so every file must have one.
- **No per-file `# ruff: noqa` lines** --- test-wide relaxations live in `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`, not in individual files.
- **Class grouping** --- all tests live inside a `class Test<Subject>:` with a one-line `"""Test ..."""` docstring.
  Test methods take `self`.
  One class per file is the default.
- **Class/method docstrings are optional** --- `D101`/`D102` are centrally ignored for tests via `pyproject.toml`; method names are self-documenting.
- **Module-level helpers stay at module scope** --- fixtures, helper functions, and constants shared within a file remain outside the test class (pytest resolves module-scoped fixtures inside class methods).
  Cross-file helpers and fixtures go in `tests/conftest.py`.
- **Non-ruff pragmas** (e.g. `# mypy: disable-error-code=import-untyped`) go directly under the module docstring, before imports.

## Architecture

**bdpinf** downloads New Testament manuscript data from the NTVMR API, runs quality control, and fits birth-death process models to estimate manuscript creation/survival rates via MLE.

### Core data structure: `BirthDeathProcess`

`models/birth_death_process.py` is the canonical object passed between modules.
It holds `creation_times`, `epoch_end_times`, per-epoch `birth_rates`/`death_rates` (all `jax.Array` float64), and optional `manuscript_records` for genealogy.
Key methods: `log_likelihood()`, `update_parameters()`, `plot_genealogy_tree()`, `to_newick()`.

### Mathematical layer: `models/`

Three submodules contain pure functions (no class state):
- `math_extinction.py` --- `calculate_extinction_pr()`: probability population dies by time T
- `math_ages.py` --- `calculate_spine_ages_log_lik()`, `calculate_non_spine_ages_log_lik()`, `_reverse_time()`
- `math_populations.py` --- `calculate_populations_log_lik()`: probability of observed population counts

`_reverse_time()` --- **time-reversal trick**: transforms multi-epoch non-spine ages into an extinction-conditioned backward-time problem, avoiding exponential complexity in the number of epochs.

`rate_functions.py` --- `RateFunction` protocol plus `PiecewiseConstantRate` and `PiecewiseLinearRate` (jumps allowed; `.continuous(knots, times)` for jump-free); JAX pytrees exposing `__call__`, exact `integral(s, e)` and `breakpoints`.
New rate families (sin/cos, splines, rational) only need to satisfy this protocol.

`math_age_densities.py` --- general-rate spine, non-spine and uniformly-chosen-individual age densities/CDFs in the paper's notation (`spine_age_log_density`, `spine_age_log_atom`, `non_spine_age_log_density`, `random_individual_cdf`, `log_nested_w`).
All reduce to one primitive, $L_w(s,e)=\int_s^e w(t)e^{\Gamma(t,e]}dt$, evaluated by Gauss--Legendre quadrature between breakpoints; no division by $\lambda-\mu$, so the critical case is exact.
Not yet wired into `BirthDeathProcess`, fitting or simulation.

Numerical derivatives used during optimization (Hessian/Jacobian for the MLE covariance) live in `inference/uncertainty.py` via `jax.hessian`/`jax.jacobian`.

`validation.py` --- `normalize_and_validate_inputs()` converts scalars/arrays to float64 tensors and enforces shape, range, and monotonicity constraints.
`ValidationConfig` (Pydantic) controls which checks run.

### Inference layer: `inference/`

`fitting.py` is the main entry point:
- `fit_birth_death_processes()` --- MLE via JAX-backed optimizers: scipy's L-BFGS-B (gradients from `jax.value_and_grad`) and optax's Adam/SGD; optimizers run sequentially (e.g. LBFGS then Adam) and each warm-starts from the previous attempt's best parameters, stopping at the first convergence.
Supports initial perturbation (`perturb_initial_params_std`) for robustness to bad initial guesses.
- `ParameterConstraints` dataclass --- fixed parameters (index → value) and equality groups (lists of indices forced equal); validated before optimization to reduce parameter space
- `estimate_mle_covariance()` --- covariance via numerical Hessian inversion
- Optimizer internals are split across `_fit_types.py` (shared dataclasses: `ConvergenceConfig`, `OptimizerStrategy`, etc.) and `_optimizer_runners.py` (`run_lbfgs_optimizer`, `run_first_order_optimizer`).
`ConvergenceConfig` is exported from `inference/__init__.py`.

`likelihood.py` --- `log_likelihood()` / `negative_log_likelihood()` wrappers that handle log-scale vs. positive-scale parameters and marginalize over unknown spine positions.

`bayesian.py` --- NumPyro/NUTS posterior inference for uncertainty quantification.

`persistence.py` --- `save_posterior_samples()` / `load_posterior_samples()`: thin `.npz` round-trip wrappers so MCMC runs (birth/death-rate draws plus epoch metadata) are written once and reloaded in downstream notebooks without re-running MCMC.

### Analysis layer: `analysis/`

Post-inference and date-processing helpers used in notebooks.

`date_density.py` --- `explode_date_ranges()`, `manuscript_date_density()`, `quantile_resample_dates()`: deterministic KDE-quantile pipeline that converts `[earlyYear, lateYear]` ranges to representative integer years via a weighted, boundary-corrected Gaussian KDE and Hazen plotting positions.
Wired into data loading via `data_cleaning.add_resampled_year_column()`.

`bootstrap_date_density.py` --- `bootstrap_quantile_resample_dates()`: vectorized reimplementation of the same pipeline, batched over 100,000 bootstrap draws without calling scipy in a loop.
`save_bootstrap_samples()` / `load_bootstrap_samples()` persist `[n_bootstrap, n]` `uint16`/`int16` arrays to `bootstrap_samples/<subgroup>.npz`.
The identity multiplicity exactly reproduces `quantile_resample_dates`.
Full mathematical derivation in `docs/date_uncertainty.qmd`.

`subset_inference.py` --- `build_manuscript_subsets()`, `fit_manuscript_subsets_bayesian()`, `ManuscriptSubset`, `SubsetFitResult`: batch MLE/Bayesian fitting across manuscript subgroups with a shared interface.

`latex_tables.py` --- LaTeX table builders for MLE and Bayesian results: `rates_latex_table`, `subset_fits_latex_by_epoch`, `bayesian_results_latex_by_epoch`, etc.

`validation_helpers.py` --- chart and model diagnostic helpers used in validation notebooks: `boundary_corrected_kde_on_support`, `boundary_corrected_kde_at_zero`, `FitModelConfig`, `fit_single_model`, `theoretical_age_density`, etc.

`spike_detection.py` --- `bootstrap_percentile_ci()`, `compute_ci_width_table()`, `select_extreme_replicates()`, `seed_from_parts()`.

`single_epoch.py` / `asymptotic_consistency.py` --- summary and consistency helpers for single-epoch fits.

### Simulation layer: `simulation/`

`simulator.py` --- two simulation engines:
- `simulate_birth_death_process()` runs a Gillespie algorithm with time-varying rates; `predict_manuscript_counts()` predicts future population via rejection sampling.
- `simulate_birth_death_lifespans()` / `simulate_birth_death_lifespans_arrays()`: JAX/JIT-compiled lifespan-at-birth alternative where each individual's death time is pre-sampled at birth from a piecewise-exponential distribution; the event loop runs inside `jax.lax.while_loop` with fixed-shape preallocated buffers.
`LifespanSimulationArrays` exposes per-individual `birth_epoch` and `survived` flags for downstream splitting/conditioning.

`splitting.py` --- Multilevel splitting for rare-event simulation (estimating small extinction probabilities).
Plotting functions for splitting results live in `visualization/splitting_plots.py`.

`survival_conditioning.py` --- estimates the proportion of ever-created manuscripts currently alive by conditioning forward lifespan simulations on observed per-era survivor counts via a filter/clone loop (sequential-ABC / multilevel-splitting scheme).
Key functions: `build_target_ranges()` (observed counts → per-era `[lo, hi]` bounds), `sample_conditioned_survival_process()` (fixed rates → matched trajectories), `sample_conditioned_survival_per_posterior()` (one trajectory per posterior draw, returns a tidy `pd.DataFrame`).
The JIT-compiled single-era core `_simulate_era_core` resumes the lifespan engine from an arbitrary alive-population state.

### Visualization layer: `visualization/`

All charts are built with Altair; there is no Plotly dependency.

`altair_theme.py` --- the house-style chokepoint: importing it registers and enables the project's Altair theme (palette, fonts, Tufte axes) process-wide, and it owns the single export path, `save_chart()` (PDF/SVG/PNG via vl-convert) and `display_svg()` (inline static SVG for data-heavy charts in Quarto).

`plotting.py` --- `convergence_chart`, `save_multi_epoch_*_figure`, `plot_parameter_consistency`.

`bayesian_plots.py` --- reusable builders for the recurring Bayesian-notebook figures: `prior_posterior_chart`, `posterior_rates_chart`, `trace_chart`, `ppc_chart`.

`age_density_plots.py` --- `rate_frame`, `age_density_frame`, `scenario_panel`, `scenario_row`: rates-above-age-density panels for time-varying-rate scenarios (non-spine or spine primary curve, optional faint secondary).

`age_fit_plot.py` --- `plot_creation_year_fit`, `creation_year_fit_chart` (observed-vs-fitted creation-year overlay).

`date_distribution_plots.py` --- `plot_date_range_kde_comparison`.

`splitting_plots.py` --- `plot_conditioned_trajectories`, `plot_empirical_population_mean`, `plot_level_survival_probabilities` (re-exported from `simulation`).

`parameter_plots.py` --- `KdePlotConfig`, `BoxPlotStyle`, `create_kde_density_plot`, `compute_boxplot_summary`, `create_parameter_box_plot` (re-exported from `analysis`).

`genealogy.py` --- `plot_genealogy_tree`, `genealogy_dfs_order` (consumed by `models.birth_death_process`, not re-exported from the package root).

This module set is intentionally the boundary between reusable, tested chart logic and notebook-local composition: named builders that are reused across notebooks or coupled to model internals live here; one-off diagnostic charts are built inline in notebooks with `alt.Chart(...)` plus the shared `PTOLQUAL` palette and `save_chart`/`display_svg`.

### Data pipeline: `data/`

All NTVMR ETL modules live in the `bdpinf/data/` subpackage.
Key public API (also re-exported from `bdpinf.data`):

`ntvmr.py` --- downloads from NTVMR API with resume support; `format_xml_files()` pretty-prints XML and strips Unicode line separators (U+2028/U+2029) for clean git diffs.

`data_cleaning.py` --- `examine_manuscript_xml()`, `clean_manuscript_dates()`, `generate_data_quality_report()`.

`xml_processing.py` --- `analyze_all_manuscripts()` batch-parses a folder for book/section content.

`content_extraction.py` --- `parse_biblical_books()`, `parse_content_overview()`.

`pattern_analysis.py` --- `analyze_section_combinations()`.

`reports.py` --- `generate_mismatch_report()`, `analyze_section_mismatches()`.

`manuscript_analysis.py` --- Typer CLI entry point wrapping `analyze_all_manuscripts`.

### Types, constants, and utilities

`constants.py` --- canonical book ordering, German abbreviations, and section patterns used throughout the XML layer.

`schemas.py` --- `ValidationConfig` Pydantic model (also available from `models.validation`) controlling which validation checks are enforced.

`paths.py` --- `find_project_root()` (exported from `bdpinf` top-level): returns the Quarto project root via `$QUARTO_PROJECT_DIR` env var, or walks up from `cwd` to find `_quarto.yml`/`pyproject.toml`/`.git`.
Use this in notebooks and scripts instead of hardcoded relative paths.

**CLI**: The package installs an `bdpinf` command (entry point: `bdpinf.cli:main`) via `[project.scripts]` in `pyproject.toml`.

## Key design decisions

1. **JAX everywhere**: All rates and ages are `jax.Array` (float64) --- enables gradient-based optimization via `jax.grad`/`jax.jit` and ensures numerical precision throughout the pipeline.
   `jax.config.update("jax_enable_x64", True)` is set at package import.

2. **Pure math functions, not methods**: `math_extinction.py`, `math_ages.py`, `math_populations.py` contain only free functions with no class state, keeping the math testable and composable independently of `BirthDeathProcess`.

3. **Time-reversal for multi-epoch efficiency**: The non-spine likelihood is reformulated as an extinction-conditioned backward-time integral.
   This avoids summing over all possible genealogy orderings (exponential in epoch count) and reduces it to a single pass.

4. **`ParameterConstraints` as first-class input**: Equality groups and fixed parameters are validated early and applied throughout the optimizer loop --- reducing effective dimension of the search space for parsimonious models.

5. **Likelihood marginalizes unknown spine**: When the spine (ancestral lineage) is unspecified, `log_likelihood()` sums over all possible spine assignments, so the fitter works with incomplete genealogy data.

## Known challenges

**Multi-epoch convergence**: High-dimensional parameter space with sensitivity at epoch boundaries.
Mitigate by: using equality constraints (constant rates across epochs), providing good initial guesses, using LBFGS first then Adam, and setting `perturb_initial_params_std > 0`.

**Numerical stability**: Likelihoods are always computed in log-space.
Extreme rate values (μ >> λ or vice versa) and ages near epoch boundaries can still cause issues.

**XML edge cases**: `format_xml_files()` must be run after downloads to clean U+2028/U+2029 characters before parsing.
Malformed or empty `originYear` elements surface in `examine_manuscript_xml()`.

## Quarto notebooks (`qmd_scripts/`)

`_quarto.yml` is the single source of truth for the research-notebook website: render globs, navbar menus, and HTML format (output to `_output/`).
Cell output is cached globally (`execute: cache: true`).
**Always use `uv run quarto ...`** --- plain `quarto` uses system Python (no venv) and fails immediately.

The **API reference** is a separate site, generated by **great-docs** from `great-docs.yml` (not part of `_quarto.yml` --- great-docs owns its own Quarto project under `great-docs/`, which cannot host the executable research notebooks because it has no `execute.cache`/`bibliography` equivalent).
See `docs/design/DOCS_TOOLING_EVALUATION.md` for why this is two sites rather than one.

### Directory taxonomy

  | Path                                             | Purpose                                  | File naming                                                                                    |
  | ------------------------------------------------ | ---------------------------------------- | ---------------------------------------------------------------------------------------------- |
  | `qmd_scripts/core_analysis/01_data_exploration/` | EDA on real manuscript data              | `<dataset>_data_exploration.qmd`                                                               |
  | `qmd_scripts/core_analysis/02_single_epoch/`     | MLE fits to real data, single-epoch      | `<dataset>_single_epoch_inference.qmd`                                                         |
  | `qmd_scripts/core_analysis/03_multi_epoch/`      | MLE fits to real data, multi-epoch       | `<dataset>_multi_epoch_inference.qmd`; topic-only (no dataset) for methods comparisons         |
  | `qmd_scripts/core_analysis/04_bayesian/`         | Bayesian inference on real data          | slow (multi-chain NUTS); CPU outperforms GPU here, render via `scripts/render_cpu_light.slurm` |
  | `qmd_scripts/validation/single_epoch/`           | Recover known params from synthetic data | `<nn>_<topic>.qmd` (e.g. `01_simulation_checks.qmd`)                                           |
  | `qmd_scripts/validation/multi_epoch/`            | Same, multi-epoch                        | `<nn>_<topic>.qmd` (e.g. `03_bulk_inference.qmd`)                                              |
  | `qmd_scripts/validation/bayesian_multi_epoch/`   | Bayesian validation                      | `<nn>_<topic>.qmd`; slow, see above                                                            |
  | `docs/*.qmd`                                     | Prose/math documentation                 | no executable cells                                                                            |
  | `great-docs.yml`                                 | API reference config                     | **drives the separate great-docs site — see above**                                            |

### Per-notebook conventions

Every executable notebook follows this structure:

- **Frontmatter**: `title`, `subtitle`, `author: "JC"`, `date: last-modified`.

- **Timing bracket**: first cell `# | label: compile-time-start` captures `time.perf_counter()`; last cell `# | label: compile-time-report` writes elapsed seconds to `_output/compile_times.json`.
  Both cells use `# | cache: false` / `# | echo: false` / `# | include: false`.

- **Cell options**: hash-pipe syntax inside code fences --- `# | label:`, `# | cache: false`, `# | echo: false`, `# | include: false`.

- **Import noqa**: import cells open with `# ruff: noqa: E402` (imports are not at the top of the Python file, which ruff would flag).

- **Keep logic in `src/`**: cells import from `bdpinf`; heavy computation belongs in the package, not inline.

- **Figures**: saved via `save_chart` / `save_chart_html` from `bdpinf.visualization`.
  All figures live under a single repo-root `figures/` tree that **mirrors** the `qmd_scripts/` layout:
  ```
  figures/
    core_analysis/01_data_exploration/   # prefix 01a_ or 01b_
    core_analysis/02_single_epoch/       # prefix 02a_ or 02b_
    core_analysis/03_multi_epoch/        # prefix 03a_ … 03e_
    core_analysis/04_bayesian/           # prefix 04a_ … 04d_
    validation/single_epoch/             # prefix vse01_, vse05_, vse06_
    validation/multi_epoch/              # prefix vme01_, vme03_, vme04_, vme05_
  ```
  In each notebook, set `figures_dir = find_project_root() / "figures" / "<mirror-subpath>"` and pass `output_dir=figures_dir` to every `save_chart` / `save_chart_html` call.
  Filenames follow the pattern `<prefix>_<descriptive>.pdf`, where the prefix is the submodule id (e.g. `04b`) assigned to each notebook.
  When `output_dir` is omitted, `save_chart` falls back to the module default set via `set_figure_dir()`, then `$BDPINF_FIGURE_DIR`, then `Path.cwd() / "figures"` --- so notebooks that opt out of the mirrored layout still get a `figures/` directory next to wherever they're run from.
  Do not use hardcoded `Path("../../../figures")`.

- **Design docs**: architecture/audit documents live in `docs/design/` (not the repo root).
