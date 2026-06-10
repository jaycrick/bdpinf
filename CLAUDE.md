# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git commit messages

Every commit message must begin with exactly one of these shortcode prefixes, followed by a single space and the description:

| Prefix | Use for |
|--------|---------|
| `:zap:` | features and primary concerns |
| `:bug:` | bug fixes |
| `:tv:` | CI, tooling, and configuration |
| `:cop:` | tests and linting |
| `:abc:` | documentation |

Use the colon-wrapped shortcode (e.g. `:zap:`), not the Unicode emoji character.

Example: `:zap: Add users endpoint, fixes #27`

**Commit granularity:** Prefer bundled commits that group related code, tests, and docs into a single commit. Choose the prefix that reflects the commit's primary concern. Do not split a feature into separate per-prefix commits (e.g. one `:zap:` for the code, one `:cop:` for its tests, one `:abc:` for its docs) unless those pieces genuinely belong in separate commits for other reasons.

**CHANGELOG.md**: Update `CHANGELOG.md` at the repo root whenever a commit adds a new feature, changes public API, or fixes a user-visible bug. Add a bullet under `## [Unreleased]` — one line describing the change and which module it affects. No need to touch `CHANGELOG.md` for pure tooling/CI changes (`:tv:`) or test-only commits (`:cop:`).

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
uv run ruff check src tests
uv run ruff format src tests
uv run isort src tests
uv run basedpyright --project pyproject.toml src/  # strict mode; 0 errors enforced

# All at once (pre-commit)
uv run pre-commit run --all-files

# Lint Python cells in Quarto notebooks — run whenever a .qmd file is created or updated
uv run lintquarto -l ruff pyright -p qmd_scripts docs -e .venv _output reference

# Documentation — always use uv run to pick up the venv; plain `quarto` uses
# system Python and fails with ModuleNotFoundError
uv run quartodoc build          # regenerate reference/*.qmd from docstrings
uv run quarto render            # full site build → _output/ (cached cells skipped)
uv run quarto preview           # incremental preview with live reload
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

from inf_bp_texts... import ...

# module-level constants, fixtures, and non-test helper functions stay here


class TestSubject:
    """Test <subject>."""

    def test_something(self) -> None:
        ...
```

### Conventions

- **Module docstring** is the first statement, single line, ends with `.`. `D100` is enforced so every file must have one.
- **No per-file `# ruff: noqa` lines** — test-wide relaxations live in `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`, not in individual files.
- **Class grouping** — all tests live inside a `class Test<Subject>:` with a one-line `"""Test ..."""` docstring. Test methods take `self`. One class per file is the default.
- **Class/method docstrings are optional** — `D101`/`D102` are centrally ignored for tests via `pyproject.toml`; method names are self-documenting.
- **Module-level helpers stay at module scope** — fixtures, helper functions, and constants shared within a file remain outside the test class (pytest resolves module-scoped fixtures inside class methods). Cross-file helpers and fixtures go in `tests/conftest.py`.
- **Non-ruff pragmas** (e.g. `# mypy: disable-error-code=import-untyped`) go directly under the module docstring, before imports.

## Architecture

**inf-bp-texts** downloads New Testament manuscript data from the NTVMR API, runs quality control, and fits birth-death process models to estimate manuscript creation/survival rates via MLE.

### Core data structure: `BirthDeathProcess`

`models/birth_death_process.py` is the canonical object passed between modules. It holds `creation_times`, `epoch_end_times`, per-epoch `birth_rates`/`death_rates` (all `torch.Tensor` float64), and optional `manuscript_records` for genealogy. Key methods: `log_likelihood()`, `update_parameters()`, `plot_genealogy_tree()`, `to_newick()`.

### Mathematical layer: `models/`

Three submodules contain pure functions (no class state):
- `math_extinction.py` — `calculate_extinction_pr()`: probability population dies by time T
- `math_ages.py` — `calculate_spine_ages_log_lik()`, `calculate_non_spine_ages_log_lik()`, `_reverse_time()`
- `math_populations.py` — `calculate_populations_log_lik()`: probability of observed population counts

`_reverse_time()` — **time-reversal trick**: transforms multi-epoch non-spine ages into an extinction-conditioned backward-time problem, avoiding exponential complexity in the number of epochs.

`mathematical_derivatives.py` — numerical derivatives used during optimization.

`validation.py` — `normalize_and_validate_inputs()` converts scalars/arrays to float64 tensors and enforces shape, range, and monotonicity constraints. `ValidationConfig` (Pydantic) controls which checks run.

### Inference layer: `inference/`

`fitting.py` is the main entry point:
- `fit_birth_death_processes()` — MLE via PyTorch optimizers (LBFGS, Adam); supports initial perturbation for robustness and convergence detection with multi-restart
- `ParameterConstraints` dataclass — fixed parameters (index → value) and equality groups (lists of indices forced equal); validated before optimization to reduce parameter space
- `estimate_mle_covariance()` — covariance via numerical Hessian inversion
- Optimizer internals are split across `_fit_types.py` (shared dataclasses: `ConvergenceConfig`, `OptimizerStrategy`, etc.) and `_optimizer_runners.py` (`_run_lbfgs_optimizer`, `_run_first_order_optimizer`). `ConvergenceConfig` is exported from `inference/__init__.py`.

`likelihood.py` — `log_likelihood()` / `negative_log_likelihood()` wrappers that handle log-scale vs. positive-scale parameters and marginalize over unknown spine positions.

`bayesian.py` — Pyro-based posterior inference for uncertainty quantification.

### Simulation layer: `simulation/`

`simulator.py` — `simulate_birth_death_process()` runs a Gillespie algorithm with time-varying rates; `predict_manuscript_counts()` predicts future population given current observations via rejection sampling.

`splitting.py` — Multilevel splitting for rare-event simulation (estimating small extinction probabilities). Plotting functions for splitting results live in `visualization/splitting_plots.py`.

### Visualization layer: `visualization/`

`plotting.py` — core Plotly helpers: `save_figure`, `save_multi_epoch_*_figure`, `apply_professional_styling`, `apply_tufte_axes`, `plot_parameter_consistency`.

`splitting_plots.py` — `plot_conditioned_trajectories`, `plot_empirical_population_mean`, `plot_level_survival_probabilities` (re-exported from `simulation`).

`parameter_plots.py` — `KdePlotConfig`, `BoxPlotStyle`, `create_kde_density_plot`, `compute_boxplot_summary`, `create_parameter_box_plot` (re-exported from `analysis`).

### Data pipeline: `data/`

All NTVMR ETL modules live in the `inf_bp_texts/data/` subpackage. Key public API (also re-exported from `inf_bp_texts.data`):

`ntvmr.py` — downloads from NTVMR API with resume support; `format_xml_files()` pretty-prints XML and strips Unicode line separators (U+2028/U+2029) for clean git diffs.

`data_cleaning.py` — `examine_manuscript_xml()`, `clean_manuscript_dates()`, `generate_data_quality_report()`.

`xml_processing.py` — `analyze_all_manuscripts()` batch-parses a folder for book/section content.

`content_extraction.py` — `parse_biblical_books()`, `parse_content_overview()`.

`pattern_analysis.py` — `analyze_section_combinations()`.

`reports.py` — `generate_mismatch_report()`, `analyze_section_mismatches()`.

`manuscript_analysis.py` — Typer CLI entry point wrapping `analyze_all_manuscripts`.

### Types, constants, and utilities

`constants.py` — canonical book ordering, German abbreviations, and section patterns used throughout the XML layer.

`schemas.py` — `ValidationConfig` Pydantic model (also available from `models.validation`) controlling which validation checks are enforced.

`_paths.py` — `get_project_root()` (exported from `inf_bp_texts` top-level): returns the Quarto project root via `$QUARTO_PROJECT_DIR` env var, or walks up from `cwd` to find `_quarto.yml`. Use this in notebooks and scripts instead of hardcoded relative paths.

**CLI**: The package installs an `inf-bp-texts` command (entry point: `inf_bp_texts.cli:main`) via `[project.scripts]` in `pyproject.toml`.

## Key design decisions

1. **PyTorch everywhere**: All rates and ages are `torch.Tensor` (float64) — enables gradient-based optimization and ensures numerical precision throughout the pipeline.

2. **Pure math functions, not methods**: `math_extinction.py`, `math_ages.py`, `math_populations.py` contain only free functions with no class state, keeping the math testable and composable independently of `BirthDeathProcess`.

3. **Time-reversal for multi-epoch efficiency**: The non-spine likelihood is reformulated as an extinction-conditioned backward-time integral. This avoids summing over all possible genealogy orderings (exponential in epoch count) and reduces it to a single pass.

4. **`ParameterConstraints` as first-class input**: Equality groups and fixed parameters are validated early and applied throughout the optimizer loop — reducing effective dimension of the search space for parsimonious models.

5. **Likelihood marginalizes unknown spine**: When the spine (ancestral lineage) is unspecified, `log_likelihood()` sums over all possible spine assignments, so the fitter works with incomplete genealogy data.

## Known challenges

**Multi-epoch convergence**: High-dimensional parameter space with sensitivity at epoch boundaries. Mitigate by: using equality constraints (constant rates across epochs), providing good initial guesses, using LBFGS first then Adam, and setting `perturb_initial_params_std > 0`.

**Numerical stability**: Likelihoods are always computed in log-space. Extreme rate values (μ >> λ or vice versa) and ages near epoch boundaries can still cause issues.

**XML edge cases**: `format_xml_files()` must be run after downloads to clean U+2028/U+2029 characters before parsing. Malformed or empty `originYear` elements surface in `examine_manuscript_xml()`.

## Quarto notebooks (`qmd_scripts/`)

`_quarto.yml` is the single source of truth for the website: render globs, navbar menus, HTML format (output to `_output/`), and the quartodoc reference config. Cell output is cached globally (`execute: cache: true`). **Always use `uv run quarto …`** — plain `quarto` uses system Python (no venv) and fails immediately.

### Directory taxonomy

| Path | Purpose | File naming |
|------|---------|-------------|
| `qmd_scripts/core_analysis/01_data_exploration/` | EDA on real manuscript data | `<dataset>_data_exploration.qmd` |
| `qmd_scripts/core_analysis/02_single_epoch/` | MLE fits to real data, single-epoch | `<dataset>_single_epoch_inference.qmd` |
| `qmd_scripts/core_analysis/03_multi_epoch/` | MLE fits to real data, multi-epoch | `<dataset>_multi_epoch_inference.qmd`; topic-only (no dataset) for methods comparisons |
| `qmd_scripts/core_analysis/04_bayesian/` | Bayesian inference on real data | excluded from default render (slow) |
| `qmd_scripts/validation/single_epoch/` | Recover known params from synthetic data | `<nn>_<topic>.qmd` (e.g. `01_simulation_checks.qmd`) |
| `qmd_scripts/validation/multi_epoch/` | Same, multi-epoch | `<nn>_<topic>.qmd` (e.g. `03_bulk_inference.qmd`) |
| `qmd_scripts/validation/bayesian_multi_epoch/` | Bayesian validation | `<nn>_<topic>.qmd`; excluded from default render (slow) |
| `docs/*.qmd` | Prose/math documentation | no executable cells |
| `reference/*.qmd` | API reference | **auto-generated by quartodoc — do not hand-edit** |

### Per-notebook conventions

Every executable notebook follows this structure:

- **Frontmatter**: `title`, `subtitle`, `author: "JC"`, `date: last-modified`.
- **Timing bracket**: first cell `# | label: compile-time-start` captures `time.perf_counter()`; last cell `# | label: compile-time-report` writes elapsed seconds to `_output/compile_times.json`. Both cells use `# | cache: false` / `# | echo: false` / `# | include: false`.
- **Cell options**: hash-pipe syntax inside code fences — `# | label:`, `# | cache: false`, `# | echo: false`, `# | include: false`.
- **Import noqa**: import cells open with `# ruff: noqa: E402` (imports are not at the top of the Python file, which ruff would flag).
- **Keep logic in `src/`**: cells import from `inf_bp_texts`; heavy computation belongs in the package, not inline.
- **Figures**: saved via `save_figure` from `inf_bp_texts.visualization` into `figures/`, resolved as `figures_dir = get_project_root() / "figures"` (import `get_project_root` from `inf_bp_texts`). Do not use hardcoded `Path("../../../figures")`.
- **Design docs**: architecture/audit documents live in `docs/design/` (not the repo root).
