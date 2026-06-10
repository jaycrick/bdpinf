# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
