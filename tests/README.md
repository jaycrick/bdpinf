# Tests for inf-bp-texts

This directory contains the test suite for `inf_bp_texts`, covering models, inference,
simulation, data parsing, and supporting helpers.

## Running tests

```bash
# Default run — slow tests excluded, coverage ≥ 80% enforced
uv run pytest

# Single file
uv run pytest tests/test_fitting.py -v

# Slow tests only
uv run pytest -m slow

# Single test by name
uv run pytest -k "test_name"
```

## Test organization

Every test file follows the canonical layout documented in `CLAUDE.md ## Tests`:

- Module docstring (mandatory).
- Optional non-ruff pragmas (e.g. `# mypy: disable-error-code=…`).
- Imports.
- Module-level helpers, fixtures, and constants (if any).
- One `class Test<Subject>:` grouping all tests for that module.

All `test_` methods take `self`. Module-level helpers and fixtures stay at module scope
(not inside the class). Cross-file fixtures live in `conftest.py`.

## Test files

| File | What it covers |
|------|---------------|
| `test_birth_death_process.py` | `BirthDeathProcess` init, manuscript addition, serialisation |
| `test_log_likelihood.py` | `BirthDeathProcess.log_likelihood()` — parameter validation, edge cases |
| `test_mathematical_functions.py` | Pure-function math: extinction probability, spine/non-spine densities, population probabilities |
| `test_mathematical_derivatives.py` | Numerical derivative calculations |
| `test_likelihood.py` | `log_likelihood()` / `negative_log_likelihood()` wrappers, spine marginalisation |
| `test_fitting.py` | MLE fitting via LBFGS/Adam, `ParameterConstraints`, covariance estimation |
| `test_bayesian_fitting.py` | Pyro-based posterior inference |
| `test_simulation.py` | Gillespie simulator, reproducibility, DataFrame structure |
| `test_splitting_simulation.py` | Multilevel splitting simulations and conditioned survival |
| `test_probability_normalization.py` | Densities integrate / mass functions sum to 1 |
| `test_asymptotic_consistency.py` | MLE recovers known parameters in large-sample regime |
| `test_lbfgs_exception_inconsistency.py` | LBFGS exception-handling edge cases |
| `test_lbfgs_improved_once.py` | Convergence detection when LBFGS improves exactly once |
| `test_population_probability.py` | `calculate_populations_log_lik` edge cases and bounds |
| `test_parameter_constraints.py` | `ParameterConstraints` validation and equality-group logic |
| `test_data_cleaning.py` | `clean_manuscript_dates`, quality reports, manual corrections |
| `test_ntvmr.py` | NTVMR download helpers, XML formatting, error detection |
| `test_xml_processing.py` | `analyze_all_manuscripts` batch XML parsing |
| `test_german_pattern_parsing.py` | German biblical abbreviation parsing from `contentOverview` strings |
| `test_pattern_analysis.py` | Pattern analysis utilities |
| `test_constants.py` | Canonical book ordering, German abbreviations, section patterns |
| `test_single_epoch_analysis.py` | Single-epoch analysis helpers |
| `test_integration.py` | End-to-end workflows (simulation → fitting) |
| `test_cli.py` | CLI entry points |
| `test_plotting.py` | Visualization / Plotly utilities |
| `test_validation_helpers.py` | Validation helper functions |
| `test_spike_detection.py` | Spike detection utilities |
| `test_reports.py` | Report generation helpers |
| `test_table_formatting.py` | Table formatting utilities |
| `test_json_safe.py` | `json_safe` type-conversion utility |
| `test_basic.py` | Basic smoke tests |
| `test_lxml_conversion.py` | XML processing with lxml |
| `test_manuscript_analysis.py` | Manuscript analysis helpers |
| `test_utils.py` | Miscellaneous utility functions |
| `test_probability_normalization.py` | Normalization checks |

## Adding new tests

Follow the canonical layout in `CLAUDE.md ## Tests`. Key points:

- One `class Test<Subject>:` per file, named after the module under test.
- Module docstring is mandatory (`D100` is enforced by ruff).
- `S101`, `D101`, and `D102` are centrally ignored for `tests/*` in `pyproject.toml` —
  do not add per-file `# ruff: noqa` lines.
- Slow tests (those hitting real network calls or expensive optimizations) should be
  marked `@pytest.mark.slow` so they are excluded from the default run.
