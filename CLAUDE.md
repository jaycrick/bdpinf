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

## Commands

```bash
# Install dependencies
uv sync --all-groups

# Run tests (slow tests excluded by default; coverage ≥80% enforced)
uv run pytest
uv run pytest tests/test_fitting.py -v        # single file
uv run pytest -m slow                          # run slow tests only
uv run pytest -k "test_name"                   # single test by name

# Linting and formatting
uv run ruff check src tests
uv run ruff format src tests
uv run isort src tests
uv run mypy src/                               # strict mode

# All at once (pre-commit)
uv run pre-commit run --all-files

# Documentation
uv run quartodoc build
uv run quarto preview
```

## Architecture

**inf-bp-texts** downloads New Testament manuscript data from the NTVMR API, runs quality control, and fits birth-death process models to estimate manuscript creation/survival rates via MLE.

### Core data structure: `BirthDeathProcess`

`models/birth_death_process.py` is the canonical object passed between modules. It holds `creation_times`, `epoch_end_times`, per-epoch `birth_rates`/`death_rates` (all `torch.Tensor` float64), and optional `manuscript_records` for genealogy. Key methods: `log_likelihood()`, `update_parameters()`, `plot_genealogy_tree()`, `to_newick()`.

### Mathematical layer: `models/`

`mathematical_functions.py` (pure functions, no class state):
- `calculate_extinction_pr()` — probability population dies by time T
- `calculate_spine_ages_log_lik()`, `calculate_non_spine_ages_log_lik()` — per-age densities
- `calculate_populations_log_lik()` — probability of observed population counts
- `_reverse_time()` — **time-reversal trick**: transforms multi-epoch non-spine ages into an extinction-conditioned backward-time problem, avoiding exponential complexity in the number of epochs

`mathematical_derivatives.py` — numerical derivatives used during optimization.

### Inference layer: `inference/`

`fitting.py` is the main entry point:
- `fit_birth_death_processes()` — MLE via PyTorch optimizers (LBFGS, Adam); supports initial perturbation for robustness and convergence detection with multi-restart
- `ParameterConstraints` dataclass — fixed parameters (index → value) and equality groups (lists of indices forced equal); validated before optimization to reduce parameter space
- `estimate_mle_covariance()` — covariance via numerical Hessian inversion

`likelihood.py` — `log_likelihood()` / `negative_log_likelihood()` wrappers that handle log-scale vs. positive-scale parameters and marginalize over unknown spine positions.

`bayesian.py` — Pyro-based posterior inference for uncertainty quantification.

### Simulation layer: `simulation/`

`simulator.py` — `simulate_birth_death_process()` runs a Gillespie algorithm with time-varying rates; `predict_manuscript_counts()` predicts future population given current observations via rejection sampling.

`splitting.py` — Multilevel splitting and Fleming–Viot particle methods for rare-event simulation (estimating small extinction probabilities).

### Data pipeline: `ntvmr.py`, `data_cleaning.py`, `xml_processing.py`

`ntvmr.py` — downloads from NTVMR API with resume support; `format_xml_files()` pretty-prints XML and strips Unicode line separators (U+2028/U+2029) for clean git diffs.

`data_cleaning.py` — `examine_manuscript_xml()`, `clean_manuscript_dates()`, `generate_data_quality_report()`.

`xml_processing.py` — `analyze_all_manuscripts()` batch-parses a folder for book/section content.

### Types and constants

`types.py` — Pydantic models `ValidBookSet`, `ValidSectionSet` for validated book/section sets.

`constants.py` — canonical book ordering, German abbreviations, and section patterns used throughout the XML layer.

## Key design decisions

1. **PyTorch everywhere**: All rates and ages are `torch.Tensor` (float64) — enables gradient-based optimization and ensures numerical precision throughout the pipeline.

2. **Pure math functions, not methods**: `mathematical_functions.py` contains only free functions with no class state, keeping the math testable and composable independently of `BirthDeathProcess`.

3. **Time-reversal for multi-epoch efficiency**: The non-spine likelihood is reformulated as an extinction-conditioned backward-time integral. This avoids summing over all possible genealogy orderings (exponential in epoch count) and reduces it to a single pass.

4. **`ParameterConstraints` as first-class input**: Equality groups and fixed parameters are validated early and applied throughout the optimizer loop — reducing effective dimension of the search space for parsimonious models.

5. **Likelihood marginalizes unknown spine**: When the spine (ancestral lineage) is unspecified, `log_likelihood()` sums over all possible spine assignments, so the fitter works with incomplete genealogy data.

## Known challenges

**Multi-epoch convergence**: High-dimensional parameter space with sensitivity at epoch boundaries. Mitigate by: using equality constraints (constant rates across epochs), providing good initial guesses, using LBFGS first then Adam, and setting `perturb_initial_params_std > 0`.

**Numerical stability**: Likelihoods are always computed in log-space. Extreme rate values (μ >> λ or vice versa) and ages near epoch boundaries can still cause issues.

**XML edge cases**: `format_xml_files()` must be run after downloads to clean U+2028/U+2029 characters before parsing. Malformed or empty `originYear` elements surface in `examine_manuscript_xml()`.
