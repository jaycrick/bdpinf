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

Full command list, with the rationale for each, is in `CONTRIBUTING.md`.
The essentials:

- Run `uv run ruff format src tests` and `uv run ruff check --fix src tests` first, and let Ruff fix what it can; do not run standalone `isort` over `src`/`tests`.
- `uv run pytest` (fast suite, no coverage; CI adds `--cov --cov-fail-under=86`, see `CONTRIBUTING.md`), `uv run basedpyright --project pyproject.toml src/` (0 errors), `uv run pre-commit run --all-files` (everything CI runs besides pytest).
- After creating or editing a `.qmd`: `uv run lintquarto -l ruff basedpyright -p qmd_scripts docs -e .venv _output`.
- Always `uv run quarto ...` (plain `quarto` uses the system Python and fails); API reference is `uv run great-docs build`.

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

**bdpinf** is a library for linear birth-death (branching) processes: exact likelihoods, MLE and Bayesian inference, simulation and plotting.
It has no domain-specific content; the New Testament manuscript application lives in the separate `nt-manuscripts` repo, and rare-event simulation methodology in `bdsurv` (a runtime dependency, installed from its public GitHub repository and pinned to release tag `v0.8.0`, used by `simulation/forward.py` and `simulation/conditioned.py`).
Public names, signatures and docstrings are in the generated API reference (`great-docs.yml` defines its structure), so this section only records the layering and the ideas that are not obvious from the code.

### Layers

Imports may only point at the same or a lower layer; `tests/test_layering.py` enforces this and must list every new module.

  | Layer | Package / module                                                                                        | Role                                                                                                      |
  | ----- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
  | 0     | `utils`, `paths`                                                                                        | Small helpers; `find_project_root()` for notebooks and scripts (never hardcode relative paths)            |
  | 1     | `types`, `models.validation`, `models.rate_functions`, `models.math_nested`                             | Input validation, `RateFunction` protocol, the closed-form core                                           |
  | 2     | `models.math_extinction`, `models.math_populations`, `models.math_age_densities`, `models.math_general` | Pure functions built on layer 1                                                                           |
  | 3     | `inference`                                                                                             | `AgeData` and likelihood, rate-model families, constraints, Newton MLE fitting, NumPyro/NUTS, persistence |
  | 4-5   | `models.genealogy`, `process`                                                                           | `BirthDeathProcess`, the canonical object passed between modules                                          |
  | 6     | `simulation`                                                                                            | Exact-event and lifespan-at-birth simulators, survival conditioning, `bdsurv` adapters                    |
  | 7     | `analysis`                                                                                              | Notebook-facing helpers: subset fits, recovery studies, date-density pipelines, LaTeX tables              |
  | 8     | `visualization`                                                                                         | Altair charts (house theme in `altair_theme.py`, the single export path `save_chart`)                     |

### Ideas worth knowing

- **Nested integral $\mathcal W$.** $\mathcal W(b)=1/\pi(b)$ obeys $\mathcal W'=\gamma\mathcal W+\mu$, so for piecewise-constant rates it has an exact epoch-wise recursion (the "restart principle").
  Age and population likelihoods follow from it in $O(K+n\log K)$, in log space, with no sum over genealogy orderings and no division by $\lambda-\mu$ (the critical case is exact).
  `math_nested` is the piecewise-constant array API (`nested_terms(birth, death, times)` then `non_spine_log_density` / `spine_log_density`).
  `math_general` is the general-rate counterpart for any `RateFunction` (piecewise constant, linear, exponential or quadratic, or sinusoidal; `breakpoints` are the interior times at which quadrature is split): `general_terms(birth, death)` does the segment quadrature once (Gauss-Legendre between breakpoints, accumulated with `cumlogsumexp`), after which each age needs two short partial-segment quadratures; `terms_for(birth, death)` picks the exact `math_nested` backend for piecewise-constant rates on shared epochs and quadrature otherwise, and its density functions accept either kind of terms.
  `math_age_densities` keeps the public `*_general` names and the CDFs on top of it.
- **Validation.** `nested_terms` and the density functions validate concrete inputs and skip validation under `jit`/`grad` tracing; `AgeData.from_groups` does the host-side work (spine resolution, counts, range checks) once, and the fitter and Bayesian model call the shared `_log_likelihood_core` on its output.
- **Unknown spine.** When no spine is given, `log_likelihood()` marginalises over which individual is the spine (`resolve_spine_ages` is the shared resolution step).
  `condition_on_survival` (default `True`) conditions the population-count term on non-extinction.
- **Fitting.** `fit(AgeData, BirthDeathModel)` maximises the exact likelihood with a trust-region Newton method (`scipy` `trust-exact`, jitted JAX gradient and Hessian), falling back to L-BFGS-B, and reports a `FitResult` (parameters by name, `at_interior_optimum`, covariance, profile-likelihood intervals via `profile_likelihood_interval`).
  `BirthDeathModel` pairs two `RateModel` families (`PiecewiseConstantModel`, `PiecewiseLinearModel`; a new family only needs `params`, `build`, `constant_values`, `values_from_rate`); every parameter has a transform (log for rates) that maps it to the unconstrained scale.
  `Constraints` (fixed values, equality groups, **by parameter name**) compile to index maps shared by the MLE and NUTS; a jump-form `PiecewiseLinearModel` with `flat_groups` constraints *is* the piecewise-constant model, so constant is nested in linear.
  `FreeKnots` (number, optionally ranges) turns the shared epoch boundaries into `tau<k>` parameters on an `interval` transform; `fit_knots` profiles them (the piecewise-constant likelihood jumps in a knot, so Newton alone is unreliable).
  Defaults are one search per model type: `fit` uses one start, `fit_knots` uses `"brent"` for one free knot and `"grid-newton"` for several; `thorough=True` uses 5 starts in `fit`, and in `fit_knots` runs brent, de-profile and grid-newton in parallel processes and keeps the best (`KnotFitResult.method`, `method_results`).
  The method comparison (sim1000, NTVMR benchmarks and their HPC scripts) is preserved at the `knot-benchmarks` tag.
  `AgeData.from_ranges` takes ages known only to lie in ranges (the density is averaged uniformly over each range; `models.math_age_masses`), which makes the likelihood continuous in a knot.
  The jitted objective is cached per model and constraint structure with the data and fixed values as arguments, and the age vector is padded (masked) to a power-of-two bucket, so repeated fits (bootstrap, profile likelihood, recovery studies) compile once; compile time, not run time, dominates a single fit (`scripts/bench_fitting.py`, `docs/design/FITTING_DERIVATIVES.md`).
  `fit_single_model` / `FitModelConfig` in `analysis.validation_helpers` is the one fit-an-age-vector helper used by the recovery studies.
- **Spline rates.** `SplineModel` (`quadratic`, `cubic_natural`, `cubic_clamped`, `pchip`, `cubic_hermite`) puts the rate values at the knots (log) and scaled slopes (natural value = slope $\times T$, identity) in the parameter vector, built into `PiecewiseQuadraticRate` / `PiecewiseCubicRate` and evaluated by the quadrature backend.
  Splines can dip below zero between knots, so `log_likelihood_rates` is the one guard (`-inf` when a rate's duck-typed `min_values()` is negative; the fitters turn that into the finite penalty and Newton steps are rejected).
  `BirthDeathModel.n_free(constraints)` is the single source of truth for $p$ in AIC/BIC (see `docs/spline_rates.qmd`).
  Optima often sit on that wall (end slope $w'(T^-)\le 0$ under a fixed zero at $T$, or an interior tangency), where plain Newton cannot converge, so `fit` adds a constrained post-step (`inference.constrained`: `nonneg_margins` rows, SLSQP, KKT Newton polish, `verify_kkt`) for rates exposing `nonneg_margins`, reporting `at_guard`, `kkt_residual` and `constrained_optimum`; `fit_knots` runs it only in its final and joint fits.
- **Plots.** Reusable, tested builders live in `visualization`; one-off diagnostic charts are built inline in notebooks with `alt.Chart(...)`, the shared `PTOLQUAL` palette and `save_chart`/`display_svg`.

## Design rules

1. **JAX everywhere**: rates and ages are float64 `jax.Array`s so `jax.grad`/`jax.jit` work and precision holds; `jax.config.update("jax_enable_x64", True)` runs at package import.
2. **Pure math functions, not methods**: the `models.math_*` modules hold free functions with no class state, so the maths stays testable and composable without `BirthDeathProcess`.
3. **Names follow the paper**: $\Gamma,\Lambda,M$ for the rate integrals, $I(T)$ (the *extinction odds*, $P_0/(1-P_0)$), $A$ non-spine age, $A^*$ spine age, $T$ observation time.

## Known challenges

**Multi-epoch convergence**: High-dimensional parameter space with sensitivity at epoch boundaries.
Mitigate by using equality constraints (constant rates across epochs), warm-starting a richer model from a simpler fit (`init=previous_result`), and `n_starts > 1` (the second start swaps birth and death, because ages alone are symmetric in the two rates).

**Weak identifiability**: from survivor ages alone, early-time rates and the death rate are weakly identified (the size term fixes only $\pi(T)$; ages separate the rest slowly), and Wald intervals for growth-type parameters are not asymptotically valid.
Flexible models (many knots) need constraints or a lot of data; prefer `profile_likelihood_interval`.

**Numerical stability**: Likelihoods are always computed in log-space.
Extreme rate values (μ >> λ or vice versa) and ages near epoch boundaries can still cause issues.

## Quarto notebooks (`qmd_scripts/`)

`_quarto.yml` is the single source of truth for the research-notebook website: render globs, navbar menus, and HTML format (output to `_output/`).
Cell output is cached globally (`execute: cache: true`).
**Always use `uv run quarto ...`** --- plain `quarto` uses system Python (no venv) and fails immediately.

The **API reference** is a separate site, generated by **great-docs** from `great-docs.yml` (not part of `_quarto.yml` --- great-docs owns its own Quarto project under `great-docs/`, which cannot host the executable research notebooks because it has no `execute.cache`/`bibliography` equivalent).
See `docs/design/DOCS_TOOLING_EVALUATION.md` for why this is two sites rather than one.

### Directory taxonomy

  | Path                                                      | Purpose                                     | File naming                                                |
  | --------------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------- |
  | `qmd_scripts/validation/single_epoch/`                    | Recover known params from synthetic data    | `<nn>_<topic>.qmd` (e.g. `01_simulation_checks.qmd`)       |
  | `qmd_scripts/validation/multi_epoch/`                     | Same, multi-epoch                           | `<nn>_<topic>.qmd` (e.g. `03_bulk_inference.qmd`)          |
  | `qmd_scripts/validation/bayesian_multi_epoch/`            | Bayesian validation                         | `<nn>_<topic>.qmd`; slow (multi-chain NUTS), render on CPU |
  | `qmd_scripts/validation/survival_conditioned_simulation/` | `bdsurv` simulators vs `bdpinf` theory      | `<nn>_<topic>.qmd`                                         |
  | `qmd_scripts/validation/time_varying_rates/`              | Age-density shapes under time-varying rates | `<nn>_<topic>.qmd`                                         |
  | `docs/*.qmd`                                              | Prose/math documentation                    | no executable cells                                        |
  | `great-docs.yml`                                          | API reference config                        | **drives the separate great-docs site, see above**         |

Real-data analyses live in the application repo (`nt-manuscripts`), not here.

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
    validation/single_epoch/         # prefix vse01_, vse05_, vse06_
    validation/multi_epoch/          # prefix vme01_, vme03_, vme04_, vme05_
    validation/survival_conditioned_simulation/   # prefix vsc01_
    validation/time_varying_rates/   # prefix per notebook
  ```
  In each notebook, set `figures_dir = find_project_root() / "figures" / "<mirror-subpath>"` and pass `output_dir=figures_dir` to every `save_chart` / `save_chart_html` call.
  Filenames follow the pattern `<prefix>_<descriptive>.pdf`, where the prefix is the submodule id (e.g. `vme03`) assigned to each notebook.
  When `output_dir` is omitted, `save_chart` falls back to the module default set via `set_figure_dir()`, then `$BDPINF_FIGURE_DIR`, then `Path.cwd() / "figures"` --- so notebooks that opt out of the mirrored layout still get a `figures/` directory next to wherever they're run from.
  Do not use hardcoded `Path("../../../figures")`.

- **Design docs**: architecture/audit documents live in `docs/design/` (not the repo root).
