# Implementation Plan: Plotly → Altair Migration

**Status:** in progress · **Author:** review by Claude · **Date:** 2026-06-12 · **Last updated:** 2026-06-17

## Progress summary

  | Phase                          | Status         | Completed  |
  | ------------------------------ | -------------- | ---------- |
  | 0 — Spike & gate               | ✅ Done        | 2026-06-12 |
  | 1 — Theme + export chokepoint  | ✅ Done        | 2026-06-12 |
  | 2 — Easy helpers + review gate | ✅ Done        | 2026-06-12 |
  | 3 — Moderate helpers           | ✅ Done        | 2026-06-13 |
  | 4 — Hard helpers + tests       | ✅ Done        | 2026-06-13 |
  | 5 — Notebooks (18 of 19)       | 🔄 In progress | see below  |
  | 6 — Remove Plotly              | ⏳ Pending     | —          |

### Phase 5 notebook status

  | Notebook group                                          | Status                                   |
  | ------------------------------------------------------- | ---------------------------------------- |
  | `01_data_exploration/manuscript_data_exploration.qmd`   | ✅ Done                                  |
  | `01_data_exploration/kashouh_data_exploration.qmd`      | ✅ Done                                  |
  | `02_single_epoch/kashouh_single_epoch_inference.qmd`    | ✅ Done                                  |
  | `02_single_epoch/ntvmr_single_epoch_inference.qmd`      | ✅ Done                                  |
  | `03_multi_epoch/kashouh_multi_epoch_inference.qmd`      | ✅ Done                                  |
  | `03_multi_epoch/ntvmr_birth_death_inference.qmd`        | ✅ Done                                  |
  | `03_multi_epoch/ntvmr_initial_population_inference.qmd` | ✅ Done                                  |
  | `03_multi_epoch/optimizer_comparison.qmd`               | ✅ Done                                  |
  | `03_multi_epoch/single_vs_multi_epoch_comparison.qmd`   | ✅ Done                                  |
  | `validation/single_epoch/` (all 5 notebooks)            | ✅ Done                                  |
  | `validation/multi_epoch/` (all 5 notebooks)             | ✅ Done                                  |
  | `validation/bayesian_multi_epoch/` (3 notebooks)        | ✅ Done (no figures; were already clean) |
  | `04_bayesian/kashouh_family_k_bayesian_inference.qmd`   | ❌ Remaining                             |

### What remains

1. **Phase 5 final notebook** — `04_bayesian/kashouh_family_k_bayesian_inference.qmd`: 4 figures
   (`prior_posterior` 2×2 KDE, `posterior_rates` bar, `trace_plots` 2×2 line, `ppc` line+band),
   all currently using `make_subplots` + `go.Scatter`. Excluded from default render (slow); do last.

2. **Phase 6** — Remove `plotly`, `kaleido`, `plotly-stubs` from `pyproject.toml` once the Bayesian
   notebook is done and `grep -rn "plotly" src tests qmd_scripts pyproject.toml` is empty.

--------------------------------------------------------------------------------

This is the actionable companion to [`PLOTLY_TO_ALTAIR_EVALUATION.md`](./PLOTLY_TO_ALTAIR_EVALUATION.md).
The evaluation answered *whether* to migrate (verdict: yes, phased, Altair-based). This document
answers *how*, in concrete ordered steps, and folds in a **review of the current visualization code**
so that the migration also fixes existing clarity / best-practice debt rather than porting it verbatim.

The migration is sequenced **easiest-first with an explicit user review gate after the first batch**,
then iterates through progressively harder figures, exactly as requested.

--------------------------------------------------------------------------------

## Part A --- Review of current visualization practices

Findings from reading all five helper modules (`plotting.py`, `parameter_plots.py`,
`splitting_plots.py`, `age_fit_plot.py`, `genealogy.py`) plus the 19 notebooks that build figures.
Each finding notes whether the migration is the natural moment to fix it.

### A1. Dead / duplicated styling state (fix during migration)

- **`layout_config`** (`plotting.py:56–77`) is defined but **never referenced anywhere**. Dead code.
- **`my_blue … my_gray`** (`plotting.py:38–44`) — seven individual colour constants, **none referenced**
  outside their own definition. Only the `ptolqual` list is actually used. Dead code.
- **`apply_tufte_legend`** (`plotting.py:420`) is **never called** by any module or notebook. Dead export.
- **`font_config` / `title_font_config`** are duplicated inline inside `apply_professional_styling`
  rather than reused.

→ In Altair this all collapses into **one registered theme**. These should not be ported; they
should be deleted and replaced by the theme chokepoint (Phase 1).

### A2. Weak typing (fix during migration)

- `apply_professional_styling`, `apply_tufte_axes`, `apply_tufte_legend`, `save_figure` all take and
  return `fig: Any`. `plot_creation_year_fit` ends with `# type: ignore[return-value]`.
- Mixed import spelling: `plotly.graph_objs as go` (most files) vs `plotly.graph_objects as go`
  (`age_fit_plot.py:16`). Cosmetic, but indicative of copy-paste drift.

→ Altair ships real types; the helper signatures become `-> alt.Chart` / `-> alt.LayerChart` etc.,
removing every `Any` and the `type: ignore`.

### A3. `save_figure` swallows all errors silently (fix during migration)

`save_figure` (`plotting.py:504`) wraps the entire body in `try/except Exception`, logs a warning,
and returns `None`. A figure that fails to export produces **no error and no file** --- easy to miss in
a 50-figure render. It also **mutates the passed figure in place** (size, margins, transparent bg) as a
side effect, so the same `fig` object renders differently inline vs on disk.

→ The Altair `save_figure` (`chart.save(..., engine="vl-convert")`) should (a) not catch-all-swallow ---
let export errors surface, or fail loudly with the filename, and (b) be pure: take a chart, write a
file, return the path, without mutating display state.

### A4. Overlapping birth-time histogram helpers (consolidate during migration)

Two functions render essentially the same "observed creation-year histogram + fitted birth-time
density overlay" figure with duplicated scaling logic:

- `save_multi_epoch_birth_time_figure` (`plotting.py:632`)
- `plot_creation_year_fit` (`age_fit_plot.py:31`)

They differ mainly in input plumbing (one takes a `manuscripts_df`, the other an `ages` array) and in
how they handle the zero-spread edge case (`age_fit_plot` handles it; `plotting.py` does not). The
density-scaling maths (`density * n * bin_width`) is copied in both.

→ Migrate to **one** Altair helper that takes a tidy frame of observed years + a density grid, with
the scaling done once. Keep two thin adapters if the call sites genuinely need different inputs.

### A5. `plot_parameter_consistency` mixes statistics with rendering (refactor during migration)

`plot_parameter_consistency` (`plotting.py:150--417`) is a ~270-line function that **computes bootstrap
CIs inline** (mean/median/variance percentile intervals) *and* builds a 3-column × N-row subplot grid
by appending ~12 traces per row by hand. Colour assignment uses fragile modular indexing
(`ptolqual[(row_index + 2) % len]`).

→ This is the single biggest beneficiary of the migration. Split into:
1. a pure `compute_consistency_bands(...) -> pd.DataFrame` returning **tidy long-form** data
   (columns: `regime, T, statistic, estimate, lower, upper, true_value`), and
2. a tiny Altair renderer using `.facet(row="regime", column="statistic")` + `mark_area(y, y2)` for
   the band + `mark_line` for the estimate + `mark_rule` for the true value.

The hand-built 36-trace loop becomes ~15 lines of declarative spec. The statistics function is
independently unit-testable (golden-value, per the project's testing strategy).

### A6. Inconsistent application of the house style (fix during migration)

`create_kde_density_plot` and `create_parameter_box_plot` (`parameter_plots.py`) return figures
**without** `apply_professional_styling` / `apply_tufte_axes`, unlike every other helper. So the KDE and
box plots are visually off-theme relative to the rest of the site.

→ Under a registered Altair theme this inconsistency disappears by construction --- the theme applies
globally, so there is no per-figure styling call to forget.

### A7. Plotting logic lives in notebooks, against the repo convention (opportunistic fix)

CLAUDE.md states *"Keep logic in `src/` ... heavy computation belongs in the package, not inline."* Yet
there are **61 inline `go.Figure`/`make_subplots` constructions across 19 notebooks**, many with
bespoke `update_layout` blocks. The migration touches every one of these cells.

→ Where a notebook figure is a near-duplicate of another (e.g. the recurring NLL-convergence line,
the rates bar chart, the pairs scatter), promote it to a reusable `visualization/` helper during its
phase rather than porting the inline Plotly to inline Altair. This shrinks the notebook surface and
is the difference between "port 61 inline figures" and "port ~20 helpers + thin call sites".

### A8. Broad `except Exception` around overlays (minor)

`save_multi_epoch_birth_time_figure`, `plot_creation_year_fit`, and `plot_parameter_consistency`'s
`_fill_color` all use bare `except Exception`. The overlay ones are arguably defensible (skip a
density curve rather than crash a render) but should narrow to the expected failure and keep the
warning. Carry the behaviour forward, narrowed.

--------------------------------------------------------------------------------

## Part B --- Migration principles

1. **Additive, never big-bang.** Add Altair *alongside* Plotly. Plotly stays installed and working
   until the final phase. Every phase leaves `uv run pytest` and `uv run quarto render` green.
2. **One theme chokepoint.** All house style (palette `ptolqual`, Tufte axes, serif font, light bg)
   lives in a single registered Altair theme. No per-figure styling calls. This replaces A1's dead
   constants and A6's inconsistency in one move.
3. **One export chokepoint.** `save_figure` becomes a thin `chart.save(..., engine="vl-convert")`
   wrapper — pure, loud on failure (fixes A3).
4. **Tidy data first.** Every hard figure (A5, genealogy) is split into a pure data-prep function
   (unit-testable) + a small declarative spec. Altair *wants* long-form frames; lean into it.
5. **Promote, don't port.** When a notebook figure duplicates another, lift it into a helper (A7)
   instead of porting inline Plotly to inline Altair.
6. **Review gates.** After the first easy batch is rendered to PDF + inline, **stop for user review**
   before continuing. Re-confirm visual parity at each subsequent phase boundary.

### Decision to confirm before Phase 1 --- math typography

The evaluation flagged one genuine fork: Vega-Lite renders axis/label math as **plain Unicode**
(`λ`, `μ`), not LaTeX. The figures currently rely on Unicode under Plotly and look fine, so Altair is
appropriate. **If** the thesis later needs LaTeX-typeset math (`$\lambda_i$`) baked into vector PDFs,
matplotlib+PGF would be the target instead. Proceeding on the assumption that **Unicode is
acceptable** (status quo). Flag now if not --- it changes the target library, not just the plan.

--------------------------------------------------------------------------------

## Part C --- Phased implementation

Difficulty and ordering follow the evaluation's feature table, refined against the actual code.

### Phase 0 --- Spike & gate (½ day)

**Goal:** prove publication-quality export and de-risk the hardest figure before committing.

- Add `altair>=5` and `vl-convert-python` to a **dev dependency group** only (not yet `[project]`).
- Reproduce in Altair, in a scratch notebook: **(a)** one easy figure (`plot_empirical_population_mean`)
  and **(b)** the riskiest one (`plot_genealogy_tree`, via `mark_rule` segments from a tidy frame).
- Export both to PDF with vl-convert; eyeball against current Plotly PDFs.

**Gate:** confirm (1) vl-convert PDF quality ≥ kaleido, (2) the genealogy tree is expressible in
Altair `mark_rule` (if not, decide now: raw Vega spec, or leave the tree on Plotly permanently). Do
not proceed until the tree path is chosen.

### Phase 1 --- Theme + export chokepoint (1 day)

Foundation reused by every later phase.

- New module `visualization/altair_theme.py`: register an Altair theme porting `ptolqual`, the serif
  font stack, `#FAFAFA` background, and Tufte axes (thin lines, outward muted ticks, no gridlines).
  `alt.themes.register("inf_bp_texts", ...)` + `alt.themes.enable(...)`.
- New `save_chart(chart, filename, *, width, height, output_dir="figures") -> Path` using
  `chart.save(path, engine="vl-convert")`. **Pure, no catch-all, fails loud with filename** (fixes A3).
- Configure Quarto/Jupyter to render Altair inline (it does so natively; verify on one notebook).
- **Delete** `layout_config`, `my_blue…my_gray`, and `apply_tufte_legend` (A1) — confirm zero
  references first (already verified: none).

No figures migrated yet; Plotly helpers untouched.

### Phase 2 --- Easy helpers + **REVIEW GATE 1** (1--2 days)

The "replace the easiest plots initially" batch. All are single-series line/scatter/bar.

  | Helper                                | Module               | Altair                                                  |
  | ------------------------------------- | -------------------- | ------------------------------------------------------- |
  | `plot_empirical_population_mean`      | `splitting_plots.py` | `mark_line(point=True)`                                 |
  | `plot_level_survival_probabilities`   | `splitting_plots.py` | `mark_line(point=True)`, y∈[0,1]                        |
  | `plot_conditioned_trajectories`       | `splitting_plots.py` | `mark_line`, `color="particle_id"` (watch 5000-row cap) |
  | `save_multi_epoch_convergence_figure` | `plotting.py`        | `mark_line(point=True)`, `axis=Axis(format="d")`        |
  | `save_multi_epoch_fit_figure`         | `plotting.py`        | `mark_bar` + `mark_text`, `.hconcat()` two panels       |

Render each to PDF **and** inline in its host notebook.

> ### ⛔ REVIEW GATE 1 --- user reviews these five figures
>
> Place new Altair PDFs side-by-side with the Plotly PDFs in `figures/`. **Stop here.** Do not start
> Phase 3 until the user has reviewed axes, colours (`ptolqual`), fonts, and overall feel, and
> confirmed the theme is right. Theme adjustments are cheapest to make now, before 50 more figures
> inherit them.

### Phase 3 --- Moderate helpers (2--3 days)

Begin only after Gate 1 sign-off.

- **Consolidate** `save_multi_epoch_birth_time_figure` + `plot_creation_year_fit` into one Altair
  histogram-plus-density helper (fixes A4): `mark_bar` (binned) layered with `mark_line` density.
- `create_kde_density_plot` → `mark_line` density + `mark_rule` true-value line. The Plotly **arrow
  annotation** has no Vega-Lite primitive → replace with `mark_text` label + small `mark_point`
  (documented fidelity compromise from the evaluation).
- `create_parameter_box_plot` → `mark_boxplot` + layered `mark_point` (true/mean/median). Note
  `go.Box(boxmean=True)`'s mean line isn't native → reproduce with a layered `mark_tick`.

**Review checkpoint** (lighter than Gate 1): user eyeballs the KDE arrow workaround and box-plot mean
tick specifically, since those are the two fidelity compromises.

### Phase 4 --- Hard helpers (3--4 days)

- **`plot_parameter_consistency`**: execute the A5 refactor. First land
  `compute_consistency_bands(...) -> pd.DataFrame` (tidy long form) with a golden-value test, then the
  Altair `.facet(row="regime", column="statistic")` renderer with `mark_area(y, y2)` CI bands,
  `scale=Scale(type="log")` x-axes, `mark_rule` for true values.
- **`plot_genealogy_tree`**: per the Phase 0 decision. If Altair: tidy segment frame →
  `mark_rule` (x/x2 for lifetimes, dashed connectors), reversed ordinal y with custom labels via
  `sort`/`scale`. If the spike showed it's intractable: leave on Plotly as the documented permanent
  exception, or drop to a raw Vega spec.
- Update `models/birth_death_process.py:570` return type (`-> go.Figure` → Altair) and its import.
- Rework the three test files (`tests/test_plotting.py`, `test_genealogy.py`,
  `test_birth_death_process.py`) to assert on `chart.to_dict()` specs instead of `go.*` /
  `monkeypatch` of `write_image`. `test_plotting.py` has the most assertions (~16).

### Phase 5 --- Notebooks, file by file (the long pole, 1--2 weeks)

19 notebooks, 61 inline figures. Convert in dependency order so reusable helpers land before their
consumers:

1. `01_data_exploration/` — `manuscript_data_exploration.qmd` (8 figs, heaviest),
   `kashouh_data_exploration.qmd` (4).
2. `02_single_epoch/` — `kashouh_…` (4), `ntvmr_…` (2).
3. `03_multi_epoch/` — `kashouh_multi_epoch_inference.qmd` (8), `optimizer_comparison.qmd` (4),
   `ntvmr_birth_death_inference.qmd` (3), `single_vs_multi_epoch_comparison.qmd` (2),
   `ntvmr_initial_population_inference.qmd` (1).
4. `validation/single_epoch/` and `validation/multi_epoch/` — mostly 1–4 figs each;
   `06_bulk_inference.qmd` and `05_bulk_inference_zero_birth.qmd` are the largest (4 each).
5. `04_bayesian/kashouh_family_k_bayesian_inference.qmd` (4) — excluded from default render (slow);
   do last.

**Per notebook:** apply A7 --- if a figure duplicates an existing helper, call the helper; only write
inline Altair for genuinely one-off charts. After each file: `uv run quarto render` that file, diff
the inline output, run `lintquarto` on it. Replace every `fig.show()` with returning/`display(chart)`.

### Phase 6 --- Remove Plotly (½ day)

Only once **zero** Plotly references remain in `src/`, `tests/`, `qmd_scripts/`.

- Drop `plotly>=6.5.2`, `kaleido>=1.2.0`, `plotly-stubs>=0.1.2` from `pyproject.toml`; `uv sync`.
- Promote `altair` / `vl-convert-python` from dev group to `[project]` dependencies.
- Full gate: `uv run pre-commit run --all-files`, `uv run pytest` (coverage ≥80%),
  `uv run lintquarto -l ruff pyright -p qmd_scripts docs -e .venv _output reference`,
  `uv run quarto render`.
- `grep -rn "plotly\|kaleido" src tests qmd_scripts pyproject.toml` → must be empty.

--------------------------------------------------------------------------------

## Part D --- Per-figure difficulty inventory

  | #   | Figure / helper                                 | Location                         | Difficulty     | Phase |
  | --- | ----------------------------------------------- | -------------------------------- | -------------- | ----- |
  | 1   | `plot_empirical_population_mean`                | `splitting_plots.py`             | Easy           | 2     |
  | 2   | `plot_level_survival_probabilities`             | `splitting_plots.py`             | Easy           | 2     |
  | 3   | `plot_conditioned_trajectories`                 | `splitting_plots.py`             | Easy (row cap) | 2     |
  | 4   | `save_multi_epoch_convergence_figure`           | `plotting.py`                    | Easy           | 2     |
  | 5   | `save_multi_epoch_fit_figure`                   | `plotting.py`                    | Easy–Mod       | 2     |
  | 6   | birth-time histogram + density (×2, **merge**)  | `plotting.py`, `age_fit_plot.py` | Moderate       | 3     |
  | 7   | `create_kde_density_plot` (arrow workaround)    | `parameter_plots.py`             | Moderate       | 3     |
  | 8   | `create_parameter_box_plot` (mean-tick)         | `parameter_plots.py`             | Moderate       | 3     |
  | 9   | `plot_parameter_consistency` (facet + CI + log) | `plotting.py`                    | Hard           | 4     |
  | 10  | `plot_genealogy_tree` (shapes → `mark_rule`)    | `genealogy.py`                   | Hard           | 4     |
  | 11  | 61 inline notebook figures                      | 19 `.qmd`                        | Easy×volume    | 5     |

--------------------------------------------------------------------------------

## Part E --- Risks, testing, rollback

**Risks (from evaluation, still live):**
- *5000-row cap* --- `plot_conditioned_trajectories` and large histograms can trip `MaxRowsError`.
  Mitigate with `alt.data_transformers.enable("vegafusion")` or `disable_max_rows()`; decide policy
  in Phase 1 theme module.
- *Arrow annotation* (KDE) and *box-plot mean line* --- documented fidelity compromises (Phase 3).
- *Genealogy tree* --- gated in Phase 0 with a fallback decision.
- *Jupyter cache* --- migrated cells invalidate `.jupyter_cache`; expect full re-render cycles.

**Verification per phase:**
- Export each new chart to PDF; place beside the Plotly PDF in `figures/` and confirm axes, bands,
  `ptolqual` colours, labels.
- Helper layer: `uv run pytest tests/test_plotting.py tests/test_genealogy.py
  tests/test_birth_death_process.py` after reworking assertions to `chart.to_dict()`.
- `uv run basedpyright --project pyproject.toml src/` (0 errors) and the `lintquarto` command.
- Site: `uv run quarto render` --- every migrated notebook renders inline, `_output/` intact.

**Rollback:** because the migration is additive and Plotly is not removed until Phase 6, any phase can
be reverted by `git revert` of that phase's commits without touching the rest. The point of no easy
return is Phase 6.

**Commit granularity** (per CLAUDE.md): one bundled commit per phase/helper grouping code + tests +
any doc, prefixed `:zap:` for the new Altair figures, `:tv:` for the dependency/tooling swap in
Phase 6. Update `CHANGELOG.md` under `## [Unreleased]` as each helper's public API changes.
