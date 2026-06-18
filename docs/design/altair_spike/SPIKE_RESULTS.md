# Phase 0 Spike --- Results & Gate Decision

Spike for the Plotly→Altair migration (see `../PLOTLY_TO_ALTAIR_IMPLEMENTATION_PLAN.md`, Phase 0).
Reproduce: `uv run python docs/design/altair_spike/phase0_spike.py` (writes 4 PDFs here).

## What was tested

  |                           | Plotly (current)  | Altair (candidate) |
  | ------------------------- | ----------------- | ------------------ |
  | Easy: empirical-mean line | `easy_plotly.pdf` | `easy_altair.pdf`  |
  | Risky: genealogy tree     | `tree_plotly.pdf` | `tree_altair.pdf`  |

Deps added to a **dev-only** group `altair-migration` in `pyproject.toml`:
`altair==6.2.1`, `vl-convert-python==1.9.0`. Plotly/kaleido untouched.

## Findings

### Gate 1 --- vl-convert PDF quality ≥ kaleido → **PASS**

Altair PDF export via vl-convert is crisp vector output, on par with kaleido. No Chromium dependency.

- ⚠️ **Font fallback:** the serif title rendered in the Libertinus serif stack, but axis **tick labels
  fell back to sans-serif** — vl-convert does not see the `Libertinus Serif` system font the way
  kaleido does. **Action for Phase 1:** the registered theme must either (a) point at a font file
  vl-convert can resolve, or (b) accept a sans tick face, or (c) embed the font. Tracked, not a blocker.

### Gate 2 --- genealogy tree expressible in Altair → **PASS (no fallback needed)**

The shapes-based Plotly drawing reproduces declaratively in Altair from the **existing tidy
`genealogy_dfs_order` frame**:
- lifetime segments → `mark_rule` (`x`=birth_time, `x2`=plot_death_time) coloured by survivor/extinct/spine
- parent→child connectors → dotted `mark_rule` (`y`=parent_y, `y2`=child dfs_index)
- reversed root-on-top ordering → ordinal `y` with `sort`

This **removes the largest risk** in the evaluation. The genealogy tree does **not** need raw Vega or
a permanent Plotly exception.

- Refinement for Phase 4 (cosmetic, not a blocker): the current Plotly tree labels the y-axis with
  `manuscript_id` via custom `ticktext`; the spike used the `dfs_index`. The real port should map the
  ordinal y labels back to `manuscript_id` to match current output.

## Decision

**Proceed to Phase 1** (theme + export chokepoint). Both gate criteria pass; Altair is the confirmed
target for *all* figures including the tree. Carry forward two tracked items: vl-convert font
resolution (Phase 1) and y-axis manuscript-id labels for the tree (Phase 4).
