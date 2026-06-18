# Contour process simulation

## Overview

This document describes the contour-process based simulator introduced in
`src/inf_bp_texts/simulation/simulator.py`.

## Concepts

- Spine component: the distinguished lineage. Built as an excursion started
  at the terminal time `T` that alternates between downward (death) and
  upward (birth) moves. If an upward jump would hit the terminal time `T`,
  the spine generation restarts from `T` (reset rule).

- Non-spine component: a single excursion started at `T` that alternates
  death/birth moves and terminates when reaching 0.

- Contour process list: the concatenation

  reverse(spine_component) + non_spine_component[1:]

  reversed spine ensures the root (0) appears first, then up-and-down
  alternations consistent with the stack-based reconstruction.

## Construction algorithm

1. Produce a spine component using `_generate_spine_component`.
2. Produce a non-spine component using `_generate_non_spine_component`.
3. Build the contour list with `_build_contour_process_list`.
4. Validate with `_validate_contour_process_list` (odd length, start/end 0,
   strict alternation of peaks/valleys).
5. Reconstruct a canonical `BirthDeathProcess` with
   `_contour_process_list_to_birth_death_process`.

## Reconstruction (stack algorithm)

- The contour is interpreted as alternating death and birth times,
  starting and ending at 0.
- Use two stacks: one for birth times and one for manuscript numbers.
- Manuscript 0 (root) is created with creation time 0.
- On reading a birth time, pop the birth-time stack until the top birth time
  is strictly less than the new birth time; the manuscript number on the
  top of the manuscript stack becomes the parent of the new manuscript.
- This reproduces the usual genealogical reconstruction from a contour.

## Assumptions and invariants

- Rates (`birth_rate`, `death_rate`) are positive.
- Terminal time `T` is positive.
- Generated contour lists must be strictly alternating and start/end with 0.
- The public API returns a `BirthDeathProcess` canonical object (use
  `BirthDeathProcess.from_manuscript_records` internally) so downstream
  code sees the same representation as other simulators in the project.

## Spine / survival semantics

- Manuscripts whose recorded death time equals the terminal time `T` are
  treated as still alive at observation and their `death_time` is stored as
  ``NaN`` in the canonical `manuscript_records` table. This ensures the
  `survived` boolean and `age` fields reflect survival to `T`.
- Exactly one manuscript will be marked `is_spine = True`: the first
  surviving manuscript (the earliest created manuscript that is alive at
  `T`) — this matches the convention used by the genealogy ordering
  utilities.

## Reproducibility and RNG

- All draws use `numpy.random.default_rng` style `Generator` objects.
- The public entrypoint accepts a `seed` and derives child `Generator`s in
  the local helper functions to make deterministic, isolated draws.

## Usage

Call `simulate_birth_death_process_from_contour_process(...)` from the
`inf_bp_texts.simulation` package. See the function docstring and
`src/inf_bp_texts/simulation/simulator.py` for parameter descriptions and
examples.

## Notes and next steps

- The reconstructed manuscripts currently have their `is_spine` field set
  using the canonical `BirthDeathProcess` logic (the genealogical ordering
  functions mark the first surviving lineage as the spine). If a different
  `is_spine` convention is desired (e.g. mark all spine-origin manuscripts
  explicitly), modify `_contour_process_list_to_birth_death_process` to
  set the `is_spine` flag during record creation.
- Add cross-references to the user-facing docs if desired.
