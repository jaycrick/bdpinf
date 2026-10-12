# bdpinf

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/jaycrick/bdpinf/graph/badge.svg?token=5QNFKWSL7G)](https://codecov.io/gh/jaycrick/bdpinf)

Inference, simulation, and visualisation for linear birth-death (branching)
processes: MLE fitting (JAX, JIT-compiled) and Bayesian inference
(NumPyro/NUTS) for single- and multi-epoch birth/death rates, Gillespie and
lifespan-at-birth simulators, multilevel splitting for rare-event
extinction-probability estimation, and a shared Altair chart theme/export
chokepoint.
No domain-specific content --- see
[`nt-manuscripts`](https://github.com/jaycrick/nt-manuscripts) for the New
Testament manuscript application built on top of this library.

## Installation

Install [uv](https://docs.astral.sh/uv/) if you haven't already:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install dependencies:

```sh
git clone https://github.com/jaycrick/bdpinf.git
cd bdpinf
uv sync
```

For development (includes testing and documentation tools):

```sh
uv sync --group dev
```

## Quick Start

```python
from bdpinf import BirthDeathProcess
from bdpinf.inference.fitting import fit_birth_death_processes

model = BirthDeathProcess(creation_times=[100, 200, 300], epoch_end_times=500)
results = fit_birth_death_processes(
    list_of_ages=[model.ages],
    times=model.epoch_end_times,
)
model.update_parameters(results["birth_rates"], results["death_rates"], results)
```

Render the research-notebook site (notebooks live under `qmd_scripts/`; output
goes to `_output/`).
This includes the slow, multi-chain Bayesian validation
notebooks --- see "HPC / batch rendering" below for a faster alternative that
precomputes the heaviest simulate-then-fit datasets first:

```sh
uv run quarto render
```

## Key Features

- Birth-death process modeling: MLE fitting (JAX, JIT-compiled) and Bayesian
  inference (NumPyro/NUTS) for single- and multi-epoch birth/death rates,
  with a Quarto research-notebook site of validation studies against
  synthetic data with known ground-truth parameters
- Gillespie and lifespan-at-birth simulators, plus multilevel splitting for
  estimating small extinction probabilities
- A shared Altair chart theme and export chokepoint (`visualization.save_chart`)
- Generic KDE-quantile date-range resampling (`analysis.date_density`) and its
  vectorized bootstrap extension (`analysis.bootstrap_date_density`) for
  quantifying uncertainty over interval-censored dates, independent of any
  particular dataset

## Date-range uncertainty

Given a dataset where each item is dated as an interval rather than a point
(e.g. `[earlyYear, lateYear]`), this library resolves that in two stages.

### Deterministic KDE-quantile resampling

`quantile_resample_dates(early, late, current_year, n_samples)` (in
`analysis/date_density.py`) assigns one integer year to each item via a
boundary-corrected weighted Gaussian KDE:

1. Each range `[early, late]` is expanded into `n = late - early + 1`
   pseudo-points with weight `1/n` (exact dates are unchanged).
2. The pseudo-points feed a weighted `gaussian_kde` with Scott's-rule bandwidth
   and boundary correction (reflection at age 0 and at the oldest item).
3. The CDF is inverted at Hazen positions `(k - 0.5) / N` to produce one
   sorted resampled year per item.

See `docs/date_uncertainty.qmd` in the application repo for the full
mathematical derivation and its NT-manuscript-specific wrapper
(`add_resampled_year_column`); see `analysis/date_density.py` and
`analysis/validation_helpers.py` here for the generic implementation.

### Bootstrap uncertainty quantification

`analysis/bootstrap_date_density.py`'s `bootstrap_quantile_resample_dates`
propagates date uncertainty through downstream inference by resampling the
dataset itself with replacement (not just the KDE), applying the identical
KDE-quantile pipeline to each bootstrap draw.
The vectorized implementation
avoids calling scipy in a loop by exploiting the fixed integer-year support:
per-item multiplicities are the only bootstrap-varying quantity, and all
bandwidth and KDE computations reduce to batched matrix operations
(`k @ A / N`, chunked einsum).
`save_bootstrap_samples`/`load_bootstrap_samples`
persist results to a compressed `.npz`:

```python
from bdpinf.analysis import load_bootstrap_samples

result = load_bootstrap_samples("bootstrap_samples/full.npz")
resampled_years = result["resampled_years"]  # int16 [n_bootstrap, n]
```

The application repo's `scripts/generate_bootstrap_kde_samples.py` is the
concrete driver for the NT-manuscript dataset; see it for a worked example.

## Development

Setup, checks (tests, Ruff, basedpyright, prose linting) and the notebook and
API-reference builds are documented in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Result numbers from the accompanying paper (in preparation) appear as
`[paper: Theorem ?]` placeholders until they are fixed; search for `[paper:`.

### HPC / batch rendering

`scripts/generate_recovery_studies.slurm` precomputes the simulate-then-fit
datasets consumed by the heaviest validation notebooks, chained via
`scripts/render_all.sh` before `scripts/render_cpu_parallel.slurm` renders
everything (including the Bayesian NUTS validation notebooks).
GPU rendering
was benchmarked and found 1.6×--8× *slower* than CPU on every config tested,
so Bayesian notebooks render on CPU too; see
`docs/design/GPU_RENDERING_PROTOCOL.md`.
