# bdpinf

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/jaycrick/bdpinf/graph/badge.svg?token=5QNFKWSL7G)](https://codecov.io/gh/jaycrick/bdpinf)

A Python platform for downloading New Testament manuscript data from the NTVMR API,
running quality control and analysis, and fitting birth-death process models
(MLE via JAX, Bayesian via NumPyro) to estimate manuscript creation/survival rates.

## Installation

Install [uv](https://docs.astral.sh/uv/) if you haven't already:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install dependencies:

```sh
git clone <repository-url>
cd bdpinf
uv sync
```

For development (includes testing and documentation tools):

```sh
uv sync --all-groups
```

## Quick Start

Download and analyze manuscripts (the `bdpinf` console script is installed by `uv sync`):

```sh
uv run bdpinf ntvmr download-all
uv run bdpinf ntvmr analyze --manuscripts-dir manuscripts --output manuscript_analysis.csv
```

Run inference from cleaned data:

```sh
uv run bdpinf inference --input-csv manuscript_analysis_cleaned.csv --plot
```

Render the research-notebook site (notebooks live under `qmd_scripts/`; output
goes to `_output/`; see `docs/cli.md` for the full CLI reference).
This
includes the slow, multi-chain Bayesian notebooks --- see "HPC / batch
rendering" below for a faster, resource-bucketed alternative:

```sh
uv run quarto render
```

## Key Features

- NTVMR data collection with resume support plus git-friendly XML formatting and validation
- Quality control, book detection, and section alignment
- Birth-death process modeling: MLE fitting (JAX, JIT-compiled) and Bayesian
  inference (NumPyro/NUTS) for single- and multi-epoch creation/death rates,
  with a Quarto research-notebook site of worked analyses and validation studies
- Manuscript date uncertainty pipeline: KDE-quantile resampling of `[earlyYear,   lateYear]` intervals into representative integer years, with a 100,000-draw
  bootstrap for downstream uncertainty quantification

## Manuscript date uncertainty

Every NTVMR manuscript is dated as an interval rather than a point.
The
pipeline resolves this in two stages.

### Deterministic KDE-quantile resampling

`add_resampled_year_column(df)` (in `data/data_cleaning.py`) assigns one
integer year to each manuscript via a boundary-corrected weighted Gaussian KDE:

1. Each range `[earlyYear, lateYear]` is expanded into `n = late - early + 1`
   pseudo-points with weight `1/n` (exact dates are unchanged).
2. The pseudo-points feed a weighted `gaussian_kde` with Scott's-rule bandwidth
   and boundary correction (reflection at age 0 and at the oldest manuscript).
3. The CDF is inverted at Hazen positions `(k - 0.5) / N` to produce one
   sorted resampled year per manuscript; years are assigned to manuscripts
   sorted by `midYear`.

The result is stored as `resampledYear` and used as the default age input to
the birth-death models.

See `docs/date_uncertainty.qmd` for the full mathematical derivation; see
`analysis/date_density.py` and `analysis/validation_helpers.py` for the
implementation.

### Bootstrap uncertainty quantification

To propagate date uncertainty through downstream inference, run:

```sh
uv run python scripts/generate_bootstrap_kde_samples.py
```

This generates `bootstrap_samples/<subgroup>.npz` for seven manuscript
subgroups (`full`, `non_lectionary`, `lectionary`, `section_e`, `section_a`,
`section_p`, `section_r`), each containing 100,000 alternative date
assignments as aligned `uint16`/`int16` arrays of shape `[100_000, n]`.

Each bootstrap draw resamples the manuscript collection with replacement and
applies the identical KDE-quantile pipeline to the bootstrapped dataset.
The
vectorized implementation avoids calling scipy in a loop by exploiting the
fixed integer-year support: per-manuscript multiplicities are the only
bootstrap-varying quantity, and all bandwidth and KDE computations reduce to
batched matrix operations (`k @ A / N`, chunked einsum).

Load samples for downstream use:

```python
from bdpinf.analysis import load_bootstrap_samples

result = load_bootstrap_samples("bootstrap_samples/full.npz")
resampled_years = result["resampled_years"]  # int16 [100_000, 5833]
```

Regenerate via the script above; the `bootstrap_samples/` directory is
`.gitignore`d (files are 50--200 MB each compressed).

## Development

Run tests:

```sh
uv run pytest
```

Generate API reference docs:

```sh
uv run great-docs build
```

Preview the research-notebook site:

```sh
uv run quarto preview
```

### HPC / batch rendering

Notebooks render in two CPU-resource buckets via `scripts/render_*.slurm`
(light/serial vs. embarrassingly-parallel).
GPU rendering of the Bayesian
notebooks was benchmarked and found 1.6×--8× *slower* than CPU on every
config tested, so they render on CPU too; see
`docs/design/GPU_RENDERING_PROTOCOL.md`.
