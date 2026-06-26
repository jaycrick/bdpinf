# inf-bp-texts

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/jaycrick/inf-bp-texts/graph/badge.svg?token=5QNFKWSL7G)](https://codecov.io/gh/jaycrick/inf-bp-texts)

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
cd inf-bp-texts
uv sync
```

For development (includes testing and documentation tools):

```sh
uv sync --all-groups
```

## Quick Start

Download and analyze manuscripts (the `inf-bp-texts` console script is installed by `uv sync`):

```sh
uv run inf-bp-texts ntvmr download-all
uv run inf-bp-texts ntvmr analyze --manuscripts-dir manuscripts --output manuscript_analysis.csv
```

Run inference from cleaned data:

```sh
uv run inf-bp-texts inference --input-csv manuscript_analysis_cleaned.csv --plot
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
