# inf-bp-texts

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/jaycrick/inf-bp-texts/graph/badge.svg?token=5QNFKWSL7G)](https://codecov.io/gh/jaycrick/inf-bp-texts)

A Python platform for downloading New Testament manuscript data from the NTVMR API
and running quality control, analysis, and birth-death process inference.

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

For static image generation with Plotly, install the Chrome dependency:

```sh
uv run plotly_get_chrome
```

## Quick Start

Download and analyze manuscripts:

```sh
uv run python -m inf_bp_texts.cli ntvmr download-all
uv run python -m inf_bp_texts.cli ntvmr analyze --manuscripts-dir manuscripts --output manuscript_analysis.csv
```

Run inference from cleaned data:

```sh
uv run python -m inf_bp_texts.cli inference --input-csv manuscript_analysis_cleaned.csv --plot
```

Render the analysis document:

```sh
uv run quarto render manuscript_analysis.qmd
```

## Key Features

- NTVMR data collection with resume support plus git-friendly XML formatting and validation
- Quality control, book detection/section alignment,
- Birth-death modeling with Quarto reports

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
