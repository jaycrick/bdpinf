# inf-bp-texts

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

- NTVMR data collection with resume support
- XML formatting and validation for git-friendly tracking
- Data quality assessment and manual corrections
- Biblical book detection and section alignment checks
- Birth-death process modeling, fitting, and visualization
- Quarto site output for interactive reports

## Development

Run tests:

```sh
uv run pytest
```

Generate API docs:

```sh
uv run quartodoc build
```

Preview documentation site:

```sh
uv run quarto preview
```
