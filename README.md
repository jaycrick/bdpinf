# inf-bp-texts

A Python platform for downloading New Testament manuscript data from the NTVMR API
and running quality control, analysis, and birth-death process inference.

## Quick Start

Render the main analysis document:

```sh
quarto render manuscript_analysis.qmd
```

Download and analyze manuscripts:

```sh
python -m inf_bp_texts.cli ntvmr download-all
python -m inf_bp_texts.cli ntvmr analyze --manuscripts-dir manuscripts --output manuscript_analysis.csv
```

Run inference from cleaned data:

```sh
python -m inf_bp_texts.cli inference --input-csv manuscript_analysis_cleaned.csv --plot
```

## Key Features

- NTVMR data collection with resume support
- XML formatting and validation for git-friendly tracking
- Data quality assessment and manual corrections
- Biblical book detection and section alignment checks
- Birth-death process modeling, fitting, and visualization
- Quarto site output for interactive reports

## Documentation

- [docs/overview.md](docs/overview.md) - Project goals, workflows, and outputs
- [docs/cli.md](docs/cli.md) - CLI commands, flags, and examples
- [docs/data_quality.md](docs/data_quality.md) - Quality checks and manual corrections
- [docs/analysis.md](docs/analysis.md) - Data columns and analysis notes
- [docs/birth_death.md](docs/birth_death.md) - Modeling and inference usage
- [docs/figures.md](docs/figures.md) - Figure generation and export
- [docs/development.md](docs/development.md) - Setup, tooling, and tests
- [docs/faq.md](docs/faq.md) - Common questions and troubleshooting

## Installation

With Poetry:

```sh
poetry install
```

Development install:

```sh
poetry install --with dev
```

Run tests:

```sh
poetry run pytest
```

Generate docs:

```sh
quartodoc build
```

Run analysis files:

```sh
quarto preview
```
