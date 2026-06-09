# Contributing

## Setup

```bash
# Install all dependencies including dev tools
uv sync --all-groups

# Install pre-commit hooks
uv run pre-commit install
```

## Running checks

```bash
# Tests (slow tests excluded by default)
uv run pytest

# All linting and formatting at once
uv run pre-commit run --all-files

# Individual tools
uv run ruff check src tests
uv run ruff format src tests
uv run mypy src/
```

## Quarto notebooks

```bash
# Always use uv run — plain quarto uses system Python (no venv)
uv run quarto render          # full site build
uv run quarto preview         # live preview

# Lint notebook Python cells
uv run lintquarto -l ruff mypy -p qmd_scripts docs -e .venv _output reference
```

## Pull requests

- One bundled commit per logical change (code + tests + docs together).
- Commit messages must start with a shortcode prefix — see `CLAUDE.md` for
  the full list (`:zap:`, `:bug:`, `:tv:`, `:cop:`, `:abc:`).
- All pre-commit hooks must pass before opening a PR.
- New public functions require NumPy-style docstrings.
