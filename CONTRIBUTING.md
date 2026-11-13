# Contributing

## Setup

```bash
uv sync --group dev            # runtime + dev tools (add --group gpu on a CUDA Linux box)
uv run pre-commit install      # run the checks below on every commit
```

## Checks

CI runs exactly `uv run pre-commit run --all-files` and `uv run pytest`.
Run the same before opening a PR.

```bash
uv run pre-commit run --all-files   # ruff check/format, basedpyright, prose lint/format

uv run pytest                       # fast suite; coverage gate in pyproject.toml
uv run pytest -m slow               # slow tests only
uv run pytest -m regression         # golden-value tests
uv run pytest -m recovery           # simulate-then-fit recovery (implies slow)
uv run pytest -k "name"             # one test by name
```

The pieces, if you want them separately:

```bash
uv run ruff format src tests
uv run ruff check --fix src tests
uv run basedpyright --project pyproject.toml src/    # strict; 0 errors enforced
uv run panache format .                              # Markdown/Quarto prose (panache.toml)
uv run panache lint --fix .
```

Always run the formatter and auto-fixer first and let Ruff correct what it can, rather than hand-editing formatting.
Ruff's `I` rules sort imports in `src` and `tests`; do not also run standalone `isort` over them.
`isort` splits any imported name that carries a trailing inline comment (for example `# pyright: ignore[...]`) onto its own import statement, which undoes Ruff's merge.
`isort` is used only by panache to format Python chunks inside `.qmd` files.

## Quarto notebooks

Always use `uv run quarto ...`; plain `quarto` uses the system Python and fails.

```bash
uv run quarto render           # research-notebook site -> _output/ (cached cells skipped)
uv run quarto preview          # live preview
uv run lintquarto -l ruff basedpyright -p qmd_scripts docs -e .venv _output   # after editing any .qmd
```

## Figure manifests

Every `save_chart` call records the file it wrote in `<output_dir>/.figure_manifest/<source-slug>.txt`.
The first line is `# source: <path>` (the Quarto document being rendered, else the running script, else `interactive`), followed by one filename per line.
The first write per (directory, source) in a process truncates the manifest and later writes append, so a manifest always describes its source's last execution; cached or frozen notebooks keep theirs.

After a full re-render, find figures that no source writes any more:

```bash
scripts/prune_figures.sh                     # dry run: lists stale and unmanaged figures, orphan manifests
scripts/prune_figures.sh --apply             # delete stale figures and orphan manifests
scripts/prune_figures.sh --apply --include-unmanaged   # also delete figures in directories without a manifest
```

A manifest whose source file no longer exists is an orphan: it is dropped and its figures become stale.
`scripts/render_cpu_parallel.slurm` ends with the dry-run report.
The same tool is available as `bdpinf.visualization.prune_figures`.

## API reference (separate site)

Generated from docstrings by great-docs; structure and ordering live in `great-docs.yml`.

```bash
uv run great-docs build        # -> great-docs/_site
uv run great-docs preview
```

Docstrings are NumPy style.
Write maths as `$...$` or `$$...$$` inside a raw docstring (`r"""..."""`); Sphinx `:math:` roles are not translated under `parser: numpy`.
Cite the accompanying paper as `[paper: Theorem ?]` until its numbering is fixed (search for `[paper:`).

## Pull requests

- One bundled commit per logical change (code, tests and docs together).
- Commit messages start with a shortcode prefix; see `CLAUDE.md` for the list and the changelog rule.
- All pre-commit hooks and the test suite must pass.
- New public functions need NumPy-style docstrings.
