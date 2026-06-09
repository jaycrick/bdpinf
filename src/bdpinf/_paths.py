"""Project-root resolution for use in Quarto notebooks."""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the Quarto project root, falling back to a cwd-based search.

    During ``uv run quarto render / preview``, Quarto sets the environment
    variable ``QUARTO_PROJECT_DIR`` to the directory containing
    ``_quarto.yml``.  When that variable is absent (e.g. during interactive
    development), the function walks up from the current working directory
    until it finds ``_quarto.yml`` and returns that parent.

    Returns
    -------
    Path
        Absolute path to the project root.

    """
    quarto_root = os.environ.get("QUARTO_PROJECT_DIR")
    if quarto_root:
        return Path(quarto_root).resolve()
    p = Path.cwd()
    for parent in [p, *p.parents]:
        if (parent / "_quarto.yml").exists():
            return parent
    return p
