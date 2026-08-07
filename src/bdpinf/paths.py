"""Project-root resolution for use in Quarto notebooks."""

import os
from pathlib import Path

_DEFAULT_MARKERS: tuple[str, ...] = ("_quarto.yml", "pyproject.toml", ".git")


def find_project_root(
    *,
    markers: tuple[str, ...] = _DEFAULT_MARKERS,
    env_var: str = "QUARTO_PROJECT_DIR",
) -> Path:
    """Return the project root, falling back to a cwd-based marker search.

    During ``uv run quarto render / preview``, Quarto sets the environment
    variable named by ``env_var`` (``QUARTO_PROJECT_DIR`` by default) to the
    directory containing ``_quarto.yml``. When that variable is absent (e.g.
    during interactive development), the function walks up from the current
    working directory until it finds a directory containing any of
    ``markers`` and returns that directory.

    Parameters
    ----------
    markers : tuple[str, ...]
        Filenames whose presence identifies the project root, checked in
        order at each directory level.
    env_var : str
        Environment variable consulted first for an explicit root.

    Returns
    -------
    Path
        Absolute path to the project root.

    """
    p = Path.cwd()
    env_root = os.environ.get(env_var)
    if env_root:
        candidate = Path(env_root).resolve()
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    for parent in [p, *p.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent
    return p
