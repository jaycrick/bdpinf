"""Figure manifests: record which script or notebook wrote which figure.

Each :func:`~bdpinf.visualization.save_chart` call records the file it wrote in
``<output_dir>/.figure_manifest/<source-slug>.txt``.
The first line is ``# source: <path>`` and every further line is one figure
filename.
The first write per (directory, source) in a process truncates the manifest and
later writes append, so a manifest always describes its source's *last*
execution; sources that are not re-executed (cached or frozen notebooks) keep
theirs.

:func:`prune_figures` (also ``python -m bdpinf.visualization.figure_manifest
prune``) uses the manifests to find figures no source writes any more.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from bdpinf.paths import find_project_root

MANIFEST_DIR = ".figure_manifest"
FIGURE_SUFFIXES = (".pdf", ".svg", ".png")
INTERACTIVE = "interactive"
_SOURCE_PREFIX = "# source:"

# (directory, source) pairs already written by this process.
_started: set[tuple[Path, str]] = set()


def _relative_to_project(path: Path) -> str:
    """Return ``path`` relative to the project root, else absolute (POSIX form)."""
    path = path.resolve()
    try:
        return path.relative_to(find_project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def current_source() -> str:
    """Identify what is producing figures: a Quarto document, a script, or neither.

    Inside ``quarto render`` the kernel sees ``QUARTO_DOCUMENT_FILE`` (the
    document's basename), ``QUARTO_DOCUMENT_PATH`` (its directory) and
    ``QUARTO_PROJECT_ROOT`` / ``QUARTO_PROJECT_DIR`` (the project root);
    these give a project-relative document path.
    Otherwise the running script (``sys.argv[0]`` if it is an existing ``.py``
    file) is used, and failing that ``"interactive"``.

    Returns
    -------
    str
        Project-relative POSIX path (absolute if outside the project), or
        ``"interactive"``.
    """
    doc_file = os.environ.get("QUARTO_DOCUMENT_FILE")
    if doc_file:
        doc_dir = os.environ.get("QUARTO_DOCUMENT_PATH")
        document = Path(doc_dir) / doc_file if doc_dir else Path(doc_file)
        if not document.is_absolute():
            document = Path.cwd() / document
        document = document.resolve()
        root = os.environ.get("QUARTO_PROJECT_ROOT") or os.environ.get(
            "QUARTO_PROJECT_DIR"
        )
        if root:
            try:
                return document.relative_to(Path(root).resolve()).as_posix()
            except ValueError:
                pass
        return document.as_posix()

    script = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if script is not None and script.suffix == ".py" and script.is_file():
        return _relative_to_project(script)
    return INTERACTIVE


def source_slug(source: str) -> str:
    """Turn a source path into a safe single filename (separators become ``__``)."""
    slug = re.sub(r"[\\/]+", "__", source.strip("\\/"))
    slug = re.sub(r"[^\w.\-]+", "_", slug)
    return slug or INTERACTIVE


def record_figure(path: Path) -> None:
    """Record ``path`` in its directory's manifest for the current source.

    Parameters
    ----------
    path : Path
        A figure file that has just been written.
    """
    directory = path.parent.resolve()
    source = current_source()
    manifest = directory / MANIFEST_DIR / f"{source_slug(source)}.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    key = (directory, source)
    if key not in _started or not manifest.exists():
        _started.add(key)
        manifest.write_text(f"{_SOURCE_PREFIX} {source}\n{path.name}\n")
        return

    if path.name not in manifest.read_text().splitlines():
        with manifest.open("a") as handle:
            handle.write(f"{path.name}\n")


@dataclass
class PruneReport:
    """Outcome of :func:`prune_figures`.

    Attributes
    ----------
    stale : list[Path]
        Figures in managed directories that no live manifest lists.
    unmanaged : list[Path]
        Figures in directories with no manifest directory.
    orphan_manifests : list[Path]
        Manifests whose source file no longer exists.
    deleted : list[Path]
        What was actually removed (empty on a dry run).
    """

    stale: list[Path] = field(default_factory=list)
    unmanaged: list[Path] = field(default_factory=list)
    orphan_manifests: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)


def _read_manifest(manifest: Path) -> tuple[str | None, set[str]]:
    """Return the manifest's source (None if unreadable) and its filenames."""
    source: str | None = None
    names: set[str] = set()
    for line in manifest.read_text().splitlines():
        if line.startswith(_SOURCE_PREFIX):
            source = line[len(_SOURCE_PREFIX) :].strip()
        elif line.strip():
            names.add(line.strip())
    return source, names


def prune_figures(
    root: Path,
    *,
    project_root: Path,
    apply: bool = False,
    include_unmanaged: bool = False,
) -> PruneReport:
    """Find (and with ``apply``, delete) figures no manifest lists.

    Parameters
    ----------
    root : Path
        Figure tree to scan recursively.
    project_root : Path
        Directory against which manifest sources are checked for existence.
    apply : bool
        Delete what is found; the default is a dry run.
    include_unmanaged : bool
        Also delete figures in directories that have no manifest directory.

    Returns
    -------
    PruneReport
        Stale figures, unmanaged figures, orphan manifests and what was deleted.
    """
    report = PruneReport()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != MANIFEST_DIR)
        directory = Path(dirpath)
        figures = sorted(
            directory / f for f in filenames if f.endswith(FIGURE_SUFFIXES)
        )
        manifest_dir = directory / MANIFEST_DIR
        if not manifest_dir.is_dir():
            report.unmanaged.extend(figures)
            continue

        listed: set[str] = set()
        for manifest in sorted(manifest_dir.glob("*.txt")):
            source, names = _read_manifest(manifest)
            if (
                source is not None
                and source != INTERACTIVE
                and not (project_root / source).exists()
            ):
                report.orphan_manifests.append(manifest)
                continue
            listed |= names
        report.stale.extend(f for f in figures if f.name not in listed)

    if apply:
        doomed = [*report.stale, *report.orphan_manifests]
        if include_unmanaged:
            doomed += report.unmanaged
        for path in doomed:
            path.unlink(missing_ok=True)
            report.deleted.append(path)
    return report


def _format_report(report: PruneReport, *, apply: bool, include_unmanaged: bool) -> str:
    lines = []
    sections = (
        ("Stale figures (not listed in any live manifest)", report.stale),
        (
            "Unmanaged figures (directory has no manifest)"
            + ("" if include_unmanaged else "; kept without --include-unmanaged"),
            report.unmanaged,
        ),
        ("Orphan manifests (source file no longer exists)", report.orphan_manifests),
    )
    for title, paths in sections:
        lines.append(f"{title}: {len(paths)}")
        lines.extend(f"  {p}" for p in paths)
    if apply:
        lines.append(f"Deleted: {len(report.deleted)}")
    else:
        lines.append("Dry run: nothing deleted (pass --apply to delete).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point (``prune`` subcommand)."""
    parser = argparse.ArgumentParser(
        prog="python -m bdpinf.visualization.figure_manifest"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prune = sub.add_parser("prune", help="remove figures no manifest lists")
    prune.add_argument("root", type=Path, help="figure tree to scan")
    prune.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="directory manifest sources are relative to (default: detected)",
    )
    prune.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    prune.add_argument(
        "--include-unmanaged",
        action="store_true",
        help="also delete figures in directories without a manifest",
    )
    args = parser.parse_args(argv)

    report = prune_figures(
        args.root,
        project_root=args.project_root or find_project_root(),
        apply=args.apply,
        include_unmanaged=args.include_unmanaged,
    )
    print(
        _format_report(
            report, apply=args.apply, include_unmanaged=args.include_unmanaged
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
