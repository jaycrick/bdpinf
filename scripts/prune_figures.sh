#!/usr/bin/env bash
# Report (or, with --apply, delete) stale figures under figures/.
#
# Run from anywhere; it works from the repo root. Dry run by default; extra
# arguments (--apply, --include-unmanaged) are passed to the prune CLI. See
# CONTRIBUTING.md ("Figure manifests").
set -euo pipefail
cd "$(dirname "$0")/.."

# -W: the package imports figure_manifest before runpy executes it as __main__,
# which would otherwise print a harmless RuntimeWarning.
uv run python -W ignore::RuntimeWarning -m bdpinf.visualization.figure_manifest \
  prune figures --project-root . "$@"
