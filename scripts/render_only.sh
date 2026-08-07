#!/usr/bin/env bash
# Submit only the notebook-rendering SLURM job, skipping the recovery-study
# generation step in scripts/render_all.sh. Use this when
# recovery_samples/*.npz already exist on disk from a prior run and only the
# notebooks themselves need to be (re-)rendered.
#
# Usage (from the repo root, on an HPC login node):
#   ./scripts/render_only.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

sbatch scripts/render_cpu_parallel.slurm
