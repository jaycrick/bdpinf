#!/bin/bash
# SLURM submit script: render the embarrassingly-parallel, no-GPU notebooks
# --- bulk simulate/fit and sensitivity-sweep notebooks that use
# joblib.Parallel(n_jobs=-1) across CPU cores.
#
# Usage (from the repo root, on an HPC login node):
#   sbatch scripts/render_cpu_parallel.slurm
#
# Adjust --account/--partition and --cpus-per-task for your allocation; these
# notebooks scale with available cores (joblib's n_jobs=-1 uses os.cpu_count(),
# which reflects the SLURM cpuset allocation).
#
# Project-rendered (in _quarto.yml's `render:` list): output lands in
# `_output/` automatically, no --execute-dir/mv needed.
#
#SBATCH --job-name=render-cpu-parallel
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1G
#SBATCH --time=10:00:00
#SBATCH --output=_output/slurm-cpu-parallel-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load quarto/1.8.26-x64

uv sync

NOTEBOOKS=(
  qmd_scripts/core_analysis/03_multi_epoch/optimizer_comparison.qmd
  qmd_scripts/validation/single_epoch/
  qmd_scripts/validation/multi_epoch/
)

for notebook in "${NOTEBOOKS[@]}"; do
  echo "=== Rendering ${notebook} ==="
  uv run quarto render "${notebook}"
done
