#!/bin/bash
# SLURM submit script: render every qmd_scripts/validation/ notebook,
# no GPU --- bulk simulate/fit and sensitivity-sweep notebooks that use
# joblib.Parallel(n_jobs=-1) across CPU cores, plus the Bayesian NUTS
# notebooks (single-threaded per chain; GPU was benchmarked and found
# 1.6x-8x slower than CPU on every config tested, see
# docs/design/GPU_RENDERING_PROTOCOL.md).
#
# Several of the heaviest notebooks under qmd_scripts/validation/ (the
# single-/multi-epoch bulk-inference and asymptotic-consistency studies) no
# longer simulate+fit inline --- they load recovery_samples/*.npz produced by
# scripts/generate_recovery_studies.slurm, which must run first (see
# render_all.sh). The remaining notebooks in these directories (e.g.
# 01_simulation_checks.qmd, 06_sensitivity_benchmark.qmd,
# 09_subcritical_splitting.qmd) still simulate/fit inline, so this job keeps a
# generous time budget; retune --time after measuring the refactored renders.
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
# Resource sizing (based on job 26813064): a 16 cpus / 32G / 6h request
# completed in 1:49:01 at 37.30% CPU efficiency (~6 cores average) and 61.90%
# memory efficiency (19.81 GB peak). This job asks for 8 cpus / 24G / 3h,
# leaving headroom above both observed averages/peaks (3G per cpu so the
# smaller cpu count doesn't undershoot the 19.81 GB peak). Job 26813064 did
# not render qmd_scripts/validation/bayesian_multi_epoch/, so retune after
# measuring a run of the current notebook set.
#SBATCH --job-name=render-cpu-parallel
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=03:00:00
#SBATCH --output=_output/slurm-cpu-parallel-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load quarto/1.8.26-x64

uv sync

NOTEBOOKS=(
  qmd_scripts/validation/single_epoch/
  qmd_scripts/validation/multi_epoch/
  qmd_scripts/validation/bayesian_multi_epoch/
)

for notebook in "${NOTEBOOKS[@]}"; do
  echo "=== Rendering ${notebook} ==="
  uv run quarto render "${notebook}" --cache-refresh
done
