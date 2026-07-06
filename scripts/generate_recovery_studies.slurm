#!/bin/bash
# SLURM submit script: precompute the recovery-study (simulate-then-fit)
# datasets consumed by the heaviest qmd_scripts/validation notebooks, so
# rendering itself only has to load and plot.
#
# This mirrors scripts/generate_bootstrap_kde_samples.slurm's role for
# bootstrap_samples/: an offline artifact generator chained ahead of the
# render jobs by render_all.sh, rather than something the render jobs compute
# inline. Each study parallelizes internally via joblib.Parallel(n_jobs=-1),
# same as render_cpu_parallel.slurm's notebooks, so this requests a similar
# many-core allocation.
#
# Usage (from the repo root, on an HPC login node):
#   sbatch scripts/generate_recovery_studies.slurm
#
# Adjust --account/--partition for your allocation if different.
#
# Resource sizing (see job 26813050): 32 cpus / 64G / 30h completed in
# 6:11:42 at 53.82% CPU efficiency (~17 cores average) and 56.16% memory
# efficiency (35.94 GB peak). Right-sized to 24 cpus / 48G / 10h, leaving
# ~40% headroom over both observed peaks rather than requesting 32 cores
# the job never sustained.
#SBATCH --job-name=generate-recovery-studies
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00:00
#SBATCH --output=_output/slurm-generate-recovery-studies-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

uv sync

uv run python scripts/generate_recovery_studies.py
