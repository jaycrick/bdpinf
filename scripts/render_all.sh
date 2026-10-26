#!/usr/bin/env bash
# Submit the recovery-study generation job and the notebook-rendering SLURM
# job as a sequential chain. generate_recovery_studies.slurm must run before
# render_cpu_parallel.slurm, since the validation notebooks it renders load
# recovery_samples/*.npz produced by that generation step.
# `--dependency=singleton` makes a job wait for any prior *running or
# pending* job with the same --job-name to finish, so giving both the same
# name chains them in submission order.
#
# The Bayesian notebooks (qmd_scripts/validation/bayesian_multi_epoch/) render
# on CPU alongside everything else via render_cpu_parallel.slurm: CPU is
# 1.6x-8x faster than GPU on every config tested (see
# docs/design/GPU_RENDERING_PROTOCOL.md).
#
# Usage (from the repo root, on an HPC login node):
#   ./scripts/render_all.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

job_name="bdpinf-render"

for job in scripts/generate_recovery_studies.slurm scripts/render_cpu_parallel.slurm; do
  job_id=$(sbatch --parsable --dependency=singleton --job-name="${job_name}" "${job}")
  echo "Submitted ${job} as job ${job_id} (chained on job-name ${job_name})"
done
