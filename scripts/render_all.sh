#!/usr/bin/env bash
# Submit the recovery-study generation job and the notebook-rendering SLURM
# job as a sequential chain. generate_recovery_studies.slurm must run before
# render_cpu_parallel.slurm, since the validation notebooks it renders load
# recovery_samples/*.npz produced by that generation step.
# `--dependency=singleton` makes a job wait for any prior *running or
# pending* job with the same --job-name to finish, so giving both the same
# name chains them in submission order.
#
# A GPU render bucket (render_gpu.slurm) used to render the Bayesian
# notebooks on a GPU node; it was removed after benchmarking showed CPU is
# 1.6x-8x faster than GPU on every config tested (see
# docs/design/GPU_RENDERING_PROTOCOL.md). Those notebooks (now
# qmd_scripts/validation/bayesian_multi_epoch/) render via
# render_cpu_parallel.slurm alongside everything else.
#
# Usage (from the repo root, on an HPC login node):
#   ./scripts/render_all.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

job_name="bdpinf-render"

for job in scripts/generate_recovery_studies.slurm scripts/render_cpu_parallel.slurm; do
  job_id=$(sbatch --parsable --dependency=singleton --job-name="${job_name}" "${job}")
  echo "Submitted ${job} as job ${job_id} (chained on job-name ${job_name})"
done
