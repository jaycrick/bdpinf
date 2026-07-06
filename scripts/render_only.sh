#!/usr/bin/env bash
# Submit only the two notebook-rendering SLURM jobs, skipping the
# recovery-study and bootstrap-KDE generation steps in scripts/render_all.sh.
# Use this when recovery_samples/*.npz and bootstrap_samples/*.npz already
# exist on disk from a prior run and only the notebooks themselves need to be
# (re-)rendered.
#
# `--dependency=singleton` makes a job wait for any prior *running or
# pending* job with the same --job-name to finish, so giving both jobs the
# same name chains them in submission order.
#
# Usage (from the repo root, on an HPC login node):
#   ./scripts/render_only.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

job_name="inf-bp-texts-render"

for job in scripts/render_cpu_light.slurm scripts/render_cpu_parallel.slurm; do
  job_id=$(sbatch --parsable --dependency=singleton --job-name="${job_name}" "${job}")
  echo "Submitted ${job} as job ${job_id} (chained on job-name ${job_name})"
done
