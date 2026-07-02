#!/usr/bin/env bash
# Submit the recovery-study generation job and both notebook-rendering SLURM
# jobs as a sequential chain. generate_recovery_studies.slurm must run before
# render_cpu_parallel.slurm, since the validation notebooks it renders load
# recovery_samples/*.npz produced by that generation step; render_cpu_light.slurm
# has no such dependency but is kept in the same chain purely to stay within a
# shared allocation/budget (as before). `--dependency=singleton` makes a job
# wait for any prior *running or pending* job with the same --job-name to
# finish, so giving all three the same name chains them in submission order.
#
# A third render bucket (render_gpu.slurm) used to render the Bayesian
# notebooks on a GPU node; it was removed after benchmarking showed CPU is
# 1.6x-8x faster than GPU on every config tested (see
# docs/design/GPU_RENDERING_PROTOCOL.md). Those notebooks now render via
# render_cpu_light.slurm.
#
# Usage (from the repo root, on an HPC login node):
#   ./scripts/render_all.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

job_name="inf-bp-texts-render"

for job in scripts/generate_recovery_studies.slurm scripts/render_cpu_light.slurm scripts/render_cpu_parallel.slurm; do
  job_id=$(sbatch --parsable --dependency=singleton --job-name="${job_name}" "${job}")
  echo "Submitted ${job} as job ${job_id} (chained on job-name ${job_name})"
done
