#!/usr/bin/env bash
# Resume scripts/render_all.sh's job chain after generate_recovery_studies.slurm
# has already completed successfully (recovery_samples/*.npz already exist on
# disk), skipping straight to the remaining jobs.
#
# Also resumes generate_bootstrap_kde_samples.slurm at the subgroup it was
# killed on: a prior run (job 26813051) timed out after saving
# bootstrap_samples/{full,non_lectionary,lectionary}.npz but before starting
# section_a/section_p/section_r, so only the four missing subgroups are
# requested here rather than re-generating the three that already succeeded.
# If bootstrap_samples/*.npz for those four subgroups already exist by the
# time you run this, adjust the --subgroup list below (or drop it entirely
# to fall back to the script's all-seven default).
#
# Usage (from the repo root, on an HPC login node):
#   ./scripts/resume_render_all.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

job_name="bdpinf-render"

bootstrap_job_id=$(sbatch --parsable --dependency=singleton --job-name="${job_name}" \
  scripts/generate_bootstrap_kde_samples.slurm \
  --subgroup section_e --subgroup section_a --subgroup section_p --subgroup section_r)
echo "Submitted scripts/generate_bootstrap_kde_samples.slurm (remaining subgroups) as job ${bootstrap_job_id} (chained on job-name ${job_name})"

for job in scripts/render_cpu_light.slurm scripts/render_cpu_parallel.slurm; do
  job_id=$(sbatch --parsable --dependency=singleton --job-name="${job_name}" "${job}")
  echo "Submitted ${job} as job ${job_id} (chained on job-name ${job_name})"
done
