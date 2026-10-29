#!/bin/bash
# SLURM array job: benchmark knot-and-rate fitting methods on the sim1000
# known-truth simulation (scripts/bench_knot_methods.py --only sim1000).
#
# Replicates are independent, so each array task fits one replicate on 4
# cores and writes _output/bench_knot_methods_sim1000_shard<i>.json.
# A second job, dependent on the whole array, merges the shards into
# _output/bench_knot_methods_sim1000.{json} and prints the summary tables.
#
# Usage (from the repo root, on an HPC login node; no nt-manuscripts needed):
#   scripts/bench_knot_methods.slurm submit
#
# Resource sizing from one local pilot replicate (n = 143, all nine methods,
# about 45 minutes using ~4 cores, including compile): one replicate per task,
# 200 tasks x 4 cores x 2G, with a 4 hour limit (about 5x headroom).
# Check the cluster's limit on array size and running jobs before submitting.
#
# Logs land in repo-root logs/ (gitignored); submit from the repo root.
#SBATCH --job-name=bench-knot-methods
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/slurm-bench-knot-methods-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

REPLICATES="${REPLICATES:-200}"
SHARDS=200

if [[ "${1:-}" == "submit" ]]; then
    mkdir -p logs
    array_id=$(sbatch --parsable "$0")
    sbatch --dependency=afterok:"${array_id}" --array=0-0 --time=00:20:00 \
        --job-name=bench-knot-methods-merge \
        --output=logs/slurm-bench-knot-methods-merge-%j-%N.log \
        --wrap="cd ${SLURM_SUBMIT_DIR:-$(pwd)} && uv run python scripts/bench_knot_methods.py --only sim1000 --merge"
    exit 0
fi

# Match the thread pools to the allocation (the pilot used about 4 cores).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

uv sync
uv run python scripts/bench_knot_methods.py --only sim1000 \
    --replicates "${REPLICATES}" --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}"
