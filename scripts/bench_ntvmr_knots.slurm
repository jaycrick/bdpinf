#!/bin/bash
# SLURM array job: benchmark the fit_knots search methods with five free knots on
# NTVMR-shaped date-range data (scripts/bench_ntvmr_knots.py).
#
# Replicates are independent parametric bootstrap samples from the truth in
# scripts/data/ntvmr_benchmark.json, so each array task fits one replicate with
# all five methods in parallel (one single-thread process per method, 5 cores) and writes
# _output/bench_ntvmr_knots_shard<i>.json.
# One extra job fits the real data with every method (--real).
# A final job, dependent on both, merges the shards and prints the tables,
# including the best sets of one, two and three methods.
#
# Usage (from the repo root, on an HPC login node; no nt-manuscripts needed):
#   scripts/bench_ntvmr_knots.slurm submit
#
# Resource sizing: PLACEHOLDER until scripts/diagnose_ntvmr_knots.slurm sets --time
# (the sim1000 pilot showed run times 5-15x above the first guesses).
# The truth must exist first: run the diagnose job, or
#   uv run python scripts/bench_ntvmr_knots.py --truth
# and keep _output/ntvmr_truth.json (or put the truth in the data file).
# Memory is 4G per core (20 GB): the sim1000 run lost one replicate to an 8 GB OOM.
# Check the cluster's limit on array size and running jobs before submitting.
#
# Logs land in repo-root logs/ (gitignored); submit from the repo root.
#SBATCH --job-name=bench-ntvmr-knots
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/slurm-bench-ntvmr-knots-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

REPLICATES="${REPLICATES:-200}"
SHARDS=200

if [[ "${1:-}" == "submit" ]]; then
    mkdir -p logs
    array_id=$(sbatch --parsable "$0")
    real_id=$(sbatch --parsable --array=0-0 --job-name=bench-ntvmr-knots-real \
        --output=logs/slurm-bench-ntvmr-knots-real-%j-%N.log \
        --export=ALL,BENCH_MODE=real "$0")
    sbatch --dependency=afterok:"${array_id}":"${real_id}" --array=0-0 \
        --time=00:20:00 --job-name=bench-ntvmr-knots-merge \
        --output=logs/slurm-bench-ntvmr-knots-merge-%j-%N.log \
        --wrap="cd ${SLURM_SUBMIT_DIR:-$(pwd)} && uv run python scripts/bench_ntvmr_knots.py --merge"
    exit 0
fi

# One thread per process; the parallelism is across the methods (--workers 5).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

uv sync
if [[ "${BENCH_MODE:-}" == "real" ]]; then
    uv run python scripts/bench_ntvmr_knots.py --real
else
    uv run python scripts/bench_ntvmr_knots.py \
        --replicates "${REPLICATES}" --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}" \
        --workers 5
fi
