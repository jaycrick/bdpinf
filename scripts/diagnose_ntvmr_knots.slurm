#!/bin/bash
# SLURM job: diagnose the cost of the NTVMR multi-knot benchmark on the HPC,
# using many cores at once (scripts/bench_ntvmr_knots.py).
#
# The first run of this job was OOM-killed: the range masses were integrated by
# quadrature over every piece of every range, so one gradient took 0.2 s and the
# Hessian needed tens of GB.
# They now use cumulative functions at the range endpoints (about 60x cheaper,
# 1 ms per likelihood) and the benchmark uses a finite-difference Hessian
# (scripts/diagnose_hessian_compile.slurm measured the alternatives), so the
# processes below need a few GB each.
# This job times what the full benchmark will cost, running independent
# single-thread processes in parallel (12 cores) that share one JAX compilation
# cache (_output/jax_cache):
#
#   A. fit the truth (all knots fixed) with 1 start and with 5 starts, at once,
#      so we see which is slow and how many iterations each needs;
#   B. after the 1-start truth is ready, fit one bootstrap replicate with each of
#      the 5 methods, and the real data with each of the 5 methods (knots free),
#      all 10 at once with fit_knots progress logged.
#
# Usage (from the repo root, on an HPC login node; no nt-manuscripts needed):
#   sbatch scripts/diagnose_ntvmr_knots.slurm
# Then read the tables in the job log, or run:
#   uv run python scripts/bench_ntvmr_knots.py --report
#
# Logs: logs/ (SLURM) and _output/diagnose/ (one log per process).
#SBATCH --job-name=diagnose-ntvmr-knots
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-diagnose-ntvmr-knots-%j-%N.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs _output/diagnose
OUT=_output/diagnose
METHODS="brent de-profile grid-newton nelder-mead profile"

# One thread per process: the parallelism is across the processes below.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

uv sync
BENCH="uv run python scripts/bench_ntvmr_knots.py"

echo "== A: truth fits ($(date))"
timeout 3h $BENCH --truth --starts 1 --tag s1 --verbose > "$OUT/truth_s1.log" 2>&1 &
pid_s1=$!
timeout 4h $BENCH --truth --starts 5 --tag s5 --verbose > "$OUT/truth_s5.log" 2>&1 &
pid_s5=$!
wait "$pid_s1"
if [[ -f _output/ntvmr_truth_s1.json ]]; then
    cp _output/ntvmr_truth_s1.json _output/ntvmr_truth.json
else
    echo "1-start truth fit did not finish; waiting for the 5-start fit"
    wait "$pid_s5"
    cp _output/ntvmr_truth_s5.json _output/ntvmr_truth.json || exit 1
fi

echo "== B: one replicate and the real data, per method ($(date))"
for m in $METHODS; do
    timeout 5h $BENCH --replicates 1 --methods "$m" --tag "$m" --verbose \
        > "$OUT/replicate_$m.log" 2>&1 &
    timeout 5h $BENCH --real --methods "$m" --tag "$m" --verbose \
        > "$OUT/real_$m.log" 2>&1 &
done
wait

echo "== Report ($(date))"
$BENCH --report
for f in "$OUT"/*.log; do echo "--- $f"; tail -n 4 "$f"; done
