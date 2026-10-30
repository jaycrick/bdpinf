#!/bin/bash
# SLURM job: measure the compile time and peak memory of the Hessian variants for
# the NTVMR linear model (scripts/diagnose_hessian_compile.py).
#
# The first diagnostic job (scripts/diagnose_ntvmr_knots.slurm) was OOM-killed:
# two processes each grew past 24 GB within 40 s while compiling the exact
# forward-over-forward Hessian.
# Here each variant runs alone, in its own process, with the whole 64 GB
# available, so the peak is measured cleanly; a variant that still exceeds it is
# killed on its own and the others carry on.
# The cheapest variants run first. Variants are:
#   fd-grad          central differences of the gradient (no second-order compile)
#   fwd-rev          forward-over-reverse
#   fd-grad-noopt    fd-grad with XLA optimisations mostly off
#   fwd-rev-noopt    fwd-rev with XLA optimisations mostly off
#   fwd-fwd          the current exact Hessian
#
# Usage (from the repo root, on an HPC login node):
#   sbatch scripts/diagnose_hessian_compile.slurm
#
# Logs: logs/ (SLURM) and _output/diagnose/.
#SBATCH --job-name=diagnose-hessian
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-diagnose-hessian-%j-%N.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs _output/diagnose
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# No shared compile cache here: every variant must really compile.
export JAX_COMPILATION_CACHE_DIR=""
uv sync
RUN="uv run python scripts/diagnose_hessian_compile.py"

for entry in "fd-grad:fd-grad:0" "fwd-rev:fwd-rev:0" \
             "fd-grad:fd-grad-noopt:1" "fwd-rev:fwd-rev-noopt:1" "fwd-fwd:fwd-fwd:0"; do
    IFS=: read -r mode tag noopt <<< "$entry"
    echo "== $tag ($(date))"
    JAX_DISABLE_MOST_OPTIMIZATIONS=$([[ $noopt == 1 ]] && echo true || echo false) \
        timeout 50m $RUN --mode "$mode" --tag "$tag" 2>&1 | tail -n 6
done
$RUN --compare
