#!/bin/bash
# SLURM submit script: render the Bayesian (NumPyro/NUTS) notebooks on a GPU
# node --- both the single-fit and the looped bulk/diagnostics notebooks,
# since every Bayesian notebook's parallelism (num_chains) runs within one
# GPU process rather than across cores/nodes; there is no separate
# "embarrassingly parallel GPU" bucket here.
#
# Usage (from the repo root, on Spartan):
#   sbatch scripts/render_gpu.slurm
#
# Adjust --account/--partition for your allocation if different.
#
#SBATCH --job-name=render-gpu
#SBATCH --account=punim1793
#SBATCH --partition=gpu-h100,gpu-a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=_output/slurm-gpu-render-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# The prebuilt jax[cuda12] wheels (installed below via the `gpu` dependency
# group) bundle their own CUDA runtime + cuDNN — no `module load` for CUDA is
# needed. A `module load` is only required to build CUDA software from source,
# which uv/jax's prebuilt wheels never do.
module load quarto/1.8.26-x64

# JAX reads JAX_PLATFORMS into its config at `import jax` time, so it must be
# exported before any python process starts (see scripts/bench_jax_jit.py and
# src/inf_bp_texts/__init__.py for the same constraint).
export JAX_PLATFORMS=cuda
# Avoid XLA's default ~90% upfront GPU memory preallocation; not required for
# correctness but keeps memory headroom predictable on a shared GPU node.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv sync --group gpu

uv run python -c "import inf_bp_texts, jax; print('JAX devices:', jax.devices())"

# These notebooks are part of _quarto.yml's `render:` list like everything
# else, so a per-file `quarto render` is project-aware: output lands directly
# in `_output/`, matching scripts/render_cpu_light.slurm and
# scripts/render_cpu_parallel.slurm. They're rendered on their own GPU node
# here (rather than relying on a plain `quarto render` of the whole site)
# because they're slow multi-chain NUTS fits that benefit from CUDA.
BAYESIAN_NOTEBOOKS=(
  qmd_scripts/core_analysis/04_bayesian/kashouh_family_k_bayesian_inference.qmd
  qmd_scripts/core_analysis/04_bayesian/kashouh_single_epoch_bayesian.qmd
  qmd_scripts/core_analysis/04_bayesian/ntvmr_single_epoch_bayesian.qmd
  qmd_scripts/core_analysis/04_bayesian/ntvmr_birth_death_bayesian.qmd
  qmd_scripts/validation/bayesian_multi_epoch/01_simulation_checks.qmd
  qmd_scripts/validation/bayesian_multi_epoch/02_single_fit_debug.qmd
  qmd_scripts/validation/bayesian_multi_epoch/03_bulk_inference.qmd
  qmd_scripts/validation/bayesian_multi_epoch/04_diagnostics_benchmark.qmd
)

for notebook in "${BAYESIAN_NOTEBOOKS[@]}"; do
  echo "=== Rendering ${notebook} ==="
  uv run quarto render "${notebook}"
done
