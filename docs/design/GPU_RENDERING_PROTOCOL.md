# GPU Rendering Protocol --- HPC GPU Benchmarking Notes (not used for routine rendering)

**GPU rendering of the Bayesian notebooks was tried, benchmarked, and
dropped.** On a real A100 (not an earlier A40-12Q vGPU), CPU beat GPU on
every configuration tested --- JIT MLE kernels, Bayesian NUTS fits at varying
sample/chain counts, and a much larger multi-subset model --- by roughly
1.6×--8×.
See `docs/design/JAX_MIGRATION_BENCHMARKS.md` for the numbers and
the likely explanation (NUTS's control-flow-heavy tree-building suits CPU
better than GPU's SIMD-throughput model, independent of problem size).

`qmd_scripts/validation/bayesian_multi_epoch/` now renders via
`scripts/render_cpu_parallel.slurm`, alongside the other validation
notebooks.
`scripts/render_gpu.slurm` has been removed.

This doc is kept for two things that remain true even though GPU rendering
isn't routine: (1) the technical gotchas below, relevant if anyone sets
`JAX_PLATFORMS=cuda` in this repo again, and (2) a manual recipe for
re-running the comparison if a future configuration looks more promising for
GPU (e.g. a much wider `vmap` batch --- dozens of chains rather than 4).

## Manual GPU re-verification recipe

```bash
module load quarto/1.8.26-x64
uv sync --group gpu
F=qmd_scripts/validation/bayesian_multi_epoch/<notebook>.qmd
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run quarto render "$F" --cache-refresh
```

`--cache-refresh` matters: this project's global `execute: cache: true`
caches cell output by source code, not by environment variable, so without it
a render under a different `JAX_PLATFORMS` value can silently replay a
previous backend's cached results instead of re-executing (see "A notebook's
own `import jax`..." below for a related gotcha that defeats the CPU default
the same way).

**Memory:** request at least `--mem-per-cpu=8G` in your `salloc`/interactive
session.
The CUDA compiler's host-side memory use (LLVM/NVPTX codegen, cuDNN
autotuning buffers) across several distinct input shapes adds up well past
the default 4000M, even though the model computation itself is tiny, and was
observed to **OOM-kill** JAX/XLA mid-compile.

No `module load` for CUDA is needed --- `jax[cuda12]` (installed via
`uv sync --group gpu`) bundles its own CUDA runtime and cuDNN.
A `module load` for CUDA is only relevant when compiling CUDA software from source,
which none of this requires.
`module load quarto/...` *is* required: the
Quarto CLI is not on `PATH` by default and is unrelated to the `quarto` Python
package in `pyproject.toml` (that's a thin Python binding, not the renderer).

### Viewing rendered output from a Mac (or any machine without HPC filesystem access)

Rendered output lands in `_output/` (per `_quarto.yml`'s `output-dir`).
Because `format.html.embed-resources` is `false`, each page's CSS/JS/figures
live in sibling `<name>_files/` directories plus a shared `site_libs/` ---
copy the **whole `_output/` tree**, not individual `.html` files, or the pages
will load with broken styling and missing figures.

```bash
# From your Mac:
rsync -avz spartan:/path/to/bdpinf/_output/ ~/bdpinf/_output/
```

Then open the rendered page directly:

```bash
open ~/bdpinf/_output/qmd_scripts/validation/bayesian_multi_epoch/01_simulation_checks.html
```

Or tunnel Quarto's live preview server instead of copying HTML:

```bash
# On Spartan (interactive session):
JAX_PLATFORMS=cuda uv run quarto preview qmd_scripts/validation/bayesian_multi_epoch/<notebook>.qmd --port 4848 --no-browser

# On your Mac, in another terminal:
ssh -L 4848:localhost:4848 spartan-gpgpu105   # or via the login node + srun, per your access path
```

Then open `http://localhost:4848` locally.
This re-executes on save, which is
convenient for debugging but re-pays JAX's cold-compile cost on each save
unless `JAX_COMPILATION_CACHE_DIR` is set to a persistent directory.

## Known issues encountered while evaluating GPU rendering

- **cuDNN version**: `nvidia-cudnn-cu12>=9.10` ships without `libcudnn.so.9`,
  causing JAX to silently fall back to CPU.
  The `gpu` dependency group in
  `pyproject.toml` pins `nvidia-cudnn-cu12<9.10`.
  See
  `docs/design/JAX_MIGRATION_BENCHMARKS.md`.
- **`JAX_PLATFORMS` must be exported before any Python process starts**, not
  set programmatically after `import jax` --- JAX reads it into its config at
  import time.
  `src/bdpinf/__init__.py` defaults it to `cpu` via
  `os.environ.setdefault`; `scripts/bench_jax_jit.py` does the same so its
  CPU baseline is reliable without depending on import order.
  Any `.qmd`
  that does `import jax` before `from bdpinf import ...` locks the backend
  at that line, bypassing the default --- import `bdpinf` first (or call
  `os.environ.setdefault("JAX_PLATFORMS", "cpu")` directly before
  `import jax`, matching `scripts/bench_jax_jit.py`) in any notebook that
  imports `jax` directly.
- **`BayesianModelConfig.device` is a no-op** --- it exists for API
  compatibility but the NumPyro backend is selected entirely by
  `JAX_PLATFORMS`.
  Don't pass `device="cuda"`; it raises a `UserWarning` and
  changes nothing.
- **Real-GPU benchmarking is done** (A100 80GB, 2026-06-22, see
  `docs/design/JAX_MIGRATION_BENCHMARKS.md`): the earlier A40-12Q vGPU was
  never a fundamental blocker, just a host limitation.
  On a real GPU, **CPU
  beats GPU on every configuration tested** --- the JIT MLE grid (≤6 epochs,
  ≤1000 ages) and Bayesian fits at 1 and 4 chains all land within a narrow
  1.6×--8× CPU-wins band, with the Bayesian configs converging on \~1.6×
  regardless of sample count, chain count, or per-call model size.
  Scaling
  each of those axes independently did **not** close the gap, which rules
  out "per-call dispatch overhead amortizes with more work" as the
  explanation.
  The likely real cause is that NUTS's tree-building is
  control-flow-heavy (recursive doubling, dynamic tree depth) in a way GPUs
  --- SIMD-throughput devices --- handle worse than CPUs, independent of
  problem size.
  The one untested, consistent-with-this-theory lever is a much
  wider `vmap` batch (dozens of chains rather than 4).
