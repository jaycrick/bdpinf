# JAX Migration Benchmarks

Phase 2a gate output — numerical equivalence and CPU performance of the Phase 2a
JAX spike (`src/inf_bp_texts/models/_jax/`) vs the torch reference implementation.

## Equivalence gate (Phase 2a.2)

`tests/test_jax_torch_equivalence.py` — **58/58 passed** on 2026-06-18.

| Check | Tolerance | Result |
|---|---|---|
| JAX reproduces all `_GOLDEN_*` constants from `test_regression_likelihoods.py` | `atol=1e-4` | ✓ |
| JAX matches torch across parameter/age/epoch grid (extinction, spine, non-spine, populations, combined LL) | `atol=1e-8` | ✓ |
| `jax.grad` matches torch autograd and analytic derivatives (`mathematical_derivatives.py`) | `atol=1e-6` | ✓ |

The equivalence gate is **passed**.  The JAX spike is numerically correct to float64
precision for all single- and multi-epoch configurations tested.

## CPU performance (Phase 2a.4)

Measured 2026-06-18 on NVIDIA A40-12Q host, JAX CPU backend.
`n_repeats=10` warm repeats; cold (JIT) not shown.  Times in milliseconds.

**Important caveat**: the Phase 2a spike runs *eagerly* (not JIT-compiled).  The
~300ms multi-epoch cost is dominated by Python-level `lax.scan` dispatch overhead,
not arithmetic.  Phase 2b will `jax.jit` the math kernels, expected to collapse
this to the XLA-compiled steady-state cost (a few ms per call).  The single-epoch
result (11 ms JAX vs 3–12 ms torch) is already meaningful because the single-epoch
path does not use `lax.scan`.

### Likelihood (forward pass): JAX eager vs torch

| epochs | n_ages | JAX warm (ms) | torch (ms) | speedup |
|--------|--------|---------------|------------|---------|
| 1 | 10 | 11.37 | 2.97 | 0.3× |
| 1 | 100 | 11.51 | 3.83 | 0.3× |
| 1 | 1000 | **11.76** | **12.44** | **1.1×** |
| 2 | 10 | 327.59 | 9.31 | 0.0× |
| 2 | 100 | 287.71 | 20.17 | 0.1× |
| 2 | 1000 | 335.76 | 152.73 | 0.5× |
| 6 | 10 | 338.62 | 10.28 | 0.0× |
| 6 | 100 | 328.26 | 24.00 | 0.1× |
| 6 | 1000 | 347.49 | 159.92 | 0.5× |

### Gradient (non-spine ages only): JAX `value_and_grad` vs torch `.backward()`

Populations log-lik is excluded from the JAX grad benchmark because the Phase 2a
spike's populations kernel retains `float()` guards preventing JAX tracing.
Both columns measure `d(non_spine_ages_log_lik)/d(birth, death)`.

| epochs | n_ages | JAX grad (ms) | torch grad (ms) | speedup |
|--------|--------|---------------|-----------------|---------|
| 1 | 10 | 24.12 | 1.25 | 0.1× |
| 1 | 100 | 24.33 | 1.71 | 0.1× |
| 1 | 1000 | **25.32** | **6.04** | **0.2×** |
| 2 | 10 | 406.56 | 3.40 | 0.0× |
| 2 | 100 | 395.04 | 15.94 | 0.0× |
| 2 | 1000 | 357.87 | 159.62 | 0.5× |
| 6 | 10 | 406.76 | 3.73 | 0.0× |
| 6 | 100 | 404.65 | 16.57 | 0.0× |
| 6 | 1000 | 399.95 | 165.85 | 0.4× |

### Simulation baseline (Gillespie, backend-independent)

| epochs | mean (ms) | mean n\_surviving |
|--------|-----------|-------------------|
| 1 | 3.6 | 4 |
| 2 | 3.0 | 1 |
| 6 | 2.9 | 0 |

## Interpretation and go/no-go

### Spike results are not representative of Phase 2b performance

The spike's eager Python dispatch explains the multi-epoch slowdown:

- **Eager `lax.scan`**: each `lax.scan` call dispatches to XLA sequentially in
  Python; there is no XLA compilation.  For the k=6 epoch scan, this adds ~300ms
  of Python overhead regardless of n_ages.
- **No JIT caching**: the first and subsequent calls pay the same overhead; there is
  no amortisation.
- **Single epoch (no scan)**: vectorised `jnp` ops with no scan — here the spike is
  only 3–4× slower than torch, close to expected eager-vs-eager parity.

In Phase 2b, `jax.jit` over the math kernels collapses XLA dispatch to a one-time
compilation cost (~seconds).  Subsequent calls on fixed-shape inputs will run at
hardware speed.  Expected Phase 2b characteristics:

- Single-epoch, small n_ages: JAX JIT ~comparable to torch (both are small-kernel,
  single-device workloads where Python overhead dominates).
- Multi-epoch, large n_ages (≥100): JAX JIT expected to match or beat torch as
  vector ops dominate; `lax.scan` fuses the epoch loop into a single kernel.
- GPU (A40): torch CUDA is available; JAX CUDA fails due to XLA CUDA VMM
  incompatibility on the A40-12Q vGPU (see Phase 3a results below).

### Go/no-go decision

**Proceed to Phase 2b.**

Rationale:

1. **Equivalence is confirmed**: all 58 equivalence-gate tests pass at the required
   tolerances.  The JAX path is numerically correct.
2. **Performance will improve with JIT**: the eager spike's Python overhead is
   the sole cause of the multi-epoch slowdown.  JIT compilation addresses this;
   there is no fundamental algorithmic reason to prefer torch.
3. **Code-tidiness benefit is concrete**: `mathematical_derivatives.py` (274 lines
   of hand-coded analytic gradients — already containing bugs) becomes deletable
   under `jax.grad`.  The `basedpyright` suppressions for torch's incomplete stubs
   (`reportUnknownMemberType`, etc.) can be re-enabled with `jaxtyping` types.
4. **Torch removal is clean**: torch touches the math/optimizer/Bayesian layers
   only.  The simulation (Gillespie, numpy) is already backend-independent.

## Phase 2b–2e completion (2026-06-19)

The full migration is complete. Final state:

- **All math kernels JIT-compiled** via `jax.jit` over `math_extinction`,
  `math_ages`, `math_populations`, and `inference/likelihood.py`. The eager-overhead
  slowdown observed in the Phase 2a spike does not apply; multi-epoch calls now run
  at XLA hardware speed after a one-time compilation.
- **Optimizer layer** replaced by `optimistix` (BFGS/L-BFGS-B via scipy) + `optax`
  (Adam/SGD) with `jax.value_and_grad` gradients. `mathematical_derivatives.py`
  (274 lines of hand-coded analytic gradients) deleted.
- **Bayesian layer** ported to NumPyro NUTS (`inference/bayesian.py`).
- **`torch` completely absent** from `src/` and `tests/`.
  `grep -r "import torch" src/ tests/` returns nothing.
- **Test suite**: 595 passed, 3 pre-existing failures unrelated to the migration.
  All 10 golden-value regression tests pass.
- **basedpyright**: 0 errors, 5 expected warnings (missing stubs for numpyro/optax).

## Post-JIT benchmark (Phase 2b–2e, 2026-06-19)

Measured on the same NVIDIA A40-12Q host (CPU backend — CUDA jaxlib not installed
on this deployment). `n_repeats=10` warm repeats; cold (first-call, includes
XLA compilation) shown separately. Script: `scripts/bench_jax_jit.py`.

The NLL closure benchmarked mirrors `_compute_nll` inside the fitting loop:
`jit(value_and_grad(lambda log_params: -(pop_ll + spine_ll + non_spine_ll)))`.
All validation bypassed (`_skip_normalization=True`) so the JIT path is fully
traceable. Eager numbers match the Phase 2a spike because neither uses JIT.

### Likelihood (forward): eager vs JIT warm (ms)

| epochs | n_ages | eager (ms) | JIT warm (ms) | speedup |
|--------|--------|-----------|---------------|---------|
| 1 | 10 | 8.37 | 0.06 | **132×** |
| 1 | 100 | 10.61 | 0.10 | 109× |
| 1 | 1000 | 11.15 | 0.17 | 65× |
| 2 | 10 | 320.74 | 0.09 | **3,512×** |
| 2 | 100 | 314.20 | 0.24 | 1,310× |
| 2 | 1000 | 319.93 | 0.41 | 772× |
| 6 | 10 | 343.07 | 0.10 | **3,518×** |
| 6 | 100 | 337.30 | 0.12 | 2,870× |
| 6 | 1000 | 346.08 | 0.22 | 1,564× |

### value_and_grad: eager vs JIT warm (ms)

| epochs | n_ages | eager (ms) | JIT warm (ms) | speedup |
|--------|--------|-----------|---------------|---------|
| 1 | 10 | 83.02 | 0.06 | **1,404×** |
| 1 | 100 | 72.22 | 0.20 | 366× |
| 1 | 1000 | 70.05 | 0.27 | 263× |
| 2 | 10 | 835.90 | 0.12 | **7,126×** |
| 2 | 100 | 844.61 | 0.27 | 3,099× |
| 2 | 1000 | 816.94 | 0.46 | 1,768× |
| 6 | 10 | 867.95 | 0.08 | **10,453×** |
| 6 | 100 | 900.92 | 0.17 | 5,160× |
| 6 | 1000 | 794.85 | 0.42 | 1,870× |

### Interpretation

JIT compilation eliminates the Python-level `lax.scan` dispatch overhead
(~320–345 ms per call in eager mode). With JIT, all nine epoch/ages combinations
settle below **0.5 ms** per forward pass and per `value_and_grad` call.

**Implication for the optimizer:** the current `_optimizer_runners.py` calls
`jax.value_and_grad(compute_nll_fn)` eagerly at each L-BFGS-B iteration.
Wrapping this with `jax.jit` before the optimization loop would reduce per-step
time from ~320–870 ms to < 0.5 ms — a 3–10,000× wall-clock speedup for the
multi-epoch fitter. The JIT cold-compile cost (a few seconds) is amortized over
hundreds of optimizer iterations.

## Phase 3a — CUDA jaxlib install + A40-12Q GPU verification (2026-06-19)

### Install

`jax[cuda12]>=0.4.38` was added to a `[dependency-groups] gpu` group in
`pyproject.toml` (linux-only marker so the default CPU lock is unaffected):

```toml
[dependency-groups]
gpu = [
    "jax[cuda12]>=0.4.38; sys_platform == 'linux'",
]
```

`uv sync --group gpu` (or `uv sync --all-groups`) installs `jax-cuda12-pjrt`,
`jax-cuda12-plugin`, and the bundled NVIDIA wheel stack (`nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`, etc.) into the venv.

**cuDNN pip-wheel caveat (≥9.10):** `nvidia-cudnn-cu12>=9.10` no longer ships
`libcudnn.so.9`; it ships only extension components
(`libcudnn_engines_tensor_ir.so.9`, `libcudnn_ext.so.9`). JAX's CUDA plugin
loads `libcudnn.so.9` dynamically; with the 9.23.x wheel installed JAX
reports *"Unable to load cuDNN"* and falls back to CPU. Versions `<9.10`
(e.g. 9.9.0.52, 664 MB) include the full library and resolve this error.
The resolved runtime issue is pinning `"nvidia-cudnn-cu12<9.10"` alongside the
`jax[cuda12]` extra in the `gpu` group if a future JAX version relaxes the
`>=9.8` lower-bound **and** the VMM issue is fixed.

### VMM incompatibility — outcome: STILL FAILS

Even after cuDNN is correctly found, XLA refuses the A40-12Q vGPU:

```
E Failed to create stream executor for device CUDA:0: Device 0 does not
  support CUDA Virtual Memory Management (VMM).  VMM is required for device
  memory allocation in XLA.
```

This is the same root cause as Phase 2a: XLA's stream executor unconditionally
requires `cuMemCreate`/`cuMemMap` (CUDA Virtual Memory Management) for device
memory, which the A40-12Q vGPU profile (`12288 MiB` virtual) does not expose to
the guest.  The standard workarounds (`XLA_PYTHON_CLIENT_ALLOCATOR=platform`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`) change JAX's high-level allocator policy
*after* XLA initialises; they do not bypass the VMM check in
`platform_util.cc:271` and have no effect.

**Conclusion for Phase 3b:** NumPyro Bayesian GPU benchmarking cannot be
completed on this host.  All subsequent benchmarking uses the CPU backend
(`JAX_PLATFORMS=cpu`).  If access to a hardware GPU (non-vGPU) becomes
available, re-run `uv sync --group gpu` and `python scripts/bench_jax_jit.py
--backend gpu` to add GPU rows.

## Phase 3c — JIT optimizer hot path (2026-06-19)

### What changed

`_log_likelihood_jit_core` (private, `inference/likelihood.py`) extracts the
jittable math from `log_likelihood`: it calls the three math kernels directly
with `_skip_normalization=True`, performs host-side spine pre-resolution once
(mirrors `log_likelihood` lines 216–232), and uses `jnp.where` instead of
`bool(traced)` guards.

In `_run_optimizer_strategies` (`inference/fitting.py`):
1. Spine is auto-detected host-side once per fit (concrete arrays).
2. A `_jittable_nll` closure is built that calls `_log_likelihood_jit_core`
   with static closure constants.
3. `val_and_grad_fn = jax.jit(jax.value_and_grad(_jittable_nll))` is built
   once; XLA compiles on the first optimizer step.
4. `val_and_grad_fn` is passed to both LBFGS and Adam/SGD runners instead of
   `compute_nll_fn`; the runners no longer call `jax.value_and_grad` internally.

**Coverage:** `initial_population == 1` (the common case) is fully JIT'd.
`initial_population > 1` also uses the same JIT path because the Python loop
in `_compute_populations_log_lik_multi_start` unrolls at XLA trace time using
the static `n0 = initial_population` constant.

**JAX tracing hazard resolved (2026-06-19):** Closure JAX arrays are abstract
tracers inside `jax.jit`, not concrete values.  The original
`_log_likelihood_jit_core` had `int(float(ns.sum()))` (an early-return guard
for zero population) that raised `ConcretizationTypeError` at trace time.
This was removed: the `n_non_spine == 0` guard already short-circuits the ages
block, and the final `jnp.where(isfinite(total), total, -inf)` handles the
zero-population edge case identically without any host-side sum.

### Expected speedup

From `bench_jax_jit.py` benchmarks (Phase 2b–2e, isolated NLL closure):

| epochs | eager warm (ms) | JIT warm (ms) | speedup |
|--------|-----------------|---------------|---------|
| 1 | 72–83 | 0.06–0.27 | 263–1,404× |
| 2 | 816–844 | 0.12–0.46 | 1,768–7,126× |
| 6 | 795–901 | 0.08–0.42 | 1,870–10,453× |

The optimizer loop (LBFGS-B + Adam) runs hundreds of `value_and_grad` calls per
fit; the JIT cold-compile cost (~seconds) is amortized.  End-to-end fitter
wall-clock speedup for 6-epoch fits is expected to be 3–4 orders of magnitude.

## Phase 3d — End-to-end fitter benchmark (2026-06-19)

Script: `uv run python scripts/bench_jax_jit.py bench-fitter`.
LBFGS only, `estimate_uncertainty=False`.  Measured on CPU backend.

### Fitter wall-clock time per call (ms)

| epochs | n_ages | cold (ms) | warm (ms) | LBFGS iters |
|--------|--------|-----------|-----------|-------------|
| 1 | 10 | 9,730 | 628 | 1 |
| 1 | 100 | 3,882 | 667 | 1 |
| 1 | 1,000 | 3,831 | 681 | 1 |
| 2 | 10 | 6,526 | 1,245 | 1 |
| 2 | 100 | 2,915 | 1,319 | 1 |
| 2 | 1,000 | 2,959 | 1,415 | 1 |
| 6 | 10 | 6,547 | 1,324 | 1 |
| 6 | 100 | 1,918 | 1,397 | 1 |
| 6 | 1,000 | 2,076 | 1,468 | 1 |

### Interpretation

**LBFGS converges in 1 iteration on these synthetic datasets.** The initial
guess (birth=0.05, death=0.04) is already near the MLE for data generated from
similar rates, so the gradient threshold (5×10⁻⁴) is met after the first line
search.  This means the fitter timings above measure the *Python overhead floor*
per call (validation, spine pre-resolution, closure building, scipy interface,
result assembly) rather than the optimizer arithmetic.

**Warm times are independent of `n_ages`** (628–681 ms for 1-epoch regardless
of 10, 100, or 1,000 ages) — confirming the bottleneck is Python overhead, not
XLA computation.

**For multi-iteration fits the JIT benefit is decisive.** Combining the Python
overhead floor with the per-step times from the Phase 2b–2e NLL closure
benchmark:

| epochs | Python overhead | JIT step (warm) | Eager step | 100-iter JIT | 100-iter eager |
|--------|----------------|-----------------|------------|--------------|----------------|
| 1 | ~650 ms | 0.06–0.27 ms | 72–83 ms | ~677 ms | ~9,000 ms |
| 6 | ~1,400 ms | 0.08–0.42 ms | 795–901 ms | ~1,440 ms | ~91,000 ms |

At 100 optimizer steps a 6-epoch fit completes in ~1.4 s under JIT vs ~91 s
under the old eager path — a **63×** end-to-end speedup.  At 500 steps (the
default L-BFGS-B budget) the gap grows to ~1.6 s vs ~450 s (~280×).

**Cold time** (2–10 s) includes XLA compilation plus the Python overhead for
the first fit.  Compilation is cached by JAX for subsequent fits with the same
input shapes, so the second and later fits for a given (epochs, n_ages)
configuration pay only the warm overhead.

## Raw data

Phase 2a torch-vs-JAX eager: `_output/jax_torch_benchmarks.json`.
Phase 2b–2e post-JIT: `_output/jax_jit_benchmarks.json` (`results` key).
Phase 3d fitter: `_output/jax_jit_benchmarks.json` (`fitter_results` key).
