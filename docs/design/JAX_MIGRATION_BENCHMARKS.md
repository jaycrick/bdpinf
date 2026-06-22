# JAX Migration Benchmarks

Record of the torch → JAX migration's equivalence gate and performance numbers.
The migration itself is complete (see `CHANGELOG.md`); this doc keeps the
headline evidence, including the CPU-vs-GPU comparison.

## Equivalence gate (2026-06-18)

`tests/test_jax_torch_equivalence.py` --- 58/58 passed: JAX reproduced all golden
regression constants, matched torch across the parameter/age/epoch grid for
extinction/spine/non-spine/populations/combined log-likelihood (`atol=1e-8`),
and `jax.grad` matched torch autograd and the analytic derivatives (`atol=1e-6`).

## JIT speedup (CPU, NVIDIA A40-12Q host, 2026-06-19)

Eager JAX (pre-`jax.jit`) paid \~300ms of Python-level `lax.scan` dispatch
overhead per multi-epoch call.
After JIT-compiling the math kernels
(`math_extinction`, `math_ages`, `math_populations`, `inference/likelihood.py`):

  | epochs | eager warm (ms) | JIT warm (ms) | speedup       |
  | ------ | --------------- | ------------- | ------------- |
  | 1      | 8–83            | 0.06–0.27     | up to 1,400×  |
  | 2      | 315–845         | 0.09–0.46     | up to 7,100×  |
  | 6      | 337–901         | 0.08–0.42     | up to 10,500× |

End-to-end fitter benchmark (LBFGS, `scripts/bench_jax_jit.py bench-fitter`):
warm fits run in \~0.6--1.5s regardless of `n_ages` (Python/scipy overhead
dominates, not per-step cost); the JIT-vs-eager wall-clock advantage scales
with iteration count, reaching 20--60× for real 6-epoch manuscript fits
(50--200 iterations).

## CPU vs GPU (NVIDIA A100 80GB, 2026-06-22)

The A40-12Q **vGPU** blocker below no longer applies --- these numbers were
collected on a real (non-vGPU) A100 node:
`uv sync --group gpu`, then `uv run python scripts/bench_jax_jit.py main` /
`bench-fitter`, once with the default (CPU) backend and once with
`JAX_PLATFORMS=cuda`.
Results merge by backend into
`_output/jax_jit_benchmarks.json`.

JIT forward warm time (ms), CPU vs GPU:

  | epochs | n_ages | CPU (ms) | GPU (ms) | GPU speedup |
  | ------ | ------ | -------- | -------- | ----------- |
  | 1      | 10     | 0.015    | 0.068    | 0.22×       |
  | 1      | 1000   | 0.038    | 0.078    | 0.49×       |
  | 2      | 10     | 0.021    | 0.146    | 0.14×       |
  | 2      | 1000   | 0.072    | 0.152    | 0.47×       |
  | 6      | 10     | 0.030    | 0.228    | 0.13×       |
  | 6      | 1000   | 0.075    | 0.225    | 0.33×       |

End-to-end fitter warm time (ms, LBFGS), CPU vs GPU:

  | epochs | n_ages | CPU (ms) | GPU (ms) | GPU speedup |
  | ------ | ------ | -------- | -------- | ----------- |
  | 1      | 10     | 347      | 2202     | 0.16×       |
  | 1      | 1000   | 339      | 730      | 0.46×       |
  | 2      | 1000   | 728      | 1384     | 0.53×       |
  | 6      | 1000   | 748      | 2560     | 0.29×       |

**GPU is slower than CPU across this entire grid** --- at these problem sizes
(≤6 epochs, ≤1000 ages) per-call kernel-launch/dispatch overhead on the GPU
dominates over the tiny amount of actual compute, so CPU wins by 2×--8×.
This
matches the prediction in `docs/design/GPU_RENDERING_PROTOCOL.md`.
The Bayesian production-config benchmark
(`kashouh_family_k_bayesian_inference.qmd`, `_output/bayesian_benchmarks.json`)
tells the same story, and **scaling up chains/samples does not change the
verdict**: a 200-sample/1-chain fit ran in 6.2s CPU vs. 10.6s GPU (0.59×), and
the notebook's actual production config (4 chains, 1000 samples, 400 warmup,
re-executed with `--cache-refresh` to force a genuine non-cached run on each
backend) ran in 10.7s CPU vs. 17.3s GPU (0.62×) --- essentially the same ratio.
This refutes the earlier hypothesis that more chains/samples would amortize
the launch overhead: `chain_method="vectorized"` batches *chains* via `vmap`
for free, but NUTS is sequential *within* a chain, so the number of
kernel-launching leapfrog steps scales with `num_samples`, not `num_chains`.
Going 200→1000 samples just multiplies the per-step dispatch overhead on both
backends roughly equally.

**A much larger per-call model doesn't help either.** The same 1-chain/50-sample
config run against the `non_lectionary` NTVMR subset (3,351 manuscripts, six
epochs --- 67× more manuscripts and 3× more epochs than the Kashouh FamilyK
model) gave 10.3s CPU vs. 16.6s GPU (0.62×): the same ratio as every smaller
configuration above.
Across three independent scaling axes tested today
(samples, chains, and manuscript/epoch count), GPU loses by the same \~1.6×
margin every time, which rules out "the model is too small" as the
explanation.
The likely real cause: NUTS's tree-building involves substantial
data-*independent* control flow (recursive doubling, dynamic tree depth,
acceptance conditionals) that XLA's GPU backend executes less efficiently than
CPU --- GPUs are SIMD-throughput devices and this control flow is branchy and
mostly serial, regardless of how much data each likelihood evaluation
processes.
The untested lever consistent with this explanation is a much
wider `vmap` batch (dozens of chains, not 4) --- the one axis GPUs genuinely
parallelize well.

## Final migration state (2026-06-19)

- Optimizer layer: `optimistix` (BFGS/L-BFGS-B) + `optax` (Adam/SGD) via
  `jax.value_and_grad`; the 274-line hand-coded `mathematical_derivatives.py`
  was deleted.
- Bayesian layer: NumPyro NUTS (`inference/bayesian.py`).
- `torch` completely absent from `src/`/`tests/`.

## Raw data

`_output/jax_torch_benchmarks.json` (torch-vs-JAX eager),
`_output/jax_jit_benchmarks.json` (`results` + `fitter_results` keys, each with
a `cpu`/`gpu`/`_speedup` block), `_output/bayesian_benchmarks.json`
(`kashouh_family_k` keyed by backend).
