# JAX Migration Benchmarks

Record of the torch → JAX migration's equivalence gate and performance numbers.
The migration itself is complete (see `CHANGELOG.md`); this doc keeps the
headline evidence and the still-open GPU item.

## Equivalence gate (2026-06-18)

`tests/test_jax_torch_equivalence.py` — 58/58 passed: JAX reproduced all golden
regression constants, matched torch across the parameter/age/epoch grid for
extinction/spine/non-spine/populations/combined log-likelihood (`atol=1e-8`),
and `jax.grad` matched torch autograd and the analytic derivatives (`atol=1e-6`).

## JIT speedup (CPU, NVIDIA A40-12Q host, 2026-06-19)

Eager JAX (pre-`jax.jit`) paid ~300ms of Python-level `lax.scan` dispatch
overhead per multi-epoch call. After JIT-compiling the math kernels
(`math_extinction`, `math_ages`, `math_populations`, `inference/likelihood.py`):

| epochs | eager warm (ms) | JIT warm (ms) | speedup |
|--------|-----------------|---------------|---------|
| 1 | 8–83 | 0.06–0.27 | up to 1,400× |
| 2 | 315–845 | 0.09–0.46 | up to 7,100× |
| 6 | 337–901 | 0.08–0.42 | up to 10,500× |

End-to-end fitter benchmark (LBFGS, `scripts/bench_jax_jit.py bench-fitter`):
warm fits run in ~0.6–1.5s regardless of `n_ages` (Python/scipy overhead
dominates, not per-step cost); the JIT-vs-eager wall-clock advantage scales
with iteration count, reaching 20–60× for real 6-epoch manuscript fits
(50–200 iterations).

## Final migration state (2026-06-19)

- Optimizer layer: `optimistix` (BFGS/L-BFGS-B) + `optax` (Adam/SGD) via
  `jax.value_and_grad`; the 274-line hand-coded `mathematical_derivatives.py`
  was deleted.
- Bayesian layer: NumPyro NUTS (`inference/bayesian.py`).
- `torch` completely absent from `src/`/`tests/`.

## Open item: real-GPU benchmarking is blocked

`jax[cuda12]` was installed via the `gpu` dependency group and a cuDNN pin
(`nvidia-cudnn-cu12<9.10` — wheels ≥9.10 drop `libcudnn.so.9`, causing a silent
CPU fallback) was added to work around a loader issue. Even with cuDNN
resolved, XLA refuses the host's A40-12Q **vGPU**: it requires CUDA Virtual
Memory Management (`cuMemCreate`/`cuMemMap`), which this vGPU profile does not
expose to the guest. `XLA_PYTHON_CLIENT_ALLOCATOR`/`_PREALLOCATE` do not help —
the check happens before those allocator settings apply.

**Status: pending** a real (non-vGPU) GPU host. Once available: `uv sync
--group gpu` then `uv run python scripts/bench_jax_jit.py --backend gpu` to
add GPU rows. See `docs/design/GPU_RENDERING_PROTOCOL.md` for the render-side
workflow once a working GPU host is in hand.

## Raw data

`_output/jax_torch_benchmarks.json` (torch-vs-JAX eager),
`_output/jax_jit_benchmarks.json` (`results` + `fitter_results` keys).
