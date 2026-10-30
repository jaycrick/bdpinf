r"""Measure the compile time, peak memory and accuracy of the Hessian variants.

The Hessian of the NTVMR linear model (13 rates, 7 segments, date ranges) took
about 285 s to compile locally and was OOM-killed above 24 GB on the HPC.
This builds the objective for the fixed-knot truth model, compiles one Hessian
variant in a fresh process, and prints the compile time, steady-state time, peak
resident memory and the Hessian itself (saved so variants can be compared).

Usage
-----
    uv run python scripts/diagnose_hessian_compile.py --mode fd-grad
    uv run python scripts/diagnose_hessian_compile.py --compare
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
from bench_ntvmr_knots import DATA_PATH, build_model, fixed_constraints, real_data

from bdpinf.inference.fitting import _problem  # pyright: ignore[reportPrivateUsage]

OUT = Path("_output/diagnose")


def peak_gb() -> float:
    """Return the peak resident memory of this process in GB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1e9 if sys.platform == "darwin" else 1e6)


def measure(mode: str, tag: str) -> None:
    """Compile one Hessian variant and report what it cost."""
    spec = json.loads(DATA_PATH.read_text())
    model = build_model(spec)
    problem = _problem(
        real_data(spec), model, fixed_constraints(spec), hessian_mode=mode
    )
    start = jnp.asarray(
        problem.compiled.reduce(
            model.constant_natural(0.0033, 0.003, spec["knot_initial"])
        )
    )
    problem.value_and_grad(start)
    print(f"[{tag}] grad compiled, peak {peak_gb():.2f} GB", flush=True)
    began = time.perf_counter()
    hess = np.asarray(problem.hessian(start))
    compile_seconds = time.perf_counter() - began
    began = time.perf_counter()
    for _ in range(5):
        problem.hessian(start).block_until_ready()
    steady = (time.perf_counter() - began) / 5
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"hessian_{tag}.npy", hess)
    print(
        f"[{tag}] mode={mode} compile+run {compile_seconds:.1f}s, "
        f"steady {steady:.3f}s, peak {peak_gb():.2f} GB, "
        f"hessian norm {np.linalg.norm(hess):.6g}",
        flush=True,
    )


def compare() -> None:
    """Print each saved Hessian's relative difference from the first one found."""
    files = sorted(OUT.glob("hessian_*.npy"))
    if not files:
        return
    ref = np.load(files[0])
    print(f"\nrelative Hessian difference from {files[0].name}")
    for path in files:
        h = np.load(path)
        print(f"  {path.name}: {np.linalg.norm(h - ref) / np.linalg.norm(ref):.2e}")


def main() -> None:
    """Parse the command line and run one variant or the comparison."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--mode", default="fd-grad")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare()
        return
    measure(args.mode, args.tag or args.mode)


if __name__ == "__main__":
    main()
