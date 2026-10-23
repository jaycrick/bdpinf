# Fitting: derivatives, optimisers and compile cost

This note records what was measured when the fitter was rebuilt around rate-function models (piecewise constant, piecewise linear, room for sinusoids), and why the implementation looks the way it does.
Everything below is reproducible with `uv run python scripts/bench_fitting.py` (add `--quick` for a short run, `--only <section>` for one section).
Numbers are from one CPU run (float64); they show orders of magnitude and rankings, not exact timings.

## Decisions

1. **Derivatives come from JAX autodiff**, not hand-coded formulas.
2. **Hessian by forward-over-forward** differentiation, gradient by reverse mode: about 35--50 % less compile time than the default Hessian for the 2--10 parameters involved, with the same accuracy.
3. **Minimise** the negative log-likelihood with a trust-region Newton method (`scipy` `trust-exact`), with L-BFGS-B as fallback, rather than solving the score equations.
   The Hessian at the optimum certifies the result (`FitResult.min_hessian_eig`, `at_interior_optimum`).
4. **Gauss-Legendre with 32 nodes per segment** for non-piecewise-constant rates.
5. **Compile once**: the jitted objective is cached per model and constraint structure, takes the data and fixed values as arguments, and the age vector is padded (masked) to a power-of-two bucket.

## What the Mathematica notebooks provide

The notebooks (`Paper01_AgeBDPVarEnv/Wolfram`) contain:

- closed forms for $\mathcal W$ and the age densities, including a piecewise-linear pair built by the restart principle from the linear-linear closed form (which involves `Erf` and `Abs` and is singular at equal slopes);
- symbolic scores and Hessians **only for one epoch of constant rates**, in $(\lambda,\gamma)$ (expression sizes of 800--2100 leaves for the individual Hessian blocks);
- the Fisher information decomposition $I_T=I_Z+I_{A^*}+\mathbb E[Z_T-1]\,I_A$ and the large-$T$ orders of the estimators.

There are no explicit derivatives for time-varying rates.
Their numerical values (for example $\mathcal W(T)=5.661968424654191$ for the kinked linear pair) are used as test fixtures (`tests/test_math_general.py`), and the constant-rate scores are used in `tests/test_explicit_scores.py` and in the benchmark below.

## Explicit scores against autodiff (one epoch, constant rates)

The explicit scores were differentiated in Mathematica with `D[Log[...], {{l, m}}]` and translated to NumPy.

| ages | value rel. diff | gradient rel. diff | autodiff eval (ms) | explicit eval (ms) |
| ---- | --------------- | ------------------ | ------------------ | ------------------ |
| 79 | 9e-16 | 5e-14 | 0.044 | 0.043 |
| 1218 | 5e-16 | 6e-14 | 0.074 | 0.095 |
| 11443 | 2e-16 | 3e-13 | 0.44 | 0.57 |

The two agree to rounding error, which independently validates both.
Per-evaluation cost is the same.
A whole single-epoch fit takes 2.6 ms warm with the jitted autodiff objective against 0.8 ms for L-BFGS on the explicit formulas, and 1.0 s on the first call (compilation).
The 2 ms is not worth a second implementation that only exists for one parameterisation, so the explicit forms stay as tests, not code.

## Derivative variants (compile and run time)

Selected rows (151 ages; "PL K" is continuous piecewise linear with $K$ epochs, so $K+1$ knots per rate).

| model | params | variant | compile (s) | run (ms) | max rel. err. vs finite differences |
| ----- | ------ | ------- | ----------- | -------- | ------------------------------------- |
| PC K=1 | 2 | gradient, reverse | 0.30 | 0.040 | 3e-10 |
| PC K=1 | 2 | Hessian, fwd-over-rev | 0.70 | 0.059 | 2e-8 |
| PC K=1 | 2 | Hessian, fwd-over-fwd | 0.54 | 0.056 | 2e-8 |
| PC K=3 | 6 | Hessian, fwd-over-rev | 0.95 | 0.17 | 1e-8 |
| PC K=3 | 6 | Hessian, fwd-over-fwd | 1.00 | 0.14 | 1e-8 |
| PL K=2 | 6 | gradient, reverse | 1.5 | 0.42 | 1e-9 |
| PL K=2 | 6 | Hessian, fwd-over-rev | 2.9 | 1.0 | 3e-8 |
| PL K=2 | 6 | Hessian, fwd-over-fwd | 1.5 | 1.3 | 3e-8 |
| PL K=4 | 10 | Hessian, fwd-over-rev | 3.2 | 1.4 | 1e-8 |
| PL K=4 | 10 | Hessian, fwd-over-fwd | 1.8 | 3.0 | 1e-8 |

Run time is a millisecond or less for every variant; compilation is the cost, and it grows with the size of the traced graph (not with the number of quadrature nodes).
Forward-over-forward halves the compile time of the piecewise-linear Hessians at a small run-time cost, which is the right trade for a Hessian that is evaluated tens of times.
Fusing value, gradient and Hessian into one jitted function did not reduce compile time.

## Gauss-Legendre nodes

Absolute error in the log-likelihood against a 512-node reference, for continuous piecewise-linear rates (segment length 3).

| scenario (knot values up to) | 8 nodes | 16 | 32 | 64 |
| ---------------------------- | ------- | -- | -- | -- |
| moderate (1.6) | 1e-12 | 6e-14 | 1e-13 | 6e-14 |
| steep (8) | 1e-7 | 0 | 2e-13 | 1e-13 |
| extreme (80) | 0.25 | 8e-5 | 2e-12 | 2e-12 |

32 nodes is safe well beyond realistic rates and costs 0.3 ms per evaluation.
It is **not** safe without limit: with fitted rates of order $10^3$ over a segment of length 5 (a boundary solution the optimiser can wander towards), a single panel of 32 nodes cannot resolve $e^{-\Gamma}$ and the likelihood is wrong, while the exact piecewise-constant backend is unaffected.
This is one reason piecewise-constant rates keep their own exact path.

## Optimisers

Ten simulated datasets per scenario; every solver starts from the same moment estimators.
"Reached best optimum" counts datasets where the solver converged and its log-likelihood is within $10^{-4}$ of the best of the three.
The gap columns are best minus achieved log-likelihood.
Seconds are warm (compiled objective reused).

| scenario | solver | reached best | median gap | max gap | median evaluations | median seconds |
| -------- | ------ | ------------ | ---------- | ------- | ------------------ | -------------- |
| constant | Newton (trust-exact) | 10/10 | 6e-14 | 1e-5 | 6 | 0.003 |
| constant | L-BFGS-B | 10/10 | 5e-11 | 2e-8 | 11 | 0.003 |
| constant | root of score | 10/10 | 0 | 4e-13 | 12 | 0.001 |
| constant, $\mu=0.02$ | Newton | 5/10 | 1e-4 | 6e-4 | 10 | 0.005 |
| constant, $\mu=0.02$ | L-BFGS-B | 7/10 | 3e-5 | 1e-3 | 24 | 0.005 |
| constant, $\mu=0.02$ | root of score | 10/10 | 0 | 0 | 87 | 0.011 |
| two epochs | Newton | 10/10 | 6e-6 | 2e-5 | 14 | 0.006 |
| two epochs | L-BFGS-B | 10/10 | 4e-8 | 7e-7 | 41 | 0.007 |
| two epochs | root of score | 6/10 | 0 | **296** | 127 | 0.012 |
| linear birth, constant death | Newton | 9/10 | 0 | 5e-12 | 14 | 0.085 |
| linear birth, constant death | L-BFGS-B | 9/10 | 1e-10 | **7e7** | 36 | 0.070 |
| linear birth, constant death | root of score | 0/10 | -- | -- | 26 | 0.036 |

Reading the table:

- With one parameter pair and an interior optimum all three agree, and root finding is as good as anything.
- Near the zero boundary the log-rate score decays smoothly towards zero, so the minimisers stop when the per-observation gradient tolerance is met, a few $10^{-4}$ log-likelihood units short of the supremum; the root finder runs the rate out to $10^{-16}$ in six to eight times as many evaluations.
  Nothing statistically relevant is lost.
- With several epochs or a piecewise-linear birth rate, solving the score equations lands on a non-maximising stationary point or fails outright (gaps up to 296 log-likelihood units; no success at all for the linear model).
  Newton with a trust region never did.
- L-BFGS-B had one catastrophic failure on the linear model (a gap of $7\times10^7$ means it stopped at a point where the likelihood was invalid), which is why the fallback chain is Newton first.
- Newton needs about one third to just over one half of the evaluations of L-BFGS-B, and each evaluation also supplies the Hessian that certifies the optimum.

The mirror-image mode $\lambda\leftrightarrow\mu$ (ages are symmetric in the two rates) is another reason for minimisation with multiple starts rather than root finding: the stationary set contains both.

## Old fitter against new

The same fifteen simulated datasets (one to three epochs) were fitted with the previous L-BFGS-then-Adam fitter and with `fit`.

| | previous | new |
| --- | --- | --- |
| total time (15 fits) | 95 s | 17.5 s |
| median per fit | 5.4 s | 1.1 s (8 ms once compiled) |
| log-likelihood | -- | never lower by more than $6\times10^{-6}$; higher by 0.1--0.4 on four boundary cases where the previous fitter stopped early |

The remaining time is compilation: roughly 1 s per model structure and array-size bucket.
Repeated fits of the same structure (bootstrap, profile likelihood, recovery replicates) cost milliseconds.

## Statistical caveats that shaped the design

- From survivor ages alone the size term identifies only $\pi(T)$, so flexible models are weakly identified: early-time rates and the death rate trade off along a ridge.
  A six-parameter piecewise-linear fit to a few hundred ages routinely puts a knot at the boundary or at a very large value.
  Constrain the model (for example linear birth with constant death) or supply more data.
- The growth rate has an $O(1/T)$ non-normal limit ($T(\hat\gamma-\gamma)\to\ln E$, $E\sim\mathrm{Exp}(1)$), so Wald intervals for growth-type parameters are not asymptotically valid; use `profile_likelihood_interval`.
- Quadrature needs extreme rates to be bounded; the optimiser works on the log scale and keeps rates positive but does not cap them.
