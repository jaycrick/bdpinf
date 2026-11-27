# bdpinf

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/jaycrick/bdpinf/graph/badge.svg?token=5QNFKWSL7G)](https://codecov.io/gh/jaycrick/bdpinf)

Inference, simulation, and visualisation for linear birth-death (branching)
processes: MLE fitting (JAX, JIT-compiled) and Bayesian inference
(NumPyro/NUTS) for single- and multi-epoch birth/death rates, exact-event and
lifespan-at-birth simulators and multilevel splitting for rare-event
simulation (implemented in [`bdsurv`](https://github.com/jaycrick/bdsurv) and
wrapped here to return `BirthDeathProcess`), and a shared Altair chart
theme/export chokepoint.
No domain-specific content --- see
[`nt-manuscripts`](https://github.com/jaycrick/nt-manuscripts) for the New
Testament manuscript application built on top of this library.

## Installation

Install [uv](https://docs.astral.sh/uv/) if you haven't already:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install dependencies:

```sh
git clone https://github.com/jaycrick/bdpinf.git
cd bdpinf
uv sync
```

To use bdpinf as a dependency of another uv project,
pin it to a release tag
(it is not on PyPI, so the tag is the release):

```sh
uv add "bdpinf @ git+https://github.com/jaycrick/bdpinf" --tag v0.8.4
```

which records a lower bound in `[project] dependencies`
and the tag in `[tool.uv.sources]`:

```toml
[project]
dependencies = ["bdpinf>=0.8.4"]

[tool.uv.sources]
bdpinf = { git = "https://github.com/jaycrick/bdpinf", tag = "v0.8.4" }
```

For development (includes testing and documentation tools):

```sh
uv sync --group dev
```

## Quick Start

```python
from bdpinf import BirthDeathProcess
from bdpinf.inference import AgeData, BirthDeathModel, fit

process = BirthDeathProcess(creation_times=[100, 200, 300], epoch_end_times=500)
data = AgeData.from_groups([process.ages], end_time=500.0)

# Piecewise-constant rates (one per epoch) ...
result = fit(data, BirthDeathModel.piecewise_constant([500.0]))
process.update_parameters(result.birth_values, result.death_values, result)

# ... or piecewise-linear rates, warm-started from the constant fit.
linear = fit(data, BirthDeathModel.piecewise_linear([250.0, 500.0]), init=result)
print(linear.params, linear.log_likelihood)
```

Render the research-notebook site (notebooks live under `qmd_scripts/`; output
goes to `_output/`).
This includes the slow, multi-chain Bayesian validation
notebooks --- see "HPC / batch rendering" below for a faster alternative that
precomputes the heaviest simulate-then-fit datasets first:

```sh
uv run quarto render
```

## Key Features

- Birth-death process modeling: MLE fitting (JAX, trust-region Newton) and
  Bayesian inference (NumPyro/NUTS) for piecewise-constant or piecewise-linear
  birth/death rates, with room for other parametric families (sinusoids,
  splines) and constraints by parameter name,
  with a Quarto research-notebook site of validation studies against
  synthetic data with known ground-truth parameters
- Exact-event (`simulate_birth_death`, constant or linear rates) and lifespan-at-birth
  simulators, plus survival-conditioned simulation (`simulate_contour_given_survival`,
  `simulate_survival_splitting`), delegated to `bdsurv` 0.9
  and returned as `BirthDeathProcess`
- A shared Altair chart theme and export chokepoint (`visualization.save_chart`)

## Date-range uncertainty

Interval-censored dates are handled by `AgeData.from_ranges` (the density is averaged over each range)
and `simulation.age_sampling.sample_ages_in_ranges_with_spine`.
The date-range KDE resampling and its bootstrap, which used to live here, are in the `nt-manuscripts` application repo.

## Development

Setup, checks (tests, Ruff, basedpyright, prose linting) and the notebook and
API-reference builds are documented in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Result numbers from the accompanying paper (in preparation) appear as
`[paper: Theorem ?]` placeholders until they are fixed; search for `[paper:`.

### HPC / batch rendering

`scripts/generate_recovery_studies.slurm` precomputes the simulate-then-fit
datasets consumed by the heaviest validation notebooks, chained via
`scripts/render_all.sh` before `scripts/render_cpu_parallel.slurm` renders
everything (including the Bayesian NUTS validation notebooks).
GPU rendering
was benchmarked and found 1.6×--8× *slower* than CPU on every config tested,
so Bayesian notebooks render on CPU too; see
`docs/design/GPU_RENDERING_PROTOCOL.md`.

To copy the rendered outputs and cached results back from the cluster,
pull the remote checkout over the local one with `rsync`
(`<hpc-host>` is your SSH host alias, and the paths are wherever each checkout lives):

```bash
rsync -avzP --exclude="/.*" <hpc-host>:/path/to/remote/bdpinf/ /path/to/local/bdpinf
```

The trailing slash on the source copies its contents rather than the directory itself,
and `--exclude="/.*"` skips top-level dotfiles and directories (`.git`, `.venv`, ...)
so each machine keeps its own environment and git state.
This is an update-only pull (no `--delete`), so local-only files survive the sync.
SLURM job logs land in repo-root `logs/` rather than inside `_output/`
(the render tree a `quarto render` wipes and regenerates on every run),
so this whole-repo pull carries both along without one clobbering the other.
