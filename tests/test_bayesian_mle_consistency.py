"""Sanity-baseline: posterior mean should land near the MLE estimate under weak priors.

These tests guard against the Bayesian path silently diverging from the validated
MLE path.  They run NUTS with real (short) chains, so they are marked slow and
excluded from the default fast suite.

Run explicitly with:
    uv run pytest tests/test_bayesian_mle_consistency.py -m slow -v
"""

import numpy as np
import pytest

from bdpinf.inference import (
    AgeData,
    BayesianMCMCConfig,
    BayesianPriorConfig,
    BirthDeathModel,
    fit,
    fit_bayesian,
)

# ---------------------------------------------------------------------------
# Shared synthetic datasets
# ---------------------------------------------------------------------------

# Single-epoch: λ=0.05, μ=0.03, T=1500 — same ages as the regression file
_AGES_SINGLE = np.array(
    [100.0, 200.0, 300.0, 450.0, 600.0, 750.0, 900.0, 1050.0, 1200.0, 1350.0]
)
_TIMES_SINGLE = 1500.0

# Two-epoch: λ=[0.05,0.01], μ=[0.03,0.02], T=[750,1500]
_AGES_MULTI = np.array(
    [80.0, 200.0, 350.0, 500.0, 650.0, 720.0, 900.0, 1050.0, 1200.0, 1400.0]
)
_TIMES_MULTI = np.array([750.0, 1500.0], dtype=np.float64)

# ---------------------------------------------------------------------------
# Weak priors — data should dominate so posterior mean ≈ MLE
# ---------------------------------------------------------------------------
_WEAK_PRIOR = BayesianPriorConfig(loc=-6.0, scale=4.0)

# Minimal NUTS config for the test (short chains, still meaningful)
_MCMC_CFG = BayesianMCMCConfig(
    num_samples=400,
    warmup_steps=200,
    num_chains=1,
    target_accept_prob=0.8,
    max_tree_depth=8,
    disable_progressbar=True,
)

# Generous tolerance for posterior mean vs MLE (relative)
_RTOL = 0.20


def _compare(ages: np.ndarray, times: np.ndarray, epoch_end_times: list[float]) -> None:
    """Assert posterior means are within _RTOL of the MLE for every parameter."""
    data = AgeData.from_groups([ages], float(times[-1]))
    model = BirthDeathModel.piecewise_constant(epoch_end_times)
    mle = fit(data, model, seed=0, compute_covariance=False)
    assert mle.converged, "MLE baseline fit failed"
    bay = fit_bayesian(
        data, model, prior=_WEAK_PRIOR, mcmc=_MCMC_CFG, init=mle, rng_seed=42
    )
    for name in model.names:
        rel = abs(bay.params[name] - mle.params[name]) / max(mle.params[name], 1e-12)
        assert rel < _RTOL, (
            f"{name}: posterior mean {bay.params[name]:.5f} vs MLE "
            f"{mle.params[name]:.5f} (rel diff {rel:.3f} > {_RTOL})"
        )


@pytest.mark.slow
class TestBayesianMleConsistency:
    """Posterior mean under weak priors should agree with MLE to within _RTOL."""

    def test_single_epoch_posterior_near_mle(self) -> None:
        """Single-epoch Bayesian fit agrees with MLE birth and death rates."""
        _compare(_AGES_SINGLE, np.array([_TIMES_SINGLE]), [_TIMES_SINGLE])

    def test_multi_epoch_posterior_near_mle(self) -> None:
        """Two-epoch Bayesian fit agrees with MLE birth and death rates."""
        _compare(_AGES_MULTI, _TIMES_MULTI, list(_TIMES_MULTI))
