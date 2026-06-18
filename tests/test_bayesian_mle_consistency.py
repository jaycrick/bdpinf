"""Sanity-baseline: posterior mean should land near the MLE estimate under weak priors.

These tests guard against the Bayesian path silently diverging from the validated
MLE path.  They run NUTS with real (short) chains, so they are marked slow and
excluded from the default fast suite.

Run explicitly with:
    uv run pytest tests/test_bayesian_mle_consistency.py -m slow -v
"""

import numpy as np
import pytest

from inf_bp_texts.inference import (
    fit_birth_death_processes,
    fit_birth_death_processes_bayesian,
)
from inf_bp_texts.inference.bayesian import (
    BayesianMCMCConfig,
    BayesianModelConfig,
    BayesianPriorConfig,
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


@pytest.mark.slow
class TestBayesianMleConsistency:
    """Posterior mean under weak priors should agree with MLE to within _RTOL."""

    def test_single_epoch_posterior_near_mle(self) -> None:
        """Single-epoch Bayesian fit agrees with MLE birth and death rates."""
        mle = fit_birth_death_processes(
            list_of_ages=[_AGES_SINGLE],
            times=_TIMES_SINGLE,
            random_seed=0,
        )
        assert mle["success"], "MLE baseline fit failed"
        mle_birth = float(mle["birth_rates"][0])
        mle_death = float(mle["death_rates"][0])

        bay = fit_birth_death_processes_bayesian(
            list_of_ages=[_AGES_SINGLE],
            times=_TIMES_SINGLE,
            model_config=BayesianModelConfig(rng_seed=42),
            mcmc_config=_MCMC_CFG,
            prior_config=_WEAK_PRIOR,
        )
        assert bay["success"]
        bay_birth = float(bay["birth_rates"][0])
        bay_death = float(bay["death_rates"][0])

        assert abs(bay_birth - mle_birth) / mle_birth < _RTOL, (
            f"Birth: posterior mean {bay_birth:.5f} vs MLE {mle_birth:.5f} "
            f"(rel diff {abs(bay_birth - mle_birth) / mle_birth:.3f} > {_RTOL})"
        )
        assert abs(bay_death - mle_death) / mle_death < _RTOL, (
            f"Death: posterior mean {bay_death:.5f} vs MLE {mle_death:.5f} "
            f"(rel diff {abs(bay_death - mle_death) / mle_death:.3f} > {_RTOL})"
        )

    def test_multi_epoch_posterior_near_mle(self) -> None:
        """Two-epoch Bayesian fit agrees with MLE birth and death rates."""
        mle = fit_birth_death_processes(
            list_of_ages=[_AGES_MULTI],
            times=_TIMES_MULTI,
            random_seed=0,
        )
        assert mle["success"], "MLE baseline multi-epoch fit failed"
        mle_birth = mle["birth_rates"]
        mle_death = mle["death_rates"]

        bay = fit_birth_death_processes_bayesian(
            list_of_ages=[_AGES_MULTI],
            times=_TIMES_MULTI,
            model_config=BayesianModelConfig(rng_seed=42),
            mcmc_config=_MCMC_CFG,
            prior_config=_WEAK_PRIOR,
        )
        assert bay["success"]
        bay_birth = bay["birth_rates"]
        bay_death = bay["death_rates"]

        for epoch in range(2):
            rel_b = abs(bay_birth[epoch] - mle_birth[epoch]) / max(
                mle_birth[epoch], 1e-12
            )
            rel_d = abs(bay_death[epoch] - mle_death[epoch]) / max(
                mle_death[epoch], 1e-12
            )
            assert rel_b < _RTOL, (
                f"Epoch {epoch} birth: posterior {bay_birth[epoch]:.5f} "
                f"vs MLE {mle_birth[epoch]:.5f} (rel diff {rel_b:.3f})"
            )
            assert rel_d < _RTOL, (
                f"Epoch {epoch} death: posterior {bay_death[epoch]:.5f} "
                f"vs MLE {mle_death[epoch]:.5f} (rel diff {rel_d:.3f})"
            )
