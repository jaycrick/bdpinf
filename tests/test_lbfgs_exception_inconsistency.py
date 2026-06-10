"""Tests for absence of spurious 'no improvement' exceptions from LBFGS."""

import time

import numpy as np
import torch

from inf_bp_texts.inference import ConvergenceConfig
from inf_bp_texts.inference.fitting import fit_birth_death_processes
from inf_bp_texts.simulation import simulate_bdp_ages_from_distributions


class TestLbfgsExceptionInconsistency:
    """Test that LBFGS does not raise false 'no improvement' exceptions."""

    def test_lbfgs_no_false_no_improvement_exception(self) -> None:
        # Simulate a dataset similar to the QMD setup but smaller for test speed
        birth_rate = 0.005
        death_rate = birth_rate - 0.004
        time_span = 1500.0
        seed = 42

        rng = np.random.default_rng(seed)
        sampled = simulate_bdp_ages_from_distributions(
            birth_rate=torch.tensor(birth_rate, dtype=torch.float64),
            death_rate=torch.tensor(death_rate, dtype=torch.float64),
            max_time=torch.tensor(time_span, dtype=torch.float64),
            rng=rng,
        )
        assert sampled.spine_ages is not None and sampled.non_spine_ages is not None
        ages = np.asarray(
            sampled.spine_ages.tolist() + sampled.non_spine_ages.tolist(),
            dtype=np.float64,
        )

        start = time.perf_counter()
        res = fit_birth_death_processes(
            list_of_ages=[torch.as_tensor(ages, dtype=torch.float64)],
            times=float(time_span),
            is_ordered=True,
            optimizers=["lbfgs"],
            optimizer_kwargs={"lbfgs": {"lr": 1.0, "record_every": 1}},
            perturb_initial_params_std=0.0,
            random_seed=seed,
            convergence_config=ConvergenceConfig(
                gradient_threshold=5e-4,
                absolute_tolerance=1e-7,
                patience=5,
            ),
            estimate_uncertainty=False,
        )
        elapsed = time.perf_counter() - start

        attempts = res.get("optimization_result", {}).get("attempts", [])
        assert attempts, "Expected at least one optimizer attempt"
        att = attempts[0]

        first_nll = att.get("first_nll")
        best_nll = att.get("best_nll")
        exception_msg = att.get("exception")

        # If the recorded best is strictly better than the first, we must not
        # report the "did not improve from initial value" error.
        improved = (
            first_nll is not None
            and best_nll is not None
            and best_nll < first_nll - 1e-12
        )
        if improved:
            assert (
                exception_msg is None
                or "did not improve from initial value" not in exception_msg
            ), (
                f"False 'no improvement' exception reported: {exception_msg} "
                f"(run {elapsed:.2f}s)"
            )
