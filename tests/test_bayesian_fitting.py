"""Tests for Bayesian birth-death inference utilities."""

import numpy as np
import pytest

from bdpinf.inference import (
    AgeData,
    BayesianMCMCConfig,
    BayesianPriorConfig,
    BirthDeathModel,
    Constraints,
    FreeKnots,
    fit,
    fit_bayesian,
    sample_prior,
)

_MCMC = BayesianMCMCConfig(
    num_samples=20,
    warmup_steps=20,
    num_chains=1,
    target_accept_prob=0.8,
    disable_progressbar=True,
)
_AGES = np.array([50.0, 120.0, 180.0, 220.0], dtype=np.float64)


class TestPriorConfig:
    """Test prior resolution against a model."""

    def test_scalar_fallbacks_and_family_overrides(self) -> None:
        model = BirthDeathModel.piecewise_constant([1.0, 2.0])
        locs, scales = BayesianPriorConfig(
            loc=-6.0, scale=2.0, death_scale=1.5
        ).resolve(model)
        np.testing.assert_allclose(locs, [-6.0] * 4)
        np.testing.assert_allclose(scales, [2.0, 2.0, 1.5, 1.5])

    def test_sequence_and_by_name_overrides(self) -> None:
        model = BirthDeathModel.piecewise_constant([1.0, 2.0])
        locs, _ = BayesianPriorConfig(
            birth_loc=[-5.0, -7.0], by_name={"death.v1": (-1.0, 0.5)}
        ).resolve(model)
        np.testing.assert_allclose(locs, [-5.0, -7.0, -6.0, -1.0])

    def test_resolve_tensors_matches_piecewise_constant_resolve(self) -> None:
        config = BayesianPriorConfig(loc=-6.0, birth_loc=[-5.0, -7.0], death_scale=1.5)
        locs, scales = config.resolve_tensors(2)
        expected = config.resolve(BirthDeathModel.piecewise_constant([1.0, 2.0]))
        np.testing.assert_allclose(locs, expected[0])
        np.testing.assert_allclose(scales, expected[1])
        assert locs.shape == (4,)

    def test_rejects_wrong_length_and_unknown_name(self) -> None:
        model = BirthDeathModel.piecewise_constant([1.0, 2.0])
        with pytest.raises(ValueError, match="does not match"):
            BayesianPriorConfig(birth_loc=[-5.0]).resolve(model)
        with pytest.raises(ValueError, match="Unknown parameter"):
            BayesianPriorConfig(by_name={"nope": (0.0, 1.0)}).resolve(model)


class TestExponentialPrior:
    """Test the Exponential prior on natural-scale rates."""

    def test_validation(self) -> None:
        model = BirthDeathModel.piecewise_constant([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown parameter"):
            BayesianPriorConfig(exponential={"nope": 0.1}).resolve(model)
        with pytest.raises(ValueError, match="positive"):
            BayesianPriorConfig(exponential={"birth.v0": 0.0}).resolve(model)
        knotted = BirthDeathModel.piecewise_constant(FreeKnots(1, 10.0))
        with pytest.raises(ValueError, match="log-transform"):
            BayesianPriorConfig(exponential={"tau0": 0.1}).resolve(knotted)
        compiled = Constraints(fixed={"birth.v1": 0.1}).compile(model.params)
        with pytest.raises(ValueError, match="fixed"):
            BayesianPriorConfig(exponential={"birth.v1": 0.1}).resolve_exponential(
                model, compiled
            )

    def test_slots_follow_equality_groups(self) -> None:
        model = BirthDeathModel.piecewise_constant([1.0, 2.0])
        compiled = Constraints(equal=[["birth.v0", "birth.v1"]]).compile(model.params)
        config = BayesianPriorConfig(exponential={"birth.v1": 0.2})
        slots, means = config.resolve_exponential(model, compiled)
        assert slots.tolist() == [0]
        assert means.tolist() == [0.2]
        clash = BayesianPriorConfig(exponential={"birth.v0": 0.2, "birth.v1": 0.3})
        with pytest.raises(ValueError, match="different"):
            clash.resolve_exponential(model, compiled)

    def test_sample_prior_mean_and_other_priors(self) -> None:
        model = BirthDeathModel.piecewise_constant([1.0, 2.0])
        draws = sample_prior(
            model,
            BayesianPriorConfig(exponential={"birth.v1": 0.01}, loc=-3.0, scale=0.5),
            20000,
            seed=0,
        )
        assert set(draws) == set(model.names)
        assert draws["birth.v1"].mean() == pytest.approx(0.01, rel=0.05)
        assert np.std(np.log(draws["birth.v0"])) == pytest.approx(0.5, rel=0.05)

    def test_sample_prior_honours_constraints(self) -> None:
        model = BirthDeathModel.piecewise_constant([1.0, 2.0])
        draws = sample_prior(
            model,
            BayesianPriorConfig(exponential={"death.v1": 0.5}),
            50,
            constraints=Constraints(
                fixed={"birth.v0": 0.1}, equal=[["death.v0", "death.v1"]]
            ),
            seed=1,
        )
        assert np.all(draws["birth.v0"] == 0.1)
        np.testing.assert_array_equal(draws["death.v0"], draws["death.v1"])

    def test_posterior_runs_and_follows_the_prior_without_data_pull(self) -> None:
        data = AgeData.from_groups([_AGES], 300.0)
        model = BirthDeathModel.piecewise_constant([150.0, 300.0])
        config = BayesianPriorConfig(exponential={"birth.v1": 0.002})
        result = fit_bayesian(
            data,
            model,
            prior=config,
            mcmc=BayesianMCMCConfig(
                num_samples=300,
                warmup_steps=200,
                num_chains=1,
                disable_progressbar=True,
            ),
            rng_seed=2,
        )
        assert np.all(np.isfinite(result.samples))
        assert result.samples[:, 1].mean() < 0.05

    def test_exponential_log_density_includes_jacobian(self) -> None:
        # One free log-rate with an Exponential prior and no data: the draws
        # of the rate should have mean close to the prior mean.
        data = AgeData.from_groups([np.array([299.0])], 300.0)
        model = BirthDeathModel.piecewise_constant([300.0])
        result = fit_bayesian(
            data,
            model,
            constraints=Constraints(fixed={"death.v0": 0.01}),
            prior=BayesianPriorConfig(exponential={"birth.v0": 0.01}),
            mcmc=BayesianMCMCConfig(
                num_samples=400,
                warmup_steps=200,
                num_chains=1,
                disable_progressbar=True,
            ),
            rng_seed=5,
        )
        assert np.all(result.samples[:, 0] > 0)
        assert result.samples[:, 0].mean() < 0.1


class TestFitBayesian:
    """Test the sampler wrapper."""

    def test_result_structure(self) -> None:
        data = AgeData.from_groups([_AGES], 300.0)
        model = BirthDeathModel.piecewise_constant([300.0])
        result = fit_bayesian(data, model, mcmc=_MCMC, rng_seed=7)
        assert result.samples_by_chain.shape == (1, 20, 2)
        assert result.samples.shape == (20, 2)
        assert result.birth_samples.shape == (20, 1)
        assert result.death_samples.shape == (20, 1)
        assert set(result.params) == {"birth.v0", "death.v0"}
        assert set(result.posterior_samples) == {"birth_rates", "death_rates"}
        lo, hi = result.credible_intervals(0.9)["birth.v0"]
        assert lo <= result.params["birth.v0"] <= hi
        assert result.num_divergences >= 0
        assert result.mcmc_config["num_chains"] == 1
        assert np.all(result.samples > 0)

    def test_constraints_are_applied_while_sampling(self) -> None:
        data = AgeData.from_groups([_AGES], 300.0)
        model = BirthDeathModel.piecewise_constant([150.0, 300.0])
        constraints = Constraints(
            fixed={"birth.v1": 0.01}, equal=[["death.v0", "death.v1"]]
        )
        result = fit_bayesian(
            data, model, constraints=constraints, mcmc=_MCMC, rng_seed=7
        )
        assert result.free_by_chain.shape[-1] == 2
        assert np.all(result.samples[:, 1] == 0.01)
        np.testing.assert_allclose(result.samples[:, 2], result.samples[:, 3])
        assert set(result.summary) == {"birth.v0", "death.v0", "death.v1"}

    def test_to_dict_has_persistence_layout(self) -> None:
        # Every age exceeds 150, so all births are in epoch 1 and the fixed zero
        # second-epoch birth rate is consistent with the data.
        data = AgeData.from_groups([np.array([160.0, 190.0, 230.0, 280.0])], 300.0)
        model = BirthDeathModel.piecewise_constant([150.0, 300.0])
        constraints = Constraints(fixed={"birth.v1": 0.0})
        payload = fit_bayesian(
            data, model, constraints=constraints, mcmc=_MCMC, rng_seed=3
        ).to_dict()
        assert payload["success"] is True
        assert payload["n_epochs"] == 2
        assert payload["birth_rates"].shape == (2,)
        assert payload["birth_rates_90_ci"].shape == (2, 2)
        assert payload["posterior_samples"]["birth_rates"].shape == (20, 2)
        assert payload["posterior_samples_by_chain"]["death_rates"].shape == (1, 20, 2)
        log_birth = payload["posterior_log_samples"]["birth_rates"]
        assert np.all(np.isfinite(log_birth))  # the fixed zero is floored, not -inf
        assert payload["diagnostics"]["death_rates"]["r_hat"].shape == (2,)
        assert payload["num_divergences"] >= 0

    def test_starts_from_mle_result(self) -> None:
        data = AgeData.from_groups([_AGES], 300.0)
        model = BirthDeathModel.piecewise_constant([300.0])
        mle = fit(data, model, compute_covariance=False)
        result = fit_bayesian(data, model, mcmc=_MCMC, init=mle, rng_seed=1)
        assert result.free_by_chain.shape == (1, 20, 2)

    def test_rejects_mismatched_window_and_unknown_spine_with_many_founders(
        self,
    ) -> None:
        data = AgeData.from_groups([_AGES], 300.0)
        with pytest.raises(ValueError, match="does not match"):
            fit_bayesian(data, BirthDeathModel.piecewise_constant([100.0]), mcmc=_MCMC)
        many = AgeData.from_groups([_AGES], 300.0, initial_population=2)
        with pytest.raises(ValueError, match="Unknown-spine marginal"):
            fit_bayesian(
                many,
                BirthDeathModel.piecewise_constant([300.0]),
                mcmc=_MCMC,
                condition_on_survival=False,
            )
