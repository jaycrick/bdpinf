"""Test maximum likelihood fitting of birth-death rate models."""

import json
import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from bdsurv import PiecewiseConstant

from bdpinf.analysis.multi_epoch_recovery import simulate_zero_birth_second_epoch
from bdpinf.inference import (
    AgeData,
    BirthDeathModel,
    Constraints,
    FitResult,
    fit,
    get_crude_parameter_estimates,
    moment_initial_rates,
    profile_likelihood_interval,
)
from bdpinf.inference.likelihood import log_likelihood_rates
from bdpinf.inference.rate_models import FreeKnots
from bdpinf.models.rate_functions import PiecewiseLinearRate
from bdpinf.simulation import simulate_birth_death


def _simulate(
    birth: list[float], death: list[float], times: list[float], seed: int
) -> np.ndarray:
    process = simulate_birth_death(
        PiecewiseConstant(np.array(birth), np.array(times)),
        PiecewiseConstant(np.array(death), np.array(times)),
        random_seed=seed,
        ensure_non_extinct=True,
    )
    return np.asarray(process.ages)


@pytest.fixture(scope="module")
def constant_case() -> tuple[AgeData, BirthDeathModel, FitResult]:
    """Return a single-epoch supercritical dataset and its fit."""
    ages = _simulate([1.0], [0.6], [9.0], seed=2)
    data = AgeData.from_groups([ages], 9.0)
    model = BirthDeathModel.piecewise_constant([9.0])
    return data, model, fit(data, model)


_FIT_PINS: dict[str, dict[str, Any]] = {
    "default": {
        "ll": -23.017548163371135,
        "params": {"birth.v0": 0.81027066688596, "death.v0": 0.6448728043452677},
        "converged": True,
        "interior": True,
        "min_eig": 4.8127539344249985,
        "score_norm": 2.654972823488031e-09,
        "n_iter": 3,
        "n_evals": 5,
        "method": "trust-exact",
        "message": "Optimization terminated successfully.",
        "starts": [[-23.017548163371135, True, 3, False]],
        "cov_free": [
            [0.062378887360233705, 0.07157296139796013],
            [0.07157296139796013, 0.17255011772961779],
        ],
        "cov": [
            [0.04095414448365189, 0.03739841238255393],
            [0.03739841238255394, 0.07175685308360041],
        ],
        "cov_unc": [
            [0.062378887360233705, 0.07157296139796013],
            [0.07157296139796013, 0.17255011772961779],
        ],
        "free_values": [-0.21038693048421825, -0.43870218419882356],
        "history": [23.017568551168765, 23.017548163814432, 23.017548163371135],
    },
    "thorough": {
        "ll": -23.01754816337113,
        "params": {"birth.v0": 0.8102706675512862, "death.v0": 0.6448728070637209},
        "converged": True,
        "interior": True,
        "min_eig": 4.812753985198691,
        "score_norm": 4.8646232447885465e-08,
        "n_iter": 3,
        "n_evals": 5,
        "method": "trust-exact",
        "message": "Optimization terminated successfully.",
        "starts": [
            [-23.017548163371135, True, 3, False],
            [-23.017548163371302, True, 4, True],
            [-23.01754816337113, True, 3, False],
            [-23.017548163371135, True, 4, False],
            [-23.01754816337114, True, 4, False],
        ],
        "cov_free": None,
        "cov": None,
        "cov_unc": None,
        "free_values": [-0.21038692966310216, -0.43870217998333627],
        "history": [23.017652205826437, 23.01754817148913, 23.01754816337113],
    },
    "multistart_fixed": {
        "ll": -49.89731547355578,
        "params": {
            "birth.v0": 0.6931465102982493,
            "birth.v1": 0.5963609629313341,
            "death.v0": 0.40574463468893446,
            "death.v1": 0.4,
        },
        "converged": True,
        "interior": False,
        "min_eig": -3.9255863830412537e-08,
        "score_norm": 1.7599746129134592e-07,
        "n_iter": 7,
        "n_evals": 9,
        "method": "trust-exact",
        "message": "Optimization terminated successfully.",
        "starts": [
            [-49.89731547355584, True, 5, False],
            [-49.89731547355578, True, 7, True],
            [-49.897315473557995, True, 6, False],
        ],
        "cov_free": [
            [142703.43244671938, -0.02186368018458719, 358008.1901878546],
            [-0.02186368018458719, 0.03336991809818176, 0.010087348473471135],
            [358008.1901878546, 0.010087348473471135, 898156.5730387102],
        ],
        "cov": [
            [68562.16161838615, -0.009037691535917127, 100686.39439560441, 0.0],
            [-0.009037691535917127, 0.011867891176790255, 0.0024408383434010483, 0.0],
            [100686.39439560441, 0.0024408383434010488, 147862.35672097458, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "cov_unc": [
            [142703.43244671938, -0.02186368018458719, 358008.1901878546, 0.0],
            [-0.02186368018458719, 0.03336991809818176, 0.010087348473471135, 0.0],
            [358008.1901878546, 0.010087348473471135, 898156.5730387102, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "free_values": [-0.36651388756535686, -0.5169091527419523, -0.9020312958691102],
        "history": [
            49.901432177459434,
            49.89732706982633,
            49.89732706982633,
            49.89732706982633,
            49.89732706982633,
            49.8973155338163,
            49.89731547355578,
        ],
    },
    "no_free": {
        "ll": -4.914391802244214,
        "params": {"birth.v0": 1.0, "death.v0": 0.5},
        "converged": True,
        "interior": True,
        "min_eig": None,
        "score_norm": 0.0,
        "n_iter": 0,
        "n_evals": 1,
        "method": "trust-exact",
        "message": "no free parameters",
        "starts": [[-4.914391802244214, True, 0, False]],
        "cov_free": None,
        "cov": None,
        "cov_unc": None,
        "free_values": [],
        "history": [4.914391802244214],
    },
    "lbfgs_nocov": {
        "ll": -23.017548163373892,
        "params": {"birth.v0": 0.8102702002206665, "death.v0": 0.6448722933997058},
        "converged": True,
        "interior": True,
        "min_eig": 4.8127485638687615,
        "score_norm": 7.704142504430446e-06,
        "n_iter": 7,
        "n_evals": 10,
        "method": "L-BFGS-B",
        "message": "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
        "starts": [[-23.017548163373892, True, 7, False]],
        "cov_free": None,
        "cov": None,
        "cov_unc": None,
        "free_values": [-0.21038750642192267, -0.43870297651904683],
        "history": [
            23.01931889748044,
            23.018420594547212,
            23.01812586564719,
            23.017613291182794,
            23.01754838458993,
            23.017548163726108,
            23.017548163373892,
        ],
    },
    "equal_group": {
        "ll": -49.88290661027099,
        "params": {
            "birth.v0": 3.7023945722508163e-06,
            "birth.v1": 0.5965110610396112,
            "death.v0": 0.11936382591850483,
            "death.v1": 0.11936382591850483,
        },
        "converged": True,
        "interior": False,
        "min_eig": 8.171519901072513e-08,
        "score_norm": 2.377434835779868e-05,
        "n_iter": 24,
        "n_evals": 26,
        "method": "trust-exact",
        "message": "Optimization terminated successfully.",
        "starts": [[-49.88290661027099, True, 24, False]],
        "cov_free": [
            [12237625.460208394, -0.04897002054750027, 122.59288095475009],
            [-0.04897002054750027, 0.03334816799129816, 0.04632029549144864],
            [122.59288095475009, 0.04632029549144864, 0.7812620828945808],
        ],
        "cov": [
            [
                0.00016775001142024427,
                -1.0815123621946175e-07,
                5.4177714761809845e-05,
                5.4177714761809845e-05,
            ],
            [
                -1.0815123621946176e-07,
                0.011866126746872497,
                0.003298090381745216,
                0.003298090381745216,
            ],
            [
                5.4177714761809845e-05,
                0.0032980903817452167,
                0.011131205698971091,
                0.011131205698971091,
            ],
            [
                5.4177714761809845e-05,
                0.0032980903817452167,
                0.011131205698971091,
                0.011131205698971091,
            ],
        ],
        "cov_unc": [
            [
                12237625.460208394,
                -0.04897002054750027,
                122.59288095475009,
                122.59288095475009,
            ],
            [
                -0.04897002054750027,
                0.03334816799129816,
                0.04632029549144864,
                0.04632029549144864,
            ],
            [
                122.59288095475009,
                0.04632029549144864,
                0.7812620828945808,
                0.7812620828945808,
            ],
            [
                122.59288095475009,
                0.04632029549144864,
                0.7812620828945808,
                0.7812620828945808,
            ],
        ],
        "free_values": [-12.506530765956443, -0.5166574943812011, -2.1255790894336157],
        "history_len": 24,
        "history_last": 49.88290661027099,
    },
    "free_knot": {
        "ll": -49.88468009558938,
        "params": {
            "birth.v0": 0.2550270377941608,
            "birth.v1": 0.5965943697204217,
            "death.v0": 0.6425201319372869,
            "death.v1": 0.14635325547221967,
            "tau0": 5.300045851050425,
        },
        "converged": False,
        "interior": False,
        "min_eig": -0.0003748154583679024,
        "score_norm": 0.006811803494154657,
        "n_iter": 88,
        "n_evals": 110,
        "method": "trust-exact",
        "message": "A bad approximation caused failure to predict improvement.",
        "starts": [
            [-49.88581873322025, False, 97, False],
            [-49.88468009558938, False, 88, True],
        ],
        "cov_free": [
            [
                -626.0093086683124,
                -0.06802933685888421,
                -1344.0992297163423,
                123.53159420266965,
                43.95937704857019,
            ],
            [
                -0.06802933685888421,
                0.03349391595338913,
                -0.09553224526492893,
                -0.15887015470625482,
                0.22195328923966345,
            ],
            [
                -1344.0992297163423,
                -0.09553224526492893,
                -1765.9002832980543,
                132.46577305447386,
                -122.66787821978197,
            ],
            [
                123.53159420266965,
                -0.15887015470625482,
                132.46577305447386,
                269.4589187255926,
                -287.91656366844234,
            ],
            [
                43.95937704857019,
                0.22195328923966345,
                -122.66787821978197,
                -287.91656366844234,
                374.2393287382681,
            ],
        ],
        "cov": [
            [
                -40.71488796831987,
                -0.01035050678691829,
                -220.24410780704338,
                4.610697819200819,
                27.926145968469942,
            ],
            [
                -0.01035050678691829,
                0.011921316743072004,
                -0.03661979217591273,
                -0.013871513733874234,
                0.32984809769613965,
            ],
            [
                -220.2441078070434,
                -0.03661979217591272,
                -729.0203575648972,
                12.456407446703382,
                -196.3318871679668,
            ],
            [
                4.610697819200819,
                -0.013871513733874234,
                12.456407446703384,
                5.7716147857520825,
                -104.96446232370309,
            ],
            [
                27.926145968469946,
                0.32984809769613965,
                -196.3318871679668,
                -104.96446232370307,
                2322.180218569553,
            ],
        ],
        "cov_unc": [
            [
                -626.0093086683124,
                -0.06802933685888421,
                -1344.0992297163423,
                123.53159420266965,
                43.95937704857019,
            ],
            [
                -0.06802933685888421,
                0.03349391595338913,
                -0.09553224526492893,
                -0.15887015470625482,
                0.22195328923966345,
            ],
            [
                -1344.0992297163423,
                -0.09553224526492893,
                -1765.9002832980543,
                132.46577305447386,
                -122.66787821978197,
            ],
            [
                123.53159420266965,
                -0.15887015470625482,
                132.46577305447386,
                269.4589187255926,
                -287.91656366844234,
            ],
            [
                43.95937704857019,
                0.22195328923966345,
                -122.66787821978197,
                -287.91656366844234,
                374.2393287382681,
            ],
        ],
        "free_values": [
            -1.3663857088792175,
            -0.5165178442232217,
            -0.44235712894912194,
            -1.9217320216624032,
            0.12016271853646106,
        ],
        "history_len": 89,
        "history_last": 49.88468009558938,
    },
    "unconverged": {
        "ll": -23.017548163374517,
        "params": {"birth.v0": 0.8102711338763855, "death.v0": 0.6448729998700735},
        "converged": False,
        "interior": False,
        "min_eig": 4.812752466848205,
        "score_norm": 1.4338318344958063e-05,
        "n_iter": 12,
        "n_evals": 18,
        "method": "trust-exact",
        "message": "Maximum number of iterations has been exceeded.",
        "starts": [[-23.017548163374517, False, 12, False]],
        "cov_free": [
            [0.06237885139544057, 0.07157295998248837],
            [0.07157295998248837, 0.17255020654932118],
        ],
        "cov": [
            [0.04095416807829015, 0.03739844453626375],
            [0.03739844453626376, 0.07175693353349004],
        ],
        "cov_unc": [
            [0.06237885139544057, 0.07157295998248837],
            [0.07157295998248837, 0.17255020654932118],
        ],
        "free_values": [-0.21038635414558224, -0.4387018809998442],
        "history_len": 12,
        "history_last": 23.017548163374517,
    },
}


def _pin_case(name: str) -> tuple[AgeData, BirthDeathModel, dict[str, Any]]:
    """Return the data, model and ``fit`` keywords of a recorded configuration."""
    one = _simulate([1.0], [0.6], [9.0], seed=2)
    two = _simulate([0.8, 0.6], [0.3, 0.4], [5.0, 10.0], seed=2)
    data1 = AgeData.from_groups([one], 9.0)
    model1 = BirthDeathModel.piecewise_constant([9.0])
    data2 = AgeData.from_groups([two], 10.0)
    model2 = BirthDeathModel.piecewise_constant([5.0, 10.0])
    cases: dict[str, tuple[AgeData, BirthDeathModel, dict[str, Any]]] = {
        "default": (data1, model1, {}),
        "thorough": (
            data1,
            model1,
            {"thorough": True, "seed": 0, "compute_covariance": False},
        ),
        "multistart_fixed": (
            data2,
            model2,
            {
                "n_starts": 3,
                "seed": 7,
                "constraints": Constraints(fixed={"death.v1": 0.4}),
            },
        ),
        "no_free": (
            AgeData.from_groups([jnp.array([0.3, 0.9, 1.5])], 2.0),
            BirthDeathModel.piecewise_constant([2.0]),
            {"constraints": Constraints(fixed={"birth.v0": 1.0, "death.v0": 0.5})},
        ),
        "lbfgs_nocov": (
            data1,
            model1,
            {"method": "L-BFGS-B", "compute_covariance": False},
        ),
        "equal_group": (
            data2,
            model2,
            {
                "constraints": Constraints(equal=[["death.v0", "death.v1"]]),
                "pad_to_bucket": False,
            },
        ),
        "free_knot": (
            data2,
            BirthDeathModel.piecewise_constant(FreeKnots(1, 10.0)),
            {"n_starts": 2, "seed": 1},
        ),
        "unconverged": (
            data1,
            model1,
            {"init": [5.0, 4.0], "max_iter": 1, "tol": 1e-14},
        ),
    }
    return cases[name]


class TestFit:
    """Test the single-start piecewise-constant fit."""

    def test_converges_to_interior_optimum(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, _, result = constant_case
        assert result.converged
        assert result.success
        assert result.at_interior_optimum
        assert result.score_norm < 1e-4
        assert result.min_hessian_eig is not None
        assert result.min_hessian_eig > 0
        assert result.method == "trust-exact"
        assert result.starts[0].converged

    def test_log_likelihood_matches_direct_evaluation(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        data, _, result = constant_case
        np.testing.assert_allclose(
            result.log_likelihood,
            float(log_likelihood_rates(data, result.birth, result.death)),
            rtol=1e-10,
        )

    def test_is_a_local_maximum(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        data, model, result = constant_case
        best = result.log_likelihood
        for scale in (0.9, 1.1):
            for i in range(2):
                natural = jnp.asarray(result.natural).at[i].mul(scale)
                moved = float(log_likelihood_rates(data, *model.rates(natural)))
                assert moved <= best + 1e-9

    def test_matches_recorded_reference_fits(self) -> None:
        """Maximised log-likelihoods of the previous optimiser on seeded data."""
        cases = [
            ([0.8], [0.3], [8.0], 2, -38.11206893044985),
            ([1.0], [0.6], [9.0], 2, -23.017548163371135),
            ([0.8, 0.6], [0.3, 0.4], [5.0, 10.0], 2, -49.882908238505465),
        ]
        for birth, death, times, seed, reference in cases:
            ages = _simulate(birth, death, times, seed)
            result = fit(
                AgeData.from_groups([ages], times[-1]),
                BirthDeathModel.piecewise_constant(times),
                compute_covariance=False,
            )
            assert result.log_likelihood >= reference - 1e-4

    def test_result_views(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, model, result = constant_case
        assert set(result.params) == set(model.names)
        np.testing.assert_allclose(result.birth_values, [result.params["birth.v0"]])
        np.testing.assert_allclose(result.death_values, [result.params["death.v0"]])
        assert all(v > 0 for v in result.params.values())

    def test_to_dict_is_json_serialisable(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, _, result = constant_case
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["converged"] is True
        assert payload["param_names"] == ["birth.v0", "death.v0"]


class TestUncertainty:
    """Test covariance, standard errors and intervals."""

    def test_covariance_is_symmetric_positive_definite(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, _, result = constant_case
        assert result.covariance is not None
        np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(result.covariance) > 0)

    def test_standard_errors_and_intervals(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, _, result = constant_case
        errors = result.standard_errors
        intervals = result.confidence_intervals(0.95)
        assert errors is not None
        assert intervals is not None
        for name, (lo, hi) in intervals.items():
            assert 0 < lo < result.params[name] < hi
            assert errors[name] > 0
        assert (
            result.confidence_intervals(0.5)["birth.v0"][1] < intervals["birth.v0"][1]
        )  # pyright: ignore[reportOptionalSubscript]

    def test_covariance_can_be_skipped(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=3)
        result = fit(
            AgeData.from_groups([ages], 9.0),
            BirthDeathModel.piecewise_constant([9.0]),
            compute_covariance=False,
        )
        assert result.covariance is None
        assert result.standard_errors is None
        assert result.confidence_intervals() is None

    def test_profile_interval_brackets_estimate(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        data, _, result = constant_case
        lo, hi = profile_likelihood_interval(result, data, "birth.v0")
        assert lo < result.params["birth.v0"] < hi
        wald_lo, wald_hi = result.confidence_intervals()["birth.v0"]  # pyright: ignore[reportOptionalSubscript]
        assert wald_lo / 1.5 < lo < wald_hi
        assert wald_lo < hi < wald_hi * 1.5


class TestConstraints:
    """Test fitting under constraints."""

    def test_equal_and_fixed_parameters(self) -> None:
        times = np.array([5.0, 10.0])
        simulated = simulate_zero_birth_second_epoch(
            first_epoch_seed=4,
            death_time_seed=5,
            birth_rate_epoch_1=1.0,
            death_rate=0.3,
            times=times,
        )
        alive = simulated.loc[simulated["death_time"].isna()]
        ages = 10.0 - alive["creation_time"].to_numpy(dtype=np.float64)
        data = AgeData.from_groups([ages], 10.0, is_ordered=False)
        model = BirthDeathModel.piecewise_constant(times)
        constraints = Constraints(
            fixed={"birth.v1": 0.0}, equal=[["death.v0", "death.v1"]]
        )
        result = fit(data, model, constraints=constraints)
        assert result.converged
        assert result.params["birth.v1"] == 0.0
        assert result.params["death.v0"] == pytest.approx(result.params["death.v1"])
        assert result.compiled.n_free == 2
        assert result.standard_errors is not None
        assert result.standard_errors["birth.v1"] == 0.0
        assert 0.0 < result.params["death.v0"] < 1.0

    def test_all_parameters_fixed(self) -> None:
        data = AgeData.from_groups([jnp.array([0.3, 0.9, 1.5])], 2.0)
        model = BirthDeathModel.piecewise_constant([2.0])
        result = fit(
            data,
            model,
            constraints=Constraints(fixed={"birth.v0": 1.0, "death.v0": 0.5}),
        )
        assert result.converged
        assert result.compiled.n_free == 0
        assert result.params == {"birth.v0": 1.0, "death.v0": 0.5}
        np.testing.assert_allclose(
            result.log_likelihood,
            float(log_likelihood_rates(data, result.birth, result.death)),
        )


class TestNesting:
    """Test that piecewise constant is a special case of piecewise linear."""

    @pytest.fixture(scope="module")
    def constant_fit(self) -> tuple[AgeData, FitResult]:
        """Return a well-conditioned single-epoch dataset and its constant fit."""
        ages = _simulate([1.0], [0.6], [9.0], seed=1)
        data = AgeData.from_groups([ages], 9.0)
        return data, fit(
            data, BirthDeathModel.piecewise_constant([9.0]), compute_covariance=False
        )

    def test_flat_jump_model_reproduces_piecewise_constant_fit(
        self, constant_fit: tuple[AgeData, FitResult]
    ) -> None:
        data, pc = constant_fit
        pl = BirthDeathModel.piecewise_linear([9.0], continuous=False)
        constraints = Constraints(
            equal=pl.birth.flat_groups("birth.") + pl.death.flat_groups("death.")  # pyright: ignore[reportAttributeAccessIssue]
        )
        flat = fit(data, pl, constraints=constraints, compute_covariance=False)
        assert flat.converged
        np.testing.assert_allclose(flat.log_likelihood, pc.log_likelihood, atol=1e-6)
        np.testing.assert_allclose(
            flat.params["birth.start0"], pc.params["birth.v0"], rtol=1e-4
        )

    def test_free_piecewise_linear_never_fits_worse(
        self, constant_fit: tuple[AgeData, FitResult]
    ) -> None:
        data, pc = constant_fit
        jumps = fit(
            data,
            BirthDeathModel.piecewise_linear([9.0], continuous=False),
            init=pc,
            compute_covariance=False,
        )
        continuous = fit(
            data,
            BirthDeathModel.piecewise_linear([9.0]),
            init=pc,
            compute_covariance=False,
        )
        assert jumps.log_likelihood >= pc.log_likelihood - 1e-6
        assert continuous.log_likelihood >= pc.log_likelihood - 1e-6

    def test_quadrature_matches_exact_at_fitted_constant_rates(
        self, constant_fit: tuple[AgeData, FitResult]
    ) -> None:
        data, pc = constant_fit
        times = jnp.array([9.0])
        birth = PiecewiseLinearRate.from_constant(jnp.asarray(pc.birth_values), times)
        death = PiecewiseLinearRate.from_constant(jnp.asarray(pc.death_values), times)
        np.testing.assert_allclose(
            float(log_likelihood_rates(data, birth, death)),
            pc.log_likelihood,
            atol=1e-8,
        )


class TestInitialisation:
    """Test the forms of ``init`` and the starting heuristics."""

    @pytest.fixture(scope="module")
    def case(self) -> tuple[AgeData, BirthDeathModel]:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        return AgeData.from_groups([ages], 9.0), BirthDeathModel.piecewise_constant(
            [9.0]
        )

    def test_mapping_array_and_rate_function_inits(
        self, case: tuple[AgeData, BirthDeathModel]
    ) -> None:
        data, model = case
        reference = fit(data, model, compute_covariance=False)
        by_name = fit(data, model, init={"birth.v0": 2.0}, compute_covariance=False)
        by_array = fit(data, model, init=[0.5, 0.2], compute_covariance=False)
        by_rates = fit(
            data,
            model,
            init=(reference.birth, reference.death),
            compute_covariance=False,
        )
        for other in (by_name, by_array, by_rates):
            np.testing.assert_allclose(
                other.log_likelihood, reference.log_likelihood, atol=1e-6
            )
        assert by_rates.n_iter <= 1

    def test_bad_init_inputs(self, case: tuple[AgeData, BirthDeathModel]) -> None:
        data, model = case
        with pytest.raises(ValueError, match="Unknown parameter"):
            fit(data, model, init={"nope": 1.0})
        with pytest.raises(ValueError, match="init has shape"):
            fit(data, model, init=[1.0, 2.0, 3.0])
        impossible = Constraints(fixed={"birth.v0": 0.0})
        with pytest.raises(ValueError, match="finite likelihood"):
            fit(data, model, constraints=impossible)

    def test_moment_estimators(self, case: tuple[AgeData, BirthDeathModel]) -> None:
        data, _ = case
        rates = moment_initial_rates(data)
        assert rates is not None
        assert rates[0] > 0
        assert rates[1] > 0
        assert (
            moment_initial_rates(AgeData.from_groups([jnp.array([1.0])], 2.0)) is None
        )

    def test_moment_estimators_match_recording(self) -> None:
        def mk(
            ages: list[float], end: float, n: float | None = None, **kw: Any
        ) -> AgeData:
            sizes = None if n is None else [n]
            return AgeData.from_groups(
                [jnp.array(ages)], end, population_sizes=sizes, **kw
            )

        five = [1.0, 2.5, 4.0, 6.5, 8.0]
        recorded = [
            (mk(five, 9.0, 40.0), (0.22727272727272727, 0.011363636363636364)),
            (mk(five, 9.0, 5.0), (0.22727272727272727, 0.09182672824359434)),
            (mk([0.5, 1.0], 20.0, 1e9), (1.3333333333333333, 0.31042147913188)),
            (mk([2.0, 2.0, 2.0], 6.0, 4.0), (0.5, 0.5)),
            (mk([1.0, 2.0, 3.0], 300.0, 2.0), (0.5, 1.0)),
            (mk([1.0, 2.0, 3.0], 300.0, 1e6), (0.5, 0.46259053839469066)),
            (
                mk([100.0, 200.0], 1000.0, 2.0),
                (0.006666666666666667, 0.013324776195792948),
            ),
            (mk([1.0], 2.0), None),
            (mk(five, 9.0, 40.0, initial_population=2), None),
            (mk([100.0, 200.0], 1000.0, 1.0), None),
            (mk([0.0, 0.0], 1000.0, 5.0), None),
        ]
        for data, expected in recorded:
            got = moment_initial_rates(data)
            if expected is None:
                assert got is None
            else:
                assert got is not None
                np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_crude_estimates_validation(self) -> None:
        birth, death = get_crude_parameter_estimates(jnp.array([1.0, 2.0, 3.0]), 10.0)
        assert float(birth[0]) > 0
        assert float(death[0]) > 0
        with pytest.raises(ValueError, match="Cannot provide both"):
            get_crude_parameter_estimates(
                jnp.array([1.0]), 10.0, birth_rates=0.1, death_rates=0.1
            )
        with pytest.raises(ValueError, match="empty age data"):
            get_crude_parameter_estimates(jnp.array([]), 10.0)


class TestMultiStart:
    """Test restarts."""

    def test_second_start_is_swapped_and_best_is_kept(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        result = fit(data, model, n_starts=3, seed=0, compute_covariance=False)
        assert len(result.starts) == 3
        assert [s.swapped for s in result.starts] == [False, True, False]
        assert result.log_likelihood == pytest.approx(
            max(s.log_likelihood for s in result.starts)
        )

    def test_thorough_uses_five_starts_and_keeps_the_best(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        result = fit(data, model, thorough=True, seed=0, compute_covariance=False)
        assert len(result.starts) == 5
        assert result.log_likelihood == pytest.approx(
            max(s.log_likelihood for s in result.starts)
        )

    def test_seeded_restarts_are_reproducible(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        a = fit(data, model, n_starts=3, seed=5, compute_covariance=False)
        b = fit(data, model, n_starts=3, seed=5, compute_covariance=False)
        assert [s.log_likelihood for s in a.starts] == [
            s.log_likelihood for s in b.starts
        ]


class TestPaddingAndCompilation:
    """Test that bucket padding is exact and shares compiled code."""

    def test_padded_and_unpadded_fits_agree(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=1)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        padded = fit(data, model)
        plain = fit(data, model, pad_to_bucket=False)
        np.testing.assert_allclose(
            padded.log_likelihood, plain.log_likelihood, rtol=1e-10
        )
        for name in model.names:
            np.testing.assert_allclose(
                padded.params[name], plain.params[name], rtol=1e-5
            )
        assert padded.covariance is not None
        assert plain.covariance is not None
        np.testing.assert_allclose(padded.covariance, plain.covariance, rtol=1e-4)

    def test_objective_is_reused_across_sample_sizes_in_a_bucket(self) -> None:
        from bdpinf.inference import fitting

        model = BirthDeathModel.piecewise_constant([9.0])
        first = _simulate([1.0], [0.6], [9.0], seed=1)
        second = first[:150]
        assert fitting._bucket(len(first)) == fitting._bucket(len(second))
        fit(AgeData.from_groups([first], 9.0), model, compute_covariance=False)
        cached = len(fitting._OBJECTIVES)
        fit(AgeData.from_groups([second], 9.0), model, compute_covariance=False)
        assert len(fitting._OBJECTIVES) == cached

    @pytest.mark.parametrize(
        ("n", "expected"), [(1, 32), (32, 32), (33, 64), (100, 128), (1000, 1024)]
    )
    def test_bucket_sizes(self, n: int, expected: int) -> None:
        from bdpinf.inference import fitting

        assert fitting._bucket(n) == expected


class TestOptimiserChoice:
    """Test method selection and failure handling."""

    def test_lbfgs_agrees_with_newton(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        newton = fit(data, model, compute_covariance=False)
        lbfgs = fit(data, model, method="L-BFGS-B", compute_covariance=False)
        assert lbfgs.converged
        np.testing.assert_allclose(
            lbfgs.log_likelihood, newton.log_likelihood, atol=1e-6
        )

    def test_rejects_unknown_method_and_window_mismatch(self) -> None:
        data = AgeData.from_groups([jnp.array([0.5, 1.0])], 2.0)
        with pytest.raises(ValueError, match="Unsupported method"):
            fit(data, BirthDeathModel.piecewise_constant([2.0]), method="Adam")
        with pytest.raises(ValueError, match="does not match the data"):
            fit(data, BirthDeathModel.piecewise_constant([3.0]))

    def test_unconverged_fit_warns_about_covariance(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        with pytest.warns(UserWarning, match="did not converge"):
            result = fit(
                AgeData.from_groups([ages], 9.0),
                BirthDeathModel.piecewise_constant([9.0]),
                init=[5.0, 4.0],
                max_iter=1,
                tol=1e-14,
            )
        assert not result.converged


class TestDataFrame:
    """Test building data from a frame."""

    def test_age_column_detection(self) -> None:
        frame = pd.DataFrame(
            {"age": [100.0, 200.0, 300.0, 400.0], "other": [1, 2, 3, 4]}
        )
        data = AgeData.from_data_frame(frame, 1000.0)
        assert data.n_observations == 4
        result = fit(data, BirthDeathModel.piecewise_constant([1000.0]))
        assert result.params["birth.v0"] > 0

    def test_custom_and_missing_columns(self) -> None:
        frame = pd.DataFrame({"creation_age": [100.0, 200.0, 300.0]})
        assert (
            AgeData.from_data_frame(
                frame, 500.0, age_column="creation_age"
            ).n_observations
            == 3
        )
        with pytest.raises(ValueError, match="not found"):
            AgeData.from_data_frame(frame, 500.0)
        ambiguous = pd.DataFrame({"age_a": [1.0], "age_b": [2.0]})
        with pytest.raises(ValueError, match="Multiple age columns"):
            AgeData.from_data_frame(ambiguous, 500.0)


class TestRecordedFits:
    """Pin every field of ``FitResult`` against values recorded before the refactor."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("name", list(_FIT_PINS))
    def test_fit_result_matches_recording(self, name: str) -> None:
        data, model, kwargs = _pin_case(name)
        result = fit(data, model, **kwargs)
        pin = _FIT_PINS[name]
        tol = {"rtol": 1e-10, "atol": 1e-12}
        np.testing.assert_allclose(result.log_likelihood, pin["ll"], rtol=1e-10)
        assert list(result.params) == list(pin["params"])
        np.testing.assert_allclose(
            list(result.params.values()), list(pin["params"].values()), rtol=1e-10
        )
        assert result.converged is pin["converged"]
        assert result.at_interior_optimum is pin["interior"]
        if pin["min_eig"] is None:
            assert result.min_hessian_eig is None
        else:
            assert result.min_hessian_eig is not None
            np.testing.assert_allclose(result.min_hessian_eig, pin["min_eig"], **tol)
        np.testing.assert_allclose(result.score_norm, pin["score_norm"], **tol)
        assert result.n_iter == pin["n_iter"]
        assert result.n_function_evals == pin["n_evals"]
        assert result.method == pin["method"]
        assert result.message == pin["message"]
        assert len(result.starts) == len(pin["starts"])
        for got, (ll, converged, n_iter, swapped) in zip(
            result.starts, pin["starts"], strict=True
        ):
            np.testing.assert_allclose(got.log_likelihood, ll, rtol=1e-10)
            assert got.converged is converged
            assert got.n_iter == n_iter
            assert got.swapped is swapped
        for got_cov, key in (
            (result.covariance_free, "cov_free"),
            (result.covariance, "cov"),
            (result.covariance_unconstrained, "cov_unc"),
        ):
            if pin[key] is None:
                assert got_cov is None
            else:
                assert got_cov is not None
                np.testing.assert_allclose(got_cov, pin[key], **tol)
        np.testing.assert_allclose(result.free_values, pin["free_values"], **tol)
        if "history" in pin:
            np.testing.assert_allclose(result.nll_history, pin["history"], rtol=1e-10)
        else:
            assert len(result.nll_history) == pin["history_len"]
            np.testing.assert_allclose(
                result.nll_history[-1], pin["history_last"], rtol=1e-10
            )

    def test_error_messages(self) -> None:
        data = AgeData.from_groups([jnp.array([0.5, 1.0])], 2.0)
        model = BirthDeathModel.piecewise_constant([2.0])
        with pytest.raises(ValueError, match="Unsupported method") as unknown:
            fit(data, model, method="Adam")
        assert str(unknown.value) == (
            "Unsupported method 'Adam'. Choose from ['trust-exact', 'L-BFGS-B']."
        )
        with pytest.raises(ValueError, match="does not match") as window:
            fit(data, BirthDeathModel.piecewise_constant([3.0]))
        assert str(window.value) == (
            "Model window 3.0 does not match the data observation time 2.0."
        )
        with pytest.raises(ValueError, match="finite likelihood") as start:
            fit(data, model, constraints=Constraints(fixed={"birth.v0": 0.0}))
        assert str(start.value) == (
            "No starting point has a finite likelihood; pass `init`."
        )

    def test_unconverged_warning_text_category_and_location(self) -> None:
        data, model, kwargs = _pin_case("unconverged")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit(data, model, **kwargs)
        ours = [w for w in caught if "did not converge" in str(w.message)]
        assert len(ours) == 1
        assert ours[0].category is UserWarning
        assert str(ours[0].message) == (
            "Estimating uncertainty from a fit that did not converge; the "
            "Hessian is evaluated at a possibly non-stationary point, so "
            "the resulting covariance/standard errors may not be valid."
        )
        assert ours[0].filename == __file__


def test_fit_emits_no_warnings_on_clean_data() -> None:
    """Test that a well-posed fit is silent."""
    ages = _simulate([1.0], [0.6], [9.0], seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit(
            AgeData.from_groups([ages], 9.0),
            BirthDeathModel.piecewise_constant([9.0]),
        )
