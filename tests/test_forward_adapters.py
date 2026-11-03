"""Test that the forward-simulator adapters return the bdsurv records."""

import bdsurv
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate
from bdpinf.process import BirthDeathProcess
from bdpinf.simulation import (
    simulate_birth_death_lifespans,
    simulate_birth_death_linear,
    simulate_birth_death_process,
)

SEED = 11
BIRTH = [0.3, 0.5]
DEATH = [0.2, 0.4]
TIMES = [10.0, 20.0]


def _assert_matches(
    process: BirthDeathProcess,
    expected: bdsurv.SimulatedProcess,
) -> None:
    assert isinstance(process, BirthDeathProcess)
    pd.testing.assert_frame_equal(
        process.records.reset_index(drop=True),
        expected.records.reset_index(drop=True),
        check_dtype=False,
    )
    np.testing.assert_array_equal(process.epoch_end_times, expected.epoch_end_times)


class TestForwardAdapters:
    """Test the `BirthDeathProcess` adapters over the bdsurv simulators."""

    def test_gillespie_matches_bdsurv(self) -> None:
        process = simulate_birth_death_process(BIRTH, DEATH, TIMES, random_seed=SEED)
        expected = bdsurv.simulate_birth_death_process(
            BIRTH, DEATH, TIMES, random_seed=SEED
        )
        _assert_matches(process, expected)
        np.testing.assert_array_equal(process.birth_rates, expected.birth_rates)
        np.testing.assert_array_equal(process.death_rates, expected.death_rates)

    def test_lifespans_matches_bdsurv(self) -> None:
        process = simulate_birth_death_lifespans(BIRTH, DEATH, TIMES, random_seed=SEED)
        expected = bdsurv.simulate_birth_death_lifespans(
            BIRTH, DEATH, TIMES, random_seed=SEED
        )
        _assert_matches(process, expected)
        assert "birth_epoch" in process.records.columns

    def test_linear_matches_bdsurv_and_stores_no_rates(self) -> None:
        birth = PiecewiseLinearRate(
            jnp.array([0.3, 0.4]), jnp.array([0.4, 0.2]), jnp.array([5.0, 10.0])
        )
        death = PiecewiseConstantRate(jnp.array([0.2, 0.3]), jnp.array([5.0, 10.0]))
        process = simulate_birth_death_linear(birth, death, random_seed=SEED)
        expected = bdsurv.simulate_birth_death_linear(birth, death, random_seed=SEED)
        _assert_matches(process, expected)
        assert expected.birth_rates is None
        assert expected.death_rates is None
