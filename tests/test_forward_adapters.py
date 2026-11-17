"""Test that the forward-simulator adapters return the bdsurv records."""

import bdsurv
import jax.numpy as jnp
import numpy as np
import pandas as pd
from bdsurv import PiecewiseConstant

from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate
from bdpinf.process import BirthDeathProcess
from bdpinf.simulation import (
    simulate_birth_death,
    simulate_birth_death_lifespans,
)

SEED = 11
BIRTH = [0.3, 0.5]
DEATH = [0.2, 0.4]
TIMES = [10.0, 20.0]


def _assert_records_match(
    process: BirthDeathProcess,
    expected: bdsurv.SimulatedTree,
) -> None:
    assert isinstance(process, BirthDeathProcess)
    assert process.records is not None
    pd.testing.assert_frame_equal(
        process.records.reset_index(drop=True),
        expected.records.reset_index(drop=True),
        check_dtype=False,
    )
    assert process.start_time == 0.0
    assert process.initial_population == 1


class TestForwardAdapters:
    """Test the `BirthDeathProcess` adapters over the bdsurv simulators."""

    def test_constant_epochs_match_bdsurv_and_store_rates(self) -> None:
        birth = PiecewiseConstant(BIRTH, TIMES)
        death = PiecewiseConstant(DEATH, TIMES)
        process = simulate_birth_death(birth, death, random_seed=SEED)
        expected = bdsurv.simulate_birth_death(birth, death, random_seed=SEED)
        _assert_records_match(process, expected)
        np.testing.assert_array_equal(process.epoch_end_times, TIMES)
        np.testing.assert_array_equal(process.birth_rates, BIRTH)
        np.testing.assert_array_equal(process.death_rates, DEATH)

    def test_float_rates_match_bdsurv_and_store_one_epoch(self) -> None:
        process = simulate_birth_death(0.5, 0.2, 15.0, random_seed=SEED)
        expected = bdsurv.simulate_birth_death(0.5, 0.2, 15.0, random_seed=SEED)
        _assert_records_match(process, expected)
        np.testing.assert_array_equal(process.epoch_end_times, [15.0])
        np.testing.assert_array_equal(process.birth_rates, [0.5])
        np.testing.assert_array_equal(process.death_rates, [0.2])

    def test_constant_rates_on_different_epochs_store_merged_epochs(self) -> None:
        birth = PiecewiseConstantRate(jnp.array([0.3, 0.5]), jnp.array([5.0, 20.0]))
        death = PiecewiseConstantRate(jnp.array([0.2, 0.4]), jnp.array([12.0, 20.0]))
        process = simulate_birth_death(birth, death, random_seed=SEED)
        expected = bdsurv.simulate_birth_death(birth, death, random_seed=SEED)
        _assert_records_match(process, expected)
        np.testing.assert_array_equal(process.epoch_end_times, [5.0, 12.0, 20.0])
        assert process.birth_rates is None
        assert process.death_rates is None

    def test_linear_rates_match_bdsurv_and_store_no_rates(self) -> None:
        birth = PiecewiseLinearRate(
            jnp.array([0.3, 0.4]), jnp.array([0.4, 0.2]), jnp.array([5.0, 10.0])
        )
        death = PiecewiseConstantRate(jnp.array([0.2, 0.3]), jnp.array([5.0, 10.0]))
        process = simulate_birth_death(birth, death, random_seed=SEED)
        expected = bdsurv.simulate_birth_death(birth, death, random_seed=SEED)
        _assert_records_match(process, expected)
        np.testing.assert_array_equal(process.epoch_end_times, [5.0, 10.0])
        assert process.birth_rates is None
        assert process.death_rates is None

    def test_linear_rates_store_union_of_epoch_ends(self) -> None:
        birth = PiecewiseLinearRate(
            jnp.array([0.3, 0.4]), jnp.array([0.4, 0.2]), jnp.array([5.0, 10.0])
        )
        death = PiecewiseConstantRate(jnp.array([0.2, 0.3]), jnp.array([4.0, 10.0]))
        process = simulate_birth_death(birth, death, random_seed=SEED)
        np.testing.assert_array_equal(process.epoch_end_times, [4.0, 5.0, 10.0])
        assert process.birth_rates is None
        assert process.death_rates is None

    def test_lifespans_matches_bdsurv(self) -> None:
        process = simulate_birth_death_lifespans(BIRTH, DEATH, TIMES, random_seed=SEED)
        expected = bdsurv.simulate_birth_death_lifespans(
            BIRTH, DEATH, TIMES, random_seed=SEED
        )
        _assert_records_match(process, expected)
        assert process.records is not None
        assert "birth_epoch" in process.records.columns
        np.testing.assert_array_equal(process.epoch_end_times, TIMES)
        np.testing.assert_array_equal(process.birth_rates, BIRTH)
        np.testing.assert_array_equal(process.death_rates, DEATH)
